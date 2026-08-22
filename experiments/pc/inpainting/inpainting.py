import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.generation.transformer import (
    PointTransformer,
    ClassConditionedPointTransformer,
    ImgConditionPointTransformer
)
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow


def save_point_cloud_ply(
    filepath: str,
    points_and_normals: torch.Tensor,
) -> None:
    """
    Saves a 3D point cloud [N, 6] (or [6, N]) to an ASCII PLY file with normal-mapped RGB colors.
    :param filepath: Output .ply file path
    :param points_and_normals: [N, 6] or [6, N] Float tensor (XYZ + NxNyNz)
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if points_and_normals.shape[0] == 6 and points_and_normals.shape[1] != 6:
        points_and_normals = points_and_normals.permute(1, 0)

    data_np = points_and_normals.detach().cpu().numpy()
    pts = data_np[:, :3]
    normals = data_np[:, 3:6] if data_np.shape[1] >= 6 else None

    if normals is not None:
        norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normals = normals / norm
        colors = (((normals + 1.0) / 2.0) * 255.0).clip(0, 255).astype("uint8")
    else:
        colors = None

    N = len(pts)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals is not None:
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header.extend(["end_header"])

    lines = ["\n".join(header)]
    for i in range(N):
        p = pts[i]
        if normals is not None:
            n = normals[i]
            c = colors[i]
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {c[0]} {c[1]} {c[2]}")
        else:
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def create_half_space_mask(
    points: torch.Tensor,
    crop_axis: str = "z",
    crop_sign: float = 1.0,
    crop_ratio: float = 0.5
) -> torch.Tensor:
    """
    Computes a boolean mask for half-space plane cropping.
    :param points: [N, 6] or [6, N] or [B, 6, N] tensor
    :param crop_axis: Cutting plane axis ('x', 'y', 'z')
    :param crop_sign: 1.0 (keep positive side) or -1.0 (keep negative side)
    :param crop_ratio: Approximate ratio of points to retain (0.0 to 1.0)
    :return: Boolean mask [N] where True indicates observed points
    """
    if points.ndim == 3:
        pts = points[0]  # [6, N] or [N, 6]
    else:
        pts = points

    if pts.shape[0] == 6 and pts.shape[1] != 6:
        pts = pts.permute(1, 0)  # [N, 6]

    axis_idx = {"x": 0, "y": 1, "z": 2}[crop_axis.lower()]
    coords = pts[:, axis_idx]

    # Compute quantile threshold for exact ratio retention
    q = (1.0 - crop_ratio) if crop_sign > 0 else crop_ratio
    thresh = torch.quantile(coords, q)

    if crop_sign >= 0:
        mask = coords >= thresh
    else:
        mask = coords <= thresh

    return mask


def initialize_inpainting_noise(
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    crop_axis: str = "z",
    crop_sign: float = 1.0,
    init_mode: str = "reflected",
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Initializes the starting noise state x_0 with a spatial prior in the missing region.

    :param x_obs: [B, 6, N] Ground-truth complete/observed point cloud
    :param mask: [N] Boolean tensor (True = observed, False = missing)
    :param crop_axis: Cutting plane axis ('x', 'y', 'z')
    :param crop_sign: 1.0 or -1.0
    :param init_mode: 'standard', 'shifted', or 'reflected'
    :param device: Execution device
    :return: [B, 6, N] Initial noise tensor x_0
    """
    B, C, N = x_obs.shape
    x_0 = torch.randn((B, C, N), device=device)
    axis_idx = {"x": 0, "y": 1, "z": 2}[crop_axis.lower()]
    num_unobs = int((~mask).sum().item())

    if init_mode == "reflected" and num_unobs > 0:
        # Reflect observed point coordinates across the cutting plane axis to provide geometric prior
        obs_pts = x_obs[:, :3, mask].clone()  # [B, 3, M]
        obs_pts[:, axis_idx, :] = -obs_pts[:, axis_idx, :]  # Mirror reflection

        # Match unobserved token count
        M = obs_pts.shape[-1]
        if M >= num_unobs:
            init_pos = obs_pts[:, :, :num_unobs]
        else:
            repeats = (num_unobs // M) + 1
            init_pos = obs_pts.repeat(1, 1, repeats)[:, :, :num_unobs]

        # Add Gaussian jitter
        x_0[:, :3, ~mask] = init_pos + 0.35 * torch.randn_like(init_pos)

    elif init_mode == "shifted" and num_unobs > 0:
        # Shift Gaussian noise center towards the missing half-space
        shift = torch.zeros((1, 3, 1), device=device)
        shift[0, axis_idx, 0] = -crop_sign * 0.45
        x_0[:, :3, ~mask] = x_0[:, :3, ~mask] * 0.6 + shift

    return x_0


def compute_boundary_energy(
    x_unobs_pos: torch.Tensor,
    x_obs_pos: torch.Tensor,
    crop_axis: str = "z",
    crop_sign: float = 1.0,
    thresh: float = 0.0,
    k_seam: int = 32
) -> torch.Tensor:
    """
    Computes smooth, differentiable seam attachment energy and soft half-space barrier.
    """
    B, _, K = x_unobs_pos.shape
    M = x_obs_pos.shape[-1]
    axis_idx = {"x": 0, "y": 1, "z": 2}[crop_axis.lower()]

    # 1. Seam Boundary Distance Energy
    obs_dist = torch.abs(x_obs_pos[:, axis_idx, :] - thresh)
    unobs_dist = torch.abs(x_unobs_pos[:, axis_idx, :] - thresh)
    k_s = min(k_seam, min(M, K))
    _, top_obs = torch.topk(obs_dist, k=k_s, largest=False, dim=-1)
    _, top_unobs = torch.topk(unobs_dist, k=k_s, largest=False, dim=-1)

    batch_idx = torch.arange(B, device=x_unobs_pos.device).unsqueeze(-1)
    p_obs = x_obs_pos[batch_idx, :, top_obs]
    p_unobs = x_unobs_pos[batch_idx, :, top_unobs]
    diff = p_unobs.unsqueeze(-1) - p_obs.unsqueeze(-2)
    e_boundary = (diff ** 2).sum(dim=1).min(dim=-1)[0].mean()

    # 2. Soft Half-Space Barrier (Smooth quadratic barrier)
    unobs_coords = x_unobs_pos[:, axis_idx, :]
    margin = 0.02
    if crop_sign > 0:
        violation = torch.relu(unobs_coords - (thresh + margin))
    else:
        violation = torch.relu((thresh - margin) - unobs_coords)
    e_halfspace = (violation ** 2).mean()

    return e_boundary + 2.0 * e_halfspace


def inpaint_flow_ode(
    model: torch.nn.Module,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    crop_axis: str = "z",
    crop_sign: float = 1.0,
    steps: int = 64,
    cfg_scale: float = 1.0,
    cond_payload: Optional[torch.Tensor] = None,
    init_mode: str = "reflected",
    boundary_guidance: float = 1.0,
    resample_steps: int = 1,
    resample_noise_strength: float = 0.1,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Option B: Differentiable Boundary Energy-Guided Flow Matching ODE Inpainting.
    No coordinate clamping and no centroid shifting; shapes evolve with natural 3D curvature.

    :param model: Trained PointTransformer / ClassConditioned / ImgCondition model
    :param x_obs: [B, 6, N] Ground-truth clean point cloud
    :param mask: [N] Boolean tensor where True = observed, False = missing
    :param crop_axis: Cutting plane normal axis ('x', 'y', 'z')
    :param crop_sign: Direction (1.0 or -1.0)
    :param steps: Number of ODE integration steps (e.g. 64)
    :param cfg_scale: Classifier-Free Guidance scale (1.0 = standard conditional pass, > 1.0 = CFG extrapolation)
    :param cond_payload: Optional class labels [B] or conditioning images [B, 1, 28, 28]
    :param init_mode: Noise prior mode ('reflected', 'shifted', 'standard')
    :param boundary_guidance: Boundary energy guidance gradient strength (-∇E_boundary)
    :param resample_steps: Harmonization repetitions per ODE step
    :param resample_noise_strength: Noise injection factor for resampling
    :param device: Execution device
    :return: [B, 6, N] Completed point cloud
    """
    B, C, N = x_obs.shape
    gt_clean = x_obs.to(device)
    M = int(mask.sum().item())
    K = int((~mask).sum().item())
    axis_idx = {"x": 0, "y": 1, "z": 2}[crop_axis.lower()]
    thresh = gt_clean[0, axis_idx, mask].min().item() if crop_sign > 0 else gt_clean[0, axis_idx, mask].max().item()

    # 1. Spatially-biased initial noise state
    x_0 = initialize_inpainting_noise(
        x_obs=gt_clean,
        mask=mask,
        crop_axis=crop_axis,
        crop_sign=crop_sign,
        init_mode=init_mode,
        device=device
    )

    times = torch.linspace(0.0, 1.0, steps + 1, device=device)
    dt = 1.0 / steps
    x_curr = x_0.clone()

    # 2. ODE Integration Loop
    for step_idx in range(steps):
        t_curr = times[step_idx]
        t_next = times[step_idx + 1]
        t_curr_batch = t_curr.expand(B)

        for r in range(resample_steps):
            # Compute velocity with Classifier-Free Guidance (if cfg_scale > 1.0)
            if cfg_scale > 1.0 and cond_payload is not None:
                v_cond = model(x_curr, t=t_curr_batch, cond=cond_payload)
                v_uncond = model(x_curr, t=t_curr_batch, cond=None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                if cond_payload is not None:
                    v = model(x_curr, t=t_curr_batch, cond=cond_payload)
                else:
                    v = model(x_curr, t=t_curr_batch)

            # Option B: Geometric Boundary Energy Guidance
            gamma_t = boundary_guidance * (1.0 - t_curr.item())
            if gamma_t > 0.0 and K > 0 and M > 0:
                with torch.enable_grad():
                    unobs_pos = x_curr[:, :3, ~mask].detach().clone().requires_grad_(True)
                    obs_pos = x_curr[:, :3, mask].detach()
                    energy = compute_boundary_energy(
                        unobs_pos, obs_pos, crop_axis=crop_axis, crop_sign=crop_sign, thresh=thresh
                    )
                    grad_unobs = torch.autograd.grad(energy, unobs_pos)[0]
                v_unobs = v[:, :, ~mask].clone()
                v_unobs[:, :3, :] = v_unobs[:, :3, :] - gamma_t * grad_unobs
            else:
                v_unobs = v[:, :, ~mask]

            # Step unobserved points forward (Continuous flow without any hard clamping or artificial centroid shifts)
            x_next_unobs = x_curr[:, :, ~mask] + v_unobs * dt

            # Exact straight-line path anchoring for observed points:
            # x_t = (1 - t) * x_0 + t * x_1
            x_next_obs = (1.0 - t_next) * x_0[:, :, mask] + t_next * gt_clean[:, :, mask]

            # Assemble composite state at t_next
            x_next = torch.empty_like(x_curr)
            x_next[:, :, mask] = x_next_obs
            x_next[:, :, ~mask] = x_next_unobs

            # Optional resampling harmonization step
            if resample_steps > 1 and r < (resample_steps - 1) and step_idx < (steps - 1):
                noise_resample = torch.randn_like(x_next[:, :, ~mask])
                x_curr[:, :, ~mask] = (
                    x_next_unobs - v_unobs * dt + resample_noise_strength * noise_resample * dt
                )
                x_curr[:, :, mask] = (1.0 - t_curr) * x_0[:, :, mask] + t_curr * gt_clean[:, :, mask]
            else:
                x_curr = x_next

    # Ensure surface normals on completed points are unit normalized
    normals = F.normalize(x_curr[:, 3:6, :], p=2, dim=1)
    x_final = torch.cat([x_curr[:, :3, :], normals], dim=1)

    return x_final


def main():
    parser = argparse.ArgumentParser(description="Option B: 3D Point Cloud Inpainting with Boundary Guidance")
    parser.add_argument("--mode", type=int, default=0, help="0: RectifiedFlow, 1: MeanFlow, 2: SoFlow")
    parser.add_argument("--ckpt", type=str, default="", help="Path to explicit .pt checkpoint")
    parser.add_argument("--class_cond", action="store_true", help="Use class conditioning")
    parser.add_argument("--img_cond", action="store_true", help="Use image conditioning")
    parser.add_argument("--class_token_cond", action="store_true", help="Pass condition as a token")
    parser.add_argument("--class_label", type=int, default=-1, help="Specify digit class (0-9)")
    parser.add_argument("--crop_axis", type=str, default="z", choices=["x", "y", "z"], help="Cutting plane normal axis")
    parser.add_argument("--crop_sign", type=float, default=1.0, choices=[1.0, -1.0], help="1.0: keep positive side, -1.0: keep negative side")
    parser.add_argument("--crop_ratio", type=float, default=0.5, help="Ratio of points retained in observed region (0.1 to 0.9)")
    parser.add_argument("--init_mode", type=str, default="reflected", choices=["reflected", "shifted", "standard"], help="Spatial noise initialization prior")
    parser.add_argument("--boundary_guidance", type=float, default=1.0, help="Boundary energy guidance weight (-∇E_boundary)")
    parser.add_argument("--steps", type=int, default=64, help="Number of ODE integration steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale (1.0 = standard conditional, > 1.0 = CFG)")
    parser.add_argument("--resample_steps", type=int, default=1, help="Harmonization resampling repetitions per ODE step")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to complete")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting offset index in test dataset")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save output .ply files")
    parser.add_argument("--exp_name", type=str, default="", help="Custom experiment name to load checkpoint from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Initialize Transformer Model Architecture
    input_channels = 6
    output_channels = 6
    n_ctx = 512
    width = 256
    layers = 6
    heads = 8
    init_scale = 0.25

    if args.img_cond:
        model = ImgConditionPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=input_channels,
            output_channels=output_channels,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            img_channels=1,
            cond_drop_prob=0.0,
            token_cond=args.class_token_cond
        )
    elif args.class_cond:
        model = ClassConditionedPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=input_channels,
            output_channels=output_channels,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            num_classes=10,
            cond_drop_prob=0.0,
            token_cond=args.class_token_cond
        )
    else:
        model = PointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=input_channels,
            output_channels=output_channels,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale
        )

    accept_cond = args.class_cond or args.img_cond
    if args.mode == 0:
        flow_model = RectifiedFlow(model, time_cond_kwarg="t", predict="flow")
        mode_name = "rectified_flow"
    elif args.mode == 1:
        flow_model = MeanFlow(model, accept_cond=accept_cond)
        mode_name = "mean_flow"
    elif args.mode == 2:
        flow_model = SoFlow(model, accept_cond=accept_cond)
        mode_name = "soflow"
        args.steps = 1
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    flow_model = flow_model.to(device)

    # 2. Locate and Load Pretrained Model Checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        exp_dir = args.exp_name if args.exp_name else mode_name
    else:
        exp_suffix = "_img_cond" if args.img_cond else ("_class_cond" if args.class_cond else "")
        exp_dir = f"{mode_name}{exp_suffix}" if not args.exp_name else args.exp_name
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "generation", "runs", exp_dir, "model.pt"
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at: {ckpt_path}")

    print(f"Loading pretrained weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        flow_model.load_state_dict(state_dict)
    except Exception:
        flow_model.load_state_dict(state_dict, strict=False)
    flow_model.eval()

    # 3. Load Test Dataset
    print("Loading PointDigit3D test dataset...")
    test_dataset = PointDigit3D(
        root="~/.conquer3d/",
        train=False,
        download=True,
        cached=True,
        num_points=n_ctx,
        return_img=True
    )

    # Resolve output directory
    save_dir = (
        args.save_dir
        if args.save_dir
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir)
    )
    os.makedirs(save_dir, exist_ok=True)

    print("==================================================================")
    print("        Option B: Energy-Guided 3D Point Cloud Inpainting         ")
    print("==================================================================")
    print(f"Model       : {mode_name} ({exp_dir})")
    print(f"Device      : {device}")
    print(f"Crop Plane  : Axis '{args.crop_axis}', Sign {args.crop_sign:+.1f}, Ratio {args.crop_ratio:.2f}")
    print(f"Noise Init  : {args.init_mode}")
    print(f"Boundary Gui: {args.boundary_guidance}")
    print(f"CFG Scale   : {args.cfg_scale}")
    print(f"ODE Steps   : {args.steps}")
    print(f"Resampling  : {args.resample_steps} passes")
    print(f"Num Samples : {args.num_samples}")
    print(f"Output Dir  : {save_dir}")
    print("------------------------------------------------------------------")

    # 4. Filter or Select Target Samples
    selected_indices = []
    if args.class_label >= 0:
        for idx in range(args.sample_offset, len(test_dataset)):
            _, _, lbl, _ = test_dataset[idx]
            if lbl == args.class_label:
                selected_indices.append(idx)
                if len(selected_indices) == args.num_samples:
                    break
    else:
        selected_indices = list(range(args.sample_offset, min(args.sample_offset + args.num_samples, len(test_dataset))))

    # 5. Inpainting Loop
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Inpainting Samples")):
            points_t, features_t, label_val, img_t = test_dataset[idx]
            # features_t shape: [512, 6] -> permute to [1, 6, 512]
            x_gt = features_t.unsqueeze(0).permute(0, 2, 1).to(device)

            # Compute Half-Space Mask
            mask = create_half_space_mask(
                x_gt[0],
                crop_axis=args.crop_axis,
                crop_sign=args.crop_sign,
                crop_ratio=args.crop_ratio
            ).to(device)

            # Prepare conditioning payload
            if args.img_cond:
                cond_payload = img_t.unsqueeze(0).to(device)  # [1, 1, 28, 28]
            elif args.class_cond:
                cond_payload = torch.tensor([label_val], device=device, dtype=torch.long)
            else:
                cond_payload = None

            # Execute Flow ODE Inpainting
            t0 = time.time()
            completed_pc = inpaint_flow_ode(
                model=model,
                x_obs=x_gt,
                mask=mask,
                crop_axis=args.crop_axis,
                crop_sign=args.crop_sign,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                cond_payload=cond_payload,
                init_mode=args.init_mode,
                boundary_guidance=args.boundary_guidance,
                resample_steps=args.resample_steps,
                device=device
            )
            torch.cuda.synchronize()
            latency_ms = (time.time() - t0) * 1000

            # Extract Observed partial points
            x_observed = x_gt[:, :, mask]

            # 6. Save Artifacts for Sample
            prefix = f"sample_{i:03d}_class_{label_val}"
            
            # Ground Truth Complete Point Cloud
            gt_ply_path = os.path.join(save_dir, f"{prefix}_gt_complete.ply")
            save_point_cloud_ply(gt_ply_path, x_gt[0])

            # Partial Input Observed Point Cloud
            partial_ply_path = os.path.join(save_dir, f"{prefix}_input_partial.ply")
            save_point_cloud_ply(partial_ply_path, x_observed[0])

            # Completed Inpainted Point Cloud
            completed_ply_path = os.path.join(save_dir, f"{prefix}_completed.ply")
            save_point_cloud_ply(completed_ply_path, completed_pc[0])

            # Paired 2D Image (if available)
            if img_t is not None:
                png_path = os.path.join(save_dir, f"{prefix}_input.png")
                torchvision.utils.save_image(img_t, png_path)

            # Save Metadata JSON
            meta = {
                "sample_idx": idx,
                "class_label": int(label_val),
                "crop_axis": args.crop_axis,
                "crop_sign": args.crop_sign,
                "crop_ratio": args.crop_ratio,
                "init_mode": args.init_mode,
                "boundary_guidance": args.boundary_guidance,
                "num_observed_points": int(mask.sum().item()),
                "num_total_points": n_ctx,
                "latency_ms": round(latency_ms, 2),
                "cfg_scale": args.cfg_scale,
                "resample_steps": args.resample_steps,
                "files": {
                    "gt_complete": f"{prefix}_gt_complete.ply",
                    "input_partial": f"{prefix}_input_partial.ply",
                    "completed": f"{prefix}_completed.ply"
                }
            }
            meta_json_path = os.path.join(save_dir, f"{prefix}_meta.json")
            with open(meta_json_path, "w") as f:
                json.dump(meta, f, indent=4)

    print(f"\n[✓] Inpainting complete! All samples and metadata exported to: {save_dir}")


if __name__ == "__main__":
    main()
