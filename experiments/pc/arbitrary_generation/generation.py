import argparse
import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import trimesh
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import Digit3D
from experiments.pc.arbitrary_generation.transformer import (
    ClassConditionedPointTransformer,
    ImgConditionPointTransformer,
    PointTransformer,
)
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from rectified_flow_pytorch.soflow import SoFlow


def save_point_cloud_ply(
    filepath: str,
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None
) -> None:
    """
    Saves an oriented 3D point cloud as an ASCII PLY file with normal-mapped RGB colors.
    :param filepath: Target .ply file path
    :param points: [P, 3] Float tensor of 3D spatial coordinates
    :param normals: Optional [P, 3] Float tensor of surface normal vectors
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    points_np = points.detach().cpu().numpy()

    if normals is not None:
        normals_np = normals.detach().cpu().numpy()
        norm = (normals_np ** 2).sum(axis=-1, keepdims=True) ** 0.5 + 1e-8
        normals_np = normals_np / norm
        colors_np = (((normals_np + 1.0) / 2.0) * 255.0).clip(0, 255).astype("uint8")
    else:
        normals_np = None
        colors_np = None

    N = len(points_np)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals_np is not None:
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
        p = points_np[i]
        if normals_np is not None:
            n = normals_np[i]
            c = colors_np[i]
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {c[0]} {c[1]} {c[2]}")
        else:
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Arbitrary-Resolution Point Cloud Generation via Self-Attention")
    parser.add_argument("--mode", type=int, default=1, help="0: RectifiedFlow, 1: MeanFlow, 2: SoFlow")
    parser.add_argument("--class_cond", action="store_true", help="Use class conditioning")
    parser.add_argument("--img_cond", action="store_true", default=True, help="Use image conditioning")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    parser.add_argument("--exp_name", type=str, default="", help="Custom experiment run folder name")
    parser.add_argument("--num_points", type=int, default=1024, help="Default points to generate")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[], help="List of resolutions (e.g. 256 512 1024 4096 16384)")
    parser.add_argument("--steps", type=int, default=64, help="Sampling steps for flow ODE solver (1 for MeanFlow, 64 for ODE)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to generate")
    parser.add_argument("--full_class", action="store_true", help="Generate 1 sample for each digit class 0 to 9")
    parser.add_argument("--save_dir", type=str, default="", help="Output directory (defaults to runs/<run_name>/ply_samples)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine run name matching train.py convention
    if args.exp_name:
        run_name = args.exp_name
    else:
        mode_str = "rf" if args.mode == 0 else ("mf" if args.mode == 1 else "soflow")
        cond_str = "img" if args.img_cond else ("class" if args.class_cond else "uncond")
        run_name = f"{mode_str}_{cond_str}_arbitrary"

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", run_name)

    # Locate checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(run_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(run_dir, "latest_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Output directory inside the corresponding run folder
    save_dir = args.save_dir if args.save_dir else os.path.join(run_dir, "ply_samples")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    mode = saved_args.get("mode", args.mode)
    img_cond = saved_args.get("img_cond", args.img_cond)
    class_cond = saved_args.get("class_cond", args.class_cond)

    if img_cond:
        model = ImgConditionPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=saved_args.get("input_channels", 6),
            output_channels=saved_args.get("output_channels", 6),
            width=saved_args.get("width", 256),
            layers=saved_args.get("layers", 6),
            heads=saved_args.get("heads", 8),
            init_scale=saved_args.get("init_scale", 0.25),
            time_token_cond=saved_args.get("time_token_cond", False),
            use_checkpoint=saved_args.get("use_checkpoint", False),
            img_channels=1,
            cond_drop_prob=0.0,
            token_cond=saved_args.get("class_token_cond", False)
        )
    elif class_cond:
        model = ClassConditionedPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=saved_args.get("input_channels", 6),
            output_channels=saved_args.get("output_channels", 6),
            width=saved_args.get("width", 256),
            layers=saved_args.get("layers", 6),
            heads=saved_args.get("heads", 8),
            init_scale=saved_args.get("init_scale", 0.25),
            time_token_cond=saved_args.get("time_token_cond", False),
            use_checkpoint=saved_args.get("use_checkpoint", False),
            num_classes=10,
            cond_drop_prob=0.0,
            token_cond=saved_args.get("class_token_cond", False)
        )
    else:
        model = PointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=saved_args.get("input_channels", 6),
            output_channels=saved_args.get("output_channels", 6),
            width=saved_args.get("width", 256),
            layers=saved_args.get("layers", 6),
            heads=saved_args.get("heads", 8),
            init_scale=saved_args.get("init_scale", 0.25),
            time_token_cond=saved_args.get("time_token_cond", False),
            use_checkpoint=saved_args.get("use_checkpoint", False),
        )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    accept_cond = img_cond or class_cond
    if mode == 0:
        flow_model = RectifiedFlow(model=model, accept_cond=accept_cond).to(device)
    elif mode == 1:
        flow_model = MeanFlow(model=model, accept_cond=accept_cond).to(device)
    elif mode == 2:
        flow_model = SoFlow(model=model, accept_cond=accept_cond).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    flow_model.eval()
    print("Model and Flow wrapper initialized successfully.")

    # Load test dataset
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=img_cond)

    # Select target samples
    selected_indices = []
    if args.full_class:
        found_classes = set()
        for idx in range(len(test_dataset)):
            _, _, label, _ = test_dataset[idx]
            if label not in found_classes:
                found_classes.add(label)
                selected_indices.append((idx, label))
            if len(found_classes) == 10:
                break
        selected_indices.sort(key=lambda x: x[1])
    else:
        for idx in range(min(args.num_samples, len(test_dataset))):
            _, _, label, _ = test_dataset[idx]
            selected_indices.append((idx, label))

    resolutions = args.resolutions if args.resolutions else [args.num_points]

    print("==================================================================")
    print("   Arbitrary-Resolution Point Cloud Generation via Self-Attn      ")
    print("==================================================================")
    print(f"Run Name     : {run_name}")
    print(f"Samples Count: {len(selected_indices)}")
    print(f"Resolutions  : {resolutions}")
    print(f"Steps        : {args.steps}")
    print(f"Output Dir   : {save_dir}")
    print("------------------------------------------------------------------")

    with torch.no_grad():
        for idx, label in tqdm(selected_indices, desc="Generating Samples"):
            v, f, _, img_t = test_dataset[idx]
            img_gpu = img_t.unsqueeze(0).to(device) if img_t is not None else None

            # Save paired 2D input image
            if img_t is not None:
                img_pil = TF.to_pil_image(img_t)
                img_save_path = os.path.join(save_dir, f"sample_{idx:03d}_class_{label}_input.png")
                img_pil.save(img_save_path)

            # Construct ground-truth mesh
            mesh = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
            gt_mesh_path = os.path.join(save_dir, f"sample_{idx:03d}_class_{label}_gt_mesh.ply")
            mesh.export(gt_mesh_path)

            meta = {
                "sample_index": idx,
                "digit_label": label,
                "steps": args.steps,
                "gt_mesh": f"sample_{idx:03d}_class_{label}_gt_mesh.ply",
                "resolutions": {}
            }

            for P in resolutions:
                t0 = time.time()
                samples = flow_model.sample(
                    batch_size=1,
                    data_shape=(P, 6),
                    steps=args.steps,
                    cond=img_gpu
                )
                torch.cuda.synchronize()
                latency_ms = (time.time() - t0) * 1000

                pts = samples[0, :, :3]
                normals = F.normalize(samples[0, :, 3:6], p=2, dim=-1)

                # Save generated predicted point cloud
                ply_filename = f"sample_{idx:03d}_class_{label}_pts_{P}.ply"
                ply_path = os.path.join(save_dir, ply_filename)
                save_point_cloud_ply(ply_path, pts, normals)

                # Sample & save ground-truth point cloud at exact resolution P
                gt_pts_np, gt_f_idx = trimesh.sample.sample_surface(mesh, P)
                gt_normals_np = mesh.face_normals[gt_f_idx]
                gt_ply_filename = f"sample_{idx:03d}_class_{label}_gt_pts_{P}.ply"
                gt_ply_path = os.path.join(save_dir, gt_ply_filename)
                save_point_cloud_ply(
                    gt_ply_path,
                    torch.tensor(gt_pts_np, dtype=torch.float32),
                    torch.tensor(gt_normals_np, dtype=torch.float32)
                )

                meta["resolutions"][str(P)] = {
                    "num_points": P,
                    "latency_ms": round(latency_ms, 2),
                    "pred_ply_file": ply_filename,
                    "gt_ply_file": gt_ply_filename
                }

            meta_save_path = os.path.join(save_dir, f"sample_{idx:03d}_class_{label}_meta.json")
            with open(meta_save_path, "w") as f:
                json.dump(meta, f, indent=2)

    print(f"\nAll samples and ground truth files generated successfully in: {save_dir}")


if __name__ == "__main__":
    main()
