import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import PointDigit3D
from conquer3d.ops.distance import chamfer_distance, hausdorff_distance
from experiments.pc.generation.transformer import (
    PointTransformer,
    ClassConditionedPointTransformer,
    ImgConditionPointTransformer
)
from experiments.pc.inpainting.inpainting import create_half_space_mask, inpaint_flow_ode
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow


def compute_metrics(
    completed_pc: torch.Tensor,
    gt_pc: torch.Tensor,
    squared: bool = False
) -> Tuple[float, float, float]:
    """
    Computes Chamfer Distance, Hausdorff Distance, and Normal Cosine Similarity using conquer3d.ops.
    :param completed_pc: [512, 6] on CUDA (XYZ + NxNyNz)
    :param gt_pc: [512, 6] on CUDA (XYZ + NxNyNz)
    :param squared: If True, returns squared distances
    :return: (chamfer_dist, hausdorff_dist, normal_cosine_sim)
    """
    pred_pts = completed_pc[:, :3].contiguous()
    pred_normals = completed_pc[:, 3:6].contiguous()

    gt_pts = gt_pc[:, :3].contiguous()
    gt_normals = gt_pc[:, 3:6].contiguous()

    # 1. GPU-accelerated symmetric Chamfer Distance and nearest neighbor indices
    cd_val, idx_pred_to_gt, _ = chamfer_distance(
        pred_pts, gt_pts, squared=squared, return_indices=True
    )

    # 2. GPU-accelerated symmetric bidirectional Hausdorff Distance
    hd_val = hausdorff_distance(
        pred_pts, gt_pts, squared=squared
    )

    # 3. Normal alignment on nearest neighbor surface pairs
    matched_gt_normals = gt_normals[idx_pred_to_gt]
    cosine_sim = (pred_normals * matched_gt_normals).sum(dim=-1).clamp(-1.0, 1.0)
    mean_cosine = cosine_sim.mean().item()

    return cd_val.item(), hd_val.item(), mean_cosine


def main():
    parser = argparse.ArgumentParser(description="Evaluate Option B: Boundary-Guided Inpainting on Digit3D")
    parser.add_argument("--mode", type=int, default=0, help="0: RectifiedFlow, 1: MeanFlow, 2: SoFlow")
    parser.add_argument("--ckpt", type=str, default="", help="Path to explicit .pt checkpoint")
    parser.add_argument("--class_cond", action="store_true", help="Use class conditioning")
    parser.add_argument("--img_cond", action="store_true", help="Use image conditioning")
    parser.add_argument("--class_token_cond", action="store_true", help="Pass condition as a token")
    parser.add_argument("--crop_axis", type=str, default="z", choices=["x", "y", "z"], help="Cutting plane axis")
    parser.add_argument("--crop_sign", type=float, default=1.0, choices=[1.0, -1.0], help="Cutting direction")
    parser.add_argument("--crop_ratio", type=float, default=0.5, help="Observed ratio")
    parser.add_argument("--init_mode", type=str, default="reflected", choices=["reflected", "shifted", "standard"], help="Spatial noise initialization prior")
    parser.add_argument("--boundary_guidance", type=float, default=1.0, help="Boundary energy guidance scale")
    parser.add_argument("--steps", type=int, default=64, help="ODE steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale (1.0 = fast, > 1.0 = CFG)")
    parser.add_argument("--resample_steps", type=int, default=1, help="Resampling repetitions")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--out_file", type=str, default="eval_metrics.json", help="Output JSON filename")
    parser.add_argument("--exp_name", type=str, default="", help="Custom experiment name")
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
    num_eval = min(args.num_samples, len(test_dataset))

    print("==================================================================")
    print("        Option B: Inpainting Benchmark (Boundary Guidance)        ")
    print("==================================================================")
    print(f"Model       : {mode_name} ({exp_dir})")
    print(f"Eval Samples: {num_eval}")
    print(f"Crop Plane  : Axis '{args.crop_axis}', Sign {args.crop_sign:+.1f}, Ratio {args.crop_ratio:.2f}")
    print(f"Noise Init  : {args.init_mode}")
    print(f"Boundary Gui: {args.boundary_guidance}")
    print(f"CFG Scale   : {args.cfg_scale}")
    print(f"ODE Steps   : {args.steps}")
    print(f"Resampling  : {args.resample_steps} passes")
    print(f"Device      : {device}")
    print("------------------------------------------------------------------")

    total_cd = 0.0
    total_hd = 0.0
    total_n_sim = 0.0
    total_time_ms = 0.0
    per_class_metrics = {str(c): {"cd": [], "hd": [], "n_sim": []} for c in range(10)}

    with torch.no_grad():
        for idx in tqdm(range(num_eval), desc="Evaluating Inpainting"):
            _, features_t, label_val, img_t = test_dataset[idx]
            x_gt = features_t.unsqueeze(0).permute(0, 2, 1).to(device)

            mask = create_half_space_mask(
                x_gt[0],
                crop_axis=args.crop_axis,
                crop_sign=args.crop_sign,
                crop_ratio=args.crop_ratio
            ).to(device)

            if args.img_cond:
                cond_payload = img_t.unsqueeze(0).to(device)
            elif args.class_cond:
                cond_payload = torch.tensor([label_val], device=device, dtype=torch.long)
            else:
                cond_payload = None

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
            total_time_ms += (time.time() - t0) * 1000

            # completed_pc is [1, 6, 512] -> permute to [512, 6]
            comp_pts = completed_pc[0].permute(1, 0)
            gt_pts = x_gt[0].permute(1, 0)

            cd, hd, n_sim = compute_metrics(comp_pts, gt_pts)
            total_cd += cd
            total_hd += hd
            total_n_sim += n_sim

            lbl_str = str(label_val)
            per_class_metrics[lbl_str]["cd"].append(cd)
            per_class_metrics[lbl_str]["hd"].append(hd)
            per_class_metrics[lbl_str]["n_sim"].append(n_sim)

    avg_cd = total_cd / num_eval
    avg_hd = total_hd / num_eval
    avg_n_sim = total_n_sim / num_eval
    avg_latency = total_time_ms / num_eval

    # Aggregate per-class averages
    class_summary = {}
    for c in range(10):
        c_str = str(c)
        if len(per_class_metrics[c_str]["cd"]) > 0:
            class_summary[c_str] = {
                "num_samples": len(per_class_metrics[c_str]["cd"]),
                "chamfer_distance": round(float(np.mean(per_class_metrics[c_str]["cd"])), 6),
                "hausdorff_distance": round(float(np.mean(per_class_metrics[c_str]["hd"])), 6),
                "normal_cosine_similarity": round(float(np.mean(per_class_metrics[c_str]["n_sim"])), 4),
            }

    results = {
        "checkpoint": os.path.basename(ckpt_path),
        "experiment": exp_dir,
        "num_eval_samples": num_eval,
        "crop_axis": args.crop_axis,
        "crop_sign": args.crop_sign,
        "crop_ratio": args.crop_ratio,
        "init_mode": args.init_mode,
        "boundary_guidance": args.boundary_guidance,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "resample_steps": args.resample_steps,
        "mean_chamfer_distance": round(avg_cd, 6),
        "mean_hausdorff_distance": round(avg_hd, 6),
        "mean_normal_cosine_similarity": round(avg_n_sim, 4),
        "mean_latency_ms": round(avg_latency, 2),
        "throughput_fps": round(1000.0 / avg_latency if avg_latency > 0 else 0.0, 1),
        "per_class": class_summary
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("                 OPTION B BENCHMARK RESULTS                           ")
    print("=" * 70)
    print(f"Evaluated Samples        : {num_eval}")
    print(f"Noise Initialization     : {args.init_mode}")
    print(f"Boundary Guidance Scale  : {args.boundary_guidance}")
    print(f"Mean Chamfer Distance    : {avg_cd:.6f}")
    print(f"Mean Hausdorff Distance  : {avg_hd:.6f}")
    print(f"Mean Normal Cosine Sim   : {avg_n_sim:.4f}")
    print(f"Mean Latency per Sample  : {avg_latency:.2f} ms ({1000.0/avg_latency:.1f} FPS)")
    print(f"Saved Results to         : {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
