import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import trimesh
from tqdm.auto import tqdm

# Append repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.ops import chamfer_distance, hausdorff_distance
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from experiments.pc.arbitrary_generation.models import ArbitraryPointFlowTransformer


def compute_metrics_with_conquer3d(
    pred_pts: torch.Tensor,
    pred_normals: torch.Tensor,
    gt_pts: torch.Tensor,
    gt_normals: torch.Tensor,
    squared: bool = False
) -> Tuple[float, float, float]:
    """
    Computes Chamfer Distance, Hausdorff Distance, and Normal Cosine Similarity using conquer3d.ops.
    :param pred_pts: [P, 3] Float tensor on CUDA
    :param pred_normals: [P, 3] Float tensor on CUDA
    :param gt_pts: [P, 3] Float tensor on CUDA
    :param gt_normals: [P, 3] Float tensor on CUDA
    :param squared: If True, computes squared distances; If False, computes true Euclidean distances
    :return: (chamfer_dist, hausdorff_dist, normal_cosine_similarity)
    """
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
    parser = argparse.ArgumentParser(description="Evaluate Arbitrary-Resolution Point Cloud Model with conquer3d")
    parser.add_argument("--ckpt", type=str, default="", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--exp_name", type=str, default="naive", help="Experiment run folder")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096], help="Point count budgets to test")
    parser.add_argument("--steps", type=int, default=1, help="Sampling steps for flow ODE solver (1 for MeanFlow)")
    parser.add_argument("--squared", action="store_true", help="Compute squared distances instead of true Euclidean")
    parser.add_argument("--save_dir", type=str, default="", help="Output metrics json path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.exp_name)
        ckpt_path = os.path.join(default_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(default_dir, "latest_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    model = ArbitraryPointFlowTransformer(
        in_channels=6,
        out_channels=6,
        embed_dim=saved_args.get("embed_dim", 256),
        depth=saved_args.get("depth", 6),
        num_heads=saved_args.get("num_heads", 8),
        mlp_ratio=4.0,
        img_channels=1,
        cond_drop_prob=0.0,
        num_freqs=saved_args.get("num_freqs", 6)
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Wrap with MeanFlow generative sampler
    flow_model = MeanFlow(model=model, accept_cond=True).to(device)
    flow_model.eval()

    test_dataset = Digit3D(
        root="~/.conquer3d/",
        train=False,
        download=True,
        cached=True,
        return_img=True
    )
    num_eval = min(args.num_samples, len(test_dataset))

    print("==================================================================")
    print("   Arbitrary-Resolution Point Cloud Evaluation (conquer3d.ops)   ")
    print("==================================================================")
    print(f"Evaluation Samples: {num_eval}")
    print(f"Tested Resolutions: {args.resolutions}")
    print(f"Sampling Steps    : {args.steps}")
    print(f"Distance Metric   : {'Squared Euclidean' if args.squared else 'True Euclidean'}")
    print(f"Device            : {device}")
    print("------------------------------------------------------------------")

    results = {}

    with torch.no_grad():
        for P in args.resolutions:
            print(f"\nEvaluating Resolution: {P} Points...")
            total_cd = 0.0
            total_hd = 0.0
            total_normal_sim = 0.0
            total_time_ms = 0.0

            for idx in tqdm(range(num_eval), desc=f"Testing P={P}"):
                v, f, label, img_t = test_dataset[idx]
                img_gpu = img_t.unsqueeze(0).to(device)

                # Sample fresh P ground-truth points and face normals directly from test mesh
                mesh = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
                points_np, face_indices = trimesh.sample.sample_surface(mesh, P)
                normals_np = mesh.face_normals[face_indices]

                gt_pts = torch.tensor(points_np, dtype=torch.float32, device=device)
                gt_normals = torch.tensor(normals_np, dtype=torch.float32, device=device)

                t0 = time.time()
                samples = flow_model.sample(
                    batch_size=1,
                    data_shape=(P, 6),
                    steps=args.steps,
                    cond=img_gpu
                )
                torch.cuda.synchronize()
                total_time_ms += (time.time() - t0) * 1000

                pred_pts = samples[0, :, :3]
                pred_normals = F.normalize(samples[0, :, 3:6], p=2, dim=-1)

                cd, hd, n_sim = compute_metrics_with_conquer3d(
                    pred_pts, pred_normals,
                    gt_pts, gt_normals,
                    squared=args.squared
                )
                total_cd += cd
                total_hd += hd
                total_normal_sim += n_sim

            avg_cd = total_cd / num_eval
            avg_hd = total_hd / num_eval
            avg_normal_sim = total_normal_sim / num_eval
            avg_latency_ms = total_time_ms / num_eval
            fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

            results[str(P)] = {
                "num_points": P,
                "chamfer_distance": round(avg_cd, 6),
                "hausdorff_distance": round(avg_hd, 6),
                "normal_cosine_similarity": round(avg_normal_sim, 4),
                "latency_ms": round(avg_latency_ms, 2),
                "throughput_fps": round(fps, 1),
            }

            print(f"  --> P={P:5d} | CD: {avg_cd:.6f} | HD: {avg_hd:.6f} | Normal Sim: {avg_normal_sim:.4f} | Latency: {avg_latency_ms:.2f} ms ({fps:.1f} FPS)")

    save_dir = args.save_dir if args.save_dir else os.path.dirname(ckpt_path)
    eval_path = os.path.join(save_dir, "eval_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===========================================================================================")
    print("                              Benchmark Summary Table (conquer3d.ops)                      ")
    print("===========================================================================================")
    print(f"{'Points (P)':<12} | {'Chamfer Dist (↓)':<18} | {'Hausdorff (↓)':<16} | {'Normal Sim (↑)':<16} | {'Latency (ms)':<14} | {'FPS (↑)':<10}")
    print("-" * 95)
    for P in args.resolutions:
        r = results[str(P)]
        print(f"{r['num_points']:<12} | {r['chamfer_distance']:<18.6f} | {r['hausdorff_distance']:<16.6f} | {r['normal_cosine_similarity']:<16.4f} | {r['latency_ms']:<14.2f} | {r['throughput_fps']:<10.1f}")
    print("===========================================================================================")
    print(f"Results saved to: {eval_path}")


if __name__ == "__main__":
    main()
