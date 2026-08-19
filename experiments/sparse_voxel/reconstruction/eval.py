import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.ops.distance import chamfer_distance
from experiments.sparse_voxel.reconstruction.models import SimpleSparseVAE
from torchsparse import SparseTensor


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sparse Voxel VAE Reconstruction on Digit3D Test Set")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). Defaults to sparse_reconstruction.pt in current folder.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples to evaluate (0 for full test set)")
    parser.add_argument("--out_file", type=str, default="eval_results.json", help="Output JSON metrics filename")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "sparse_reconstruction.pt")

    if not os.path.exists(ckpt_path):
        parent_ckpt = os.path.abspath(os.path.join(curr_dir, "../sparse_reconstruction.pt"))
        if os.path.exists(parent_ckpt):
            ckpt_path = parent_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {parent_ckpt}")

    print("==================================================================")
    print("        Digit3D Sparse Voxel VAE Reconstruction Benchmark         ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Num Samples  : {args.num_samples if args.num_samples > 0 else 'Full Test Set'}")
    print(f"Device       : {args.device}")
    print("------------------------------------------------------------------")

    # 1. Load Model
    model = SimpleSparseVAE(
        in_channels=8,
        hidden_channels=32,
        latent_channels=16,
        out_channels=8,
        num_layers=3
    ).to(args.device)

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Test Dataset
    print("Initializing test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=bmesh_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    per_sample_metrics = []
    cd_scores = []
    mse_scores = []

    limit = args.num_samples if args.num_samples > 0 else len(test_dataset)

    print(f"Evaluating reconstruction quality across {limit} test samples...")
    with torch.no_grad():
        for i, (bmesh, batched_labels) in enumerate(tqdm(test_loader, total=limit, desc="Evaluating")):
            if i >= limit:
                break

            bmesh = bmesh.to(args.device)
            orig_points = bmesh.vertices.float()

            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            batched_coords = batched_coords.to(args.device)
            batched_sdf = batched_sdf.to(args.device)

            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())

            # Forward pass
            pred_feats, _ = model(x)
            mse_val = torch.nn.functional.mse_loss(pred_feats, batched_sdf).item()
            mse_scores.append(mse_val)

            # Marching cubes surface extraction
            pred_unique_vertices, pred_local_voxels, pred_merged_sdfs = c3d.conversion.sparse2voxel(
                batched_coords, pred_feats,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )
            pred_vert, pred_tri, _, _ = c3d.ops.diff_marching_cubes(pred_unique_vertices, pred_local_voxels, pred_merged_sdfs, iso=0.0)

            if len(pred_vert) == 0:
                cd_val = float("inf")
            else:
                pred_vert = pred_vert.float()
                cd_val = chamfer_distance(orig_points, pred_vert, squared=True).item()

            if cd_val != float("inf"):
                cd_scores.append(cd_val)

            per_sample_metrics.append({
                "sample_idx": i,
                "label": int(batched_labels[0].item()),
                "sdf_mse": mse_val,
                "chamfer_distance": cd_val
            })

    mean_cd = float(np.mean(cd_scores)) if cd_scores else float("nan")
    median_cd = float(np.median(cd_scores)) if cd_scores else float("nan")
    mean_mse = float(np.mean(mse_scores)) if mse_scores else float("nan")

    results = {
        "mean_chamfer_distance": mean_cd,
        "median_chamfer_distance": median_cd,
        "mean_sdf_mse": mean_mse,
        "total_evaluated_samples": len(per_sample_metrics),
        "valid_mesh_reconstructions": len(cd_scores),
        "per_sample_metrics": per_sample_metrics
    }

    print("\n" + "=" * 65)
    print("                 RECONSTRUCTION BENCHMARK SUMMARY                 ")
    print("=" * 65)
    print(f"Total Evaluated Samples     : {len(per_sample_metrics)}")
    print(f"Valid Mesh Extractions      : {len(cd_scores)} / {len(per_sample_metrics)}")
    print(f"Mean SDF MSE                : {mean_mse:.6f}")
    print(f"Mean Chamfer Distance       : {mean_cd:.6f}")
    print(f"Median Chamfer Distance     : {median_cd:.6f}")
    print("-" * 65)

    out_json_path = os.path.join(curr_dir, args.out_file)
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Benchmark evaluation results saved to: {out_json_path}")


if __name__ == "__main__":
    main()
