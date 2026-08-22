import argparse
import json
import os
import sys
import time
import torch
from tqdm.auto import tqdm
import trimesh

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import sparse_coo2dense_occ, dense_occ2sparse_coo
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.pccond_generation.stage1.models import StructureDiT


def compute_voxel_iou(pred_occ: torch.Tensor, gt_occ: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = pred_occ > threshold
    gt_bin = gt_occ > threshold
    intersection = (pred_bin & gt_bin).sum().float().item()
    union = (pred_bin | gt_bin).sum().float().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Point-Conditioned Structure DiT on Digit3D")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_points", type=int, default=512, help="Number of input points")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")
    parser.add_argument("--steps", type=int, default=16, help="Sampling steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "stage1_structure.pt")

    print("==================================================================")
    print("  Digit3D Stage 1 Point-Conditioned Voxel Occupancy Evaluation    ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Num Samples  : {args.num_samples}")
    print(f"Num Points   : {args.num_points}")
    print(f"Threshold    : {args.threshold}")
    print(f"Steps        : {args.steps}")
    print("------------------------------------------------------------------")

    model = StructureDiT(
        grid_res=32,
        patch_size=4,
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=4,
        cond_dim=256,
        pc_channels=6
    ).to(args.device)

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.eval()
    mf = MeanFlow(model=model, accept_cond=True, data_shape=(1, 32, 32, 32)).to(args.device)

    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
    num_eval = min(args.num_samples, len(test_dataset))

    total_iou = 0.0
    total_latency_ms = 0.0
    per_class_ious = {c: [] for c in range(10)}

    print(f"Evaluating {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Stage 1"):
        v, f, label = test_dataset[idx]
        mesh_np = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
        pts_np, face_idx = trimesh.sample.sample_surface(mesh_np, args.num_points)
        normals_np = mesh_np.face_normals[face_idx]
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        normals_t = torch.tensor(normals_np, dtype=torch.float32)
        pc_feat = torch.cat([pts_t, normals_t], dim=-1).permute(1, 0).unsqueeze(0).to(args.device) # [1, 6, N]

        # Ground truth voxel occupancy
        bmesh = BTriangleMesh(
            vertices=v.to(args.device),
            faces=f.to(args.device),
            vertbids=torch.zeros(v.shape[0], dtype=torch.int32, device=args.device),
            facebids=torch.zeros(f.shape[0], dtype=torch.int32, device=args.device),
            batch_size=1
        )
        with torch.no_grad():
            sparse_coords, _ = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            gt_occ = sparse_coo2dense_occ(sparse_coords, 1, (32, 32, 32))

        t0 = time.time()
        with torch.no_grad():
            pred_occ = mf.sample(batch_size=1, cond=pc_feat, steps=args.steps)
        latency = (time.time() - t0) * 1000.0

        iou = compute_voxel_iou(pred_occ[0, 0], gt_occ[0, 0], threshold=args.threshold)
        total_iou += iou
        total_latency_ms += latency
        per_class_ious[int(label)].append(iou)

    mean_iou = total_iou / num_eval
    mean_latency = total_latency_ms / num_eval

    print("\n" + "=" * 60)
    print("           STAGE 1 POINT-CONDITIONED EVALUATION RESULTS       ")
    print("=" * 60)
    print(f"Mean Voxel IoU       : {mean_iou * 100:.2f}%")
    print(f"Mean Latency         : {mean_latency:.2f} ms ({1000.0/mean_latency:.1f} FPS)")
    for c in range(10):
        if len(per_class_ious[c]) > 0:
            c_mean = sum(per_class_ious[c]) / len(per_class_ious[c])
            print(f"  Class {c} ({len(per_class_ious[c]):2d} samples) : {c_mean * 100:.2f}% IoU")
    print("=" * 60)

    results = {
        "mean_iou": round(mean_iou, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "num_eval_samples": num_eval,
        "steps": args.steps,
        "threshold": args.threshold,
        "per_class_iou": {str(c): round(sum(per_class_ious[c]) / len(per_class_ious[c]), 4) for c in range(10) if len(per_class_ious[c]) > 0}
    }
    with open(os.path.join(curr_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
