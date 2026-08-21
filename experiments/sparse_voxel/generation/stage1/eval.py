import argparse
import json
import os
import sys
import time
import torch
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import sparse_coo2dense_occ
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.generation.stage1.models import StructureDiT


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Structure DiT MeanFlow Occupancy Generation on Digit3D")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")
    parser.add_argument("--steps", type=int, default=16, help="Sampling steps (1 = single-step fast sampling)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "stage1_structure.pt")

    print("==================================================================")
    print("        Digit3D Stage 1 Structure DiT MeanFlow Evaluation         ")
    print("==================================================================")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Threshold  : {args.threshold}")
    print(f"Steps      : {args.steps}")
    print("------------------------------------------------------------------")

    # 1. Load Model & MeanFlow
    model = StructureDiT(
        grid_res=32,
        patch_size=8,
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=4,
        cond_dim=256,
        img_channels=1
    ).to(args.device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found.")

    model.eval()
    mf = MeanFlow(model=model, accept_cond=True, data_shape=(1, 32, 32, 32)).to(args.device)

    # 2. Dataset
    dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)
    num_eval = min(args.num_samples, len(dataset))

    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_latency = 0.0

    print(f"Evaluating on {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Stage 1"):
        raw_sample = dataset[idx]
        v_gt, f_gt, label = raw_sample[0], raw_sample[1], raw_sample[2]
        img = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

        # Ground Truth Occupancy
        mesh = c3d.data_structure.bmesh.BTriangleMesh(
            vertices=v_gt.to(args.device),
            faces=f_gt.to(args.device),
            vertbids=torch.zeros(v_gt.shape[0], dtype=torch.int32, device=args.device),
            facebids=torch.zeros(f_gt.shape[0], dtype=torch.int32, device=args.device),
            batch_size=1
        )
        with torch.no_grad():
            sparse_coords, _ = mesh2sparse(mesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            gt_occ = sparse_coo2dense_occ(sparse_coords, 1, (32, 32, 32)) > 0.5

        # Generated Occupancy with MeanFlow
        img_batch = img.unsqueeze(0).to(args.device)
        start_t = time.time()
        with torch.no_grad():
            pred_occ = mf.sample(batch_size=1, cond=img_batch, steps=args.steps)
        latency = time.time() - start_t
        total_latency += latency

        pred_binary = pred_occ > args.threshold

        intersection = (pred_binary & gt_occ).sum().float().item()
        union = (pred_binary | gt_occ).sum().float().item()
        pred_pos = pred_binary.sum().float().item()
        gt_pos = gt_occ.sum().float().item()

        iou = intersection / max(union, 1.0)
        precision = intersection / max(pred_pos, 1.0)
        recall = intersection / max(gt_pos, 1.0)

        total_iou += iou
        total_precision += precision
        total_recall += recall

    avg_iou = total_iou / num_eval
    avg_prec = total_precision / num_eval
    avg_rec = total_recall / num_eval
    avg_lat = (total_latency / num_eval) * 1000.0

    print("\n==================================================================")
    print("                Stage 1 MeanFlow Evaluation Results               ")
    print("==================================================================")
    print(f"Occupancy IoU      : {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"Occupancy Precision: {avg_prec:.4f} ({avg_prec*100:.2f}%)")
    print(f"Occupancy Recall   : {avg_rec:.4f} ({avg_rec*100:.2f}%)")
    print(f"Avg Inference Time : {avg_lat:.2f} ms / sample")
    print("==================================================================")


if __name__ == "__main__":
    main()
