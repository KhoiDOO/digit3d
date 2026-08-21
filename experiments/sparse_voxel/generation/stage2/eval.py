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
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.generation.stage2.models import SparseVertexSDFTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Sparse Vertex SDF MeanFlow on Digit3D")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--steps", type=int, default=16, help="Sampling steps (1 = single-step fast sampling)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "stage2_sdf.pt")

    print("==================================================================")
    print("     Digit3D Stage 2 Sparse Vertex SDF MeanFlow Evaluation        ")
    print("==================================================================")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Steps      : {args.steps}")
    print("------------------------------------------------------------------")

    # 1. Load Model & MeanFlow
    model = SparseVertexSDFTransformer(
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
    mf = MeanFlow(model=model, accept_cond=True).to(args.device)

    # 2. Dataset
    dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)
    num_eval = min(args.num_samples, len(dataset))

    total_sdf_l1 = 0.0
    total_sign_acc = 0.0
    total_latency = 0.0

    print(f"Evaluating on {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Stage 2"):
        raw_sample = dataset[idx]
        v_gt, f_gt = raw_sample[0], raw_sample[1]
        img = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

        # Extract GT vertex graph & scalar SDFs
        mesh = BTriangleMesh(
            vertices=v_gt.to(args.device),
            faces=f_gt.to(args.device),
            vertbids=torch.zeros(v_gt.shape[0], dtype=torch.int32, device=args.device),
            facebids=torch.zeros(f_gt.shape[0], dtype=torch.int32, device=args.device),
            batch_size=1
        )
        with torch.no_grad():
            sparse_coords, sparse_sdfs = mesh2sparse(mesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            u_verts, local_vox, gt_sdfs = c3d.conversion.sparse2voxel(
                sparse_coords, sparse_sdfs,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )

        M = u_verts.shape[0]
        img_batch = img.unsqueeze(0).to(args.device)

        start_t = time.time()
        with torch.no_grad():
            c_img = model.img_encoder(img_batch)
            cond_tensor = torch.cat([u_verts.unsqueeze(0), c_img.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            pred_sdfs_batch = mf.sample(
                data_shape=(M, 1),
                noise=noise,
                cond=cond_tensor,
                steps=args.steps
            )
        latency = time.time() - start_t
        total_latency += latency

        pred_sdfs = pred_sdfs_batch[0, :, 0]
        l1_diff = (pred_sdfs - gt_sdfs).abs().mean().item()
        sign_match = ((pred_sdfs < 0) == (gt_sdfs < 0)).float().mean().item()

        total_sdf_l1 += l1_diff
        total_sign_acc += sign_match

    avg_l1 = total_sdf_l1 / num_eval
    avg_sign = (total_sign_acc / num_eval) * 100.0
    avg_lat = (total_latency / num_eval) * 1000.0

    print("\n==================================================================")
    print("                Stage 2 MeanFlow Evaluation Results               ")
    print("==================================================================")
    print(f"Vertex SDF L1 Error : {avg_l1:.4f}")
    print(f"SDF Sign Accuracy   : {avg_sign:.2f}%")
    print(f"Avg Inference Time  : {avg_lat:.2f} ms / sample")
    print("==================================================================")


if __name__ == "__main__":
    main()
