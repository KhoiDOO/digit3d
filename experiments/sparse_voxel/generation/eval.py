import argparse
import json
import os
import sys
import time
import torch
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import sparse_coo2dense_occ, dense_occ2sparse_coo
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from experiments.sparse_voxel.generation.stage1.models import StructureDiT
from experiments.sparse_voxel.generation.stage2.models import SparseVertexSDFTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Full End-to-End Two-Stage MeanFlow 3D Generation Pipeline on Digit3D")
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Stage 1 checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, default="", help="Stage 2 checkpoint")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")
    parser.add_argument("--stage1_steps", type=int, default=16, help="Stage 1 MeanFlow steps")
    parser.add_argument("--stage2_steps", type=int, default=16, help="Stage 2 MeanFlow steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_ckpt = args.stage1_ckpt if args.stage1_ckpt else os.path.join(curr_dir, "stage1", "stage1_structure.pt")
    stage2_ckpt = args.stage2_ckpt if args.stage2_ckpt else os.path.join(curr_dir, "stage2", "stage2_sdf.pt")

    print("==================================================================")
    print("     Digit3D End-to-End Two-Stage MeanFlow Pipeline Eval          ")
    print("==================================================================")
    print(f"Stage 1 Checkpoint : {stage1_ckpt}")
    print(f"Stage 2 Checkpoint : {stage2_ckpt}")
    print(f"Num Samples        : {args.num_samples}")
    print(f"Occupancy Threshold: {args.threshold}")
    print("------------------------------------------------------------------")

    # Load Models
    model_s1 = StructureDiT(
        grid_res=32, patch_size=4, in_channels=1, out_channels=1,
        embed_dim=256, depth=6, num_heads=4, cond_dim=256, img_channels=1
    ).to(args.device)
    if os.path.exists(stage1_ckpt):
        model_s1.load_state_dict(torch.load(stage1_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 1 weights from {stage1_ckpt}")
    model_s1.eval()
    mf_s1 = MeanFlow(model=model_s1, accept_cond=True, data_shape=(1, 32, 32, 32)).to(args.device)

    model_s2 = SparseVertexSDFTransformer(
        in_channels=1, out_channels=1,
        embed_dim=256, depth=6, num_heads=4, cond_dim=256, img_channels=1
    ).to(args.device)
    if os.path.exists(stage2_ckpt):
        model_s2.load_state_dict(torch.load(stage2_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 2 weights from {stage2_ckpt}")
    model_s2.eval()
    rf_s2 = RectifiedFlow(model=model_s2).to(args.device)

    # Dataset
    dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)
    num_eval = min(args.num_samples, len(dataset))

    total_iou = 0.0
    total_mesh_verts = 0
    total_mesh_faces = 0
    total_latency = 0.0
    valid_mesh_count = 0

    print(f"Running full evaluation on {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Full MeanFlow Pipeline"):
        raw_sample = dataset[idx]
        v_gt, f_gt = raw_sample[0], raw_sample[1]
        img = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

        mesh_gt = BTriangleMesh(
            vertices=v_gt.to(args.device),
            faces=f_gt.to(args.device),
            vertbids=torch.zeros(v_gt.shape[0], dtype=torch.int32, device=args.device),
            facebids=torch.zeros(f_gt.shape[0], dtype=torch.int32, device=args.device),
            batch_size=1
        )
        with torch.no_grad():
            sparse_coords_gt, _ = mesh2sparse(mesh_gt, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            gt_occ = sparse_coo2dense_occ(sparse_coords_gt, 1, (32, 32, 32)) > 0.5

        img_batch = img.unsqueeze(0).to(args.device)

        start_t = time.time()
        with torch.no_grad():
            # Stage 1: Generate occupancy via MeanFlow
            pred_occ = mf_s1.sample(batch_size=1, cond=img_batch, steps=args.stage1_steps)
            sparse_coords = dense_occ2sparse_coo(pred_occ, threshold=args.threshold)
            num_active = int(sparse_coords.shape[0])

            if num_active > 3000:
                flat_occ = pred_occ[0, 0, sparse_coords[:, 1].long(), sparse_coords[:, 2].long(), sparse_coords[:, 3].long()]
                topk_indices = torch.topk(flat_occ, k=3000).indices
                sparse_coords = sparse_coords[topk_indices]
                num_active = 3000
            elif num_active == 0:
                sparse_coords = torch.tensor([[0, 15, 15, 15], [0, 16, 16, 16]], dtype=torch.int32, device=args.device)
                num_active = 2

            # Compute IoU
            pred_binary = pred_occ > args.threshold
            intersection = (pred_binary & gt_occ).sum().float().item()
            union = (pred_binary | gt_occ).sum().float().item()
            total_iou += (intersection / max(union, 1.0))

            # Stage 2: Generate vertex SDFs via MeanFlow
            dummy_sdfs = torch.zeros((num_active, 8), device=args.device, dtype=torch.float32)
            u_verts, local_vox, _ = c3d.conversion.sparse2voxel(
                sparse_coords, dummy_sdfs,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )

            M = u_verts.shape[0]
            c_img = model_s2.img_encoder(img_batch)
            cond_tensor = torch.cat([u_verts.unsqueeze(0), c_img.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            model_s2.set_seq_lens([M])
            pred_sdfs_batch = rf_s2.sample(
                data_shape=(M, 1),
                noise=noise,
                cond=cond_tensor,
                steps=args.stage2_steps
            )
            pred_sdfs = pred_sdfs_batch[0, :, 0]

            # Surface extraction
            recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(u_verts, local_vox, pred_sdfs, iso=0.0)

        latency = time.time() - start_t
        total_latency += latency

        if recon_verts.shape[0] > 0 and recon_tris.shape[0] > 0:
            valid_mesh_count += 1
            total_mesh_verts += recon_verts.shape[0]
            total_mesh_faces += recon_tris.shape[0]

    avg_iou = total_iou / num_eval
    avg_lat = (total_latency / num_eval) * 1000.0
    mesh_success_rate = (valid_mesh_count / num_eval) * 100.0
    avg_verts = total_mesh_verts / max(valid_mesh_count, 1)
    avg_faces = total_mesh_faces / max(valid_mesh_count, 1)

    print("\n==================================================================")
    print("            Full Pipeline MeanFlow Evaluation Results             ")
    print("==================================================================")
    print(f"Stage 1 Occupancy IoU  : {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"Mesh Extraction Success: {mesh_success_rate:.1f}% ({valid_mesh_count}/{num_eval})")
    print(f"Avg Reconstructed Verts: {avg_verts:.1f}")
    print(f"Avg Reconstructed Faces: {avg_faces:.1f}")
    print(f"Avg End-to-End Latency : {avg_lat:.2f} ms / sample")
    print("==================================================================")


if __name__ == "__main__":
    main()
