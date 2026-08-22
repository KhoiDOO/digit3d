import argparse
import json
import os
import sys
import time
import torch
from tqdm.auto import tqdm
import trimesh

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import sparse_coo2dense_occ, dense_occ2sparse_coo
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.ops.distance import chamfer_distance, hausdorff_distance
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from experiments.sparse_voxel.pccond_generation.stage1.models import StructureDiT
from experiments.sparse_voxel.pccond_generation.stage2.models import SparseVertexSDFTransformer


def sample_mesh_points(vertices, faces, num_samples: int = 1000):
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    pts_np, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals_np = mesh.face_normals[face_idx]
    return torch.tensor(pts_np, dtype=torch.float32), torch.tensor(normals_np, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Full End-to-End Two-Stage Point-Conditioned 3D Generation Pipeline on Digit3D")
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Stage 1 checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, default="", help="Stage 2 checkpoint")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--num_points", type=int, default=512, help="Number of input points")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")
    parser.add_argument("--stage1_steps", type=int, default=16, help="Stage 1 MeanFlow steps")
    parser.add_argument("--stage2_steps", type=int, default=16, help="Stage 2 Rectified Flow steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_ckpt = args.stage1_ckpt if args.stage1_ckpt else os.path.join(curr_dir, "stage1", "stage1_structure.pt")
    stage2_ckpt = args.stage2_ckpt if args.stage2_ckpt else os.path.join(curr_dir, "stage2", "stage2_sdf.pt")

    print("==================================================================")
    print("  Digit3D Full End-to-End Point-Conditioned Pipeline Evaluation   ")
    print("==================================================================")
    print(f"Stage 1 Checkpoint : {stage1_ckpt}")
    print(f"Stage 2 Checkpoint : {stage2_ckpt}")
    print(f"Num Samples        : {args.num_samples}")
    print(f"Num Points         : {args.num_points}")
    print(f"Occupancy Threshold: {args.threshold}")
    print(f"Stage 1 Steps      : {args.stage1_steps}")
    print(f"Stage 2 Steps      : {args.stage2_steps}")
    print("------------------------------------------------------------------")

    # Load Models
    model_s1 = StructureDiT(
        grid_res=32, patch_size=4, in_channels=1, out_channels=1,
        embed_dim=256, depth=6, num_heads=4, cond_dim=256, pc_channels=6
    ).to(args.device)
    if os.path.exists(stage1_ckpt):
        model_s1.load_state_dict(torch.load(stage1_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 1 weights from {stage1_ckpt}")
    model_s1.eval()
    mf_s1 = MeanFlow(model=model_s1, accept_cond=True, data_shape=(1, 32, 32, 32)).to(args.device)

    model_s2 = SparseVertexSDFTransformer(
        in_channels=1, out_channels=1,
        embed_dim=256, depth=6, num_heads=4, cond_dim=256, pc_channels=6
    ).to(args.device)
    if os.path.exists(stage2_ckpt):
        model_s2.load_state_dict(torch.load(stage2_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 2 weights from {stage2_ckpt}")
    model_s2.eval()
    rf_s2 = RectifiedFlow(model=model_s2).to(args.device)

    # Dataset
    dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
    num_eval = min(args.num_samples, len(dataset))

    total_iou = 0.0
    total_cd = 0.0
    total_hd = 0.0
    total_latency = 0.0
    valid_count = 0

    print(f"Running full evaluation on {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Full Pipeline"):
        v_gt, f_gt, label = dataset[idx]
        mesh_np = trimesh.Trimesh(vertices=v_gt.numpy(), faces=f_gt.numpy(), process=False)
        pts_np, face_idx = trimesh.sample.sample_surface(mesh_np, args.num_points)
        normals_np = mesh_np.face_normals[face_idx]
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        normals_t = torch.tensor(normals_np, dtype=torch.float32)
        pc_feat = torch.cat([pts_t, normals_t], dim=-1).permute(1, 0).unsqueeze(0).to(args.device)

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

        start_t = time.time()
        with torch.no_grad():
            # Stage 1: Generate occupancy via MeanFlow
            pred_occ = mf_s1.sample(batch_size=1, cond=pc_feat, steps=args.stage1_steps)
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

            dummy_sdfs = torch.zeros((num_active, 8), device=args.device, dtype=torch.float32)
            unique_vertices, local_voxels, _ = c3d.conversion.sparse2voxel(
                sparse_coords, dummy_sdfs,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )

            M = unique_vertices.shape[0]

            # Stage 2: Predict Vertex SDFs via Rectified Flow
            c_pc = model_s2.pc_encoder(pc_feat)
            cond_tensor = torch.cat([unique_vertices.unsqueeze(0), c_pc.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            model_s2.set_seq_lens([M])

            pred_sdfs_batch = rf_s2.sample(
                data_shape=(M, 1), noise=noise, cond=cond_tensor, steps=args.stage2_steps
            )
            pred_sdfs = pred_sdfs_batch[0, :, 0]

            # Marching Cubes
            recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(
                unique_vertices, local_voxels, pred_sdfs, iso=0.0
            )

        latency = (time.time() - start_t) * 1000.0
        total_latency += latency

        # Stage 1 Voxel IoU
        pred_bin = (pred_occ[0, 0] > args.threshold)
        intersection = (pred_bin & gt_occ[0, 0]).sum().float().item()
        union = (pred_bin | gt_occ[0, 0]).sum().float().item()
        iou = intersection / union if union > 0 else 0.0
        total_iou += iou

        # Stage 2 Mesh Metrics
        if recon_verts.shape[0] > 0 and recon_tris.shape[0] > 0:
            pred_sample_pts, _ = sample_mesh_points(recon_verts, recon_tris, 1000)
            gt_sample_pts, _ = sample_mesh_points(v_gt, f_gt, 1000)

            cd = chamfer_distance(pred_sample_pts.to(args.device), gt_sample_pts.to(args.device)).item()
            hd = hausdorff_distance(pred_sample_pts.to(args.device), gt_sample_pts.to(args.device)).item()

            total_cd += cd
            total_hd += hd
            valid_count += 1

    avg_iou = total_iou / num_eval
    avg_latency = total_latency / num_eval
    avg_cd = (total_cd / valid_count) if valid_count > 0 else float("nan")
    avg_hd = (total_hd / valid_count) if valid_count > 0 else float("nan")

    print("\n" + "=" * 60)
    print("      END-TO-END POINT-CONDITIONED BENCHMARK RESULTS          ")
    print("=" * 60)
    print(f"Evaluated Samples    : {num_eval}")
    print(f"Stage 1 Mean IoU     : {avg_iou * 100:.2f}%")
    print(f"Valid Mesh Recoveries: {valid_count}/{num_eval}")
    print(f"Mean Chamfer Distance: {avg_cd:.6f}")
    print(f"Mean Hausdorff Dist  : {avg_hd:.6f}")
    print(f"Mean Latency         : {avg_latency:.2f} ms ({1000.0/avg_latency:.1f} FPS)")
    print("=" * 60)

    results = {
        "num_eval_samples": num_eval,
        "stage1_mean_iou": round(avg_iou, 4),
        "valid_mesh_recoveries": f"{valid_count}/{num_eval}",
        "mean_chamfer_distance": round(avg_cd, 6) if not torch.isnan(torch.tensor(avg_cd)) else None,
        "mean_hausdorff_distance": round(avg_hd, 6) if not torch.isnan(torch.tensor(avg_hd)) else None,
        "mean_latency_ms": round(avg_latency, 2),
        "stage1_steps": args.stage1_steps,
        "stage2_steps": args.stage2_steps,
        "threshold": args.threshold
    }
    with open(os.path.join(curr_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
