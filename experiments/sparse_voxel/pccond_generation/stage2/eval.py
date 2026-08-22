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
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.ops.distance import chamfer_distance, hausdorff_distance
from rectified_flow_pytorch import RectifiedFlow
from experiments.sparse_voxel.pccond_generation.stage2.models import SparseVertexSDFTransformer


def sample_mesh_points(vertices, faces, num_samples: int = 1000):
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    pts_np, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals_np = mesh.face_normals[face_idx]
    return torch.tensor(pts_np, dtype=torch.float32), torch.tensor(normals_np, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Point-Conditioned Vertex SDF Rectified Flow on Digit3D")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--num_points", type=int, default=512, help="Number of input points")
    parser.add_argument("--steps", type=int, default=16, help="Sampling steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "stage2_sdf.pt")

    print("==================================================================")
    print("  Digit3D Stage 2 Point-Conditioned Vertex SDF Evaluation         ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Num Samples  : {args.num_samples}")
    print(f"Num Points   : {args.num_points}")
    print(f"Steps        : {args.steps}")
    print("------------------------------------------------------------------")

    model = SparseVertexSDFTransformer(
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
    rf = RectifiedFlow(model=model).to(args.device)

    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
    num_eval = min(args.num_samples, len(test_dataset))

    total_cd = 0.0
    total_hd = 0.0
    total_latency = 0.0
    valid_count = 0

    print(f"Evaluating {num_eval} samples...")
    for idx in tqdm(range(num_eval), desc="Evaluating Stage 2"):
        v, f, label = test_dataset[idx]
        mesh_np = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
        pts_np, face_idx = trimesh.sample.sample_surface(mesh_np, args.num_points)
        normals_np = mesh_np.face_normals[face_idx]
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        normals_t = torch.tensor(normals_np, dtype=torch.float32)
        pc_feat = torch.cat([pts_t, normals_t], dim=-1).permute(1, 0).unsqueeze(0).to(args.device)

        bmesh = BTriangleMesh(
            vertices=v.to(args.device),
            faces=f.to(args.device),
            vertbids=torch.zeros(v.shape[0], dtype=torch.int32, device=args.device),
            facebids=torch.zeros(f.shape[0], dtype=torch.int32, device=args.device),
            batch_size=1
        )
        with torch.no_grad():
            sparse_coords, sparse_sdfs = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)

        unique_vertices, local_voxels, _ = c3d.conversion.sparse2voxel(
            sparse_coords, sparse_sdfs,
            grid_min=[-1.2, -1.2, -1.2],
            grid_max=[1.2, 1.2, 1.2],
            res=[32, 32, 32]
        )

        M = unique_vertices.shape[0]

        t0 = time.time()
        with torch.no_grad():
            c_pc = model.pc_encoder(pc_feat)
            cond_tensor = torch.cat([unique_vertices.unsqueeze(0), c_pc.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            model.set_seq_lens([M])

            pred_sdfs_batch = rf.sample(data_shape=(M, 1), noise=noise, cond=cond_tensor, steps=args.steps)
            pred_sdfs = pred_sdfs_batch[0, :, 0]

            recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(
                unique_vertices, local_voxels, pred_sdfs, iso=0.0
            )

        latency = (time.time() - t0) * 1000.0

        if recon_verts.shape[0] > 0 and recon_tris.shape[0] > 0:
            pred_sample_pts, _ = sample_mesh_points(recon_verts, recon_tris, 1000)
            gt_sample_pts, _ = sample_mesh_points(v, f, 1000)

            cd = chamfer_distance(pred_sample_pts.to(args.device), gt_sample_pts.to(args.device)).item()
            hd = hausdorff_distance(pred_sample_pts.to(args.device), gt_sample_pts.to(args.device)).item()

            total_cd += cd
            total_hd += hd
            total_latency += latency
            valid_count += 1

    if valid_count > 0:
        avg_cd = total_cd / valid_count
        avg_hd = total_hd / valid_count
        avg_latency = total_latency / valid_count

        print("\n" + "=" * 60)
        print("           STAGE 2 POINT-CONDITIONED EVALUATION RESULTS       ")
        print("=" * 60)
        print(f"Valid Mesh Recoveries: {valid_count}/{num_eval}")
        print(f"Mean Chamfer Distance: {avg_cd:.6f}")
        print(f"Mean Hausdorff Dist  : {avg_hd:.6f}")
        print(f"Mean Latency         : {avg_latency:.2f} ms ({1000.0/avg_latency:.1f} FPS)")
        print("=" * 60)

        results = {
            "mean_chamfer_distance": round(avg_cd, 6),
            "mean_hausdorff_distance": round(avg_hd, 6),
            "mean_latency_ms": round(avg_latency, 2),
            "valid_recoveries": f"{valid_count}/{num_eval}"
        }
        with open(os.path.join(curr_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
