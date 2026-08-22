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
from conquer3d.conversion.grid import dense_occ2sparse_coo
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from experiments.sparse_voxel.pccond_generation.stage1.models import StructureDiT
from experiments.sparse_voxel.pccond_generation.stage2.models import SparseVertexSDFTransformer


def save_ply_mesh(filename: str, vertices: torch.Tensor, faces: torch.Tensor):
    """
    Saves a 3D triangle mesh to an ASCII PLY file.
    """
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if faces is not None and len(faces) > 0:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None and len(faces) > 0:
            for face in faces:
                f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_point_cloud_ply(filename: str, points_and_normals: torch.Tensor):
    """
    Saves a 3D point cloud [6, N] or [N, 6] to an ASCII PLY file.
    """
    if points_and_normals.shape[0] == 6 and points_and_normals.shape[1] != 6:
        points_and_normals = points_and_normals.permute(1, 0)
    data = points_and_normals.detach().cpu().numpy()
    N = len(data)
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for p in data:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f} {p[5]:.6f}\n")


def save_voxel_quad_mesh_ply(filename: str, vertices: torch.Tensor, quads: torch.Tensor):
    """
    Saves sparse voxel grid as a quad mesh (vertices and 4-vertex quad faces) to an ASCII PLY file.
    """
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(quads):
        quads = quads.detach().cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if quads is not None and len(quads) > 0:
            f.write(f"element face {len(quads)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if quads is not None and len(quads) > 0:
            for q in quads:
                f.write(f"4 {int(q[0])} {int(q[1])} {int(q[2])} {int(q[3])}\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Two-Stage Point Cloud to 3D Mesh Generation (Stage 1 MeanFlow + Stage 2 Rectified Flow)")
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Path to Stage 1 checkpoint (.pt)")
    parser.add_argument("--stage2_ckpt", type=str, default="", help="Path to Stage 2 checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--full_class", action="store_true", help="Generate num_samples for each digit class (0-9)")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter for specific class digit (0-9)")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--num_points", type=int, default=512, help="Number of input conditioning points")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold for Stage 1 active voxel extraction")
    parser.add_argument("--stage1_steps", type=int, default=64, help="Stage 1 MeanFlow sampling steps (1 = fast, >1 = slow ODE)")
    parser.add_argument("--stage2_steps", type=int, default=64, help="Stage 2 Rectified Flow sampling steps (1 = fast, >1 = slow ODE)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory to save ply_samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_ckpt = args.stage1_ckpt if args.stage1_ckpt else os.path.join(base_dir, "stage1", "stage1_structure.pt")
    stage2_ckpt = args.stage2_ckpt if args.stage2_ckpt else os.path.join(base_dir, "stage2", "stage2_sdf.pt")
    ply_samples_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "ply_samples")
    os.makedirs(ply_samples_dir, exist_ok=True)

    print("==================================================================")
    print("  Digit3D End-to-End Point-Conditioned Two-Stage 3D Mesh Gen      ")
    print("==================================================================")
    print(f"Stage 1 Checkpoint : {stage1_ckpt}")
    print(f"Stage 2 Checkpoint : {stage2_ckpt}")
    print(f"Full Class Coverage: {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples        : {args.num_samples}")
    print(f"Num Points         : {args.num_points} (XYZ + Normals)")
    print(f"Occupancy Threshold: {args.threshold}")
    print(f"Stage 1 Steps      : {args.stage1_steps}")
    print(f"Stage 2 Steps      : {args.stage2_steps}")
    print(f"Output Directory   : {ply_samples_dir}")
    print("------------------------------------------------------------------")

    # 1. Load Stage 1 Model & MeanFlow
    print("Loading Stage 1 (Structure DiT MeanFlow)...")
    model_stage1 = StructureDiT(
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

    if os.path.exists(stage1_ckpt):
        model_stage1.load_state_dict(torch.load(stage1_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 1 weights from {stage1_ckpt}")
    else:
        print(f"[!] Warning: Stage 1 checkpoint {stage1_ckpt} not found. Running with initialized weights.")
    model_stage1.eval()
    mf_stage1 = MeanFlow(model=model_stage1, accept_cond=True, data_shape=(1, 32, 32, 32)).to(args.device)

    # 2. Load Stage 2 Model & Rectified Flow
    print("Loading Stage 2 (Sparse Vertex SDF Rectified Flow)...")
    model_stage2 = SparseVertexSDFTransformer(
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=4,
        cond_dim=256,
        pc_channels=6
    ).to(args.device)

    if os.path.exists(stage2_ckpt):
        model_stage2.load_state_dict(torch.load(stage2_ckpt, map_location=args.device))
        print(f"[*] Loaded Stage 2 weights from {stage2_ckpt}")
    else:
        print(f"[!] Warning: Stage 2 checkpoint {stage2_ckpt} not found. Running with initialized weights.")
    model_stage2.eval()
    rf_stage2 = RectifiedFlow(model=model_stage2).to(args.device)

    # 3. Load Test Dataset
    print("Loading test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)

    # 4. Filter Samples
    selected_indices = []
    if args.full_class:
        class_buckets = {c: [] for c in range(10)}
        for idx in range(len(test_dataset)):
            raw_sample = test_dataset[idx]
            label = int(raw_sample[2])
            if len(class_buckets[label]) < args.num_samples:
                class_buckets[label].append(idx)
            if all(len(b) >= args.num_samples for b in class_buckets.values()):
                break
        for c in sorted(class_buckets.keys()):
            selected_indices.extend(class_buckets[c])
    elif args.class_label >= 0:
        for idx in range(len(test_dataset)):
            raw_sample = test_dataset[idx]
            if int(raw_sample[2]) == args.class_label:
                selected_indices.append(idx)
                if len(selected_indices) >= args.num_samples:
                    break
    else:
        selected_indices = list(range(args.sample_offset, min(args.sample_offset + args.num_samples, len(test_dataset))))

    quad_template = torch.tensor([
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
    ], dtype=torch.int64, device=args.device)

    print(f"Running two-stage 3D generation on {len(selected_indices)} samples...")

    for i, idx in enumerate(tqdm(selected_indices, desc="Generating 3D Meshes")):
        v_gt, f_gt, label_val = test_dataset[idx]
        mesh_np = trimesh.Trimesh(vertices=v_gt.numpy(), faces=f_gt.numpy(), process=False)
        pts_np, face_idx = trimesh.sample.sample_surface(mesh_np, args.num_points)
        normals_np = mesh_np.face_normals[face_idx]
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        normals_t = torch.tensor(normals_np, dtype=torch.float32)
        pc_feat = torch.cat([pts_t, normals_t], dim=-1).permute(1, 0).unsqueeze(0).to(args.device) # [1, 6, N]

        # Save input point cloud
        pc_ply_name = f"sample_{i:03d}_class_{label_val}_input_pc.ply"
        save_point_cloud_ply(os.path.join(ply_samples_dir, pc_ply_name), pc_feat[0])

        start_time = time.time()

        # =====================================================================
        # STAGE 1: Generate Dense Occupancy via MeanFlow
        # =====================================================================
        t0_s1 = time.time()
        with torch.no_grad():
            pred_occ = mf_stage1.sample(batch_size=1, cond=pc_feat, steps=args.stage1_steps)

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
        cube_quads = local_voxels[:, quad_template].reshape(-1, 4)
        stage1_latency_ms = (time.time() - t0_s1) * 1000.0

        # Save Stage 1 Voxel Quad Mesh
        stage1_ply_name = f"sample_{i:03d}_class_{label_val}_stage1_voxels.ply"
        save_voxel_quad_mesh_ply(os.path.join(ply_samples_dir, stage1_ply_name), unique_vertices, cube_quads)

        # =====================================================================
        # STAGE 2: Generate 1D Scalar SDFs at Active Grid Vertices via Rectified Flow
        # =====================================================================
        t0_s2 = time.time()
        with torch.no_grad():
            c_pc = model_stage2.pc_encoder(pc_feat)
            cond_tensor = torch.cat([unique_vertices.unsqueeze(0), c_pc.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            model_stage2.set_seq_lens([M])
            pred_sdfs_batch = rf_stage2.sample(
                data_shape=(M, 1),
                noise=noise,
                cond=cond_tensor,
                steps=args.stage2_steps
            )
            pred_sdfs = pred_sdfs_batch[0, :, 0]

        # =====================================================================
        # Surface Extraction: Differentiable Marching Cubes
        # =====================================================================
        recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(
            unique_vertices, local_voxels, pred_sdfs, iso=0.0
        )
        stage2_latency_ms = (time.time() - t0_s2) * 1000.0
        total_latency = (time.time() - start_time) * 1000.0

        # Save Stage 2 Reconstructed 3D Mesh
        recon_ply_name = f"sample_{i:03d}_class_{label_val}_recon.ply"
        save_ply_mesh(os.path.join(ply_samples_dir, recon_ply_name), recon_verts, recon_tris)

        # Save Ground Truth 3D Mesh
        gt_ply_name = f"sample_{i:03d}_class_{label_val}_gt.ply"
        save_ply_mesh(os.path.join(ply_samples_dir, gt_ply_name), v_gt, f_gt)

        # Save Detailed Metadata JSON
        recon_meta = {
            "sample_idx": idx,
            "class_label": int(label_val),
            "num_input_points": args.num_points,
            "num_active_voxels": int(num_active),
            "num_grid_vertices": int(unique_vertices.shape[0]),
            "num_mesh_vertices": int(recon_verts.shape[0]),
            "num_mesh_triangles": int(recon_tris.shape[0]),
            "threshold": args.threshold,
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "stage1_latency_ms": round(stage1_latency_ms, 2),
            "stage2_latency_ms": round(stage2_latency_ms, 2),
            "total_latency_ms": round(total_latency, 2)
        }
        with open(os.path.join(ply_samples_dir, f"sample_{i:03d}_class_{label_val}_recon.json"), "w") as f:
            json.dump(recon_meta, f, indent=4)

    print(f"\n[✓] Two-Stage Generation Complete! Artifacts saved to: {ply_samples_dir}")


if __name__ == "__main__":
    main()
