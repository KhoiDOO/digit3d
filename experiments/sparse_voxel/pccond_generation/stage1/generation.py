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
from conquer3d.conversion.grid import dense_occ2sparse_coo
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.pccond_generation.stage1.models import StructureDiT


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
    parser = argparse.ArgumentParser(description="Generate 3D Active Voxel Occupancy from Point Clouds using Stage 1 Structure DiT MeanFlow")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). Defaults to stage1_structure.pt")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (or per class if --full_class is set)")
    parser.add_argument("--full_class", action="store_true", help="Generate num_samples for each digit class (0-9)")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter for specific class digit (0-9)")
    parser.add_argument("--num_points", type=int, default=512, help="Number of input points")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold for active voxel extraction")
    parser.add_argument("--steps", type=int, default=64, help="Sampling steps (1 = fast 1-step, >1 = slow ODE integration)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory to save samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "stage1_structure.pt")
    out_dir = args.out_dir if args.out_dir else os.path.join(curr_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print("  Digit3D Stage 1 Point-Conditioned Structure DiT Voxel Gen       ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Full Class   : {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples  : {args.num_samples}")
    print(f"Num Points   : {args.num_points} (XYZ + Normals)")
    print(f"Threshold    : {args.threshold}")
    print(f"Steps        : {args.steps} (1 = direct 1-step, >1 = ODE integration)")
    print(f"Output Dir   : {out_dir}")
    print("------------------------------------------------------------------")

    # 1. Load Model & MeanFlow
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

    # 2. Load Dataset
    print("Loading test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)

    # 3. Filter Samples
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
        selected_indices = list(range(args.num_samples))

    # Cube quad face template
    quad_template = torch.tensor([
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
    ], dtype=torch.int64, device=args.device)

    print(f"Generating voxel quad meshes for {len(selected_indices)} samples...")

    for i, idx in enumerate(tqdm(selected_indices, desc="Sampling Voxel Meshes")):
        v, f, label = test_dataset[idx]
        mesh_np = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
        pts_np, face_idx = trimesh.sample.sample_surface(mesh_np, args.num_points)
        normals_np = mesh_np.face_normals[face_idx]
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        normals_t = torch.tensor(normals_np, dtype=torch.float32)
        pc_feat = torch.cat([pts_t, normals_t], dim=-1).permute(1, 0).unsqueeze(0).to(args.device) # [1, 6, N]

        # Save input point cloud
        pc_ply_name = f"sample_{i:03d}_class_{label}_input_pc.ply"
        save_point_cloud_ply(os.path.join(out_dir, pc_ply_name), pc_feat[0])

        t0 = time.time()
        with torch.no_grad():
            pred_occ = mf.sample(batch_size=1, cond=pc_feat, steps=args.steps)

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

        cube_quads = local_voxels[:, quad_template].reshape(-1, 4)
        latency_ms = (time.time() - t0) * 1000.0

        # Save Voxel Quad Mesh
        ply_name = f"sample_{i:03d}_class_{label}_voxels.ply"
        save_voxel_quad_mesh_ply(os.path.join(out_dir, ply_name), unique_vertices, cube_quads)

        # Save Metadata
        meta = {
            "sample_idx": idx,
            "class_label": int(label),
            "num_input_points": args.num_points,
            "num_active_voxels": int(num_active),
            "num_grid_vertices": int(unique_vertices.shape[0]),
            "num_quad_faces": int(cube_quads.shape[0]),
            "threshold": args.threshold,
            "steps": args.steps,
            "latency_ms": round(latency_ms, 2)
        }
        with open(os.path.join(out_dir, f"sample_{i:03d}_class_{label}.json"), "w") as f:
            json.dump(meta, f, indent=4)

    print(f"\n[✓] Stage 1 Sampling complete! Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
