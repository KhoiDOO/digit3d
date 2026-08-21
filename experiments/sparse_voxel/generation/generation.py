import argparse
import json
import os
import sys
import time
import torch
import torchvision
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import dense_occ2sparse_coo
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from experiments.sparse_voxel.generation.stage1.models import StructureDiT
from experiments.sparse_voxel.generation.stage2.models import SparseVertexSDFTransformer


def save_ply_mesh(filename, vertices, faces):
    """
    Saves a 3D triangle mesh to an ASCII PLY file.
    """
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if faces is not None and len(faces) > 0:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None and len(faces) > 0:
            for face in faces:
                f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_voxel_quad_mesh_ply(filename, vertices, quads):
    """
    Saves sparse voxel grid as a quad mesh (vertices and 4-vertex quad faces) to an ASCII PLY file.
    """
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(quads):
        quads = quads.detach().cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
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
    parser = argparse.ArgumentParser(description="End-to-End Two-Stage 2D Image to 3D Mesh Generation (Stage 1 MeanFlow + Stage 2 Rectified Flow)")
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Path to Stage 1 checkpoint (.pt). Defaults to stage1/stage1_structure.pt")
    parser.add_argument("--stage2_ckpt", type=str, default="", help="Path to Stage 2 checkpoint (.pt). Defaults to stage2/stage2_sdf.pt")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (or per class if --full_class is set)")
    parser.add_argument("--full_class", action="store_true", help="Generate num_samples for each digit class (0-9)")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter for specific class digit (0-9)")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold for Stage 1 active voxel extraction")
    parser.add_argument("--stage1_steps", type=int, default=64, help="Stage 1 MeanFlow sampling steps (1 = single-step fast)")
    parser.add_argument("--stage2_steps", type=int, default=64, help="Stage 2 Rectified Flow sampling steps (1 = single-step fast)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory to save ply_samples. Defaults to ply_samples/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_ckpt = args.stage1_ckpt if args.stage1_ckpt else os.path.join(base_dir, "stage1", "stage1_structure.pt")
    stage2_ckpt = args.stage2_ckpt if args.stage2_ckpt else os.path.join(base_dir, "stage2", "stage2_sdf.pt")
    ply_samples_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "ply_samples")
    os.makedirs(ply_samples_dir, exist_ok=True)

    print("==================================================================")
    print("      Digit3D End-to-End Two-Stage 3D Mesh Generation             ")
    print("==================================================================")
    print(f"Stage 1 Checkpoint : {stage1_ckpt}")
    print(f"Stage 2 Checkpoint : {stage2_ckpt}")
    print(f"Full Class Coverage: {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples        : {args.num_samples}")
    print(f"Class Filter       : {args.class_label if args.class_label >= 0 and not args.full_class else 'All Classes'}")
    print(f"Occupancy Threshold: {args.threshold}")
    print(f"Stage 1 Steps      : {args.stage1_steps} (1 = direct 1-step, >1 = slow ODE)")
    print(f"Stage 2 Steps      : {args.stage2_steps} (1 = direct 1-step, >1 = slow ODE)")
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
        img_channels=1
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
        img_channels=1
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
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)

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

    # Quad face template for 8 corner vertices of a cube (CCW outward normals)
    quad_template = torch.tensor([
        [0, 3, 2, 1], # Bottom (-Z)
        [4, 5, 6, 7], # Top (+Z)
        [0, 1, 5, 4], # Front (-Y)
        [2, 3, 7, 6], # Back (+Y)
        [0, 4, 7, 3], # Left (-X)
        [1, 2, 6, 5], # Right (+X)
    ], dtype=torch.int64, device=args.device)

    print(f"Running two-stage 3D generation on {len(selected_indices)} samples...")

    for i, idx in enumerate(tqdm(selected_indices, desc="Generating 3D Meshes")):
        raw_sample = test_dataset[idx]
        gt_vertices, gt_faces, label_val = raw_sample[0], raw_sample[1], int(raw_sample[2])
        img_tensor = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

        # Save paired 2D input image
        img_path = os.path.join(ply_samples_dir, f"sample_{i:03d}_class_{label_val}_input.png")
        torchvision.utils.save_image(img_tensor, img_path)

        img_batch = img_tensor.unsqueeze(0).to(args.device)

        start_time = time.time()

        # =====================================================================
        # STAGE 1: Generate Dense Occupancy via MeanFlow
        # =====================================================================
        t0_s1 = time.time()
        with torch.no_grad():
            pred_occ = mf_stage1.sample(batch_size=1, cond=img_batch, steps=args.stage1_steps)

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

        # Construct local indexed voxel graph and unique vertices
        dummy_sdfs = torch.zeros((num_active, 8), device=args.device, dtype=torch.float32)
        unique_vertices, local_voxels, _ = c3d.conversion.sparse2voxel(
            sparse_coords, dummy_sdfs,
            grid_min=[-1.2, -1.2, -1.2],
            grid_max=[1.2, 1.2, 1.2],
            res=[32, 32, 32]
        )

        M = unique_vertices.shape[0]

        # Build 6 quad faces per active voxel cube for Stage 1 visual inspection
        cube_quads = local_voxels[:, quad_template].reshape(-1, 4)
        stage1_latency_ms = (time.time() - t0_s1) * 1000.0

        # Save Stage 1 Voxel Quad Mesh
        stage1_ply_name = f"sample_{i:03d}_class_{label_val}_stage1_voxels.ply"
        stage1_ply_path = os.path.join(ply_samples_dir, stage1_ply_name)
        save_voxel_quad_mesh_ply(stage1_ply_path, unique_vertices, cube_quads)

        # Save Stage 1 Metadata
        stage1_meta = {
            "sample_idx": idx,
            "class_label": int(label_val),
            "num_active_voxels": int(num_active),
            "num_grid_vertices": int(M),
            "num_quad_faces": int(cube_quads.shape[0]),
            "threshold": args.threshold,
            "stage1_steps": args.stage1_steps,
            "stage1_latency_ms": round(stage1_latency_ms, 2)
        }
        stage1_json_name = f"sample_{i:03d}_class_{label_val}_stage1.json"
        stage1_json_path = os.path.join(ply_samples_dir, stage1_json_name)
        with open(stage1_json_path, "w") as f:
            json.dump(stage1_meta, f, indent=4)

        # =====================================================================
        # STAGE 2: Generate 1D Scalar SDFs at Active Grid Vertices via Rectified Flow
        # =====================================================================
        t0_s2 = time.time()
        with torch.no_grad():
            c_img = model_stage2.img_encoder(img_batch)
            cond_tensor = torch.cat([unique_vertices.unsqueeze(0), c_img.unsqueeze(1).expand(1, M, -1)], dim=-1)
            noise = torch.randn(1, M, 1, device=args.device)
            model_stage2.set_seq_lens([M])
            pred_sdfs_batch = rf_stage2.sample(
                data_shape=(M, 1),
                noise=noise,
                cond=cond_tensor,
                steps=args.stage2_steps
            )
            pred_sdfs = pred_sdfs_batch[0, :, 0] # [M]

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
        recon_ply_path = os.path.join(ply_samples_dir, recon_ply_name)
        save_ply_mesh(recon_ply_path, recon_verts, recon_tris)

        # Save Ground Truth 3D Mesh
        gt_ply_name = f"sample_{i:03d}_class_{label_val}_gt.ply"
        gt_ply_path = os.path.join(ply_samples_dir, gt_ply_name)
        save_ply_mesh(gt_ply_path, gt_vertices, gt_faces)

        # Save Detailed Stage 2 Topology Metadata
        recon_meta = {
            "sample_idx": idx,
            "class_label": int(label_val),
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
        recon_json_name = f"sample_{i:03d}_class_{label_val}_recon.json"
        recon_json_path = os.path.join(ply_samples_dir, recon_json_name)
        with open(recon_json_path, "w") as f:
            json.dump(recon_meta, f, indent=4)

        gt_meta = {
            "sample_idx": idx,
            "class_label": int(label_val),
            "num_mesh_vertices": int(gt_vertices.shape[0]),
            "num_mesh_triangles": int(gt_faces.shape[0])
        }
        gt_json_name = f"sample_{i:03d}_class_{label_val}_gt.json"
        gt_json_path = os.path.join(ply_samples_dir, gt_json_name)
        with open(gt_json_path, "w") as f:
            json.dump(gt_meta, f, indent=4)

    print(f"\nTwo-Stage Generation Complete! Saved Stage 1 voxels, Stage 2 meshes, GT, and metadata to {ply_samples_dir}")


if __name__ == "__main__":
    main()
