import argparse
import json
import os
import sys
import torch
import torchvision
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import RectifiedFlow
from experiments.sparse_voxel.generation.stage2.models import SparseVertexSDFTransformer


def save_mesh_ply(vertices: torch.Tensor, faces: torch.Tensor, save_path: str):
    """
    Saves triangular mesh vertices and faces to standard ASCII PLY.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()

    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(v_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(f_np)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in v_np:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in f_np:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Standalone Stage 2 Sparse Vertex SDF Rectified Flow 3D Mesh Generation")
    parser.add_argument("--ckpt", type=str, default="", help="Path to Stage 2 checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--full_class", action="store_true", help="Generate samples for all classes 0-9")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter specific class (0-9)")
    parser.add_argument("--steps", type=int, default=64, help="Sampling steps for Rectified Flow ODE")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for PLY meshes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = args.ckpt if args.ckpt else os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_sdf.pt")
    out_dir = args.out_dir if args.out_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print(" Digit3D Stage 2 Sparse Vertex SDF Rectified Flow Mesh Gen        ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Full Class   : {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples  : {args.num_samples}")
    print(f"Sampling Step: {args.steps}")
    print(f"Output Dir   : {out_dir}")
    print("------------------------------------------------------------------")

    # 1. Load Model & Rectified Flow
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
        print(f"[*] Loaded Stage 2 weights from {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.eval()
    rf = RectifiedFlow(model=model).to(args.device)

    # 2. Dataset
    print("Loading test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)

    sample_indices = []
    if args.full_class:
        class_buckets = {c: [] for c in range(10)}
        for idx in range(len(test_dataset)):
            _, _, label, _ = test_dataset[idx]
            label_val = int(label)
            if len(class_buckets[label_val]) < args.num_samples:
                class_buckets[label_val].append(idx)
            if all(len(b) >= args.num_samples for b in class_buckets.values()):
                break
        for c in sorted(class_buckets.keys()):
            sample_indices.extend(class_buckets[c])
    elif args.class_label >= 0:
        for idx in range(len(test_dataset)):
            _, _, label, _ = test_dataset[idx]
            if int(label) == args.class_label:
                sample_indices.append(idx)
            if len(sample_indices) >= args.num_samples:
                break
    else:
        sample_indices = list(range(min(args.num_samples, len(test_dataset))))

    print(f"Generating Stage 2 meshes for {len(sample_indices)} samples...")

    with torch.no_grad():
        for i, idx in enumerate(tqdm(sample_indices, desc="Generating Stage 2 Meshes")):
            raw_sample = test_dataset[idx]
            v, f = raw_sample[0], raw_sample[1]
            label_val = int(raw_sample[2])
            img = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

            # 1. Save input image
            img_path = os.path.join(out_dir, f"sample_{i:03d}_class_{label_val}_input.png")
            torchvision.utils.save_image(img, img_path)

            bmesh = BTriangleMesh(
                vertices=v.to(args.device),
                faces=f.to(args.device),
                vertbids=torch.zeros(v.shape[0], dtype=torch.int32, device=args.device),
                facebids=torch.zeros(f.shape[0], dtype=torch.int32, device=args.device),
                batch_size=1
            )
            img_t = img.unsqueeze(0).to(args.device) # [1, 1, 28, 28]

            sparse_coords, sparse_sdfs = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            num_active = int(sparse_coords.shape[0])
            dummy_sdfs = torch.zeros((num_active, 8), device=args.device, dtype=torch.float32)
            u_verts, local_voxels, _ = c3d.conversion.sparse2voxel(
                sparse_coords, dummy_sdfs,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )

            M = u_verts.shape[0]
            c_img = model.img_encoder(img_t) # [1, 256]
            cond_tensor = torch.cat([u_verts.unsqueeze(0), c_img.unsqueeze(1).expand(-1, M, -1)], dim=-1)

            noise = torch.randn(1, M, 1, device=args.device)
            model.set_seq_lens([M])

            # Rectified Flow Sampling
            s_pred_1 = rf.sample(
                data_shape=(M, 1),
                noise=noise,
                cond=cond_tensor,
                steps=args.steps
            )

            pred_sdfs = s_pred_1.squeeze(0).squeeze(-1) # [M]

            # Differentiable Marching Cubes surface extraction
            recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(
                u_verts, local_voxels, pred_sdfs, iso=0.0
            )

            # 2. Save Mesh PLY
            ply_path = os.path.join(out_dir, f"sample_{i:03d}_class_{label_val}_mesh.ply")
            save_mesh_ply(recon_verts, recon_tris, ply_path)

            # 3. Save Metadata JSON
            meta = {
                "sample_idx": idx,
                "class_label": label_val,
                "num_active_voxels": num_active,
                "num_grid_vertices": int(M),
                "num_reconstructed_vertices": int(recon_verts.shape[0]),
                "num_reconstructed_faces": int(recon_tris.shape[0]),
                "steps": args.steps
            }
            meta_path = os.path.join(out_dir, f"sample_{i:03d}_class_{label_val}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

    print(f"\nStage 2 Generation Complete! Meshes, input images, and metadata saved to {out_dir}\n")


if __name__ == "__main__":
    main()
