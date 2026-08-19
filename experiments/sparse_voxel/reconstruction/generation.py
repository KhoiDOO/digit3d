import argparse
import json
import os
import sys
import torch
import torchvision
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.data.dataset.digit3d import Digit3D
from experiments.sparse_voxel.reconstruction.models import SimpleSparseVAE
from torchsparse import SparseTensor


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


def main():
    parser = argparse.ArgumentParser(description="Reconstruct 3D Meshes from Sparse Voxel VAE on Digit3D")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). Defaults to sparse_reconstruction.pt in current folder.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to reconstruct (or per class if --full_class is set)")
    parser.add_argument("--full_class", action="store_true", help="Generate num_samples for each digit class (0-9)")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter for specific class digit (0-9)")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory to save ply_samples. Defaults to experiments/sparse_voxel/reconstruction/ply_samples/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "sparse_reconstruction.pt")

    if not os.path.exists(ckpt_path):
        parent_ckpt = os.path.abspath(os.path.join(curr_dir, "../sparse_reconstruction.pt"))
        if os.path.exists(parent_ckpt):
            ckpt_path = parent_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {parent_ckpt}")

    ply_samples_dir = args.out_dir if args.out_dir else os.path.join(curr_dir, "ply_samples")
    os.makedirs(ply_samples_dir, exist_ok=True)

    print("==================================================================")
    print("        Digit3D Sparse Voxel VAE Mesh Reconstruction              ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Full Class   : {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples  : {args.num_samples}")
    print(f"Class Filter : {args.class_label if args.class_label >= 0 and not args.full_class else 'All Classes'}")
    print(f"Sample Offset: {args.sample_offset}")
    print(f"Output Dir   : {ply_samples_dir}")
    print("------------------------------------------------------------------")

    # 1. Load Model
    model = SimpleSparseVAE(
        in_channels=8,
        hidden_channels=32,
        latent_channels=16,
        out_channels=8,
        num_layers=3
    ).to(args.device)

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Dataset
    print("Loading test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)

    # 3. Collect Samples to Reconstruct
    selected_indices = []
    if args.full_class:
        for c in range(10):
            collected_for_c = 0
            curr_idx = args.sample_offset
            while collected_for_c < args.num_samples and curr_idx < len(test_dataset):
                sample = test_dataset[curr_idx]
                label = sample[2] if len(sample) > 2 else 0
                if label == c:
                    selected_indices.append(curr_idx)
                    collected_for_c += 1
                curr_idx += 1
    else:
        curr_idx = args.sample_offset
        while len(selected_indices) < args.num_samples and curr_idx < len(test_dataset):
            sample = test_dataset[curr_idx]
            label = sample[2] if len(sample) > 2 else 0
            if args.class_label < 0 or label == args.class_label:
                selected_indices.append(curr_idx)
            curr_idx += 1

    print(f"Selected {len(selected_indices)} test samples: {selected_indices}")

    # 4. Reconstruction Loop
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Reconstructing Meshes")):
            raw_sample = test_dataset[idx]
            gt_vertices = raw_sample[0]
            gt_faces = raw_sample[1]
            label_val = raw_sample[2] if len(raw_sample) > 2 else 0

            batch = bmesh_collate_fn([raw_sample])
            bmesh, _ = batch

            bmesh = bmesh.to(args.device)
            bmesh.vertices = bmesh.vertices.float()

            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            batched_coords = batched_coords.to(args.device)
            batched_sdf = batched_sdf.to(args.device)

            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())

            # VAE Forward Pass
            pred_sdf, posterior = model(x)

            # Isosurface Extraction via Differentiable Marching Cubes
            unique_vertices, local_voxels, merged_sdfs = c3d.conversion.sparse2voxel(
                batched_coords, pred_sdf,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )
            recon_verts, recon_tris, _, _ = c3d.ops.diff_marching_cubes(unique_vertices, local_voxels, merged_sdfs, iso=0.0)

            # 1. Save Reconstructed 3D Mesh & Metadata
            recon_ply_name = f"sample_{i:03d}_class_{label_val}_recon.ply"
            recon_ply_path = os.path.join(ply_samples_dir, recon_ply_name)
            save_ply_mesh(recon_ply_path, recon_verts, recon_tris)

            recon_meta = {
                "sample_idx": idx,
                "class_label": int(label_val),
                "num_active_voxels": int(batched_coords.shape[0]),
                "num_grid_vertices": int(unique_vertices.shape[0]),
                "num_mesh_vertices": int(recon_verts.shape[0]),
                "num_mesh_triangles": int(recon_tris.shape[0])
            }
            recon_json_name = f"sample_{i:03d}_class_{label_val}_recon.json"
            recon_json_path = os.path.join(ply_samples_dir, recon_json_name)
            with open(recon_json_path, "w") as f:
                json.dump(recon_meta, f, indent=4)

            # 2. Save Ground Truth 3D Mesh & Metadata
            gt_unique_vertices, _, _ = c3d.conversion.sparse2voxel(
                batched_coords, batched_sdf,
                grid_min=[-1.2, -1.2, -1.2],
                grid_max=[1.2, 1.2, 1.2],
                res=[32, 32, 32]
            )

            gt_ply_name = f"sample_{i:03d}_class_{label_val}_gt.ply"
            gt_ply_path = os.path.join(ply_samples_dir, gt_ply_name)
            save_ply_mesh(gt_ply_path, gt_vertices, gt_faces)

            gt_meta = {
                "sample_idx": idx,
                "class_label": int(label_val),
                "num_active_voxels": int(batched_coords.shape[0]),
                "num_grid_vertices": int(gt_unique_vertices.shape[0]),
                "num_mesh_vertices": int(gt_vertices.shape[0]),
                "num_mesh_triangles": int(gt_faces.shape[0])
            }
            gt_json_name = f"sample_{i:03d}_class_{label_val}_gt.json"
            gt_json_path = os.path.join(ply_samples_dir, gt_json_name)
            with open(gt_json_path, "w") as f:
                json.dump(gt_meta, f, indent=4)

            # 3. Save Paired 2D Input Image (if available)
            if len(raw_sample) > 3 and raw_sample[3] is not None:
                img_tensor = raw_sample[3]
                if torch.is_tensor(img_tensor):
                    png_name = f"sample_{i:03d}_class_{label_val}_input.png"
                    png_path = os.path.join(ply_samples_dir, png_name)
                    torchvision.utils.save_image(img_tensor.float(), png_path)

    print("\n==================================================================")
    print("           RECONSTRUCTION EXPORT COMPLETED SUCCESSFULLY           ")
    print("==================================================================")
    print(f"Exported {len(selected_indices)} pairs (Reconstructed + Ground Truth) to: {ply_samples_dir}/")


if __name__ == "__main__":
    main()
