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
from conquer3d.conversion.grid import dense_occ2sparse_coo
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.generation.stage1.models import StructureDiT


def save_voxel_quad_mesh_ply(filename: str, vertices: torch.Tensor, quads: torch.Tensor):
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
    parser = argparse.ArgumentParser(description="Generate 3D Active Voxel Occupancy from 2D Images using Stage 1 Structure DiT MeanFlow")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). Defaults to stage1_structure.pt")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (or per class if --full_class is set)")
    parser.add_argument("--full_class", action="store_true", help="Generate num_samples for each digit class (0-9)")
    parser.add_argument("--class_label", type=int, default=-1, help="Filter for specific class digit (0-9)")
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
    print("      Digit3D Stage 1 Structure DiT MeanFlow Voxel Quad Gen       ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Full Class   : {args.full_class} ({args.num_samples} per class if True)")
    print(f"Num Samples  : {args.num_samples}")
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
        img_channels=1
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
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)

    # 3. Filter Samples
    selected_indices = []
    if args.full_class:
        class_samples = {c: [] for c in range(10)}
        for idx in range(len(test_dataset)):
            raw_sample = test_dataset[idx]
            label_val = int(raw_sample[2])
            if len(class_samples[label_val]) < args.num_samples:
                class_samples[label_val].append(idx)
            if all(len(s) >= args.num_samples for s in class_samples.values()):
                break
        for c in sorted(class_samples.keys()):
            selected_indices.extend(class_samples[c])
    elif args.class_label >= 0:
        for idx in range(len(test_dataset)):
            raw_sample = test_dataset[idx]
            if int(raw_sample[2]) == args.class_label:
                selected_indices.append(idx)
                if len(selected_indices) >= args.num_samples:
                    break
    else:
        selected_indices = list(range(min(args.num_samples, len(test_dataset))))

    # Quad face indices template for 8 corner vertices of a cube (CCW outward normals)
    quad_template = torch.tensor([
        [0, 3, 2, 1], # Bottom (-Z)
        [4, 5, 6, 7], # Top (+Z)
        [0, 1, 5, 4], # Front (-Y)
        [2, 3, 7, 6], # Back (+Y)
        [0, 4, 7, 3], # Left (-X)
        [1, 2, 6, 5], # Right (+X)
    ], dtype=torch.int64, device=args.device)

    print(f"Generating voxel quad meshes for {len(selected_indices)} samples...")

    for i, idx in enumerate(tqdm(selected_indices, desc="Generating Stage 1 Voxels")):
        raw_sample = test_dataset[idx]
        label_val = int(raw_sample[2])
        img_tensor = raw_sample[3] if len(raw_sample) > 3 and raw_sample[3] is not None else torch.zeros((1, 28, 28))

        # Save input image
        img_path = os.path.join(out_dir, f"sample_{i:03d}_class_{label_val}_input.png")
        torchvision.utils.save_image(img_tensor, img_path)

        # Generate dense occupancy grid with MeanFlow sample
        img_batch = img_tensor.unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred_occ = mf.sample(batch_size=1, cond=img_batch, steps=args.steps)

        # Extract active voxel coordinates
        sparse_coords = dense_occ2sparse_coo(pred_occ, threshold=args.threshold)
        num_active = int(sparse_coords.shape[0])

        if num_active > 3000:
            flat_occ = pred_occ[0, 0, sparse_coords[:, 1].long(), sparse_coords[:, 2].long(), sparse_coords[:, 3].long()]
            topk_indices = torch.topk(flat_occ, k=3000).indices
            sparse_coords = sparse_coords[topk_indices]
            num_active = 3000
        elif num_active == 0:
            sparse_coords = torch.tensor([[0, 15, 15, 15]], dtype=torch.int32, device=args.device)
            num_active = 1

        # Convert sparse active voxel coordinates to unique grid vertices and local voxel corner graph
        dummy_sdfs = torch.zeros((num_active, 8), device=args.device, dtype=torch.float32)
        unique_vertices, local_voxels, _ = c3d.conversion.sparse2voxel(
            sparse_coords, dummy_sdfs,
            grid_min=[-1.2, -1.2, -1.2],
            grid_max=[1.2, 1.2, 1.2],
            res=[32, 32, 32]
        )

        # Build 6 quad faces per voxel cube mapping to unique vertices
        cube_quads = local_voxels[:, quad_template].reshape(-1, 4)

        # Save as quad mesh PLY file
        ply_name = f"sample_{i:03d}_class_{label_val}_voxels.ply"
        ply_path = os.path.join(out_dir, ply_name)
        save_voxel_quad_mesh_ply(ply_path, unique_vertices, cube_quads)

        # Save metadata
        meta = {
            "sample_idx": idx,
            "class_label": label_val,
            "num_active_voxels": int(num_active),
            "num_grid_vertices": int(unique_vertices.shape[0]),
            "num_quad_faces": int(cube_quads.shape[0]),
            "threshold": args.threshold,
            "steps": args.steps
        }
        meta_path = os.path.join(out_dir, f"sample_{i:03d}_class_{label_val}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    print(f"\nStage 1 Voxel Quad Generation Complete! Saved to {out_dir}")


if __name__ == "__main__":
    main()
