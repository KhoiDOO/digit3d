import argparse
import json
import os
import sys
import numpy as np
import torch
import torchvision
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Append digit3d root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.data.dataset.digit3d import Digit3D
from experiments.sparse_voxel.classification.models import SparseClassifier
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
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sparse Voxel ResNet Classifier on Digit3D Test Set")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). Defaults to sparse_classification.pt in current folder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--out_file", type=str, default="eval_results.json", help="Output JSON metrics filename")
    parser.add_argument("--wrong_dir", type=str, default="wrong", help="Directory to save misclassified cases")
    parser.add_argument("--no_save_wrong", action="store_true", help="Disable saving misclassified cases")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if args.ckpt else os.path.join(curr_dir, "sparse_classification.pt")

    if not os.path.exists(ckpt_path):
        parent_ckpt = os.path.abspath(os.path.join(curr_dir, "../sparse_classification.pt"))
        if os.path.exists(parent_ckpt):
            ckpt_path = parent_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {parent_ckpt}")

    print("==================================================================")
    print("        Digit3D Sparse Voxel Classification Benchmark Evaluation  ")
    print("==================================================================")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Device       : {args.device}")
    print("------------------------------------------------------------------")

    # 1. Load Model
    model = SparseClassifier(in_channels=8, num_classes=10).to(args.device)
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Test Dataset
    print("Loading Digit3D test dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bmesh_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. Evaluation Loop
    all_preds = []
    all_targets = []
    wrong_cases = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        global_idx = 0
        for bmesh, batched_labels in tqdm(test_loader, desc="Evaluating"):
            batch_size = batched_labels.size(0)
            bmesh = bmesh.to(args.device)
            bmesh.vertices = bmesh.vertices.float()
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)

            batched_coords = batched_coords.to(args.device)
            batched_sdf = batched_sdf.to(args.device)
            batched_labels = batched_labels.to(args.device)

            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
            with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu', dtype=torch.float16 if 'cuda' in args.device else torch.float32):
                logits = model(x)

            preds = logits.argmax(dim=-1).cpu().numpy()
            targets = batched_labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

            if not args.no_save_wrong:
                for b_i in range(batch_size):
                    sample_idx = global_idx + b_i
                    pred_cls = int(preds[b_i])
                    true_cls = int(targets[b_i])
                    if pred_cls != true_cls:
                        raw_sample = test_dataset[sample_idx]
                        v_slice = raw_sample[0]
                        f_slice = raw_sample[1]
                        img_slice = raw_sample[3] if len(raw_sample) > 3 else None

                        wrong_cases.append({
                            "sample_idx": sample_idx,
                            "pred": pred_cls,
                            "target": true_cls,
                            "vertices": v_slice,
                            "faces": f_slice,
                            "image": img_slice
                        })

            global_idx += batch_size

    # 4. Compute Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    report = classification_report(all_targets, all_preds, digits=4, output_dict=True)

    results = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_test_samples": len(all_targets),
        "total_misclassified": len(wrong_cases),
        "per_class_report": report
    }

    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS SUMMARY                   ")
    print("=" * 65)
    print(f"Total Test Samples   : {len(all_targets)}")
    print(f"Overall Accuracy     : {acc * 100:.2f}%")
    print(f"Macro Precision      : {macro_precision * 100:.2f}%")
    print(f"Macro Recall         : {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score       : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score    : {weighted_f1 * 100:.2f}%")
    print(f"Misclassified Cases  : {len(wrong_cases)}")
    print("-" * 65)

    out_json_path = os.path.join(curr_dir, args.out_file)
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Benchmark evaluation metrics saved to: {out_json_path}")

    # 5. Export Misclassified Samples
    if not args.no_save_wrong and wrong_cases:
        wrong_dir_path = os.path.join(curr_dir, args.wrong_dir)
        os.makedirs(wrong_dir_path, exist_ok=True)
        print(f"\n[*] Exporting {len(wrong_cases)} misclassified 3D meshes to: {wrong_dir_path}/")

        for item in tqdm(wrong_cases, desc="Saving Wrong Cases"):
            s_idx = item["sample_idx"]
            pred = item["pred"]
            tgt = item["target"]
            base_name = f"wrong_{s_idx:05d}_pred_{pred}_true_{tgt}"
            ply_path = os.path.join(wrong_dir_path, f"{base_name}.ply")
            save_ply_mesh(ply_path, item["vertices"], item["faces"])

            if item["image"] is not None and torch.is_tensor(item["image"]):
                png_path = os.path.join(wrong_dir_path, f"{base_name}.png")
                torchvision.utils.save_image(item["image"].float(), png_path)

        print("[*] All misclassified cases successfully exported!")


if __name__ == "__main__":
    main()
