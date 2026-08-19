import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Append digit3d root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.classification.models import PointTransformerCls


def save_ply(filename, sample):
    """
    Saves a point cloud sample [N, 6] with normals to an ASCII PLY file.
    Matches the exact formatting from experiments/pc/generation/generation.py.
    """
    with open(filename, "w") as f:
        # Write PLY Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(sample)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        # Write Points and Normals
        for pt in sample:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {pt[3]:.6f} {pt[4]:.6f} {pt[5]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointTransformer Classifier on Digit3D Point Cloud Test Set")
    parser.add_argument("--ckpt", type=str, default="", help="Path to model checkpoint (.pt). Defaults to point_classification.pt in the same folder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_points", type=int, default=512, help="Number of points per sample")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--out_file", type=str, default="eval_results.json", help="Output JSON filename")
    parser.add_argument("--wrong_dir", type=str, default="wrong", help="Directory name to save misclassified cases")
    parser.add_argument("--no_save_wrong", action="store_true", help="Disable saving misclassified cases")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Resolve Checkpoint Path
    if not args.ckpt:
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "point_classification.pt")
    else:
        ckpt_path = args.ckpt

    wrong_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.wrong_dir)
    save_wrong = not args.no_save_wrong
    if save_wrong:
        os.makedirs(wrong_dir, exist_ok=True)

    print("==================================================================")
    print("        Point Cloud Classification Evaluation Pipeline           ")
    print("==================================================================")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Num Points  : {args.num_points}")
    print(f"Batch Size  : {args.batch_size}")
    if save_wrong:
        print(f"Wrong Dir   : {wrong_dir}")
    print("------------------------------------------------------------------")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Load Test Dataset & DataLoader
    print("Loading PointDigit3D Test Dataset...")
    test_dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=True, cached=True, num_points=args.num_points, return_img=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    print(f"Total Test Samples: {len(test_dataset)}")

    # 3. Initialize Model
    print("Initializing PointTransformerCls...")
    model = PointTransformerCls(
        depth=4,
        in_channels=6,
        num_classes=10,
        dim=128,
        share_planes=8,
        patch_size=32
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights successfully.")

    # 4. Inference Loop
    all_preds = []
    all_targets = []
    num_wrong_saved = 0
    sample_offset = 0

    print("\nRunning Inference...")
    with torch.inference_mode():
        for points, features, labels, imgs in tqdm(test_loader, desc="Evaluating"):
            features_dev = features.to(device, non_blocking=True)
            
            if device.type == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(features_dev)
            else:
                logits = model(features_dev)

            preds = torch.argmax(logits, dim=1).cpu()
            labels_cpu = labels.cpu()

            # Save misclassified samples
            if save_wrong:
                wrong_mask = (preds != labels_cpu)
                if wrong_mask.any():
                    wrong_indices = torch.where(wrong_mask)[0]
                    for idx_in_batch in wrong_indices:
                        global_idx = sample_offset + idx_in_batch.item()
                        gt_label = labels_cpu[idx_in_batch].item()
                        pred_label = preds[idx_in_batch].item()
                        
                        prefix = f"sample_{global_idx:05d}_gt_{gt_label}_pred_{pred_label}"
                        
                        # 1. Save 2D Image (.png)
                        img_path = os.path.join(wrong_dir, f"{prefix}.png")
                        torchvision.utils.save_image(imgs[idx_in_batch], img_path)
                        
                        # 2. Save 3D Point Cloud with Normals (.ply)
                        ply_path = os.path.join(wrong_dir, f"{prefix}.ply")
                        pts_feat_np = features[idx_in_batch].numpy()
                        save_ply(ply_path, pts_feat_np)
                        
                        num_wrong_saved += 1

            all_preds.extend(preds.numpy().tolist())
            all_targets.extend(labels_cpu.numpy().tolist())
            sample_offset += len(labels_cpu)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Compute Metrics
    accuracy = float(accuracy_score(all_targets, all_preds))
    precision_macro = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall_macro = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    precision_weighted = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    # Per-Class Breakdown
    cls_report = classification_report(all_targets, all_preds, digits=4, output_dict=True, zero_division=0)
    per_class_metrics = {}
    for c in range(10):
        c_str = str(c)
        if c_str in cls_report:
            per_class_metrics[c_str] = {
                "precision": float(cls_report[c_str]["precision"]),
                "recall": float(cls_report[c_str]["recall"]),
                "f1-score": float(cls_report[c_str]["f1-score"]),
                "support": int(cls_report[c_str]["support"])
            }

    results = {
        "checkpoint": os.path.basename(ckpt_path),
        "num_test_samples": len(all_targets),
        "num_wrong_samples": int((all_preds != all_targets).sum()),
        "wrong_dir": os.path.relpath(wrong_dir, os.path.dirname(os.path.abspath(__file__))) if save_wrong else None,
        "num_points": args.num_points,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class_metrics
    }

    # 6. Save Results to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    # 7. Display Summary
    print("\n" + "=" * 66)
    print("                       EVALUATION RESULTS                         ")
    print("=" * 66)
    print(classification_report(all_targets, all_preds, digits=4, zero_division=0))
    print("-" * 66)
    print(f"Overall Accuracy   : {accuracy * 100:.2f}%")
    print(f"Macro Precision    : {precision_macro * 100:.2f}%")
    print(f"Macro Recall       : {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score     : {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score  : {f1_weighted * 100:.2f}%")
    print(f"Wrong Cases Saved  : {num_wrong_saved} samples in {wrong_dir}")
    print(f"Saved Results to   : {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
