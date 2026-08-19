import argparse
import json
import math
import os
import sys
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.data.dataset.digit3d import Digit3D
from experiments.sparse_voxel.classification.models import SparseClassifier
from torchsparse import SparseTensor


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Voxel ResNet Classifier on Digit3D")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="", help="Custom directory to save weights and logs")
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    print("==================================================================")
    print("        Digit3D Sparse Voxel Classification Training              ")
    print("==================================================================")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Save Dir     : {save_dir}")
    print("------------------------------------------------------------------")

    # 1. Initialize Datasets & DataLoaders
    print("Initializing Datasets...")
    train_dataset = Digit3D(root="~/.conquer3d/", train=True, download=True, cached=True)
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bmesh_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bmesh_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True
    )

    # 2. Model, Loss, Optimizer & Scheduler
    model = SparseClassifier(in_channels=8, num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # 3. Training Loop
    history = []
    best_test_acc = 0.0

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for bmesh, batched_labels in progress_bar:
            bmesh = bmesh.cuda(non_blocking=True)
            bmesh.vertices = bmesh.vertices.float()
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)

            batched_coords = batched_coords.cuda(non_blocking=True)
            batched_sdf = batched_sdf.cuda(non_blocking=True)
            batched_labels = batched_labels.cuda(non_blocking=True)

            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, batched_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batched_labels.size(0)
            correct += predicted.eq(batched_labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100. * correct / total:.2f}%"
            })

        # Evaluate on Test Set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for bmesh, batched_labels in test_loader:
                bmesh = bmesh.cuda(non_blocking=True)
                bmesh.vertices = bmesh.vertices.float()
                batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
                batched_coords = batched_coords.cuda(non_blocking=True)
                batched_sdf = batched_sdf.cuda(non_blocking=True)
                batched_labels = batched_labels.cuda(non_blocking=True)

                x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)

                _, predicted = logits.max(1)
                test_total += batched_labels.size(0)
                test_correct += predicted.eq(batched_labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total

        print(f"==> Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = os.path.join(save_dir, "sparse_classification.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Saved new best model to {ckpt_path} with test acc: {best_test_acc:.2f}%")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })

        json_path = os.path.join(save_dir, "sparse_classification.json")
        with open(json_path, "w") as f:
            json.dump(history, f, indent=4)

        scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    main()
