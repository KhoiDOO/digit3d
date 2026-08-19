import argparse
import json
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
from experiments.sparse_voxel.reconstruction.models import SimpleSparseVAE
from torchsparse import SparseTensor


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Voxel VAE for 3D Reconstruction on Digit3D")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="KL divergence loss weight")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="", help="Custom directory to save weights and logs")
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    print("==================================================================")
    print("        Digit3D Sparse Voxel VAE Reconstruction Training          ")
    print("==================================================================")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"KL Weight    : {args.kl_weight}")
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

    # 2. Initialize Model & Optimizer
    model = SimpleSparseVAE(
        in_channels=8,
        hidden_channels=32,
        latent_channels=16,
        out_channels=8,
        num_layers=3
    ).cuda()

    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # 3. Training Loop
    history = []
    best_test_loss = float("inf")

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for bmesh, _ in progress_bar:
            bmesh = bmesh.cuda(non_blocking=True)
            bmesh.vertices = bmesh.vertices.float()
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)

            batched_coords = batched_coords.cuda(non_blocking=True)
            batched_sdf = batched_sdf.cuda(non_blocking=True)

            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_sdf, posterior = model(x)
                recon_loss = mse_criterion(pred_sdf, batched_sdf)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + args.kl_weight * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MSE': f"{recon_loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}"
            })

        # Evaluate on Test Set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for bmesh, _ in test_loader:
                bmesh = bmesh.cuda(non_blocking=True)
                bmesh.vertices = bmesh.vertices.float()
                batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
                batched_coords = batched_coords.cuda(non_blocking=True)
                batched_sdf = batched_sdf.cuda(non_blocking=True)

                x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_sdf, posterior = model(x)
                    recon_loss = mse_criterion(pred_sdf, batched_sdf)
                    kl_loss = posterior.kl().mean()
                    test_loss = recon_loss + args.kl_weight * kl_loss

                total_test_loss += test_loss.item()

        train_loss_avg = total_train_loss / len(train_loader)
        test_loss_avg = total_test_loss / len(test_loader)

        print(f"==> Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss_avg:.4f}, Test Loss: {test_loss_avg:.4f}")

        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            ckpt_path = os.path.join(save_dir, "sparse_reconstruction.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Saved best model to {ckpt_path} (Test Loss: {best_test_loss:.4f})")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "test_loss": test_loss_avg,
            "recon_mse": total_recon_loss / len(train_loader),
            "kl_loss": total_kl_loss / len(train_loader)
        })

        json_path = os.path.join(save_dir, "sparse_reconstruction.json")
        with open(json_path, "w") as f:
            json.dump(history, f, indent=4)

    print("Training finished.")


if __name__ == "__main__":
    main()
