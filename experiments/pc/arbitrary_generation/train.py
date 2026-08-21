import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import trimesh
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Append repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow

from experiments.pc.arbitrary_generation.models import ArbitraryPointFlowTransformer


class TrimeshRandomPointCollate:
    """
    Collate function that dynamically samples a synchronized random point count P ~ Uniform(min_points, max_points)
    per batch and extracts 6D surface points [P, 6] (positions + exact face normals) using trimesh across DataLoader workers.
    """
    def __init__(self, min_points: int = 256, max_points: int = 512):
        self.min_points = min_points
        self.max_points = max_points

    def __call__(self, batch):
        if self.min_points == self.max_points:
            P = self.min_points
        else:
            P = int(np.random.randint(self.min_points, self.max_points + 1))

        all_feats = []
        all_imgs = []
        all_labels = []

        for item in batch:
            v, f, label, img = item
            mesh = trimesh.Trimesh(vertices=v.numpy(), faces=f.numpy(), process=False)
            pts_np, f_idx = trimesh.sample.sample_surface(mesh, P)
            normals_np = mesh.face_normals[f_idx]

            pts = torch.tensor(pts_np, dtype=torch.float32)
            normals = torch.tensor(normals_np, dtype=torch.float32)
            feats = torch.cat([pts, normals], dim=-1)  # [P, 6]

            all_feats.append(feats)
            all_imgs.append(img if img is not None else torch.zeros((1, 28, 28), dtype=torch.float32))
            all_labels.append(label)

        return (
            torch.stack(all_feats, dim=0),      # [B, P, 6]
            torch.stack(all_imgs, dim=0),       # [B, 1, 28, 28]
            torch.tensor(all_labels, dtype=torch.long)  # [B]
        )


def main():
    parser = argparse.ArgumentParser(description="Train Arbitrary-Resolution Point Cloud MeanFlow on Digit3D")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--embed_dim", type=int, default=256, help="Transformer hidden embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--min_points", type=int, default=256, help="Minimum point count per batch during dynamic training")
    parser.add_argument("--max_points", type=int, default=512, help="Maximum point count per batch during dynamic training")
    parser.add_argument("--cond_drop_prob", type=float, default=0.15, help="Classifier-Free Guidance dropout probability")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save model checkpoints")
    parser.add_argument("--exp_name", type=str, default="naive", help="Experiment name")
    args = parser.parse_args()

    default_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.exp_name)
    save_dir = args.save_dir if args.save_dir else default_save_dir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================================")
    print("   Arbitrary-Resolution Point Cloud MeanFlow Training (Digit3D)   ")
    print("==================================================================")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Embed Dim    : {args.embed_dim}")
    print(f"Depth        : {args.depth}")
    print(f"Heads        : {args.num_heads}")
    print(f"Points Range : [{args.min_points}, {args.max_points}] (Dynamic Trimesh Collation)")
    print(f"Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Save Dir     : {save_dir}")
    print("------------------------------------------------------------------")

    # 1. Datasets & DataLoaders with Dynamic Trimesh Collation
    print("Initializing Digit3D datasets...")
    train_dataset = Digit3D(root="~/.conquer3d/", train=True, download=True, cached=True, return_img=True)
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)

    train_collate_fn = TrimeshRandomPointCollate(min_points=args.min_points, max_points=args.max_points)
    test_collate_fn = TrimeshRandomPointCollate(min_points=512, max_points=512)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Model & MeanFlow Setup
    model = ArbitraryPointFlowTransformer(
        in_channels=6,
        out_channels=6,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        img_channels=1,
        cond_drop_prob=args.cond_drop_prob,
        num_freqs=6
    ).to(device)

    mf = MeanFlow(model=model, accept_cond=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} [Train]")

        for feats, imgs, _ in pbar:
            feats = feats.to(device)
            imgs = imgs.to(device)
            B = feats.shape[0]
            P = feats.shape[1]

            optimizer.zero_grad(set_to_none=True)
            loss = mf(feats, cond=imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * B
            total_train_samples += B
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "P": P, "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_train_loss = train_loss / total_train_samples

        # Validation Step
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for feats, imgs, _ in test_loader:
                feats = feats.to(device)
                imgs = imgs.to(device)
                B = feats.shape[0]
                loss = mf(feats, cond=imgs)
                val_loss += loss.item() * B
                total_val_samples += B

        avg_val_loss = val_loss / total_val_samples
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_loss": avg_val_loss,
        }
        torch.save(ckpt, os.path.join(save_dir, "latest_model.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"  --> Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")

        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining completed successfully!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
