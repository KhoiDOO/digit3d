import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Append root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import conquer3d as c3d
from conquer3d.conversion.grid import sparse_coo2dense_occ
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data_structure.bmesh import BTriangleMesh
from conquer3d.data.dataset.digit3d import Digit3D
from rectified_flow_pytorch import MeanFlow
from experiments.sparse_voxel.generation.stage1.models import StructureDiT


def bmesh_img_collate_fn(batch):
    all_vertices, all_faces, all_vertbids, all_facebids, all_labels, all_imgs = [], [], [], [], [], []
    for b, item in enumerate(batch):
        if len(item) == 4:
            v, f, l, img = item
        else:
            v, f, l = item[:3]
            img = torch.zeros((1, 28, 28), dtype=torch.float32)
        all_vertices.append(v)
        all_faces.append(f)
        all_vertbids.append(torch.full((v.shape[0],), b, dtype=torch.int32))
        all_facebids.append(torch.full((f.shape[0],), b, dtype=torch.int32))
        all_labels.append(l)
        all_imgs.append(img if img is not None else torch.zeros((1, 28, 28), dtype=torch.float32))

    bmesh = BTriangleMesh(
        vertices=torch.cat(all_vertices, dim=0),
        faces=torch.cat(all_faces, dim=0),
        vertbids=torch.cat(all_vertbids, dim=0),
        facebids=torch.cat(all_facebids, dim=0),
        batch_size=len(batch)
    )
    return bmesh, torch.stack(all_imgs, dim=0), torch.tensor(all_labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Structure DiT MeanFlow on Digit3D")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patch_size", type=int, default=4, help="3D patch size (8 for 4^3=64 tokens, 2 for 16^3=4096 tokens)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save model weights")
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================================")
    print("      Digit3D Stage 1 Structure DiT MeanFlow Training             ")
    print("==================================================================")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Embed Dim    : {args.embed_dim}")
    print(f"Depth        : {args.depth}")
    print(f"Heads        : {args.num_heads}")
    print(f"Device       : {device}")
    print(f"Save Dir     : {save_dir}")
    print("------------------------------------------------------------------")

    # 1. Datasets & Loaders
    print("Initializing Digit3D datasets...")
    train_dataset = Digit3D(root="~/.conquer3d/", train=True, download=True, cached=True, return_img=True)
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bmesh_img_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bmesh_img_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Model & MeanFlow Setup
    model = StructureDiT(
        grid_res=32,
        patch_size=args.patch_size,
        in_channels=1,
        out_channels=1,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        cond_dim=args.embed_dim,
        img_channels=1
    ).to(device)

    mf = MeanFlow(model=model, accept_cond=True, data_shape=(1, 32, 32, 32)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")

        for bmesh, imgs, _ in pbar:
            bmesh = bmesh.to(device)
            imgs = imgs.to(device)

            with torch.no_grad():
                sparse_coords, _ = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
                x_1 = sparse_coo2dense_occ(sparse_coords, bmesh.batch_size, (32, 32, 32))

            optimizer.zero_grad()
            loss = mf(x_1, cond=imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bmesh.batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bmesh, imgs, _ in test_loader:
                bmesh = bmesh.to(device)
                imgs = imgs.to(device)
                sparse_coords, _ = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
                x_1 = sparse_coo2dense_occ(sparse_coords, bmesh.batch_size, (32, 32, 32))

                loss = mf(x_1, cond=imgs)
                val_loss += loss.item() * bmesh.batch_size

        val_loss /= len(test_dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save Best & Latest
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "stage1_structure.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Best model saved to {ckpt_path} (Val Loss: {best_val_loss:.4f})")

        with open(os.path.join(save_dir, "stage1_structure_train.json"), "w") as f:
            json.dump(history, f, indent=4)

    print(f"\nStage 1 Training Complete! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
