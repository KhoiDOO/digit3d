import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import trimesh
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import Digit3D
from experiments.pc.arbitrary_generation.transformer import (
    ClassConditionedPointTransformer,
    ImgConditionPointTransformer,
    PointTransformer,
)
from rectified_flow_pytorch import MeanFlow, RectifiedFlow
from rectified_flow_pytorch.soflow import SoFlow


class TrimeshRandomPointCollate:
    """
    Collate function that samples a synchronized point budget P ~ Uniform(min_points, max_points)
    for each batch and extracts 6D surface points [P, 6] (coordinates + face normals) across DataLoader workers.
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
    parser = argparse.ArgumentParser(description="Train Size-Varying Point Cloud Flow Model")
    parser.add_argument('--mode', type=int, default=1, help='0: RectifiedFlow, 1: MeanFlow, 2: SoFlow')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--min_points', type=int, default=256, help="Minimum point count per batch")
    parser.add_argument('--max_points', type=int, default=512, help="Maximum point count per batch")
    
    # PointTransformer args
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--output_channels', type=int, default=6)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--init_scale', type=float, default=0.25)
    parser.add_argument('--time_token_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--class_cond', action='store_true', help="Use class conditioning")
    parser.add_argument('--img_cond', action='store_true', default=True, help="Use image conditioning")
    parser.add_argument('--class_token_cond', action='store_true', help="Pass condition as a token")
    parser.add_argument('--cond_drop_prob', type=float, default=0.15, help="CFG drop probability")
    parser.add_argument('--exp_name', type=str, default="naive", help="Custom experiment name for the run folder")
    parser.add_argument('--num_workers', type=int, default=8, help="DataLoader workers")
    
    args = parser.parse_args()

    # Determine save directory
    if args.exp_name:
        run_name = args.exp_name
    else:
        mode_str = "rf" if args.mode == 0 else ("mf" if args.mode == 1 else "soflow")
        cond_str = "img" if args.img_cond else ("class" if args.class_cond else "uncond")
        run_name = f"{mode_str}_{cond_str}_arbitrary"
        
    save_dir = os.path.join(os.path.dirname(__file__), "runs", run_name)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==================================================================")
    print("   Arbitrary Point Cloud Generation Training (Self-Attention)     ")
    print("==================================================================")
    print(f"Mode         : {args.mode} (0: RectifiedFlow, 1: MeanFlow, 2: SoFlow)")
    print(f"Condition    : {'Image' if args.img_cond else ('Class' if args.class_cond else 'Unconditional')}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Points Range : [{args.min_points}, {args.max_points}] (Dynamic Trimesh Collation)")
    print(f"Save Dir     : {save_dir}")
    print(f"Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("------------------------------------------------------------------")

    print("Initializing Digit3D datasets...")
    train_dataset = Digit3D(root="~/.conquer3d/", train=True, download=True, cached=True, return_img=args.img_cond)
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True, return_img=args.img_cond)

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

    print("Initializing Model...")
    if args.img_cond:
        model = ImgConditionPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
            img_channels=1,
            cond_drop_prob=args.cond_drop_prob,
            token_cond=args.class_token_cond
        )
    elif args.class_cond:
        model = ClassConditionedPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
            num_classes=10,
            cond_drop_prob=args.cond_drop_prob,
            token_cond=args.class_token_cond
        )
    else:
        model = PointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
        )

    accept_cond = args.class_cond or args.img_cond
    if args.mode == 0:
        flow_model = RectifiedFlow(model=model, accept_cond=accept_cond).to(device)
    elif args.mode == 1:
        flow_model = MeanFlow(model=model, accept_cond=accept_cond).to(device)
    elif args.mode == 2:
        flow_model = SoFlow(model=model, accept_cond=accept_cond).to(device)
    else:
        raise ValueError("Mode must be 0 (RectifiedFlow), 1 (MeanFlow), or 2 (SoFlow)")

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

        for feats, imgs, labels in pbar:
            feats = feats.to(device)  # [B, P, 6]
            B = feats.shape[0]
            P = feats.shape[1]

            if args.img_cond:
                cond = imgs.to(device)
            elif args.class_cond:
                cond = labels.to(device)
            else:
                cond = None

            optimizer.zero_grad(set_to_none=True)
            if cond is not None:
                loss = flow_model(feats, cond=cond)
            else:
                loss = flow_model(feats)

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
            for feats, imgs, labels in test_loader:
                feats = feats.to(device)
                B = feats.shape[0]

                if args.img_cond:
                    cond = imgs.to(device)
                elif args.class_cond:
                    cond = labels.to(device)
                else:
                    cond = None

                if cond is not None:
                    loss = flow_model(feats, cond=cond)
                else:
                    loss = flow_model(feats)

                val_loss += loss.item() * B
                total_val_samples += B

        avg_val_loss = val_loss / total_val_samples
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
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
