import argparse
import os
import sys
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.generation.transformer import PointTransformer, ClassConditionedPointTransformer, ImgConditionPointTransformer
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow


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
    parser = argparse.ArgumentParser(description="Point Cloud Generation Pipeline with Optional Class or Image Conditioning")
    parser.add_argument('--mode', type=int, default=0, help='0: RectifiedFlow, 1: MeanFlow, 2: SoFlow')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=64, help='Sampling steps for ODE solver')

    # PointTransformer args (must match training!)
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--output_channels', type=int, default=6)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--init_scale', type=float, default=0.25)
    parser.add_argument('--time_token_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--class_cond', action='store_true', help="Use class conditioning")
    parser.add_argument('--img_cond', action='store_true', help="Use image conditioning")
    parser.add_argument('--class_token_cond', action='store_true', help="Pass condition as a token")

    # Image-conditioned specific arguments
    parser.add_argument('--img_path', type=str, default="", help="Path to a custom input image file to condition on")
    parser.add_argument('--class_label', type=int, default=-1, help="Filter dataset by specific digit class (0-9)")
    parser.add_argument('--sample_offset', type=int, default=0, help="Offset in dataset when selecting test samples")
    parser.add_argument('--cfg_scale', type=float, default=1.0, help="Classifier-free guidance scale (default: 1.0)")

    parser.add_argument('--exp_name', type=str, default="", help="Custom experiment name to load from")
    parser.add_argument('--ckpt', type=str, default="", help="Explicit path to model checkpoint (.pt)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Initialize Model
    if args.img_cond:
        model = ImgConditionPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            n_ctx=args.n_ctx,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
            img_channels=1,
            cond_drop_prob=0.0,  # No dropout at inference
            token_cond=args.class_token_cond
        )
    elif args.class_cond:
        model = ClassConditionedPointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            n_ctx=args.n_ctx,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
            num_classes=10,
            cond_drop_prob=0.0,  # No dropout at inference
            token_cond=args.class_token_cond
        )
    else:
        model = PointTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            n_ctx=args.n_ctx,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint
        )

    accept_cond = args.class_cond or args.img_cond
    if args.mode == 0:
        flow_model = RectifiedFlow(model, time_cond_kwarg='t', predict='flow')
        mode_name = "rectified_flow"
    elif args.mode == 1:
        flow_model = MeanFlow(model, accept_cond=accept_cond)
        mode_name = "mean_flow"
    elif args.mode == 2:
        flow_model = SoFlow(model, accept_cond=accept_cond)
        mode_name = "soflow"
        args.steps = 1  # SoFlow is a 1-step generative model!
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    flow_model = flow_model.to(device)

    # 2. Resolve Model Checkpoint Path
    if args.ckpt:
        ckpt_path = args.ckpt
        exp_dir = args.exp_name if args.exp_name else mode_name
    else:
        exp_suffix = "_img_cond" if args.img_cond else ("_class_cond" if args.class_cond else "")
        exp_dir = f"{mode_name}{exp_suffix}" if not args.exp_name else args.exp_name
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "model.pt")

    if not os.path.exists(ckpt_path):
        print(f"Warning: Could not find model weights at {ckpt_path}. Proceeding with initialized weights for testing.")
    else:
        print(f"Loading weights from {ckpt_path}...")
        flow_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    flow_model.eval()

    # 3. Prepare Conditioning Inputs
    cond_imgs = None
    cond_classes = None
    labels = []

    if args.img_cond:
        if args.img_path:
            # Custom 2D image file provided
            if not os.path.exists(args.img_path):
                raise FileNotFoundError(f"Image not found at {args.img_path}")
            raw_img = Image.open(args.img_path).convert("L").resize((28, 28))
            img_tensor = TF.to_tensor(raw_img).to(device)  # [1, 28, 28]
            cond_imgs = img_tensor.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)  # [B, 1, 28, 28]
            base_name = os.path.splitext(os.path.basename(args.img_path))[0]
            labels = [base_name] * args.num_samples
            print(f"Conditioning on custom image: {args.img_path} ({args.num_samples} generations)")
        elif args.class_label >= 0:
            # Filter test dataset by class
            print(f"Querying test dataset for class {args.class_label} samples...")
            test_dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=True, cached=True, num_points=args.n_ctx, return_img=True)
            matched_imgs = []
            matched_labels = []
            for i in range(args.sample_offset, len(test_dataset)):
                _, _, lbl, img = test_dataset[i]
                if lbl == args.class_label:
                    matched_imgs.append(img)
                    matched_labels.append(lbl)
                    if len(matched_imgs) == args.num_samples:
                        break
            if len(matched_imgs) == 0:
                raise RuntimeError(f"No samples found for class {args.class_label}")
            cond_imgs = torch.stack(matched_imgs).to(device)
            labels = matched_labels
        else:
            # Sliced sequential test dataset samples
            print(f"Streaming {args.num_samples} samples from test dataset (offset={args.sample_offset})...")
            test_dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=True, cached=True, num_points=args.n_ctx, return_img=True)
            cond_imgs = torch.stack([test_dataset[args.sample_offset + i][3] for i in range(args.num_samples)]).to(device)
            labels = [test_dataset[args.sample_offset + i][2] for i in range(args.num_samples)]
    elif args.class_cond:
        if args.class_label >= 0:
            cond_classes = torch.full((args.num_samples,), args.class_label, device=device, dtype=torch.long)
        else:
            cond_classes = (torch.arange(args.num_samples, device=device) % 10).long()
        labels = [c.item() for c in cond_classes]

    print(f"Generating {args.num_samples} samples using {mode_name} (steps={args.steps}, cfg_scale={args.cfg_scale})...")

    # 4. Inference / Sampling Loop
    with torch.no_grad():
        if args.cfg_scale > 1.0 and (args.img_cond or args.class_cond) and args.mode == 0:
            # Custom Euler ODE sampling with Classifier-Free Guidance (CFG)
            batch_size = args.num_samples
            shape = (batch_size, args.input_channels, args.n_ctx)
            x = torch.randn(shape, device=device)
            times = torch.linspace(0.0, 1.0, args.steps, device=device)
            dt = 1.0 / (args.steps - 1) if args.steps > 1 else 1.0

            cond_payload = cond_imgs if args.img_cond else cond_classes
            for step_idx in range(len(times) - 1):
                t_curr = times[step_idx].expand(batch_size)
                # Conditional velocity
                v_cond = model(x, t=t_curr, cond=cond_payload)
                # Unconditional velocity
                v_uncond = model(x, t=t_curr, cond=None)
                # CFG extrapolation
                v = v_uncond + args.cfg_scale * (v_cond - v_uncond)
                x = x + v * dt
            samples = x
        else:
            if args.img_cond:
                samples = flow_model.sample(
                    batch_size=args.num_samples,
                    data_shape=(args.input_channels, args.n_ctx),
                    steps=args.steps,
                    cond=cond_imgs
                )
            elif args.class_cond:
                samples = flow_model.sample(
                    batch_size=args.num_samples,
                    data_shape=(args.input_channels, args.n_ctx),
                    steps=args.steps,
                    cond=cond_classes
                )
            else:
                samples = flow_model.sample(
                    batch_size=args.num_samples,
                    data_shape=(args.input_channels, args.n_ctx),
                    steps=args.steps
                )

    # Permute back from [B, C, L] -> [B, L, C] (e.g. [B, 512, 6])
    samples = samples.permute(0, 2, 1).cpu()

    # 5. Save Output Artifacts
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "samples.pt")
    torch.save(samples, save_path)
    print(f"Saved generated samples tensor to {save_path} (Shape: {samples.shape})")

    samples_dir = os.path.join(save_dir, "ply_samples")
    os.makedirs(samples_dir, exist_ok=True)

    for i in range(args.num_samples):
        if args.img_cond:
            class_label = labels[i]
            ply_path = os.path.join(samples_dir, f"sample_{i:03d}_class_{class_label}.ply")
            png_path = os.path.join(samples_dir, f"sample_{i:03d}_class_{class_label}_input.png")
            torchvision.utils.save_image(cond_imgs[i].cpu(), png_path)
        elif args.class_cond:
            class_label = labels[i]
            ply_path = os.path.join(samples_dir, f"sample_{i:03d}_class_{class_label}.ply")
        else:
            ply_path = os.path.join(samples_dir, f"sample_{i:03d}.ply")

        save_ply(ply_path, samples[i].numpy())

    print(f"Saved {args.num_samples} individual .ply point cloud files to {samples_dir}")


if __name__ == "__main__":
    main()
