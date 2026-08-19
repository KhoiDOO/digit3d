import argparse
import os
import sys
import torch
import torchvision

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.generation.transformer import PointTransformer
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
    parser = argparse.ArgumentParser(description="Continuous Point Cloud Class Interpolation with Unconditional Flow Models")
    parser.add_argument('--mode', type=int, default=0, help='0: RectifiedFlow, 1: MeanFlow, 2: SoFlow')
    parser.add_argument('--class_start', type=int, default=0, help='Starting digit class (0-9)')
    parser.add_argument('--class_end', type=int, default=8, help='Target digit class (0-9)')
    parser.add_argument('--k', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--steps', type=int, default=64, help='Sampling steps for ODE solver')
    parser.add_argument('--noise_strength', type=float, default=1.0, help='Noise weighting (0.0=pure data, 1.0=pure Gaussian noise mapped to data direction)')

    # PointTransformer args (must match trained model)
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--output_channels', type=int, default=6)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--init_scale', type=float, default=0.25)
    parser.add_argument('--time_token_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--exp_name', type=str, default="", help='Custom experiment name to load from (defaults to mode name)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Initialize Unconditional Model
    print("==================================================================")
    print("        Point Cloud Latent Interpolation Pipeline                 ")
    print("==================================================================")
    print(f"Device        : {device}")
    print(f"Mode          : {args.mode} (0: RF, 1: MeanFlow, 2: SoFlow)")
    print(f"Class Start   : {args.class_start}")
    print(f"Class End     : {args.class_end}")
    print(f"Interp Steps  : {args.k}")
    print(f"Noise Strength: {args.noise_strength}")
    print("------------------------------------------------------------------")

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

    if args.mode == 0:
        flow_model = RectifiedFlow(model, time_cond_kwarg='t', predict='flow')
        mode_name = "rectified_flow"
    elif args.mode == 1:
        flow_model = MeanFlow(model, accept_cond=False)
        mode_name = "mean_flow"
    elif args.mode == 2:
        flow_model = SoFlow(model, accept_cond=False)
        mode_name = "soflow"
        args.steps = 1  # SoFlow is a 1-step generative model!
    else:
        raise ValueError("Invalid mode")

    flow_model = flow_model.to(device)

    # Resolve paths locally to the runs directory
    exp_dir = mode_name if not args.exp_name else args.exp_name
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir)
    ckpt_path = os.path.join(model_dir, "model.pt")

    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Proceeding with initialized weights for testing.")
    else:
        print(f"Loading weights from {ckpt_path}...")
        flow_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    flow_model.eval()

    # 2. Retrieve Source Samples for class_start and class_end
    print(f"Fetching point clouds for classes {args.class_start} and {args.class_end}...")
    dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=True, cached=True, num_points=args.n_ctx, return_img=True)

    start_sample = None
    end_sample = None

    for i in range(len(dataset)):
        points, features, label, img = dataset[i]
        if label == args.class_start and start_sample is None:
            start_sample = (features, img, label)
        if label == args.class_end and end_sample is None:
            end_sample = (features, img, label)
        if start_sample is not None and end_sample is not None:
            break

    if start_sample is None or end_sample is None:
        raise RuntimeError(f"Could not find samples for class {args.class_start} and/or {args.class_end} in dataset.")

    # 3. Create Latent Noise Interpolation Trajectory
    # features shape: [512, 6] -> permute to [1, 6, 512] for transformer
    pc_start = start_sample[0].to(device).permute(1, 0).unsqueeze(0)  # [1, 6, 512]
    pc_end = end_sample[0].to(device).permute(1, 0).unsqueeze(0)      # [1, 6, 512]

    noise_start = torch.randn_like(pc_start)
    noise_end = torch.randn_like(pc_end)

    # Standard Flow Matching lerp: noise -> data from 0.0 to 1.0
    t_data = 1.0 - args.noise_strength
    z_start = noise_start.lerp(pc_start, t_data)
    z_end = noise_end.lerp(pc_end, t_data)

    alphas = torch.linspace(0.0, 1.0, args.k, device=device)
    z_interp = torch.stack([z_start.lerp(z_end, alpha.item()).squeeze(0) for alpha in alphas], dim=0)

    print(f"Generated {args.k} interpolated noise tensors of shape {z_interp.shape}")

    # 4. Generate Interpolated Point Clouds with Unconditional Model
    print(f"Generating interpolated point clouds with {mode_name}...")
    with torch.no_grad():
        samples = flow_model.sample(
            batch_size=args.k,
            data_shape=(args.input_channels, args.n_ctx),
            steps=args.steps,
            noise=z_interp
        )

    # Permute back from [K, C, L] -> [K, L, C] (e.g. [K, 512, 6])
    samples = samples.permute(0, 2, 1).cpu()

    # 5. Save Interpolated Assets to Model's Run Folder
    interp_dir = os.path.join(model_dir, "interpolation", f"from_{args.class_start}_to_{args.class_end}")
    os.makedirs(interp_dir, exist_ok=True)

    # Save Ground Truth Source Assets for Reference
    save_ply(os.path.join(interp_dir, f"source_start_class_{args.class_start}.ply"), start_sample[0].numpy())
    torchvision.utils.save_image(start_sample[1], os.path.join(interp_dir, f"source_start_class_{args.class_start}.png"))
    save_ply(os.path.join(interp_dir, f"source_end_class_{args.class_end}.ply"), end_sample[0].numpy())
    torchvision.utils.save_image(end_sample[1], os.path.join(interp_dir, f"source_end_class_{args.class_end}.png"))

    # Save Interpolated Point Clouds (.ply)
    for i in range(args.k):
        alpha = alphas[i].item()
        ply_path = os.path.join(interp_dir, f"sample_{i:02d}_alpha_{alpha:.2f}.ply")
        save_ply(ply_path, samples[i].numpy())

    # Save Tensor Artifact (.pt)
    tensor_path = os.path.join(interp_dir, "interpolated_samples.pt")
    torch.save(samples, tensor_path)

    print("\n" + "=" * 66)
    print("                 INTERPOLATION GENERATION FINISHED                ")
    print("=" * 66)
    print(f"Saved {args.k} interpolated .ply files to: {interp_dir}")
    print(f"Saved complete trajectory tensor to     : {tensor_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
