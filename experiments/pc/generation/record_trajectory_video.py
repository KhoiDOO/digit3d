import os
import sys
import math
import argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from experiments.pc.generation.transformer import ImgConditionPointTransformer
from rectified_flow_pytorch import MeanFlow, RectifiedFlow


def render_aa_point_cloud(pts, norms, angle_rad, width=320, height=320, scale_ssaa=2, pitch=0.15):
    """
    Sub-pixel super-sampled anti-aliased 3D point cloud renderer.
    Maps Digit3D coordinates (X: horizontal, Z: vertical, -Y: depth)
    with 3D depth-sorting, bright pastel/normal shading, and directional illumination.
    """
    w_hi = width * scale_ssaa
    h_hi = height * scale_ssaa
    
    # Digit3D Coordinate Transformation: X -> X, Z -> Y (up), -Y -> Z (depth)
    p_x = pts[:, 0]
    p_y = pts[:, 2] # Z is digit up
    p_z = -pts[:, 1] # -Y is thickness / depth
    
    n_x = norms[:, 0]
    n_y = norms[:, 2]
    n_z = -norms[:, 1]
    
    norm_len = np.sqrt(n_x**2 + n_y**2 + n_z**2)
    norm_len[norm_len == 0] = 1.0
    n_x, n_y, n_z = n_x / norm_len, n_y / norm_len, n_z / norm_len
    
    # Rotation matrices: Yaw (around Y) & Pitch (around X)
    cos_yaw, sin_yaw = math.cos(angle_rad), math.sin(angle_rad)
    cos_pit, sin_pit = math.cos(pitch), math.sin(pitch)
    
    # Apply Yaw
    rx1 = cos_yaw * p_x + sin_yaw * p_z
    ry1 = p_y
    rz1 = -sin_yaw * p_x + cos_yaw * p_z
    
    # Apply Pitch
    rx = rx1
    ry = cos_pit * ry1 - sin_pit * rz1
    rz = sin_pit * ry1 + cos_pit * rz1
    
    # Normals rotation
    nrx1 = cos_yaw * n_x + sin_yaw * n_z
    nry1 = n_y
    nrz1 = -sin_yaw * n_x + cos_yaw * n_z
    
    nrx = nrx1
    nry = cos_pit * nry1 - sin_pit * nrz1
    nrz = sin_pit * nry1 + cos_pit * nrz1
    
    # Depth sorting (far to near)
    sort_idx = np.argsort(rz)
    s_rx = rx[sort_idx]
    s_ry = ry[sort_idx]
    s_rz = rz[sort_idx]
    s_nrx = nrx[sort_idx]
    s_nry = nry[sort_idx]
    s_nrz = nrz[sort_idx]
    
    # Perspective projection: Prominent 3D size (fov scale 0.90, cam_dist 2.2)
    cam_dist = 2.2
    depth = cam_dist - s_rz
    xs = (s_rx / depth * (w_hi * 0.90) + w_hi / 2).astype(np.int32)
    ys = (-s_ry / depth * (h_hi * 0.90) + h_hi / 2).astype(np.int32)
    
    img = Image.new("RGBA", (w_hi, h_hi), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    # Directional illumination (top-right-front)
    light_dir = np.array([0.4, 0.6, 0.7], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diff = np.clip(s_nrx * light_dir[0] + s_nry * light_dir[1] + s_nrz * light_dir[2], 0.0, 1.0)
    
    # Light, attractive, vibrant normal-mapped palette
    base_r = 0.5 * s_nrx + 0.5
    base_g = 0.5 * s_nry + 0.5
    base_b = 0.5 * s_nrz + 0.5
    
    # High ambient baseline (0.65 ambient + 0.35 diffuse) + soft highlight lift
    bright_factor = 0.68 + 0.32 * diff
    r = np.clip(base_r * bright_factor * 255 + 35, 45, 255).astype(np.uint8)
    g = np.clip(base_g * bright_factor * 255 + 35, 45, 255).astype(np.uint8)
    b = np.clip(base_b * bright_factor * 255 + 45, 55, 255).astype(np.uint8)
    
    base_rad = 6.5 * scale_ssaa # Prominent particle size
    for i in range(len(xs)):
        x_pt, y_pt = xs[i], ys[i]
        if 0 <= x_pt < w_hi and 0 <= y_pt < h_hi:
            rad = max(2, int(base_rad / (depth[i] * 0.75)))
            
            # Glowing core + bright halo
            col_main = (int(r[i]), int(g[i]), int(b[i]), 255)
            col_halo = (min(255, int(r[i]) + 35), min(255, int(g[i]) + 35), min(255, int(b[i]) + 35), 130)
            
            draw.ellipse([x_pt - rad - 2, y_pt - rad - 2, x_pt + rad + 2, y_pt + rad + 2], fill=col_halo)
            draw.ellipse([x_pt - rad, y_pt - rad, x_pt + rad, y_pt + rad], fill=col_main)
            
    # Downsample SSAA 2x -> 1x with box filter for anti-aliasing
    return img.resize((width, height), Image.LANCZOS)


def record_image_conditional_trajectory(
    img_path="docs/data/img_cond/sample_004_class_0_input.png",
    ckpt_path="experiments/pc/generation/runs/mean_flow_img_cond/model.pt",
    output_gif="experiments/pc/generation/image_conditional_trajectory.gif",
    num_ode_steps=50,
    num_rot_frames=36,
    fps=20,
    seed=42,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"Sampling MeanFlow on device: {device}")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 1. Initialize PointTransformer Architecture
    model = ImgConditionPointTransformer(
        device=device,
        dtype=torch.float32,
        input_channels=6,
        output_channels=6,
        n_ctx=512,
        width=256,
        layers=6,
        heads=8,
        init_scale=0.25,
        time_token_cond=False,
        use_checkpoint=False,
        img_channels=1,
        cond_drop_prob=0.0,
        token_cond=False
    )
    
    flow_model = MeanFlow(model, accept_cond=True).to(device)
    if os.path.exists(ckpt_path):
        print(f"Loading MeanFlow checkpoint weights from {ckpt_path}...")
        flow_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    flow_model.eval()
    
    # 2. Prepare 2D Conditioning Image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Condition image not found at {img_path}")
    raw_img = Image.open(img_path).convert("L").resize((28, 28))
    img_tensor = TF.to_tensor(raw_img).to(device).unsqueeze(0)
    raw_2d_rgba = Image.open(img_path).convert("RGBA").resize((260, 260), Image.NEAREST)
    
    # 3. Exact MeanFlow Sampling Trajectory (slow_sample step-by-step)
    print(f"Solving continuous MeanFlow ODE across {num_ode_steps} steps...")
    batch_size = 1
    data_shape = (6, 512)
    noise = flow_model.get_noise(batch_size, data_shape=data_shape)
    times = torch.linspace(1.0, 0.0, num_ode_steps + 1, device=device)[:-1]
    delta = 1.0 / num_ode_steps
    denoised = noise
    maybe_cond = (img_tensor,)
    delta_time = torch.zeros(batch_size, device=device)
    
    # Record intermediate states along ODE integration
    trajectory = []
    init_pc = flow_model.unnormalize_data_fn(denoised).permute(0, 2, 1).cpu().numpy()[0]
    trajectory.append(init_pc)
    
    with torch.no_grad():
        for time in times:
            t_exp = time.expand(batch_size)
            pred_flow = flow_model.model(denoised, t_exp, delta_time, *maybe_cond)
            denoised = denoised - delta * pred_flow
            curr_pc = flow_model.unnormalize_data_fn(denoised).permute(0, 2, 1).cpu().numpy()[0]
            trajectory.append(curr_pc)
            
    print(f"Recorded trajectory: {len(trajectory)} steps. Final shape: {trajectory[-1].shape}")
    
    # 4. Render Multi-Panel Animation Frames
    print("Rendering animation frames with vibrant, prominent 3D point cloud styling...")
    frames = []
    
    # Phase 1: Iterative ODE Generation (t = 0.00 -> 1.00)
    total_traj = len(trajectory)
    for k, pts_k in enumerate(trajectory):
        progress = k / (total_traj - 1)
        pts = pts_k[:, :3]
        norms = pts_k[:, 3:6]
        
        # Center & normalize point cloud for consistent view
        center = pts.mean(axis=0)
        pts_c = pts - center
        rad = np.max(np.linalg.norm(pts_c, axis=1))
        if rad > 0:
            pts_c /= (rad * 1.25)
        
        norm_len = np.linalg.norm(norms, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        norms_n = norms / norm_len
        
        cam_ang = 0.20 + progress * 0.25
        pc_img = render_aa_point_cloud(pts_c, norms_n, cam_ang, width=320, height=320)
        
        panel = Image.new("RGBA", (660, 350), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        
        # Paste 2D Image and 3D Viewport side-by-side
        panel.paste(raw_2d_rgba.resize((300, 300), Image.NEAREST), (20, 20))
        panel.paste(pc_img.resize((300, 300), Image.LANCZOS), (340, 20))
        
        # Progress Bar at bottom
        bar_x0, bar_y0, bar_x1, bar_y1 = 20, 330, 640, 338
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(30, 41, 59, 255))
        fill_w = int((bar_x1 - bar_x0) * progress)
        if fill_w > 0:
            draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], radius=4, fill=(99, 102, 241, 255))
            
        frames.append(panel)
        
    # Phase 2: Converged 360-degree rotation
    final_pts = trajectory[-1][:, :3]
    final_norms = trajectory[-1][:, 3:6]
    
    center = final_pts.mean(axis=0)
    final_pts_c = final_pts - center
    rad = np.max(np.linalg.norm(final_pts_c, axis=1))
    if rad > 0:
        final_pts_c /= (rad * 1.25)
    norm_len = np.linalg.norm(final_norms, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    final_norms_n = final_norms / norm_len
    
    for r in range(num_rot_frames):
        ang = 0.45 + (r / num_rot_frames) * math.pi * 2
        pc_img = render_aa_point_cloud(final_pts_c, final_norms_n, ang, width=320, height=320)
        
        panel = Image.new("RGBA", (660, 350), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        
        panel.paste(raw_2d_rgba.resize((300, 300), Image.NEAREST), (20, 20))
        panel.paste(pc_img.resize((300, 300), Image.LANCZOS), (340, 20))
        
        # Full Progress Bar
        bar_x0, bar_y0, bar_x1, bar_y1 = 20, 330, 640, 338
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(56, 189, 248, 255))
        
        frames.append(panel)
        
    # 5. Save GIF
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[SUCCESS] Trajectory GIF saved to {output_gif} ({len(frames)} frames, {os.path.getsize(output_gif) / 1024:.1f} KB)")
    
    # Mirror to docs/static/gifs/
    docs_gif = os.path.join("docs/static/gifs", os.path.basename(output_gif))
    os.makedirs(os.path.dirname(docs_gif), exist_ok=True)
    frames[0].save(docs_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[MIRRORED] Copied to {docs_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record iterative continuous flow point cloud trajectory GIF.")
    parser.add_argument("--img_path", type=str, default="docs/data/img_cond/sample_004_class_0_input.png", help="Path to 2D MNIST input image")
    parser.add_argument("--ckpt", type=str, default="experiments/pc/generation/runs/mean_flow_img_cond/model.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--output_gif", type=str, default="experiments/pc/generation/image_conditional_trajectory.gif", help="Output GIF path")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE steps")
    parser.add_argument("--rot_frames", type=int, default=36, help="Number of 360-degree rotation inspection frames")
    parser.add_argument("--fps", type=int, default=20, help="GIF playback frame rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Gaussian noise")
    args = parser.parse_args()
    
    record_image_conditional_trajectory(
        img_path=args.img_path,
        ckpt_path=args.ckpt,
        output_gif=args.output_gif,
        num_ode_steps=args.steps,
        num_rot_frames=args.rot_frames,
        fps=args.fps,
        seed=args.seed
    )
