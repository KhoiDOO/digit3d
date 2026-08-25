import os
import sys
import math
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath("."))

from experiments.pc.generation.transformer import PointTransformer
from rectified_flow_pytorch import MeanFlow


def render_aa_point_cloud(pts, norms, angle_rad, width=320, height=320, scale_ssaa=2, pitch=0.15):
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
    
    cos_yaw, sin_yaw = math.cos(angle_rad), math.sin(angle_rad)
    cos_pit, sin_pit = math.cos(pitch), math.sin(pitch)
    
    rx1 = cos_yaw * p_x + sin_yaw * p_z
    ry1 = p_y
    rz1 = -sin_yaw * p_x + cos_yaw * p_z
    
    rx = rx1
    ry = cos_pit * ry1 - sin_pit * rz1
    rz = sin_pit * ry1 + cos_pit * rz1
    
    nrx1 = cos_yaw * n_x + sin_yaw * n_z
    nry1 = n_y
    nrz1 = -sin_yaw * n_x + cos_yaw * n_z
    
    nrx = nrx1
    nry = cos_pit * nry1 - sin_pit * nrz1
    nrz = sin_pit * nry1 + cos_pit * nrz1
    
    sort_idx = np.argsort(rz)
    s_rx = rx[sort_idx]
    s_ry = ry[sort_idx]
    s_rz = rz[sort_idx]
    s_nrx = nrx[sort_idx]
    s_nry = nry[sort_idx]
    s_nrz = nrz[sort_idx]
    
    cam_dist = 2.2
    depth = cam_dist - s_rz
    xs = (s_rx / depth * (w_hi * 0.90) + w_hi / 2).astype(np.int32)
    ys = (-s_ry / depth * (h_hi * 0.90) + h_hi / 2).astype(np.int32)
    
    img = Image.new("RGBA", (w_hi, h_hi), (10, 14, 26, 255))
    draw = ImageDraw.Draw(img)
    
    light_dir = np.array([0.4, 0.6, 0.7], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diff = np.clip(s_nrx * light_dir[0] + s_nry * light_dir[1] + s_nrz * light_dir[2], 0.0, 1.0)
    
    base_r = 0.5 * s_nrx + 0.5
    base_g = 0.5 * s_nry + 0.5
    base_b = 0.5 * s_nrz + 0.5
    
    bright_factor = 0.68 + 0.32 * diff
    r = np.clip(base_r * bright_factor * 255 + 35, 45, 255).astype(np.uint8)
    g = np.clip(base_g * bright_factor * 255 + 35, 45, 255).astype(np.uint8)
    b = np.clip(base_b * bright_factor * 255 + 45, 55, 255).astype(np.uint8)
    
    base_rad = 6.5 * scale_ssaa
    for i in range(len(xs)):
        x_pt, y_pt = xs[i], ys[i]
        if 0 <= x_pt < w_hi and 0 <= y_pt < h_hi:
            rad = max(2, int(base_rad / (depth[i] * 0.75)))
            col_main = (int(r[i]), int(g[i]), int(b[i]), 255)
            col_halo = (min(255, int(r[i]) + 35), min(255, int(g[i]) + 35), min(255, int(b[i]) + 35), 130)
            
            draw.ellipse([x_pt - rad - 2, y_pt - rad - 2, x_pt + rad + 2, y_pt + rad + 2], fill=col_halo)
            draw.ellipse([x_pt - rad, y_pt - rad, x_pt + rad, y_pt + rad], fill=col_main)
            
    return img.resize((width, height), Image.LANCZOS)


def record_unconditional_trajectory(
    ckpt_path="experiments/pc/generation/runs/mean_flow/model.pt",
    output_gif="docs/static/gifs/unconditional_trajectory.gif",
    num_ode_steps=50,
    num_rot_frames=36,
    fps=20,
    seeds=[42, 105, 2024, 777],
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"Sampling Unconditional MeanFlow on device: {device}")
    
    # Initialize Model
    model = PointTransformer(
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
        use_checkpoint=False
    )
    
    flow_model = MeanFlow(model, accept_cond=False).to(device)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights from {ckpt_path}...")
        flow_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    flow_model.eval()
    
    # Sample trajectories for each seed
    all_trajectories = []
    
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        
        batch_size = 1
        data_shape = (6, 512)
        noise = flow_model.get_noise(batch_size, data_shape=data_shape)
        times = torch.linspace(1.0, 0.0, num_ode_steps + 1, device=device)[:-1]
        delta = 1.0 / num_ode_steps
        denoised = noise
        delta_time = torch.zeros(batch_size, device=device)
        
        traj = []
        init_pc = flow_model.unnormalize_data_fn(denoised).permute(0, 2, 1).cpu().numpy()[0]
        traj.append(init_pc)
        
        with torch.no_grad():
            for time in times:
                t_exp = time.expand(batch_size)
                pred_flow = flow_model.model(denoised, t_exp, delta_time)
                denoised = denoised - delta * pred_flow
                curr_pc = flow_model.unnormalize_data_fn(denoised).permute(0, 2, 1).cpu().numpy()[0]
                traj.append(curr_pc)
                
        all_trajectories.append(traj)
        
    print(f"Generated {len(all_trajectories)} unconditional trajectories of {len(all_trajectories[0])} steps each.")
    
    # Render 4-sample 2x2 grid animation GIF
    num_samples = len(all_trajectories)
    w_tile, h_tile = 260, 260
    total_traj = len(all_trajectories[0])
    
    frames = []
    
    # Phase 1: ODE Iterations (t = 1.00 -> 0.00)
    for k in range(total_traj):
        progress = k / (total_traj - 1)
        cam_ang = 0.20 + progress * 0.25
        
        # 2x2 grid canvas: width = 2*260 + 3*15 = 565, height = 2*260 + 3*15 + 15 = 580
        panel = Image.new("RGBA", (565, 575), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        
        for idx in range(num_samples):
            pts_k = all_trajectories[idx][k]
            pts = pts_k[:, :3]
            norms = pts_k[:, 3:6]
            
            center = pts.mean(axis=0)
            pts_c = pts - center
            rad = np.max(np.linalg.norm(pts_c, axis=1))
            if rad > 0:
                pts_c /= (rad * 1.25)
                
            norm_len = np.linalg.norm(norms, axis=1, keepdims=True)
            norm_len[norm_len == 0] = 1.0
            norms_n = norms / norm_len
            
            tile_img = render_aa_point_cloud(pts_c, norms_n, cam_ang, width=w_tile, height=h_tile)
            
            row, col = idx // 2, idx % 2
            pos_x = 15 + col * (w_tile + 15)
            pos_y = 15 + row * (h_tile + 15)
            panel.paste(tile_img, (pos_x, pos_y))
            
        # Progress Bar at bottom
        bar_x0, bar_y0, bar_x1, bar_y1 = 15, 555, 550, 563
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(30, 41, 59, 255))
        fill_w = int((bar_x1 - bar_x0) * progress)
        if fill_w > 0:
            draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], radius=4, fill=(99, 102, 241, 255))
            
        frames.append(panel)
        
    # Phase 2: Converged 360-degree rotation
    for r in range(num_rot_frames):
        ang = 0.45 + (r / num_rot_frames) * math.pi * 2
        
        panel = Image.new("RGBA", (565, 575), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        
        for idx in range(num_samples):
            final_pts = all_trajectories[idx][-1][:, :3]
            final_norms = all_trajectories[idx][-1][:, 3:6]
            
            center = final_pts.mean(axis=0)
            final_pts_c = final_pts - center
            rad = np.max(np.linalg.norm(final_pts_c, axis=1))
            if rad > 0:
                final_pts_c /= (rad * 1.25)
                
            norm_len = np.linalg.norm(final_norms, axis=1, keepdims=True)
            norm_len[norm_len == 0] = 1.0
            final_norms_n = final_norms / norm_len
            
            tile_img = render_aa_point_cloud(final_pts_c, final_norms_n, ang, width=w_tile, height=h_tile)
            
            row, col = idx // 2, idx % 2
            pos_x = 15 + col * (w_tile + 15)
            pos_y = 15 + row * (h_tile + 15)
            panel.paste(tile_img, (pos_x, pos_y))
            
        # Full green/cyan progress bar
        bar_x0, bar_y0, bar_x1, bar_y1 = 15, 555, 550, 563
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(56, 189, 248, 255))
        
        frames.append(panel)
        
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[SUCCESS] Saved unconditional trajectory GIF to {output_gif} ({len(frames)} frames, {os.path.getsize(output_gif) / 1024:.1f} KB)")
    
    # Also save to experiments/pc/generation/unconditional_trajectory.gif
    exp_gif = "experiments/pc/generation/unconditional_trajectory.gif"
    frames[0].save(exp_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[MIRRORED] Saved to {exp_gif}")

    # Also render single-sample high-res GIF
    single_gif = "docs/static/gifs/unconditional_single_trajectory.gif"
    single_frames = []
    traj_single = all_trajectories[0]
    
    for k in range(total_traj):
        progress = k / (total_traj - 1)
        cam_ang = 0.20 + progress * 0.25
        pts = traj_single[k][:, :3]
        norms = traj_single[k][:, 3:6]
        
        center = pts.mean(axis=0)
        pts_c = pts - center
        rad = np.max(np.linalg.norm(pts_c, axis=1))
        if rad > 0:
            pts_c /= (rad * 1.25)
        norm_len = np.linalg.norm(norms, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        norms_n = norms / norm_len
        
        tile_img = render_aa_point_cloud(pts_c, norms_n, cam_ang, width=320, height=320)
        
        panel = Image.new("RGBA", (350, 360), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        panel.paste(tile_img, (15, 15))
        
        bar_x0, bar_y0, bar_x1, bar_y1 = 15, 342, 335, 350
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(30, 41, 59, 255))
        fill_w = int((bar_x1 - bar_x0) * progress)
        if fill_w > 0:
            draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], radius=4, fill=(99, 102, 241, 255))
        single_frames.append(panel)
        
    for r in range(num_rot_frames):
        ang = 0.45 + (r / num_rot_frames) * math.pi * 2
        final_pts = traj_single[-1][:, :3]
        final_norms = traj_single[-1][:, 3:6]
        center = final_pts.mean(axis=0)
        final_pts_c = final_pts - center
        rad = np.max(np.linalg.norm(final_pts_c, axis=1))
        if rad > 0:
            final_pts_c /= (rad * 1.25)
        norm_len = np.linalg.norm(final_norms, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        final_norms_n = final_norms / norm_len
        
        tile_img = render_aa_point_cloud(final_pts_c, final_norms_n, ang, width=320, height=320)
        
        panel = Image.new("RGBA", (350, 360), (10, 14, 26, 255))
        draw = ImageDraw.Draw(panel)
        panel.paste(tile_img, (15, 15))
        
        bar_x0, bar_y0, bar_x1, bar_y1 = 15, 342, 335, 350
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(56, 189, 248, 255))
        single_frames.append(panel)
        
    single_frames[0].save(single_gif, save_all=True, append_images=single_frames[1:], duration=duration_ms, loop=0)
    print(f"[SUCCESS] Saved single unconditional trajectory GIF to {single_gif} ({len(single_frames)} frames, {os.path.getsize(single_gif) / 1024:.1f} KB)")


if __name__ == "__main__":
    record_unconditional_trajectory()
