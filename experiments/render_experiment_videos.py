import os
import math
import argparse
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. 3D Loading & Normalization Utilities
# ==============================================================================

def load_ply_points(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_ended = False
    coords = []
    normals = []
    num_verts = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        if not header_ended:
            if line.startswith("element vertex"):
                num_verts = int(line.split()[2])
            elif line == "end_header":
                header_ended = True
            continue
        parts = line.split()
        if len(coords) < num_verts and len(parts) >= 3:
            coords.append([float(parts[0]), float(parts[2]), -float(parts[1])])
            if len(parts) >= 6:
                normals.append([float(parts[3]), float(parts[5]), -float(parts[4])])
    
    pts = np.array(coords, dtype=np.float32)
    if len(normals) == len(coords) and len(coords) > 0:
        norms = np.array(normals, dtype=np.float32)
    else:
        norms = np.zeros_like(pts)
    
    # Center & normalize
    if len(pts) > 0:
        center = pts.mean(axis=0)
        pts -= center
        radius = np.max(np.linalg.norm(pts, axis=1))
        if radius > 0:
            pts /= (radius * 1.25)
            
    return pts, norms


def load_ply_mesh(path):
    mesh = trimesh.load(path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, np.zeros((0, 3), dtype=np.float32)
        
    # Center & normalize
    center = verts.mean(axis=0)
    verts -= center
    radius = np.max(np.linalg.norm(verts, axis=1))
    if radius > 0:
        verts /= (radius * 1.25)
    
    # Compute face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    face_normals /= norm_len
    
    # Coordinate system adjustment (Y up, Z depth)
    verts = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=-1)
    face_normals = np.stack([face_normals[:, 0], face_normals[:, 2], -face_normals[:, 1]], axis=-1)
    
    return verts, faces, face_normals


# ==============================================================================
# 2. Software Rasterization Viewport Renderers
# ==============================================================================

def render_point_cloud_viewport(pts, norms, angle_rad, width=280, height=280, pt_radius=3, custom_color=None):
    if len(pts) == 0:
        return Image.new("RGBA", (width, height), (15, 23, 42, 255))
        
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    
    rotated_pts = pts @ R.T
    rotated_norms = norms @ R.T
    
    # Sort points back-to-front (depth = z)
    sort_idx = np.argsort(rotated_pts[:, 2])
    sorted_pts = rotated_pts[sort_idx]
    sorted_norms = rotated_norms[sort_idx]
    
    # Perspective projection
    fov = 2.4
    depth = fov - sorted_pts[:, 2]
    xs = (sorted_pts[:, 0] / depth * (width * 0.8) + width / 2).astype(np.int32)
    ys = (-sorted_pts[:, 1] / depth * (height * 0.8) + height / 2).astype(np.int32)
    
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    if custom_color is not None:
        r_val, g_val, b_val = custom_color
        r = np.full(len(xs), r_val, dtype=np.uint8)
        g = np.full(len(xs), g_val, dtype=np.uint8)
        b = np.full(len(xs), b_val, dtype=np.uint8)
    else:
        # Normal map coloring
        nx = sorted_norms[:, 0]
        ny = sorted_norms[:, 1]
        nz = sorted_norms[:, 2]
        r = np.clip((0.5 * nx + 0.5) * 255, 30, 255).astype(np.uint8)
        g = np.clip((0.5 * ny + 0.5) * 255, 30, 255).astype(np.uint8)
        b = np.clip((0.5 * nz + 0.5) * 255, 30, 255).astype(np.uint8)
    
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if 0 <= x < width and 0 <= y < height:
            rad = max(1, int(pt_radius / (depth[i] * 0.7)))
            color = (int(r[i]), int(g[i]), int(b[i]), 240)
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=color)
            
    return img


def render_mesh_viewport(verts, faces, face_normals, angle_rad, width=280, height=280, is_voxel=False):
    if len(verts) == 0 or len(faces) == 0:
        return Image.new("RGBA", (width, height), (15, 23, 42, 255))
        
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    
    rot_verts = verts @ R.T
    rot_normals = face_normals @ R.T
    
    # Project vertices
    fov = 2.4
    depths = fov - rot_verts[:, 2]
    xs = rot_verts[:, 0] / depths * (width * 0.8) + width / 2
    ys = -rot_verts[:, 1] / depths * (height * 0.8) + height / 2
    
    # Face centers and depth sorting
    face_depths = depths[faces].mean(axis=1)
    sorted_face_indices = np.argsort(-face_depths) # Far to near
    
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    # Directional illumination
    light_dir = np.array([0.4, 0.6, 0.7], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    
    for fi in sorted_face_indices:
        fn = rot_normals[fi]
        if fn[2] < -0.1: # Back-face culling
            continue
            
        f_idx = faces[fi]
        poly = [(xs[f_idx[0]], ys[f_idx[0]]), (xs[f_idx[1]], ys[f_idx[1]]), (xs[f_idx[2]], ys[f_idx[2]])]
        
        diff = max(0.15, float(np.dot(fn, light_dir)))
        if is_voxel:
            r = int(diff * 99 + (1 - diff) * 40)
            g = int(diff * 102 + (1 - diff) * 45)
            b = int(diff * 241 + (1 - diff) * 120)
            edge_col = (56, 189, 248, 200) # cyan edge
        else:
            r = int(np.clip((0.5 * fn[0] + 0.5) * 255 * diff + 30, 0, 255))
            g = int(np.clip((0.5 * fn[1] + 0.5) * 255 * diff + 30, 0, 255))
            b = int(np.clip((0.5 * fn[2] + 0.5) * 255 * diff + 30, 0, 255))
            edge_col = None
            
        draw.polygon(poly, fill=(r, g, b, 255), outline=edge_col)
        
    return img


def render_condition_badge(title, subtitle, icon_text, width=280, height=280, badge_color=(99, 102, 241)):
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer rounded box
    box_pad = 20
    draw.rounded_rectangle([box_pad, box_pad, width - box_pad, height - box_pad], radius=16, fill=(10, 14, 26, 255), outline=badge_color, width=2)
    
    # Large Badge Icon / Symbol
    draw.text((width // 2 - 30, height // 2 - 45), icon_text, fill=badge_color)
    draw.text((width // 2 - 45, height // 2 + 15), title, fill=(241, 245, 249, 255))
    draw.text((width // 2 - 55, height // 2 + 40), subtitle, fill=(148, 163, 184, 255))
    
    return img


# ==============================================================================
# 3. Experiment Video Composers
# ==============================================================================

def compose_multi_panel_frame(panels, labels, banner_title, subtitle="", width_total=840, height_total=360):
    canvas = Image.new("RGBA", (width_total, height_total), (10, 14, 26, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Header Banner
    draw.text((20, 12), banner_title, fill=(241, 245, 249, 255))
    if subtitle:
        draw.text((20, 32), subtitle, fill=(148, 163, 184, 255))
    draw.line([20, 52, width_total - 20, 52], fill=(255, 255, 255, 30), width=1)
    
    # Place panels
    num_p = len(panels)
    pad_x = 15
    y_top = 60
    p_w = (width_total - (num_p + 1) * pad_x) // num_p
    
    for i, p_img in enumerate(panels):
        x_pos = pad_x + i * (p_w + pad_x)
        resized_p = p_img.resize((p_w, p_w), Image.BILINEAR)
        canvas.paste(resized_p, (x_pos, y_top))
        
        # Sub-panel label
        if i < len(labels):
            draw.text((x_pos + 10, y_top + p_w + 10), labels[i], fill=(56, 189, 248, 255))
            
    return canvas


def export_video_file(frames, output_path, fps=20):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[EXPORTED] {output_path} ({len(frames)} frames, {os.path.getsize(output_path) / 1024:.1f} KB)")


# ==============================================================================
# 4. Individual Experiment Generators
# ==============================================================================

def generate_exp1_unconditional(num_frames=36):
    print("Generating Experiment 1: Unconditional Generation Video...")
    ply_path = "docs/data/unconditional/sample_000_class_7.ply"
    if not os.path.exists(ply_path):
        ply_path = "experiments/pc/generation/runs/mean_flow/ply_samples/sample_000_class_7.ply"
    pts, norms = load_ply_points(ply_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        badge = render_condition_badge("Gaussian Prior", "x0 ~ N(0, I)", "N(0, I)", 280, 280, (99, 102, 241))
        pc_vp = render_point_cloud_viewport(pts, norms, ang, 280, 280, pt_radius=3)
        frame = compose_multi_panel_frame(
            [badge, pc_vp],
            ["Input Prior: Gaussian Noise", "Output: 3D Point Cloud (360°)"],
            "1. Unconditional Point Cloud Generation (Mean Flow)",
            "Spontaneous 3D generation from Gaussian velocity field",
            width_total=640, height_total=380
        )
        frames.append(frame)
        
    out1 = "experiments/pc/generation/unconditional_generation.webp"
    export_video_file(frames, out1)
    export_video_file(frames, "docs/static/videos/unconditional_generation.webp")


def generate_exp2_class_conditional(num_frames=36):
    print("Generating Experiment 2: Class-Conditional Generation Video...")
    ply_path = "docs/data/class_cond/sample_000_class_0.ply"
    if not os.path.exists(ply_path):
        ply_path = "experiments/pc/generation/runs/mean_flow_class_cond/ply_samples/sample_000_class_0.ply"
    pts, norms = load_ply_points(ply_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        badge = render_condition_badge("Class Label", "Digit: y = 0", "CLASS 0", 280, 280, (56, 189, 248))
        pc_vp = render_point_cloud_viewport(pts, norms, ang, 280, 280, pt_radius=3)
        frame = compose_multi_panel_frame(
            [badge, pc_vp],
            ["Class Condition: Digit '0'", "Generated Point Cloud (360°)"],
            "2. Class-Conditional Point Cloud Generation (CFG Flow)",
            "Targeted class-guided 3D generation across all 10 digits",
            width_total=640, height_total=380
        )
        frames.append(frame)
        
    out2 = "experiments/pc/generation/class_conditional_generation.webp"
    export_video_file(frames, out2)
    export_video_file(frames, "docs/static/videos/class_conditional_generation.webp")


def generate_exp3_image_conditional(num_frames=36):
    print("Generating Experiment 3: Image-Conditional Generation Video...")
    img_path = "docs/data/img_cond/sample_004_class_0_input.png"
    ply_path = "docs/data/img_cond/sample_004_class_0.ply"
    if not os.path.exists(img_path):
        img_path = "experiments/pc/generation/runs/mean_flow_img_cond/ply_samples/sample_004_class_0_input.png"
        ply_path = "experiments/pc/generation/runs/mean_flow_img_cond/ply_samples/sample_004_class_0.ply"
    
    raw_img = Image.open(img_path).convert("RGBA")
    pts, norms = load_ply_points(ply_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        pc_vp = render_point_cloud_viewport(pts, norms, ang, 280, 280, pt_radius=3)
        frame = compose_multi_panel_frame(
            [raw_img, pc_vp],
            ["2D MNIST Input (28x28)", "6D Normal-Mapped Point Cloud (360°)"],
            "3. Image-Conditional 3D Point Cloud Generation",
            "Single-view monocular image to 3D point cloud via Continuous Flow",
            width_total=640, height_total=380
        )
        frames.append(frame)
        
    out3 = "experiments/pc/generation/image_conditional_generation.webp"
    export_video_file(frames, out3)
    export_video_file(frames, "docs/static/videos/image_conditional_generation.webp")


def generate_exp4_arbitrary_resolution(num_frames=36):
    print("Generating Experiment 4: Arbitrary-Resolution Generation Video...")
    img_path = "docs/data/arbitrary_pc_gen/sample_004_class_0_input.png"
    ply_256 = "docs/data/arbitrary_pc_gen/sample_004_class_0_pts_256.ply"
    ply_1024 = "docs/data/arbitrary_pc_gen/sample_004_class_0_pts_1024.ply"
    ply_4096 = "docs/data/arbitrary_pc_gen/sample_004_class_0_pts_4096.ply"
    ply_16384 = "docs/data/arbitrary_pc_gen/sample_004_class_0_pts_16384.ply"
    
    raw_img = Image.open(img_path).convert("RGBA")
    pts_256, n_256 = load_ply_points(ply_256)
    pts_1024, n_1024 = load_ply_points(ply_1024)
    pts_4096, n_4096 = load_ply_points(ply_4096)
    pts_16k, n_16k = load_ply_points(ply_16384)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        vp_256 = render_point_cloud_viewport(pts_256, n_256, ang, 200, 200, pt_radius=4)
        vp_1024 = render_point_cloud_viewport(pts_1024, n_1024, ang, 200, 200, pt_radius=3)
        vp_4096 = render_point_cloud_viewport(pts_4096, n_4096, ang, 200, 200, pt_radius=2)
        vp_16k = render_point_cloud_viewport(pts_16k, n_16k, ang, 200, 200, pt_radius=1)
        
        frame = compose_multi_panel_frame(
            [raw_img, vp_256, vp_1024, vp_4096, vp_16k],
            ["2D Input", "N = 256", "N = 1024", "N = 4096", "N = 16,384"],
            "4. Arbitrary-Resolution Point Cloud Generation",
            "Multi-scale continuous point cloud synthesis from 2D images",
            width_total=1080, height_total=320
        )
        frames.append(frame)
        
    out4 = "experiments/pc/arbitrary_generation/arbitrary_pc_generation.webp"
    export_video_file(frames, out4)
    export_video_file(frames, "docs/static/videos/arbitrary_pc_generation.webp")


def generate_exp5_inpainting(num_frames=36):
    print("Generating Experiment 5: Point Cloud Inpainting Video...")
    partial_path = "experiments/pc/inpainting/runs/rectified_flow_img_cond/sample_000_class_1_input_partial.ply"
    complete_path = "experiments/pc/inpainting/runs/rectified_flow_img_cond/sample_000_class_1_completed.ply"
    if not os.path.exists(partial_path):
        partial_path = "experiments/pc/inpainting/runs/rectified_flow_class_cond/sample_008_class_5_input_partial.ply"
        complete_path = "experiments/pc/inpainting/runs/rectified_flow_class_cond/sample_008_class_5_completed.ply"
        
    pts_part, n_part = load_ply_points(partial_path)
    pts_comp, n_comp = load_ply_points(complete_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        vp_part = render_point_cloud_viewport(pts_part, n_part, ang, 280, 280, pt_radius=3, custom_color=(239, 68, 68)) # Red partial
        vp_comp = render_point_cloud_viewport(pts_comp, n_comp, ang, 280, 280, pt_radius=3)
        
        frame = compose_multi_panel_frame(
            [vp_part, vp_comp],
            ["Partial Input Point Cloud", "Inpainted & Completed 3D Cloud"],
            "5. 3D Point Cloud Inpainting & Surface Completion",
            "Continuous flow trajectory completion from masked partial points",
            width_total=640, height_total=380
        )
        frames.append(frame)
        
    out5 = "experiments/pc/inpainting/pc_inpainting.webp"
    export_video_file(frames, out5)
    export_video_file(frames, "docs/static/videos/pc_inpainting.webp")


def generate_exp6_sparse_reconstruction(num_frames=36):
    print("Generating Experiment 6: Sparse Voxel Reconstruction Video...")
    gt_path = "docs/data/sparse_recon/sample_001_class_0_gt.ply"
    recon_path = "docs/data/sparse_recon/sample_001_class_0_recon.ply"
    if not os.path.exists(gt_path):
        gt_path = "experiments/sparse_voxel/reconstruction/ply_samples/sample_000_class_0_gt.ply"
        recon_path = "experiments/sparse_voxel/reconstruction/ply_samples/sample_000_class_0_recon.ply"
        
    v_gt, f_gt, n_gt = load_ply_mesh(gt_path)
    v_rec, f_rec, n_rec = load_ply_mesh(recon_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        vp_gt = render_mesh_viewport(v_gt, f_gt, n_gt, ang, 280, 280, is_voxel=False)
        vp_rec = render_mesh_viewport(v_rec, f_rec, n_rec, ang, 280, 280, is_voxel=False)
        
        frame = compose_multi_panel_frame(
            [vp_gt, vp_rec],
            ["Ground Truth Watertight Mesh", "Sparse VAE Reconstructed Mesh"],
            "6. Sparse Voxel VAE Mesh Autoencoding",
            "Hierarchical sparse convolutional autoencoding (16-channel latent)",
            width_total=640, height_total=380
        )
        frames.append(frame)
        
    out6 = "experiments/sparse_voxel/reconstruction/sparse_voxel_reconstruction.webp"
    export_video_file(frames, out6)
    export_video_file(frames, "docs/static/videos/sparse_voxel_reconstruction.webp")


def generate_exp7_image_to_mesh(num_frames=36):
    print("Generating Experiment 7: Image-Conditional Mesh Generation Video...")
    img_path = "docs/data/sparse_voxel_gen/sample_000_class_0_input.png"
    vox_path = "docs/data/sparse_voxel_gen/sample_000_class_0_stage1_voxels.ply"
    mesh_path = "docs/data/sparse_voxel_gen/sample_000_class_0_recon.ply"
    
    raw_img = Image.open(img_path).convert("RGBA")
    v_vox, f_vox, n_vox = load_ply_mesh(vox_path)
    v_mesh, f_mesh, n_mesh = load_ply_mesh(mesh_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        vp_vox = render_mesh_viewport(v_vox, f_vox, n_vox, ang, 250, 250, is_voxel=True)
        vp_mesh = render_mesh_viewport(v_mesh, f_mesh, n_mesh, ang, 250, 250, is_voxel=False)
        
        frame = compose_multi_panel_frame(
            [raw_img, vp_vox, vp_mesh],
            ["2D MNIST Input", "Stage 1: Active Voxels", "Stage 2: Watertight Mesh"],
            "7. Image-Conditional 3D Mesh Generation",
            "Two-Stage Structure DiT + Sparse Vertex SDF Cascaded Flow",
            width_total=820, height_total=360
        )
        frames.append(frame)
        
    out7 = "experiments/sparse_voxel/generation/image_conditional_mesh_generation.webp"
    export_video_file(frames, out7)
    export_video_file(frames, "docs/static/videos/image_conditional_mesh_generation.webp")


def generate_exp8_pc_to_mesh(num_frames=36):
    print("Generating Experiment 8: PointCloud-Conditional Mesh Generation Video...")
    pc_path = "docs/data/pccond_voxel_gen/sample_000_class_0_input_pc.ply"
    vox_path = "docs/data/pccond_voxel_gen/sample_000_class_0_stage1_voxels.ply"
    mesh_path = "docs/data/pccond_voxel_gen/sample_000_class_0_recon.ply"
    
    pts, norms = load_ply_points(pc_path)
    v_vox, f_vox, n_vox = load_ply_mesh(vox_path)
    v_mesh, f_mesh, n_mesh = load_ply_mesh(mesh_path)
    
    frames = []
    for f in range(num_frames):
        ang = (f / num_frames) * math.pi * 2
        vp_pc = render_point_cloud_viewport(pts, norms, ang, 250, 250, pt_radius=3)
        vp_vox = render_mesh_viewport(v_vox, f_vox, n_vox, ang, 250, 250, is_voxel=True)
        vp_mesh = render_mesh_viewport(v_mesh, f_mesh, n_mesh, ang, 250, 250, is_voxel=False)
        
        frame = compose_multi_panel_frame(
            [vp_pc, vp_vox, vp_mesh],
            ["Input Point Cloud (N=512)", "Stage 1: Active Voxels", "Stage 2: Watertight Mesh"],
            "8. PointCloud-Conditional 3D Mesh Generation",
            "Point-conditioned Structure DiT + Sparse Vertex SDF Cascaded Flow",
            width_total=820, height_total=360
        )
        frames.append(frame)
        
    out8 = "experiments/sparse_voxel/pccond_generation/pccond_mesh_generation.webp"
    export_video_file(frames, out8)
    export_video_file(frames, "docs/static/videos/pccond_mesh_generation.webp")


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 360-degree representation videos for Digit3D experiments.")
    parser.add_argument("--all", action="store_true", help="Render representation videos for all 8 experiments.")
    parser.add_argument("--exp", type=int, default=0, help="Render specific experiment index (1-8).")
    args = parser.parse_args()
    
    os.makedirs("docs/static/videos", exist_ok=True)
    
    if args.all or args.exp == 1: generate_exp1_unconditional()
    if args.all or args.exp == 2: generate_exp2_class_conditional()
    if args.all or args.exp == 3: generate_exp3_image_conditional()
    if args.all or args.exp == 4: generate_exp4_arbitrary_resolution()
    if args.all or args.exp == 5: generate_exp5_inpainting()
    if args.all or args.exp == 6: generate_exp6_sparse_reconstruction()
    if args.all or args.exp == 7: generate_exp7_image_to_mesh()
    if args.all or args.exp == 8: generate_exp8_pc_to_mesh()
    
    print("\n[SUCCESS] All requested experiment representation videos successfully rendered!")
