import os
import io
import math
import zipfile
import numpy as np
from PIL import Image, ImageDraw

def load_mesh_from_zip(zip_path, obj_name):
    with zipfile.ZipFile(zip_path, "r") as z:
        content = z.read(obj_name).decode("utf-8")
    
    verts = []
    faces = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            face_idx = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
            faces.append(face_idx)
            
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    
    # Center & normalize
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts = verts - center
    scale = np.max(np.abs(verts))
    if scale > 0:
        verts = verts / scale
        
    return verts, faces


def render_mesh_frame(verts, faces, angle_rad, width=200, height=200, scale_ssaa=2, pitch=0.18):
    w_hi = width * scale_ssaa
    h_hi = height * scale_ssaa
    
    # Digit3D Coordinate Transformation: X -> X, Z -> Y (up), -Y -> Z (depth)
    p_x = verts[:, 0]
    p_y = verts[:, 2] # Z is digit up
    p_z = -verts[:, 1] # -Y is thickness / depth
    
    cos_yaw, sin_yaw = math.cos(angle_rad), math.sin(angle_rad)
    cos_pit, sin_pit = math.cos(pitch), math.sin(pitch)
    
    # Yaw rotation
    rx1 = cos_yaw * p_x + sin_yaw * p_z
    ry1 = p_y
    rz1 = -sin_yaw * p_x + cos_yaw * p_z
    
    # Pitch rotation
    rx = rx1
    ry = cos_pit * ry1 - sin_pit * rz1
    rz = sin_pit * ry1 + cos_pit * rz1
    
    # Perspective projection
    cam_dist = 2.4
    depth = cam_dist - rz
    xs = rx / depth * (w_hi * 0.85) + w_hi / 2
    ys = -ry / depth * (h_hi * 0.85) + h_hi / 2
    
    f_v0 = faces[:, 0]
    f_v1 = faces[:, 1]
    f_v2 = faces[:, 2]
    
    # 3D face normal calculation in camera space
    v0_3d = np.stack([rx[f_v0], ry[f_v0], rz[f_v0]], axis=-1)
    v1_3d = np.stack([rx[f_v1], ry[f_v1], rz[f_v1]], axis=-1)
    v2_3d = np.stack([rx[f_v2], ry[f_v2], rz[f_v2]], axis=-1)
    
    edge1 = v1_3d - v0_3d
    edge2 = v2_3d - v0_3d
    fn = np.cross(edge1, edge2)
    norm = np.linalg.norm(fn, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    fn = fn / norm
    
    face_depths = (rz[f_v0] + rz[f_v1] + rz[f_v2]) / 3.0
    sort_idx = np.argsort(face_depths)
    
    img = Image.new("RGBA", (w_hi, h_hi), (10, 14, 26, 255))
    draw = ImageDraw.Draw(img)
    
    # Top-right-front light direction
    light_dir = np.array([0.4, 0.6, 0.7], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diff = np.clip(np.sum(fn * light_dir, axis=-1), 0.0, 1.0)
    
    # Vibrant, light studio normal-mapped mesh shading
    base_r = 0.5 * fn[:, 0] + 0.5
    base_g = 0.5 * fn[:, 1] + 0.5
    base_b = 0.5 * fn[:, 2] + 0.5
    
    bright_factor = 0.65 + 0.35 * diff
    r = np.clip(base_r * bright_factor * 255 + 35, 40, 255).astype(np.uint8)
    g = np.clip(base_g * bright_factor * 255 + 35, 40, 255).astype(np.uint8)
    b = np.clip(base_b * bright_factor * 255 + 45, 50, 255).astype(np.uint8)
    
    for idx in sort_idx:
        poly = [
            (xs[f_v0[idx]], ys[f_v0[idx]]),
            (xs[f_v1[idx]], ys[f_v1[idx]]),
            (xs[f_v2[idx]], ys[f_v2[idx]])
        ]
        color = (int(r[idx]), int(g[idx]), int(b[idx]), 255)
        outline_color = (min(255, int(r[idx]) + 25), min(255, int(g[idx]) + 25), min(255, int(b[idx]) + 25), 160)
        draw.polygon(poly, fill=color, outline=outline_color)
        
    return img.resize((width, height), Image.LANCZOS)


def generate_all_digit_gifs(zip_path, output_dir="docs/static/gifs", num_frames=36, fps=20):
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        
    # Find clean test samples for digits 0 to 9
    digit_samples = {}
    for name in names:
        if name.startswith("src/test/") and name.endswith(".obj"):
            parts = os.path.basename(name).split("_")
            digit = int(parts[0])
            if digit not in digit_samples:
                digit_samples[digit] = name
                if len(digit_samples) == 10:
                    break
                    
    duration_ms = int(1000 / fps)
    
    for digit in range(10):
        obj_name = digit_samples[digit]
        verts, faces = load_mesh_from_zip(zip_path, obj_name)
        print(f"Rendering Digit {digit} ({obj_name}) - {len(verts)} verts, {len(faces)} faces...")
        
        frames = []
        for f in range(num_frames):
            ang = (f / num_frames) * math.pi * 2
            img = render_mesh_frame(verts, faces, angle_rad=ang, width=200, height=200)
            frames.append(img)
            
        out_path = os.path.join(output_dir, f"mesh_digit_{digit}.gif")
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        print(f"  -> Saved {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    zip_path = os.path.expanduser("~/.conquer3d/digit3d.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find {zip_path}")
    generate_all_digit_gifs(zip_path)
