import os
import sys
import warnings
import torch
from torch.utils.data import DataLoader
from torchsparse import SparseTensor

# Append digit3d root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import conquer3d as c3d
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.data.dataset.digit3d import Digit3D
from experiments.sparse_voxel.reconstruction.models import DiagonalGaussianDistribution, SimpleSparseVAE

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
dataset = Digit3D(root="~/.conquer3d/", train=False, download=True, cached=True)
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=bmesh_collate_fn)

batch = next(iter(loader))
bmesh, _ = batch
bmesh = bmesh.to(device)
bmesh.vertices = bmesh.vertices.float()

batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
batched_coords = batched_coords.to(device)
batched_sdf = batched_sdf.to(device)

x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
print(f"Input Coords Min: {x.coords[:, 1:].min(dim=0)[0].tolist()}")
print(f"Input Coords Max: {x.coords[:, 1:].max(dim=0)[0].tolist()}")
print(f"Input Coords Shape: {x.coords.shape}")

# 2. Load Model
model = SimpleSparseVAE(in_channels=8, hidden_channels=32, latent_channels=16, out_channels=8, num_layers=3).to(device)
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_reconstruction.pt")
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Model loaded successfully.")
else:
    print("Running with uninitialized model (no checkpoint found).")
model.eval()

# 3. Layer-by-Layer Forward Pass (Encoder)
print("\n=== ENCODER ===")
x._caches.cmaps.setdefault(x.stride, x.coords)

with torch.no_grad():
    h = model.stem(x)
    print(f"[Stem] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")

    for i, layer in enumerate(model.enc_layers):
        h = layer(h)
        print(f"[Enc Layer {i+1} (Downsample)] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")

    enc_out = model.enc_out(h)
    print(f"[Enc Out (Latent)] Coords Min: {enc_out.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {enc_out.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {enc_out.stride}")

# 4. Layer-by-Layer Forward Pass (Decoder)
print("\n=== DECODER ===")
with torch.no_grad():
    posterior = DiagonalGaussianDistribution(enc_out.feats, feat_dim=-1)
    z_feats = posterior.mean

    h = SparseTensor(coords=enc_out.coords, feats=z_feats, stride=enc_out.stride)
    h._caches = x._caches  # Crucial for decoder to find upsampling maps!

    h = model.dec_in(h)
    print(f"[Dec In] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")

    for i, layer in enumerate(model.dec_layers):
        h = layer(h)
        print(f"[Dec Layer {i+1} (Upsample)] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")

    out = model.dec_out(h)
    print(f"[Dec Out] Coords Min: {out.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {out.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {out.stride}")
