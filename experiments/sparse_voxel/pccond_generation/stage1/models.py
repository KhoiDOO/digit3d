import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PointTransformerBlock(nn.Module):
    """
    Standard Self-Attention Transformer block for Point Cloud tokens.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PointTransformerEncoder(nn.Module):
    """
    Encodes 3D point clouds with surface normals [B, 6, N] or [B, N, 6] into a continuous
    conditioning vector [B, embed_dim] via direct 6D feature projection, self-attention transformer blocks,
    and permutation-invariant pooling. Supports classifier-free guidance conditional dropout.
    """
    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        cond_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.cond_drop_prob = cond_drop_prob

        # Direct 6D feature projection (XYZ + Normals -> embed_dim)
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PointTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable null token for CFG dropout
        self.null_embed = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.null_embed, std=0.02)

    def forward(self, pc: Optional[torch.Tensor], batch_size: Optional[int] = None) -> torch.Tensor:
        """
        :param pc: [B, 6, N] or [B, N, 6] float point cloud tensor (or None for unconditional CFG)
        :param batch_size: Required if pc is None
        :return: [B, embed_dim] continuous conditioning embedding
        """
        if pc is not None:
            if pc.shape[1] == self.in_channels and pc.shape[2] != self.in_channels:
                pc = pc.permute(0, 2, 1)  # [B, N, C]
            
            # 1. Project 6D coordinates and surface normals to feature dimension
            h = self.input_proj(pc)  # [B, N, D]

            # 2. Self-Attention Transformer blocks
            for block in self.blocks:
                h = block(h)

            h = self.norm(h)

            # 3. Permutation-invariant mean pooling + MLP projection
            pooled = h.mean(dim=1)  # [B, D]
            emb = self.out_proj(pooled)
        else:
            assert batch_size is not None, "Must provide batch_size when pc is None"
            emb = self.null_embed.expand(batch_size, -1)

        # CFG conditional dropout during training
        if self.training and self.cond_drop_prob > 0.0:
            mask = torch.rand(len(emb), device=emb.device) < self.cond_drop_prob
            if mask.any():
                null_expanded = self.null_embed.expand(len(emb), -1)
                emb = torch.where(mask[:, None], null_expanded, emb)

        return emb


class DiTBlock3D(nn.Module):
    """
    3D Modulated Transformer Block with Adaptive LayerNorm (AdaLN) conditioning.
    Uses forward-mode AD compatible attention for MeanFlow JVP.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, L, D]
        :param c: [B, D] combined condition + timestep embedding
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # 1. Modulated Self-Attention (pure PyTorch matrix multiply for JVP forward-mode AD support)
        norm_x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        B, L, D = norm_x.shape
        qkv = self.qkv(norm_x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = attn @ v

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
        x = x + gate_msa.unsqueeze(1) * self.proj(attn_out)

        # 2. Modulated MLP
        norm_x = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)
        return x


class StructureDiT(nn.Module):
    """
    Patchified 3D Diffusion/MeanFlow Transformer for Dense Voxel Occupancy Flow Matching
    conditioned on 3D Point Clouds.
    Resolution: 32x32x32.
    Patch Size: 4x4x4 -> (32/4)^3 = 8^3 = 512 tokens.
    """
    def __init__(
        self,
        grid_res: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        cond_dim: int = 256,
        pc_channels: int = 6,
    ):
        super().__init__()
        self.grid_res = grid_res
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.tokens_per_dim = grid_res // patch_size
        self.num_tokens = self.tokens_per_dim ** 3

        # 3D Point Cloud Conditioner
        self.pc_encoder = PointTransformerEncoder(
            in_channels=pc_channels,
            embed_dim=cond_dim,
            depth=4,
            num_heads=num_heads
        )

        # Timestep & Delta-Timestep Embedders (for MeanFlow)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.dt_embedder = TimestepEmbedder(embed_dim)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

        # Patch Embed & Unembed
        patch_dim = in_channels * (patch_size ** 3)
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock3D(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer (AdaLN modulation + Linear projection)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        out_patch_dim = out_channels * (patch_size ** 3)
        self.final_proj = nn.Linear(embed_dim, out_patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, R, R, R] where R = 32
        :return: [B, L, patch_dim] where L = 512, patch_dim = C * 4^3 = 64
        """
        B, C, D, H, W = x.shape
        p = self.patch_size
        k = self.tokens_per_dim
        x = x.view(B, C, k, p, k, p, k, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return x.view(B, k * k * k, p * p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, L, patch_dim]
        :return: [B, C, R, R, R]
        """
        B, L, _ = x.shape
        p = self.patch_size
        k = self.tokens_per_dim
        C = self.out_channels
        x = x.view(B, k, k, k, p, p, p, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(B, C, k * p, k * p, k * p)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        delta_times: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        pc: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        MeanFlow and Flow Matching forward pass.
        :param x: [B, 1, 32, 32, 32] noisy dense occupancy grid
        :param times: [B] timesteps t
        :param delta_times: [B] delta timesteps (for MeanFlow)
        :param cond: [B, cond_dim] precomputed condition (or raw point cloud [B, 6, 512])
        :param pc: Optional raw point cloud tensor [B, 6, 512]
        """
        if delta_times is None:
            delta_times = torch.zeros_like(times)

        if cond is None:
            if pc is not None:
                c_emb = self.pc_encoder(pc)
            else:
                c_emb = self.pc_encoder(None, batch_size=len(x))
        elif cond.ndim == 3:
            # Raw point cloud passed through cond kwarg [B, 6, 512] or [B, 512, 6]
            c_emb = self.pc_encoder(cond)
        else:
            c_emb = cond

        # 1. Combined Conditioning vector: t + delta_t + point_cloud_condition
        t_emb = self.t_embedder(times)
        dt_emb = self.dt_embedder(delta_times)
        c_proj = self.cond_proj(c_emb)
        c = t_emb + dt_emb + c_proj

        # 2. Patchify & add 3D Positional Embedding
        h = self.patch_embed(self.patchify(x)) + self.pos_embed

        # 3. Transformer Blocks
        for block in self.blocks:
            h = block(h, c)

        # 4. Final AdaLN Modulation & Linear Projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.final_proj(h)

        # 5. Unpatchify back to 3D grid
        return self.unpatchify(out)
