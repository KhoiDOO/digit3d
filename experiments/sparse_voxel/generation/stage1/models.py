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


class ImageEncoder(nn.Module):
    """
    Encodes 2D MNIST/Digit images [B, 1, 28, 28] into continuous conditioning vectors [B, embed_dim].
    Supports classifier-free guidance conditional dropout.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, cond_drop_prob: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.cond_drop_prob = cond_drop_prob

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )
        self.null_embed = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.null_embed, std=0.02)

    def forward(self, img: Optional[torch.Tensor], batch_size: Optional[int] = None) -> torch.Tensor:
        if img is not None:
            if img.shape[1] == 3 and self.in_channels == 1:
                img = img.mean(dim=1, keepdim=True)
            emb = self.net(img)
        else:
            assert batch_size is not None, "Must provide batch_size when img is None"
            emb = self.null_embed.expand(batch_size, -1)

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
    Patchified 3D Diffusion/MeanFlow Transformer for Dense Voxel Occupancy Flow Matching.
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
        img_channels: int = 1,
    ):
        super().__init__()
        self.grid_res = grid_res
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.tokens_per_dim = grid_res // patch_size
        self.num_tokens = self.tokens_per_dim ** 3

        # 2D Image Conditioner
        self.img_encoder = ImageEncoder(in_channels=img_channels, embed_dim=cond_dim)

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
        :return: [B, L, patch_dim] where L = 4096, patch_dim = C * 2^3
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
        img: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        MeanFlow and Flow Matching forward pass.
        :param x: [B, 1, 32, 32, 32] noisy dense occupancy grid
        :param times: [B] timesteps t
        :param delta_times: [B] delta timesteps (for MeanFlow)
        :param cond: [B, cond_dim] precomputed condition (or raw image [B, 1, 28, 28])
        """
        if delta_times is None:
            delta_times = torch.zeros_like(times)

        if cond is None:
            if img is not None:
                c_emb = self.img_encoder(img)
            else:
                c_emb = self.img_encoder(None, batch_size=len(x))
        elif cond.ndim == 4:
            c_emb = self.img_encoder(cond)
        else:
            c_emb = cond

        # 1. Combined Conditioning vector: t + delta_t + image_condition
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
