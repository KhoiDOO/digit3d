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


class CoordinateFourierEmbedder(nn.Module):
    """
    Multi-frequency sinusoidal Fourier positional embedding for continuous 3D spatial coordinates [B, P, 3].
    Uses bounded frequency bands to ensure smooth Jacobian Vector Products (JVP) in MeanFlow.
    """
    def __init__(self, in_dim: int = 3, embed_dim: int = 256, num_freqs: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs

        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

        raw_dim = in_dim + in_dim * num_freqs * 2
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        :param coords: [..., 3] continuous positions in [-1.5, 1.5]^3
        :return: [..., embed_dim]
        """
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi  # [..., 3, num_freqs]
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        fourier = torch.cat([coords, sin.flatten(start_dim=-2), cos.flatten(start_dim=-2)], dim=-1)
        return self.proj(fourier)


class ImagePatchEncoder(nn.Module):
    """
    High-capacity deep convolutional encoder for 2D image conditioning [B, 1, 28, 28].
    Extracts:
    1. Spatial Context Tokens [B, K, D] (where K = 7x7 = 49 tokens) for Multi-Head Cross-Attention.
    2. Global Conditioning Vector [B, D] for AdaLN modulation.
    Supports Classifier-Free Guidance (CFG) random dropout during training.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, cond_drop_prob: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.cond_drop_prob = cond_drop_prob

        # Deep 5-stage convolutional backbone -> 7x7 spatial feature map
        self.stem = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            # Stage 2: 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),

            # Stage 3: Feature projection to embed_dim (7x7)
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, embed_dim),
            nn.GELU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Learnable null tokens for unconditional / CFG sampling
        self.num_spatial_tokens = 7 * 7  # 49
        self.null_spatial_tokens = nn.Parameter(torch.zeros(1, self.num_spatial_tokens, embed_dim))
        self.null_global_embed = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.null_spatial_tokens, std=0.02)
        nn.init.normal_(self.null_global_embed, std=0.02)

    def forward(
        self,
        img: Optional[torch.Tensor],
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param img: Optional [B, 1, 28, 28] or [B, 3, 28, 28] 2D conditioning images
        :param batch_size: Required if img is None
        :return: (spatial_tokens [B, 49, D], global_embed [B, D])
        """
        if img is not None:
            if img.shape[1] == 3 and self.in_channels == 1:
                img = img.mean(dim=1, keepdim=True)
            feat = self.stem(img)  # [B, D, 7, 7]
            B, D, H, W = feat.shape
            spatial_tokens = feat.flatten(2).permute(0, 2, 1)  # [B, 49, D]
            global_embed = self.global_proj(self.global_pool(feat).flatten(1))  # [B, D]
        else:
            assert batch_size is not None, "Must provide batch_size when img is None"
            spatial_tokens = self.null_spatial_tokens.expand(batch_size, -1, -1)
            global_embed = self.null_global_embed.expand(batch_size, -1)

        # Classifier-Free Guidance dropout during training
        if self.training and self.cond_drop_prob > 0.0:
            mask = torch.rand(len(spatial_tokens), device=spatial_tokens.device) < self.cond_drop_prob
            if mask.any():
                null_sp_exp = self.null_spatial_tokens.expand(len(spatial_tokens), -1, -1)
                null_gl_exp = self.null_global_embed.expand(len(spatial_tokens), -1)
                spatial_tokens = torch.where(mask[:, None, None], null_sp_exp, spatial_tokens)
                global_embed = torch.where(mask[:, None], null_gl_exp, global_embed)

        return spatial_tokens, global_embed


class AdaLNCrossAttnBlock(nn.Module):
    """
    Modulated Transformer Block with Adaptive LayerNorm (AdaLN) and Independent Cross-Attention.
    Points are processed point-wise independently (no self-attention across points, O(P) scaling).
    Every block is modulated by the combined condition vector c (time + global image context).
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Cross-Attention projections (Q from query points, K and V from spatial context tokens)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN modulation generates 6 modulation parameters from condition c
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x: torch.Tensor, context: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, P, D] Point queries feature tensor (arbitrary point count P)
        :param context: [B, K, D] Spatial image tokens (K tokens)
        :param c: [B, D] Global condition embedding (time + image)
        :return: [B, P, D] Updated point feature tensor
        """
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # 1. Modulated Cross-Attention
        norm_x = self.norm1(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
        B, P, D = norm_x.shape
        _, K, _ = context.shape

        q = self.q_proj(norm_x).reshape(B, P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = attn @ v

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, P, D)
        x = x + gate_ca.unsqueeze(1) * self.out_proj(attn_out)

        # 2. Modulated Point-Wise MLP
        norm_x2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)
        return x


class ArbitraryPointFlowTransformer(nn.Module):
    """
    Arbitrary-Resolution 3D Point Cloud & Surface Normal Generation Model.
    Ingests continuous 6D points (3D positions + 3D normals) and predicts 6D flow velocity.
    Supports arbitrary point count P in a single batch (O(P) linear complexity).
    """
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 6,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        img_channels: int = 1,
        cond_drop_prob: float = 0.15,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depth = depth

        # High-capacity 2D Image Conditioning
        self.img_encoder = ImagePatchEncoder(
            in_channels=img_channels,
            embed_dim=embed_dim,
            cond_drop_prob=cond_drop_prob
        )

        # Timestep Embedders (t for Rectified Flow / MeanFlow, and s / dt for MeanFlow)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.s_embedder = TimestepEmbedder(embed_dim)
        self.cond_proj = nn.Linear(embed_dim, embed_dim)

        # 3D Coordinate Fourier Positional Embedder + Feature Projector
        self.fourier_embedder = CoordinateFourierEmbedder(in_dim=3, embed_dim=embed_dim, num_freqs=num_freqs)
        self.point_proj = nn.Linear(in_channels, embed_dim)

        # Stack of AdaLN-Modulated Cross-Attention Blocks
        self.blocks = nn.ModuleList([
            AdaLNCrossAttnBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer (AdaLN modulation + Linear projection)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        self.final_proj = nn.Linear(embed_dim, out_channels, bias=True)

        self._init_final_layer()

    def _init_final_layer(self):
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        :param x: [B, P, 6] or [B, 6, P] Tensor of 6D points (positions + normals)
        :param t: [B] Float tensor of current timesteps in [0, 1]
        :param s: Optional [B] Float tensor of previous timesteps / delta times (MeanFlow)
        :param cond: Optional [B, 1, 28, 28] 2D conditioning images
        :return: [B, P, 6] or [B, 6, P] Predicted 6D flow velocity
        """
        transposed = False
        if x.shape[1] == self.in_channels and x.shape[2] != self.in_channels:
            x = x.permute(0, 2, 1)  # Convert [B, 6, P] -> [B, P, 6]
            transposed = True

        B, P, _ = x.shape

        if t is None and 't' in kwargs:
            t = kwargs['t']
        if t is None:
            raise ValueError("Timestep tensor 't' must be provided.")
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B)
        elif len(t) == 1 and B > 1:
            t = t.expand(B)

        # 1. Encode 2D Image Conditioning
        spatial_context, global_embed = self.img_encoder(cond, batch_size=B)  # [B, 49, D], [B, D]

        # 2. Compute Combined AdaLN Condition Vector c
        c = self.t_embedder(t) + self.cond_proj(global_embed)
        if s is not None:
            if s.ndim == 0:
                s = s.unsqueeze(0).expand(B)
            elif len(s) == 1 and B > 1:
                s = s.expand(B)
            c = c + self.s_embedder(s)
        elif 's' in kwargs and kwargs['s'] is not None:
            kw_s = kwargs['s']
            if kw_s.ndim == 0:
                kw_s = kw_s.unsqueeze(0).expand(B)
            c = c + self.s_embedder(kw_s)

        # 3. Input Embedding: Linear projection + 3D Coordinate Fourier Embedding
        coords = x[..., :3]
        h = self.point_proj(x) + self.fourier_embedder(coords)  # [B, P, D]

        # 4. Pass through AdaLN Cross-Attention Transformer Blocks
        for block in self.blocks:
            h = block(h, context=spatial_context, c=c)

        # 5. Final Output Layer
        shift_final, scale_final = self.final_adaLN(c).chunk(2, dim=-1)
        norm_h = self.final_norm(h) * (1 + scale_final.unsqueeze(1)) + shift_final.unsqueeze(1)
        v = self.final_proj(norm_h)  # [B, P, 6]

        if transposed:
            v = v.permute(0, 2, 1)

        return v
