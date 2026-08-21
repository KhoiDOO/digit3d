import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.sparse_voxel.generation.stage1.models import ImageEncoder, TimestepEmbedder, timestep_embedding


class CoordinateFourierEmbedder(nn.Module):
    """
    Sinusoidal Fourier positional embedding for continuous 3D vertex coordinates [..., 3].
    """
    def __init__(self, in_dim: int = 3, embed_dim: int = 256, num_freqs: int = 32):
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
        :param coords: [..., 3] continuous positions in [-1.2, 1.2]^3 (or [-1, 1])
        :return: [..., embed_dim]
        """
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi # [..., 3, num_freqs]
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        fourier = torch.cat([coords, sin.flatten(start_dim=-2), cos.flatten(start_dim=-2)], dim=-1)
        return self.proj(fourier)


class ModulatedVarLenVertexBlock(nn.Module):
    """
    Modulated Transformer Block for Stacked 1D Sparse Vertex Sequences (Zero Padding, O(B) Memory).
    Linear projections run across all T_total tokens in a unified GPU GEMM, while self-attention
    splits by seq_lens to eliminate quadratic cross-sample memory buffers.
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

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        seq_lens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        :param x: [T_total, D] or [1, T_total, D] stacked active vertex features
        :param c: [T_total, D] or [1, T_total, D] token-aligned conditioning embedding
        :param seq_lens: Optional[List[int]] sequence lengths per batch sample [M_1, M_2, ..., M_B]
        """
        is_3d = (x.ndim == 3)
        if is_3d:
            x = x.squeeze(0)
            c = c.squeeze(0)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # 1. Modulated QKV Projection (Single Unified Fast GPU GEMM across all T_total tokens)
        norm_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        T_total, D = norm_x.shape
        qkv = self.qkv(norm_x).reshape(T_total, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] # each [T_total, H, head_dim]

        # 2. Per-Sample Self-Attention (No Padding, O(B) Linear Memory Scaling)
        if seq_lens is not None and len(seq_lens) > 1:
            q_splits = q.split(seq_lens, dim=0)
            k_splits = k.split(seq_lens, dim=0)
            v_splits = v.split(seq_lens, dim=0)

            out_splits = []
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                for q_b, k_b, v_b in zip(q_splits, k_splits, v_splits):
                    # [M_b, H, head_dim] -> [1, H, M_b, head_dim]
                    q_b = q_b.permute(1, 0, 2).unsqueeze(0)
                    k_b = k_b.permute(1, 0, 2).unsqueeze(0)
                    v_b = v_b.permute(1, 0, 2).unsqueeze(0)

                    out_b = F.scaled_dot_product_attention(q_b, k_b, v_b)
                    out_splits.append(out_b.squeeze(0).permute(1, 0, 2).reshape(-1, D))

            attn_out = torch.cat(out_splits, dim=0)
        else:
            # Single sample (B=1 or unbatched inference)
            q_p = q.permute(1, 0, 2).unsqueeze(0)
            k_p = k.permute(1, 0, 2).unsqueeze(0)
            v_p = v.permute(1, 0, 2).unsqueeze(0)
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                attn_out = F.scaled_dot_product_attention(q_p, k_p, v_p)
            attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(T_total, D)

        x = x + gate_msa * self.proj(attn_out)

        # 3. Modulated MLP (Single Unified Fast GPU GEMM across all T_total tokens)
        norm_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(norm_x)

        if is_3d:
            x = x.unsqueeze(0)
        return x


class SparseVertexSDFTransformer(nn.Module):
    """
    Modulated Rectified Flow Transformer operating over Stacked 1D active grid vertices.
    Uses Native PyTorch VarLen Split Attention for 100% zero padding and O(B) memory scaling.
    """
    def __init__(
        self,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim

        # 2D Image Conditioner
        self.img_encoder = ImageEncoder(in_channels=img_channels, embed_dim=cond_dim)

        # 3D Vertex Coordinate Embedder
        self.coord_embedder = CoordinateFourierEmbedder(in_dim=3, embed_dim=embed_dim)

        # Timestep Embedder for Rectified Flow
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

        # Input projection (scalar SDF feature -> embed_dim)
        self.feat_embed = nn.Linear(in_channels, embed_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            ModulatedVarLenVertexBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer (AdaLN modulation + Linear projection to scalar velocity)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        self.final_proj = nn.Linear(embed_dim, out_channels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
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

    def set_seq_lens(self, seq_lens: Optional[Union[List[int], torch.Tensor]] = None):
        if seq_lens is None:
            self._seq_lens = None
        elif isinstance(seq_lens, torch.Tensor):
            self._seq_lens = seq_lens.tolist()
        else:
            self._seq_lens = list(seq_lens)

    def forward(
        self,
        s: torch.Tensor,
        times: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        seq_lens: Optional[Union[List[int], torch.Tensor]] = None,
        coords: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        :param s: [T_total, 1] or [1, T_total, 1] noisy scalar SDF state
        :param times: [1] or [B] timesteps t in [0, 1]
        :param cond: [T_total, 3 + cond_dim] unified condition tensor (or coords / image)
        :param seq_lens: Optional list or tensor of active vertex counts per sample [M_1, M_2, ..., M_B]
        :return: [T_total, 1] or [1, T_total, 1] predicted scalar velocity field
        """
        is_3d = (s.ndim == 3)
        if is_3d:
            s = s.squeeze(0)

        # Resolve seq_lens from argument, attribute, or fallback
        if seq_lens is None:
            if hasattr(self, '_seq_lens') and self._seq_lens is not None:
                seq_lens_list = self._seq_lens
            else:
                seq_lens_list = [s.shape[0]]
        elif isinstance(seq_lens, torch.Tensor):
            seq_lens_list = seq_lens.tolist()
        else:
            seq_lens_list = list(seq_lens)

        seq_lens_tensor = torch.tensor(seq_lens_list, dtype=torch.long, device=s.device)
        batch_size = len(seq_lens_list)

        # Parse condition tensor
        if cond is not None and cond.shape[-1] >= 3 + self.cond_dim:
            if cond.ndim == 3:
                cond = cond.squeeze(0)
            coords = cond[:, :3]
            c_img_tokens = cond[:, 3:3+self.cond_dim]
        elif cond is not None and cond.shape[-1] == 3:
            if cond.ndim == 3:
                cond = cond.squeeze(0)
            coords = cond
            c_img_tokens = None
        else:
            if coords is None and 'coords' in kwargs:
                coords = kwargs['coords']
                if coords.ndim == 3:
                    coords = coords.squeeze(0)
            c_img_tokens = None

        # 1. Conditioning vector = Timestep Embedding + Condition Projection
        if times.ndim == 0 or times.numel() == 1:
            t_emb = self.t_embedder(times) # [1, embed_dim]
            t_emb_stacked = t_emb.expand(s.shape[0], -1)
        else:
            t_emb = self.t_embedder(times) # [B, embed_dim]
            t_emb_stacked = torch.repeat_interleave(t_emb, seq_lens_tensor, dim=0)

        if c_img_tokens is not None:
            c_img_proj = self.cond_proj(c_img_tokens) # [T_total, embed_dim]
            c = t_emb_stacked + c_img_proj
        elif img is not None:
            c_img = self.img_encoder(img)
            c_img_proj = self.cond_proj(c_img)
            if batch_size > 1 and len(c_img_proj) == batch_size:
                c_img_stacked = torch.repeat_interleave(c_img_proj, seq_lens_tensor, dim=0)
            else:
                c_img_stacked = c_img_proj.expand(s.shape[0], -1)
            c = t_emb_stacked + c_img_stacked
        else:
            c_img = self.img_encoder(None, batch_size=batch_size)
            c_img_proj = self.cond_proj(c_img)
            c_img_stacked = torch.repeat_interleave(c_img_proj, seq_lens_tensor, dim=0)
            c = t_emb_stacked + c_img_stacked

        # 2. Token embedding = Feature embedding (SDF state) + 3D Coordinate Fourier Embedding
        h = self.feat_embed(s) + self.coord_embedder(coords) # [T_total, embed_dim]

        # 3. Transformer Blocks with VarLen Split Attention
        for block in self.blocks:
            h = block(h, c, seq_lens=seq_lens_list)

        # 4. Final Modulation and Projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        out = self.final_proj(h)

        if is_3d:
            out = out.unsqueeze(0)
        return out
