"""Per-corner token feature embedding (with vectorized wall augmentation) and timestep embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from condiffplan.config import ModelConfig


def coords_to_bits(coords: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Quantize normalized 2D coordinates in [-1, 1] to integer [0, 2^B - 1] and convert via int2bit.

    Implements HouseDiffusion Section 4.1-4.2 (Analog Bits binary representation).
    Args:
        coords: [..., 2] continuous coordinates in [-1, 1]
        num_bits: B (default 8 -> [0, 255], 16 bits per 2D corner)
    Returns:
        bits: [..., 2 * num_bits] float tensor with values in {0.0, 1.0}
    """
    max_val = (1 << num_bits) - 1
    scaled = (coords.float().clamp(-1.0, 1.0) + 1.0) * (0.5 * float(max_val))
    x_int = scaled.round().to(torch.long).clamp(0, max_val)
    shifts = torch.arange(num_bits - 1, -1, -1, device=coords.device, dtype=torch.long)
    bits = ((x_int.unsqueeze(-1) >> shifts) & 1).to(dtype=coords.dtype)
    return bits.flatten(start_dim=-2)


def bits_to_coords(bits: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Binary-threshold predicted bit vectors (>= 0.5) and convert via bit2int back to [-1, 1].

    Args:
        bits: [..., 2 * num_bits] continuous or binary bit predictions
        num_bits: B (default 8)
    Returns:
        coords: [..., 2] quantized coordinates in [-1, 1]
    """
    max_val = (1 << num_bits) - 1
    bits_2d = bits.unflatten(-1, (2, num_bits))
    hard_bits = (bits_2d >= 0.5).to(torch.long)
    weights = torch.tensor(
        [1 << (num_bits - 1 - i) for i in range(num_bits)],
        device=bits.device,
        dtype=torch.long,
    )
    x_int = (hard_bits * weights).sum(dim=-1)
    return (x_int.to(dtype=bits.dtype) / float(max_val)) * 2.0 - 1.0


def wall_augment(
    coords: torch.Tensor,
    next_corner_idx: torch.Tensor,
    num_points: int = 8,
) -> torch.Tensor:
    """Uniformly sample L points along the wall segment from corner (i, j) to corner (i, j+1).

     Implements AU(C^t_{i,j}) in Eq. (8) of ConDiffPlan:
      For l = 0 .. L-1:  p_l = (1 - l/L) * C_{i,j} + (l/L) * C_{i, j+1}
    Args:
        coords: [B, M, 2] corner coordinates at timestep t
        next_corner_idx: [B, M] index of the next corner in the same polygon
        num_points: L (default 8 -> 16D feature vector per corner)
    Returns:
        aug_coords: [B, M, 2 * num_points]
    """
    idx_expanded = next_corner_idx.unsqueeze(-1).expand(-1, -1, 2)
    next_coords = torch.gather(coords, dim=1, index=idx_expanded)  # [B, M, 2]

    alphas = torch.linspace(
        0.0,
        1.0 - 1.0 / max(num_points, 1),
        steps=num_points,
        device=coords.device,
        dtype=coords.dtype,
    ).view(1, 1, num_points, 1)

    c_curr = coords.unsqueeze(2)       # [B, M, 1, 2]
    c_next = next_coords.unsqueeze(2)  # [B, M, 1, 2]
    interp = (1.0 - alphas) * c_curr + alphas * c_next  # [B, M, L, 2]
    return interp.flatten(start_dim=2)  # [B, M, 2*L]


class CornerFeatureEmbedding(nn.Module):
    """Constructs per-corner feature vector P^t_{i,j} (Eq. 8) and projects to d_model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.token_input_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        coords_t: torch.Tensor,
        elem_type: torch.Tensor,
        elem_idx: torch.Tensor,
        corner_idx: torch.Tensor,
        next_corner_idx: torch.Tensor,
        fixed_mask: torch.Tensor,
        norm_area: torch.Tensor,
        has_area: torch.Tensor,
    ) -> torch.Tensor:
        au = wall_augment(coords_t, next_corner_idx, num_points=self.cfg.wall_aug_points)

        type_oh = F.one_hot(
            elem_type.clamp(0, self.cfg.num_elem_types - 1),
            num_classes=self.cfg.num_elem_types,
        ).to(coords_t.dtype)

        eidx_oh = F.one_hot(
            elem_idx.clamp(0, self.cfg.max_elements - 1),
            num_classes=self.cfg.max_elements,
        ).to(coords_t.dtype)

        cidx_oh = F.one_hot(
            corner_idx.clamp(0, self.cfg.max_corners - 1),
            num_classes=self.cfg.max_corners,
        ).to(coords_t.dtype)

        f_flag = fixed_mask.to(coords_t.dtype).unsqueeze(-1)
        a_val = norm_area.to(coords_t.dtype).unsqueeze(-1)
        a_flag = has_area.to(coords_t.dtype).unsqueeze(-1)

        raw_feat = torch.cat([au, type_oh, eidx_oh, cidx_oh, f_flag, a_val, a_flag], dim=-1)
        return self.norm(self.proj(raw_feat))


class DiscreteCornerFeatureEmbedding(nn.Module):
    """Constructs HouseDiffusion Eq. (4) discrete corner embedding: [AU(C^0), int2bit(C^0), conds]."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.discrete_token_input_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        coords_0: torch.Tensor,
        elem_type: torch.Tensor,
        elem_idx: torch.Tensor,
        corner_idx: torch.Tensor,
        next_corner_idx: torch.Tensor,
        fixed_mask: torch.Tensor,
        norm_area: torch.Tensor,
        has_area: torch.Tensor,
    ) -> torch.Tensor:
        au = wall_augment(coords_0, next_corner_idx, num_points=self.cfg.wall_aug_points)
        bits = coords_to_bits(coords_0, num_bits=self.cfg.discrete_num_bits).to(coords_0.dtype)

        type_oh = F.one_hot(
            elem_type.clamp(0, self.cfg.num_elem_types - 1),
            num_classes=self.cfg.num_elem_types,
        ).to(coords_0.dtype)

        eidx_oh = F.one_hot(
            elem_idx.clamp(0, self.cfg.max_elements - 1),
            num_classes=self.cfg.max_elements,
        ).to(coords_0.dtype)

        cidx_oh = F.one_hot(
            corner_idx.clamp(0, self.cfg.max_corners - 1),
            num_classes=self.cfg.max_corners,
        ).to(coords_0.dtype)

        f_flag = fixed_mask.to(coords_0.dtype).unsqueeze(-1)
        a_val = norm_area.to(coords_0.dtype).unsqueeze(-1)
        a_flag = has_area.to(coords_0.dtype).unsqueeze(-1)

        raw_feat = torch.cat([au, bits, type_oh, eidx_oh, cidx_oh, f_flag, a_val, a_flag], dim=-1)
        return self.norm(self.proj(raw_feat))


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard DDPM sinusoidal timestep embedding followed by a 2-layer SiLU MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * (-emb_scale)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)
