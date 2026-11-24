"""Tri-branch masked multi-head self-attention for ConDiffPlan (Section 3.4.3).

Combines three parallel attention branches in every layer:
  1. Global self-attention (all valid corners attend to all valid corners)
  2. Room-wise self-attention (corners attend only to corners within the same polygon)
  3. Relational self-attention (corners attend to connected rooms/openings/boundary
     with a learnable per-head bias indexed by typed relational edges)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from condiffplan.config import ModelConfig


class TriBranchAttention(nn.Module):
    """Three-way multi-head self-attention with separate QKV matrices summed before out_proj."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout_p = cfg.dropout

        # Three separate sets of Q, K, V matrices per layer (Section 3.4.3)
        self.qkv_global = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.qkv_room = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.qkv_rel = nn.Linear(cfg.d_model, 3 * cfg.d_model)

        # Learnable typed-edge relative attention bias for the relational branch
        self.edge_bias = nn.Embedding(cfg.num_rel_edge_types, cfg.n_heads)
        nn.init.zeros_(self.edge_bias.weight)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _split_qkv(self, qkv_proj: nn.Linear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        qkv = qkv_proj(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, M, D_h]
        return qkv[0], qkv[1], qkv[2]

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        room_mask: torch.Tensor,
        rel_mask: torch.Tensor,
        rel_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute summed Global + Room-wise + Relational self-attention.

        Args:
            x: [B, M, d_model]
            padding_mask: [B, M] bool (True = valid token)
            room_mask: [B, M, M] bool (True = same element)
            rel_mask: [B, M, M] bool (True = connected in graph)
            rel_edge_type: [B, M, M] long (relational edge type ID in 0..7)
        """
        bsz, seqlen, _ = x.shape
        dtype = x.dtype
        neg_inf = torch.finfo(dtype).min

        # Valid key positions: [B, 1, 1, M]
        valid_keys = padding_mask.unsqueeze(1).unsqueeze(2)
        diag = torch.eye(seqlen, device=x.device, dtype=torch.bool).unsqueeze(0)  # [1, M, M]

        # 1. Global attention bias: [B, 1, M, M]
        global_allow = valid_keys.expand(bsz, 1, seqlen, seqlen) | diag.unsqueeze(1)
        global_bias = torch.where(global_allow, torch.zeros((), device=x.device, dtype=dtype), neg_inf)

        # 2. Room-wise attention bias: [B, 1, M, M]
        room_allow = (room_mask & padding_mask.unsqueeze(1)) | diag
        room_bias = torch.where(
            room_allow.unsqueeze(1),
            torch.zeros((), device=x.device, dtype=dtype),
            neg_inf,
        )

        # 3. Relational attention bias with typed-edge embedding: [B, H, M, M]
        rel_allow = (rel_mask & padding_mask.unsqueeze(1)) | diag
        eb = self.edge_bias(rel_edge_type).permute(0, 3, 1, 2).to(dtype)  # [B, H, M, M]
        rel_bias = torch.where(rel_allow.unsqueeze(1), eb, neg_inf)

        drop_p = self.dropout_p if self.training else 0.0

        qg, kg, vg = self._split_qkv(self.qkv_global, x)
        out_global = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=global_bias, dropout_p=drop_p)

        qr, kr, vr = self._split_qkv(self.qkv_room, x)
        out_room = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=room_bias, dropout_p=drop_p)

        qrel, krel, vrel = self._split_qkv(self.qkv_rel, x)
        out_rel = F.scaled_dot_product_attention(qrel, krel, vrel, attn_mask=rel_bias, dropout_p=drop_p)

        # Sum the results of the three attentions (Section 3.4.3)
        merged = (out_global + out_room + out_rel).transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        out = self.resid_dropout(self.out_proj(merged))
        # Mask out padded positions
        return out * padding_mask.unsqueeze(-1).to(dtype)
