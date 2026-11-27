"""Tri-branch masked multi-head self-attention for ConDiffPlan (Section 3.4.3).

Combines three parallel attention branches in every layer:
  1. Global self-attention (all valid corners attend to all valid corners)
  2. Room-wise self-attention (corners attend only to corners within the same polygon)
  3. Relational self-attention (corners attend to connected rooms/openings/boundary
     with a learnable per-head bias indexed by typed relational edges)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from condiffplan.config import ModelConfig


def prepare_tri_branch_masks(
    padding_mask: torch.Tensor,
    room_mask: torch.Tensor,
    rel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute boolean SDPA attention masks once per forward pass across all layers.

    Returns:
        global_allow: [B, 1, M, M] bool (True = attend)
        room_allow:   [B, 1, M, M] bool (True = attend)
        rel_allow:    [B, 1, M, M] bool (True = attend)
    """
    bsz, seqlen = padding_mask.shape
    diag = torch.eye(seqlen, device=padding_mask.device, dtype=torch.bool).unsqueeze(0)  # [1, M, M]

    valid_keys = padding_mask.unsqueeze(1)  # [B, 1, M]
    global_allow = (valid_keys.expand(bsz, seqlen, seqlen) | diag).unsqueeze(1)
    room_allow = ((room_mask & valid_keys) | diag).unsqueeze(1)
    rel_allow = ((rel_mask & valid_keys) | diag).unsqueeze(1)
    return global_allow, room_allow, rel_allow


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

        # Fused Q, K, V projection for all 3 branches (Global + Room + Relational) in 1 GEMM
        self.qkv_all = nn.Linear(cfg.d_model, 9 * cfg.d_model)

        # Learnable typed-edge relative attention bias for the relational branch
        self.edge_bias = nn.Embedding(cfg.num_rel_edge_types, cfg.n_heads)
        nn.init.zeros_(self.edge_bias.weight)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Backward-compatible loader: pack legacy qkv_global/qkv_room/qkv_rel into qkv_all."""
        g_w = prefix + "qkv_global.weight"
        r_w = prefix + "qkv_room.weight"
        rel_w = prefix + "qkv_rel.weight"
        all_w = prefix + "qkv_all.weight"
        if all_w not in state_dict and g_w in state_dict and r_w in state_dict and rel_w in state_dict:
            state_dict[all_w] = torch.cat(
                [state_dict.pop(g_w), state_dict.pop(r_w), state_dict.pop(rel_w)], dim=0
            )
            g_b = prefix + "qkv_global.bias"
            r_b = prefix + "qkv_room.bias"
            rel_b = prefix + "qkv_rel.bias"
            if g_b in state_dict and r_b in state_dict and rel_b in state_dict:
                state_dict[prefix + "qkv_all.bias"] = torch.cat(
                    [state_dict.pop(g_b), state_dict.pop(r_b), state_dict.pop(rel_b)], dim=0
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        room_mask: torch.Tensor,
        rel_mask: torch.Tensor,
        rel_edge_type: torch.Tensor,
        precomputed_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute summed Global + Room-wise + Relational self-attention.

        Args:
            x: [B, M, d_model]
            padding_mask: [B, M] bool (True = valid token)
            room_mask: [B, M, M] bool (True = same element)
            rel_mask: [B, M, M] bool (True = connected in graph)
            rel_edge_type: [B, M, M] uint8 or long (relational edge type ID in 0..7)
            precomputed_masks: optional (global_allow, room_allow, rel_allow) boolean masks
        """
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        if precomputed_masks is not None:
            global_allow, room_allow, rel_allow = precomputed_masks
        else:
            global_allow, room_allow, rel_allow = prepare_tri_branch_masks(
                padding_mask=padding_mask,
                room_mask=room_mask,
                rel_mask=rel_mask,
            )

        # Relational attention bias with learnable typed-edge embedding: [B, H, M, M]
        edge_ids = rel_edge_type if rel_edge_type.dtype == torch.long else rel_edge_type.long()
        eb = self.edge_bias(edge_ids).permute(0, 3, 1, 2).to(dtype)
        rel_bias = eb.masked_fill(~rel_allow, float("-inf"))

        drop_p = self.dropout_p if self.training else 0.0

        # Single fused projection -> [9, B, H, M, D_h]
        qkv = self.qkv_all(x).view(bsz, seqlen, 9, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        qg, kg, vg, qr, kr, vr, qrel, krel, vrel = qkv.unbind(dim=0)

        out_global = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=global_allow, dropout_p=drop_p)
        out_room = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=room_allow, dropout_p=drop_p)
        out_rel = F.scaled_dot_product_attention(qrel, krel, vrel, attn_mask=rel_bias, dropout_p=drop_p)

        # Sum the results of the three attentions (Section 3.4.3)
        merged = (out_global + out_room + out_rel).transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        out = self.resid_dropout(self.out_proj(merged))
        # Mask out padded positions
        return out * padding_mask.unsqueeze(-1).to(dtype)
