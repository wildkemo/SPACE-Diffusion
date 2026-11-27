"""ConDiffPlan Transformer denoiser backbone with AdaLN-Zero and HouseDiffusion discrete branch."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from condiffplan.config import ModelConfig
from condiffplan.data.dataset import materialize_attention_masks_gpu
from condiffplan.model.attention import TriBranchAttention, prepare_tri_branch_masks
from condiffplan.model.embeddings import (
    CornerFeatureEmbedding,
    DiscreteCornerFeatureEmbedding,
    SinusoidalTimestepEmbedding,
)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ConDiffPlanBlock(nn.Module):
    """Transformer block combining TriBranchAttention, FFN, and AdaLN-Zero timestep conditioning."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = TriBranchAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 6 * cfg.d_model),
        )
        # Initialize modulation projection to zero so block acts as identity at start of training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        padding_mask: torch.Tensor,
        room_mask: torch.Tensor,
        rel_mask: torch.Tensor,
        rel_edge_type: torch.Tensor,
        precomputed_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        )
        h_attn = self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa),
            padding_mask=padding_mask,
            room_mask=room_mask,
            rel_mask=rel_mask,
            rel_edge_type=rel_edge_type,
            precomputed_masks=precomputed_masks,
        )
        x = x + gate_msa.unsqueeze(1) * h_attn
        h_ffn = self.ffn(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h_ffn
        return x * padding_mask.unsqueeze(-1).to(x.dtype)


class ConDiffPlanTransformer(nn.Module):
    """Full ConDiffPlan vector conditional diffusion Transformer with optional discrete branch."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable_discrete = bool(cfg.enable_discrete)
        self.corner_embed = CornerFeatureEmbedding(cfg)
        self.time_embed = SinusoidalTimestepEmbedding(cfg.d_model)
        self.blocks = nn.ModuleList([ConDiffPlanBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.noise_head = nn.Linear(cfg.d_model, 2)

        # HouseDiffusion Discrete Denoising Branch (Section 4.2, Eq. 4)
        if self.enable_discrete:
            self.discrete_embed = DiscreteCornerFeatureEmbedding(cfg)
            self.discrete_blocks = nn.ModuleList(
                [ConDiffPlanBlock(cfg) for _ in range(cfg.discrete_n_layers)]
            )
            self.discrete_final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
            self.discrete_final_adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cfg.d_model, 2 * cfg.d_model),
            )
            nn.init.zeros_(self.discrete_final_adaLN[1].weight)
            nn.init.zeros_(self.discrete_final_adaLN[1].bias)
            self.discrete_head = nn.Linear(cfg.d_model, cfg.discrete_out_dim)
        else:
            self.discrete_embed = None
            self.discrete_blocks = None
            self.discrete_final_norm = None
            self.discrete_final_adaLN = None
            self.discrete_head = None

    def _run_block(
        self,
        blk: ConDiffPlanBlock,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        padding_mask: torch.Tensor,
        room_mask: torch.Tensor,
        rel_mask: torch.Tensor,
        rel_edge_type: torch.Tensor,
        precomputed_masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.training and self.cfg.grad_checkpointing:
            return grad_checkpoint(
                blk,
                x,
                t_emb,
                padding_mask,
                room_mask,
                rel_mask,
                rel_edge_type,
                precomputed_masks,
                use_reentrant=False,
            )
        return blk(
            x=x,
            t_emb=t_emb,
            padding_mask=padding_mask,
            room_mask=room_mask,
            rel_mask=rel_mask,
            rel_edge_type=rel_edge_type,
            precomputed_masks=precomputed_masks,
        )

    def forward(
        self,
        coords_t: torch.Tensor,
        timesteps: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Continuous denoising branch: predict noise epsilon_theta(P^t_{i,j}, t) of shape [B, M, 2]."""
        padding_mask = batch["padding_mask"]
        room_mask, rel_mask, rel_edge_type = materialize_attention_masks_gpu(batch)
        precomputed_masks = prepare_tri_branch_masks(padding_mask, room_mask, rel_mask)

        x = self.corner_embed(
            coords_t=coords_t,
            elem_type=batch["elem_type"],
            elem_idx=batch["elem_idx"],
            corner_idx=batch["corner_idx"],
            next_corner_idx=batch["next_corner_idx"],
            fixed_mask=batch["fixed_mask"],
            norm_area=batch["norm_area"],
            has_area=batch["has_area"],
        )
        t_emb = self.time_embed(timesteps)

        for blk in self.blocks:
            x = self._run_block(
                blk=blk,
                x=x,
                t_emb=t_emb,
                padding_mask=padding_mask,
                room_mask=room_mask,
                rel_mask=rel_mask,
                rel_edge_type=rel_edge_type,
                precomputed_masks=precomputed_masks,
            )

        shift_f, scale_f = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift_f, scale_f)
        pred_noise = self.noise_head(x)
        return pred_noise * padding_mask.unsqueeze(-1).to(pred_noise.dtype)

    def forward_discrete(
        self,
        coords_0: torch.Tensor,
        timesteps: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Discrete denoising branch: predict binary-encoded x_0 of shape [B, M, 2 * num_bits]."""
        if not self.enable_discrete or self.discrete_embed is None or self.discrete_blocks is None:
            raise RuntimeError("Discrete branch is disabled on this model (enable_discrete=False).")

        padding_mask = batch["padding_mask"]
        room_mask, rel_mask, rel_edge_type = materialize_attention_masks_gpu(batch)
        precomputed_masks = prepare_tri_branch_masks(padding_mask, room_mask, rel_mask)

        x = self.discrete_embed(
            coords_0=coords_0,
            elem_type=batch["elem_type"],
            elem_idx=batch["elem_idx"],
            corner_idx=batch["corner_idx"],
            next_corner_idx=batch["next_corner_idx"],
            fixed_mask=batch["fixed_mask"],
            norm_area=batch["norm_area"],
            has_area=batch["has_area"],
        )
        t_emb = self.time_embed(timesteps)

        for blk in self.discrete_blocks:
            x = self._run_block(
                blk=blk,
                x=x,
                t_emb=t_emb,
                padding_mask=padding_mask,
                room_mask=room_mask,
                rel_mask=rel_mask,
                rel_edge_type=rel_edge_type,
                precomputed_masks=precomputed_masks,
            )

        assert self.discrete_final_adaLN is not None
        assert self.discrete_final_norm is not None
        assert self.discrete_head is not None
        shift_f, scale_f = self.discrete_final_adaLN(t_emb).chunk(2, dim=-1)
        x = _modulate(self.discrete_final_norm(x), shift_f, scale_f)
        pred_bits = self.discrete_head(x)
        return pred_bits * padding_mask.unsqueeze(-1).to(pred_bits.dtype)
