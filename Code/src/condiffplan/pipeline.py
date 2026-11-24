"""High-level inference pipeline mapping a room-only FloorplanGraph -> VectorFloorplan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from condiffplan.config import DiffusionConfig, ModelConfig
from condiffplan.data.dataset import (
    build_attention_masks,
    collate_floorplans,
    expand_graph_to_tokens,
)
from condiffplan.model.diffusion import ConditionalGaussianDiffusion
from condiffplan.model.transformer import ConDiffPlanTransformer
from condiffplan.postprocess import select_best_candidate, tokens_to_vector_floorplan
from condiffplan.schema import FloorplanGraph, VectorFloorplan


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class ConDiffPlanPipeline:
    """End-to-end conditional vector floorplan generator with Multi-Candidate Selection."""

    def __init__(
        self,
        model: ConDiffPlanTransformer,
        diffusion: ConditionalGaussianDiffusion,
        corner_stats: Optional[Dict[str, Dict[int, int]]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.device = resolve_device(str(device)) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.model.eval()
        self.diffusion = diffusion.to(self.device)
        self.corner_stats = corner_stats

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        corner_stats_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        use_ema: bool = True,
    ) -> "ConDiffPlanPipeline":
        dev = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

        model_cfg = ModelConfig(**ckpt["model_config"])
        diff_cfg = DiffusionConfig(**ckpt["diffusion_config"])

        model = ConDiffPlanTransformer(model_cfg)
        state_key = "ema_state_dict" if (use_ema and "ema_state_dict" in ckpt) else "model_state_dict"
        model.load_state_dict(ckpt[state_key])

        diffusion = ConditionalGaussianDiffusion(diff_cfg)

        corner_stats = ckpt.get("corner_stats")
        if corner_stats_path is not None and Path(corner_stats_path).exists():
            with open(corner_stats_path, "r") as f:
                raw_stats = json.load(f)
            corner_stats = {
                t: {int(k): int(v) for k, v in d.items()} for t, d in raw_stats.items()
            }

        return cls(
            model=model,
            diffusion=diffusion,
            corner_stats=corner_stats,
            device=dev,
        )

    def _prepare_candidate_item(
        self,
        graph: FloorplanGraph,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        expanded = expand_graph_to_tokens(
            graph=graph,
            corner_stats=self.corner_stats,
            rng=rng,
            max_elements=self.model.cfg.max_elements,
            max_corners=self.model.cfg.max_corners,
        )
        room_mask, rel_mask, rel_edge_type = build_attention_masks(
            elem_idx=expanded["elem_idx"],
            edge_index=expanded["edge_index"],
            edge_type=expanded["edge_type"],
            num_elements=len(expanded["manifest"]),
        )
        return {
            "coords": torch.from_numpy(expanded["coords"]),
            "elem_type": torch.from_numpy(expanded["elem_type"]),
            "elem_idx": torch.from_numpy(expanded["elem_idx"]),
            "corner_idx": torch.from_numpy(expanded["corner_idx"]),
            "next_corner_idx": torch.from_numpy(expanded["next_corner_idx"]),
            "fixed_mask": torch.from_numpy(expanded["fixed_mask"]),
            "norm_area": torch.from_numpy(expanded["norm_area"]),
            "has_area": torch.from_numpy(expanded["has_area"]),
            "room_mask": torch.from_numpy(room_mask),
            "rel_mask": torch.from_numpy(rel_mask),
            "rel_edge_type": torch.from_numpy(rel_edge_type),
            "manifest": expanded["manifest"],
            "graph": expanded["graph"],
        }

    @torch.no_grad()
    def generate(
        self,
        graph: FloorplanGraph,
        num_candidates: int = 10,
        sampler: Literal["ddpm", "ddim"] = "ddim",
        num_steps: int = 100,
        eta: float = 0.0,
        seed: Optional[int] = None,
        return_all_candidates: bool = False,
    ) -> Union[VectorFloorplan, List[VectorFloorplan]]:
        """Generate vector floorplan coordinates from a room-only connection graph.

        Args:
            graph: Input room-only FloorplanGraph with room attributes and typed edges.
            num_candidates: Number of parallel candidates N for Multi-Candidate Selection (default 10).
            sampler: 'ddim' (fast strided sampling) or 'ddpm' (full 1000-step sampling).
            num_steps: Number of denoising steps when sampler == 'ddim'.
            eta: DDIM stochasticity parameter in [0, 1].
            seed: Optional random seed for deterministic generation.
            return_all_candidates: If True, return all N candidates sorted best-first.
        """
        n_cand = max(1, int(num_candidates))
        rng = np.random.default_rng(seed)

        items = [self._prepare_candidate_item(graph, rng) for _ in range(n_cand)]
        batch = collate_floorplans(items)

        tensor_keys = [
            "coords",
            "elem_type",
            "elem_idx",
            "corner_idx",
            "next_corner_idx",
            "fixed_mask",
            "norm_area",
            "has_area",
            "padding_mask",
            "room_mask",
            "rel_mask",
            "rel_edge_type",
        ]
        device_batch = {k: batch[k].to(self.device) for k in tensor_keys}

        gen: Optional[torch.Generator] = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))

        if sampler == "ddpm":
            sampled_coords = self.diffusion.sample_ddpm(self.model, device_batch, generator=gen)
        elif sampler == "ddim":
            sampled_coords = self.diffusion.sample_ddim(
                self.model,
                device_batch,
                num_steps=num_steps,
                eta=eta,
                generator=gen,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler!r}")

        sampled_np = sampled_coords.cpu().numpy()
        candidates: List[VectorFloorplan] = []
        for i in range(n_cand):
            manifest_i = batch["manifests"][i]
            graph_i = batch["graphs"][i]
            m_i = sum(it.num_corners for it in manifest_i)
            vplan = tokens_to_vector_floorplan(
                coords_norm=sampled_np[i, :m_i],
                manifest=manifest_i,
                graph=graph_i,
                compute_metrics=True,
            )
            candidates.append(vplan)

        if return_all_candidates:
            candidates.sort(
                key=lambda c: (
                    c.metrics.get("selection_score", 0.0),
                    c.metrics.get("occupancy_ratio", 0.0),
                ),
                reverse=True,
            )
            return candidates

        return select_best_candidate(candidates)
