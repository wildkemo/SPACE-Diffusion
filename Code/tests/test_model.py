"""Unit tests for ConDiffPlan embeddings, TriBranchAttention, Transformer, and Diffusion."""

from __future__ import annotations

import numpy as np
import torch

from condiffplan.config import DiffusionConfig, ModelConfig
from condiffplan.data.dataset import (
    ResPlanDiffusionDataset,
    apply_dihedral_transform,
    collate_floorplans,
    expand_graph_to_tokens,
)
from condiffplan.data.preprocess import _signed_area
from condiffplan.model.diffusion import ConditionalGaussianDiffusion
from condiffplan.model.transformer import ConDiffPlanTransformer
from condiffplan.schema import BoundarySpec, ConnectionEdge, FloorplanGraph, RoomNode


def _make_synthetic_graphs() -> list[FloorplanGraph]:
    bnd = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]], dtype=np.float32)
    r1 = np.array([[-0.6, -0.6], [0.0, -0.6], [0.0, 0.6], [-0.6, 0.6]], dtype=np.float32)
    r2 = np.array([[0.0, -0.6], [0.6, -0.6], [0.6, 0.6], [0.0, 0.6]], dtype=np.float32)
    r3 = np.array([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]], dtype=np.float32)

    g1 = FloorplanGraph(
        nodes=[
            RoomNode("living_0", "living", coords=r1, target_area=0.4, has_front_door=True, num_exterior_windows=1),
            RoomNode("bedroom_0", "bedroom", coords=r2, target_area=0.3, num_exterior_windows=1),
        ],
        edges=[ConnectionEdge("living_0", "bedroom_0", type="via_door", count=1)],
        boundary=BoundarySpec(coords=bnd, is_fixed=True),
        plan_id=1,
    )
    # Graph 2 includes an isolated room node (no edges) to verify relational attention never NaNs
    g2 = FloorplanGraph(
        nodes=[
            RoomNode("living_0", "living", coords=r1),
            RoomNode("bathroom_0", "bathroom", coords=r2),
            RoomNode("storage_0", "storage", coords=r3),
        ],
        edges=[ConnectionEdge("living_0", "bathroom_0", type="via_door", count=1)],
        boundary=None,
        plan_id=2,
    )
    return [g1, g2]


def test_dihedral_reflection_preserves_ccw_and_cycles() -> None:
    g = _make_synthetic_graphs()[0]
    tok = expand_graph_to_tokens(g)
    coords = tok["coords"]
    manifest = tok["manifest"]

    for k_rot in range(4):
        for flip_x in (False, True):
            pts, c_idx, nxt_idx = apply_dihedral_transform(coords, manifest, k_rot90=k_rot, flip_x=flip_x)
            for item in manifest:
                s, e = item.start_token, item.end_token
                poly_pts = pts[s:e]
                if np.ptp(poly_pts) > 0.0:
                    assert _signed_area(poly_pts) > 0.0, f"CCW violated for rot={k_rot}, flip={flip_x}"
                assert np.array_equal(nxt_idx[s:e], np.roll(np.arange(s, e), -1))
                assert np.array_equal(c_idx[s:e], np.arange(e - s))


def test_forward_shapes_padding_and_no_nans() -> None:
    graphs = _make_synthetic_graphs()
    ds = ResPlanDiffusionDataset(graphs, is_train=True)
    batch = collate_floorplans([ds[0], ds[1]])

    cfg = ModelConfig.preset("tiny")
    model = ConDiffPlanTransformer(cfg)
    t = torch.tensor([3, 7], dtype=torch.long)

    out = model(batch["coords"], t, batch)
    assert out.shape == batch["coords"].shape
    assert torch.isfinite(out).all()

    # Padded tokens must be strictly zero
    pad_mask = batch["padding_mask"]
    if (~pad_mask).any():
        assert (out[~pad_mask] == 0.0).all()


def test_fixed_corner_invariance_in_forward_and_reverse_diffusion() -> None:
    graphs = _make_synthetic_graphs()
    ds = ResPlanDiffusionDataset(graphs, is_train=False)
    item = ds[0]
    # Force boundary + living_0 to be fixed
    item["fixed_mask"][:8] = True
    batch = collate_floorplans([item])

    cfg = ModelConfig.preset("tiny")
    model = ConDiffPlanTransformer(cfg)
    model.eval()
    diff = ConditionalGaussianDiffusion(DiffusionConfig(num_timesteps=12))

    x_0 = batch["coords"]
    f_mask = batch["fixed_mask"]

    # 1. Forward q_sample must keep fixed tokens identical to x_0
    t = torch.tensor([10], dtype=torch.long)
    x_t = diff.q_sample(x_0, t=t, fixed_mask=f_mask)
    assert torch.max(torch.abs(x_t[f_mask] - x_0[f_mask])).item() == 0.0

    # 2. Reverse DDPM sampling must keep fixed tokens identical to x_0
    x_ddpm = diff.sample_ddpm(model, batch)
    assert torch.max(torch.abs(x_ddpm[f_mask] - x_0[f_mask])).item() == 0.0

    # 3. Reverse DDIM sampling must keep fixed tokens identical to x_0
    x_ddim = diff.sample_ddim(model, batch, num_steps=4, eta=0.0)
    assert torch.max(torch.abs(x_ddim[f_mask] - x_0[f_mask])).item() == 0.0


def test_gradient_flow_and_single_batch_overfit() -> None:
    torch.manual_seed(0)
    g = _make_synthetic_graphs()[0]
    ds = ResPlanDiffusionDataset([g], is_train=False)
    batch = collate_floorplans([ds[0]])

    cfg = ModelConfig.preset("tiny")
    model = ConDiffPlanTransformer(cfg)
    diff = ConditionalGaussianDiffusion(DiffusionConfig(num_timesteps=10))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    t_fixed = torch.tensor([5], dtype=torch.long)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = diff.training_loss(model, batch, t=t_fixed)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Verify gradients are finite and non-zero
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(gn) for gn in grad_norms)
    assert any(gn > 0.0 for gn in grad_norms)
    # Loss should decrease when overfitting a fixed batch/timestep
    assert losses[-1] < losses[0]
