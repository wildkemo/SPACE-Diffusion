"""Unit tests for ConDiffPlan embeddings, TriBranchAttention, Transformer, and Diffusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from condiffplan.config import DiffusionConfig, ModelConfig
from condiffplan.data.dataset import (
    BucketBatchSampler,
    ResPlanDiffusionDataset,
    apply_dihedral_transform,
    collate_floorplans,
    expand_graph_to_tokens,
    materialize_attention_masks_gpu,
)
from condiffplan.data.preprocess import _signed_area
from condiffplan.hardware import is_oom_error, resolve_micro_batch_size, resolve_precision
from condiffplan.model.attention import TriBranchAttention
from condiffplan.model.diffusion import ConditionalGaussianDiffusion
from condiffplan.model.embeddings import bits_to_coords, coords_to_bits
from condiffplan.model.transformer import ConDiffPlanTransformer
from condiffplan.pipeline import ConDiffPlanPipeline
from condiffplan.schema import BoundarySpec, ConnectionEdge, FloorplanGraph, RoomNode
from condiffplan.train import EMA


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


def test_binary_coordinate_roundtrip() -> None:
    coords = torch.tensor(
        [[[-1.0, 1.0], [0.0, -0.5], [0.95, -0.95], [1.5, -1.5]]],
        dtype=torch.float32,
    )
    bits = coords_to_bits(coords, num_bits=8)
    assert bits.shape == (1, 4, 16)
    assert set(bits.unique().tolist()).issubset({0.0, 1.0})

    recovered = bits_to_coords(bits, num_bits=8)
    assert recovered.shape == coords.shape
    clamped = coords.clamp(-1.0, 1.0)
    # 8-bit quantization max error on [-1, 1] is 1 / 255
    assert torch.max(torch.abs(recovered - clamped)).item() <= (1.0 / 255.0 + 1e-6)


def test_discrete_branch_forward_loss_and_sampling() -> None:
    torch.manual_seed(1)
    graphs = _make_synthetic_graphs()
    ds = ResPlanDiffusionDataset(graphs, is_train=False)
    item0 = ds[0]
    item0["fixed_mask"][:4] = True
    batch = collate_floorplans([item0, ds[1]])

    cfg = ModelConfig.preset("tiny")
    assert cfg.enable_discrete is True
    model = ConDiffPlanTransformer(cfg)
    diff = ConditionalGaussianDiffusion(
        DiffusionConfig(num_timesteps=50, discrete_train_t=20, discrete_sample_t=32)
    )

    # 1. Direct forward_discrete shape & padding zeroing
    t_low = torch.tensor([5, 10], dtype=torch.long)
    pred_bits = model.forward_discrete(batch["coords"], t_low, batch)
    assert pred_bits.shape == (2, batch["coords"].shape[1], 16)
    assert torch.isfinite(pred_bits).all()
    pad_mask = batch["padding_mask"]
    if (~pad_mask).any():
        assert (pred_bits[~pad_mask] == 0.0).all()

    # 2. When all t >= discrete_train_t, discrete branch receives no gradient
    model.zero_grad()
    t_high = torch.tensor([25, 40], dtype=torch.long)
    loss_high = diff.training_loss(model, batch, t=t_high)
    loss_high.backward()
    assert model.discrete_head is not None
    assert model.discrete_head.weight.grad is None

    # 3. When any t < discrete_train_t, BOTH continuous and discrete heads receive gradients
    model.zero_grad()
    t_mix = torch.tensor([5, 40], dtype=torch.long)
    loss_mix = diff.training_loss(model, batch, t=t_mix)
    loss_mix.backward()
    assert model.noise_head.weight.grad is not None
    assert model.discrete_head.weight.grad is not None
    assert model.discrete_head.weight.grad.norm().item() > 0.0

    # 4. Reverse DDPM & DDIM sampling preserve 0.0 drift on fixed corners and quantize unfixed corners
    model.eval()
    x_0 = batch["coords"]
    f_mask = batch["fixed_mask"]
    unfixed_valid = (~f_mask) & pad_mask

    for x_gen in (
        diff.sample_ddpm(model, batch),
        diff.sample_ddim(model, batch, num_steps=10, eta=0.0),
    ):
        assert torch.max(torch.abs(x_gen[f_mask] - x_0[f_mask])).item() == 0.0
        # Unfixed valid coordinates at t=0 must lie on the exact 8-bit grid
        scaled = (x_gen[unfixed_valid] + 1.0) * 0.5 * 255.0
        assert torch.max(torch.abs(scaled - scaled.round())).item() < 1e-4


def test_hardware_precision_and_vram_profiles() -> None:
    # Ampere+ (RTX 3080/4090/A100): BF16 + TF32, no GradScaler
    dtype, amp, scaler, tf32, label = resolve_precision("auto", "cuda", (8, 6), bf16_supported=True)
    assert dtype == torch.bfloat16 and amp is True and scaler is False and tf32 is True
    assert "BF16" in label

    # Turing/Volta (RTX 2060/T4/V100): FP16 + GradScaler
    dtype, amp, scaler, tf32, _ = resolve_precision("auto", "cuda", (7, 5), bf16_supported=False)
    assert dtype == torch.float16 and amp is True and scaler is True and tf32 is False

    # Pascal (GTX 1080 Ti / P100): FP32
    dtype, amp, scaler, tf32, _ = resolve_precision("auto", "cuda", (6, 1), bf16_supported=False)
    assert dtype is None and amp is False and scaler is False and tf32 is False

    # CPU: FP32
    dtype, amp, scaler, tf32, _ = resolve_precision("auto", "cpu", None, bf16_supported=False)
    assert dtype is None and amp is False and scaler is False and tf32 is False

    # VRAM micro-batching tiers
    assert resolve_micro_batch_size(64, total_vram_gb=10.0, device_type="cuda") == (64, 1)
    assert resolve_micro_batch_size(64, total_vram_gb=8.0, device_type="cuda") == (32, 2)
    assert resolve_micro_batch_size(64, total_vram_gb=4.0, device_type="cuda") == (16, 4)
    assert resolve_micro_batch_size(64, total_vram_gb=24.0, requested_micro_batch_size=16, device_type="cuda") == (16, 4)

    # OOM error detection
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_oom_error(RuntimeError("some unrelated error"))


def test_backward_compatible_continuous_checkpoint_loading(tmp_path: Path) -> None:
    cfg = ModelConfig.preset("tiny")
    cfg.enable_discrete = False
    model = ConDiffPlanTransformer(cfg)
    diff_cfg = DiffusionConfig(num_timesteps=10)

    # Simulate a legacy checkpoint dict lacking `enable_discrete`
    legacy_model_dict = cfg.to_dict()
    legacy_model_dict.pop("enable_discrete", None)
    legacy_model_dict.pop("discrete_num_bits", None)
    legacy_model_dict.pop("discrete_n_layers", None)
    legacy_model_dict.pop("grad_checkpointing", None)

    ckpt_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "step": 10,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": model.state_dict(),
            "model_config": legacy_model_dict,
            "diffusion_config": {"num_timesteps": 10, "schedule": "cosine", "cosine_s": 0.008},
        },
        ckpt_path,
    )

    pipe = ConDiffPlanPipeline.from_checkpoint(ckpt_path, device="cpu")
    assert pipe.model.enable_discrete is False
    assert pipe.model.discrete_head is None


def test_compact_gpu_masks_match_cpu_masks_bit_for_bit() -> None:
    graphs = _make_synthetic_graphs()
    ds_full = ResPlanDiffusionDataset(graphs, is_train=False, include_metadata=True, compact_masks=False)
    ds_compact = ResPlanDiffusionDataset(graphs, is_train=False, include_metadata=False, compact_masks=True)

    b_full = collate_floorplans([ds_full[0], ds_full[1]])
    b_compact = collate_floorplans([ds_compact[0], ds_compact[1]])
    assert "elem_adj" in b_compact and "room_mask" not in b_compact

    rm, relm, relet = materialize_attention_masks_gpu(b_compact)
    assert torch.equal(rm, b_full["room_mask"])
    assert torch.equal(relm, b_full["rel_mask"])
    assert torch.equal(relet.long(), b_full["rel_edge_type"])


def test_legacy_separate_qkv_checkpoint_loads_into_fused_attention() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig.preset("tiny")
    attn = TriBranchAttention(cfg)
    attn.eval()

    # Split fused qkv_all back into legacy qkv_global/qkv_room/qkv_rel keys
    orig_sd = {k: v.clone() for k, v in attn.state_dict().items()}
    w_g, w_r, w_rel = orig_sd.pop("qkv_all.weight").chunk(3, dim=0)
    b_g, b_r, b_rel = orig_sd.pop("qkv_all.bias").chunk(3, dim=0)
    legacy_sd = {
        **orig_sd,
        "qkv_global.weight": w_g,
        "qkv_global.bias": b_g,
        "qkv_room.weight": w_r,
        "qkv_room.bias": b_r,
        "qkv_rel.weight": w_rel,
        "qkv_rel.bias": b_rel,
    }

    attn2 = TriBranchAttention(cfg)
    attn2.load_state_dict(legacy_sd, strict=True)
    assert torch.equal(attn.qkv_all.weight, attn2.qkv_all.weight)
    assert torch.equal(attn.qkv_all.bias, attn2.qkv_all.bias)


def test_bucket_batch_sampler_and_foreach_ema() -> None:
    lengths = [60 + (i * 17) % 140 for i in range(256)]
    sampler = BucketBatchSampler(lengths, batch_size=32, drop_last=True, seed=0)
    epoch1 = list(sampler)
    epoch2 = list(sampler)
    assert len(epoch1) == 8
    assert sorted(idx for b in epoch1 for idx in b) == list(range(256))
    assert epoch1 != epoch2

    # Verify foreach EMA matches exact manual lerp
    torch.manual_seed(0)
    cfg = ModelConfig.preset("tiny")
    model = ConDiffPlanTransformer(cfg)
    ema = EMA(model, decay=0.9)
    before = [p.detach().clone() for p in ema.shadow.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for b_p, s_p, m_p in zip(before, ema.shadow.parameters(), model.parameters()):
        expected = b_p * 0.9 + m_p * 0.1
        assert torch.allclose(s_p, expected, atol=1e-6)
