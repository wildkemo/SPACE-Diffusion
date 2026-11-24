"""End-to-end CPU integration smoke test: preprocess -> train -> sample -> evaluate -> export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from condiffplan.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from condiffplan.eval import compute_fixed_drift, compute_graph_compatibility, evaluate_dataset
from condiffplan.export import plot_vector_floorplan, to_geojson, to_json_file, to_svg
from condiffplan.pipeline import ConDiffPlanPipeline
from condiffplan.sample import prepare_graph_for_mode
from condiffplan.schema import FloorplanGraph, VectorFloorplan
from condiffplan.train import run_training

RESPLAN_PKL = Path(__file__).resolve().parents[2] / "Dataset" / "ResPlan.pkl"
SPLIT_JSON = Path(__file__).resolve().parents[2] / "Dataset" / "split.json"


@pytest.mark.skipif(not RESPLAN_PKL.exists(), reason="Dataset/ResPlan.pkl not found")
def test_end_to_end_pipeline_smoke(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "outputs"

    data_cfg = DataConfig(
        data_path=str(RESPLAN_PKL),
        split_path=str(SPLIT_JSON),
        cache_dir=str(cache_dir),
    )
    model_cfg = ModelConfig.preset("tiny")
    diff_cfg = DiffusionConfig(num_timesteps=10)
    train_cfg = TrainConfig(
        batch_size=4,
        lr=1e-3,
        lr_min=5e-4,
        total_steps=3,
        log_every=1,
        val_every=3,
        save_every=3,
        checkpoint_dir=str(ckpt_dir),
        amp=False,
    )

    ckpt_path = run_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        diff_cfg=diff_cfg,
        train_cfg=train_cfg,
        device_str="cpu",
        limit_per_split=8,
    )
    assert ckpt_path.exists()
    assert (cache_dir / "corner_stats.json").exists()

    pipeline = ConDiffPlanPipeline.from_checkpoint(
        checkpoint_path=ckpt_path,
        corner_stats_path=cache_dir / "corner_stats.json",
        device="cpu",
    )

    import torch

    test_records = torch.load(cache_dir / "test.pt", weights_only=False)
    ref_graph = FloorplanGraph.from_dict(test_records[0])

    # 1. Boundary-conditioned generation with Multi-Candidate Selection (N=3)
    bnd_graph = prepare_graph_for_mode(ref_graph, mode="boundary")
    vplan = pipeline.generate(
        graph=bnd_graph,
        num_candidates=3,
        sampler="ddim",
        num_steps=4,
        seed=123,
    )
    assert isinstance(vplan, VectorFloorplan)
    assert len(vplan.rooms) == len(ref_graph.nodes)
    assert "occupancy_ratio" in vplan.metrics
    assert compute_fixed_drift(bnd_graph, vplan) == 0.0

    # 2. Iterative partial editing: lock first room + boundary, regenerate remaining rooms
    fixed_rid = ref_graph.nodes[0].id
    edit_graph = bnd_graph.with_fixed_rooms(vplan, fixed_room_ids={fixed_rid}, fix_boundary=True)
    edited_plan = pipeline.generate(
        graph=edit_graph,
        num_candidates=2,
        sampler="ddim",
        num_steps=4,
        seed=456,
    )
    assert isinstance(edited_plan, VectorFloorplan)
    assert compute_fixed_drift(edit_graph, edited_plan) == 0.0
    orig_room = next(r for r in vplan.rooms if r.id == fixed_rid)
    new_room = next(r for r in edited_plan.rooms if r.id == fixed_rid)
    assert np.allclose(orig_room.coords, new_room.coords, atol=1e-4)

    # 3. Verify evaluation & export utilities
    gc = compute_graph_compatibility(edit_graph, edited_plan)
    assert "edge_match_f1" in gc and "graph_edit_distance" in gc

    eval_summary = evaluate_dataset(
        pipeline=pipeline,
        graphs=[ref_graph],
        num_fixed_rooms=1,
        with_boundary=True,
        num_candidates=2,
        sampler="ddim",
        num_steps=3,
    )
    assert eval_summary["max_fixed_corner_drift"] == 0.0

    json_file = to_json_file(edited_plan, out_dir / "edited.json")
    assert json_file.exists()
    assert to_geojson(edited_plan)["type"] == "FeatureCollection"
    to_svg(edited_plan, path=out_dir / "edited.svg")
    plot_vector_floorplan(edited_plan, save_path=out_dir / "edited.png")
    assert (out_dir / "edited.svg").exists()
    assert (out_dir / "edited.png").exists()
