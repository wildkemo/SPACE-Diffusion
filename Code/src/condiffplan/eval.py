"""Quantitative evaluation of ConDiffPlan on ResPlan:
- Occupancy Ratio (OR) & Gap Ratio (1 - OR) across constraint levels
- Overlap Ratio & Polygon Validity Rate
- Graph Compatibility (Edge Match F1 & Modified Graph Edit Distance)
- Fixed-Element Preservation Error (verifies 0.0 coordinate drift)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from condiffplan.pipeline import ConDiffPlanPipeline
from condiffplan.postprocess import coords_to_valid_polygon
from condiffplan.sample import prepare_graph_for_mode
from condiffplan.schema import FloorplanGraph, VectorFloorplan


def extract_generated_room_edges(plan: VectorFloorplan, buffer_dist: float = 4.0) -> Set[Tuple[str, str]]:
    """Recover room-to-room connectivity edges geometrically from generated vector coordinates."""
    room_polys = {}
    room_bufs = {}
    for r in plan.rooms:
        poly, _ = coords_to_valid_polygon(r.coords)
        if not poly.is_empty:
            room_polys[r.id] = poly
            room_bufs[r.id] = poly.buffer(buffer_dist)

    rids = list(room_polys.keys())
    recovered: Set[Tuple[str, str]] = set()

    # 1. Connections bridged by generated door or window polygons
    for op in plan.openings:
        if op.type not in ("door", "window"):
            continue
        op_poly, _ = coords_to_valid_polygon(op.coords)
        if op_poly.is_empty:
            continue
        touching = [rid for rid in rids if op_poly.intersects(room_bufs[rid])]
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                u, v = min(touching[i], touching[j]), max(touching[i], touching[j])
                recovered.add((u, v))

    # 2. Direct geometric adjacency between buffered room polygons
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            u, v = min(rids[i], rids[j]), max(rids[i], rids[j])
            if (u, v) in recovered:
                continue
            inter = room_bufs[u].intersection(room_bufs[v])
            if not inter.is_empty and inter.area >= buffer_dist * buffer_dist * 0.25:
                recovered.add((u, v))

    return recovered


def compute_graph_compatibility(
    input_graph: FloorplanGraph,
    generated_plan: VectorFloorplan,
) -> Dict[str, float]:
    """Compare target room connectivity edges against geometrically recovered room connections."""
    target_edges: Set[Tuple[str, str]] = {
        (min(e.source, e.target), max(e.source, e.target)) for e in input_graph.edges
    }
    pred_edges = extract_generated_room_edges(generated_plan)

    if not target_edges and not pred_edges:
        return {"edge_match_f1": 1.0, "edge_recall": 1.0, "graph_edit_distance": 0.0}

    tp = len(target_edges & pred_edges)
    fp = len(pred_edges - target_edges)
    fn = len(target_edges - pred_edges)

    recall = tp / max(len(target_edges), 1)
    precision = tp / max(len(pred_edges), 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
    # Modified Graph Edit Distance: symmetric edge difference (missing + spurious connections)
    ged = float(fp + fn)
    return {
        "edge_match_f1": float(f1),
        "edge_recall": float(recall),
        "graph_edit_distance": ged,
    }


def compute_fixed_drift(
    input_graph: FloorplanGraph,
    generated_plan: VectorFloorplan,
) -> float:
    """Compute maximum L2 coordinate error on fixed boundary and fixed rooms (must be 0.0)."""
    g = input_graph.normalized_copy()
    max_err = 0.0

    if g.boundary is not None and g.boundary.is_fixed and g.boundary.coords is not None:
        if generated_plan.boundary is not None:
            diff = np.abs(g.boundary.coords - generated_plan.boundary.coords_norm).max()
            max_err = max(max_err, float(diff))

    gen_rooms = {r.id: r for r in generated_plan.rooms}
    for n in g.nodes:
        if n.is_fixed and n.coords is not None and n.id in gen_rooms:
            diff = np.abs(n.coords - gen_rooms[n.id].coords_norm).max()
            max_err = max(max_err, float(diff))

    return max_err


def evaluate_dataset(
    pipeline: ConDiffPlanPipeline,
    graphs: List[FloorplanGraph],
    num_fixed_rooms: int = 0,
    with_boundary: bool = True,
    num_candidates: int = 10,
    sampler: str = "ddim",
    num_steps: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate ConDiffPlan across a list of test FloorplanGraphs under specified constraints."""
    rng = np.random.default_rng(seed)

    or_list: List[float] = []
    gap_list: List[float] = []
    ov_list: List[float] = []
    val_list: List[float] = []
    f1_list: List[float] = []
    ged_list: List[float] = []
    drift_list: List[float] = []

    for idx, base_g in enumerate(graphs):
        g_norm = base_g.normalized_copy()
        if num_fixed_rooms > 0:
            k_fix = min(num_fixed_rooms, max(len(g_norm.nodes) - 1, 0))
            chosen_idx = rng.choice(len(g_norm.nodes), size=k_fix, replace=False) if k_fix > 0 else []
            fixed_ids = {g_norm.nodes[int(i)].id for i in chosen_idx}
            cond_g = prepare_graph_for_mode(g_norm, mode="partial_edit", fixed_rooms=fixed_ids)
            if not with_boundary and cond_g.boundary is not None:
                cond_g.boundary.is_fixed = False
                cond_g.boundary.coords = None
        else:
            mode = "boundary" if with_boundary else "unconstrained"
            cond_g = prepare_graph_for_mode(g_norm, mode=mode)

        pred_plan = pipeline.generate(
            graph=cond_g,
            num_candidates=num_candidates,
            sampler=sampler,  # type: ignore[arg-type]
            num_steps=num_steps,
            seed=seed + idx,
            return_all_candidates=False,
        )
        assert isinstance(pred_plan, VectorFloorplan)

        or_list.append(pred_plan.metrics.get("occupancy_ratio", 0.0))
        gap_list.append(pred_plan.metrics.get("gap_ratio", 1.0))
        ov_list.append(pred_plan.metrics.get("overlap_ratio", 0.0))
        val_list.append(pred_plan.metrics.get("valid_polygon_rate", 0.0))

        gc = compute_graph_compatibility(cond_g, pred_plan)
        f1_list.append(gc["edge_match_f1"])
        ged_list.append(gc["graph_edit_distance"])
        drift_list.append(compute_fixed_drift(cond_g, pred_plan))

    return {
        "num_samples": float(len(graphs)),
        "num_fixed_rooms": float(num_fixed_rooms),
        "with_boundary": float(with_boundary),
        "num_candidates": float(num_candidates),
        "occupancy_ratio_mean": float(np.mean(or_list)),
        "gap_ratio_mean": float(np.mean(gap_list)),
        "overlap_ratio_mean": float(np.mean(ov_list)),
        "valid_polygon_rate_mean": float(np.mean(val_list)),
        "edge_match_f1_mean": float(np.mean(f1_list)),
        "graph_compatibility_ged_mean": float(np.mean(ged_list)),
        "max_fixed_corner_drift": float(np.max(drift_list)) if drift_list else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ConDiffPlan checkpoint on ResPlan.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-cache", type=str, default="data_cache/test.pt")
    parser.add_argument("--corner-stats", type=str, default="data_cache/corner_stats.json")
    parser.add_argument("--limit", type=int, default=50, help="Number of test plans to evaluate")
    parser.add_argument("--fixed-rooms", type=int, default=0, help="Number of fixed rooms (0..5)")
    parser.add_argument("--no-boundary", action="store_true", help="Evaluate without fixed house boundary")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    raw_list = torch.load(args.test_cache, weights_only=False)[: args.limit]
    graphs = [FloorplanGraph.from_dict(d) for d in raw_list]

    pipeline = ConDiffPlanPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        corner_stats_path=args.corner_stats,
        device=args.device,
    )

    metrics = evaluate_dataset(
        pipeline=pipeline,
        graphs=graphs,
        num_fixed_rooms=args.fixed_rooms,
        with_boundary=not args.no_boundary,
        num_candidates=args.num_candidates,
        sampler=args.sampler,
        num_steps=args.steps,
    )
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
