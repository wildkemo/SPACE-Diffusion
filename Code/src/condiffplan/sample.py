"""CLI for generating vector floorplan coordinates from a room-only connection graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Set

import torch

from condiffplan.export import plot_vector_floorplan, to_geojson, to_json_file, to_svg
from condiffplan.pipeline import ConDiffPlanPipeline
from condiffplan.schema import BoundarySpec, FloorplanGraph, RoomNode


def prepare_graph_for_mode(
    graph: FloorplanGraph,
    mode: str = "boundary",
    fixed_rooms: Optional[Set[str]] = None,
) -> FloorplanGraph:
    """Configure a FloorplanGraph for unconstrained, boundary-conditioned, or partial-edit generation."""
    fixed_set = fixed_rooms or set()
    g = graph.normalized_copy()

    new_nodes: List[RoomNode] = []
    for n in g.nodes:
        is_fix = (mode == "partial_edit") and (n.id in fixed_set) and (n.coords is not None)
        new_nodes.append(
            RoomNode(
                id=n.id,
                type=n.type,
                num_corners=n.num_corners,
                is_fixed=is_fix,
                coords=n.coords if is_fix else None,
                target_area=n.target_area,
                has_front_door=n.has_front_door,
                num_exterior_windows=n.num_exterior_windows,
                front_door_coords=n.front_door_coords if is_fix else None,
                exterior_window_coords=n.exterior_window_coords if is_fix else None,
            )
        )

    new_bnd: Optional[BoundarySpec] = None
    if mode in ("boundary", "partial_edit") and g.boundary is not None:
        new_bnd = BoundarySpec(
            coords=g.boundary.coords,
            num_corners=g.boundary.num_corners,
            is_fixed=(g.boundary.coords is not None),
        )
    elif mode == "unconstrained" and g.boundary is not None:
        new_bnd = BoundarySpec(
            coords=None,
            num_corners=g.boundary.num_corners,
            is_fixed=False,
        )

    return FloorplanGraph(
        nodes=new_nodes,
        edges=g.edges,
        boundary=new_bnd,
        coord_frame=g.coord_frame,
        coords_are_normalized=True,
        plan_id=g.plan_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vector floorplan coordinates from a room-only connection graph."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--corner-stats", type=str, default="data_cache/corner_stats.json")
    parser.add_argument("--input-json", type=str, default=None, help="Path to input FloorplanGraph JSON")
    parser.add_argument(
        "--from-cache",
        type=str,
        default="data_cache/test.pt",
        help="Path to cached split .pt file if --input-json is not provided",
    )
    parser.add_argument("--plan-index", type=int, default=0, help="Index in --from-cache split")
    parser.add_argument(
        "--mode",
        type=str,
        default="boundary",
        choices=["unconstrained", "boundary", "partial_edit", "as_is"],
        help="Conditioning mode when loading from a reference graph",
    )
    parser.add_argument(
        "--fixed-rooms",
        type=str,
        default="",
        help="Comma-separated room IDs to keep fixed when --mode=partial_edit (e.g. living_0,kitchen_0)",
    )
    parser.add_argument("--num-candidates", type=int, default=10, help="N for Multi-Candidate Selection")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=100, help="Sampling steps for DDIM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/sample_vector_plan.json")
    parser.add_argument("--export-geojson", type=str, default=None)
    parser.add_argument("--export-svg", type=str, default=None)
    parser.add_argument("--export-png", type=str, default=None)
    args = parser.parse_args()

    if args.input_json is not None:
        raw_text = Path(args.input_json).read_text(encoding="utf-8")
        graph = FloorplanGraph.from_json(raw_text)
    else:
        cached = torch.load(args.from_cache, weights_only=False)
        idx = min(max(args.plan_index, 0), len(cached) - 1)
        graph = FloorplanGraph.from_dict(cached[idx])

    fixed_rooms = {r.strip() for r in args.fixed_rooms.split(",") if r.strip()}
    if args.mode != "as_is":
        if args.mode == "partial_edit" and not fixed_rooms and len(graph.nodes) > 1:
            fixed_rooms = {graph.nodes[0].id}
        graph = prepare_graph_for_mode(graph, mode=args.mode, fixed_rooms=fixed_rooms)

    pipeline = ConDiffPlanPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        corner_stats_path=args.corner_stats,
        device=args.device,
    )

    vector_plan = pipeline.generate(
        graph=graph,
        num_candidates=args.num_candidates,
        sampler=args.sampler,
        num_steps=args.steps,
        seed=args.seed,
        return_all_candidates=False,
    )

    out_json = to_json_file(vector_plan, args.output_json)
    print(f"Saved vector coordinates JSON to: {out_json}")
    print(f"Metrics: {json.dumps(vector_plan.metrics, indent=2)}")

    if args.export_geojson:
        gj_path = Path(args.export_geojson)
        gj_path.parent.mkdir(parents=True, exist_ok=True)
        gj_path.write_text(json.dumps(to_geojson(vector_plan), indent=2), encoding="utf-8")
        print(f"Saved GeoJSON to: {gj_path}")

    if args.export_svg:
        to_svg(vector_plan, path=args.export_svg)
        print(f"Saved SVG to: {args.export_svg}")

    if args.export_png:
        plot_vector_floorplan(
            vector_plan,
            title=f"Plan {vector_plan.plan_id} ({args.mode}, OR={vector_plan.metrics.get('occupancy_ratio', 0):.3f})",
            save_path=args.export_png,
        )
        print(f"Saved PNG visualization to: {args.export_png}")


if __name__ == "__main__":
    main()
