"""Optional downstream representation adapters for VectorFloorplan.

The diffusion model outputs pure vector polygon coordinates (`VectorFloorplan`).
This module converts `VectorFloorplan` into any downstream format required:
  - JSON (`to_json_file`)
  - RFC 7946 GeoJSON (`to_geojson`)
  - ResPlan-compatible Shapely dictionary (`to_shapely_dict`)
  - Vector SVG (`to_svg`)
  - Matplotlib visualization (`plot_vector_floorplan`)
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon

from condiffplan.postprocess import coords_to_valid_polygon
from condiffplan.schema import VectorFloorplan

CATEGORY_COLORS: Dict[str, str] = {
    "boundary": "#f7f7f7",
    "living": "#d9d9d9",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "kitchen": "#8da0cb",
    "door": "#e78ac3",
    "window": "#a6d854",
    "front_door": "#a63603",
    "balcony": "#b3b3b3",
    "storage": "#FF8C69",
    "stair": "#9e9ac8",
}


def to_json_file(plan: VectorFloorplan, path: Union[str, Path], indent: int = 2) -> Path:
    """Save raw vector coordinates and metadata to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(plan.to_json(indent=indent), encoding="utf-8")
    return out_path


def to_geojson(plan: VectorFloorplan, use_normalized: bool = False) -> Dict[str, Any]:
    """Convert VectorFloorplan into an RFC 7946 GeoJSON FeatureCollection."""
    features: List[Dict[str, Any]] = []
    for el in plan.all_elements:
        pts = el.coords_norm if use_normalized else el.coords
        ring = pts.tolist()
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": el.id,
                    "type": el.type,
                    "is_fixed": bool(el.is_fixed),
                    "connected_rooms": list(el.connected_rooms),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "properties": {
            "plan_id": plan.plan_id,
            "metrics": plan.metrics,
            "edges": [e.to_dict() for e in plan.edges],
        },
        "features": features,
    }


def to_shapely_dict(plan: VectorFloorplan) -> Dict[str, Any]:
    """Convert VectorFloorplan into a ResPlan-compatible dict of Shapely geometries.

    Compatible with `Dataset/resplan_utils.py`'s `plot_plan` and `plan_to_graph`.
    """
    by_type: Dict[str, List[Polygon]] = collections.defaultdict(list)
    for r in plan.rooms:
        poly, _ = coords_to_valid_polygon(r.coords)
        if not poly.is_empty:
            by_type[r.type].append(poly)
    for op in plan.openings:
        poly, _ = coords_to_valid_polygon(op.coords)
        if not poly.is_empty:
            by_type[op.type].append(poly)

    out: Dict[str, Any] = {"id": plan.plan_id, "wall_depth": 4.0}
    if plan.boundary is not None:
        bpoly, _ = coords_to_valid_polygon(plan.boundary.coords)
        out["inner"] = bpoly
    else:
        out["inner"] = Polygon()

    all_keys = [
        "living",
        "bedroom",
        "bathroom",
        "kitchen",
        "balcony",
        "storage",
        "stair",
        "door",
        "window",
        "front_door",
    ]
    for k in all_keys:
        polys = by_type.get(k, [])
        if not polys:
            out[k] = Polygon()
        elif len(polys) == 1:
            out[k] = polys[0]
        else:
            out[k] = MultiPolygon(polys)
    return out


def to_svg(
    plan: VectorFloorplan,
    path: Optional[Union[str, Path]] = None,
    canvas_size: int = 512,
    padding: int = 24,
) -> str:
    """Export VectorFloorplan to a standalone SVG string (and optionally write to `path`)."""
    elems = plan.all_elements
    if not elems:
        svg_str = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}"/>'
        if path is not None:
            Path(path).write_text(svg_str, encoding="utf-8")
        return svg_str

    all_pts = [el.coords for el in elems if len(el.coords) > 0]
    import numpy as np

    cat_pts = np.concatenate(all_pts, axis=0)
    xmin, ymin = cat_pts.min(axis=0)
    xmax, ymax = cat_pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    draw_span = canvas_size - 2 * padding

    def _map_pt(x: float, y: float) -> tuple[float, float]:
        sx = padding + ((x - xmin) / span) * draw_span
        # Flip Y for SVG screen coordinates
        sy = canvas_size - (padding + ((y - ymin) / span) * draw_span)
        return sx, sy

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_size} {canvas_size}" width="{canvas_size}" height="{canvas_size}">',
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    # Draw in layer order: boundary -> rooms -> openings
    ordered: List[Any] = []
    if plan.boundary is not None:
        ordered.append(plan.boundary)
    ordered.extend(plan.rooms)
    ordered.extend(plan.openings)

    for el in ordered:
        pts_svg = " ".join(f"{_map_pt(float(p[0]), float(p[1]))[0]:.2f},{_map_pt(float(p[0]), float(p[1]))[1]:.2f}" for p in el.coords)
        fill = CATEGORY_COLORS.get(el.type, "#cccccc")
        fill_opacity = "0.25" if el.type == "boundary" else "0.85"
        stroke_dash = ' stroke-dasharray="6,3"' if el.is_fixed else ""
        stroke_w = "2.2" if el.is_fixed else "1.4"
        lines.append(
            f'  <polygon id="{el.id}" data-type="{el.type}" points="{pts_svg}" '
            f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="#222222" stroke-width="{stroke_w}"{stroke_dash}/>'
        )

    lines.append("</svg>")
    svg_str = "\n".join(lines)
    if path is not None:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_str, encoding="utf-8")
    return svg_str


def plot_vector_floorplan(
    plan: VectorFloorplan,
    ax: Optional[plt.Axes] = None,
    show_graph: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """Render a VectorFloorplan using Matplotlib with optional connectivity graph overlay."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_fig = True

    present_types: List[str] = []

    if plan.boundary is not None:
        bnd_patch = MplPolygon(
            plan.boundary.coords,
            closed=True,
            facecolor=CATEGORY_COLORS["boundary"],
            edgecolor="#333333",
            linewidth=2.0 if plan.boundary.is_fixed else 1.2,
            linestyle="--" if plan.boundary.is_fixed else "-",
            alpha=0.35,
        )
        ax.add_patch(bnd_patch)
        present_types.append("boundary")

    room_centroids: Dict[str, tuple[float, float]] = {}
    for r in plan.rooms:
        color = CATEGORY_COLORS.get(r.type, "#cccccc")
        patch = MplPolygon(
            r.coords,
            closed=True,
            facecolor=color,
            edgecolor="#111111",
            linewidth=2.0 if r.is_fixed else 1.0,
            alpha=0.85,
        )
        ax.add_patch(patch)
        present_types.append(r.type)
        cx = float(r.coords[:, 0].mean())
        cy = float(r.coords[:, 1].mean())
        room_centroids[r.id] = (cx, cy)
        ax.text(cx, cy, r.id, fontsize=7, ha="center", va="center", color="#111111")

    for op in plan.openings:
        color = CATEGORY_COLORS.get(op.type, "#e78ac3")
        patch = MplPolygon(
            op.coords,
            closed=True,
            facecolor=color,
            edgecolor="#000000",
            linewidth=1.0,
            alpha=0.95,
        )
        ax.add_patch(patch)
        present_types.append(op.type)

    if show_graph:
        edge_colors = {
            "via_door": "#1f4e78",
            "adjacency": "#2e7d32",
            "via_window": "#ef6c00",
        }
        for e in plan.edges:
            if e.source in room_centroids and e.target in room_centroids:
                x1, y1 = room_centroids[e.source]
                x2, y2 = room_centroids[e.target]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_colors.get(e.type, "#444444"),
                    linestyle="-" if e.type == "via_door" else "--",
                    linewidth=1.3,
                    alpha=0.75,
                )

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)

    uniq_types = list(dict.fromkeys(present_types))
    if uniq_types:
        handles = [
            Patch(facecolor=CATEGORY_COLORS.get(t, "#cccccc"), edgecolor="black", label=t.replace("_", " "))
            for t in uniq_types
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    if save_path is not None:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight", dpi=160)
        if created_fig:
            plt.close(ax.figure)

    return ax
