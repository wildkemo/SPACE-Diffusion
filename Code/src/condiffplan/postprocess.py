"""Vector polygon reconstruction, Occupancy/Gap/Overlap metrics, and Multi-Candidate Selection."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from condiffplan.data.dataset import ElementManifestItem
from condiffplan.schema import FloorplanGraph, VectorElement, VectorFloorplan


def coords_to_valid_polygon(coords: np.ndarray) -> tuple[Polygon, bool]:
    """Convert [N, 2] vertex coordinates to a valid Shapely Polygon and report raw validity."""
    pts = np.asarray(coords, dtype=np.float64)
    if len(pts) < 3:
        return Polygon(), False
    try:
        raw_poly = Polygon(pts)
        is_valid = bool(raw_poly.is_valid and not raw_poly.is_empty and raw_poly.area > 1e-6)
        if is_valid:
            return raw_poly, True
        fixed = make_valid(raw_poly)
        if isinstance(fixed, Polygon) and not fixed.is_empty:
            return fixed, False
        if hasattr(fixed, "geoms"):
            polys = [g for g in fixed.geoms if isinstance(g, Polygon) and not g.is_empty]
            if polys:
                return max(polys, key=lambda g: g.area), False
        buf_poly = raw_poly.buffer(0)
        if isinstance(buf_poly, Polygon) and not buf_poly.is_empty:
            return buf_poly, False
    except Exception:
        pass
    return Polygon(), False


def compute_layout_metrics(plan: VectorFloorplan) -> Dict[str, float]:
    """Compute Occupancy Ratio (Eq. 11), Gap Ratio, Overlap Ratio, and Polygon Validity Rate."""
    all_elems = plan.all_elements
    if not all_elems:
        return {
            "occupancy_ratio": 0.0,
            "gap_ratio": 1.0,
            "overlap_ratio": 0.0,
            "valid_polygon_rate": 0.0,
            "selection_score": 0.0,
        }

    valid_flags: List[bool] = []
    interior_room_polys: List[Polygon] = []
    all_room_polys: List[Polygon] = []

    for r in plan.rooms:
        poly, is_val = coords_to_valid_polygon(r.coords)
        valid_flags.append(is_val)
        if not poly.is_empty and poly.area > 1e-4:
            all_room_polys.append(poly)
            if r.type != "balcony":
                interior_room_polys.append(poly)

    for op in plan.openings:
        _, is_val = coords_to_valid_polygon(op.coords)
        valid_flags.append(is_val)

    bnd_poly: Optional[Polygon] = None
    if plan.boundary is not None:
        bp, is_val = coords_to_valid_polygon(plan.boundary.coords)
        valid_flags.append(is_val)
        if not bp.is_empty and bp.area > 1e-4:
            bnd_poly = bp

    valid_rate = float(np.mean(valid_flags)) if valid_flags else 0.0

    if not all_room_polys:
        return {
            "occupancy_ratio": 0.0,
            "gap_ratio": 1.0,
            "overlap_ratio": 0.0,
            "valid_polygon_rate": valid_rate,
            "selection_score": 0.0,
        }

    sum_room_area = float(sum(p.area for p in all_room_polys))
    try:
        union_all_rooms = unary_union(all_room_polys)
        union_all_area = float(union_all_rooms.area)
    except Exception:
        union_all_area = sum_room_area

    overlap_area = max(0.0, sum_room_area - union_all_area)
    overlap_ratio = float(np.clip(overlap_area / max(sum_room_area, 1e-6), 0.0, 1.0))

    if bnd_poly is not None and bnd_poly.area > 1e-4:
        target_rooms = interior_room_polys if interior_room_polys else all_room_polys
        try:
            room_union = unary_union(target_rooms)
            covered_area = float(room_union.intersection(bnd_poly).area)
        except Exception:
            covered_area = min(float(sum(p.area for p in target_rooms)), float(bnd_poly.area))
        occupancy_ratio = float(np.clip(covered_area / float(bnd_poly.area), 0.0, 1.0))
    else:
        # Unconstrained boundary-free generation: measure compactness against convex hull
        try:
            hull_area = float(unary_union(all_room_polys).convex_hull.area)
            occupancy_ratio = float(np.clip(union_all_area / max(hull_area, 1e-6), 0.0, 1.0))
        except Exception:
            occupancy_ratio = 0.0

    gap_ratio = float(np.clip(1.0 - occupancy_ratio, 0.0, 1.0))
    selection_score = float(occupancy_ratio - 0.5 * overlap_ratio)

    return {
        "occupancy_ratio": occupancy_ratio,
        "gap_ratio": gap_ratio,
        "overlap_ratio": overlap_ratio,
        "valid_polygon_rate": valid_rate,
        "selection_score": selection_score,
    }


def tokens_to_vector_floorplan(
    coords_norm: np.ndarray,
    manifest: Sequence[ElementManifestItem],
    graph: FloorplanGraph,
    compute_metrics: bool = True,
) -> VectorFloorplan:
    """Reconstruct a structured VectorFloorplan from denoised token coordinates [M, 2]."""
    pts_norm = np.asarray(coords_norm, dtype=np.float32)
    cf = graph.coord_frame
    pts_raw = cf.denormalize(pts_norm)

    rooms: List[VectorElement] = []
    openings: List[VectorElement] = []
    boundary: Optional[VectorElement] = None

    for item in manifest:
        s, e = item.start_token, item.end_token
        el = VectorElement(
            id=item.id,
            type=item.type,
            coords=pts_raw[s:e].copy(),
            coords_norm=pts_norm[s:e].copy(),
            is_fixed=item.is_fixed,
            connected_rooms=item.connected_rooms,
        )
        if item.category == "boundary":
            boundary = el
        elif item.category == "room":
            rooms.append(el)
        else:
            openings.append(el)

    vplan = VectorFloorplan(
        rooms=rooms,
        openings=openings,
        boundary=boundary,
        edges=list(graph.edges),
        coord_frame=cf,
        plan_id=graph.plan_id,
    )
    if compute_metrics:
        vplan.metrics = compute_layout_metrics(vplan)
    return vplan


def select_best_candidate(candidates: Sequence[VectorFloorplan]) -> VectorFloorplan:
    """Multi-Candidate Selection (Section 3.5): return the layout with highest Occupancy Ratio."""
    if not candidates:
        raise ValueError("select_best_candidate received an empty candidate list.")
    for c in candidates:
        if "occupancy_ratio" not in c.metrics:
            c.metrics = compute_layout_metrics(c)
    return max(
        candidates,
        key=lambda c: (
            c.metrics.get("selection_score", c.metrics.get("occupancy_ratio", 0.0)),
            c.metrics.get("occupancy_ratio", 0.0),
        ),
    )
