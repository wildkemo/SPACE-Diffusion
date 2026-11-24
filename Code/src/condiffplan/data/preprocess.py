"""Convert raw ResPlan.pkl records into room-only FloorplanGraphs and cached tensor splits."""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon, box
from tqdm import tqdm

from condiffplan.config import DEFAULT_CORNER_COUNTS, ELEM_TYPES, ROOM_TYPES, DataConfig
from condiffplan.schema import (
    BoundarySpec,
    ConnectionEdge,
    CoordFrame,
    FloorplanGraph,
    RoomNode,
)


def _extract_polygons(geom: Any) -> List[Polygon]:
    """Extract non-empty Shapely Polygons from a Polygon/MultiPolygon/GeometryCollection."""
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 1e-3 else []
    if hasattr(geom, "geoms"):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_extract_polygons(g))
        return out
    return []


def _signed_area(coords: np.ndarray) -> float:
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _canonicalize_ccw(coords: np.ndarray) -> np.ndarray:
    """Remove near-duplicate consecutive vertices, enforce CCW winding, and start at min (y, x)."""
    pts = np.asarray(coords, dtype=np.float32)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-4:
        pts = pts[:-1]

    # Filter consecutive duplicates
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 1e-3:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(pts[keep[-1]] - pts[keep[0]]) < 1e-3:
        keep.pop()
    pts = pts[keep]

    if len(pts) < 3:
        # Fallback tiny triangle/box around centroid
        c = pts.mean(axis=0) if len(pts) > 0 else np.zeros(2, dtype=np.float32)
        pts = np.array(
            [[c[0] - 1, c[1] - 1], [c[0] + 1, c[1] - 1], [c[0] + 1, c[1] + 1], [c[0] - 1, c[1] + 1]],
            dtype=np.float32,
        )

    # Enforce CCW (positive signed area)
    if _signed_area(pts) < 0.0:
        pts = pts[::-1].copy()

    # Circular shift so corner 0 is bottom-left: smallest (y, x)
    lex_idx = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    pts = np.roll(pts, -lex_idx, axis=0)
    return pts.astype(np.float32)


def simplify_polygon_ccw(
    poly: Polygon,
    base_tol: float = 1.0,
    max_corners: int = 64,
    force_four_corners: bool = False,
) -> np.ndarray:
    """Simplify a Shapely Polygon with adaptive tolerance so vertex count <= max_corners."""
    if force_four_corners:
        rect = poly.minimum_rotated_rectangle
        if isinstance(rect, Polygon) and not rect.is_empty:
            rcoords = np.asarray(rect.exterior.coords[:-1], dtype=np.float32)
            if len(rcoords) == 4:
                return _canonicalize_ccw(rcoords)
        minx, miny, maxx, maxy = poly.bounds
        bpoly = box(minx, miny, max(maxx, minx + 0.5), max(maxy, miny + 0.5))
        return _canonicalize_ccw(np.asarray(bpoly.exterior.coords[:-1], dtype=np.float32))

    tol = float(base_tol)
    curr = poly
    for _ in range(10):
        sp = curr.simplify(tol, preserve_topology=True)
        if isinstance(sp, MultiPolygon) and len(sp.geoms) > 0:
            sp = max(sp.geoms, key=lambda g: g.area)
        if isinstance(sp, Polygon) and not sp.is_empty:
            raw_pts = np.asarray(sp.exterior.coords[:-1], dtype=np.float32)
            pts = _canonicalize_ccw(raw_pts)
            if len(pts) <= max_corners:
                return pts
        tol *= 1.5

    # Final fallback: uniform vertex subsampling if still > max_corners
    raw_pts = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
    pts = _canonicalize_ccw(raw_pts)
    if len(pts) > max_corners:
        idx = np.linspace(0, len(pts) - 1, max_corners, dtype=int)
        pts = _canonicalize_ccw(pts[idx])
    return pts


def _contact_score(opening_poly: Polygon, room_poly: Polygon, buf: float) -> float:
    try:
        return float(opening_poly.boundary.intersection(room_poly.buffer(buf * 0.6)).length)
    except Exception:
        return 0.0


def resplan_to_floorplan_graph(plan: Dict[str, Any], cfg: Optional[DataConfig] = None) -> FloorplanGraph:
    """Convert a raw ResPlan dictionary into a normalized, room-only FloorplanGraph."""
    if cfg is None:
        cfg = DataConfig()

    # Handle typo in ResPlan keys
    if "balacony" in plan and "balcony" not in plan:
        plan = dict(plan)
        plan["balcony"] = plan.pop("balacony")

    raw_graph = plan.get("graph")
    inner_polys = _extract_polygons(plan.get("inner"))
    main_inner = max(inner_polys, key=lambda g: g.area) if inner_polys else None

    if main_inner is not None:
        x1, y1, x2, y2 = main_inner.bounds
        plan_span = max(x2 - x1, y2 - y1, 1.0)
        plan_area = max(float(main_inner.area), 1.0)
    else:
        plan_span = 256.0
        plan_area = 256.0 * 256.0

    buf = max(plan_span * 0.01, float(plan.get("wall_depth") or 4.0) * 0.65, 1.0)

    # 1. Extract room polygons and build room table
    room_polys: Dict[str, Polygon] = {}
    room_types: Dict[str, str] = {}
    extra_adj_pairs: List[Tuple[str, str]] = []
    split_parent_map: Dict[str, List[str]] = {}

    if raw_graph is not None:
        for nid, ndata in raw_graph.nodes(data=True):
            ntype = str(ndata.get("type", ""))
            if ntype not in ROOM_TYPES:
                continue
            parts = _extract_polygons(ndata.get("geometry"))
            if not parts:
                continue
            parts.sort(key=lambda g: g.area, reverse=True)
            largest_area = parts[0].area
            kept = [parts[0]] + [p for p in parts[1:] if p.area >= 0.05 * largest_area]
            if len(kept) == 1:
                sid = str(nid)
                room_polys[sid] = kept[0]
                room_types[sid] = ntype
                split_parent_map[sid] = [sid]
            else:
                sub_ids: List[str] = []
                for k, part in enumerate(kept):
                    sid = f"{nid}_part{k}" if k > 0 else str(nid)
                    room_polys[sid] = part
                    room_types[sid] = ntype
                    sub_ids.append(sid)
                split_parent_map[str(nid)] = sub_ids
                for k in range(len(sub_ids) - 1):
                    extra_adj_pairs.append((sub_ids[k], sub_ids[k + 1]))

    # Fallback if graph had no valid rooms
    if not room_polys:
        for rtype in ROOM_TYPES:
            for idx, poly in enumerate(_extract_polygons(plan.get(rtype))):
                sid = f"{rtype}_{idx}"
                room_polys[sid] = poly
                room_types[sid] = rtype
                split_parent_map[sid] = [sid]

    room_ids = list(room_polys.keys())
    room_bufs = {rid: room_polys[rid].buffer(buf) for rid in room_ids}

    # 2. Collect base room-to-room edges from plan["graph"]
    edge_table: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _add_or_merge_edge(u: str, v: str, etype: str) -> Tuple[str, str]:
        if u == v:
            return (u, v)
        key = (min(u, v), max(u, v))
        if etype not in ("via_door", "adjacency", "via_window"):
            etype = "adjacency"
        if key not in edge_table:
            edge_table[key] = {"type": etype, "opening_polys": []}
        else:
            # Priority: via_door > via_window > adjacency
            priority = {"via_door": 3, "via_window": 2, "adjacency": 1}
            if priority.get(etype, 1) > priority.get(edge_table[key]["type"], 1):
                edge_table[key]["type"] = etype
        return key

    if raw_graph is not None:
        for u, v, edata in raw_graph.edges(data=True):
            su, sv = str(u), str(v)
            if su not in split_parent_map or sv not in split_parent_map:
                continue
            u_list = split_parent_map[su]
            v_list = split_parent_map[sv]
            # Pick closest sub-polygon pair if a MultiPolygon living room was split
            best_pair = (u_list[0], v_list[0])
            if len(u_list) > 1 or len(v_list) > 1:
                best_d = float("inf")
                for uu in u_list:
                    for vv in v_list:
                        d = room_polys[uu].distance(room_polys[vv])
                        if d < best_d:
                            best_d = d
                            best_pair = (uu, vv)
            etype = str(edata.get("type", "adjacency"))
            if etype in ("via_opening", "fallback", "direct"):
                etype = "adjacency"
            _add_or_merge_edge(best_pair[0], best_pair[1], etype)

    for u, v in extra_adj_pairs:
        _add_or_merge_edge(u, v, "adjacency")

    # 3. Match front_door polygons to entry room(s)
    fd_by_room: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    for fd_poly in _extract_polygons(plan.get("front_door")):
        fd_coords = simplify_polygon_ccw(fd_poly, force_four_corners=True)
        touching = [rid for rid in room_ids if fd_poly.intersects(room_bufs[rid])]
        if not touching:
            # Nearest room fallback
            nearest = min(room_ids, key=lambda r: fd_poly.distance(room_polys[r]))
            touching = [nearest]
        else:
            touching.sort(
                key=lambda r: _contact_score(fd_poly, room_polys[r], buf),
                reverse=True,
            )
        fd_by_room[touching[0]].append(fd_coords)

    # 4. Match door polygons -> via_door edges
    for d_poly in _extract_polygons(plan.get("door")):
        d_coords = simplify_polygon_ccw(d_poly, force_four_corners=True)
        touching = [rid for rid in room_ids if d_poly.intersects(room_bufs[rid])]
        if len(touching) >= 2:
            scored = sorted(
                ((rid, _contact_score(d_poly, room_polys[rid], buf)) for rid in touching),
                key=lambda x: x[1],
                reverse=True,
            )
            r1, r2 = scored[0][0], scored[1][0]
            key = _add_or_merge_edge(r1, r2, "via_door")
            edge_table[key]["type"] = "via_door"
            edge_table[key]["opening_polys"].append(d_coords)

    # 5. Match window polygons -> via_window edges (2 rooms) or exterior windows (1 room)
    ext_win_by_room: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    for w_poly in _extract_polygons(plan.get("window")):
        w_coords = simplify_polygon_ccw(w_poly, force_four_corners=True)
        touching = [rid for rid in room_ids if w_poly.intersects(room_bufs[rid])]
        if not touching:
            nearest = min(room_ids, key=lambda r: w_poly.distance(room_polys[r]))
            ext_win_by_room[nearest].append(w_coords)
        elif len(touching) == 1:
            ext_win_by_room[touching[0]].append(w_coords)
        else:
            scored = sorted(
                ((rid, _contact_score(w_poly, room_polys[rid], buf)) for rid in touching),
                key=lambda x: x[1],
                reverse=True,
            )
            r1, r2 = scored[0][0], scored[1][0]
            # Only treat as via_window edge if second room has meaningful contact (e.g. room <-> balcony)
            if scored[1][1] >= 0.25 * max(scored[0][1], 1e-3):
                key = _add_or_merge_edge(r1, r2, "via_window")
                if edge_table[key]["type"] != "via_door":
                    edge_table[key]["type"] = "via_window"
                    edge_table[key]["opening_polys"].append(w_coords)
                else:
                    ext_win_by_room[r1].append(w_coords)
            else:
                ext_win_by_room[r1].append(w_coords)

    # 6. Respect max_elements budget (boundary + rooms + openings <= cfg.max_elements)
    bnd_coords = (
        simplify_polygon_ccw(
            main_inner,
            base_tol=cfg.simplify_tolerance,
            max_corners=cfg.max_corners_per_elem,
        )
        if main_inner is not None
        else None
    )

    all_point_arrays: List[np.ndarray] = []
    if bnd_coords is not None:
        all_point_arrays.append(bnd_coords)

    raw_nodes: List[RoomNode] = []
    for rid in room_ids:
        rcoords = simplify_polygon_ccw(
            room_polys[rid],
            base_tol=cfg.simplify_tolerance,
            max_corners=cfg.max_corners_per_elem,
        )
        all_point_arrays.append(rcoords)
        fd_list = fd_by_room.get(rid)
        if fd_list:
            all_point_arrays.extend(fd_list)
        ew_list = ext_win_by_room.get(rid)
        if ew_list:
            all_point_arrays.extend(ew_list)

        norm_area = float(np.clip(np.sqrt(room_polys[rid].area / plan_area), 0.01, 1.0))
        raw_nodes.append(
            RoomNode(
                id=rid,
                type=room_types[rid],
                num_corners=int(rcoords.shape[0]),
                is_fixed=False,
                coords=rcoords,
                target_area=norm_area,
                has_front_door=bool(fd_list),
                num_exterior_windows=len(ew_list) if ew_list else 0,
                front_door_coords=fd_list if fd_list else None,
                exterior_window_coords=ew_list if ew_list else None,
            )
        )

    raw_edges: List[ConnectionEdge] = []
    for (u, v), ed in sorted(edge_table.items()):
        op_list: List[np.ndarray] = ed["opening_polys"]
        if op_list:
            all_point_arrays.extend(op_list)
        raw_edges.append(
            ConnectionEdge(
                source=u,
                target=v,
                type=ed["type"],
                count=max(1, len(op_list)),
                opening_coords=op_list if op_list else None,
                is_fixed=False,
            )
        )

    # 7. Compute per-plan aspect-preserving CoordFrame and return normalized FloorplanGraph
    all_pts = np.concatenate(all_point_arrays, axis=0)
    coord_frame = CoordFrame.from_point_cloud(all_pts, margin=cfg.norm_margin)

    raw_graph_obj = FloorplanGraph(
        nodes=raw_nodes,
        edges=raw_edges,
        boundary=BoundarySpec(
            coords=bnd_coords,
            num_corners=int(bnd_coords.shape[0]) if bnd_coords is not None else None,
            is_fixed=False,
        )
        if bnd_coords is not None
        else None,
        coord_frame=coord_frame,
        coords_are_normalized=False,
        plan_id=plan.get("id"),
    )
    return raw_graph_obj.normalized_copy()


def _accumulate_corner_stats(
    counts: Dict[str, collections.Counter],
    g: FloorplanGraph,
) -> None:
    if g.boundary is not None and g.boundary.num_corners is not None:
        counts["boundary"][int(g.boundary.num_corners)] += 1
    for n in g.nodes:
        if n.num_corners is not None:
            counts[n.type][int(n.num_corners)] += 1
        if n.has_front_door:
            counts["front_door"][4] += 1
        if n.num_exterior_windows > 0:
            counts["window"][4] += n.num_exterior_windows
    for e in g.edges:
        if e.type == "via_door":
            counts["door"][4] += e.count
        elif e.type == "via_window":
            counts["window"][4] += e.count


def _finalize_corner_stats(counts: Dict[str, collections.Counter]) -> Dict[str, Dict[int, int]]:
    result: Dict[str, Dict[int, int]] = {}
    for t in ELEM_TYPES:
        if counts[t]:
            result[t] = {int(k): int(v) for k, v in sorted(counts[t].items())}
        else:
            result[t] = {DEFAULT_CORNER_COUNTS[t]: 1}
    return result


def compute_corner_stats(graphs: Sequence[FloorplanGraph]) -> Dict[str, Dict[int, int]]:
    """Compute empirical corner-count frequency distribution per element type."""
    counts: Dict[str, collections.Counter] = {t: collections.Counter() for t in ELEM_TYPES}
    for g in graphs:
        _accumulate_corner_stats(counts, g)
    return _finalize_corner_stats(counts)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def preprocess_resplan_splits(
    cfg: Optional[DataConfig] = None,
    limit_per_split: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Path]:
    """Preprocess ResPlan.pkl according to split.json and save cached .pt split files."""
    if cfg is None:
        cfg = DataConfig()

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.split_path, "r") as f:
        splits: Dict[str, List[int]] = json.load(f)

    data_path = Path(cfg.data_path)
    if show_progress and data_path.exists():
        size_mb = data_path.stat().st_size / (1024 ** 2)
        print(f"Loading {data_path.name} ({size_mb:.1f} MB) from disk...", flush=True)

    with open(data_path, "rb") as f:
        raw_plans: List[Dict[str, Any]] = pickle.load(f)

    plan_by_id: Dict[int, Dict[str, Any]] = {int(p["id"]): p for p in raw_plans}
    del raw_plans
    gc.collect()

    saved_paths: Dict[str, Path] = {}
    train_corner_counts: Dict[str, collections.Counter] = {t: collections.Counter() for t in ELEM_TYPES}

    split_names = ["train", "val", "test"]
    if cfg.include_augmented_split and "augmented" in splits:
        split_names.append("augmented")

    for sname in split_names:
        ids = splits.get(sname, [])
        if limit_per_split is not None:
            ids = ids[:limit_per_split]

        serialized_graphs: List[Dict[str, Any]] = []
        iterator = tqdm(ids, desc=f"Preprocessing {sname}") if show_progress else ids
        for pid in iterator:
            plan = plan_by_id.get(int(pid))
            if plan is None:
                continue
            fg = resplan_to_floorplan_graph(plan, cfg=cfg)
            serialized_graphs.append(fg.to_dict())
            if sname == "train":
                _accumulate_corner_stats(train_corner_counts, fg)

        out_path = cache_dir / f"{sname}.pt"
        if show_progress:
            print(f"Saving {out_path} ({len(serialized_graphs):,} plans)...", flush=True)
        _atomic_torch_save(serialized_graphs, out_path)
        saved_paths[sname] = out_path
        del serialized_graphs
        gc.collect()

    del plan_by_id
    gc.collect()

    # Save empirical corner count distribution
    corner_stats = _finalize_corner_stats(train_corner_counts)
    stats_path = cache_dir / "corner_stats.json"
    with open(stats_path, "w") as f:
        json.dump(corner_stats, f, indent=2)
    saved_paths["corner_stats"] = stats_path

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ResPlan.pkl for ConDiffPlan training.")
    parser.add_argument("--data", type=str, default="../Dataset/ResPlan.pkl", help="Path to ResPlan.pkl")
    parser.add_argument("--split", type=str, default="../Dataset/split.json", help="Path to split.json")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Output cache directory")
    parser.add_argument("--tol", type=float, default=1.0, help="Base Shapely simplify tolerance")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per split for quick runs")
    parser.add_argument("--include-augmented", action="store_true", help="Also preprocess augmented split")
    args = parser.parse_args()

    cfg = DataConfig(
        data_path=args.data,
        split_path=args.split,
        cache_dir=args.cache_dir,
        simplify_tolerance=args.tol,
        include_augmented_split=args.include_augmented,
    )
    paths = preprocess_resplan_splits(cfg=cfg, limit_per_split=args.limit, show_progress=True)
    for k, p in paths.items():
        print(f"Saved {k}: {p}")


if __name__ == "__main__":
    main()
