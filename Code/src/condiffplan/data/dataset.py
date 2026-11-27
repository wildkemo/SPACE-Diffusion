"""Shared graph-to-token expansion, heuristic conditional masking, and PyTorch Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from condiffplan.config import (
    DEFAULT_CORNER_COUNTS,
    ELEM_TYPE_TO_ID,
    REL_EDGE_TYPE_TO_ID,
    DataConfig,
)
from condiffplan.schema import FloorplanGraph


@dataclass
class ElementManifestItem:
    """Bookkeeping entry mapping a slice of corner tokens [start:end] back to a geometric element."""

    id: str
    type: str                      # one of ELEM_TYPES
    elem_idx: int                  # 0 .. K-1
    start_token: int               # inclusive token index
    end_token: int                 # exclusive token index
    is_fixed: bool
    category: str                  # "boundary" | "room" | "opening"
    connected_rooms: Tuple[str, ...] = ()

    @property
    def num_corners(self) -> int:
        return self.end_token - self.start_token


def sample_num_corners(
    elem_type: str,
    corner_stats: Optional[Dict[str, Dict[Union[int, str], int]]] = None,
    rng: Optional[np.random.Generator] = None,
    max_corners: int = 64,
) -> int:
    """Sample a polygon corner count for `elem_type` from empirical ResPlan statistics."""
    if elem_type in ("front_door", "door", "window"):
        return 4
    if corner_stats and elem_type in corner_stats and corner_stats[elem_type]:
        dist = corner_stats[elem_type]
        keys = np.array([min(max(int(k), 3), max_corners) for k in dist.keys()], dtype=np.int64)
        weights = np.array([float(v) for v in dist.values()], dtype=np.float64)
        weights = weights / weights.sum()
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(keys, p=weights))
    return min(DEFAULT_CORNER_COUNTS.get(elem_type, 4), max_corners)


def expand_graph_to_tokens(
    graph: FloorplanGraph,
    corner_stats: Optional[Dict[str, Dict[Union[int, str], int]]] = None,
    rng: Optional[np.random.Generator] = None,
    max_elements: int = 64,
    max_corners: int = 64,
) -> Dict[str, Any]:
    """Deterministically expand a room-only FloorplanGraph into corner tokens and relational edges.

    Element ordering (up to `max_elements`):
      1. boundary (if graph.boundary is present)
      2. rooms (in graph.nodes order)
      3. front_door polygons (1 per room with has_front_door=True, or len(front_door_coords))
      4. door polygons (edge.count per via_door edge)
      5. edge window polygons (edge.count per via_window edge)
      6. exterior window polygons (room.num_exterior_windows per room)
    """
    g = graph.normalized_copy()
    if rng is None:
        rng = np.random.default_rng()

    manifest: List[ElementManifestItem] = []
    coords_list: List[np.ndarray] = []
    elem_type_list: List[np.ndarray] = []
    elem_idx_list: List[np.ndarray] = []
    corner_idx_list: List[np.ndarray] = []
    next_corner_idx_list: List[np.ndarray] = []
    fixed_mask_list: List[np.ndarray] = []
    norm_area_list: List[np.ndarray] = []
    has_area_list: List[np.ndarray] = []

    token_cursor = 0
    elem_cursor = 0

    def _append_element(
        el_id: str,
        el_type: str,
        category: str,
        coords: Optional[np.ndarray],
        num_corners: Optional[int],
        is_fixed: bool,
        target_area: Optional[float] = None,
        connected_rooms: Tuple[str, ...] = (),
    ) -> Optional[int]:
        nonlocal token_cursor, elem_cursor
        if elem_cursor >= max_elements:
            return None

        if coords is not None:
            c = np.asarray(coords, dtype=np.float32)
            if len(c) > max_corners:
                idx = np.linspace(0, len(c) - 1, max_corners, dtype=int)
                c = c[idx]
            n_c = int(c.shape[0])
        else:
            n_c = int(num_corners) if num_corners is not None else sample_num_corners(
                el_type, corner_stats=corner_stats, rng=rng, max_corners=max_corners
            )
            n_c = min(max(n_c, 3), max_corners)
            c = np.zeros((n_c, 2), dtype=np.float32)

        e_idx = elem_cursor
        start_tok = token_cursor
        end_tok = start_tok + n_c

        coords_list.append(c)
        elem_type_list.append(np.full(n_c, ELEM_TYPE_TO_ID[el_type], dtype=np.int64))
        elem_idx_list.append(np.full(n_c, e_idx, dtype=np.int64))
        corner_idx_list.append(np.arange(n_c, dtype=np.int64))
        next_idx = np.arange(start_tok, end_tok, dtype=np.int64)
        next_idx = np.roll(next_idx, -1)
        next_corner_idx_list.append(next_idx)
        fixed_mask_list.append(np.full(n_c, bool(is_fixed), dtype=bool))

        has_a = target_area is not None and float(target_area) > 0.0
        a_val = float(target_area) if has_a else 0.0
        norm_area_list.append(np.full(n_c, a_val, dtype=np.float32))
        has_area_list.append(np.full(n_c, has_a, dtype=bool))

        manifest.append(
            ElementManifestItem(
                id=el_id,
                type=el_type,
                elem_idx=e_idx,
                start_token=start_tok,
                end_token=end_tok,
                is_fixed=bool(is_fixed),
                category=category,
                connected_rooms=connected_rooms,
            )
        )
        token_cursor = end_tok
        elem_cursor += 1
        return e_idx

    # Element-level relational edges: (elem_u, elem_v) -> rel_edge_type_id
    elem_rel_edges: Dict[Tuple[int, int], int] = {}

    def _add_elem_edge(u_idx: Optional[int], v_idx: Optional[int], rel_type: str) -> None:
        if u_idx is None or v_idx is None or u_idx == v_idx:
            return
        rt_id = REL_EDGE_TYPE_TO_ID[rel_type]
        elem_rel_edges[(u_idx, v_idx)] = rt_id
        elem_rel_edges[(v_idx, u_idx)] = rt_id

    # 1. Optional boundary
    bnd_elem_idx: Optional[int] = None
    if g.boundary is not None:
        bnd_elem_idx = _append_element(
            el_id="boundary",
            el_type="boundary",
            category="boundary",
            coords=g.boundary.coords,
            num_corners=g.boundary.num_corners,
            is_fixed=g.boundary.is_fixed,
            target_area=None,
        )

    # 2. Room nodes
    room_id_to_elem_idx: Dict[str, int] = {}
    room_id_to_node = {n.id: n for n in g.nodes}
    for n in g.nodes:
        r_idx = _append_element(
            el_id=n.id,
            el_type=n.type,
            category="room",
            coords=n.coords,
            num_corners=n.num_corners,
            is_fixed=n.is_fixed,
            target_area=n.target_area,
        )
        if r_idx is not None:
            room_id_to_elem_idx[n.id] = r_idx
            if bnd_elem_idx is not None:
                _add_elem_edge(bnd_elem_idx, r_idx, "in_boundary")

    # 3. Room-to-room relational edges
    for e in g.edges:
        u_idx = room_id_to_elem_idx.get(e.source)
        v_idx = room_id_to_elem_idx.get(e.target)
        _add_elem_edge(u_idx, v_idx, e.type)

    # 4. Front doors (from room.has_front_door)
    fd_counter = 0
    for n in g.nodes:
        if not n.has_front_door:
            continue
        r_idx = room_id_to_elem_idx.get(n.id)
        fd_coords_list = n.front_door_coords or [None]
        for fdc in fd_coords_list:
            fd_idx = _append_element(
                el_id=f"front_door_{fd_counter}",
                el_type="front_door",
                category="opening",
                coords=fdc,
                num_corners=4,
                is_fixed=bool(n.is_fixed and fdc is not None),
                connected_rooms=(n.id,),
            )
            fd_counter += 1
            _add_elem_edge(fd_idx, r_idx, "direct")
            if bnd_elem_idx is not None:
                _add_elem_edge(fd_idx, bnd_elem_idx, "in_boundary")

    # 5. Internal doors (from via_door edges)
    door_counter = 0
    for e in g.edges:
        if e.type != "via_door":
            continue
        u_idx = room_id_to_elem_idx.get(e.source)
        v_idx = room_id_to_elem_idx.get(e.target)
        u_fixed = room_id_to_node[e.source].is_fixed
        v_fixed = room_id_to_node[e.target].is_fixed
        for k in range(e.count):
            dc = e.opening_coords[k] if (e.opening_coords and k < len(e.opening_coords)) else None
            d_fixed = bool((e.is_fixed or (u_fixed and v_fixed)) and dc is not None)
            d_idx = _append_element(
                el_id=f"door_{door_counter}",
                el_type="door",
                category="opening",
                coords=dc,
                num_corners=4,
                is_fixed=d_fixed,
                connected_rooms=(e.source, e.target),
            )
            door_counter += 1
            _add_elem_edge(d_idx, u_idx, "door_to_room")
            _add_elem_edge(d_idx, v_idx, "door_to_room")
            if bnd_elem_idx is not None:
                _add_elem_edge(d_idx, bnd_elem_idx, "in_boundary")

    # 6. Shared-boundary windows (from via_window edges)
    win_counter = 0
    for e in g.edges:
        if e.type != "via_window":
            continue
        u_idx = room_id_to_elem_idx.get(e.source)
        v_idx = room_id_to_elem_idx.get(e.target)
        u_fixed = room_id_to_node[e.source].is_fixed
        v_fixed = room_id_to_node[e.target].is_fixed
        for k in range(e.count):
            wc = e.opening_coords[k] if (e.opening_coords and k < len(e.opening_coords)) else None
            w_fixed = bool((e.is_fixed or (u_fixed and v_fixed)) and wc is not None)
            w_idx = _append_element(
                el_id=f"window_{win_counter}",
                el_type="window",
                category="opening",
                coords=wc,
                num_corners=4,
                is_fixed=w_fixed,
                connected_rooms=(e.source, e.target),
            )
            win_counter += 1
            _add_elem_edge(w_idx, u_idx, "window_to_room")
            _add_elem_edge(w_idx, v_idx, "window_to_room")
            if bnd_elem_idx is not None:
                _add_elem_edge(w_idx, bnd_elem_idx, "in_boundary")

    # 7. Exterior windows (from room.num_exterior_windows)
    for n in g.nodes:
        if n.num_exterior_windows <= 0:
            continue
        r_idx = room_id_to_elem_idx.get(n.id)
        for k in range(n.num_exterior_windows):
            wc = (
                n.exterior_window_coords[k]
                if (n.exterior_window_coords and k < len(n.exterior_window_coords))
                else None
            )
            w_fixed = bool(n.is_fixed and wc is not None)
            w_idx = _append_element(
                el_id=f"window_{win_counter}",
                el_type="window",
                category="opening",
                coords=wc,
                num_corners=4,
                is_fixed=w_fixed,
                connected_rooms=(n.id,),
            )
            win_counter += 1
            _add_elem_edge(w_idx, r_idx, "window_to_room")
            if bnd_elem_idx is not None:
                _add_elem_edge(w_idx, bnd_elem_idx, "in_boundary")

    coords_arr = np.concatenate(coords_list, axis=0)
    elem_type_arr = np.concatenate(elem_type_list, axis=0)
    elem_idx_arr = np.concatenate(elem_idx_list, axis=0)
    corner_idx_arr = np.concatenate(corner_idx_list, axis=0)
    next_corner_idx_arr = np.concatenate(next_corner_idx_list, axis=0)
    fixed_mask_arr = np.concatenate(fixed_mask_list, axis=0)
    norm_area_arr = np.concatenate(norm_area_list, axis=0)
    has_area_arr = np.concatenate(has_area_list, axis=0)

    if elem_rel_edges:
        pairs = list(elem_rel_edges.keys())
        edge_index = np.array(pairs, dtype=np.int64).T  # [2, E]
        edge_type = np.array([elem_rel_edges[p] for p in pairs], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    return {
        "coords": coords_arr,
        "elem_type": elem_type_arr,
        "elem_idx": elem_idx_arr,
        "corner_idx": corner_idx_arr,
        "next_corner_idx": next_corner_idx_arr,
        "fixed_mask": fixed_mask_arr,
        "norm_area": norm_area_arr,
        "has_area": has_area_arr,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "manifest": manifest,
        "graph": g,
    }


def build_attention_masks(
    elem_idx: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    num_elements: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build [M, M] room_mask, rel_mask, and rel_edge_type matrices from element-level edges."""
    m = len(elem_idx)
    room_mask = elem_idx[:, None] == elem_idx[None, :]

    elem_adj = np.zeros((num_elements, num_elements), dtype=bool)
    elem_etype = np.zeros((num_elements, num_elements), dtype=np.int64)
    if edge_index.shape[1] > 0:
        u = edge_index[0]
        v = edge_index[1]
        elem_adj[u, v] = True
        elem_etype[u, v] = edge_type

    rel_mask = elem_adj[elem_idx[:, None], elem_idx[None, :]]
    rel_edge_type = elem_etype[elem_idx[:, None], elem_idx[None, :]]
    # Always allow self-attention on the diagonal so isolated elements never cause softmax NaNs
    np.fill_diagonal(rel_mask, True)
    return room_mask, rel_mask, rel_edge_type


def apply_dihedral_transform(
    coords: np.ndarray,
    manifest: Sequence[ElementManifestItem],
    k_rot90: int,
    flip_x: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one of the 8 dihedral symmetries of [-1, 1]^2 while preserving CCW polygon winding."""
    pts = coords.copy()
    k = k_rot90 % 4
    if k == 1:
        pts = np.stack([-pts[:, 1], pts[:, 0]], axis=-1)
    elif k == 2:
        pts = -pts
    elif k == 3:
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=-1)

    if flip_x:
        pts[:, 0] = -pts[:, 0]

    corner_idx = np.zeros(len(pts), dtype=np.int64)
    next_corner_idx = np.zeros(len(pts), dtype=np.int64)

    for item in manifest:
        s, e = item.start_token, item.end_token
        poly_pts = pts[s:e]
        if flip_x:
            # Reflection reverses orientation -> reverse vertices to keep CCW
            poly_pts = poly_pts[::-1].copy()
        # Re-anchor vertex 0 to bottom-left (min y, min x)
        lex_idx = int(np.lexsort((poly_pts[:, 0], poly_pts[:, 1]))[0])
        poly_pts = np.roll(poly_pts, -lex_idx, axis=0)
        pts[s:e] = poly_pts

        n_c = e - s
        corner_idx[s:e] = np.arange(n_c, dtype=np.int64)
        nxt = np.arange(s, e, dtype=np.int64)
        next_corner_idx[s:e] = np.roll(nxt, -1)

    return pts, corner_idx, next_corner_idx


class BucketBatchSampler(Sampler[List[int]]):
    """Randomized length-bucketed batch sampler to eliminate O(M^2) padding waste.

    Groups samples of similar corner counts M within randomized megabatches, then shuffles
    batch order every epoch so every sample is visited once per epoch with minimal padding.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        pool_multiplier: int = 16,
        seed: int = 42,
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = max(int(batch_size), 1)
        self.drop_last = bool(drop_last)
        self.pool_size = max(self.batch_size * max(int(pool_multiplier), 1), self.batch_size)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        n = len(self.lengths)
        order = rng.permutation(n)

        batches: List[List[int]] = []
        for pool_start in range(0, n, self.pool_size):
            pool = order[pool_start : pool_start + self.pool_size]
            # Sort within the randomized pool by sequence length
            pool_sorted = pool[np.argsort(self.lengths[pool], kind="mergesort")]
            for b_start in range(0, len(pool_sorted), self.batch_size):
                chunk = pool_sorted[b_start : b_start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk.tolist())

        # Shuffle the order of batches across the epoch
        batch_perm = rng.permutation(len(batches))
        for idx in batch_perm:
            yield batches[int(idx)]


def materialize_attention_masks_gpu(
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (room_mask, rel_mask, rel_edge_type), building them on-device from compact element
    adjacency matrices when not already materialized on CPU.
    """
    if "room_mask" in batch and "rel_mask" in batch and "rel_edge_type" in batch:
        return batch["room_mask"], batch["rel_mask"], batch["rel_edge_type"]

    elem_idx = batch["elem_idx"]          # [B, M] long
    padding_mask = batch["padding_mask"]  # [B, M] bool
    elem_adj = batch["elem_adj"]          # [B, K, K] bool
    elem_etype = batch["elem_etype"]      # [B, K, K] uint8

    bsz, seqlen = elem_idx.shape
    device = elem_idx.device

    e_i = elem_idx.unsqueeze(2)  # [B, M, 1]
    e_j = elem_idx.unsqueeze(1)  # [B, 1, M]
    valid_pair = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
    diag = torch.eye(seqlen, device=device, dtype=torch.bool).unsqueeze(0)

    room_mask = ((e_i == e_j) & valid_pair) | diag
    b_idx = torch.arange(bsz, device=device).view(bsz, 1, 1)
    rel_mask = (elem_adj[b_idx, e_i, e_j] & valid_pair) | diag
    zero = torch.zeros((), device=device, dtype=elem_etype.dtype)
    rel_edge_type = torch.where(valid_pair, elem_etype[b_idx, e_i, e_j], zero)
    return room_mask, rel_mask, rel_edge_type


class ResPlanDiffusionDataset(Dataset):
    """PyTorch Dataset yielding tokenized floorplans with heuristic conditional masking.

    Pre-expands static graph structures once in __init__ so __getitem__ only performs fast
    dihedral augmentation and conditional masking with zero redundant graph traversal.
    """

    def __init__(
        self,
        graphs_or_cache_path: Union[str, Path, Sequence[Union[FloorplanGraph, Dict[str, Any]]]],
        cfg: Optional[DataConfig] = None,
        is_train: bool = True,
        include_metadata: bool = True,
        compact_masks: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DataConfig()
        self.is_train = is_train
        self.include_metadata = include_metadata
        # By default, use compact element-level masks during training/val (include_metadata=False)
        # to avoid O(M^2) CPU allocations and PCIe transfers; materialize on GPU in microseconds.
        self.compact_masks = (not include_metadata) if compact_masks is None else bool(compact_masks)

        if isinstance(graphs_or_cache_path, (str, Path)):
            raw_list = torch.load(graphs_or_cache_path, weights_only=False)
        else:
            raw_list = list(graphs_or_cache_path)

        self._records: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        for g_raw in raw_list:
            g = g_raw if isinstance(g_raw, FloorplanGraph) else FloorplanGraph.from_dict(g_raw)
            expanded = expand_graph_to_tokens(
                g,
                max_elements=self.cfg.max_elements,
                max_corners=self.cfg.max_corners_per_elem,
            )
            manifest: List[ElementManifestItem] = expanded["manifest"]
            num_elems = len(manifest)
            elem_adj = np.zeros((num_elems, num_elems), dtype=bool)
            elem_etype = np.zeros((num_elems, num_elems), dtype=np.uint8)
            edge_index = expanded["edge_index"]
            edge_type = expanded["edge_type"]
            if edge_index.shape[1] > 0:
                u, v = edge_index[0], edge_index[1]
                elem_adj[u, v] = True
                elem_etype[u, v] = edge_type.astype(np.uint8)

            rec: Dict[str, Any] = {
                "coords": expanded["coords"],
                "elem_type": expanded["elem_type"],
                "elem_idx": expanded["elem_idx"],
                "corner_idx": expanded["corner_idx"],
                "next_corner_idx": expanded["next_corner_idx"],
                "norm_area": expanded["norm_area"],
                "has_area": expanded["has_area"],
                "elem_adj": elem_adj,
                "elem_etype": elem_etype,
                "manifest": manifest,
                "room_items": [it for it in manifest if it.category == "room"],
                "bnd_items": [it for it in manifest if it.category == "boundary"],
                "opening_items": [it for it in manifest if it.category == "opening" and it.connected_rooms],
            }
            if include_metadata:
                rec["graph"] = expanded["graph"]
            self._records.append(rec)
            self.lengths.append(int(expanded["coords"].shape[0]))
        del raw_list

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        coords = rec["coords"]
        elem_type = rec["elem_type"]
        elem_idx = rec["elem_idx"]
        corner_idx = rec["corner_idx"]
        next_corner_idx = rec["next_corner_idx"]
        norm_area = rec["norm_area"].copy()
        has_area = rec["has_area"].copy()
        manifest: List[ElementManifestItem] = rec["manifest"]

        rng = np.random.default_rng()

        # 1. Optional random dihedral augmentation during training
        if self.is_train and self.cfg.random_dihedral:
            k_rot = int(rng.integers(0, 4))
            flip_x = bool(rng.random() < 0.5)
            coords, corner_idx, next_corner_idx = apply_dihedral_transform(
                coords, manifest, k_rot90=k_rot, flip_x=flip_x
            )

        # 2. Heuristic conditional masking (Section 3.4.3 of ConDiffPlan)
        fixed_mask = np.zeros(len(coords), dtype=bool)
        if self.is_train:
            fix_bnd = bool(rng.random() < self.cfg.p_fix_boundary)
            fix_rooms = bool(rng.random() < self.cfg.p_fix_rooms)
            keep_area = bool(rng.random() < self.cfg.p_condition_area)
        else:
            fix_bnd = True
            fix_rooms = False
            keep_area = True

        room_items: List[ElementManifestItem] = rec["room_items"]
        fixed_room_ids = set()
        if fix_rooms and len(room_items) > 1:
            num_to_fix = int(rng.integers(1, len(room_items)))
            chosen = rng.choice(len(room_items), size=num_to_fix, replace=False)
            for c_idx in chosen:
                it = room_items[int(c_idx)]
                fixed_mask[it.start_token : it.end_token] = True
                fixed_room_ids.add(it.id)

        if fix_bnd:
            for it in rec["bnd_items"]:
                fixed_mask[it.start_token : it.end_token] = True

        if fixed_room_ids:
            for it in rec["opening_items"]:
                if all(rid in fixed_room_ids for rid in it.connected_rooms):
                    fixed_mask[it.start_token : it.end_token] = True

        # Guaranteed at least one unfixed room so training loss is always well-defined
        if fixed_mask.all() and room_items:
            last_room = room_items[-1]
            fixed_mask[last_room.start_token : last_room.end_token] = False

        if not keep_area:
            # Drop area conditioning on unfixed tokens
            unfixed = ~fixed_mask
            norm_area[unfixed] = 0.0
            has_area[unfixed] = False

        sample: Dict[str, Any] = {
            "coords": torch.from_numpy(coords),
            "elem_type": torch.from_numpy(elem_type),
            "elem_idx": torch.from_numpy(elem_idx),
            "corner_idx": torch.from_numpy(corner_idx),
            "next_corner_idx": torch.from_numpy(next_corner_idx),
            "fixed_mask": torch.from_numpy(fixed_mask),
            "norm_area": torch.from_numpy(norm_area),
            "has_area": torch.from_numpy(has_area),
        }

        if self.compact_masks:
            sample["elem_adj"] = torch.from_numpy(rec["elem_adj"])
            sample["elem_etype"] = torch.from_numpy(rec["elem_etype"])
        else:
            elem_adj = rec["elem_adj"]
            elem_etype = rec["elem_etype"]
            room_mask = elem_idx[:, None] == elem_idx[None, :]
            rel_mask = elem_adj[elem_idx[:, None], elem_idx[None, :]]
            np.fill_diagonal(rel_mask, True)
            rel_edge_type = elem_etype[elem_idx[:, None], elem_idx[None, :]].astype(np.int64)
            sample["room_mask"] = torch.from_numpy(room_mask)
            sample["rel_mask"] = torch.from_numpy(rel_mask)
            sample["rel_edge_type"] = torch.from_numpy(rel_edge_type)

        if self.include_metadata:
            sample["manifest"] = manifest
            sample["graph"] = rec["graph"]
        return sample


def collate_floorplans(batch: Sequence[Dict[str, Any]], pad_to_multiple: int = 8) -> Dict[str, Any]:
    """Pad variable-length floorplan token sequences to a Tensor-Core-aligned multiple."""
    bsz = len(batch)
    raw_max_m = max(int(item["coords"].shape[0]) for item in batch)
    if pad_to_multiple > 1:
        max_m = ((raw_max_m + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    else:
        max_m = raw_max_m

    coords = torch.zeros((bsz, max_m, 2), dtype=torch.float32)
    elem_type = torch.zeros((bsz, max_m), dtype=torch.long)
    elem_idx = torch.zeros((bsz, max_m), dtype=torch.long)
    corner_idx = torch.zeros((bsz, max_m), dtype=torch.long)
    next_corner_idx = torch.zeros((bsz, max_m), dtype=torch.long)
    fixed_mask = torch.zeros((bsz, max_m), dtype=torch.bool)
    norm_area = torch.zeros((bsz, max_m), dtype=torch.float32)
    has_area = torch.zeros((bsz, max_m), dtype=torch.bool)
    padding_mask = torch.zeros((bsz, max_m), dtype=torch.bool)

    use_compact = ("elem_adj" in batch[0]) and ("room_mask" not in batch[0])
    if use_compact:
        max_k = max(int(item["elem_adj"].shape[0]) for item in batch)
        elem_adj = torch.zeros((bsz, max_k, max_k), dtype=torch.bool)
        elem_etype = torch.zeros((bsz, max_k, max_k), dtype=torch.uint8)
    else:
        room_mask = torch.zeros((bsz, max_m, max_m), dtype=torch.bool)
        rel_mask = torch.zeros((bsz, max_m, max_m), dtype=torch.bool)
        rel_edge_type = torch.zeros((bsz, max_m, max_m), dtype=torch.long)
        # Diagonal self-attention enabled everywhere to avoid NaN in softmax on padded rows
        diag_idx = torch.arange(max_m)
        room_mask[:, diag_idx, diag_idx] = True
        rel_mask[:, diag_idx, diag_idx] = True

    has_meta = "manifest" in batch[0] and "graph" in batch[0]
    manifests: List[List[ElementManifestItem]] = []
    graphs: List[FloorplanGraph] = []

    for i, item in enumerate(batch):
        m = int(item["coords"].shape[0])
        coords[i, :m] = item["coords"]
        elem_type[i, :m] = item["elem_type"]
        elem_idx[i, :m] = item["elem_idx"]
        corner_idx[i, :m] = item["corner_idx"]
        next_corner_idx[i, :m] = item["next_corner_idx"]
        # For padded positions, point next_corner_idx to themselves
        if m < max_m:
            next_corner_idx[i, m:] = torch.arange(m, max_m, dtype=torch.long)
        fixed_mask[i, :m] = item["fixed_mask"]
        norm_area[i, :m] = item["norm_area"]
        has_area[i, :m] = item["has_area"]
        padding_mask[i, :m] = True

        if use_compact:
            k = int(item["elem_adj"].shape[0])
            elem_adj[i, :k, :k] = item["elem_adj"]
            elem_etype[i, :k, :k] = item["elem_etype"]
        else:
            room_mask[i, :m, :m] = item["room_mask"]
            rel_mask[i, :m, :m] = item["rel_mask"]
            rel_edge_type[i, :m, :m] = item["rel_edge_type"]

        if has_meta:
            manifests.append(item["manifest"])
            graphs.append(item["graph"])

    out: Dict[str, Any] = {
        "coords": coords,
        "elem_type": elem_type,
        "elem_idx": elem_idx,
        "corner_idx": corner_idx,
        "next_corner_idx": next_corner_idx,
        "fixed_mask": fixed_mask,
        "norm_area": norm_area,
        "has_area": has_area,
        "padding_mask": padding_mask,
    }
    if use_compact:
        out["elem_adj"] = elem_adj
        out["elem_etype"] = elem_etype
    else:
        out["room_mask"] = room_mask
        out["rel_mask"] = rel_mask
        out["rel_edge_type"] = rel_edge_type

    if has_meta:
        out["manifests"] = manifests
        out["graphs"] = graphs
    return out
