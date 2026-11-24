"""Tests for ResPlan preprocessing, polygon simplification, and graph-to-token expansion."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from condiffplan.config import ROOM_TYPES, DataConfig
from condiffplan.data.dataset import build_attention_masks, expand_graph_to_tokens
from condiffplan.data.preprocess import compute_corner_stats, resplan_to_floorplan_graph

RESPLAN_PKL = Path(__file__).resolve().parents[2] / "Dataset" / "ResPlan.pkl"


@pytest.mark.skipif(not RESPLAN_PKL.exists(), reason="Dataset/ResPlan.pkl not found")
def test_preprocess_real_resplan_records() -> None:
    with open(RESPLAN_PKL, "rb") as f:
        plans = pickle.load(f)

    # Pick first 35 plans + explicitly search for a multi-polygon living plan and high-vertex plan
    selected = list(plans[:35])
    for p in plans[35:1500]:
        g = p.get("graph")
        if g is not None:
            for _, d in g.nodes(data=True):
                geom = d.get("geometry")
                if geom is not None and geom.geom_type == "MultiPolygon":
                    selected.append(p)
                    break
        if len(selected) >= 42:
            break

    cfg = DataConfig()
    graphs = []
    for p in selected:
        fg = resplan_to_floorplan_graph(p, cfg=cfg)
        fg.validate()
        graphs.append(fg)

        # Every node in FloorplanGraph must be a room
        assert all(n.type in ROOM_TYPES for n in fg.nodes)
        assert fg.boundary is not None and fg.boundary.coords is not None

        # Check coordinate frame round-trip accuracy
        c_norm = fg.nodes[0].coords
        assert c_norm is not None
        assert np.all(np.isfinite(c_norm))
        assert np.max(np.abs(c_norm)) <= 1.0 + 1e-4
        c_raw = fg.coord_frame.denormalize(c_norm)
        c_back = fg.coord_frame.normalize(c_raw)
        assert np.max(np.abs(c_norm - c_back)) < 1e-4

        # Check token expansion
        tok = expand_graph_to_tokens(
            fg,
            max_elements=cfg.max_elements,
            max_corners=cfg.max_corners_per_elem,
        )
        coords = tok["coords"]
        assert coords.ndim == 2 and coords.shape[1] == 2
        assert np.all(np.isfinite(coords))
        assert np.max(np.abs(coords)) <= 1.0 + 1e-4
        assert np.all(tok["elem_idx"] < cfg.max_elements)
        assert np.all(tok["corner_idx"] < cfg.max_corners_per_elem)

        # Verify next_corner_idx forms a valid closed cycle per element
        for item in tok["manifest"]:
            s, e = item.start_token, item.end_token
            nxt = tok["next_corner_idx"][s:e]
            expected = np.roll(np.arange(s, e), -1)
            assert np.array_equal(nxt, expected)

        room_mask, rel_mask, rel_edge_type = build_attention_masks(
            elem_idx=tok["elem_idx"],
            edge_index=tok["edge_index"],
            edge_type=tok["edge_type"],
            num_elements=len(tok["manifest"]),
        )
        assert room_mask.shape == (len(coords), len(coords))
        assert rel_mask.shape == (len(coords), len(coords))
        assert rel_edge_type.shape == (len(coords), len(coords))
        assert np.all(np.diag(rel_mask))

    stats = compute_corner_stats(graphs)
    assert "living" in stats and "boundary" in stats and "door" in stats
    assert stats["door"] == {4: sum(stats["door"].values())}
