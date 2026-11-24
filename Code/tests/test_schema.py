"""Unit tests for room-only FloorplanGraph input schema, VectorFloorplan output schema, and exports."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from condiffplan.export import (
    plot_vector_floorplan,
    to_geojson,
    to_json_file,
    to_shapely_dict,
    to_svg,
)
from condiffplan.schema import (
    BoundarySpec,
    ConnectionEdge,
    CoordFrame,
    FloorplanGraph,
    RoomNode,
    VectorElement,
    VectorFloorplan,
)


def _make_sample_graph() -> FloorplanGraph:
    living_coords = np.array([[-0.5, -0.5], [0.2, -0.5], [0.2, 0.3], [-0.5, 0.3]], dtype=np.float32)
    bed_coords = np.array([[0.2, -0.5], [0.7, -0.5], [0.7, 0.3], [0.2, 0.3]], dtype=np.float32)
    bnd_coords = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]], dtype=np.float32)

    nodes = [
        RoomNode(
            id="living_0",
            type="living",
            num_corners=4,
            is_fixed=True,
            coords=living_coords,
            target_area=0.45,
            has_front_door=True,
            num_exterior_windows=1,
        ),
        RoomNode(
            id="bedroom_0",
            type="bedroom",
            num_corners=4,
            is_fixed=False,
            coords=bed_coords,
            target_area=0.30,
            has_front_door=False,
            num_exterior_windows=2,
        ),
        RoomNode(
            id="kitchen_0",
            type="kitchen",
            num_corners=6,
            is_fixed=False,
            coords=None,
        ),
    ]
    edges = [
        ConnectionEdge(source="living_0", target="bedroom_0", type="via_door", count=1),
        ConnectionEdge(source="living_0", target="kitchen_0", type="adjacency", count=1),
    ]
    boundary = BoundarySpec(coords=bnd_coords, num_corners=4, is_fixed=True)
    return FloorplanGraph(nodes=nodes, edges=edges, boundary=boundary, plan_id=101)


def test_room_only_validation_rejects_non_room_nodes() -> None:
    with pytest.raises(ValueError, match="strictly rooms"):
        RoomNode(id="door_0", type="door")

    with pytest.raises(ValueError, match="strictly rooms"):
        RoomNode(id="window_0", type="window")

    with pytest.raises(ValueError, match="strictly rooms"):
        RoomNode(id="bnd", type="boundary")


def test_fixed_node_requires_coords() -> None:
    with pytest.raises(ValueError, match="coords=None"):
        RoomNode(id="living_0", type="living", is_fixed=True, coords=None)

    with pytest.raises(ValueError, match="must provide coords"):
        BoundarySpec(is_fixed=True, coords=None)


def test_edge_validation() -> None:
    with pytest.raises(ValueError, match="ConnectionEdge.type"):
        ConnectionEdge(source="living_0", target="bedroom_0", type="invalid_edge")

    with pytest.raises(ValueError, match="Self-loop"):
        ConnectionEdge(source="living_0", target="living_0", type="via_door")

    n1 = RoomNode(id="living_0", type="living")
    with pytest.raises(ValueError, match="unknown room"):
        FloorplanGraph(
            nodes=[n1],
            edges=[ConnectionEdge(source="living_0", target="missing_room", type="via_door")],
        )


def test_graph_json_and_networkx_roundtrip() -> None:
    g = _make_sample_graph()
    json_str = g.to_json()
    g2 = FloorplanGraph.from_json(json_str)

    assert len(g2.nodes) == 3
    assert len(g2.edges) == 2
    assert g2.boundary is not None and g2.boundary.is_fixed
    assert np.allclose(g2.nodes[0].coords, g.nodes[0].coords)

    nx_g = g.to_networkx()
    g3 = FloorplanGraph.from_networkx(nx_g)
    assert [n.id for n in g3.nodes] == [n.id for n in g.nodes]
    assert len(g3.edges) == 2
    assert g3.nodes[0].has_front_door is True


def test_networkx_folds_raw_front_door_node() -> None:
    G = nx.Graph()
    G.add_node("living_0", type="living")
    G.add_node("bedroom_0", type="bedroom")
    G.add_node("front_door_0", type="front_door")
    G.add_edge("living_0", "bedroom_0", type="via_door")
    G.add_edge("front_door_0", "living_0", type="direct")

    fg = FloorplanGraph.from_networkx(G)
    assert len(fg.nodes) == 2
    assert all(n.type in ("living", "bedroom") for n in fg.nodes)
    living_node = next(n for n in fg.nodes if n.id == "living_0")
    assert living_node.has_front_door is True


def test_vector_floorplan_and_iterative_editing_exports(tmp_path: Path) -> None:
    cf = CoordFrame(center_x=128.0, center_y=128.0, scale=100.0)
    r1_norm = np.array([[-0.5, -0.5], [0.0, -0.5], [0.0, 0.5], [-0.5, 0.5]], dtype=np.float32)
    r2_norm = np.array([[0.0, -0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    d_norm = np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1]], dtype=np.float32)
    b_norm = np.array([[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]], dtype=np.float32)

    vplan = VectorFloorplan(
        rooms=[
            VectorElement("living_0", "living", cf.denormalize(r1_norm), r1_norm, is_fixed=False),
            VectorElement("bedroom_0", "bedroom", cf.denormalize(r2_norm), r2_norm, is_fixed=False),
        ],
        openings=[
            VectorElement(
                "door_0",
                "door",
                cf.denormalize(d_norm),
                d_norm,
                is_fixed=False,
                connected_rooms=("living_0", "bedroom_0"),
            )
        ],
        boundary=VectorElement("boundary", "boundary", cf.denormalize(b_norm), b_norm, is_fixed=True),
        edges=[ConnectionEdge("living_0", "bedroom_0", type="via_door", count=1)],
        coord_frame=cf,
        metrics={"occupancy_ratio": 0.85, "gap_ratio": 0.15},
        plan_id=7,
    )

    # Test with_fixed_rooms iterative edit workflow
    base_graph = FloorplanGraph(
        nodes=[
            RoomNode("living_0", "living", num_corners=4),
            RoomNode("bedroom_0", "bedroom", num_corners=4),
        ],
        edges=[ConnectionEdge("living_0", "bedroom_0", type="via_door")],
    )
    edited_graph = base_graph.with_fixed_rooms(vplan, fixed_room_ids={"living_0"}, fix_boundary=True)
    assert edited_graph.nodes[0].is_fixed is True
    assert np.allclose(edited_graph.nodes[0].coords, r1_norm)
    assert edited_graph.nodes[1].is_fixed is False
    assert edited_graph.nodes[1].coords is None
    assert edited_graph.boundary is not None and edited_graph.boundary.is_fixed is True

    # Test downstream export adapters
    json_path = to_json_file(vplan, tmp_path / "plan.json")
    assert json_path.exists()

    gj = to_geojson(vplan)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 4

    sh_dict = to_shapely_dict(vplan)
    assert not sh_dict["living"].is_empty
    assert not sh_dict["bedroom"].is_empty
    assert not sh_dict["inner"].is_empty

    svg_str = to_svg(vplan, path=tmp_path / "plan.svg")
    assert "<svg" in svg_str and "</svg>" in svg_str
    assert (tmp_path / "plan.svg").exists()

    png_path = tmp_path / "plan.png"
    plot_vector_floorplan(vplan, save_path=png_path)
    assert png_path.exists()
