"""Strict connection-graph input (FloorplanGraph) and vector-coordinate output (VectorFloorplan) schemas.

Design invariant:
- FloorplanGraph.nodes contains ONLY room nodes (ROOM_TYPES).
- FloorplanGraph.edges contains ONLY typed room-to-room connections (GRAPH_EDGE_TYPES).
- The optional building boundary is stored at FloorplanGraph.boundary.
- Explicit 4-corner door, window, and front_door polygons are spawned internally from
  via_door/via_window edges and room attributes (has_front_door, num_exterior_windows),
  and returned as raw 2D vector coordinates in VectorFloorplan.openings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np

from condiffplan.config import ELEM_TYPES, GRAPH_EDGE_TYPES, ROOM_TYPES


def _as_coords_array(coords: Optional[Union[Sequence[Sequence[float]], np.ndarray]]) -> Optional[np.ndarray]:
    if coords is None:
        return None
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(f"Polygon coords must have shape [N>=3, 2], got {arr.shape}")
    return arr


def _as_coords_list(
    coords_list: Optional[Sequence[Union[Sequence[Sequence[float]], np.ndarray]]]
) -> Optional[List[np.ndarray]]:
    if coords_list is None:
        return None
    out: List[np.ndarray] = []
    for c in coords_list:
        arr = _as_coords_array(c)
        if arr is not None:
            out.append(arr)
    return out


@dataclass
class CoordFrame:
    """Aspect-preserving affine coordinate frame mapping raw coordinates <-> [-1, 1]."""

    center_x: float = 128.0
    center_y: float = 128.0
    scale: float = 128.0  # half-span: raw = center + norm * scale

    def normalize(self, coords: np.ndarray) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float32)
        center = np.array([self.center_x, self.center_y], dtype=np.float32)
        return (arr - center) / max(float(self.scale), 1e-6)

    def denormalize(self, coords_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(coords_norm, dtype=np.float32)
        center = np.array([self.center_x, self.center_y], dtype=np.float32)
        return arr * float(self.scale) + center

    @classmethod
    def from_point_cloud(cls, all_points: np.ndarray, margin: float = 0.95) -> "CoordFrame":
        pts = np.asarray(all_points, dtype=np.float32).reshape(-1, 2)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        cx = 0.5 * float(xmin + xmax)
        cy = 0.5 * float(ymin + ymax)
        half_span = 0.5 * max(float(xmax - xmin), float(ymax - ymin), 1.0)
        scale = half_span / max(float(margin), 1e-3)
        return cls(center_x=cx, center_y=cy, scale=scale)

    def to_dict(self) -> Dict[str, float]:
        return {
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "scale": float(self.scale),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordFrame":
        return cls(
            center_x=float(d.get("center_x", 128.0)),
            center_y=float(d.get("center_y", 128.0)),
            scale=float(d.get("scale", 128.0)),
        )


@dataclass
class BoundarySpec:
    """Optional house boundary specification on a FloorplanGraph."""

    coords: Optional[np.ndarray] = None
    num_corners: Optional[int] = None
    is_fixed: bool = False

    def __post_init__(self) -> None:
        self.coords = _as_coords_array(self.coords)
        if self.coords is not None:
            self.num_corners = int(self.coords.shape[0])
        if self.is_fixed and self.coords is None:
            raise ValueError("BoundarySpec with is_fixed=True must provide coords.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coords": self.coords.tolist() if self.coords is not None else None,
            "num_corners": self.num_corners,
            "is_fixed": bool(self.is_fixed),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundarySpec":
        return cls(
            coords=d.get("coords"),
            num_corners=d.get("num_corners"),
            is_fixed=bool(d.get("is_fixed", False)),
        )


@dataclass
class RoomNode:
    """A room node in the bubble diagram (strictly one of ROOM_TYPES)."""

    id: str
    type: str
    num_corners: Optional[int] = None
    is_fixed: bool = False
    coords: Optional[np.ndarray] = None
    target_area: Optional[float] = None
    has_front_door: bool = False
    num_exterior_windows: int = 0
    # Optional reference coordinates for attached openings (used in training / fixed-room edits)
    front_door_coords: Optional[List[np.ndarray]] = None
    exterior_window_coords: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.type not in ROOM_TYPES:
            raise ValueError(
                f"RoomNode.type must be one of {ROOM_TYPES} (nodes are strictly rooms), got {self.type!r} for node {self.id!r}"
            )
        self.coords = _as_coords_array(self.coords)
        if self.coords is not None:
            self.num_corners = int(self.coords.shape[0])
        if self.is_fixed and self.coords is None:
            raise ValueError(f"RoomNode {self.id!r} has is_fixed=True but coords=None.")
        self.num_exterior_windows = max(0, int(self.num_exterior_windows))
        self.front_door_coords = _as_coords_list(self.front_door_coords)
        self.exterior_window_coords = _as_coords_list(self.exterior_window_coords)
        if self.exterior_window_coords and self.num_exterior_windows == 0:
            self.num_exterior_windows = len(self.exterior_window_coords)
        if self.front_door_coords:
            self.has_front_door = True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "num_corners": self.num_corners,
            "is_fixed": bool(self.is_fixed),
            "coords": self.coords.tolist() if self.coords is not None else None,
            "target_area": float(self.target_area) if self.target_area is not None else None,
            "has_front_door": bool(self.has_front_door),
            "num_exterior_windows": int(self.num_exterior_windows),
        }
        if self.front_door_coords is not None:
            out["front_door_coords"] = [c.tolist() for c in self.front_door_coords]
        if self.exterior_window_coords is not None:
            out["exterior_window_coords"] = [c.tolist() for c in self.exterior_window_coords]
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoomNode":
        return cls(
            id=str(d["id"]),
            type=str(d["type"]),
            num_corners=d.get("num_corners"),
            is_fixed=bool(d.get("is_fixed", False)),
            coords=d.get("coords"),
            target_area=d.get("target_area"),
            has_front_door=bool(d.get("has_front_door", False)),
            num_exterior_windows=int(d.get("num_exterior_windows", 0)),
            front_door_coords=d.get("front_door_coords"),
            exterior_window_coords=d.get("exterior_window_coords"),
        )


@dataclass
class ConnectionEdge:
    """A typed room-to-room connection edge in the bubble diagram."""

    source: str
    target: str
    type: str = "via_door"
    count: int = 1
    opening_coords: Optional[List[np.ndarray]] = None
    is_fixed: bool = False

    def __post_init__(self) -> None:
        if self.type not in GRAPH_EDGE_TYPES:
            raise ValueError(
                f"ConnectionEdge.type must be one of {GRAPH_EDGE_TYPES}, got {self.type!r} ({self.source} <-> {self.target})"
            )
        if self.source == self.target:
            raise ValueError(f"Self-loop edges are not allowed: {self.source!r}")
        self.count = max(1, int(self.count))
        self.opening_coords = _as_coords_list(self.opening_coords)
        if self.opening_coords and self.type in ("via_door", "via_window"):
            self.count = max(self.count, len(self.opening_coords))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "count": int(self.count),
            "is_fixed": bool(self.is_fixed),
        }
        if self.opening_coords is not None:
            out["opening_coords"] = [c.tolist() for c in self.opening_coords]
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConnectionEdge":
        return cls(
            source=str(d["source"]),
            target=str(d["target"]),
            type=str(d.get("type", "via_door")),
            count=int(d.get("count", 1)),
            opening_coords=d.get("opening_coords"),
            is_fixed=bool(d.get("is_fixed", False)),
        )


@dataclass
class FloorplanGraph:
    """Room-only bubble diagram representing the full floorplan topology and attributes."""

    nodes: List[RoomNode]
    edges: List[ConnectionEdge]
    boundary: Optional[BoundarySpec] = None
    coord_frame: CoordFrame = field(default_factory=CoordFrame)
    coords_are_normalized: bool = True
    plan_id: Optional[Union[int, str]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("FloorplanGraph must contain at least one RoomNode.")
        seen_ids: Set[str] = set()
        for node in self.nodes:
            if not isinstance(node, RoomNode):
                raise TypeError(f"Expected RoomNode, got {type(node).__name__}")
            if node.id in seen_ids:
                raise ValueError(f"Duplicate RoomNode id: {node.id!r}")
            seen_ids.add(node.id)
        for edge in self.edges:
            if not isinstance(edge, ConnectionEdge):
                raise TypeError(f"Expected ConnectionEdge, got {type(edge).__name__}")
            if edge.source not in seen_ids or edge.target not in seen_ids:
                raise ValueError(
                    f"Edge ({edge.source!r}, {edge.target!r}) references unknown room node(s). Known rooms: {sorted(seen_ids)}"
                )

    def normalized_copy(self) -> "FloorplanGraph":
        """Return a copy where all present coordinates are normalized into [-1, 1] via coord_frame."""
        if self.coords_are_normalized:
            return self
        cf = self.coord_frame
        new_nodes: List[RoomNode] = []
        for n in self.nodes:
            new_nodes.append(
                RoomNode(
                    id=n.id,
                    type=n.type,
                    num_corners=n.num_corners,
                    is_fixed=n.is_fixed,
                    coords=cf.normalize(n.coords) if n.coords is not None else None,
                    target_area=n.target_area,
                    has_front_door=n.has_front_door,
                    num_exterior_windows=n.num_exterior_windows,
                    front_door_coords=[cf.normalize(c) for c in n.front_door_coords]
                    if n.front_door_coords
                    else None,
                    exterior_window_coords=[cf.normalize(c) for c in n.exterior_window_coords]
                    if n.exterior_window_coords
                    else None,
                )
            )
        new_edges: List[ConnectionEdge] = []
        for e in self.edges:
            new_edges.append(
                ConnectionEdge(
                    source=e.source,
                    target=e.target,
                    type=e.type,
                    count=e.count,
                    opening_coords=[cf.normalize(c) for c in e.opening_coords]
                    if e.opening_coords
                    else None,
                    is_fixed=e.is_fixed,
                )
            )
        new_bnd: Optional[BoundarySpec] = None
        if self.boundary is not None:
            new_bnd = BoundarySpec(
                coords=cf.normalize(self.boundary.coords)
                if self.boundary.coords is not None
                else None,
                num_corners=self.boundary.num_corners,
                is_fixed=self.boundary.is_fixed,
            )
        return FloorplanGraph(
            nodes=new_nodes,
            edges=new_edges,
            boundary=new_bnd,
            coord_frame=cf,
            coords_are_normalized=True,
            plan_id=self.plan_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "coords_are_normalized": bool(self.coords_are_normalized),
            "coord_frame": self.coord_frame.to_dict(),
            "boundary": self.boundary.to_dict() if self.boundary is not None else None,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FloorplanGraph":
        bnd_raw = payload.get("boundary")
        cf_raw = payload.get("coord_frame")
        return cls(
            nodes=[RoomNode.from_dict(n) for n in payload["nodes"]],
            edges=[ConnectionEdge.from_dict(e) for e in payload.get("edges", [])],
            boundary=BoundarySpec.from_dict(bnd_raw) if bnd_raw is not None else None,
            coord_frame=CoordFrame.from_dict(cf_raw) if cf_raw is not None else CoordFrame(),
            coords_are_normalized=bool(payload.get("coords_are_normalized", True)),
            plan_id=payload.get("plan_id"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FloorplanGraph":
        return cls.from_dict(json.loads(json_str))

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph(
            plan_id=self.plan_id,
            coord_frame=self.coord_frame.to_dict(),
            coords_are_normalized=self.coords_are_normalized,
            boundary=self.boundary.to_dict() if self.boundary is not None else None,
        )
        for n in self.nodes:
            G.add_node(
                n.id,
                type=n.type,
                num_corners=n.num_corners,
                is_fixed=n.is_fixed,
                coords=n.coords,
                target_area=n.target_area,
                has_front_door=n.has_front_door,
                num_exterior_windows=n.num_exterior_windows,
                front_door_coords=n.front_door_coords,
                exterior_window_coords=n.exterior_window_coords,
            )
        for e in self.edges:
            G.add_edge(
                e.source,
                e.target,
                type=e.type,
                count=e.count,
                opening_coords=e.opening_coords,
                is_fixed=e.is_fixed,
            )
        return G

    @classmethod
    def from_networkx(
        cls,
        G: nx.Graph,
        boundary: Optional[BoundarySpec] = None,
        coord_frame: Optional[CoordFrame] = None,
        coords_are_normalized: Optional[bool] = None,
    ) -> "FloorplanGraph":
        """Build a room-only FloorplanGraph from a NetworkX graph.

        If the NetworkX graph comes from raw ResPlan (which may contain a 'front_door_*'
        node with 'direct' edges), those 'front_door_*' nodes are automatically folded
        into `has_front_door=True` on the connected room node(s), and any 'fallback' or
        'via_opening' edge types are normalized to 'adjacency'.
        """
        front_door_rooms: Set[str] = set()
        for nid, ndata in G.nodes(data=True):
            if ndata.get("type") == "front_door":
                for nbr in G.neighbors(nid):
                    if G.nodes[nbr].get("type") in ROOM_TYPES:
                        front_door_rooms.add(str(nbr))

        nodes: List[RoomNode] = []
        room_ids: Set[str] = set()
        for nid, ndata in G.nodes(data=True):
            ntype = str(ndata.get("type", ""))
            if ntype == "front_door":
                continue
            if ntype not in ROOM_TYPES:
                raise ValueError(
                    f"Graph node {nid!r} has non-room type {ntype!r}; allowed room types: {ROOM_TYPES}"
                )
            coords = ndata.get("coords")
            if coords is None and "geometry" in ndata and ndata["geometry"] is not None:
                geom = ndata["geometry"]
                if hasattr(geom, "exterior"):
                    coords = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
            node = RoomNode(
                id=str(nid),
                type=ntype,
                num_corners=ndata.get("num_corners"),
                is_fixed=bool(ndata.get("is_fixed", False)),
                coords=coords,
                target_area=ndata.get("target_area"),
                has_front_door=bool(ndata.get("has_front_door", False)) or (str(nid) in front_door_rooms),
                num_exterior_windows=int(ndata.get("num_exterior_windows", 0)),
                front_door_coords=ndata.get("front_door_coords"),
                exterior_window_coords=ndata.get("exterior_window_coords"),
            )
            nodes.append(node)
            room_ids.add(node.id)

        edges: List[ConnectionEdge] = []
        for u, v, edata in G.edges(data=True):
            su, sv = str(u), str(v)
            if su not in room_ids or sv not in room_ids:
                continue
            etype = str(edata.get("type", "via_door"))
            if etype in ("via_opening", "fallback", "direct"):
                etype = "adjacency"
            edges.append(
                ConnectionEdge(
                    source=su,
                    target=sv,
                    type=etype,
                    count=int(edata.get("count", 1)),
                    opening_coords=edata.get("opening_coords"),
                    is_fixed=bool(edata.get("is_fixed", False)),
                )
            )

        if boundary is None and G.graph.get("boundary") is not None:
            boundary = BoundarySpec.from_dict(G.graph["boundary"])
        if coord_frame is None:
            cf_dict = G.graph.get("coord_frame")
            coord_frame = CoordFrame.from_dict(cf_dict) if cf_dict else CoordFrame()
        if coords_are_normalized is None:
            coords_are_normalized = bool(G.graph.get("coords_are_normalized", True))

        return cls(
            nodes=nodes,
            edges=edges,
            boundary=boundary,
            coord_frame=coord_frame,
            coords_are_normalized=coords_are_normalized,
            plan_id=G.graph.get("plan_id"),
        )

    def with_fixed_rooms(
        self,
        vector_plan: "VectorFloorplan",
        fixed_room_ids: Iterable[str],
        fix_boundary: bool = False,
    ) -> "FloorplanGraph":
        """Create an updated FloorplanGraph where specified rooms (and optionally boundary)
        are locked to their vector coordinates from `vector_plan` for iterative editing.
        """
        fixed_set = set(fixed_room_ids)
        room_map = {r.id: r for r in vector_plan.rooms}

        # Group openings by attached rooms
        fd_by_room: Dict[str, List[np.ndarray]] = {}
        ext_win_by_room: Dict[str, List[np.ndarray]] = {}
        edge_openings: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
        for op in vector_plan.openings:
            if op.type == "front_door" and len(op.connected_rooms) == 1:
                fd_by_room.setdefault(op.connected_rooms[0], []).append(op.coords_norm.copy())
            elif op.type == "window" and len(op.connected_rooms) == 1:
                ext_win_by_room.setdefault(op.connected_rooms[0], []).append(op.coords_norm.copy())
            elif op.type in ("door", "window") and len(op.connected_rooms) >= 2:
                r1, r2 = op.connected_rooms[0], op.connected_rooms[1]
                etype = "via_door" if op.type == "door" else "via_window"
                key = (min(r1, r2), max(r1, r2), etype)
                edge_openings.setdefault(key, []).append(op.coords_norm.copy())

        new_nodes: List[RoomNode] = []
        for n in self.nodes:
            is_fix = n.id in fixed_set and n.id in room_map
            vec_r = room_map.get(n.id)
            new_nodes.append(
                RoomNode(
                    id=n.id,
                    type=n.type,
                    num_corners=vec_r.coords_norm.shape[0] if is_fix and vec_r else n.num_corners,
                    is_fixed=is_fix,
                    coords=vec_r.coords_norm.copy() if is_fix and vec_r else None,
                    target_area=n.target_area,
                    has_front_door=n.has_front_door,
                    num_exterior_windows=n.num_exterior_windows,
                    front_door_coords=fd_by_room.get(n.id) if is_fix else None,
                    exterior_window_coords=ext_win_by_room.get(n.id) if is_fix else None,
                )
            )

        new_edges: List[ConnectionEdge] = []
        for e in self.edges:
            both_fixed = (e.source in fixed_set) and (e.target in fixed_set)
            key = (min(e.source, e.target), max(e.source, e.target), e.type)
            op_coords = edge_openings.get(key) if both_fixed else None
            new_edges.append(
                ConnectionEdge(
                    source=e.source,
                    target=e.target,
                    type=e.type,
                    count=e.count,
                    opening_coords=op_coords,
                    is_fixed=both_fixed and (op_coords is not None),
                )
            )

        new_bnd = self.boundary
        if fix_boundary and vector_plan.boundary is not None:
            new_bnd = BoundarySpec(
                coords=vector_plan.boundary.coords_norm.copy(),
                num_corners=int(vector_plan.boundary.coords_norm.shape[0]),
                is_fixed=True,
            )

        return FloorplanGraph(
            nodes=new_nodes,
            edges=new_edges,
            boundary=new_bnd,
            coord_frame=vector_plan.coord_frame,
            coords_are_normalized=True,
            plan_id=self.plan_id,
        )


@dataclass
class VectorElement:
    """A single output vector polygon with both denormalized and normalized [-1, 1] coordinates."""

    id: str
    type: str
    coords: np.ndarray
    coords_norm: np.ndarray
    is_fixed: bool = False
    connected_rooms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in ELEM_TYPES:
            raise ValueError(f"Unknown VectorElement.type: {self.type!r}")
        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.coords_norm = np.asarray(self.coords_norm, dtype=np.float32)
        self.connected_rooms = tuple(str(r) for r in self.connected_rooms)

    @property
    def num_corners(self) -> int:
        return int(self.coords.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "num_corners": self.num_corners,
            "is_fixed": bool(self.is_fixed),
            "connected_rooms": list(self.connected_rooms),
            "coords": np.round(self.coords, 4).tolist(),
            "coords_norm": np.round(self.coords_norm, 5).tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorElement":
        return cls(
            id=str(d["id"]),
            type=str(d["type"]),
            coords=np.asarray(d["coords"], dtype=np.float32),
            coords_norm=np.asarray(d["coords_norm"], dtype=np.float32),
            is_fixed=bool(d.get("is_fixed", False)),
            connected_rooms=tuple(d.get("connected_rooms", ())),
        )


@dataclass
class VectorFloorplan:
    """Representation-agnostic vector coordinate output from ConDiffPlan."""

    rooms: List[VectorElement]
    openings: List[VectorElement]
    boundary: Optional[VectorElement]
    edges: List[ConnectionEdge]
    coord_frame: CoordFrame = field(default_factory=CoordFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    plan_id: Optional[Union[int, str]] = None

    @property
    def all_elements(self) -> List[VectorElement]:
        out: List[VectorElement] = []
        if self.boundary is not None:
            out.append(self.boundary)
        out.extend(self.rooms)
        out.extend(self.openings)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "coord_frame": self.coord_frame.to_dict(),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "boundary": self.boundary.to_dict() if self.boundary is not None else None,
            "rooms": [r.to_dict() for r in self.rooms],
            "openings": [o.to_dict() for o in self.openings],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorFloorplan":
        bnd = d.get("boundary")
        return cls(
            rooms=[VectorElement.from_dict(r) for r in d.get("rooms", [])],
            openings=[VectorElement.from_dict(o) for o in d.get("openings", [])],
            boundary=VectorElement.from_dict(bnd) if bnd is not None else None,
            edges=[ConnectionEdge.from_dict(e) for e in d.get("edges", [])],
            coord_frame=CoordFrame.from_dict(d["coord_frame"]) if "coord_frame" in d else CoordFrame(),
            metrics={k: float(v) for k, v in d.get("metrics", {}).items()},
            plan_id=d.get("plan_id"),
        )
