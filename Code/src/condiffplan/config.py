"""Vocabularies and configuration dataclasses for ConDiffPlan on ResPlan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Strict 7-class room vocabulary allowed in FloorplanGraph.nodes
ROOM_TYPES: tuple[str, ...] = (
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "storage",
    "stair",
)
ROOM_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(ROOM_TYPES)}

# Internal 11-class geometric element vocabulary used by the diffusion Transformer
ELEM_TYPES: tuple[str, ...] = (
    "boundary",    # 0 (ResPlan 'inner')
    "living",      # 1
    "bedroom",     # 2
    "bathroom",    # 3
    "kitchen",     # 4
    "balcony",     # 5
    "storage",     # 6
    "stair",       # 7
    "front_door",  # 8 (instantiated from room.has_front_door)
    "door",        # 9 (instantiated from via_door edge)
    "window",      # 10 (instantiated from via_window edge / room.num_exterior_windows)
)
ELEM_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(ELEM_TYPES)}

# Allowed room-to-room connection edge types in the input bubble diagram
GRAPH_EDGE_TYPES: tuple[str, ...] = (
    "via_door",
    "adjacency",
    "via_window",
)

# Internal 8-class relational attention edge vocabulary
REL_EDGE_TYPES: tuple[str, ...] = (
    "none",            # 0 (no direct link / self)
    "via_door",        # 1 (room <-> room connected via door)
    "adjacency",       # 2 (room <-> room open passage / adjacency)
    "via_window",      # 3 (room <-> room connected via window)
    "direct",          # 4 (front_door <-> entry room)
    "door_to_room",    # 5 (door polygon <-> connected rooms)
    "window_to_room",  # 6 (window polygon <-> connected room(s))
    "in_boundary",     # 7 (room / opening <-> house boundary)
)
REL_EDGE_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(REL_EDGE_TYPES)}

# Default corner count fallbacks per element type if corner_stats.json is unavailable
DEFAULT_CORNER_COUNTS: dict[str, int] = {
    "boundary": 12,
    "living": 12,
    "bedroom": 4,
    "bathroom": 4,
    "kitchen": 6,
    "balcony": 4,
    "storage": 4,
    "stair": 6,
    "front_door": 4,
    "door": 4,
    "window": 4,
}


@dataclass
class DataConfig:
    """Dataset preprocessing and heuristic conditioning parameters."""

    data_path: str = "../Dataset/ResPlan.pkl"
    split_path: str = "../Dataset/split.json"
    cache_dir: str = "data_cache"
    simplify_tolerance: float = 1.0
    max_elements: int = 64
    max_corners_per_elem: int = 64
    wall_aug_points: int = 8
    norm_margin: float = 0.95
    include_augmented_split: bool = False
    p_fix_boundary: float = 0.5
    p_fix_rooms: float = 0.5
    p_condition_area: float = 0.5
    random_dihedral: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """ConDiffPlan Transformer backbone hyperparameters (with HouseDiffusion discrete branch)."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    num_elem_types: int = len(ELEM_TYPES)
    max_elements: int = 64
    max_corners: int = 64
    num_rel_edge_types: int = len(REL_EDGE_TYPES)
    wall_aug_points: int = 8
    enable_discrete: bool = True
    discrete_num_bits: int = 8
    discrete_n_layers: int = 2
    grad_checkpointing: bool = False

    @property
    def token_input_dim(self) -> int:
        # 2*L wall points + num_elem_types + max_elements + max_corners + fixed_flag(1) + norm_area(1) + has_area(1)
        return (
            2 * self.wall_aug_points
            + self.num_elem_types
            + self.max_elements
            + self.max_corners
            + 3
        )

    @property
    def discrete_out_dim(self) -> int:
        return 2 * self.discrete_num_bits

    @property
    def discrete_token_input_dim(self) -> int:
        return self.token_input_dim + self.discrete_out_dim

    @classmethod
    def preset(cls, name: str = "default") -> "ModelConfig":
        if name == "default":
            return cls(d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, discrete_n_layers=2)
        if name == "small":
            return cls(d_model=256, n_heads=8, n_layers=4, d_ff=1024, dropout=0.1, discrete_n_layers=2)
        if name == "tiny":
            return cls(d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.0, discrete_n_layers=1)
        raise ValueError(f"Unknown ModelConfig preset: {name!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffusionConfig:
    """Forward and reverse diffusion schedule configuration."""

    num_timesteps: int = 1000
    schedule: str = "cosine"
    cosine_s: float = 0.008
    clip_denoised: bool = True
    clip_range: float = 1.0
    discrete_train_t: int = 20
    discrete_sample_t: int = 32
    discrete_loss_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training loop hyperparameters (matching ConDiffPlan section 4.2)."""

    batch_size: int = 64
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    total_steps: int = 500_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    amp: bool = True
    precision: str = "auto"
    micro_batch_size: Optional[int] = None
    grad_checkpointing: bool = False
    compile_model: bool = False
    bucket_batching: bool = True
    num_workers: int = -1
    log_every: int = 100
    val_every: int = 2_000
    save_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
