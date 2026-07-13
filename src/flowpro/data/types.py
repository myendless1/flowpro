from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np


@dataclass
class Frame:
    """One robot tick; stored actions use canonical delta EEF semantics."""

    observation: dict[str, Any]
    action: np.ndarray
    timestamp: float = field(default_factory=time.time)
    source: str = "policy"

    def __post_init__(self) -> None:
        self.action = np.asarray(self.action, dtype=np.float32).reshape(16)


@dataclass
class TrajectoryPair:
    pair_id: str
    loser: list[Frame]
    winner: list[Frame]
    rollback_index: int
    round_id: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.loser or not self.winner:
            raise ValueError("A preference pair needs non-empty winner and loser trajectories")
        if self.loser[0].action.shape != (16,) or self.winner[0].action.shape != (16,):
            raise ValueError("Astribot preference actions must be 16-D")


@dataclass
class PreferenceSample:
    observation: dict[str, Any]
    winner: np.ndarray
    loser: np.ndarray
    source: str
    pair_id: str = ""

    def __post_init__(self) -> None:
        self.winner = np.asarray(self.winner, dtype=np.float32)
        self.loser = np.asarray(self.loser, dtype=np.float32)
        if self.winner.shape != self.loser.shape or self.winner.ndim != 2:
            raise ValueError("winner and loser must have equal [H,D] shapes")
