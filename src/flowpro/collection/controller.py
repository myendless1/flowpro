from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
import uuid
import numpy as np

from flowpro.data.types import Frame, TrajectoryPair
from flowpro.data.store import PairStore
from .rollback import RollbackBuffer, RollbackConfig, observation_state16
from .protocol import Policy, RobotIO


class Phase(Enum):
    POLICY = auto()
    ARMED = auto()
    ROLLED_BACK = auto()
    TAKEOVER = auto()


@dataclass
class InputState:
    b: bool = False
    a: bool = False
    middle: float = 0.0
    expert_action: np.ndarray | None = None
    record: bool = True


class InterventionCollector:
    """B -> rollback; hold middle -> teleoperate/record; A -> commit pair."""

    def __init__(self, robot: RobotIO, policy: Policy, store: PairStore, *,
                 rollback: RollbackConfig | None = None, round_id: int = 1,
                 trigger_threshold: float = .5) -> None:
        self.robot, self.policy, self.store = robot, policy, store
        self.buffer = RollbackBuffer(rollback)
        self.round_id, self.threshold = round_id, trigger_threshold
        self.phase = Phase.POLICY
        self._loser: list[Frame] = []
        self._winner: list[Frame] = []
        self._prev_b = self._prev_a = False

    def tick(self, controls: InputState) -> Phase:
        b_edge, a_edge = controls.b and not self._prev_b, controls.a and not self._prev_a
        self._prev_b, self._prev_a = controls.b, controls.a
        if b_edge and self.phase is Phase.POLICY:
            self._loser = self.buffer.segment()
            if not self._loser:
                raise RuntimeError("Cannot rollback before any policy frame was buffered")
            self.phase = Phase.ARMED
            self.buffer.execute(self.robot, self._loser)
            restored = observation_state16(self._loser[0])
            if hasattr(self.robot, "reset_history"):
                self.robot.reset_history(
                    self._loser[0].action if restored is None else restored
                )
            self.policy.reset(self._loser[0].observation)
            self.phase = Phase.ROLLED_BACK

        if self.phase in (Phase.ROLLED_BACK, Phase.TAKEOVER):
            if controls.middle >= self.threshold:
                if controls.expert_action is None:
                    raise ValueError("middle trigger takeover requires expert_action")
                action = np.asarray(controls.expert_action, dtype=np.float32).reshape(16)
                obs = self.robot.observe() if controls.record else None
                self.robot.execute(action)
                if controls.record:
                    assert obs is not None
                    self._winner.append(Frame(obs, action, source="human"))
                self.phase = Phase.TAKEOVER
            if a_edge:
                if not self._winner:
                    raise RuntimeError("A cannot finish before a middle-trigger correction is recorded")
                pair = TrajectoryPair(
                    pair_id=f"r{self.round_id:02d}-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
                    loser=self._loser, winner=self._winner, rollback_index=0,
                    round_id=self.round_id,
                    metadata={"control": "B rollback, middle takeover, A finish"},
                )
                self.store.save(pair)
                self._loser, self._winner = [], []
                self.buffer = RollbackBuffer(self.buffer.config)
                self.phase = Phase.POLICY
            return self.phase

        obs = self.robot.observe()
        chunk = np.asarray(self.policy.infer(obs), dtype=np.float32)
        action = chunk[0] if chunk.ndim == 2 else chunk
        self.robot.execute(action)
        self.buffer.append(Frame(obs, action, source="policy"))
        return self.phase
