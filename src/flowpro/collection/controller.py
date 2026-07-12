from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
import uuid
import numpy as np

from flowpro.data.types import Frame, TrajectoryPair
from flowpro.data.store import PairStore
from wan_va.action_representation import relative_pose7
from .rollback import RollbackBuffer, RollbackConfig, rollback_target16
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
        self._winner_previous_observation: dict | None = None
        self._prev_b = self._prev_a = False
        self._prev_middle_active = False

    def start_episode(self) -> None:
        """Clear transient collection state before policy control is enabled."""
        self.buffer = RollbackBuffer(self.buffer.config)
        self._loser, self._winner = [], []
        self._winner_previous_observation = None
        self._prev_b = self._prev_a = False
        self._prev_middle_active = False
        self.phase = Phase.POLICY
        state = self.robot.state_action16()
        if hasattr(self.robot, "reset_history"):
            self.robot.reset_history(state)
        self.policy.reset(None)

    def tick(self, controls: InputState) -> Phase:
        b_edge, a_edge = controls.b and not self._prev_b, controls.a and not self._prev_a
        middle_active = controls.middle >= self.threshold
        middle_edge = middle_active and not self._prev_middle_active
        self._prev_b, self._prev_a = controls.b, controls.a
        self._prev_middle_active = middle_active
        if b_edge and self.phase is Phase.POLICY:
            self._loser = self.buffer.segment()
            if not self._loser:
                raise RuntimeError("Cannot rollback before any policy frame was buffered")
            print(
                f"Rollback requested: replaying {len(self._loser)} recorded frames in reverse.",
                flush=True,
            )
            self.phase = Phase.ARMED
            self.buffer.execute(self.robot, self._loser)
            restored = rollback_target16(self._loser[0])
            if hasattr(self.robot, "reset_history"):
                self.robot.reset_history(
                    self._loser[0].action if restored is None else restored
                )
            self.policy.reset(self._loser[0].observation)
            self._winner_previous_observation = self.robot.observe()
            self.phase = Phase.ROLLED_BACK

        if self.phase in (Phase.ROLLED_BACK, Phase.TAKEOVER):
            if self.phase is Phase.TAKEOVER and controls.record:
                self._append_winner_state_transition(self.robot.observe())
            if middle_active:
                if controls.expert_action is None:
                    raise ValueError("middle trigger takeover requires expert_action")
                action = np.asarray(controls.expert_action, dtype=np.float32).reshape(16)
                if middle_edge:
                    begin_takeover = getattr(self.robot, "begin_takeover", None)
                    if callable(begin_takeover):
                        begin_takeover()
                    self._winner_previous_observation = self.robot.observe()
                execute_takeover = getattr(self.robot, "execute_takeover_absolute", None)
                if callable(execute_takeover):
                    execute_takeover(action)
                else:
                    execute_absolute = getattr(self.robot, "execute_absolute", None)
                    if callable(execute_absolute):
                        execute_absolute(action)
                    else:
                        self.robot.execute(action)
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
                self._winner_previous_observation = None
                self.buffer = RollbackBuffer(self.buffer.config)
                end_takeover = getattr(self.robot, "end_takeover", None)
                if callable(end_takeover):
                    end_takeover()
                self.phase = Phase.POLICY
            return self.phase

        obs = self.robot.observe()
        chunk = np.asarray(self.policy.infer(obs), dtype=np.float32)
        chunk = chunk.reshape(-1, 16)
        execute_waypoints = getattr(self.robot, "execute_policy_waypoints", None)
        if callable(execute_waypoints):
            initial_target = self.robot.command_target16()
            targets = np.asarray(execute_waypoints(chunk), np.float32).reshape(-1, 16)
            self.buffer = RollbackBuffer(self.buffer.config)
            references = np.concatenate([initial_target.reshape(1, 16), targets[:-1]], axis=0)
            for action, reference, target in zip(chunk, references, targets):
                frame_obs = dict(obs)
                frame_obs["state_action16"] = reference.copy()
                frame_obs["_flowpro_rollback_target16"] = target.copy()
                self.buffer.append(Frame(frame_obs, action, source="policy"))
        else:
            action = chunk[0]
            self.robot.execute(action)
            if bool(getattr(self.policy, "last_inference_started_chunk", False)):
                self.buffer = RollbackBuffer(self.buffer.config)
            command_target = getattr(self.robot, "command_target16", None)
            if callable(command_target):
                obs = dict(obs)
                obs["_flowpro_rollback_target16"] = command_target()
            self.buffer.append(Frame(obs, action, source="policy"))
        return self.phase

    def _append_winner_state_transition(self, current_observation: dict) -> None:
        previous_observation = self._winner_previous_observation
        if previous_observation is None:
            self._winner_previous_observation = current_observation
            return
        previous = np.asarray(previous_observation["state_action16"], np.float32).reshape(16)
        current = np.asarray(current_observation["state_action16"], np.float32).reshape(16)
        delta = current.copy()
        delta[0:7] = relative_pose7(previous[0:7], current[0:7])
        delta[8:15] = relative_pose7(previous[8:15], current[8:15])
        self._winner.append(Frame(previous_observation, delta, source="human"))
        self._winner_previous_observation = current_observation
