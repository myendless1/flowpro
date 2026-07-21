from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import time
import uuid
import numpy as np

from flowpro.data.types import Frame, TrajectoryPair
from flowpro.data.store import PairStore
from wan_va.action_representation import relative_pose7
from .rollback import RollbackBuffer, RollbackConfig, frame_start16
from .inference_worker import PolicyInferenceWorker
from .execution_worker import ObservationSampler, PolicyExecutionResult, PolicyExecutionWorker
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
    active_arms: dict[str, bool] | None = None
    record: bool = True
    policy_step: bool = True


class InterventionCollector:
    """B triggers rollback/retry, middle teleoperates, and A commits the pair."""

    def __init__(self, robot: RobotIO, policy: Policy, store: PairStore, *,
                 rollback: RollbackConfig | None = None, round_id: int = 1,
                 trigger_threshold: float = .5,
                 async_inference: bool = False,
                 async_execution: bool = False,
                 observation_rate_hz: float = 40.0,
                 policy_waypoint_batch_actions: int = 8) -> None:
        self.robot, self.policy, self.store = robot, policy, store
        self.buffer = RollbackBuffer(rollback)
        self.round_id, self.threshold = round_id, trigger_threshold
        self.phase = Phase.POLICY
        self._loser: list[Frame] = []
        self._winner: list[Frame] = []
        self._winner_previous_observation: dict | None = None
        self._pair_id: str | None = None
        self._stream = None
        self._policy_actions: deque[np.ndarray] = deque()
        self._policy_chunk_size = 0
        self._policy_chunk_index = 0
        self._policy_waypoint_batch_actions = int(policy_waypoint_batch_actions)
        if self._policy_waypoint_batch_actions <= 0:
            raise ValueError("policy_waypoint_batch_actions 必须大于 0")
        self._pending_inference_observation: dict | None = None
        self._chunk_start_observation: dict | None = None
        self._inference_worker = (
            PolicyInferenceWorker(policy) if async_inference else None
        )
        self._execution_worker = (
            PolicyExecutionWorker(robot, batch_size=policy_waypoint_batch_actions)
            if async_execution else None
        )
        self._observation_sampler = (
            ObservationSampler(robot, rate_hz=observation_rate_hz)
            if async_execution else None
        )
        self._failure_requested_at: float | None = None
        self.last_policy_status: str | None = None
        self._prev_b = self._prev_a = False
        self._prev_middle_active = False
        self.last_pair_saved = False
        self.last_pair_discarded = False

    def start_episode(self) -> None:
        """Clear transient collection state before policy control is enabled."""
        self._abort_stream()
        self.buffer = RollbackBuffer(self.buffer.config)
        self._loser, self._winner = [], []
        self._winner_previous_observation = None
        self._pair_id = None
        self._prev_b = self._prev_a = False
        self._prev_middle_active = False
        self.last_pair_saved = False
        self.last_pair_discarded = False
        self._failure_requested_at = None
        self.phase = Phase.POLICY
        state = self.robot.state_action16()
        if hasattr(self.robot, "reset_history"):
            self.robot.reset_history(state)
        if self._execution_worker is not None:
            self._execution_worker.reset()
        if self._observation_sampler is not None:
            self._observation_sampler.start(clear=True)
        self._reset_policy(None)

    def tick(self, controls: InputState) -> Phase:
        self.last_pair_saved = False
        self.last_pair_discarded = False
        self.last_policy_status = None
        b_edge, a_edge = controls.b and not self._prev_b, controls.a and not self._prev_a
        middle_active = controls.middle >= self.threshold
        middle_edge = middle_active and not self._prev_middle_active
        self._prev_b, self._prev_a = controls.b, controls.a
        self._prev_middle_active = middle_active
        if self._execution_worker is not None and self.phase in (Phase.POLICY, Phase.ARMED):
            self._tick_async_policy(b_edge)
            return self.phase

        if b_edge and self.phase is Phase.POLICY:
            candidates = list(self.buffer.frames)
            if not candidates:
                print("尚未执行任何策略动作，暂时无法回退。", flush=True)
                return self.phase
            self._loser = self._fixed_loser(candidates)
            self._perform_rollback(
                self._loser or candidates,
                loser=self._loser,
            )
            return self.phase

        if self.phase in (Phase.ROLLED_BACK, Phase.TAKEOVER):
            if b_edge:
                print(
                    "接管期间按下 B：丢弃当前正负样本，并重新采集本轮。",
                    flush=True,
                )
                self.last_pair_discarded = True
                self._clear_takeover()
                return self.phase
            if self.phase is Phase.TAKEOVER and controls.record:
                self._append_winner_state_transition(self._current_observation())
            if middle_active:
                if controls.expert_action is None:
                    raise ValueError("使用 middle 扳机接管时必须提供 expert_action")
                action = np.asarray(controls.expert_action, dtype=np.float32).reshape(16)
                if middle_edge:
                    begin_takeover = getattr(self.robot, "begin_takeover", None)
                    if callable(begin_takeover):
                        begin_takeover()
                    self._winner_previous_observation = self._current_observation()
                execute_takeover = getattr(self.robot, "execute_takeover_absolute", None)
                if callable(execute_takeover):
                    execute_takeover(action, arm_command_mask=controls.active_arms)
                else:
                    execute_absolute = getattr(self.robot, "execute_absolute", None)
                    if callable(execute_absolute):
                        execute_absolute(action)
                    else:
                        self.robot.execute(action)
                self.phase = Phase.TAKEOVER
            if a_edge:
                if not self._winner or not self._loser:
                    print(
                        "接管动作或固定长度 loser 不完整：丢弃本次回退数据并结束本轮。",
                        flush=True,
                    )
                else:
                    pair = TrajectoryPair(
                        pair_id=self._pair_id or f"r{self.round_id:02d}-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
                        loser=self._loser, winner=self._winner, rollback_index=0,
                        round_id=self.round_id,
                        metadata=self._pair_metadata(),
                    )
                    print("正在保存，请等待...", flush=True)
                    if self._stream is None:
                        self.store.save(pair)
                    else:
                        stream = self._stream
                        try:
                            stream.commit()
                        except Exception as exc:
                            print(f"警告：后台流式保存失败，正在改用同步保存：{exc}", flush=True)
                            stream.abort()
                            self.store.save(pair)
                        finally:
                            self._stream = None
                    self.last_pair_saved = True
                    print("保存完成。", flush=True)
                self._clear_takeover()
            return self.phase

        self._advance_policy(controls.policy_step)
        return self.phase

    def _clear_takeover(self) -> None:
        self._abort_stream()
        self._loser, self._winner = [], []
        self._winner_previous_observation = None
        self._pair_id = None
        self._failure_requested_at = None
        self.buffer = RollbackBuffer(self.buffer.config)
        end_takeover = getattr(self.robot, "end_takeover", None)
        if callable(end_takeover):
            end_takeover()
        self.phase = Phase.POLICY

    def close(self) -> None:
        """Stop any background writer without publishing an incomplete pair."""
        self._abort_stream()
        if self._execution_worker is not None:
            self._execution_worker.close()
        if self._observation_sampler is not None:
            self._observation_sampler.close()
        if self._inference_worker is not None:
            self._inference_worker.close()

    def _tick_async_policy(self, b_edge: bool) -> None:
        result = self._execution_worker.poll()
        if result is not None:
            frames = self._frames_from_execution(result)
            previous_frames = list(self.buffer.frames)
            for frame in frames:
                self.buffer.append(frame)
            if self._failure_requested_at is not None:
                before_failure = [
                    frame
                    for frame, start_time in zip(frames, result.action_start_times)
                    if float(start_time) <= self._failure_requested_at
                ]
                after_failure = [
                    frame
                    for frame, start_time in zip(frames, result.action_start_times)
                    if float(start_time) > self._failure_requested_at
                ]
                candidates = previous_frames + before_failure
                self._loser = self._fixed_loser(candidates)
                if not candidates:
                    print("B 按下时尚未开始执行策略动作，本次不回退。", flush=True)
                    self._failure_requested_at = None
                    self.phase = Phase.POLICY
                    return
                rollback_frames = (self._loser or candidates) + after_failure
                self._failure_requested_at = None
                self._perform_rollback(rollback_frames, loser=self._loser)
                return
            if b_edge:
                self._clear_pending_policy_chunk()
                candidates = list(self.buffer.frames)
                if not candidates:
                    print("尚未执行任何策略动作，暂时无法回退。", flush=True)
                    return
                self._loser = self._fixed_loser(candidates)
                self._perform_rollback(
                    self._loser or candidates,
                    loser=self._loser,
                )
                return
            if result.last_in_chunk:
                self._clear_pending_policy_chunk()
                self.last_policy_status = "chunk_finished"
            else:
                self.last_policy_status = "waypoint_batch_finished"
            return

        if self.phase is Phase.ARMED:
            return

        if b_edge:
            if self._execution_worker.pending:
                self._failure_requested_at = time.time()
                self.phase = Phase.ARMED
                self.last_policy_status = "rollback_deferred"
                self._clear_pending_policy_chunk()
                if self._inference_worker is not None:
                    self._inference_worker.reset(None)
                    self._pending_inference_observation = None
                return
            self._clear_pending_policy_chunk()
            candidates = list(self.buffer.frames)
            if not candidates:
                print("尚未执行任何策略动作，暂时无法回退。", flush=True)
                return
            self._loser = self._fixed_loser(candidates)
            self._perform_rollback(
                self._loser or candidates,
                loser=self._loser,
            )
            return

        if self._execution_worker.pending:
            return

        if self._policy_actions:
            self._submit_next_waypoint_batch()
            return

        chunk = None
        if self._inference_worker is None:
            observation = self._current_observation()
            chunk = np.asarray(self.policy.infer(observation), np.float32).reshape(-1, 16)
        else:
            chunk = self._inference_worker.poll()
            if chunk is None and not self._inference_worker.pending:
                observation = self._current_observation(wait_s=0.1)
                if observation is not None and self._inference_worker.request(observation):
                    self._pending_inference_observation = observation
                    self.last_policy_status = "inference_started"
                return

        if chunk is not None:
            self._accept_policy_chunk(chunk, self._pending_inference_observation)
            self._pending_inference_observation = None
            self._submit_next_waypoint_batch()

    def _submit_next_waypoint_batch(self) -> None:
        if not self._policy_actions or self._execution_worker.pending:
            return
        count = min(
            self._policy_waypoint_batch_actions,
            len(self._policy_actions),
        )
        batch = np.stack(list(self._policy_actions)[:count]).astype(np.float32, copy=False)
        first_in_chunk = self._policy_chunk_index == 0
        last_in_chunk = count == len(self._policy_actions)
        if not self._execution_worker.request(
            batch,
            first_in_chunk=first_in_chunk,
            last_in_chunk=last_in_chunk,
        ):
            return
        for _ in range(count):
            self._policy_actions.popleft()
        self._policy_chunk_index += count
        self.last_policy_status = (
            "chunk_started" if first_in_chunk else "waypoint_batch_started"
        )

    def _clear_pending_policy_chunk(self) -> None:
        self._policy_actions.clear()
        self._policy_chunk_size = 0
        self._policy_chunk_index = 0
        self._pending_inference_observation = None
        self._chunk_start_observation = None

    def _fixed_loser(self, candidates: list[Frame]) -> list[Frame]:
        horizon = max(1, int(self.buffer.config.default_horizon))
        if len(candidates) < horizon:
            print(
                f"策略历史只有 {len(candidates)} 步，少于固定 loser 长度 {horizon}；"
                "本次只执行物理回退，不保存偏好对。",
                flush=True,
            )
            return []
        return list(candidates[-horizon:])

    def _frames_from_execution(self, result: PolicyExecutionResult) -> list[Frame]:
        frames = []
        for action, target, start_target, started_at, arrived_at in zip(
            result.actions,
            result.targets,
            result.start_targets,
            result.action_start_times,
            result.action_arrival_times,
        ):
            observation = self._observation_sampler.latest_at_or_before(float(started_at))
            measured_state = np.asarray(observation["state_action16"], np.float32).reshape(16)
            frame_observation = dict(observation)
            timing = dict(frame_observation.get("_flowpro_timing", {}))
            timing["action_start_timestamp"] = float(started_at)
            timing["action_arrival_timestamp"] = float(arrived_at)
            frame_observation["_flowpro_timing"] = timing
            frame_observation["_flowpro_rollback_start16"] = np.asarray(
                start_target, np.float32
            ).copy()
            frame_observation["_flowpro_rollback_target16"] = np.asarray(
                target, np.float32
            ).copy()
            frames.append(
                Frame(
                    frame_observation,
                    self._delta_between(measured_state, target),
                    timestamp=float(frame_observation.get("time", started_at)),
                    source="policy",
                )
            )
        return frames

    def _perform_rollback(
        self,
        rollback_frames: list[Frame],
        *,
        loser: list[Frame] | None = None,
    ) -> None:
        self._loser = list(rollback_frames if loser is None else loser)
        self._pair_id = f"r{self.round_id:02d}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        if self._loser:
            try:
                self._stream = self.store.begin_stream(
                    pair_id=self._pair_id,
                    loser=self._loser,
                    rollback_index=0,
                    round_id=self.round_id,
                    metadata=self._pair_metadata(),
                )
            except Exception as exc:
                self._stream = None
                print(f"警告：无法启动后台流式保存，将在按 A 后同步保存：{exc}", flush=True)
        else:
            self._stream = None
        print(
            f"收到回退请求：负样本 {len(self._loser)} 步，"
            f"正在反向执行 {len(rollback_frames)} 步策略动作。",
            flush=True,
        )
        self.phase = Phase.ARMED
        try:
            self.buffer.execute(self.robot, rollback_frames)
        except Exception:
            self._abort_stream()
            raise
        restored_observation = self.robot.observe()
        restored = np.asarray(restored_observation["state_action16"], np.float32).reshape(16)
        if hasattr(self.robot, "reset_history"):
            self.robot.reset_history(restored)
        self._reset_policy(restored_observation)
        self._winner_previous_observation = restored_observation
        self.phase = Phase.ROLLED_BACK

    def _current_observation(self, *, wait_s: float = 0.0) -> dict | None:
        if self._observation_sampler is None:
            return self.robot.observe()
        observation = self._observation_sampler.latest(wait_s=wait_s)
        return observation

    def _abort_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.abort()

    def _pair_metadata(self) -> dict[str, str]:
        return {
            "control": "B rollback, middle takeover, A finish",
            "policy_action_representation": str(
                getattr(self.policy, "action_representation", "delta")
            ),
            "stored_action_representation": "delta",
            "history_action_representation": "absolute",
        }

    def _reset_policy(self, observation: dict | None) -> None:
        self._clear_pending_policy_chunk()
        if self._inference_worker is None:
            self.policy.reset(observation)
        else:
            self._inference_worker.reset(observation)

    def _advance_policy(self, execute_step: bool) -> None:
        if self._inference_worker is not None:
            chunk = self._inference_worker.poll()
            if chunk is not None:
                self._accept_policy_chunk(chunk, self._pending_inference_observation)
                self._pending_inference_observation = None

        if not self._policy_actions:
            if self._inference_worker is None:
                observation = self.robot.observe()
                chunk = np.asarray(self.policy.infer(observation), np.float32).reshape(-1, 16)
                self._accept_policy_chunk(chunk, observation)
            elif not self._inference_worker.pending:
                observation = self.robot.observe()
                if self._inference_worker.request(observation):
                    self._pending_inference_observation = observation
                    self.last_policy_status = "inference_started"
            if not self._policy_actions:
                return

        if not execute_step:
            return

        index = self._policy_chunk_index
        action = self._policy_actions.popleft()
        first_in_chunk = index == 0
        last_in_chunk = not self._policy_actions
        observation = (
            self._chunk_start_observation
            if first_in_chunk and self._chunk_start_observation is not None
            else self.robot.observe()
        )
        if first_in_chunk:
            self._chunk_start_observation = None
        measured_state = np.asarray(
            observation["state_action16"], np.float32
        ).reshape(16)
        command_target = getattr(self.robot, "command_target16", None)
        command_start_target = (
            np.asarray(command_target(), np.float32).reshape(16)
            if callable(command_target)
            else measured_state.copy()
        )

        execute_policy_step = getattr(self.robot, "execute_policy_step", None)
        if callable(execute_policy_step):
            target = np.asarray(
                execute_policy_step(
                    action,
                    first_in_chunk=first_in_chunk,
                    last_in_chunk=last_in_chunk,
                ),
                np.float32,
            ).reshape(16)
        else:
            self.robot.execute(action)
            target = (
                np.asarray(command_target(), np.float32).reshape(16)
                if callable(command_target)
                else np.asarray(self.robot.state_action16(), np.float32).reshape(16)
            )

        frame_observation = dict(observation)
        frame_observation["_flowpro_rollback_start16"] = command_start_target.copy()
        frame_observation["_flowpro_rollback_target16"] = target.copy()
        timestamp = float(frame_observation.get("time", time.time()))
        self.buffer.append(
            Frame(
                frame_observation,
                self._delta_between(measured_state, target),
                timestamp=timestamp,
                source="policy",
            )
        )
        self._policy_chunk_index += 1
        self.last_policy_status = "chunk_finished" if last_in_chunk else "action_executed"

    def _accept_policy_chunk(
        self,
        chunk: np.ndarray,
        observation: dict | None,
    ) -> None:
        actions = np.asarray(chunk, np.float32).reshape(-1, 16)
        if not len(actions):
            raise ValueError("策略推理返回了空 action chunk")
        self._policy_actions = deque(action.copy() for action in actions)
        self._policy_chunk_size = len(actions)
        self._policy_chunk_index = 0
        self._chunk_start_observation = observation
        self.last_policy_status = "chunk_ready"

    @staticmethod
    def _delta_between(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, np.float32).reshape(16)
        target = np.asarray(target, np.float32).reshape(16)
        delta = target.copy()
        delta[0:7] = relative_pose7(reference[0:7], target[0:7])
        delta[8:15] = relative_pose7(reference[8:15], target[8:15])
        return delta

    def _append_winner_state_transition(self, current_observation: dict) -> None:
        previous_observation = self._winner_previous_observation
        if previous_observation is None:
            self._winner_previous_observation = current_observation
            return
        previous = np.asarray(previous_observation["state_action16"], np.float32).reshape(16)
        current = np.asarray(current_observation["state_action16"], np.float32).reshape(16)
        delta = self._delta_between(previous, current)
        frame = Frame(previous_observation, delta, source="human")
        self._winner.append(frame)
        if self._stream is not None:
            self._stream.append_winner(frame)
        self._winner_previous_observation = current_observation
