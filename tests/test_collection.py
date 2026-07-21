import numpy as np
import pytest
import time
from flowpro.collection import InputState, InterventionCollector, Phase
from flowpro.cli.collect import (
    _gate_takeover_retry_b,
    _resume_progress,
    _wait_for_a_reset,
    _wait_for_a_start,
)
from flowpro.collection.rollback import RollbackConfig
from flowpro.collection.execution_worker import ObservationSampler
from flowpro.data import Frame, PairStore
from wan_va.action_representation import apply_relative_pose7


class Robot:
    def __init__(self): self.action = np.r_[np.zeros(3), [1,0,0,0], 0, [0.4,0,0], [1,0,0,0], 0].astype(np.float32); self.n=0; self.reset_count=0
    def observe(self):
        self.n += 1
        return {
            "step": self.n,
            "time": self.n / 10.0,
            "state_action16": self.action.copy(),
        }
    def state_action16(self): return self.action.copy()
    def command_target16(self): return self.action.copy()
    def execute(self, action):
        delta = np.asarray(action).copy(); target = self.action.copy()
        target[0:7] = apply_relative_pose7(target[0:7], delta[0:7]); target[7] = delta[7]
        target[8:15] = apply_relative_pose7(target[8:15], delta[8:15]); target[15] = delta[15]
        self.action = target
    def execute_absolute(self, action): self.action = np.asarray(action).copy()
    def reset_history(self, action): self.action = np.asarray(action).copy(); self.reset_count += 1


class Policy:
    def __init__(self): self.reset_count = 0
    def reset(self, observation): self.reset_count += 1
    def infer(self, observation):
        delta = np.zeros(16, np.float32); delta[[3, 11]] = 1
        state = observation["state_action16"]; delta[[7, 15]] = state[[7, 15]]
        return np.tile(delta, (2,1))


def test_reset_gate_requires_a_release_before_new_press(capsys):
    class Controls:
        def __init__(self):
            self.states = iter([True, True, False, True])
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return InputState(a=next(self.states))

    controls = Controls()

    assert _wait_for_a_reset(controls, lambda: False, 0.0, 3) is True
    assert controls.poll_count == 4
    assert "按 A 抬起双臂" in capsys.readouterr().out


def test_policy_start_gate_requires_a_release_before_new_press(capsys):
    class Controls:
        def __init__(self):
            self.states = iter([True, False, True])
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return InputState(a=next(self.states))

    controls = Controls()

    assert _wait_for_a_start(controls, lambda: False, 0.0, 2) is True
    assert controls.poll_count == 3
    assert "按 A 开始策略推理" in capsys.readouterr().out


def test_takeover_retry_b_requires_release_after_rollback_press():
    held = InputState(b=True)
    armed = _gate_takeover_retry_b(held, False)
    assert armed is False
    assert held.b is False

    released = InputState(b=False)
    armed = _gate_takeover_retry_b(released, armed)
    assert armed is True

    pressed_again = InputState(b=True)
    assert _gate_takeover_retry_b(pressed_again, armed) is True
    assert pressed_again.b is True


def test_resume_uses_existing_pairs_and_total_target(tmp_path):
    class Store:
        def completed_count(self):
            return 7

    assert _resume_progress(Store(), 10) == (7, 8, False)
    assert _resume_progress(Store(), 7) == (7, 8, True)
    assert _resume_progress(Store(), 0) == (7, 8, False)


def test_observation_alignment_uses_latest_causal_frame():
    import threading
    from collections import deque

    sampler = ObservationSampler.__new__(ObservationSampler)
    sampler._lock = threading.Lock()
    sampler._samples = deque([
        (1.0, {"time": 1.0, "state_action16": np.zeros(16)}),
        (1.2, {"time": 1.2, "state_action16": np.ones(16)}),
    ])

    observation = sampler.latest_at_or_before(1.1)

    assert observation["time"] == 1.0
    assert observation["_flowpro_timing"]["action_observation_offset_s"] == pytest.approx(-0.1)


def test_observation_alignment_rejects_future_only_frame():
    import threading
    from collections import deque

    sampler = ObservationSampler.__new__(ObservationSampler)
    sampler._lock = threading.Lock()
    sampler._samples = deque([
        (1.2, {"time": 1.2, "state_action16": np.ones(16)}),
    ])

    with pytest.raises(RuntimeError, match="没有 action 开始前"):
        sampler.latest_at_or_before(1.1)


def test_b_middle_a_commits_atomic_pair_with_save_progress(tmp_path, capsys):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=2))
    c.tick(InputState()); c.tick(InputState())
    assert c.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert c.tick(InputState(middle=1, expert_action=robot.action)) is Phase.TAKEOVER
    assert c.tick(InputState(a=True)) is Phase.POLICY
    assert c.last_pair_saved is True
    output = capsys.readouterr().out
    saving = output.index("正在保存，请等待...")
    saved = output.index("保存完成。")
    assert saving < saved
    target = next(tmp_path.iterdir())
    assert (target / "trajectories.h5").exists()
    pair = PairStore(tmp_path).load(target)
    assert pair.metadata["stored_action_representation"] == "delta"
    assert pair.metadata["history_action_representation"] == "absolute"


def test_a_before_takeover_discards_rollback_without_saving_pair(tmp_path, capsys):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1))
    c.tick(InputState()); c.tick(InputState(b=True)); c.tick(InputState())

    assert c.tick(InputState(a=True)) is Phase.POLICY
    assert c.last_pair_saved is False
    assert list(tmp_path.iterdir()) == []
    assert "接管动作或固定长度 loser 不完整" in capsys.readouterr().out


def test_stream_failure_falls_back_to_legacy_atomic_save(tmp_path, capsys):
    class FailedStream:
        def append_winner(self, frame):
            pass

        def commit(self):
            raise RuntimeError("disk writer failed")

        def abort(self):
            pass

    class FallbackStore(PairStore):
        def begin_stream(self, **kwargs):
            return FailedStream()

    robot = Robot()
    collector = InterventionCollector(
        robot,
        Policy(),
        FallbackStore(tmp_path),
        rollback=RollbackConfig(default_horizon=1),
    )
    collector.tick(InputState())
    collector.tick(InputState(b=True))
    collector.tick(InputState())
    collector.tick(InputState(middle=1, expert_action=robot.action))

    assert collector.tick(InputState(a=True)) is Phase.POLICY
    assert collector.last_pair_saved is True
    target = next(path for path in tmp_path.iterdir() if not path.name.startswith("."))
    assert (target / "winner.npz").is_file()
    assert "后台流式保存失败，正在改用同步保存" in capsys.readouterr().out


def test_b_during_takeover_discards_pair_and_requests_episode_retry(tmp_path, capsys):
    robot = Robot()
    collector = InterventionCollector(
        robot,
        Policy(),
        PairStore(tmp_path),
        rollback=RollbackConfig(default_horizon=1),
    )
    collector.tick(InputState())
    collector.tick(InputState(b=True))
    collector.tick(InputState())
    collector.tick(InputState(middle=1, expert_action=robot.action))

    assert collector.tick(InputState(b=True)) is Phase.POLICY
    assert collector.last_pair_discarded is True
    assert collector.last_pair_saved is False
    assert len(collector.buffer.frames) == 0
    assert list(tmp_path.iterdir()) == []
    assert "丢弃当前正负样本" in capsys.readouterr().out


def test_start_episode_clears_state_and_resets_runtime(tmp_path):
    robot = Robot(); policy = Policy()
    c = InterventionCollector(robot, policy, PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1))
    c.tick(InputState())
    c.start_episode()
    assert c.phase is Phase.POLICY
    assert len(c.buffer.frames) == 0
    assert robot.reset_count == 1
    assert policy.reset_count == 1


def test_rollback_buffer_keeps_two_policy_chunks_as_loser_frames(tmp_path):
    c = InterventionCollector(
        Robot(),
        Policy(),
        PairStore(tmp_path),
        rollback=RollbackConfig(default_horizon=4),
    )
    c.tick(InputState())
    assert len(c.buffer.frames) == 1
    for _ in range(3):
        c.tick(InputState())
    assert len(c.buffer.frames) == 4
    assert len(c.buffer.segment()) == 4
    assert [frame.observation["step"] for frame in c.buffer.frames] == [1, 2, 3, 4]
    assert c.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert len(c._loser) == 4
    c.close()


def test_default_rollback_horizon_is_two_32_step_chunks():
    assert RollbackConfig().default_horizon == 64
    assert RollbackConfig().capacity == 72


def test_policy_continues_across_chunks_and_selects_last_64_negative_frames(tmp_path):
    robot = Robot()
    collector = InterventionCollector(robot, Policy(), PairStore(tmp_path))

    for _ in range(70):
        assert collector.tick(InputState()) is Phase.POLICY

    assert len(collector.buffer.frames) == 70
    steps = [frame.observation["step"] for frame in collector.buffer.frames]
    assert steps == list(range(1, 71))
    np.testing.assert_allclose(
        np.diff([frame.timestamp for frame in collector.buffer.frames]),
        0.1,
        atol=1e-7,
    )
    assert collector.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert [frame.observation["step"] for frame in collector._loser] == list(range(7, 71))
    collector.close()


def test_b_finishes_current_waypoint_batch_and_rolls_back_at_most_72_frames(tmp_path):
    import threading

    class Chunk32Policy(Policy):
        def infer(self, observation):
            delta = np.zeros(16, np.float32)
            delta[[3, 11]] = 1
            return np.tile(delta, (32, 1))

    class BatchRobot(Robot):
        def __init__(self):
            super().__init__()
            self.execution_started = threading.Event()
            self.release_execution = threading.Event()
            self.rollback_targets = None
            self.policy_batches = []

        def observe(self):
            self.n += 1
            return {
                "step": self.n,
                "time": time.time(),
                "state_action16": self.action.copy(),
            }

        def execute_policy_waypoint_batch(
            self,
            actions,
            *,
            first_in_chunk,
            last_in_chunk,
        ):
            self.policy_batches.append(np.asarray(actions, np.float32).copy())
            starts = []
            targets = []
            for action in np.asarray(actions, np.float32):
                starts.append(self.action.copy())
                self.execute(action)
                targets.append(self.action.copy())
            # Put all starts after the B timestamp so the current 8-action
            # waypoint batch is rollback-only and excluded from loser data.
            first_start = time.time() + 0.2
            action_starts = first_start + np.arange(len(actions)) * 0.1
            self.execution_started.set()
            self.release_execution.wait(timeout=2.0)
            return {
                "targets": np.asarray(targets, np.float32),
                "start_targets": np.asarray(starts, np.float32),
                "action_start_times": action_starts,
                "action_arrival_times": action_starts + 0.1,
                "finished_at": time.time(),
            }

        def execute_rollback_waypoints(self, targets, *, step_duration_s):
            self.rollback_targets = np.asarray(targets, np.float32).copy()
            self.action = self.rollback_targets[-1].copy()

    robot = BatchRobot()
    collector = InterventionCollector(
        robot,
        Chunk32Policy(),
        PairStore(tmp_path),
        rollback=RollbackConfig(capacity=72, default_horizon=64),
        async_execution=True,
        observation_rate_hz=100,
    )
    collector.start_episode()
    pose = robot.action.copy()
    for index in range(64):
        start = pose.copy()
        target = pose.copy()
        target[8] += (index + 1) * 1e-4
        collector.buffer.append(
            Frame(
                {
                    "step": index,
                    "time": time.time() - 1 + index * 0.001,
                    "state_action16": start,
                    "_flowpro_rollback_start16": start,
                    "_flowpro_rollback_target16": target,
                },
                np.zeros(16, np.float32),
            )
        )

    collector.tick(InputState())
    assert robot.execution_started.wait(timeout=1.0)
    assert collector.tick(InputState(b=True)) is Phase.ARMED
    robot.release_execution.set()
    deadline = time.monotonic() + 2.0
    while collector.phase is Phase.ARMED and time.monotonic() < deadline:
        collector.tick(InputState())
        time.sleep(0.005)

    assert collector.phase is Phase.ROLLED_BACK
    assert len(collector._loser) == 64
    assert len(robot.policy_batches) == 1
    assert robot.policy_batches[0].shape == (8, 16)
    assert robot.rollback_targets.shape == (72, 16)
    collector.close()


def test_async_execution_submits_32_actions_as_four_independent_batches(tmp_path):
    class Chunk32Policy(Policy):
        def infer(self, observation):
            delta = np.zeros(16, np.float32)
            delta[[3, 11]] = 1
            return np.tile(delta, (32, 1))

    class BatchRobot(Robot):
        def __init__(self):
            super().__init__()
            self.batch_calls = []

        def observe(self):
            self.n += 1
            return {
                "step": self.n,
                "time": time.time(),
                "state_action16": self.action.copy(),
            }

        def execute_policy_waypoint_batch(
            self,
            actions,
            *,
            first_in_chunk,
            last_in_chunk,
        ):
            actions = np.asarray(actions, np.float32)
            self.batch_calls.append((len(actions), first_in_chunk, last_in_chunk))
            starts = []
            targets = []
            started_at = time.time() + 0.01
            for action in actions:
                starts.append(self.action.copy())
                self.execute(action)
                targets.append(self.action.copy())
            offsets = np.arange(len(actions), dtype=np.float64) * 0.1
            return {
                "targets": np.asarray(targets, np.float32),
                "start_targets": np.asarray(starts, np.float32),
                "action_start_times": started_at + offsets,
                "action_arrival_times": started_at + offsets + 0.1,
                "finished_at": time.time(),
            }

    robot = BatchRobot()
    collector = InterventionCollector(
        robot,
        Chunk32Policy(),
        PairStore(tmp_path),
        async_execution=True,
        observation_rate_hz=200,
        policy_waypoint_batch_actions=8,
    )
    collector.start_episode()
    deadline = time.monotonic() + 2.0
    while len(collector.buffer.frames) < 32 and time.monotonic() < deadline:
        collector.tick(InputState())
        time.sleep(0.005)

    assert robot.batch_calls == [
        (8, True, False),
        (8, False, False),
        (8, False, False),
        (8, False, True),
    ]
    assert len(collector.buffer.frames) == 32
    collector.close()


def test_b_is_not_lost_when_waypoint_result_arrives_on_same_tick(tmp_path):
    class ChunkPolicy(Policy):
        def infer(self, observation):
            delta = np.zeros(16, np.float32)
            delta[[3, 11]] = 1
            return np.tile(delta, (16, 1))

    class BatchRobot(Robot):
        def __init__(self):
            super().__init__()
            self.rollback_targets = None
            self.batch_calls = 0

        def observe(self):
            self.n += 1
            return {
                "step": self.n,
                "time": time.time(),
                "state_action16": self.action.copy(),
            }

        def execute_policy_waypoint_batch(
            self,
            actions,
            *,
            first_in_chunk,
            last_in_chunk,
        ):
            self.batch_calls += 1
            starts = []
            targets = []
            started_at = time.time()
            for action in np.asarray(actions, np.float32):
                starts.append(self.action.copy())
                self.execute(action)
                targets.append(self.action.copy())
            offsets = np.arange(len(actions), dtype=np.float64) * 0.1
            return {
                "targets": np.asarray(targets, np.float32),
                "start_targets": np.asarray(starts, np.float32),
                "action_start_times": started_at + offsets,
                "action_arrival_times": started_at + offsets + 0.1,
                "finished_at": time.time(),
            }

        def execute_rollback_waypoints(self, targets, *, step_duration_s):
            self.rollback_targets = np.asarray(targets, np.float32).copy()
            self.action = self.rollback_targets[-1].copy()

    robot = BatchRobot()
    collector = InterventionCollector(
        robot,
        ChunkPolicy(),
        PairStore(tmp_path),
        rollback=RollbackConfig(capacity=72, default_horizon=64),
        async_execution=True,
        observation_rate_hz=200,
    )
    collector.start_episode()
    collector.tick(InputState())
    deadline = time.monotonic() + 1.0
    while collector._execution_worker._results.empty() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert collector.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert robot.batch_calls == 1
    assert robot.rollback_targets is not None
    collector.close()


def test_async_inference_does_not_block_control_tick(tmp_path):
    class SlowPolicy(Policy):
        def infer(self, observation):
            time.sleep(0.15)
            return super().infer(observation)

    collector = InterventionCollector(
        Robot(),
        SlowPolicy(),
        PairStore(tmp_path),
        async_inference=True,
    )
    collector.start_episode()
    started = time.monotonic()
    collector.tick(InputState(policy_step=False))
    elapsed = time.monotonic() - started
    collector.close()

    assert elapsed < 0.05


def test_async_inference_executes_returned_chunks_automatically(tmp_path):
    collector = InterventionCollector(
        Robot(),
        Policy(),
        PairStore(tmp_path),
        async_inference=True,
    )
    collector.start_episode()
    deadline = time.monotonic() + 1.0
    while len(collector.buffer.frames) < 4 and time.monotonic() < deadline:
        collector.tick(InputState(policy_step=True))
        time.sleep(0.005)
    collector.close()

    assert len(collector.buffer.frames) == 4
    assert [frame.observation["step"] for frame in collector.buffer.frames] == [1, 2, 3, 4]


def test_policy_frame_records_the_commanded_target_for_rollback(tmp_path):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path))
    c.tick(InputState())
    frame = c.buffer.frames[-1]
    np.testing.assert_allclose(frame.observation["_flowpro_rollback_target16"], robot.action)


def test_winner_records_measured_state_delta_not_absolute_quest_target(tmp_path):
    robot = Robot(); c = InterventionCollector(
        robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1)
    )
    c.tick(InputState()); c.tick(InputState(b=True)); c.tick(InputState())
    target = robot.state_action16(); target[8] += 0.02
    c.tick(InputState(middle=1, expert_action=target))
    c.tick(InputState(a=True))

    pair = PairStore(tmp_path).load(next(tmp_path.iterdir()))
    np.testing.assert_allclose(pair.winner[-1].action[8], 0.02, atol=1e-6)
    assert abs(float(pair.winner[-1].action[8]) - float(target[8])) > 0.1


def test_rollback_reverses_absolute_chunk_in_one_waypoint_call():
    class WaypointRobot:
        def __init__(self): self.calls = []
        def execute_rollback_waypoints(self, targets, *, step_duration_s):
            self.calls.append((np.asarray(targets).copy(), step_duration_s))

    def pose(x):
        value = np.zeros(16, np.float32); value[[3, 11]] = 1; value[8] = x
        return value

    frames = [
        Frame({"state_action16": pose(i), "_flowpro_rollback_target16": pose(i + 1)}, pose(.01))
        for i in range(3)
    ]
    robot = WaypointRobot()
    from flowpro.collection.rollback import RollbackBuffer
    buffer = RollbackBuffer(RollbackConfig(step_interval_s=.1))
    buffer.execute(robot, frames)

    assert len(robot.calls) == 1
    targets, duration = robot.calls[0]
    np.testing.assert_allclose(targets[:, 8], [2, 1, 0])
    assert duration == pytest.approx(.1)
