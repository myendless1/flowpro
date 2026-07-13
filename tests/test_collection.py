import numpy as np
import pytest
from flowpro.collection import InputState, InterventionCollector, Phase
from flowpro.cli.collect import (
    _gate_takeover_retry_b,
    _wait_for_a_reset,
    _wait_for_a_start,
)
from flowpro.collection.rollback import RollbackConfig
from flowpro.data import Frame, PairStore
from wan_va.action_representation import apply_relative_pose7


class Robot:
    def __init__(self): self.action = np.r_[np.zeros(3), [1,0,0,0], 0, [0.4,0,0], [1,0,0,0], 0].astype(np.float32); self.n=0; self.reset_count=0
    def observe(self): self.n += 1; return {"step": self.n, "state_action16": self.action.copy()}
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
    assert "Press A to move the robot" in capsys.readouterr().out


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
    assert "press A to start policy inference" in capsys.readouterr().out


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


def test_b_middle_a_commits_atomic_pair(tmp_path):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=2))
    c.tick(InputState()); c.tick(InputState())
    assert c.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert c.tick(InputState(middle=1, expert_action=robot.action)) is Phase.TAKEOVER
    assert c.tick(InputState(a=True)) is Phase.POLICY
    assert c.last_pair_saved is True
    target = next(tmp_path.iterdir())
    assert (target / "winner.npz").exists() and (target / "loser.npz").exists()
    pair = PairStore(tmp_path).load(target)
    assert pair.metadata["stored_action_representation"] == "delta"
    assert pair.metadata["history_action_representation"] == "absolute"


def test_a_before_takeover_discards_rollback_without_saving_pair(tmp_path, capsys):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1))
    c.tick(InputState()); c.tick(InputState(b=True)); c.tick(InputState())

    assert c.tick(InputState(a=True)) is Phase.POLICY
    assert c.last_pair_saved is False
    assert list(tmp_path.iterdir()) == []
    assert "No middle-trigger correction recorded" in capsys.readouterr().out


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
    assert "discarding the pair" in capsys.readouterr().out


def test_start_episode_clears_state_and_resets_runtime(tmp_path):
    robot = Robot(); policy = Policy()
    c = InterventionCollector(robot, policy, PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1))
    c.tick(InputState())
    c.start_episode()
    assert c.phase is Phase.POLICY
    assert len(c.buffer.frames) == 0
    assert robot.reset_count == 1
    assert policy.reset_count == 1


def test_rollback_buffer_only_keeps_the_current_policy_chunk(tmp_path):
    class ChunkPolicy(Policy):
        def __init__(self): super().__init__(); self.calls = 0; self.last_inference_started_chunk = False
        def infer(self, observation):
            self.calls += 1
            self.last_inference_started_chunk = self.calls in (1, 3)
            return super().infer(observation)

    c = InterventionCollector(Robot(), ChunkPolicy(), PairStore(tmp_path))
    c.tick(InputState()); c.tick(InputState())
    assert len(c.buffer.frames) == 2
    c.tick(InputState())
    assert len(c.buffer.frames) == 1


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
