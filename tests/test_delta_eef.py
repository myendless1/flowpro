import json
from collections import deque
from pathlib import Path

import numpy as np

from flowpro.collection.astribot_runtime import (
    AstribotRobotIO,
    AstribotRuntimeConfig,
    FakeAstribotRobotIO,
    QuestControlSource,
    WanVAPolicy,
)
from wan_va.action_representation import (
    EXECUTION_CHANNEL_IDS,
    decode_execution_sequence,
    delta16_to_model30,
    model30_to_execution16,
)


def _pose16():
    value = np.zeros(16, np.float32)
    value[[3, 11]] = 1.0
    return value


def test_delta_model_encoding_and_sequential_decoding_round_trip():
    initial = _pose16()
    targets = np.stack([initial.copy(), initial.copy()])
    targets[0, 0] = 0.01
    targets[1, 0] = 0.03
    targets[:, [7, 15]] = [[0.2, 0.8], [0.3, 0.7]]
    references = np.stack([initial, targets[0]])

    model, mask = delta16_to_model30(targets, references=references)
    deltas = model30_to_execution16(model)

    np.testing.assert_allclose(deltas[:, 0], [0.01, 0.02], atol=1e-7)
    np.testing.assert_allclose(decode_execution_sequence(deltas, initial_absolute=initial), targets)
    assert mask[:, EXECUTION_CHANNEL_IDS].all()


def test_fake_robot_applies_each_delta_against_live_state():
    robot = FakeAstribotRobotIO()
    delta = _pose16()
    delta[0] = 0.01
    robot.execute(delta)
    robot.execute(delta)
    np.testing.assert_allclose(robot.state_action16()[0], 0.02, atol=1e-7)


def test_real_robot_adapter_submits_eef_and_grippers_in_one_mixed_command():
    class RecordingRobot:
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"
        whole_body_names = [
            "chassis",
            arm_left_name,
            effector_left_name,
            arm_right_name,
            effector_right_name,
        ]

        def __init__(self):
            self.calls = []

        def set_different_type_command(self, names, types, commands, **kwargs):
            self.calls.append((names, types, commands, kwargs))

    initial = _pose16()
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=None)
    robot.robot = RecordingRobot()
    robot.action_history = deque()
    robot.state_action16 = lambda: initial.copy()

    delta = _pose16()
    delta[[7, 15]] = [0.2, 0.8]
    robot.execute(delta)

    assert len(robot.robot.calls) == 1
    names, types, commands, kwargs = robot.robot.calls[0]
    assert names == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    assert types == ["cartesian", "joints", "cartesian", "joints"]
    np.testing.assert_allclose(commands[1], [80.0])
    np.testing.assert_allclose(commands[3], [20.0])
    assert kwargs == {"control_way": "filter", "use_wbc": False}


def test_right_only_takeover_sends_a_fixed_left_target_in_complete_mixed_command():
    class RecordingRobot:
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"
        whole_body_names = [
            "chassis", arm_left_name, effector_left_name, arm_right_name, effector_right_name
        ]

        def __init__(self): self.calls = []; self.arm_calls = []; self.gripper_calls = []
        def set_different_type_command(self, names, types, commands, **kwargs):
            self.calls.append((names, types, commands, kwargs))
        def set_cartesian_pose(self, names, poses, **kwargs):
            self.arm_calls.append((names, poses, kwargs))
        def set_joints_position(self, names, positions, **kwargs):
            self.gripper_calls.append((names, positions, kwargs))

    initial = _pose16()
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=None)
    robot.robot = RecordingRobot()
    robot._last_target = initial.copy()
    robot._takeover_limited_target = initial.copy()
    robot.action_history = deque()
    target = initial.copy(); target[8] += 0.005

    robot.execute_takeover_absolute(
        target, arm_command_mask={"left": False, "right": True}
    )

    assert len(robot.robot.calls) == 1
    assert robot.robot.arm_calls == []
    assert robot.robot.gripper_calls == []
    names, types, commands, kwargs = robot.robot.calls[0]
    assert names == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    assert types == ["cartesian", "joints", "cartesian", "joints"]
    np.testing.assert_allclose(commands[0][:3], initial[:3])
    np.testing.assert_allclose(commands[2][:3], target[8:11])
    assert kwargs == {"control_way": "filter", "use_wbc": False}
    np.testing.assert_allclose(robot.command_target16()[0:8], initial[0:8])


def test_quest_rotation_does_not_modify_takeover_translation():
    class Robot:
        config = AstribotRuntimeConfig(right_min_z=None)

    source = QuestControlSource.__new__(QuestControlSource)
    source.robot = Robot()
    source.anchor = _pose16()
    source.anchor[8:11] = [0.4, -0.2, 0.9]
    residual = np.zeros(14, np.float32)
    info = {
        "right": {
            "active": True,
            "relative_position": [0.0, 0.0, 0.0],
            "scaled_rotvec": [0.3, -0.2, 0.1],
        }
    }

    rotated = source._expert_action(residual, info)
    np.testing.assert_allclose(rotated[8:11], source.anchor[8:11], atol=1e-7)

    info["right"]["relative_position"] = [0.02, -0.01, 0.03]
    translated = source._expert_action(residual, info)
    np.testing.assert_allclose(
        translated[8:11], source.anchor[8:11] + [0.02, -0.01, 0.03], atol=1e-7
    )


def test_real_robot_adapter_clamps_a_tiny_right_arm_min_z_undershoot():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=0.862)
    current = _pose16()
    current[10] = 0.8618
    robot.state_action16 = lambda: current.copy()

    target = robot._delta_to_target(_pose16())

    assert target[10] >= 0.862


def test_delta_target_uses_the_last_sent_target_not_measured_robot_pose():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=None)
    measured = _pose16()
    measured[[0, 8]] = [-0.3, -0.2]
    last_target = _pose16()
    last_target[[0, 8]] = [0.3, 0.2]
    robot._last_target = last_target.copy()
    robot.state_action16 = lambda: measured.copy()

    delta = _pose16()
    delta[[0, 8]] = [0.01, -0.02]
    target = robot._delta_to_target(delta)

    np.testing.assert_allclose(target[[0, 8]], [0.31, 0.18], atol=1e-7)


def test_takeover_rebases_command_gap_and_limits_each_streaming_step():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(
        right_min_z=None,
        takeover_max_translation_step_m=0.01,
        takeover_max_rotation_step_deg=2.5,
    )
    measured = _pose16()
    commanded = measured.copy(); commanded[8] = 0.08
    robot._last_target = commanded
    robot.action_history = deque()
    robot.observation_history = deque()
    robot.state_action16 = lambda: measured.copy()
    sent = []
    robot._send_target = lambda target, **_kwargs: sent.append(np.asarray(target).copy())

    robot.begin_takeover()
    target = measured.copy(); target[8] = 0.08
    robot.execute_takeover_absolute(target)

    np.testing.assert_allclose(robot.command_target16()[8], 0.01, atol=1e-7)
    np.testing.assert_allclose(sent[-1][8], 0.01, atol=1e-7)


def test_takeover_clamps_measured_right_z_below_safety_floor():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=0.862)
    measured = _pose16(); measured[10] = 0.8603
    robot._last_target = measured.copy()
    robot._takeover_limited_target = measured.copy()
    robot.action_history = deque()
    sent = []
    robot._send_target = lambda target, **_kwargs: sent.append(np.asarray(target).copy())

    robot.execute_takeover_absolute(measured)

    assert sent[-1][10] >= 0.862


def test_policy_delta_chunk_is_sent_as_one_absolute_waypoint_trajectory():
    class WaypointRobot:
        torso_name = "torso"
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"

        def __init__(self): self.calls = []
        def get_desired_cartesian_pose(self, names): return [[0, 0, 1, 0, 0, 0, 1]]
        def move_cartesian_waypoints(self, names, waypoints, time_list, **kwargs):
            self.calls.append((names, waypoints, time_list, kwargs))

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(right_min_z=None)
    adapter.robot = WaypointRobot()
    adapter._last_target = _pose16()
    adapter._policy_chunk_count = 0
    adapter.action_history = deque()
    deltas = np.tile(_pose16(), (2, 1)); deltas[:, 8] = [0.01, 0.02]

    targets = adapter.execute_policy_waypoints(deltas)

    assert len(adapter.robot.calls) == 1
    names, waypoints, times, kwargs = adapter.robot.calls[0]
    assert names == ["torso", "left_arm", "left_gripper", "right_arm", "right_gripper"]
    assert len(waypoints) == 2
    np.testing.assert_allclose(times, [0.6, 0.7])
    np.testing.assert_allclose(targets[:, 8], [0.01, 0.03], atol=1e-7)
    assert kwargs == {"use_wbc": True, "add_default_torso": False}


def test_policy_returns_the_unsmoothed_server_chunk():
    policy = WanVAPolicy(host="127.0.0.1", port=8006, prompt="test", replan_steps=3, fake=True)
    chunk = policy.infer({"wam4d": {}, "state_action16": _pose16()})
    assert chunk.shape == (3, 16)
    np.testing.assert_allclose(chunk[:, 8], 0.0)


def test_policy_logs_de_normalized_server_delta_xyz_before_execution(capsys):
    policy = WanVAPolicy(host="127.0.0.1", port=8006, prompt="test", replan_steps=1, fake=True)
    policy.infer({"wam4d": {}, "state_action16": _pose16()})

    output = capsys.readouterr().out
    assert "WAM4D server action #1 (de-normalized delta xyz)" in output
    assert "left=[+0.00000, +0.00000, +0.00000]" in output
    assert "right=[+0.00000, +0.00000, +0.00000]" in output


def test_policy_can_lock_the_left_arm_without_changing_the_right_action(capsys):
    policy = WanVAPolicy(
        host="127.0.0.1",
        port=8006,
        prompt="test",
        replan_steps=1,
        fake=True,
        control_left_arm=False,
    )
    state = _pose16()
    state[7] = 0.3
    action = policy.infer({"wam4d": {}, "state_action16": state})[0]

    np.testing.assert_allclose(action[0:3], 0.0)
    np.testing.assert_allclose(action[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(action[7], 0.3)
    np.testing.assert_allclose(action[8:15], _pose16()[8:15])
    assert "left arm/gripper locked" in capsys.readouterr().out


def test_flowpro_uses_reference_delta_experiment_without_representation_switch():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/flowpro.json").read_text())
    assert config["model"]["experiment_config"].endswith("/delta.json")
    assert "action_representation" not in config["model"]
