import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astribot_env.quest_intervention import index_trigger_to_gripper_action
from astribot_env.utils import (
    action_quat_to_sdk_xyzw,
    quat_multiply_xyzw,
    quat_xyzw_to_matrix,
    quat_xyzw_to_rotvec,
    rotvec_to_quat_xyzw,
)

from flowpro.collection.astribot_runtime import (
    AstribotRobotIO,
    AstribotRuntimeConfig,
    FakeAstribotRobotIO,
    QuestControlSource,
    WanVAPolicy,
)
from wan_va.action_representation import (
    EXECUTION_CHANNEL_IDS,
    decode_action_sequence,
    decode_execution_sequence,
    delta16_to_model30,
    encode_absolute_history,
    encode_action_targets,
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


def test_absolute_history_is_never_delta_encoded():
    history = np.stack([_pose16(), _pose16()])
    history[:, 0] = [0.25, 0.31]

    model_history = encode_absolute_history(history)

    np.testing.assert_allclose(model30_to_execution16(model_history), history)


def test_absolute_and_delta_targets_only_differ_in_prediction_semantics():
    initial = _pose16()
    initial[0] = 0.2
    target = initial.copy()
    target[0] = 0.23

    absolute, _ = encode_action_targets(
        target[None], representation="absolute", references=initial[None]
    )
    delta, _ = encode_action_targets(
        target[None], representation="delta", references=initial[None]
    )

    np.testing.assert_allclose(model30_to_execution16(absolute)[0, 0], 0.23)
    np.testing.assert_allclose(model30_to_execution16(delta)[0, 0], 0.03)
    np.testing.assert_allclose(
        decode_action_sequence(
            model30_to_execution16(absolute),
            representation="absolute",
            initial_absolute=initial,
        ),
        target[None],
    )
    np.testing.assert_allclose(
        decode_action_sequence(
            model30_to_execution16(delta),
            representation="delta",
            initial_absolute=initial,
        ),
        target[None],
    )


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


def test_initial_pose_reset_lifts_both_arms_before_joint_motion():
    class RecordingRobot:
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"
        whole_body_names = [
            "chassis", arm_left_name, effector_left_name, arm_right_name,
            effector_right_name, "head",
        ]

        def __init__(self):
            self.calls = []

        def move_cartesian_pose(self, names, commands, **kwargs):
            self.calls.append(("lift", names, commands, kwargs))

        def move_joints_position(self, names, target, **kwargs):
            self.calls.append(("home", names, target, kwargs))

    initial = _pose16()
    initial[[2, 10]] = [0.72, 0.81]
    initial[[7, 15]] = [0.2, 0.8]
    measured_after_reset = initial.copy()

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(
        reset_prelift_height_m=0.10,
        reset_prelift_duration=1.25,
        right_min_z=None,
    )
    adapter.robot = RecordingRobot()
    adapter.action_history = deque()
    adapter.observation_history = deque()
    adapter._policy_chunk_count = 4
    states = iter([initial.copy(), measured_after_reset.copy()])
    adapter.state_action16 = lambda: next(states)

    adapter.move_to_initial_pose()

    assert [call[0] for call in adapter.robot.calls] == ["lift", "home"]
    _, names, commands, kwargs = adapter.robot.calls[0]
    assert names == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    np.testing.assert_allclose(commands[0][2], 0.82, atol=1e-7)
    np.testing.assert_allclose(commands[2][2], 0.91, atol=1e-7)
    np.testing.assert_allclose(commands[1], [80.0], atol=1e-7)
    np.testing.assert_allclose(commands[3], [20.0], atol=1e-7)
    assert kwargs == {
        "duration": 1.25,
        "use_wbc": False,
        "add_default_torso": False,
    }
    np.testing.assert_allclose(adapter.command_target16(), measured_after_reset)
    assert adapter._policy_chunk_count == 0


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


def test_right_quest_rotation_deltas_accumulate_on_current_absolute_target():
    class Robot:
        config = AstribotRuntimeConfig(
            right_min_z=None,
            right_gripper_angle_constraint_during_takeover=False,
        )

        def __init__(self):
            self.command = _pose16()
            self.command[11:15] = [0.9238795, 0.0, 0.3826834, 0.0]

        def command_target16(self):
            return self.command.copy()

    source = QuestControlSource.__new__(QuestControlSource)
    source.robot = Robot()
    source.anchor = source.robot.command_target16()
    source._right_quest_rotation_xyzw = None
    source._right_target_rotation_xyzw = None
    first_quest_rotation = rotvec_to_quat_xyzw([0.2, 0.0, 0.0])
    step_rotation = rotvec_to_quat_xyzw([0.0, 0.15, 0.0])
    second_quest_rotation = quat_multiply_xyzw(first_quest_rotation, step_rotation)

    first = source._expert_action(
        np.zeros(14, np.float32),
        {"right": {"active": True, "scaled_rotvec": [0.2, 0.0, 0.0]}},
    )
    second = source._expert_action(
        np.zeros(14, np.float32),
        {
            "right": {
                "active": True,
                "scaled_rotvec": quat_xyzw_to_rotvec(second_quest_rotation),
            }
        },
    )

    base = action_quat_to_sdk_xyzw(source.robot.command[11:15], use_xyzw=False)
    first_expected = quat_multiply_xyzw(base, first_quest_rotation)
    second_expected = quat_multiply_xyzw(first_expected, step_rotation)
    np.testing.assert_allclose(
        action_quat_to_sdk_xyzw(first[11:15], use_xyzw=False),
        first_expected,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action_quat_to_sdk_xyzw(second[11:15], use_xyzw=False),
        second_expected,
        atol=1e-6,
    )


def test_index_trigger_maps_continuously_to_gripper_action():
    np.testing.assert_allclose(index_trigger_to_gripper_action(0.0, 0.2), 1.0)
    np.testing.assert_allclose(index_trigger_to_gripper_action(0.2, 0.2), 1.0)
    np.testing.assert_allclose(index_trigger_to_gripper_action(0.6, 0.2), 0.5)
    np.testing.assert_allclose(index_trigger_to_gripper_action(1.0, 0.2), 0.0)


def test_quest_expert_action_uses_continuous_index_trigger_value():
    class Robot:
        config = AstribotRuntimeConfig(right_min_z=None)

    source = QuestControlSource.__new__(QuestControlSource)
    source.robot = Robot()
    source.quest = SimpleNamespace(gripper_threshold=0.2)
    source.anchor = _pose16()
    source.anchor[15] = 1.0
    info = {
        "right": {
            "active": True,
            "index": 0.6,
            "relative_position": [0.0, 0.0, 0.0],
            "scaled_rotvec": [0.0, 0.0, 0.0],
        }
    }

    target = source._expert_action(np.zeros(14, np.float32), info)

    np.testing.assert_allclose(target[15], 0.5)


def test_real_robot_adapter_clamps_a_tiny_right_arm_min_z_undershoot():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=0.862)
    current = _pose16()
    current[10] = 0.8618
    robot.state_action16 = lambda: current.copy()

    target = robot._delta_to_target(_pose16())

    assert target[10] >= 0.862


def test_policy_translation_safety_limit_is_six_centimeters():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(right_min_z=None)
    reference = _pose16()

    within_limit = reference.copy()
    within_limit[8] += 0.05
    robot._validate_step(robot._absolute_to_delta(reference, within_limit), within_limit)

    over_limit = reference.copy()
    over_limit[8] += 0.061
    with pytest.raises(ValueError, match="Unsafe Cartesian step"):
        robot._validate_step(robot._absolute_to_delta(reference, over_limit), over_limit)


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


def test_takeover_constrains_right_gripper_absolute_orientation_and_keeps_min_z():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(
        right_min_z=0.862,
        max_rotation_step_deg=0.0,
        takeover_max_rotation_step_deg=0.0,
    )
    previous = _pose16()
    previous[10] = 0.9
    robot._last_target = previous.copy()
    robot._takeover_limited_target = previous.copy()
    robot.action_history = deque()
    sent = []
    robot._send_target = lambda target, **_kwargs: sent.append(np.asarray(target).copy())
    target = previous.copy()
    target[10] = 0.7
    target[11:15] = [0.8, 0.2, -0.3, 0.45]

    robot.execute_takeover_absolute(
        target, arm_command_mask={"left": False, "right": True}
    )

    constrained = sent[-1]
    rotation = quat_xyzw_to_matrix(
        action_quat_to_sdk_xyzw(constrained[11:15], use_xyzw=False)
    )
    ray = rotation[:, 2]
    level = rotation[:, 0]
    ray_angle_deg = np.degrees(np.arcsin(np.clip(abs(float(ray[2])), 0.0, 1.0)))
    np.testing.assert_allclose(ray_angle_deg, 45.0, atol=1e-5)
    np.testing.assert_allclose(level[2], 0.0, atol=1e-6)
    assert constrained[10] >= 0.862


def test_takeover_rate_limits_continuous_gripper_target():
    robot = AstribotRobotIO.__new__(AstribotRobotIO)
    robot.config = AstribotRuntimeConfig(
        right_min_z=None,
        takeover_max_gripper_step=0.02,
    )
    previous = _pose16()
    previous[[7, 15]] = 1.0
    robot._last_target = previous.copy()
    robot._takeover_limited_target = previous.copy()
    robot.action_history = deque()
    sent = []
    robot._send_target = lambda target, **_kwargs: sent.append(np.asarray(target).copy())
    target = previous.copy()
    target[15] = 0.0

    robot.execute_takeover_absolute(
        target, arm_command_mask={"left": False, "right": True}
    )

    np.testing.assert_allclose(sent[-1][15], 0.98, atol=1e-7)
    np.testing.assert_allclose(sent[-1][7], 1.0, atol=1e-7)


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


def test_policy_steps_preserve_wbc_waypoints_and_chunk_durations():
    class WaypointRobot:
        torso_name = "torso"
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"

        def __init__(self):
            self.calls = []

        def get_desired_cartesian_pose(self, names):
            return [[0, 0, 1, 0, 0, 0, 1]]

        def move_cartesian_waypoints(self, names, waypoints, time_list, **kwargs):
            self.calls.append((waypoints, time_list, kwargs))

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(right_min_z=None)
    adapter.robot = WaypointRobot()
    adapter._last_target = _pose16()
    adapter._policy_chunk_count = 0
    adapter.action_history = deque()
    first = _pose16(); first[8] = 0.01
    second = _pose16(); second[8] = 0.02

    target_1 = adapter.execute_policy_step(
        first, first_in_chunk=True, last_in_chunk=False
    )
    target_2 = adapter.execute_policy_step(
        second, first_in_chunk=False, last_in_chunk=True
    )

    assert len(adapter.robot.calls) == 2
    assert adapter.robot.calls[0][1] == [0.6]
    assert adapter.robot.calls[1][1] == [0.1]
    assert all(call[2] == {"use_wbc": True, "add_default_torso": False}
               for call in adapter.robot.calls)
    np.testing.assert_allclose([target_1[8], target_2[8]], [0.01, 0.03], atol=1e-7)
    assert adapter._policy_chunk_count == 1


def test_policy_chunk_uses_four_overlapping_nine_point_waypoint_batches():
    class WaypointRobot:
        torso_name = "torso"
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"

        def __init__(self):
            self.calls = []

        def get_desired_cartesian_pose(self, names):
            return [[0, 0, 1, 0, 0, 0, 1]]

        def move_cartesian_waypoints(self, names, waypoints, time_list, **kwargs):
            self.calls.append((waypoints, time_list, kwargs))

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(right_min_z=None)
    adapter.robot = WaypointRobot()
    adapter._last_target = _pose16()
    adapter._policy_chunk_count = 0
    adapter._policy_torso_pose = None
    adapter.action_history = deque()
    adapter.observation_history = deque()
    actions = np.tile(_pose16(), (32, 1))
    actions[:, 8] = 0.001

    result = adapter.execute_policy_waypoint_batches(actions, batch_size=8)

    assert len(adapter.robot.calls) == 4
    assert all(len(waypoints) == 9 for waypoints, _, _ in adapter.robot.calls)
    assert all(len(times) == 9 and times[0] == 0.0
               for _, times, _ in adapter.robot.calls)
    np.testing.assert_allclose(adapter.robot.calls[0][1], [0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    for previous, current in zip(adapter.robot.calls, adapter.robot.calls[1:]):
        assert current[0][0] == previous[0][-1]
        np.testing.assert_allclose(current[1], np.arange(9) * 0.1)
    assert len(adapter.action_history) == 32
    assert result["targets"].shape == (32, 16)


def test_rollback_clamps_legacy_measured_start_above_right_min_z():
    class WaypointRobot:
        torso_name = "torso"
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"

        def __init__(self):
            self.calls = []

        def get_desired_cartesian_pose(self, names):
            return [[0, 0, 1, 0, 0, 0, 1]]

        def move_cartesian_waypoints(self, names, waypoints, time_list, **kwargs):
            self.calls.append((waypoints, time_list, kwargs))

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(right_min_z=0.862)
    adapter.robot = WaypointRobot()
    adapter._last_target = _pose16()
    adapter._last_target[10] = 0.87
    adapter._policy_torso_pose = None
    adapter.action_history = deque()
    target = adapter._last_target.copy()
    target[10] = 0.8617

    adapter.execute_rollback_waypoints(target[None], step_duration_s=0.1)

    right_arm_pose = adapter.robot.calls[0][0][-1][3]
    assert right_arm_pose[2] >= 0.862


def test_policy_absolute_chunk_is_sent_without_delta_integration():
    class WaypointRobot:
        torso_name = "torso"
        arm_left_name = "left_arm"
        effector_left_name = "left_gripper"
        arm_right_name = "right_arm"
        effector_right_name = "right_gripper"

        def get_desired_cartesian_pose(self, names): return [[0, 0, 1, 0, 0, 0, 1]]
        def move_cartesian_waypoints(self, names, waypoints, time_list, **kwargs): pass

    adapter = AstribotRobotIO.__new__(AstribotRobotIO)
    adapter.config = AstribotRuntimeConfig(
        action_representation="absolute", right_min_z=None
    )
    adapter.action_representation = "absolute"
    adapter.robot = WaypointRobot()
    adapter._last_target = _pose16()
    adapter._policy_chunk_count = 0
    adapter.action_history = deque()
    targets = np.tile(_pose16(), (2, 1))
    targets[:, 8] = [0.01, 0.02]

    executed = adapter.execute_policy_waypoints(targets)

    np.testing.assert_allclose(executed[:, 8], [0.01, 0.02], atol=1e-7)
    np.testing.assert_allclose(np.asarray(adapter.action_history)[:, 8], [0.01, 0.02])


def test_policy_waypoint_clips_slightly_out_of_range_gripper_target():
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
    delta = _pose16()
    delta[15] = 1.0071225

    targets = adapter.execute_policy_waypoints(delta.reshape(1, 16))

    np.testing.assert_allclose(targets[0, 15], 1.0)
    np.testing.assert_allclose(adapter.robot.calls[0][1][0][4], [0.0])


def test_policy_returns_the_unsmoothed_server_chunk():
    policy = WanVAPolicy(host="127.0.0.1", port=8006, prompt="test", replan_steps=3, fake=True)
    chunk = policy.infer({"wam4d": {}, "state_action16": _pose16()})
    assert chunk.shape == (3, 16)
    np.testing.assert_allclose(chunk[:, 8], 0.0)


def test_policy_logs_de_normalized_server_delta_xyz_before_execution(capsys):
    policy = WanVAPolicy(host="127.0.0.1", port=8006, prompt="test", replan_steps=1, fake=True)
    policy.infer({"wam4d": {}, "state_action16": _pose16()})

    output = capsys.readouterr().out
    assert "WAM4D 服务端动作 #1 （反归一化后的 delta xyz）" in output
    assert "左臂=[+0.00000, +0.00000, +0.00000]" in output
    assert "右臂=[+0.00000, +0.00000, +0.00000]" in output


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
    assert "左臂和左夹爪已锁定" in capsys.readouterr().out


def test_absolute_policy_locks_left_arm_to_last_executed_cmd():
    policy = WanVAPolicy(
        host="127.0.0.1",
        port=8006,
        prompt="test",
        replan_steps=1,
        fake=True,
        control_left_arm=False,
        action_representation="absolute",
    )
    measured = _pose16()
    measured[0] = 0.1
    commanded = measured.copy()
    commanded[0] = 0.25
    payload = {"observation.executed_action_history": commanded[None]}

    action = policy.infer({"wam4d": payload, "state_action16": measured})[0]

    np.testing.assert_allclose(action[0:8], commanded[0:8])


def test_flowpro_default_explicitly_uses_delta_representation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/flowpro.json").read_text())
    assert config["model"]["experiment_config"].endswith("/delta.json")
    assert config["model"]["action_representation"] == "delta"
