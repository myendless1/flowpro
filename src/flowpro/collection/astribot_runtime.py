from __future__ import annotations

"""Concrete Astribot/Quest/Wan-VA adapters used by the preference collector.

Heavy robot and websocket dependencies are imported lazily so the collection
state machine and its fake mode remain testable on a development machine.
"""

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np

from astribot_env.quest_intervention import (
    QuestResidualIntervention,
    index_trigger_to_gripper_action,
)
from astribot_env.initial_pose import default_init_joint_action, normalize_init_joint_action
from astribot_env.rgbd import RGBDReader, build_wam4d_observation_payload, get_current_eef_state
from astribot_env.sdk_loader import DEFAULT_ASTRIBOT_SDK_ROOT, load_astribot_class
from astribot_env.utils import (
    ACTION16_DIM,
    action16_to_sdk_commands,
    action_quat_to_sdk_xyzw,
    apply_right_gripper_orientation_constraint,
    convert_gripper_cmd_value_to_action_value,
    quat_inverse_xyzw,
    quat_multiply_xyzw,
    rotvec_to_quat_xyzw,
    sdk_xyzw_to_action_quat,
)
from astribot_env.wam4d_policy import WAM4DPriorClient
from wan_va.action_representation import (
    apply_relative_pose7,
    relative_pose7,
    validate_action_representation,
)

from .controller import InputState


@dataclass
class AstribotRuntimeConfig:
    action_representation: str = "delta"
    sdk_root: str = ""
    robot_type: str = "S1"
    sdk_frequency: float = 100.0
    cartesian_frame: str = "chassis"
    control_way: str = "filter"
    use_xyzw: bool = False
    camera_timeout: float = 0.3
    image_from_s1_topic: bool = True
    camera_sync_slop_s: float = 0.05
    camera_sync_rate_hz: float = 40.0
    max_translation_step_m: float = 0.06
    max_rotation_step_deg: float = 15.0
    takeover_max_translation_step_m: float = 0.01
    takeover_max_rotation_step_deg: float = 2.5
    takeover_max_gripper_step: float = 0.02
    first_policy_waypoint_duration: float = 0.6
    policy_waypoint_duration: float = 0.1
    init_joint_action: list[list[float]] = field(default_factory=default_init_joint_action)
    initial_joint_duration: float = 4.0
    reset_prelift_height_m: float = 0.10
    reset_prelift_duration: float = 1.0
    reset_to_initial_on_startup: bool = True
    left_xyz_low: tuple[float, float, float] | None = None
    left_xyz_high: tuple[float, float, float] | None = None
    right_xyz_low: tuple[float, float, float] | None = None
    right_xyz_high: tuple[float, float, float] | None = None
    right_min_z: float | None = 0.862
    right_gripper_angle_constraint_during_takeover: bool = True
    right_gripper_target_angle_deg: float = 45.0
    right_gripper_ray_axis: str = "+z"
    right_gripper_twist_level_constraint: bool = True
    right_gripper_level_axis: str = "+x"
    state_history_len: int = 16
    obs_history_len: int = 9


class AstribotRobotIO:
    """Astribot adapter with configurable policy action semantics."""

    def __init__(self, config: AstribotRuntimeConfig | None = None) -> None:
        import os

        self.config = config or AstribotRuntimeConfig()
        self.action_representation = validate_action_representation(
            self.config.action_representation
        )
        os.environ.setdefault("ROBOT_TYPE", self.config.robot_type)
        Astribot = load_astribot_class(self.config.sdk_root)
        self.robot = Astribot(freq=self.config.sdk_frequency, high_control_rights=True)
        if hasattr(self.robot, "set_filter_parameters"):
            self.robot.set_filter_parameters(0.1, 0.5)
        if self.config.reset_to_initial_on_startup:
            self._move_to_initial_joint_pose()
        self.rgbd = RGBDReader(
            self.robot,
            camera_timeout=self.config.camera_timeout,
            use_topic=self.config.image_from_s1_topic,
            sync_slop_s=self.config.camera_sync_slop_s,
            sync_rate_hz=self.config.camera_sync_rate_hz,
        )
        self.action_history: deque[np.ndarray] = deque(maxlen=self.config.state_history_len)
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=self.config.obs_history_len)
        self._history_lock = threading.Lock()
        # Delta policy actions are integrated against the target actually sent
        # to the SDK. Both modes expose those absolute cmd targets as history.
        self._last_target = self.state_action16()
        self._takeover_limited_target: np.ndarray | None = None
        self._policy_chunk_count = 0
        self._policy_torso_pose = None

    def _move_to_initial_joint_pose(self) -> None:
        target = normalize_init_joint_action(self.config.init_joint_action)
        self.robot.move_joints_position(
            self.robot.whole_body_names[1:], target,
            duration=self.config.initial_joint_duration, use_wbc=False,
        )

    def _raise_current_arms_before_initial_reset(self) -> None:
        lift_height = float(self.config.reset_prelift_height_m)
        if lift_height <= 0.0:
            return

        duration = max(0.0, float(self.config.reset_prelift_duration))
        current_action = self.state_action16()
        lifted_action = current_action.copy()
        lifted_action[2] += lift_height
        lifted_action[10] += lift_height
        arm_poses, grippers = action16_to_sdk_commands(
            lifted_action,
            use_xyzw=self.config.use_xyzw,
        )
        names = [
            self.robot.arm_left_name,
            self.robot.effector_left_name,
            self.robot.arm_right_name,
            self.robot.effector_right_name,
        ]
        commands = [arm_poses[0], grippers[0], arm_poses[1], grippers[1]]
        print(
            "复位到初始位姿前正在抬起 Astribot 双臂："
            f"左臂 z {float(current_action[2]):.3f}->{float(lifted_action[2]):.3f}，"
            f"右臂 z {float(current_action[10]):.3f}->{float(lifted_action[10]):.3f}，"
            f"用时 {duration:.3f} 秒。",
            flush=True,
        )
        if hasattr(self.robot, "move_cartesian_pose"):
            self.robot.move_cartesian_pose(
                names,
                commands,
                duration=duration,
                use_wbc=False,
                add_default_torso=False,
            )
            return
        if not hasattr(self.robot, "set_different_type_command"):
            raise RuntimeError(
                "Astribot SDK must provide a mixed-command API for the "
                "pre-reset arm lift."
            )
        self.robot.set_different_type_command(
            names,
            ["cartesian", "joints", "cartesian", "joints"],
            commands,
            control_way=self.config.control_way,
            use_wbc=False,
        )
        if duration > 0.0:
            time.sleep(duration)

    def move_to_initial_pose(self) -> None:
        """Raise both arms, move non-chassis joints home, and rebase control."""
        self._raise_current_arms_before_initial_reset()
        self._move_to_initial_joint_pose()
        measured = self.state_action16()
        self.reset_history(measured)

    def state_action16(self) -> np.ndarray:
        return get_current_eef_state(
            self.robot,
            use_xyzw=self.config.use_xyzw,
            frame=self.config.cartesian_frame,
        )

    def command_target16(self) -> np.ndarray:
        return self._last_target.copy()

    def _configured_action_representation(self) -> str:
        return validate_action_representation(
            getattr(
                self,
                "action_representation",
                getattr(self.config, "action_representation", "delta"),
            )
        )

    def begin_takeover(self) -> None:
        """Rebase teleoperation on measured state without treating servo lag as motion."""
        measured = self.state_action16()
        self._takeover_limited_target = measured.copy()
        self.reset_history(measured)

    def end_takeover(self) -> None:
        self._takeover_limited_target = None

    def observe(self) -> dict[str, Any]:
        images, timing = self.rgbd.get_bgr_images_snapshot()
        state = self.state_action16()
        state_timestamp = time.time()
        timing = dict(timing)
        timing["state_timestamp"] = state_timestamp
        image_timestamp = timing.get("image_timestamp")
        timing["state_image_skew_s"] = (
            None
            if image_timestamp is None
            else abs(state_timestamp - float(image_timestamp))
        )
        payload = build_wam4d_observation_payload(
            bgr_images=images,
            prompt="",
            action_history=self._action_history_snapshot(),
            history_len=self.config.state_history_len,
        )
        with self._history_lock:
            self.observation_history.append(payload)
            observation_history = list(self.observation_history)
        return {
            "state_action16": state,
            "wam4d": payload,
            "wam4d_history": observation_history,
            "time": state_timestamp,
            "_flowpro_timing": timing,
        }

    def _action_history_snapshot(self) -> deque[np.ndarray]:
        lock = getattr(self, "_history_lock", None)
        if lock is None:
            return deque(self.action_history, maxlen=getattr(self.action_history, "maxlen", None))
        with lock:
            return deque(self.action_history, maxlen=self.action_history.maxlen)

    def _append_action_history(self, targets: np.ndarray) -> None:
        values = np.asarray(targets, np.float32).reshape(-1, ACTION16_DIM)
        lock = getattr(self, "_history_lock", None)
        if lock is None:
            for target in values:
                self.action_history.append(target.copy())
            return
        with lock:
            for target in values:
                self.action_history.append(target.copy())

    @staticmethod
    def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        a = a / max(float(np.linalg.norm(a)), 1e-8)
        b = b / max(float(np.linalg.norm(b)), 1e-8)
        return float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0, 1))))

    def _delta_to_target(
        self,
        delta: np.ndarray,
        *,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        if reference is None:
            reference = getattr(self, "_last_target", None)
        if reference is None:
            reference = self.state_action16()
        reference = np.asarray(reference, dtype=np.float32).reshape(ACTION16_DIM)
        target = reference.copy()
        for off, low, high in (
            (0, self.config.left_xyz_low, self.config.left_xyz_high),
            (8, self.config.right_xyz_low, self.config.right_xyz_high),
        ):
            target[off : off + 7] = apply_relative_pose7(
                reference[off : off + 7], delta[off : off + 7]
            )
            target[off + 7] = np.clip(delta[off + 7], 0.0, 1.0)
            if low is not None:
                target[off : off + 3] = np.maximum(
                    target[off : off + 3], np.asarray(low, dtype=np.float32)
                )
            if high is not None:
                target[off : off + 3] = np.minimum(
                    target[off : off + 3], np.asarray(high, dtype=np.float32)
                )
        if self.config.right_min_z is not None:
            minimum_z = np.float32(self.config.right_min_z)
            # Store the next representable float32 above the requested bound;
            # assigning exactly 0.862 to a float32 otherwise becomes
            # 0.86199999 and fails a float64 safety comparison.
            target[10] = max(
                target[10],
                np.nextafter(minimum_z, np.float32(np.inf)),
            )
        return target

    @staticmethod
    def _absolute_to_delta(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=np.float32).reshape(ACTION16_DIM)
        target = np.asarray(target, dtype=np.float32).reshape(ACTION16_DIM)
        delta = target.copy()
        delta[0:7] = relative_pose7(reference[0:7], target[0:7])
        delta[8:15] = relative_pose7(reference[8:15], target[8:15])
        return delta

    def _validate_step(
        self,
        delta: np.ndarray,
        target: np.ndarray,
        *,
        arm_command_mask: dict[str, bool] | None = None,
    ) -> None:
        active = arm_command_mask or {"left": True, "right": True}
        grip_indices = [7 if hand == "left" else 15 for hand in ("left", "right") if active.get(hand, False)]
        if grip_indices and (
            np.any(target[grip_indices] < 0) or np.any(target[grip_indices] > 1)
        ):
            raise ValueError(f"Gripper targets must be in [0,1], got {target[grip_indices]}")
        for hand, off, low, high in (
            ("left", 0, self.config.left_xyz_low, self.config.left_xyz_high),
            ("right", 8, self.config.right_xyz_low, self.config.right_xyz_high),
        ):
            if not active.get(hand, False):
                continue
            xyz = target[off : off + 3]
            if low is not None and np.any(xyz < np.asarray(low)):
                raise ValueError(f"Arm@{off} target below workspace lower bound: {xyz}")
            if high is not None and np.any(xyz > np.asarray(high)):
                raise ValueError(f"Arm@{off} target above workspace upper bound: {xyz}")
            if (
                off == 8
                and self.config.right_min_z is not None
                and xyz[2] < float(self.config.right_min_z) - 1e-6
            ):
                raise ValueError(f"Right arm z={xyz[2]:.4f} below minimum {self.config.right_min_z:.4f}")
            if float(np.linalg.norm(delta[off + 3 : off + 7])) < 1e-6:
                raise ValueError(f"Arm@{off} quaternion has zero norm")
            distance = float(np.linalg.norm(delta[off : off + 3]))
            if self.config.max_translation_step_m > 0 and distance > self.config.max_translation_step_m:
                raise ValueError(f"Unsafe Cartesian step for arm@{off}: {distance:.4f}m")
            angle = self._quat_angle_deg(delta[off + 3 : off + 7], np.array([1, 0, 0, 0]))
            if self.config.max_rotation_step_deg > 0 and angle > self.config.max_rotation_step_deg:
                raise ValueError(f"Unsafe rotation step for arm@{off}: {angle:.2f}deg")

    def _send_target(
        self,
        target: np.ndarray,
        *,
        arm_command_mask: dict[str, bool] | None = None,
    ) -> None:
        arm_poses, grippers = action16_to_sdk_commands(target, use_xyzw=self.config.use_xyzw)
        arm_names = [self.robot.arm_left_name, self.robot.arm_right_name]
        grip_names = [self.robot.effector_left_name, self.robot.effector_right_name]
        active = arm_command_mask or {"left": True, "right": True}
        commands = {}
        for index, hand in enumerate(("left", "right")):
            if active.get(hand, False):
                commands[arm_names[index]] = ("cartesian", arm_poses[index])
                commands[grip_names[index]] = ("joints", grippers[index])
        if not commands:
            return
        active_indices = [
            index for index, hand in enumerate(("left", "right")) if active.get(hand, False)
        ]
        if self.config.control_way == "filter" and len(active_indices) == 1:
            index = active_indices[0]
            self.robot.set_cartesian_pose(
                [arm_names[index]],
                [arm_poses[index]],
                control_way=self.config.control_way,
                use_wbc=False,
                add_default_torso=False,
            )
            self.robot.set_joints_position(
                [grip_names[index]],
                [grippers[index]],
                control_way=self.config.control_way,
                use_wbc=False,
                add_default_torso=False,
            )
            return
        if not hasattr(self.robot, "set_different_type_command"):
            raise RuntimeError(
                "Astribot SDK must provide set_different_type_command so EEF and "
                "gripper targets can be sent atomically."
            )
        order = [name for name in getattr(self.robot, "whole_body_names", []) if name in commands]
        if len(order) != len(commands):
            order = [
                name
                for name in (arm_names[0], grip_names[0], arm_names[1], grip_names[1])
                if name in commands
            ]
        self.robot.set_different_type_command(
            order,
            [commands[name][0] for name in order],
            [commands[name][1] for name in order],
            control_way=self.config.control_way,
            use_wbc=False,
        )

    def execute(self, action16: np.ndarray) -> None:
        action = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        if not np.isfinite(action).all():
            raise ValueError("Robot command contains NaN/Inf")
        reference = getattr(self, "_last_target", None)
        if reference is None:
            reference = self.state_action16()
        reference = np.asarray(reference, np.float32).reshape(ACTION16_DIM).copy()
        if self._configured_action_representation() == "delta":
            target = self._delta_to_target(action, reference=reference)
        else:
            target = self._delta_to_target(
                self._absolute_to_delta(reference, action), reference=reference
            )
        delta = self._absolute_to_delta(reference, target)
        self._validate_step(delta, target)
        self._send_target(target)
        self._last_target = target.copy()
        self._append_action_history(target)

    def execute_policy_waypoints(self, actions16: np.ndarray) -> np.ndarray:
        """Decode one policy chunk and submit absolute SDK waypoints."""
        actions = np.asarray(actions16, np.float32).reshape(-1, ACTION16_DIM)
        if not len(actions):
            raise ValueError("Policy waypoint chunk cannot be empty")
        reference = self._last_target.copy()
        targets = []
        for action in actions:
            target = self._policy_action_to_target(action, reference=reference)
            targets.append(target)
            reference = target
        targets_array = np.asarray(targets, np.float32)

        durations = []
        for index in range(len(targets_array)):
            durations.append(
                float(self.config.first_policy_waypoint_duration)
                if self._policy_chunk_count == 0 and index == 0
                else float(self.config.policy_waypoint_duration)
            )
        self._execute_absolute_waypoint_trajectory(targets_array, durations)
        self._policy_chunk_count += 1
        return targets_array

    def execute_policy_step(
        self,
        action16: np.ndarray,
        *,
        first_in_chunk: bool,
        last_in_chunk: bool,
    ) -> np.ndarray:
        """Execute one policy action as an interruptible one-waypoint trajectory."""
        reference = self._last_target.copy()
        target = self._policy_action_to_target(action16, reference=reference)
        duration = (
            float(self.config.first_policy_waypoint_duration)
            if self._policy_chunk_count == 0 and first_in_chunk
            else float(self.config.policy_waypoint_duration)
        )
        self._execute_absolute_waypoint_trajectory(
            target.reshape(1, ACTION16_DIM),
            [duration],
        )
        if last_in_chunk:
            self._policy_chunk_count += 1
        return target.copy()

    def execute_policy_waypoint_batch(
        self,
        actions16: np.ndarray,
        *,
        first_in_chunk: bool,
        last_in_chunk: bool,
    ) -> dict[str, Any]:
        """Submit one waypoint call containing an anchor and new policy actions."""
        actions = np.asarray(actions16, np.float32).reshape(-1, ACTION16_DIM)
        if not len(actions):
            raise ValueError("策略 waypoint batch 不能为空")

        reference = self._last_target.copy()
        targets = []
        starts = []
        durations = []
        for index, action in enumerate(actions):
            starts.append(reference.copy())
            target = self._policy_action_to_target(action, reference=reference)
            targets.append(target)
            durations.append(
                float(self.config.first_policy_waypoint_duration)
                if self._policy_chunk_count == 0 and first_in_chunk and index == 0
                else float(self.config.policy_waypoint_duration)
            )
            reference = target

        targets_array = np.asarray(targets, np.float32)
        starts_array = np.asarray(starts, np.float32)
        action_start_times = np.empty(len(actions), np.float64)
        action_arrival_times = np.empty(len(actions), np.float64)

        batch_durations = np.asarray(durations, np.float64)
        cumulative = np.cumsum(batch_durations)
        # SDK's move_cartesian_waypoints internally prepends the current robot
        # state at t=0, so time_list must start at the first positive duration.
        # Passing an explicit anchor at t=0.0 would produce a duplicate zero
        # and trigger "x must be strictly increasing" in scipy's cubic spline.
        submitted_at = self._execute_absolute_waypoint_schedule(
            targets_array,
            cumulative.tolist(),
            history_start_index=0,
        )
        action_start_times[:] = submitted_at + np.concatenate(
            [[0.0], cumulative[:-1]]
        )
        action_arrival_times[:] = submitted_at + cumulative

        if last_in_chunk:
            self._policy_chunk_count += 1
        return {
            "targets": targets_array,
            "start_targets": starts_array,
            "action_start_times": action_start_times,
            "action_arrival_times": action_arrival_times,
            "finished_at": time.time(),
        }

    def execute_policy_waypoint_batches(
        self,
        actions16: np.ndarray,
        *,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Compatibility helper that executes a full chunk batch by batch."""
        actions = np.asarray(actions16, np.float32).reshape(-1, ACTION16_DIM)
        if not len(actions):
            raise ValueError("策略 waypoint chunk 不能为空")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("waypoint batch_size 必须大于 0")
        results = []
        for begin in range(0, len(actions), batch_size):
            end = min(begin + batch_size, len(actions))
            results.append(
                self.execute_policy_waypoint_batch(
                    actions[begin:end],
                    first_in_chunk=begin == 0,
                    last_in_chunk=end == len(actions),
                )
            )
        combined = {
            key: np.concatenate([result[key] for result in results], axis=0)
            for key in (
                "targets",
                "start_targets",
                "action_start_times",
                "action_arrival_times",
            )
        }
        combined["finished_at"] = float(results[-1]["finished_at"])
        return combined

    def _policy_action_to_target(
        self,
        action16: np.ndarray,
        *,
        reference: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action16, np.float32).reshape(ACTION16_DIM)
        reference = np.asarray(reference, np.float32).reshape(ACTION16_DIM)
        if not np.isfinite(action).all():
            raise ValueError("Robot command contains NaN/Inf")
        if self._configured_action_representation() == "delta":
            target = self._delta_to_target(action, reference=reference)
        else:
            target = self._delta_to_target(
                self._absolute_to_delta(reference, action), reference=reference
            )
        self._validate_step(self._absolute_to_delta(reference, target), target)
        return target

    def _execute_absolute_waypoint_trajectory(
        self,
        targets16: np.ndarray,
        step_durations: list[float],
    ) -> None:
        targets_array = np.asarray(targets16, np.float32).reshape(-1, ACTION16_DIM)
        if len(targets_array) != len(step_durations) or not len(targets_array):
            raise ValueError("Waypoint targets and durations must have equal non-zero lengths")
        time_list = []
        elapsed = 0.0
        for duration in step_durations:
            elapsed += float(duration)
            time_list.append(elapsed)
        self._execute_absolute_waypoint_schedule(targets_array, time_list)

    def _execute_absolute_waypoint_schedule(
        self,
        targets16: np.ndarray,
        time_list: list[float],
        *,
        history_start_index: int = 0,
    ) -> float:
        targets_array = np.asarray(targets16, np.float32).reshape(-1, ACTION16_DIM)
        if len(targets_array) != len(time_list) or not len(targets_array):
            raise ValueError("Waypoint targets and times must have equal non-zero lengths")
        times = np.asarray(time_list, np.float64)
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("Waypoint times must be non-negative and monotonic")
        history_start_index = int(history_start_index)
        if not 0 <= history_start_index < len(targets_array):
            raise ValueError("Invalid waypoint history_start_index")

        names = [
            self.robot.torso_name,
            self.robot.arm_left_name,
            self.robot.effector_left_name,
            self.robot.arm_right_name,
            self.robot.effector_right_name,
        ]
        torso_pose = getattr(self, "_policy_torso_pose", None)
        if torso_pose is None:
            torso_pose = self.robot.get_desired_cartesian_pose([self.robot.torso_name])[0]
            self._policy_torso_pose = list(torso_pose)
        waypoints = []
        reference = self._last_target.copy()
        for target in targets_array:
            self._validate_step(self._absolute_to_delta(reference, target), target)
            arm_poses, grippers = action16_to_sdk_commands(
                target, use_xyzw=self.config.use_xyzw
            )
            waypoints.append([
                list(torso_pose),
                np.asarray(arm_poses[0]).tolist(),
                np.asarray(grippers[0]).tolist(),
                np.asarray(arm_poses[1]).tolist(),
                np.asarray(grippers[1]).tolist(),
            ])
            reference = target

        submitted_at = time.time()
        self.robot.move_cartesian_waypoints(
            names,
            waypoints,
            times.tolist(),
            use_wbc=True,
            add_default_torso=False,
        )
        self._last_target = targets_array[-1].copy()
        self._append_action_history(targets_array[history_start_index:])
        return submitted_at

    def execute_rollback_waypoints(
        self,
        targets16: np.ndarray,
        *,
        step_duration_s: float,
    ) -> None:
        """Submit a reversed absolute chunk as one continuous SDK trajectory."""
        targets = np.asarray(targets16, np.float32).reshape(-1, ACTION16_DIM)
        reference = self._last_target.copy()
        safe_targets = []
        for target in targets:
            safe_target = self._delta_to_target(
                self._absolute_to_delta(reference, target),
                reference=reference,
            )
            safe_targets.append(safe_target)
            reference = safe_target
        duration = max(float(step_duration_s), 1e-3)
        self._execute_absolute_waypoint_trajectory(
            np.asarray(safe_targets, np.float32),
            [duration] * len(safe_targets),
        )

    def execute_absolute(self, action16: np.ndarray) -> None:
        """Send an absolute target and make it the base for subsequent deltas."""
        target = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        if not np.isfinite(target).all():
            raise ValueError("Robot target contains NaN/Inf")
        reference = getattr(self, "_last_target", None)
        if reference is None:
            reference = self.state_action16()
        # Reuse the workspace/min-z clamps used for a policy delta without
        # altering an already absolute pose by applying a nonzero delta.
        target = self._delta_to_target(
            self._absolute_to_delta(reference, target),
            reference=reference,
        )
        self._validate_step(self._absolute_to_delta(reference, target), target)
        self._send_target(target)
        self._last_target = target.copy()
        self._append_action_history(target)

    @staticmethod
    def _slerp_action_quat(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        a = np.asarray(a, np.float32).reshape(4)
        b = np.asarray(b, np.float32).reshape(4)
        a /= max(float(np.linalg.norm(a)), 1e-8)
        b /= max(float(np.linalg.norm(b)), 1e-8)
        dot = float(np.dot(a, b))
        if dot < 0.0:
            b = -b
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            value = a + float(alpha) * (b - a)
            return (value / max(float(np.linalg.norm(value)), 1e-8)).astype(np.float32)
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        value = (
            np.sin((1.0 - float(alpha)) * theta) / sin_theta * a
            + np.sin(float(alpha) * theta) / sin_theta * b
        )
        return value.astype(np.float32)

    def execute_takeover_absolute(
        self,
        action16: np.ndarray,
        *,
        arm_command_mask: dict[str, bool] | None = None,
    ) -> None:
        """Stream a rate-limited teleoperation target anchored on measured state."""
        if self._takeover_limited_target is None:
            self.begin_takeover()
        assert self._takeover_limited_target is not None
        previous = self._takeover_limited_target
        target = np.asarray(action16, np.float32).reshape(ACTION16_DIM).copy()
        active = arm_command_mask or {"left": True, "right": True}
        if active.get("right", False):
            target = apply_right_gripper_orientation_constraint(
                target,
                enabled=self.config.right_gripper_angle_constraint_during_takeover,
                use_xyzw=self.config.use_xyzw,
                target_angle_deg=self.config.right_gripper_target_angle_deg,
                ray_axis=self.config.right_gripper_ray_axis,
                level_axis=self.config.right_gripper_level_axis,
                keep_level_axis_horizontal=self.config.right_gripper_twist_level_constraint,
            )
        limited = target.copy()
        for hand, off in (("left", 0), ("right", 8)):
            if not active.get(hand, False):
                limited[off : off + 8] = previous[off : off + 8]
                continue
            delta_xyz = target[off : off + 3] - previous[off : off + 3]
            distance = float(np.linalg.norm(delta_xyz))
            max_translation = float(self.config.takeover_max_translation_step_m)
            if max_translation > 0.0 and distance > max_translation:
                limited[off : off + 3] = (
                    previous[off : off + 3] + delta_xyz * (max_translation / distance)
                )
            angle = np.deg2rad(self._quat_angle_deg(
                previous[off + 3 : off + 7], target[off + 3 : off + 7]
            ))
            max_rotation = np.deg2rad(float(self.config.takeover_max_rotation_step_deg))
            if max_rotation > 0.0 and angle > max_rotation:
                limited[off + 3 : off + 7] = self._slerp_action_quat(
                    previous[off + 3 : off + 7],
                    target[off + 3 : off + 7],
                    max_rotation / angle,
                )
            max_gripper = float(self.config.takeover_max_gripper_step)
            if max_gripper > 0.0:
                gripper_delta = float(target[off + 7] - previous[off + 7])
                limited[off + 7] = previous[off + 7] + np.clip(
                    gripper_delta, -max_gripper, max_gripper
                )
        if active.get("right", False) and self.config.right_min_z is not None:
            minimum_z = np.float32(self.config.right_min_z)
            limited[10] = max(
                limited[10], np.nextafter(minimum_z, np.float32(np.inf))
            )
        limited[[7, 15]] = np.clip(limited[[7, 15]], 0.0, 1.0)
        delta = self._absolute_to_delta(previous, limited)
        self._validate_step(delta, limited, arm_command_mask=active)
        # Submit a complete mixed command so the SDK receives one atomic
        # Cartesian+gripper update. Inactive arms were pinned to `previous`
        # above, so they hold their takeover-start targets.
        self._send_target(limited)
        self._takeover_limited_target = limited.copy()
        self._last_target = limited.copy()
        self._append_action_history(limited)

    def reset_history(self, action16: np.ndarray) -> None:
        target = np.asarray(action16, np.float32).reshape(16).copy()
        lock = getattr(self, "_history_lock", None)
        if lock is None:
            self.action_history.clear()
            self.action_history.append(target)
            self.observation_history.clear()
        else:
            with lock:
                self.action_history.clear()
                self.action_history.append(target)
                self.observation_history.clear()
        self._last_target = target.copy()
        self._policy_chunk_count = 0
        self._policy_torso_pose = None


class WanVAPolicy:
    def __init__(self, *, host: str, port: int, prompt: str, state_history_len: int = 16,
                 obs_history_len: int = 9, replan_steps: int = 8, fake: bool = False,
                 control_left_arm: bool = True, video_guidance_scale: float = 1.0,
                 action_guidance_scale: float = 1.0,
                 action_representation: str = "delta") -> None:
        self.prompt = prompt
        self.replan_steps = int(replan_steps)
        self.control_left_arm = bool(control_left_arm)
        self.action_representation = validate_action_representation(action_representation)
        self.client = WAM4DPriorClient(
            host=host, port=port, prompt=prompt, state_history_len=state_history_len,
            obs_history_len=obs_history_len,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
            action_representation=self.action_representation,
            fake=fake,
        )
        self._executed_server_action_count = 0
        self.last_inference_started_chunk = False

    def reset(self, observation: dict[str, Any] | None = None) -> None:
        self.client.reset()
        self._executed_server_action_count = 0
        self.last_inference_started_chunk = False

    def _action_for_execution(
        self,
        action16: np.ndarray,
        *,
        current_command16: np.ndarray | None,
    ) -> np.ndarray:
        """Log the server's de-normalized action immediately before use."""
        action = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        self._executed_server_action_count += 1
        left_xyz = action[0:3]
        right_xyz = action[8:11]
        print(
            "WAM4D 服务端动作 "
            f"#{self._executed_server_action_count} "
            f"（反归一化后的 {self.action_representation} xyz）："
            f"左臂=[{left_xyz[0]:+.5f}, {left_xyz[1]:+.5f}, {left_xyz[2]:+.5f}] "
            f"右臂=[{right_xyz[0]:+.5f}, {right_xyz[1]:+.5f}, {right_xyz[2]:+.5f}]",
            flush=True,
        )
        if not self.control_left_arm:
            action = action.copy()
            current = None
            if current_command16 is not None:
                current = np.asarray(current_command16, dtype=np.float32).reshape(ACTION16_DIM)
            if self.action_representation == "delta":
                action[0:3] = 0.0
                action[3:7] = [1.0, 0.0, 0.0, 0.0]
                if current is not None:
                    action[7] = current[7]
            elif current is not None:
                action[0:8] = current[0:8]
            print("WAM4D 策略：左臂和左夹爪已锁定，仅执行右臂动作。", flush=True)
        return action

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        current_command16 = observation.get("state_action16")
        self.last_inference_started_chunk = True
        payload = dict(observation["wam4d"])
        executed_history = payload.get("observation.executed_action_history")
        if executed_history is not None and len(executed_history):
            current_command16 = np.asarray(executed_history, np.float32)[-1]
        payload["task"] = self.prompt
        chunk = self.client.infer_prior_chunk(
            payload,
            fallback_state16=observation.get("state_action16"),
            max_steps=self.replan_steps,
        )
        return np.stack([
            self._action_for_execution(action, current_command16=current_command16)
            for action in np.asarray(chunk, np.float32).reshape(-1, ACTION16_DIM)
        ])


class QuestControlSource:
    """Maps right-controller B/A edges and middle-trigger motion to InputState."""

    def __init__(self, robot: AstribotRobotIO, *, state_url: str,
                 trigger_threshold: float = 0.5, button_a_index: int = 4,
                 button_b_index: int = 5,
                 gripper_trigger_threshold: float = 0.2) -> None:
        self.robot = robot
        self.quest = QuestResidualIntervention(
            state_url=state_url, trigger_threshold=trigger_threshold,
            gripper_threshold=gripper_trigger_threshold, position_scale=1.0,
            residual_position_scale=0.2, residual_rotation_scale=np.deg2rad(30),
            episode_button_hand="right", success_button_index=button_a_index,
            failure_button_index=button_b_index, episode_button_threshold=0.5,
        )
        self.anchor: np.ndarray | None = None
        self._right_quest_rotation_xyzw: np.ndarray | None = None
        self._right_target_rotation_xyzw: np.ndarray | None = None
        self.consecutive_errors = 0

    def reset(self) -> None:
        self.quest.reset()
        self.anchor = None
        self._right_quest_rotation_xyzw = None
        self._right_target_rotation_xyzw = None
        self.consecutive_errors = 0

    def _current_command16(self) -> np.ndarray:
        command_target = getattr(self.robot, "command_target16", None)
        if callable(command_target):
            return np.asarray(command_target(), np.float32).reshape(ACTION16_DIM).copy()
        assert self.anchor is not None
        return self.anchor.copy()

    def _apply_right_rotation_delta(
        self,
        target: np.ndarray,
        cumulative_rotvec: np.ndarray,
    ) -> None:
        current_quest_rotation = rotvec_to_quat_xyzw(cumulative_rotvec)
        previous_quest_rotation = getattr(self, "_right_quest_rotation_xyzw", None)
        if previous_quest_rotation is None:
            step_delta = current_quest_rotation
        else:
            step_delta = quat_multiply_xyzw(
                quat_inverse_xyzw(previous_quest_rotation),
                current_quest_rotation,
            )

        current_target_rotation = getattr(self, "_right_target_rotation_xyzw", None)
        if current_target_rotation is None:
            current_command = self._current_command16()
            current_target_rotation = action_quat_to_sdk_xyzw(
                current_command[11:15], use_xyzw=self.robot.config.use_xyzw
            )
        desired_rotation = quat_multiply_xyzw(current_target_rotation, step_delta)
        target[11:15] = sdk_xyzw_to_action_quat(
            desired_rotation, use_xyzw=self.robot.config.use_xyzw
        )
        target[:] = apply_right_gripper_orientation_constraint(
            target,
            enabled=self.robot.config.right_gripper_angle_constraint_during_takeover,
            use_xyzw=self.robot.config.use_xyzw,
            target_angle_deg=self.robot.config.right_gripper_target_angle_deg,
            ray_axis=self.robot.config.right_gripper_ray_axis,
            level_axis=self.robot.config.right_gripper_level_axis,
            keep_level_axis_horizontal=self.robot.config.right_gripper_twist_level_constraint,
        )
        self._right_quest_rotation_xyzw = current_quest_rotation.copy()
        self._right_target_rotation_xyzw = action_quat_to_sdk_xyzw(
            target[11:15], use_xyzw=self.robot.config.use_xyzw
        )

    def _expert_action(self, residual: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if self.anchor is None:
            self.anchor = self.robot.state_action16().copy()
        target = self.anchor.copy()
        for hand, off, roff in (("left", 0, 0), ("right", 8, 7)):
            hand_info = info.get(hand, {})
            if not hand_info.get("active", False):
                if hand == "right":
                    self._right_quest_rotation_xyzw = None
                    self._right_target_rotation_xyzw = None
                continue
            delta_xyz = np.asarray(hand_info.get("relative_position", residual[roff:roff+3] * .2))
            delta_rot = np.asarray(hand_info.get("scaled_rotvec", residual[roff+3:roff+6] * np.deg2rad(30)))
            target[off:off+3] = self.anchor[off:off+3] + delta_xyz
            if hand == "right":
                self._apply_right_rotation_delta(target, delta_rot)
            else:
                base_q = action_quat_to_sdk_xyzw(
                    self.anchor[off+3:off+7], use_xyzw=self.robot.config.use_xyzw
                )
                target_q = quat_multiply_xyzw(base_q, rotvec_to_quat_xyzw(delta_rot))
                target[off+3:off+7] = sdk_xyzw_to_action_quat(
                    target_q, use_xyzw=self.robot.config.use_xyzw
                )
            if "index" in hand_info:
                target[off + 7] = index_trigger_to_gripper_action(
                    float(hand_info["index"]), self.quest.gripper_threshold
                )
        return target.astype(np.float32)

    def poll(self) -> InputState:
        residual, active, info = self.quest.get_residual_action()
        if "quest_error" in info:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 5:
                raise ConnectionError(
                    f"Quest state unavailable for {self.consecutive_errors} polls: {info['quest_error']}"
                )
        else:
            self.consecutive_errors = 0
        buttons = info.get("episode_buttons", {})
        active_arms = info.get("active_arms")
        if not isinstance(active_arms, dict):
            active_arms = {
                hand: bool((info.get(hand) or {}).get("active", False))
                for hand in ("left", "right")
            }
        if not active:
            self.anchor = None
            self._right_quest_rotation_xyzw = None
            self._right_target_rotation_xyzw = None
        return InputState(
            b=bool(buttons.get("failure_value", 0) >= .5),
            a=bool(buttons.get("success_value", 0) >= .5),
            middle=1.0 if active else 0.0,
            expert_action=self._expert_action(residual, info) if active else None,
            active_arms={hand: bool(active_arms.get(hand, False)) for hand in ("left", "right")},
        )


class FakeAstribotRobotIO:
    """Deterministic adapter for deployment smoke tests."""

    def __init__(self, action_representation: str = "delta") -> None:
        self.action_representation = validate_action_representation(action_representation)
        self.action = np.zeros(16, np.float32)
        self.action[[3, 11]] = 1
        self.step = 0

    def state_action16(self) -> np.ndarray:
        return self.action.copy()

    def command_target16(self) -> np.ndarray:
        return self.action.copy()

    def observe(self) -> dict[str, Any]:
        self.step += 1
        image = np.zeros((8, 8, 3), np.uint8)
        payload = {
            "observation.images.cam_high": image,
            "observation.images.cam_left_wrist": image,
            "observation.images.cam_right_wrist": image,
            "observation.state": np.asarray([self.action]),
            "observation.executed_action_history": np.asarray([self.action]),
            "task": "fake",
        }
        return {"state_action16": self.action.copy(), "wam4d": payload, "step": self.step}

    def execute(self, action16: np.ndarray) -> None:
        if self.action_representation == "absolute":
            self.execute_absolute(action16)
            return
        delta = np.asarray(action16, np.float32).reshape(16)
        target = self.action.copy()
        target[0:7] = apply_relative_pose7(self.action[0:7], delta[0:7])
        target[7] = delta[7]
        target[8:15] = apply_relative_pose7(self.action[8:15], delta[8:15])
        target[15] = delta[15]
        self.action = target

    def execute_absolute(self, action16: np.ndarray) -> None:
        self.action = np.asarray(action16, np.float32).reshape(16).copy()

    def execute_policy_waypoints(self, actions16: np.ndarray) -> np.ndarray:
        targets = []
        for action in np.asarray(actions16, np.float32).reshape(-1, 16):
            self.execute(action)
            targets.append(self.action.copy())
        return np.asarray(targets, np.float32)

    def execute_policy_waypoint_batch(
        self,
        actions16: np.ndarray,
        *,
        first_in_chunk: bool,
        last_in_chunk: bool,
    ) -> dict[str, Any]:
        actions = np.asarray(actions16, np.float32).reshape(-1, 16)
        targets = []
        starts = []
        started_at = time.time()
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

    def execute_policy_waypoint_batches(
        self,
        actions16: np.ndarray,
        *,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        actions = np.asarray(actions16, np.float32).reshape(-1, 16)
        results = []
        for begin in range(0, len(actions), int(batch_size)):
            end = min(begin + int(batch_size), len(actions))
            results.append(
                self.execute_policy_waypoint_batch(
                    actions[begin:end],
                    first_in_chunk=begin == 0,
                    last_in_chunk=end == len(actions),
                )
            )
        combined = {
            key: np.concatenate([result[key] for result in results], axis=0)
            for key in (
                "targets",
                "start_targets",
                "action_start_times",
                "action_arrival_times",
            )
        }
        combined["finished_at"] = float(results[-1]["finished_at"])
        return combined

    def execute_policy_step(
        self,
        action16: np.ndarray,
        *,
        first_in_chunk: bool,
        last_in_chunk: bool,
    ) -> np.ndarray:
        self.execute(action16)
        return self.action.copy()

    def reset_history(self, action16: np.ndarray) -> None:
        self.action = np.asarray(action16, np.float32).reshape(16).copy()
