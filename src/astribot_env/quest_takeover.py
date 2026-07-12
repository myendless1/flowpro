from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np

from astribot_env.initial_pose import default_init_joint_action, normalize_init_joint_action
from astribot_env.quest_intervention import QuestResidualIntervention
from astribot_env.rgbd import get_current_eef_state
from astribot_env.sdk_loader import DEFAULT_ASTRIBOT_SDK_ROOT, load_astribot_class
from astribot_env.utils import (
    ACTION16_DIM,
    RESIDUAL14_DIM,
    action16_to_sdk_commands,
    action_quat_to_sdk_xyzw,
    apply_right_gripper_orientation_constraint,
    apply_xyz_limits,
    convert_gripper_cmd_value_to_action_value,
    quat_multiply_xyzw,
    rotvec_to_quat_xyzw,
    sdk_xyzw_to_action_quat,
)


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32, copy=False)


def _slerp_xyzw(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = _normalize_quat_xyzw(a)
    b = _normalize_quat_xyzw(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat_xyzw(a + alpha * (b - a))

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    scale_a = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    scale_b = np.sin(theta) / sin_theta_0
    return _normalize_quat_xyzw(scale_a * a + scale_b * b)


def _quat_angle_between_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_quat_xyzw(a)
    b = _normalize_quat_xyzw(b)
    dot = abs(float(np.dot(a, b)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _format_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.4f}" for v in np.asarray(values).reshape(-1)) + "]"


@dataclass
class QuestTakeoverConfig:
    sdk_root: str = str(DEFAULT_ASTRIBOT_SDK_ROOT)
    robot_type: str = "S1"
    sdk_frequency: float = 100.0
    high_control_rights: bool = True
    robot_command_enabled: bool = True
    control_way: str = "filter"
    cartesian_frame: str = "chassis"
    robot_filter_scale: float = 0.1
    robot_gripper_filter_scale: float = 0.5
    use_xyzw: bool = False
    use_wbc_during_takeover: bool = False
    right_arm_min_z: float | None = 0.862
    right_gripper_angle_constraint_during_takeover: bool = True
    right_gripper_target_angle_deg: float = 45.0
    right_gripper_ray_axis: str = "+z"
    right_gripper_twist_level_constraint: bool = True
    right_gripper_level_axis: str = "+x"
    left_xyz_low: tuple[float, float, float] | None = None
    left_xyz_high: tuple[float, float, float] | None = None
    right_xyz_low: tuple[float, float, float] | None = None
    right_xyz_high: tuple[float, float, float] | None = None
    residual_position_scale: float = 0.2
    residual_rotation_scale: float = 0.5235987755982988
    residual_gripper_deadband: float = 0.5
    quest_state_url: str = ""
    quest_trigger_threshold: float = 0.5
    quest_gripper_threshold: float = 0.2
    quest_position_scale: float = 1.0
    quest_rotation_gain: float = 1.0
    quest_rotation_mode: str = "full"
    quest_timeout: float = 0.05
    quest_takeover_release_grace: float = 0.35
    quest_episode_button_hand: str = "right"
    quest_success_button_index: int = 4
    quest_failure_button_index: int = 5
    quest_episode_button_threshold: float = 0.5
    quest_neutral_gripper_when_released: bool = True
    quest_verify_ssl: bool = False
    quest_control_rate_hz: float = 100.0
    quest_poll_rate_hz: float = 100.0
    quest_stream_gripper_every_tick: bool = False
    quest_max_translation_step_m: float = 0.01
    quest_max_rotation_step_deg: float = 2.5
    quest_sync_all_arm_targets_on_takeover: bool = False
    init_joint_action: list[list[float]] = field(default_factory=default_init_joint_action)
    initial_joint_duration: float = 4.0
    reset_to_initial_on_start: bool = True
    reset_to_initial_on_episode_end: bool = True
    debug_takeover_actions: bool = False
    debug_sdk_command_state: bool = False

    @classmethod
    def from_env_config(cls, env_config: Any) -> "QuestTakeoverConfig":
        fields = cls.__dataclass_fields__
        values = {
            name: getattr(env_config, name)
            for name in fields
            if hasattr(env_config, name)
        }
        return cls(**values)


class QuestTakeoverController:
    def __init__(self, config: QuestTakeoverConfig) -> None:
        self.config = config
        self._stop = threading.Event()
        self._quest_lock = threading.RLock()
        self._sdk_command_lock = threading.RLock()
        self._latest_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._latest_quest_intervened = False
        self._latest_quest_info: dict[str, Any] = {}
        self._latest_quest_active_arms = {"left": False, "right": False}
        self._last_quest_takeover_time: float | None = None
        self._last_active_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._last_active_quest_info: dict[str, Any] = {}
        self._last_active_quest_active_arms = {"left": False, "right": False}
        self._takeover_arm_anchor_poses = {"left": None, "right": None}
        self._takeover_limited_action: np.ndarray | None = None
        self._latest_takeover_command_error = ""
        self._latest_takeover_command_sent = False
        self._takeover_control_active = False
        self._takeover_control_sent_count = 0
        self._last_status_print = 0.0
        self._last_streaming_command_time: float | None = None

        self.astribot = None
        self._initial_joint_target = normalize_init_joint_action(self.config.init_joint_action)
        self.quest = QuestResidualIntervention(
            state_url=self.config.quest_state_url,
            trigger_threshold=self.config.quest_trigger_threshold,
            gripper_threshold=self.config.quest_gripper_threshold,
            position_scale=self.config.quest_position_scale,
            residual_position_scale=self.config.residual_position_scale,
            residual_rotation_scale=self.config.residual_rotation_scale,
            residual_gripper_deadband=self.config.residual_gripper_deadband,
            rotation_gain=self.config.quest_rotation_gain,
            rotation_mode=self.config.quest_rotation_mode,
            timeout=self.config.quest_timeout,
            episode_button_hand=self.config.quest_episode_button_hand,
            success_button_index=self.config.quest_success_button_index,
            failure_button_index=self.config.quest_failure_button_index,
            episode_button_threshold=self.config.quest_episode_button_threshold,
            neutral_gripper_when_released=self.config.quest_neutral_gripper_when_released,
            verify_ssl=self.config.quest_verify_ssl,
        )

    def start(self) -> None:
        if not self.config.quest_state_url:
            raise ValueError("Quest takeover requires quest_state_url.")
        self._connect_robot()
        if self.config.robot_command_enabled and self.config.reset_to_initial_on_start:
            self._move_to_initial_joint_pose(reason="standalone takeover startup")
        self.quest.reset()
        self._stop.clear()
        print(
            "Standalone Quest takeover started: "
            f"quest={self.config.quest_state_url} "
            f"poll={float(self.config.quest_poll_rate_hz):.1f}Hz "
            f"control={float(self.config.quest_control_rate_hz):.1f}Hz "
            f"robot_commands={bool(self.config.robot_command_enabled)}",
            flush=True,
        )
        self._run_loop()

    def stop(self) -> None:
        self._stop.set()

    def _connect_robot(self) -> None:
        if not self.config.robot_command_enabled:
            print("Robot command output disabled; Quest takeover will run as a dry poll loop.", flush=True)
            return
        import os

        os.environ.setdefault("ROBOT_TYPE", self.config.robot_type)
        Astribot = load_astribot_class(self.config.sdk_root)
        self.astribot = Astribot(
            freq=float(self.config.sdk_frequency),
            high_control_rights=self.config.high_control_rights,
        )
        if hasattr(self.astribot, "set_filter_parameters"):
            self.astribot.set_filter_parameters(
                float(self.config.robot_filter_scale),
                float(self.config.robot_gripper_filter_scale),
            )

    def _run_loop(self) -> None:
        poll_interval = 1.0 / max(float(self.config.quest_poll_rate_hz), 1e-6)
        control_interval = 1.0 / max(float(self.config.quest_control_rate_hz), 1e-6)
        next_poll = time.monotonic()
        next_control = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_poll:
                    self._poll_quest_residual(apply_release_grace=True)
                    next_poll = max(next_poll + poll_interval, now + poll_interval)
                if now >= next_control:
                    self._control_tick()
                    next_control = max(next_control + control_interval, now + control_interval)
                self._stop.wait(max(0.0, min(next_poll, next_control) - time.monotonic()))
        finally:
            self._reset_takeover_control_state()

    def _control_tick(self) -> None:
        residual, intervened, info = self._get_latest_quest_snapshot(apply_release_grace=True)
        outcome = self.quest.pop_episode_outcome()
        if outcome is not None:
            print(f"Quest episode outcome: {outcome}", flush=True)
            if self.config.reset_to_initial_on_episode_end:
                self._reset_takeover_control_state()
                self._move_to_initial_joint_pose()
            return
        if intervened:
            self._send_direct_takeover_action(
                residual,
                quest_info=info,
                arm_command_mask=self._active_arms_from_quest_info(info),
            )
            self._print_status(info, active=True)
        else:
            if self._takeover_control_active or self._latest_takeover_command_sent:
                self._reset_takeover_control_state()
                self._print_status(info, active=False)

    @staticmethod
    def _active_arms_from_quest_info(info: dict[str, Any]) -> dict[str, bool]:
        active_arms = info.get("active_arms")
        if isinstance(active_arms, dict):
            return {hand: bool(active_arms.get(hand, False)) for hand in ("left", "right")}
        return {
            hand: bool((info.get(hand) or {}).get("active", False))
            for hand in ("left", "right")
        }

    def _poll_quest_residual(self, *, apply_release_grace: bool = False) -> tuple[np.ndarray, bool, dict[str, Any]]:
        residual, intervened, info = self.quest.get_residual_action()
        residual = np.asarray(residual, dtype=np.float32).reshape(RESIDUAL14_DIM).copy()
        raw_intervened = bool(intervened)
        now = time.monotonic()
        with self._quest_lock:
            if raw_intervened:
                self._last_quest_takeover_time = now
                self._latest_quest_active_arms = self._active_arms_from_quest_info(dict(info))
                self._last_active_quest_residual = residual.copy()
                self._last_active_quest_info = dict(info)
                self._last_active_quest_active_arms = dict(self._latest_quest_active_arms)
            elif (
                apply_release_grace
                and self._last_quest_takeover_time is not None
                and now - self._last_quest_takeover_time <= float(self.config.quest_takeover_release_grace)
            ):
                residual = self._last_active_quest_residual.copy()
                intervened = True
                info = dict(self._last_active_quest_info or info)
                info["takeover_release_grace"] = True
                info["raw_intervened"] = False
                info["active_arms"] = dict(self._last_active_quest_active_arms)
            else:
                intervened = False
                self._latest_quest_active_arms = {"left": False, "right": False}
            self._latest_quest_residual = residual.copy()
            self._latest_quest_intervened = bool(intervened)
            self._latest_quest_info = dict(info)
            return self._latest_quest_residual.copy(), self._latest_quest_intervened, dict(self._latest_quest_info)

    def _get_latest_quest_snapshot(self, *, apply_release_grace: bool = False) -> tuple[np.ndarray, bool, dict[str, Any]]:
        with self._quest_lock:
            residual = self._latest_quest_residual.copy()
            intervened = bool(self._latest_quest_intervened)
            info = dict(self._latest_quest_info)
            now = time.monotonic()
            if (
                not intervened
                and apply_release_grace
                and self._last_quest_takeover_time is not None
                and now - self._last_quest_takeover_time <= float(self.config.quest_takeover_release_grace)
            ):
                residual = self._last_active_quest_residual.copy()
                intervened = True
                info = dict(self._last_active_quest_info or info)
                info["takeover_release_grace"] = True
                info["raw_intervened"] = False
                info["active_arms"] = dict(self._last_active_quest_active_arms)
            return residual, intervened, info

    def _send_direct_takeover_action(
        self,
        residual_action: np.ndarray,
        *,
        quest_info: dict[str, Any],
        arm_command_mask: dict[str, bool],
    ) -> None:
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        with self._sdk_command_lock:
            active_mask = {
                hand: bool(arm_command_mask.get(hand, False))
                for hand in ("left", "right")
            }
            if not any(active_mask.values()):
                self._takeover_control_active = False
                return

            if self._takeover_limited_action is None:
                current_action = self._read_current_state16()
                self._takeover_limited_action = current_action.copy()
                if bool(self.config.quest_sync_all_arm_targets_on_takeover):
                    self._sync_takeover_targets_to_current_pose(current_action)
            target_action = self._takeover_limited_action.copy()

            for hand, pose_start in (("left", 0), ("right", 8)):
                if not active_mask.get(hand, False):
                    self._takeover_arm_anchor_poses[hand] = None
                    continue
                if self._takeover_arm_anchor_poses[hand] is None:
                    anchor_pose = self._read_current_arm_action_pose(hand)
                    self._takeover_arm_anchor_poses[hand] = anchor_pose.copy()
                    target_action[pose_start : pose_start + 7] = anchor_pose
                    self._takeover_limited_action[pose_start : pose_start + 7] = anchor_pose
                anchor_pose = np.asarray(self._takeover_arm_anchor_poses[hand], dtype=np.float32).reshape(7)
                target_action[pose_start : pose_start + 7] = self._build_direct_takeover_pose(
                    hand,
                    anchor_pose,
                    residual_action,
                    quest_info,
                )

            self._apply_direct_takeover_grippers(target_action, residual_action)
            constrained = self._constrain_action(target_action)
            limited_action = self._limit_takeover_action_step(constrained)
            if self.config.robot_command_enabled:
                self._send_streaming_action(
                    limited_action,
                    arm_command_mask=None,
                    use_wbc=self.config.use_wbc_during_takeover,
                )
            self._latest_takeover_command_sent = bool(self.config.robot_command_enabled)
            self._latest_takeover_command_error = ""
            self._takeover_control_sent_count += 1
            self._takeover_control_active = True

    def _build_direct_takeover_pose(
        self,
        hand: str,
        anchor_pose: np.ndarray,
        residual_action: np.ndarray,
        quest_info: dict[str, Any],
    ) -> np.ndarray:
        residual_start = 0 if hand == "left" else 7
        hand_info = quest_info.get(hand)
        if not isinstance(hand_info, dict):
            relative_position = (
                residual_action[residual_start : residual_start + 3]
                * float(self.config.residual_position_scale)
            )
            relative_rotvec = (
                residual_action[residual_start + 3 : residual_start + 6]
                * float(self.config.residual_rotation_scale)
            )
        else:
            relative_position = np.asarray(
                hand_info.get("relative_position", hand_info.get("robot_delta", [0.0, 0.0, 0.0])),
                dtype=np.float32,
            ).reshape(3)
            relative_rotvec = np.asarray(
                hand_info.get(
                    "scaled_rotvec",
                    hand_info.get("relative_rotvec", hand_info.get("rotvec", [0.0, 0.0, 0.0])),
                ),
                dtype=np.float32,
            ).reshape(3)

        pose = np.asarray(anchor_pose, dtype=np.float32).reshape(7).copy()
        pose[:3] = anchor_pose[:3] + relative_position
        anchor_quat_xyzw = action_quat_to_sdk_xyzw(anchor_pose[3:7], use_xyzw=self.config.use_xyzw)
        delta_quat_xyzw = rotvec_to_quat_xyzw(relative_rotvec)
        target_quat_xyzw = quat_multiply_xyzw(anchor_quat_xyzw, delta_quat_xyzw)
        pose[3:7] = sdk_xyzw_to_action_quat(target_quat_xyzw, use_xyzw=self.config.use_xyzw)
        return pose

    def _apply_direct_takeover_grippers(self, action16: np.ndarray, residual_action: np.ndarray) -> None:
        for grip_index, residual_index in ((7, 6), (15, 13)):
            action16[grip_index] = self._direct_takeover_gripper_value(
                float(residual_action[residual_index]),
                float(action16[grip_index]),
            )

    def _direct_takeover_gripper_value(self, grip_residual: float, current_value: float) -> float:
        deadband = float(self.config.residual_gripper_deadband)
        if grip_residual <= -deadband:
            return 1.0
        if grip_residual >= deadband:
            close_fraction = (grip_residual - deadband) / max(1.0 - deadband, 1e-6)
            return 1.0 - float(np.clip(close_fraction, 0.0, 1.0))
        return float(np.clip(current_value, 0.0, 1.0))

    def _constrain_action(self, action16: np.ndarray) -> np.ndarray:
        action = apply_xyz_limits(
            action16,
            left_low=self.config.left_xyz_low,
            left_high=self.config.left_xyz_high,
            right_low=self.config.right_xyz_low,
            right_high=self.config.right_xyz_high,
            right_min_z=None,
        )
        return apply_right_gripper_orientation_constraint(
            action,
            enabled=self.config.right_gripper_angle_constraint_during_takeover,
            use_xyzw=self.config.use_xyzw,
            target_angle_deg=self.config.right_gripper_target_angle_deg,
            ray_axis=self.config.right_gripper_ray_axis,
            level_axis=self.config.right_gripper_level_axis,
            keep_level_axis_horizontal=self.config.right_gripper_twist_level_constraint,
        )

    def _limit_takeover_action_step(self, target_action: np.ndarray) -> np.ndarray:
        target = np.asarray(target_action, dtype=np.float32).reshape(ACTION16_DIM)
        previous = self._takeover_limited_action
        if previous is None:
            self._takeover_limited_action = target.copy()
            return target.copy()

        limited = target.copy()
        previous = np.asarray(previous, dtype=np.float32).reshape(ACTION16_DIM)
        for pose_start in (0, 8):
            limited[pose_start : pose_start + 3] = self._limit_takeover_xyz_step(
                previous[pose_start : pose_start + 3],
                target[pose_start : pose_start + 3],
            )
            limited[pose_start + 3 : pose_start + 7] = self._limit_takeover_quat_step(
                previous[pose_start + 3 : pose_start + 7],
                target[pose_start + 3 : pose_start + 7],
            )
        self._takeover_limited_action = limited.copy()
        return limited

    def _limit_takeover_xyz_step(self, previous_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
        max_step = float(self.config.quest_max_translation_step_m)
        previous = np.asarray(previous_xyz, dtype=np.float32).reshape(3)
        target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        if max_step <= 0.0:
            return target
        delta = target - previous
        distance = float(np.linalg.norm(delta))
        if distance <= max_step or distance <= 1e-9:
            return target
        return (previous + delta * (max_step / distance)).astype(np.float32, copy=False)

    def _limit_takeover_quat_step(self, previous_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        max_step_deg = float(self.config.quest_max_rotation_step_deg)
        target_xyzw = action_quat_to_sdk_xyzw(target_quat, use_xyzw=self.config.use_xyzw)
        if max_step_deg <= 0.0:
            return sdk_xyzw_to_action_quat(target_xyzw, use_xyzw=self.config.use_xyzw)
        previous_xyzw = action_quat_to_sdk_xyzw(previous_quat, use_xyzw=self.config.use_xyzw)
        angle = _quat_angle_between_xyzw(previous_xyzw, target_xyzw)
        max_step = np.deg2rad(max_step_deg)
        if angle <= max_step or angle <= 1e-9:
            limited_xyzw = target_xyzw
        else:
            limited_xyzw = _slerp_xyzw(previous_xyzw, target_xyzw, max_step / angle)
        return sdk_xyzw_to_action_quat(limited_xyzw, use_xyzw=self.config.use_xyzw)

    def _read_state16(self) -> np.ndarray:
        if self.astribot is None:
            raise RuntimeError("Quest takeover requires Astribot to read desired cartesian pose.")
        arm_names = [self.astribot.arm_left_name, self.astribot.arm_right_name]
        desired_poses = self._read_desired_cartesian_action_poses(arm_names)
        joint_state = self.astribot.get_current_joints_position(
            names=[self.astribot.effector_left_name, self.astribot.effector_right_name]
        )
        left_gripper = convert_gripper_cmd_value_to_action_value(
            float(np.asarray(joint_state[0], dtype=np.float32).reshape(-1)[0])
        )
        right_gripper = convert_gripper_cmd_value_to_action_value(
            float(np.asarray(joint_state[1], dtype=np.float32).reshape(-1)[0])
        )
        return np.concatenate(
            [desired_poses[0], [left_gripper], desired_poses[1], [right_gripper]]
        ).astype(np.float32)

    def _read_current_state16(self) -> np.ndarray:
        if self.astribot is None:
            raise RuntimeError("Quest takeover requires Astribot to read current cartesian pose.")
        return get_current_eef_state(
            self.astribot,
            use_xyzw=self.config.use_xyzw,
            frame=self.config.cartesian_frame,
        )

    def _read_current_arm_action_pose(self, hand: str) -> np.ndarray:
        pose_start = 0 if hand == "left" else 8
        return self._read_current_state16()[pose_start : pose_start + 7].astype(np.float32, copy=True)

    def _read_desired_arm_action_pose(self, hand: str) -> np.ndarray:
        if self.astribot is None:
            raise RuntimeError("Quest takeover requires Astribot to read desired cartesian pose.")
        name = self.astribot.arm_left_name if hand == "left" else self.astribot.arm_right_name
        return self._read_desired_cartesian_action_poses([name])[0]

    def _read_desired_cartesian_action_poses(self, names: list[str]) -> list[np.ndarray]:
        if self.astribot is None or not hasattr(self.astribot, "get_desired_cartesian_pose"):
            raise RuntimeError("Astribot SDK does not provide get_desired_cartesian_pose.")
        try:
            try:
                poses = self.astribot.get_desired_cartesian_pose(
                    names=names,
                    frame=self.config.cartesian_frame,
                )
            except TypeError:
                poses = self.astribot.get_desired_cartesian_pose(names=names)
        except Exception as exc:
            raise RuntimeError(f"Failed to read desired cartesian pose for {names}: {exc!r}") from exc
        if len(poses) != len(names):
            raise RuntimeError(f"Expected {len(names)} desired cartesian poses, got {len(poses)}.")
        action_poses = []
        for name, pose_values in zip(names, poses):
            pose = np.asarray(pose_values, dtype=np.float32).reshape(-1)
            if pose.size < 7:
                raise RuntimeError(
                    f"Desired cartesian pose for {name} has invalid length {pose.size}; expected at least 7."
                )
            pose = pose[:7].copy()
            pose[3:7] = sdk_xyzw_to_action_quat(pose[3:7], use_xyzw=self.config.use_xyzw)
            action_poses.append(pose.astype(np.float32, copy=False))
        return action_poses

    def _send_streaming_action(
        self,
        action16: np.ndarray,
        *,
        arm_command_mask: dict[str, bool] | None,
        use_wbc: bool,
    ) -> None:
        if self.astribot is None:
            return
        arm_poses, grippers = action16_to_sdk_commands(action16, use_xyzw=self.config.use_xyzw)
        arm_names = [self.astribot.arm_left_name, self.astribot.arm_right_name]
        if arm_command_mask is None:
            enabled = [True, True]
        else:
            enabled = [bool(arm_command_mask.get("left", False)), bool(arm_command_mask.get("right", False))]
        command_arm_names = [name for name, active in zip(arm_names, enabled) if active]
        command_arm_poses = [pose for pose, active in zip(arm_poses, enabled) if active]
        gripper_names = [
            name
            for name, active in zip(
                [self.astribot.effector_left_name, self.astribot.effector_right_name],
                enabled,
            )
            if active
        ]
        gripper_targets = [target for target, active in zip(grippers, enabled) if active]

        command_names, command_types, command_list = self._merge_streaming_command(
            command_arm_names=command_arm_names,
            command_arm_poses=command_arm_poses,
            gripper_names=gripper_names,
            gripper_targets=gripper_targets,
        )
        if bool(getattr(self.config, "debug_takeover_actions", False)) and command_arm_names:
            xyz_parts = [
                f"{name} xyz={_format_vec(np.asarray(pose, dtype=np.float32)[:3])}"
                for name, pose in zip(command_arm_names, command_arm_poses)
            ]
            print("Quest takeover command " + " | ".join(xyz_parts), flush=True)
        if command_names:
            if not hasattr(self.astribot, "set_different_type_command"):
                raise RuntimeError(
                    "Astribot SDK must provide set_different_type_command so EEF and "
                    "gripper targets can be sent atomically."
                )
            self.astribot.set_different_type_command(
                command_names,
                command_types,
                command_list,
                control_way=self.config.control_way,
                use_wbc=use_wbc,
            )
        self._last_streaming_command_time = time.time()

    def _merge_streaming_command(
        self,
        *,
        command_arm_names: list[str],
        command_arm_poses: list[list[float]],
        gripper_names: list[str],
        gripper_targets: list[list[float]],
    ) -> tuple[list[str], list[str], list[list[float]]]:
        arm_by_name = {
            name: pose
            for name, pose in zip(command_arm_names, command_arm_poses)
        }
        gripper_by_name = {
            name: target
            for name, target in zip(gripper_names, gripper_targets)
        }
        ordered_names = []
        ordered_types = []
        ordered_commands = []
        for name in getattr(self.astribot, "whole_body_names", []):
            if name in arm_by_name:
                ordered_names.append(name)
                ordered_types.append("cartesian")
                ordered_commands.append(arm_by_name[name])
            elif name in gripper_by_name:
                ordered_names.append(name)
                ordered_types.append("joints")
                ordered_commands.append(gripper_by_name[name])

        if ordered_names:
            return ordered_names, ordered_types, ordered_commands

        names = list(command_arm_names) + list(gripper_names)
        types = ["cartesian"] * len(command_arm_names) + ["joints"] * len(gripper_names)
        commands = list(command_arm_poses) + list(gripper_targets)
        return names, types, commands

    def _sync_takeover_targets_to_current_pose(self, current_action16: np.ndarray) -> None:
        if self.astribot is None:
            return
        # The next control tick submits the current complete target atomically.
        # Do not emit an extra arm-only SDK command here.
        return

    def _move_to_initial_joint_pose(self, *, reason: str = "episode reset") -> None:
        if not self.config.robot_command_enabled or self._initial_joint_target is None or self.astribot is None:
            return
        print(
            f"Moving Astribot non-chassis joints to configured initial pose ({reason}).",
            flush=True,
        )
        with self._sdk_command_lock:
            self.astribot.move_joints_position(
                self.astribot.whole_body_names[1:],
                self._initial_joint_target,
                duration=float(self.config.initial_joint_duration),
                use_wbc=False,
            )

    def _reset_takeover_control_state(self) -> None:
        with self._sdk_command_lock:
            self._takeover_arm_anchor_poses = {"left": None, "right": None}
            self._takeover_limited_action = None
            self._takeover_control_active = False
            self._latest_takeover_command_sent = False
            self._latest_takeover_command_error = ""
            self._last_streaming_command_time = None

    def _print_status(self, info: dict[str, Any], *, active: bool) -> None:
        now = time.monotonic()
        if now - self._last_status_print < 0.5:
            return
        self._last_status_print = now
        arms = self._active_arms_from_quest_info(info)
        active_label = "".join(hand[0].upper() for hand, enabled in arms.items() if enabled) or "-"
        state = "ACTIVE" if active else "idle"
        print(
            f"Quest takeover {state}: arms={active_label} "
            f"commands={self._takeover_control_sent_count} "
            f"error={self._latest_takeover_command_error}",
            flush=True,
        )
