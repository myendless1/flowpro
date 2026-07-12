from __future__ import annotations

"""Concrete Astribot/Quest/Wan-VA adapters used by the preference collector.

Heavy robot and websocket dependencies are imported lazily so the collection
state machine and its fake mode remain testable on a development machine.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from astribot_env.quest_intervention import QuestResidualIntervention
from astribot_env.initial_pose import default_init_joint_action, normalize_init_joint_action
from astribot_env.rgbd import RGBDReader, build_wam4d_observation_payload, get_current_eef_state
from astribot_env.sdk_loader import DEFAULT_ASTRIBOT_SDK_ROOT, load_astribot_class
from astribot_env.utils import (
    ACTION16_DIM,
    action16_to_sdk_commands,
    action_quat_to_sdk_xyzw,
    convert_gripper_cmd_value_to_action_value,
    quat_multiply_xyzw,
    rotvec_to_quat_xyzw,
    sdk_xyzw_to_action_quat,
)
from astribot_env.wam4d_policy import WAM4DPriorClient
from wan_va.action_representation import apply_relative_pose7, relative_pose7

from .controller import InputState


@dataclass
class AstribotRuntimeConfig:
    sdk_root: str = ""
    robot_type: str = "S1"
    sdk_frequency: float = 100.0
    cartesian_frame: str = "chassis"
    control_way: str = "filter"
    use_xyzw: bool = False
    camera_timeout: float = 0.3
    image_from_s1_topic: bool = False
    max_translation_step_m: float = 0.04
    max_rotation_step_deg: float = 15.0
    takeover_max_translation_step_m: float = 0.01
    takeover_max_rotation_step_deg: float = 2.5
    first_policy_waypoint_duration: float = 0.6
    policy_waypoint_duration: float = 0.1
    init_joint_action: list[list[float]] = field(default_factory=default_init_joint_action)
    initial_joint_duration: float = 4.0
    reset_to_initial_on_startup: bool = True
    left_xyz_low: tuple[float, float, float] | None = None
    left_xyz_high: tuple[float, float, float] | None = None
    right_xyz_low: tuple[float, float, float] | None = None
    right_xyz_high: tuple[float, float, float] | None = None
    right_min_z: float | None = 0.862
    state_history_len: int = 16
    obs_history_len: int = 9


class AstribotRobotIO:
    """Delta-EEF adapter following Astribot's online Cartesian-control example."""

    def __init__(self, config: AstribotRuntimeConfig | None = None) -> None:
        import os

        self.config = config or AstribotRuntimeConfig()
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
        )
        self.action_history: deque[np.ndarray] = deque(maxlen=self.config.state_history_len)
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=self.config.obs_history_len)
        # Delta policy actions are integrated against the target actually sent
        # to the SDK, not noisy/lagged measured Cartesian feedback.
        self._last_target = self.state_action16()
        self._takeover_limited_target: np.ndarray | None = None
        self._policy_chunk_count = 0

    def _move_to_initial_joint_pose(self) -> None:
        target = normalize_init_joint_action(self.config.init_joint_action)
        self.robot.move_joints_position(
            self.robot.whole_body_names[1:], target,
            duration=self.config.initial_joint_duration, use_wbc=False,
        )

    def move_to_initial_pose(self) -> None:
        """Move all non-chassis joints home and rebase Cartesian control."""
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

    def begin_takeover(self) -> None:
        """Rebase teleoperation on measured state without treating servo lag as motion."""
        measured = self.state_action16()
        self._takeover_limited_target = measured.copy()
        self.reset_history(measured)

    def end_takeover(self) -> None:
        self._takeover_limited_target = None

    def observe(self) -> dict[str, Any]:
        state = self.state_action16()
        images = self.rgbd.get_bgr_images_dict()
        payload = build_wam4d_observation_payload(
            bgr_images=images,
            prompt="",
            action_history=self.action_history,
            history_len=self.action_history.maxlen or 16,
        )
        self.observation_history.append(payload)
        return {
            "state_action16": state,
            "wam4d": payload,
            "wam4d_history": list(self.observation_history),
            "time": time.time(),
        }

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
            target[off + 7] = delta[off + 7]
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

    def _validate_step(self, delta: np.ndarray, target: np.ndarray) -> None:
        if np.any(delta[[7, 15]] < 0) or np.any(delta[[7, 15]] > 1):
            raise ValueError(f"Gripper targets must be in [0,1], got {delta[[7, 15]]}")
        for off, low, high in (
            (0, self.config.left_xyz_low, self.config.left_xyz_high),
            (8, self.config.right_xyz_low, self.config.right_xyz_high),
        ):
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

    def _send_target(self, target: np.ndarray) -> None:
        arm_poses, grippers = action16_to_sdk_commands(target, use_xyzw=self.config.use_xyzw)
        arm_names = [self.robot.arm_left_name, self.robot.arm_right_name]
        grip_names = [self.robot.effector_left_name, self.robot.effector_right_name]
        if not hasattr(self.robot, "set_different_type_command"):
            raise RuntimeError(
                "Astribot SDK must provide set_different_type_command so EEF and "
                "gripper targets can be sent atomically."
            )
        commands = {
            arm_names[0]: ("cartesian", arm_poses[0]),
            arm_names[1]: ("cartesian", arm_poses[1]),
            grip_names[0]: ("joints", grippers[0]),
            grip_names[1]: ("joints", grippers[1]),
        }
        order = [name for name in getattr(self.robot, "whole_body_names", []) if name in commands]
        if len(order) != len(commands):
            order = [arm_names[0], grip_names[0], arm_names[1], grip_names[1]]
        self.robot.set_different_type_command(
            order,
            [commands[name][0] for name in order],
            [commands[name][1] for name in order],
            control_way=self.config.control_way,
            use_wbc=False,
        )

    def execute(self, action16: np.ndarray) -> None:
        delta = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        if not np.isfinite(delta).all():
            raise ValueError("Robot command contains NaN/Inf")
        target = self._delta_to_target(delta)
        self._validate_step(delta, target)
        self._send_target(target)
        self._last_target = target.copy()
        self.action_history.append(target.copy())

    def execute_policy_waypoints(self, actions16: np.ndarray) -> np.ndarray:
        """Decode one delta chunk and submit it as one continuous SDK trajectory."""
        deltas = np.asarray(actions16, np.float32).reshape(-1, ACTION16_DIM)
        if not len(deltas):
            raise ValueError("Policy waypoint chunk cannot be empty")
        reference = self._last_target.copy()
        targets = []
        for delta in deltas:
            target = self._delta_to_target(delta, reference=reference)
            self._validate_step(delta, target)
            targets.append(target)
            reference = target
        targets_array = np.asarray(targets, np.float32)

        names = [
            self.robot.torso_name,
            self.robot.arm_left_name,
            self.robot.effector_left_name,
            self.robot.arm_right_name,
            self.robot.effector_right_name,
        ]
        torso_pose = self.robot.get_desired_cartesian_pose([self.robot.torso_name])[0]
        waypoints = []
        for target in targets_array:
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

        durations = []
        elapsed = 0.0
        for index in range(len(waypoints)):
            duration = (
                float(self.config.first_policy_waypoint_duration)
                if self._policy_chunk_count == 0 and index == 0
                else float(self.config.policy_waypoint_duration)
            )
            elapsed += duration
            durations.append(elapsed)
        self.robot.move_cartesian_waypoints(
            names,
            waypoints,
            durations,
            use_wbc=True,
            add_default_torso=False,
        )
        self._policy_chunk_count += 1
        self._last_target = targets_array[-1].copy()
        for target in targets_array:
            self.action_history.append(target.copy())
        return targets_array

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
        self.action_history.append(target.copy())

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

    def execute_takeover_absolute(self, action16: np.ndarray) -> None:
        """Stream a rate-limited teleoperation target anchored on measured state."""
        if self._takeover_limited_target is None:
            self.begin_takeover()
        assert self._takeover_limited_target is not None
        previous = self._takeover_limited_target
        target = np.asarray(action16, np.float32).reshape(ACTION16_DIM).copy()
        limited = target.copy()
        for off in (0, 8):
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
        delta = self._absolute_to_delta(previous, limited)
        self._validate_step(delta, limited)
        self._send_target(limited)
        self._takeover_limited_target = limited.copy()
        self._last_target = limited.copy()
        self.action_history.append(limited.copy())

    def reset_history(self, action16: np.ndarray) -> None:
        self.action_history.clear()
        target = np.asarray(action16, np.float32).reshape(16).copy()
        self.action_history.append(target)
        self._last_target = target.copy()
        self.observation_history.clear()
        self._policy_chunk_count = 0


class WanVAPolicy:
    def __init__(self, *, host: str, port: int, prompt: str, state_history_len: int = 16,
                 obs_history_len: int = 9, replan_steps: int = 8, fake: bool = False,
                 control_left_arm: bool = True, video_guidance_scale: float = 1.0,
                 action_guidance_scale: float = 1.0) -> None:
        self.prompt = prompt
        self.replan_steps = int(replan_steps)
        self.control_left_arm = bool(control_left_arm)
        self.client = WAM4DPriorClient(
            host=host, port=port, prompt=prompt, state_history_len=state_history_len,
            obs_history_len=obs_history_len,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
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
        current_state16: np.ndarray | None,
    ) -> np.ndarray:
        """Log the server's de-normalized delta action immediately before use."""
        action = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        self._executed_server_action_count += 1
        left_xyz = action[0:3]
        right_xyz = action[8:11]
        print(
            "WAM4D server action "
            f"#{self._executed_server_action_count} (de-normalized delta xyz): "
            f"left=[{left_xyz[0]:+.5f}, {left_xyz[1]:+.5f}, {left_xyz[2]:+.5f}] "
            f"right=[{right_xyz[0]:+.5f}, {right_xyz[1]:+.5f}, {right_xyz[2]:+.5f}]",
            flush=True,
        )
        if not self.control_left_arm:
            action = action.copy()
            action[0:3] = 0.0
            action[3:7] = [1.0, 0.0, 0.0, 0.0]
            if current_state16 is not None:
                action[7] = np.asarray(current_state16, dtype=np.float32).reshape(ACTION16_DIM)[7]
            print("WAM4D policy: left arm/gripper locked; executing right arm only.", flush=True)
        return action

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        current_state16 = observation.get("state_action16")
        self.last_inference_started_chunk = True
        payload = dict(observation["wam4d"])
        payload["task"] = self.prompt
        chunk = self.client.infer_prior_chunk(
            payload,
            fallback_state16=observation.get("state_action16"),
            max_steps=self.replan_steps,
        )
        return np.stack([
            self._action_for_execution(action, current_state16=current_state16)
            for action in np.asarray(chunk, np.float32).reshape(-1, ACTION16_DIM)
        ])


class QuestControlSource:
    """Maps right-controller B/A edges and middle-trigger motion to InputState."""

    def __init__(self, robot: AstribotRobotIO, *, state_url: str,
                 trigger_threshold: float = 0.5, button_a_index: int = 4,
                 button_b_index: int = 5) -> None:
        self.robot = robot
        self.quest = QuestResidualIntervention(
            state_url=state_url, trigger_threshold=trigger_threshold,
            gripper_threshold=0.2, position_scale=1.0,
            residual_position_scale=0.2, residual_rotation_scale=np.deg2rad(30),
            episode_button_hand="right", success_button_index=button_a_index,
            failure_button_index=button_b_index, episode_button_threshold=0.5,
        )
        self.anchor: np.ndarray | None = None
        self.consecutive_errors = 0

    def reset(self) -> None:
        self.quest.reset()
        self.anchor = None
        self.consecutive_errors = 0

    def _expert_action(self, residual: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if self.anchor is None:
            self.anchor = self.robot.state_action16().copy()
        target = self.anchor.copy()
        for hand, off, roff in (("left", 0, 0), ("right", 8, 7)):
            hand_info = info.get(hand, {})
            if not hand_info.get("active", False):
                target[off : off + 8] = self.robot.state_action16()[off : off + 8]
                continue
            delta_xyz = np.asarray(hand_info.get("relative_position", residual[roff:roff+3] * .2))
            delta_rot = np.asarray(hand_info.get("scaled_rotvec", residual[roff+3:roff+6] * np.deg2rad(30)))
            target[off:off+3] = self.anchor[off:off+3] + delta_xyz
            base_q = action_quat_to_sdk_xyzw(self.anchor[off+3:off+7], use_xyzw=self.robot.config.use_xyzw)
            target_q = quat_multiply_xyzw(base_q, rotvec_to_quat_xyzw(delta_rot))
            target[off+3:off+7] = sdk_xyzw_to_action_quat(target_q, use_xyzw=self.robot.config.use_xyzw)
            grip = float(residual[roff + 6])
            target[off+7] = 1.0 if grip <= -.5 else (0.0 if grip >= .5 else target[off+7])
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
        if not active:
            self.anchor = None
        return InputState(
            b=bool(buttons.get("failure_value", 0) >= .5),
            a=bool(buttons.get("success_value", 0) >= .5),
            middle=1.0 if active else 0.0,
            expert_action=self._expert_action(residual, info) if active else None,
        )


class FakeAstribotRobotIO:
    """Deterministic adapter for deployment smoke tests."""

    def __init__(self) -> None:
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
            "task": "fake",
        }
        return {"state_action16": self.action.copy(), "wam4d": payload, "step": self.step}

    def execute(self, action16: np.ndarray) -> None:
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

    def reset_history(self, action16: np.ndarray) -> None:
        self.action = np.asarray(action16, np.float32).reshape(16).copy()
