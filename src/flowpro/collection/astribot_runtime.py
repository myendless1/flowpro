from __future__ import annotations

"""Concrete Astribot/Quest/Wan-VA adapters used by the preference collector.

Heavy robot and websocket dependencies are imported lazily so the collection
state machine and its fake mode remain testable on a development machine.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from astribot_env.quest_intervention import QuestResidualIntervention
from astribot_env.initial_pose import load_init_joint_target_from_hdf5
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
    init_hdf5: str = ""
    init_frame_idx: int = 0
    initial_joint_duration: float = 4.0
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
        if self.config.init_hdf5:
            target = load_init_joint_target_from_hdf5(
                self.config.init_hdf5, frame_idx=self.config.init_frame_idx
            )
            self.robot.move_joints_position(
                self.robot.whole_body_names[1:], target,
                duration=self.config.initial_joint_duration, use_wbc=False,
            )
        self.rgbd = RGBDReader(
            self.robot,
            camera_timeout=self.config.camera_timeout,
            use_topic=self.config.image_from_s1_topic,
        )
        self.action_history: deque[np.ndarray] = deque(maxlen=self.config.state_history_len)
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=self.config.obs_history_len)

    def state_action16(self) -> np.ndarray:
        return get_current_eef_state(
            self.robot,
            use_xyzw=self.config.use_xyzw,
            frame=self.config.cartesian_frame,
        )

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

    def _delta_to_target(self, delta: np.ndarray) -> np.ndarray:
        reference = self.state_action16()
        target = reference.copy()
        for off in (0, 8):
            target[off : off + 7] = apply_relative_pose7(
                reference[off : off + 7], delta[off : off + 7]
            )
            target[off + 7] = delta[off + 7]
        return target

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
            if off == 8 and self.config.right_min_z is not None and xyz[2] < self.config.right_min_z:
                raise ValueError(f"Right arm z={xyz[2]:.4f} below minimum {self.config.right_min_z:.4f}")
            if float(np.linalg.norm(delta[off + 3 : off + 7])) < 1e-6:
                raise ValueError(f"Arm@{off} quaternion has zero norm")
            distance = float(np.linalg.norm(delta[off : off + 3]))
            if self.config.max_translation_step_m > 0 and distance > self.config.max_translation_step_m:
                raise ValueError(f"Unsafe Cartesian step for arm@{off}: {distance:.4f}m")
            angle = self._quat_angle_deg(delta[off + 3 : off + 7], np.array([1, 0, 0, 0]))
            if self.config.max_rotation_step_deg > 0 and angle > self.config.max_rotation_step_deg:
                raise ValueError(f"Unsafe rotation step for arm@{off}: {angle:.2f}deg")

    def execute(self, action16: np.ndarray) -> None:
        delta = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM)
        if not np.isfinite(delta).all():
            raise ValueError("Robot command contains NaN/Inf")
        target = self._delta_to_target(delta)
        self._validate_step(delta, target)
        arm_poses, grippers = action16_to_sdk_commands(target, use_xyzw=self.config.use_xyzw)
        arm_names = [self.robot.arm_left_name, self.robot.arm_right_name]
        grip_names = [self.robot.effector_left_name, self.robot.effector_right_name]
        if hasattr(self.robot, "set_different_type_command"):
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
        else:
            self.robot.set_cartesian_pose(
                arm_names, arm_poses, control_way=self.config.control_way,
                use_wbc=False, add_default_torso=False,
            )
            self.robot.set_joints_position(
                grip_names, grippers, control_way=self.config.control_way,
                use_wbc=False, add_default_torso=False,
            )
        self.action_history.append(target.copy())

    def reset_history(self, action16: np.ndarray) -> None:
        self.action_history.clear()
        self.action_history.append(np.asarray(action16, np.float32).reshape(16).copy())
        self.observation_history.clear()


class WanVAPolicy:
    def __init__(self, *, host: str, port: int, prompt: str, state_history_len: int = 16,
                 obs_history_len: int = 9, replan_steps: int = 8, fake: bool = False) -> None:
        self.prompt = prompt
        self.replan_steps = int(replan_steps)
        self.client = WAM4DPriorClient(
            host=host, port=port, prompt=prompt, state_history_len=state_history_len,
            obs_history_len=obs_history_len, fake=fake,
        )
        self._chunk: deque[np.ndarray] = deque()

    def reset(self, observation: dict[str, Any] | None = None) -> None:
        self.client.reset()
        self._chunk.clear()

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        if self._chunk:
            return np.asarray([self._chunk.popleft()])
        payload = dict(observation["wam4d"])
        payload["task"] = self.prompt
        chunk = self.client.infer_prior_chunk(
            payload,
            fallback_state16=observation.get("state_action16"),
            max_steps=self.replan_steps,
        )
        self._chunk.extend(np.asarray(chunk)[1:])
        return np.asarray(chunk[:1])


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
        current = self.robot.state_action16()
        delta = target.copy()
        delta[0:7] = relative_pose7(current[0:7], target[0:7])
        delta[8:15] = relative_pose7(current[8:15], target[8:15])
        return delta.astype(np.float32)

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

    def reset_history(self, action16: np.ndarray) -> None:
        self.action = np.asarray(action16, np.float32).reshape(16).copy()
