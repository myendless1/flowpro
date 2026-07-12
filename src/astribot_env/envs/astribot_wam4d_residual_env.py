from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import pickle as pkl
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np

from astribot_env.initial_pose import load_init_joint_target_from_hdf5
from astribot_env.manual_control import StdinEpisodeController, prompt_for_success
from astribot_env.quest_intervention import QuestResidualIntervention
from astribot_env.rgbd import (
    build_serl_images,
    build_wam4d_observation_payload,
    get_current_eef_state,
    RGBDReader,
)
from astribot_env.sdk_loader import DEFAULT_ASTRIBOT_SDK_ROOT, load_astribot_class
from astribot_env.utils import (
    ACTION16_DIM,
    RESIDUAL14_DIM,
    action16_to_sdk_commands,
    action_quat_to_sdk_xyzw,
    apply_xyz_limits,
    apply_right_gripper_orientation_constraint,
    convert_gripper_cmd_value_to_action_value,
    fuse_wam4d_prior_with_residual,
    normalize_quat_xyzw,
    quat_multiply_xyzw,
    rotvec_to_quat_xyzw,
    sdk_xyzw_to_action_quat,
)
from astribot_env.wam4d_policy import WAM4DPriorClient


def _slerp_xyzw(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = normalize_quat_xyzw(a)
    b = normalize_quat_xyzw(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_xyzw(a + alpha * (b - a))

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    scale_a = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    scale_b = np.sin(theta) / sin_theta_0
    return normalize_quat_xyzw(scale_a * a + scale_b * b)


def _quat_angle_between_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_quat_xyzw(a)
    b = normalize_quat_xyzw(b)
    dot = abs(float(np.dot(a, b)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


@dataclass
class AstribotWAM4DResidualEnvConfig:
    sdk_root: str = str(DEFAULT_ASTRIBOT_SDK_ROOT)
    robot_type: str = "S1"
    sdk_frequency: float = 250.0
    high_control_rights: bool = True
    wam4d_host: str = "0.0.0.0"
    wam4d_port: int = 8006
    prompt: str = "pick up white plate"
    image_shape: tuple[int, int, int] = (256, 256, 3)
    camera_timeout: float = 0.3
    use_topic_camera: bool = False
    use_fake_images: bool = False
    state_history_len: int = 16
    obs_history_len: int = 9
    num_action_groups: int = 2
    save_visualization: bool = False
    disable_wam4d_inference: bool = False
    robot_command_enabled: bool = True
    stream_policy_actions: bool = False
    manual_takeover_only: bool = False
    video_guidance_scale: float = 5.0
    action_guidance_scale: float = 5.0
    residual_position_scale: float = 0.2
    residual_rotation_scale: float = 0.5235987755982988
    residual_gripper_deadband: float = 0.5
    action_chunk_steps: int = 32
    action_chunk_send_batch_size: int = 8
    action_chunk_step_duration: float = 0.1
    first_action_chunk_step_duration: float | None = 0.6
    action_sample_period: float = 0.1
    takeover_observation_period: float = 0.8
    use_xyzw: bool = False
    control_way: str = "filter"
    cartesian_frame: str = "chassis"
    robot_filter_scale: float = 0.1
    robot_gripper_filter_scale: float = 0.5
    use_wbc: bool = False
    use_wbc_during_takeover: bool = False
    add_default_torso: bool = False
    right_arm_min_z: float | None = 0.862
    right_gripper_angle_constraint: bool = True
    right_gripper_angle_constraint_during_takeover: bool = True
    right_gripper_target_angle_deg: float = 45.0
    right_gripper_ray_axis: str = "+z"
    right_gripper_twist_level_constraint: bool = True
    right_gripper_level_axis: str = "+x"
    left_xyz_low: tuple[float, float, float] | None = None
    left_xyz_high: tuple[float, float, float] | None = None
    right_xyz_low: tuple[float, float, float] | None = None
    right_xyz_high: tuple[float, float, float] | None = None
    max_episode_length: int = 200
    manual_control: bool = True
    prompt_on_timeout: bool = True
    episode_log_dir: str = "astribot_online_logs"
    quest_state_url: str = ""
    quest_trigger_threshold: float = 0.5
    quest_gripper_threshold: float = 0.02
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
    quest_high_freq_control: bool = True
    quest_control_rate_hz: float = 100.0
    quest_poll_rate_hz: float = 100.0
    quest_stream_gripper_every_tick: bool = False
    quest_max_translation_step_m: float = 0.01
    quest_max_rotation_step_deg: float = 2.5
    quest_sync_all_arm_targets_on_takeover: bool = False
    debug_takeover_actions: bool = False
    debug_sdk_command_state: bool = False
    init_hdf5: str = "/home/xddex05/Desktop/data/hdf5_output_multidrop/multidrop_episode_104.hdf5"
    init_frame_idx: int = 0
    initial_joint_duration: float = 4.0
    reset_to_initial_on_start: bool = True
    reset_grippers_to_initial_on_start: bool | None = None
    initial_gripper_duration: float = 1.0
    reset_to_initial_on_episode_end: bool = True
    reset_prelift_height_m: float = 0.05
    reset_prelift_duration: float = 1.0
    fake_seed: int = 0


class AstribotWAM4DResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: AstribotWAM4DResidualEnvConfig | None = None,
        fake_env: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or AstribotWAM4DResidualEnvConfig()
        self.fake_env = bool(fake_env)
        self.rng = np.random.default_rng(self.config.fake_seed)
        self.step_count = 0
        self._episode_records: list[dict[str, Any]] = []
        self._episode_index = 0
        self._last_residual_action = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._cached_prior_action = np.zeros((ACTION16_DIM,), dtype=np.float32)
        self._cached_prior_chunk = np.zeros((1, ACTION16_DIM), dtype=np.float32)
        self._alignment_prior_chunk = np.zeros((1, ACTION16_DIM), dtype=np.float32)
        self._alignment_prior_index = 0
        self._alignment_segment_id = 0
        self._cached_state16 = self._initial_fake_state16()
        self._cached_bgr_images = self._fake_bgr_images()
        self._startup_initial_reset_done = False
        self._initial_joint_reset_count = 0
        self._takeover_anchor_prior_action: np.ndarray | None = None
        self._next_step_time: float | None = None
        self._last_takeover_observation_time: float | None = None
        self._last_quest_takeover_time: float | None = None
        self._quest_client_lock = threading.RLock()
        self._quest_lock = threading.RLock()
        self._sdk_command_lock = threading.RLock()
        self._takeover_control_stop = threading.Event()
        self._quest_poll_thread: threading.Thread | None = None
        self._takeover_control_thread: threading.Thread | None = None
        self._takeover_control_anchor_action: np.ndarray | None = None
        self._takeover_arm_anchor_poses = {"left": None, "right": None}
        self._takeover_limited_action: np.ndarray | None = None
        self._takeover_control_active = False
        self._latest_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._latest_quest_intervened = False
        self._latest_quest_info: dict[str, Any] = {}
        self._latest_quest_active_arms = {"left": False, "right": False}
        self._last_active_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._last_active_quest_info: dict[str, Any] = {}
        self._last_active_quest_active_arms = {"left": False, "right": False}
        self._latest_takeover_executed_action: np.ndarray | None = None
        self._latest_takeover_command_sent = False
        self._latest_takeover_command_error = ""
        self._takeover_control_sent_count = 0
        self._latest_streaming_command_debug: dict[str, Any] = {}
        self._latest_takeover_action_debug: dict[str, Any] = {}
        self._last_streaming_command_time: float | None = None

        self.action_space = gym.spaces.Box(
            low=-np.ones((RESIDUAL14_DIM,), dtype=np.float32),
            high=np.ones((RESIDUAL14_DIM,), dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "eef_state16": gym.spaces.Box(-np.inf, np.inf, shape=(ACTION16_DIM,), dtype=np.float32),
                        "prior_action16": gym.spaces.Box(-np.inf, np.inf, shape=(ACTION16_DIM,), dtype=np.float32),
                        "last_residual_action14": gym.spaces.Box(-1.0, 1.0, shape=(RESIDUAL14_DIM,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "cam_high": gym.spaces.Box(0, 255, shape=self.config.image_shape, dtype=np.uint8),
                        "cam_left_wrist": gym.spaces.Box(0, 255, shape=self.config.image_shape, dtype=np.uint8),
                        "cam_right_wrist": gym.spaces.Box(0, 255, shape=self.config.image_shape, dtype=np.uint8),
                    }
                ),
            }
        )

        self.astribot = None
        self.rgbd: RGBDReader | None = None
        if not self.fake_env:
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
                print(
                    "Astribot filter parameters: "
                    f"filter_scale={float(self.config.robot_filter_scale):.3f}, "
                    f"gripper_filter_scale={float(self.config.robot_gripper_filter_scale):.3f}",
                    flush=True,
                )
            if not self.config.use_fake_images:
                self.rgbd = RGBDReader(
                    self.astribot,
                    camera_timeout=self.config.camera_timeout,
                    use_topic=self.config.use_topic_camera,
                )

        use_fake_wam4d_prior = self.fake_env or self.config.disable_wam4d_inference
        if self.config.disable_wam4d_inference and not self.fake_env:
            print(
                "WAM4D prior inference disabled; using current robot state as the prior action.",
                flush=True,
            )
        if not self.config.robot_command_enabled and not self.fake_env:
            print("Robot command output disabled; environment will not send SDK motion commands.", flush=True)
        if self.config.manual_takeover_only and not self.fake_env:
            print("Manual takeover-only mode enabled; idle policy actions will not be sent.", flush=True)
        if self.config.use_fake_images and not self.fake_env:
            print("Fake camera images enabled; camera frames will not be read.", flush=True)
        if float(self.config.action_sample_period) > 0.0:
            print(
                f"Astribot action sampling period: {float(self.config.action_sample_period):.3f}s.",
                flush=True,
            )
        if self._high_freq_takeover_enabled():
            print(
                "Quest takeover parallel control enabled: "
                f"{float(self.config.quest_control_rate_hz):.1f} Hz command stream; "
                f"{float(self.config.quest_poll_rate_hz):.1f} Hz Quest polling; "
                f"sampling/logging period remains {float(self.config.action_sample_period):.3f}s.",
                flush=True,
            )
        self.wam4d_prior = WAM4DPriorClient(
            host=self.config.wam4d_host,
            port=self.config.wam4d_port,
            prompt=self.config.prompt,
            state_history_len=self.config.state_history_len,
            obs_history_len=self.config.obs_history_len,
            num_action_groups=self.config.num_action_groups,
            save_visualization=self.config.save_visualization,
            video_guidance_scale=self.config.video_guidance_scale,
            action_guidance_scale=self.config.action_guidance_scale,
            fake=use_fake_wam4d_prior,
        )
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
        self.manual = StdinEpisodeController(enabled=self.config.manual_control and not self.fake_env)
        self._initial_joint_target = None
        if not self.fake_env and self.config.init_hdf5:
            self._initial_joint_target = load_init_joint_target_from_hdf5(
                self.config.init_hdf5,
                frame_idx=self.config.init_frame_idx,
            )

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get("seed"))
        self._stop_takeover_control_thread()
        self._flush_episode_log(final=False)
        self.step_count = 0
        self._episode_records = []
        self._last_residual_action = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        self._cached_prior_action = np.zeros((ACTION16_DIM,), dtype=np.float32)
        self._cached_prior_chunk = np.zeros((1, ACTION16_DIM), dtype=np.float32)
        self._alignment_prior_chunk = np.zeros((1, ACTION16_DIM), dtype=np.float32)
        self._alignment_prior_index = 0
        self._alignment_segment_id = 0
        self._takeover_anchor_prior_action = None
        self._reset_takeover_control_state()
        self._reset_quest_snapshot_state()
        self._last_takeover_observation_time = None
        self._last_quest_takeover_time = None
        self._next_step_time = time.monotonic()
        self.wam4d_prior.reset()
        with self._quest_client_lock:
            self.quest.reset()
        if (
            self.config.robot_command_enabled
            and not self._startup_initial_reset_done
            and self._reset_to_initial_on_start_enabled()
        ):
            self._reset_to_initial_joint_pose(reason="initial startup")
            self._startup_initial_reset_done = True
        obs = self._observe_and_cache_prior()
        self._ensure_takeover_control_thread()
        return obs, {"succeed": False}

    def step(self, action):
        self._wait_for_next_sample()
        self.step_count += 1
        state_before_action = self._cached_state16.copy()
        policy_residual = np.asarray(action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        policy_residual = np.clip(policy_residual, -1.0, 1.0)

        quest_residual, intervened, quest_info = self._get_quest_residual_for_step()
        pre_command_state16 = None
        if intervened:
            pre_command_state16 = self._read_state16()
        quest_outcome = self._pop_quest_episode_outcome()
        if quest_outcome is not None:
            return self._finish_episode_without_new_command(
                outcome=quest_outcome,
                policy_residual=policy_residual,
                quest_info=quest_info,
            )

        residual_action = quest_residual if intervened else policy_residual
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)

        should_send_command = True
        gripper_command_mask = None
        if intervened:
            prior_chunk = self._manual_takeover_prior_chunk(True)
            if self.config.manual_takeover_only:
                gripper_command_mask = self._manual_takeover_gripper_command_mask(residual_action)
        else:
            if self.config.manual_takeover_only:
                prior_chunk = self._manual_takeover_prior_chunk(False)
                should_send_command = False
                gripper_command_mask = self._manual_takeover_gripper_command_mask(residual_action)
            else:
                self._takeover_anchor_prior_action = None
                prior_chunk = self._cached_prior_chunk.copy()
        fused_chunk = np.asarray(
            [
                self._fuse_action(
                    prior,
                    residual_action,
                    clip_residual=not bool(intervened),
                )
                for prior in prior_chunk
            ],
            dtype=np.float32,
        )
        executed_chunk = np.asarray(
            [
                self._constrain_action(
                    fused,
                    prior,
                    residual_action,
                    preserve_neutral_grippers=self.config.manual_takeover_only,
                    apply_right_gripper_constraint=(
                        not intervened or self.config.right_gripper_angle_constraint_during_takeover
                    ),
                    apply_right_arm_min_z=not intervened,
                )
                for fused, prior in zip(fused_chunk, prior_chunk)
            ],
            dtype=np.float32,
        )
        prior_action = prior_chunk[0].copy()
        command_prior_action = prior_chunk[-1].copy()
        fused_action = fused_chunk[-1].copy()
        executed_action = executed_chunk[-1].copy()
        alignment_prior_index = int(self._alignment_prior_index if intervened else 0)
        alignment_prior_valid = (
            self._alignment_prior_chunk.size > 0
            and alignment_prior_index < int(self._alignment_prior_chunk.shape[0])
        )
        if alignment_prior_valid:
            alignment_prior_action = self._alignment_prior_chunk[alignment_prior_index].copy()
        else:
            alignment_prior_action = prior_action.copy()
        alignment_drop_reason = None if alignment_prior_valid else "alignment_horizon_exhausted"
        command_action: np.ndarray | None = None
        command_action_chunk: np.ndarray | None = None

        high_freq_takeover = self._high_freq_takeover_enabled() and bool(intervened)
        command_sent = False
        step_streaming_command_debug = None
        if not self.fake_env:
            if should_send_command and self.config.robot_command_enabled:
                if high_freq_takeover:
                    with self._sdk_command_lock:
                        command_sent = bool(self._latest_takeover_command_sent)
                        latest_takeover_action = (
                            None
                            if self._latest_takeover_executed_action is None
                            else self._latest_takeover_executed_action.copy()
                        )
                    if latest_takeover_action is not None:
                        executed_action = latest_takeover_action
                        executed_chunk = executed_action.reshape(1, ACTION16_DIM)
                        fused_action = executed_action.copy()
                        fused_chunk = executed_chunk.copy()
                        command_action = executed_action.copy()
                        command_action_chunk = executed_chunk.copy()
                elif executed_chunk.shape[0] == 1 or self.config.stream_policy_actions:
                    streamed_action = (
                        executed_action
                        if executed_chunk.shape[0] == 1
                        else executed_chunk[0].copy()
                    )
                    self._send_streaming_action(streamed_action, gripper_command_mask=gripper_command_mask)
                    if executed_chunk.shape[0] != 1:
                        executed_action = streamed_action
                        fused_action = fused_chunk[0].copy()
                        command_prior_action = prior_chunk[0].copy()
                        executed_chunk = streamed_action.reshape(1, ACTION16_DIM)
                        fused_chunk = fused_action.reshape(1, ACTION16_DIM)
                        prior_chunk = prior_chunk[0].reshape(1, ACTION16_DIM)
                    with self._sdk_command_lock:
                        step_streaming_command_debug = dict(self._latest_streaming_command_debug)
                    self._reset_takeover_control_state()
                    command_sent = True
                    command_action = executed_action.copy()
                    command_action_chunk = executed_chunk.copy()
                else:
                    self._send_streaming_action_chunk(executed_chunk)
                    self._reset_takeover_control_state()
                    command_sent = True
                    command_action = executed_action.copy()
                    command_action_chunk = executed_chunk.copy()
        else:
            self._cached_state16 = executed_action.copy()
            command_action = executed_action.copy()
            command_action_chunk = executed_chunk.copy()

        if command_action_chunk is not None:
            for executed in command_action_chunk:
                self.wam4d_prior.append_executed_action(executed)
        self._last_residual_action = residual_action.copy()
        obs = self._observe_and_cache_prior(
            skip_wam4d_inference=bool(intervened),
            fallback_prior_action=executed_action,
        )
        post_command_state16 = self._cached_state16.copy() if intervened else None
        with self._sdk_command_lock:
            streaming_command_debug = (
                step_streaming_command_debug
                if step_streaming_command_debug is not None
                else dict(self._latest_streaming_command_debug)
            )
            takeover_action_debug = dict(self._latest_takeover_action_debug)

        reward = 0.0
        terminated = False
        truncated = False
        outcome = None
        manual_command = self.manual.poll()
        if manual_command is not None:
            outcome = manual_command.outcome
            terminated = True
            reward = 1.0 if outcome == "success" else 0.0
        elif self.step_count >= self.config.max_episode_length:
            if self.config.prompt_on_timeout and not self.fake_env:
                outcome = prompt_for_success()
                terminated = True
                reward = 1.0 if outcome == "success" else 0.0
            else:
                truncated = True
                outcome = "timeout"

        info: dict[str, Any] = {
            "succeed": bool(reward > 0.0),
            "outcome": outcome,
            "grasp_penalty": 0.0,
            "state16": state_before_action.tolist(),
            "prior_action16": prior_action.tolist(),
            "wam4d_prior_action16": alignment_prior_action.tolist(),
            "command_prior_action16": command_prior_action.tolist(),
            "action_chunk_len": int(prior_chunk.shape[0]),
            "alignment_segment_id": int(self._alignment_segment_id),
            "prior_action_index": int(alignment_prior_index),
            "sac_alignment_valid": bool(alignment_prior_valid),
            "alignment_drop_reason": alignment_drop_reason,
            "prior_chunk_right_z": prior_chunk[:, 10].astype(float).tolist(),
            "fused_chunk_right_z": fused_chunk[:, 10].astype(float).tolist(),
            "executed_chunk_right_z": executed_chunk[:, 10].astype(float).tolist(),
            "policy_residual_action14": policy_residual.tolist(),
            "residual_action14": residual_action.tolist(),
            "fused_action16": fused_action.tolist(),
            "executed_action16": executed_action.tolist(),
            "executed_action16_post_clamp": executed_action.tolist(),
            "command_action16": None if command_action is None else command_action.tolist(),
            "intervened": bool(intervened),
            "quest": quest_info,
            "command_sent": command_sent,
            "takeover_high_freq_control": bool(high_freq_takeover),
            "takeover_control_rate_hz": float(self.config.quest_control_rate_hz),
            "takeover_poll_rate_hz": float(self.config.quest_poll_rate_hz),
            "takeover_control_sent_count": int(self._takeover_control_sent_count),
            "takeover_control_error": self._latest_takeover_command_error,
        }
        if pre_command_state16 is not None:
            info["pre_command_state16"] = pre_command_state16.tolist()
        if post_command_state16 is not None:
            info["post_command_state16"] = post_command_state16.tolist()
        if intervened and streaming_command_debug:
            info["streaming_command_debug"] = streaming_command_debug
        if intervened and takeover_action_debug:
            info["takeover_action_debug"] = takeover_action_debug
        if not command_sent and not self.fake_env:
            info["skipped_execution"] = True
        if intervened:
            info["human_executed_action16"] = executed_action.tolist()
            info["quest_residual_action14"] = residual_action.tolist()
            info["intervene_action"] = residual_action.copy()
            self._alignment_prior_index += 1

        self._episode_records.append(
            {
                "timestamp": time.time(),
                "step": self.step_count,
                "state16": self._cached_state16.copy(),
                "prior_action16": prior_action.copy(),
                "command_prior_action16": command_prior_action.copy(),
                "prior_action16_chunk": prior_chunk.copy(),
                "policy_residual_action14": policy_residual.copy(),
                "residual_action14": residual_action.copy(),
                "fused_action16": fused_action.copy(),
                "fused_action16_chunk": fused_chunk.copy(),
                "executed_action16": executed_action.copy(),
                "executed_action16_chunk": executed_chunk.copy(),
                "command_action16": None if command_action is None else command_action.copy(),
                "command_action16_chunk": None if command_action_chunk is None else command_action_chunk.copy(),
                "intervened": bool(intervened),
                "command_sent": bool(command_sent),
                "skipped_execution": bool(not command_sent and not self.fake_env),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "outcome": outcome,
            }
        )
        if terminated or truncated:
            self._flush_episode_log(final=True)
            if outcome in {"success", "failure"} and self.config.reset_to_initial_on_episode_end:
                self._stop_takeover_control_thread()
                self._reset_takeover_control_state()
                self._reset_to_initial_joint_pose()
        return obs, reward, terminated, truncated, info

    def _observe_and_cache_prior(
        self,
        *,
        skip_wam4d_inference: bool = False,
        fallback_prior_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        state16 = self._read_state16()
        if skip_wam4d_inference:
            bgr_images = self._maybe_append_takeover_observation()
            fallback = state16 if fallback_prior_action is None else np.asarray(
                fallback_prior_action,
                dtype=np.float32,
            ).reshape(ACTION16_DIM)
            prior_chunk = np.repeat(
                fallback.reshape(1, ACTION16_DIM),
                max(1, int(self.config.action_chunk_steps)),
                axis=0,
            )
        else:
            self._last_takeover_observation_time = None
            bgr_images = self._read_bgr_images()
            wam4d_obs = build_wam4d_observation_payload(
                bgr_images=bgr_images,
                prompt=self.config.prompt,
                action_history=self.wam4d_prior.action_history,
                history_len=self.config.state_history_len,
            )
            prior_chunk = self.wam4d_prior.infer_prior_chunk(
                wam4d_obs,
                fallback_state16=state16,
                max_steps=max(1, int(self.config.action_chunk_steps)),
            )
            self._alignment_prior_chunk = prior_chunk.copy()
            self._alignment_prior_index = 0
            self._alignment_segment_id += 1
        self._cached_state16 = state16.copy()
        self._cached_bgr_images = bgr_images
        self._cached_prior_chunk = prior_chunk.copy()
        self._cached_prior_action = prior_chunk[0].copy()
        return self._build_cached_observation()

    def _maybe_append_takeover_observation(self) -> dict[str, np.ndarray]:
        now = time.monotonic()
        period = float(self.config.takeover_observation_period)
        should_capture = (
            self._last_takeover_observation_time is None
            or period <= 0.0
            or now - self._last_takeover_observation_time >= period
        )
        if should_capture:
            bgr_images = self._read_bgr_images()
            wam4d_obs = build_wam4d_observation_payload(
                bgr_images=bgr_images,
                prompt=self.config.prompt,
                action_history=self.wam4d_prior.action_history,
                history_len=self.config.state_history_len,
            )
            self.wam4d_prior.append_observation(wam4d_obs)
            self._last_takeover_observation_time = now
            return bgr_images
        return {key: value.copy() for key, value in self._cached_bgr_images.items()}

    def _build_cached_observation(self) -> dict[str, Any]:
        return {
            "state": {
                "eef_state16": self._cached_state16.astype(np.float32, copy=True),
                "prior_action16": self._cached_prior_action.astype(np.float32, copy=True),
                "last_residual_action14": self._last_residual_action.astype(np.float32, copy=True),
            },
            "images": build_serl_images(
                bgr_images=self._cached_bgr_images,
                image_shape=self.config.image_shape,
            ),
        }

    def _fuse_action(
        self,
        prior_action: np.ndarray,
        residual_action: np.ndarray,
        *,
        clip_residual: bool = True,
    ) -> np.ndarray:
        prior_action = np.asarray(prior_action, dtype=np.float32).reshape(ACTION16_DIM)
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        return fuse_wam4d_prior_with_residual(
            prior_action,
            residual_action,
            position_scale=self.config.residual_position_scale,
            rotation_scale=self.config.residual_rotation_scale,
            use_xyzw=self.config.use_xyzw,
            gripper_deadband=self.config.residual_gripper_deadband,
            clip_residual=clip_residual,
        )

    def _constrain_action(
        self,
        fused_action: np.ndarray,
        prior_action: np.ndarray,
        residual_action: np.ndarray,
        *,
        preserve_neutral_grippers: bool = False,
        apply_right_gripper_constraint: bool = True,
        apply_right_arm_min_z: bool = True,
    ) -> np.ndarray:
        fused_action = np.asarray(fused_action, dtype=np.float32).reshape(ACTION16_DIM)
        prior_action = np.asarray(prior_action, dtype=np.float32).reshape(ACTION16_DIM)
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        executed_action = fused_action.copy()
        executed_action = apply_xyz_limits(
            executed_action,
            left_low=self.config.left_xyz_low,
            left_high=self.config.left_xyz_high,
            right_low=self.config.right_xyz_low,
            right_high=self.config.right_xyz_high,
            right_min_z=self.config.right_arm_min_z if apply_right_arm_min_z else None,
        )
        executed_action = apply_right_gripper_orientation_constraint(
            executed_action,
            enabled=self.config.right_gripper_angle_constraint and apply_right_gripper_constraint,
            use_xyzw=self.config.use_xyzw,
            target_angle_deg=self.config.right_gripper_target_angle_deg,
            ray_axis=self.config.right_gripper_ray_axis,
            level_axis=self.config.right_gripper_level_axis,
            keep_level_axis_horizontal=self.config.right_gripper_twist_level_constraint,
        )
        if preserve_neutral_grippers:
            for grip_index, residual_index in ((7, 6), (15, 13)):
                if abs(float(residual_action[residual_index])) < float(self.config.residual_gripper_deadband):
                    executed_action[grip_index] = prior_action[grip_index]
        return executed_action

    def _fuse_and_constrain_action(
        self,
        prior_action: np.ndarray,
        residual_action: np.ndarray,
        *,
        preserve_neutral_grippers: bool = False,
        apply_right_gripper_constraint: bool = True,
        apply_right_arm_min_z: bool = True,
    ) -> np.ndarray:
        fused_action = self._fuse_action(prior_action, residual_action)
        return self._constrain_action(
            fused_action,
            prior_action,
            residual_action,
            preserve_neutral_grippers=preserve_neutral_grippers,
            apply_right_gripper_constraint=apply_right_gripper_constraint,
            apply_right_arm_min_z=apply_right_arm_min_z,
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

    def _sync_takeover_targets_to_current_pose(self, current_action16: np.ndarray) -> None:
        if not bool(self.config.quest_sync_all_arm_targets_on_takeover):
            return
        if self.fake_env or self.astribot is None or not bool(self.config.robot_command_enabled):
            return
        arm_poses, _grippers = action16_to_sdk_commands(current_action16, use_xyzw=self.config.use_xyzw)
        self.astribot.set_cartesian_pose(
            [self.astribot.arm_left_name, self.astribot.arm_right_name],
            arm_poses,
            control_way=self.config.control_way,
            use_wbc=False,
            add_default_torso=False,
        )

    def _constrain_takeover_action(self, action16: np.ndarray) -> np.ndarray:
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

    def _manual_takeover_prior_chunk(self, intervened: bool) -> np.ndarray:
        if not intervened:
            self._takeover_anchor_prior_action = None
            return self._cached_state16.reshape(1, ACTION16_DIM).astype(np.float32, copy=True)
        if self._takeover_anchor_prior_action is None:
            self._takeover_anchor_prior_action = self._cached_state16.astype(np.float32, copy=True)
        return self._takeover_anchor_prior_action.reshape(1, ACTION16_DIM).astype(np.float32, copy=True)

    def _manual_takeover_gripper_command_mask(self, residual_action: np.ndarray) -> list[bool]:
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        deadband = float(self.config.residual_gripper_deadband)
        return [abs(float(residual_action[6])) >= deadband, abs(float(residual_action[13])) >= deadband]

    def _wait_for_next_sample(self) -> None:
        period = float(self.config.action_sample_period)
        if period <= 0.0:
            return
        now = time.monotonic()
        if self._next_step_time is None:
            self._next_step_time = now
        wait_s = self._next_step_time - now
        if wait_s > 0.0:
            time.sleep(wait_s)
            now = time.monotonic()
        self._next_step_time = max(self._next_step_time + period, now + period)

    def _high_freq_takeover_enabled(self) -> bool:
        return (
            bool(self.config.quest_high_freq_control)
            and bool(self.config.quest_state_url)
            and float(self.config.quest_control_rate_hz) > 0.0
            and float(self.config.quest_poll_rate_hz) > 0.0
            and bool(self.config.robot_command_enabled)
            and not self.fake_env
        )

    def _ensure_takeover_control_thread(self) -> None:
        if not self._high_freq_takeover_enabled():
            return
        self._takeover_control_stop.clear()
        if self._quest_poll_thread is None or not self._quest_poll_thread.is_alive():
            self._quest_poll_thread = threading.Thread(
                target=self._quest_poll_loop,
                name="astribot-quest-poll",
                daemon=True,
            )
            self._quest_poll_thread.start()
        if self._takeover_control_thread is None or not self._takeover_control_thread.is_alive():
            self._takeover_control_thread = threading.Thread(
                target=self._takeover_control_loop,
                name="astribot-quest-takeover-stream",
                daemon=True,
            )
            self._takeover_control_thread.start()

    def _stop_takeover_control_thread(self) -> None:
        threads = [self._quest_poll_thread, self._takeover_control_thread]
        if all(thread is None for thread in threads):
            return
        self._takeover_control_stop.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=1.0)
        self._quest_poll_thread = None
        self._takeover_control_thread = None

    def _reset_takeover_control_state(self) -> None:
        with self._sdk_command_lock:
            self._takeover_control_anchor_action = None
            self._takeover_arm_anchor_poses = {"left": None, "right": None}
            self._takeover_limited_action = None
            self._takeover_anchor_prior_action = None
            self._takeover_control_active = False
            self._latest_takeover_executed_action = None
            self._latest_takeover_command_sent = False
            self._latest_takeover_command_error = ""
            self._takeover_control_sent_count = 0
            self._latest_streaming_command_debug = {}
            self._latest_takeover_action_debug = {}
            self._last_streaming_command_time = None

    def _reset_quest_snapshot_state(self) -> None:
        with self._quest_lock:
            self._latest_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
            self._latest_quest_intervened = False
            self._latest_quest_info = {}
            self._latest_quest_active_arms = {"left": False, "right": False}
            self._last_active_quest_residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
            self._last_active_quest_info = {}
            self._last_active_quest_active_arms = {"left": False, "right": False}

    @staticmethod
    def _active_arms_from_quest_info(info: dict[str, Any]) -> dict[str, bool]:
        active_arms = info.get("active_arms")
        if isinstance(active_arms, dict):
            return {hand: bool(active_arms.get(hand, False)) for hand in ("left", "right")}
        return {
            hand: bool((info.get(hand) or {}).get("active", False))
            for hand in ("left", "right")
        }

    def _poll_quest_residual(
        self,
        *,
        apply_release_grace: bool = False,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        with self._quest_client_lock:
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

    def _get_latest_quest_snapshot(
        self,
        *,
        apply_release_grace: bool = False,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
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

    def _get_quest_residual_for_step(self) -> tuple[np.ndarray, bool, dict[str, Any]]:
        if self._high_freq_takeover_enabled():
            return self._get_latest_quest_snapshot(apply_release_grace=True)
        return self._poll_quest_residual(apply_release_grace=True)

    def _pop_quest_episode_outcome(self) -> str | None:
        with self._quest_client_lock:
            return self.quest.pop_episode_outcome()

    def _quest_poll_loop(self) -> None:
        interval = 1.0 / max(float(self.config.quest_poll_rate_hz), 1e-6)
        while not self._takeover_control_stop.is_set():
            started = time.monotonic()
            try:
                self._poll_quest_residual(apply_release_grace=True)
            except Exception as exc:
                with self._sdk_command_lock:
                    self._latest_takeover_command_error = f"quest_poll: {exc}"
                print(f"WARNING: Quest polling error: {exc!r}", flush=True)
                self._takeover_control_stop.wait(0.2)
                continue

            elapsed = time.monotonic() - started
            self._takeover_control_stop.wait(max(0.0, interval - elapsed))

    def _takeover_control_loop(self) -> None:
        interval = 1.0 / max(float(self.config.quest_control_rate_hz), 1e-6)
        while not self._takeover_control_stop.is_set():
            started = time.monotonic()
            try:
                residual, intervened, info = self._get_latest_quest_snapshot(apply_release_grace=True)
                if intervened:
                    self._send_high_freq_takeover_action(
                        residual,
                        quest_info=info,
                        arm_command_mask=self._active_arms_from_quest_info(info),
                    )
                else:
                    with self._sdk_command_lock:
                        needs_reset = (
                            self._takeover_control_active
                            or self._latest_takeover_executed_action is not None
                            or self._latest_takeover_command_sent
                        )
                    if needs_reset:
                        self._reset_takeover_control_state()
            except Exception as exc:
                with self._sdk_command_lock:
                    self._latest_takeover_command_sent = False
                    self._latest_takeover_command_error = str(exc)
                print(f"WARNING: Quest high-frequency takeover control error: {exc!r}", flush=True)
                self._takeover_control_stop.wait(0.2)
                continue

            elapsed = time.monotonic() - started
            self._takeover_control_stop.wait(max(0.0, interval - elapsed))

    def _send_high_freq_takeover_action(
        self,
        residual_action: np.ndarray,
        *,
        quest_info: dict[str, Any] | None = None,
        arm_command_mask: dict[str, bool] | None = None,
    ) -> None:
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(RESIDUAL14_DIM)
        quest_info = {} if quest_info is None else dict(quest_info)
        with self._sdk_command_lock:
            active_mask = self._active_arms_from_quest_info(quest_info)
            if arm_command_mask is not None:
                requested_mask = {hand: bool(arm_command_mask.get(hand, False)) for hand in ("left", "right")}
                if any(active_mask.values()):
                    active_mask = {
                        hand: bool(active_mask.get(hand, False)) and requested_mask[hand]
                        for hand in ("left", "right")
                    }
                else:
                    active_mask = requested_mask
            if not any(active_mask.values()):
                self._takeover_control_active = False
                return

            if self._takeover_limited_action is None:
                current_action = self._read_current_state16()
                self._takeover_control_anchor_action = current_action.copy()
                self._takeover_anchor_prior_action = current_action.copy()
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
            constrained = self._constrain_takeover_action(target_action)
            limited_action = self._limit_takeover_action_step(constrained)
            self._send_streaming_action(
                limited_action,
                gripper_command_mask=None,
                arm_command_mask=None,
                use_wbc=self.config.use_wbc_during_takeover,
                add_default_torso=False,
            )
            self._latest_takeover_executed_action = limited_action.copy()
            self._latest_takeover_action_debug = {
                "mode": "direct_relative_pose",
                "active_arms": dict(active_mask),
                "anchor_poses": {
                    hand: (
                        None
                        if self._takeover_arm_anchor_poses[hand] is None
                        else self._takeover_arm_anchor_poses[hand].tolist()
                    )
                    for hand in ("left", "right")
                },
                "residual_action14": residual_action.tolist(),
                "direct_target_action16": target_action.tolist(),
                "constrained_action16": constrained.tolist(),
                "executed_action16": limited_action.tolist(),
            }
            self._latest_takeover_command_sent = True
            self._latest_takeover_command_error = ""
            self._takeover_control_sent_count += 1
            self._takeover_control_active = True

    def _read_desired_arm_action_pose(self, hand: str) -> np.ndarray:
        pose_start = 0 if hand == "left" else 8
        if self.fake_env:
            return self._cached_state16[pose_start : pose_start + 7].astype(np.float32, copy=True)
        if self.astribot is None:
            raise RuntimeError("Quest takeover requires Astribot to read desired cartesian pose.")
        name = self.astribot.arm_left_name if hand == "left" else self.astribot.arm_right_name
        return self._read_desired_cartesian_action_poses([name])[0]

    def _read_current_arm_action_pose(self, hand: str) -> np.ndarray:
        pose_start = 0 if hand == "left" else 8
        return self._read_current_state16()[pose_start : pose_start + 7].astype(np.float32, copy=True)

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
                    hand_info.get(
                        "relative_rotvec",
                        hand_info.get("rotvec", [0.0, 0.0, 0.0]),
                    ),
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

    def _finish_episode_without_new_command(
        self,
        *,
        outcome: str,
        policy_residual: np.ndarray,
        quest_info: dict[str, Any],
    ):
        reward = 1.0 if outcome == "success" else 0.0
        obs = self._build_cached_observation()
        residual_action = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        executed_action = self._cached_state16.copy()
        info: dict[str, Any] = {
            "succeed": bool(reward > 0.0),
            "outcome": outcome,
            "grasp_penalty": 0.0,
            "prior_action16": self._cached_prior_action.tolist(),
            "executed_action16": executed_action.tolist(),
            "command_action16": None,
            "quest": quest_info,
            "skipped_execution": True,
        }
        self._episode_records.append(
            {
                "timestamp": time.time(),
                "step": self.step_count,
                "state16": self._cached_state16.copy(),
                "prior_action16": self._cached_prior_action.copy(),
                "policy_residual_action14": policy_residual.copy(),
                "residual_action14": residual_action,
                "executed_action16": executed_action,
                "command_action16": None,
                "command_action16_chunk": None,
                "intervened": False,
                "skipped_execution": True,
                "reward": float(reward),
                "terminated": True,
                "truncated": False,
                "outcome": outcome,
            }
        )
        self._flush_episode_log(final=True)
        if self.config.reset_to_initial_on_episode_end:
            self._stop_takeover_control_thread()
            self._reset_takeover_control_state()
            self._reset_to_initial_joint_pose()
        return obs, reward, True, False, info

    def _read_state16(self) -> np.ndarray:
        if self.fake_env:
            return self._cached_state16.astype(np.float32, copy=True)
        assert self.astribot is not None
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
        if self.fake_env:
            return self._cached_state16.astype(np.float32, copy=True)
        if self.astribot is None:
            raise RuntimeError("Quest takeover requires Astribot to read current cartesian pose.")
        return get_current_eef_state(
            self.astribot,
            use_xyzw=self.config.use_xyzw,
            frame=self.config.cartesian_frame,
        )

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

    def _read_bgr_images(self) -> dict[str, np.ndarray]:
        if self.fake_env or self.config.use_fake_images:
            return {key: value.copy() for key, value in self._cached_bgr_images.items()}
        assert self.rgbd is not None
        return self.rgbd.get_bgr_images_dict()

    def _send_streaming_action(
        self,
        action16: np.ndarray,
        gripper_command_mask: list[bool] | None = None,
        arm_command_mask: dict[str, bool] | None = None,
        use_wbc: bool | None = None,
        add_default_torso: bool | None = None,
    ) -> None:
        assert self.astribot is not None
        with self._sdk_command_lock:
            use_wbc = self.config.use_wbc if use_wbc is None else bool(use_wbc)
            add_default_torso = (
                self.config.add_default_torso
                if add_default_torso is None
                else bool(add_default_torso)
            )
            arm_poses, grippers = action16_to_sdk_commands(action16, use_xyzw=self.config.use_xyzw)
            arm_names = [self.astribot.arm_left_name, self.astribot.arm_right_name]
            if arm_command_mask is None:
                command_arm_names = arm_names
                command_arm_poses = arm_poses
            else:
                enabled = [bool(arm_command_mask.get("left", False)), bool(arm_command_mask.get("right", False))]
                command_arm_names = [name for name, active in zip(arm_names, enabled) if active]
                command_arm_poses = [pose for pose, active in zip(arm_poses, enabled) if active]
            if gripper_command_mask is None:
                gripper_names = [self.astribot.effector_left_name, self.astribot.effector_right_name]
                gripper_targets = grippers
            else:
                gripper_names = [
                    name
                    for name, enabled in zip(
                        [self.astribot.effector_left_name, self.astribot.effector_right_name],
                        gripper_command_mask,
                    )
                    if enabled
                ]
                gripper_targets = [target for target, enabled in zip(grippers, gripper_command_mask) if enabled]

            sdk_desired_before = None
            sdk_current_before = None
            if command_arm_names and self.config.debug_sdk_command_state:
                sdk_desired_before = self.astribot.get_desired_cartesian_pose(
                    names=command_arm_names,
                    frame=self.config.cartesian_frame,
                )
                sdk_current_before = self.astribot.get_current_cartesian_pose(
                    names=command_arm_names,
                    frame=self.config.cartesian_frame,
                )

            command_api = "none"
            command_names, command_types, command_list = self._merge_streaming_command(
                command_arm_names=command_arm_names,
                command_arm_poses=command_arm_poses,
                gripper_names=gripper_names,
                gripper_targets=gripper_targets,
            )
            if command_names and self._should_use_mixed_streaming_command(command_arm_names):
                self.astribot.set_different_type_command(
                    command_names,
                    command_types,
                    command_list,
                    control_way=self.config.control_way,
                    use_wbc=use_wbc,
                )
                command_api = "set_different_type_command"
            else:
                if command_arm_names:
                    self.astribot.set_cartesian_pose(
                        command_arm_names,
                        command_arm_poses,
                        control_way=self.config.control_way,
                        use_wbc=use_wbc,
                        add_default_torso=add_default_torso,
                    )
                    command_api = "set_cartesian_pose"
                if gripper_names:
                    self.astribot.set_joints_position(
                        gripper_names,
                        gripper_targets,
                        control_way=self.config.control_way,
                        use_wbc=False,
                        add_default_torso=False,
                    )
                    command_api = (
                        "set_joints_position"
                        if command_api == "none"
                        else f"{command_api}+set_joints_position"
                    )

            sdk_desired_after = None
            sdk_current_after = None
            if command_arm_names and self.config.debug_sdk_command_state:
                sdk_desired_after = self.astribot.get_desired_cartesian_pose(
                    names=command_arm_names,
                    frame=self.config.cartesian_frame,
                )
                sdk_current_after = self.astribot.get_current_cartesian_pose(
                    names=command_arm_names,
                    frame=self.config.cartesian_frame,
                )
            command_time = time.time()
            previous_command_time = self._last_streaming_command_time
            self._last_streaming_command_time = command_time
            self._latest_streaming_command_debug = {
                "timestamp": command_time,
                "dt_since_previous_command": (
                    None
                    if previous_command_time is None
                    else command_time - previous_command_time
                ),
                "command_arm_names": list(command_arm_names),
                "command_arm_poses": [list(pose) for pose in command_arm_poses],
                "gripper_names": list(gripper_names),
                "gripper_targets": [list(target) for target in gripper_targets],
                "command_api": command_api,
                "control_way": self.config.control_way,
                "use_wbc": bool(use_wbc),
                "add_default_torso": bool(add_default_torso),
                "arm_command_mask": None if arm_command_mask is None else dict(arm_command_mask),
                "gripper_command_mask": None if gripper_command_mask is None else list(gripper_command_mask),
            }
            if self.config.debug_sdk_command_state:
                self._latest_streaming_command_debug.update(
                    {
                        "sdk_desired_before": sdk_desired_before,
                        "sdk_current_before": sdk_current_before,
                        "sdk_desired_after": sdk_desired_after,
                        "sdk_current_after": sdk_current_after,
                    }
                )

    def _should_use_mixed_streaming_command(self, command_arm_names: list[str]) -> bool:
        if not hasattr(self.astribot, "set_different_type_command"):
            return False
        return not (
            self.config.control_way == "filter"
            and len(command_arm_names) == 1
        )

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

    def _send_streaming_action_chunk(self, action_chunk16: np.ndarray) -> None:
        assert self.astribot is not None
        with self._sdk_command_lock:
            action_chunk16 = np.asarray(action_chunk16, dtype=np.float32).reshape(-1, ACTION16_DIM)
            names = [
                self.astribot.arm_left_name,
                self.astribot.effector_left_name,
                self.astribot.arm_right_name,
                self.astribot.effector_right_name,
            ]
            waypoints = []
            for action16 in action_chunk16:
                arm_poses, grippers = action16_to_sdk_commands(action16, use_xyzw=self.config.use_xyzw)
                waypoints.append(
                    [
                        arm_poses[0],
                        grippers[0],
                        arm_poses[1],
                        grippers[1],
                    ]
                )
            batch_size = max(1, int(self.config.action_chunk_send_batch_size))
            first_step_duration = (
                self.config.first_action_chunk_step_duration
                if self.step_count <= 1
                else None
            )
            for batch_start in range(0, len(waypoints), batch_size):
                segment_waypoints = waypoints[batch_start : batch_start + batch_size]
                time_list = []
                elapsed = 0.0
                for local_idx in range(len(segment_waypoints)):
                    global_idx = batch_start + local_idx
                    step_duration = (
                        float(first_step_duration)
                        if global_idx == 0 and first_step_duration is not None
                        else float(self.config.action_chunk_step_duration)
                    )
                    elapsed += step_duration
                    time_list.append(elapsed)
                self.astribot.move_cartesian_waypoints(
                    names,
                    segment_waypoints,
                    time_list,
                    use_wbc=False,
                    add_default_torso=False,
                )

    def _reset_to_initial_joint_pose(self, *, reason: str = "episode reset") -> bool:
        if self._initial_joint_reset_count > 0:
            self._raise_current_arms_before_initial_reset()
        moved = self._move_to_initial_joint_pose(reason=reason)
        if moved:
            self._initial_joint_reset_count += 1
        return moved

    def _raise_current_arms_before_initial_reset(self) -> None:
        if not self.config.robot_command_enabled or self.astribot is None:
            return
        lift_height = float(self.config.reset_prelift_height_m)
        if lift_height <= 0.0:
            return

        duration = max(0.0, float(self.config.reset_prelift_duration))
        with self._sdk_command_lock:
            current_action = self._read_state16()
            lifted_action = current_action.copy()
            lifted_action[2] += lift_height
            lifted_action[10] += lift_height
            arm_poses, _grippers = action16_to_sdk_commands(
                lifted_action,
                use_xyzw=self.config.use_xyzw,
            )
            arm_names = [self.astribot.arm_left_name, self.astribot.arm_right_name]
            print(
                "Raising Astribot arms before initial-pose reset: "
                f"left_z {float(current_action[2]):.3f}->{float(lifted_action[2]):.3f}, "
                f"right_z {float(current_action[10]):.3f}->{float(lifted_action[10]):.3f}, "
                f"duration={duration:.3f}s.",
                flush=True,
            )
            if hasattr(self.astribot, "move_cartesian_pose"):
                self.astribot.move_cartesian_pose(
                    arm_names,
                    arm_poses,
                    duration=duration,
                    use_wbc=False,
                    add_default_torso=False,
                )
            else:
                self.astribot.set_cartesian_pose(
                    arm_names,
                    arm_poses,
                    control_way=self.config.control_way,
                    use_wbc=False,
                    add_default_torso=False,
                )
                if duration > 0.0:
                    time.sleep(duration)

    def _move_to_initial_joint_pose(self, *, reason: str = "episode reset") -> bool:
        if not self.config.robot_command_enabled or self._initial_joint_target is None or self.astribot is None:
            return False
        assert self.astribot is not None
        print(
            f"Moving Astribot non-chassis joints to initial pose ({reason}) from {self.config.init_hdf5} "
            f"frame={self.config.init_frame_idx}.",
            flush=True,
        )
        with self._sdk_command_lock:
            self.astribot.move_joints_position(
                self.astribot.whole_body_names[1:],
                self._initial_joint_target,
                duration=float(self.config.initial_joint_duration),
                use_wbc=False,
            )
        return True

    def _reset_to_initial_on_start_enabled(self) -> bool:
        if self.config.reset_grippers_to_initial_on_start is not None:
            return bool(self.config.reset_grippers_to_initial_on_start)
        return bool(self.config.reset_to_initial_on_start)

    def _initial_fake_state16(self) -> np.ndarray:
        return np.asarray(
            [
                0.40,
                0.30,
                1.00,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.40,
                -0.30,
                1.00,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def _fake_bgr_images(self) -> dict[str, np.ndarray]:
        height, width, _channels = self.config.image_shape
        zeros = np.zeros((height, width, 3), dtype=np.uint8)
        return {
            "Bolt": zeros.copy(),
            "left_D405": zeros.copy(),
            "right_D405": zeros.copy(),
        }

    def _flush_episode_log(self, *, final: bool) -> None:
        if not self._episode_records:
            return
        log_dir = Path(self.config.episode_log_dir).expanduser()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            suffix = "final" if final else "reset"
            path = log_dir / f"episode_{self._episode_index:06d}_{stamp}_{suffix}.pkl"
            payload = {
                "config": self.config,
                "episode_index": self._episode_index,
                "records": self._episode_records,
            }
            with path.open("wb") as handle:
                pkl.dump(payload, handle)
            self._episode_index += 1
            self._episode_records = []
        except Exception as exc:
            print(f"WARNING: failed to write Astribot episode log: {exc!r}", flush=True)

    def close(self) -> None:
        self._stop_takeover_control_thread()
        self._flush_episode_log(final=False)
        return super().close()
