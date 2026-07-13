from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np
import requests
import urllib3

from astribot_env.utils import (
    RESIDUAL14_DIM,
    quat_inverse_xyzw,
    quat_multiply_xyzw,
    quat_xyzw_to_rotvec,
)


WEBXR_ALIGNED_TO_ROBOT = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
QUEST_FORWARD_QUAT_XYZW = np.asarray([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float32)


@dataclass
class _HandAnchor:
    position: np.ndarray | None = None
    quat_xyzw: np.ndarray | None = None
    active: bool = False


def _normalize_quat_xyzw(quat: Any) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _quat_to_matrix_xyzw(quat: Any) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    return _normalize_quat_xyzw(quat)


def project_quat_to_vertical_roll_xyzw(quat_xyzw: Any) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat_xyzw)
    projected = np.asarray([0.0, 0.0, quat[2], quat[3]], dtype=np.float64)
    return _normalize_quat_xyzw(projected).astype(np.float32)


def webxr_aligned_quat_to_robot(quat_xyzw: Any) -> np.ndarray:
    rotation = WEBXR_ALIGNED_TO_ROBOT @ _quat_to_matrix_xyzw(quat_xyzw) @ WEBXR_ALIGNED_TO_ROBOT.T
    return _matrix_to_quat_xyzw(rotation).astype(np.float32)


def webxr_aligned_delta_to_robot(delta_xyz: Any, scale: float) -> np.ndarray:
    x_right, y_up, z_back = (float(v) for v in np.asarray(delta_xyz).reshape(3))
    return np.asarray([-z_back * scale, -x_right * scale, y_up * scale], dtype=np.float32)


def webxr_aligned_position_to_robot(position_xyz: Any) -> np.ndarray:
    x_right, y_up, z_back = (float(v) for v in np.asarray(position_xyz).reshape(3))
    return np.asarray([-z_back, -x_right, y_up], dtype=np.float64)


def quest_controller_quat_to_robot_target(quat_xyzw: Any) -> np.ndarray:
    return quat_multiply_xyzw(quat_xyzw, QUEST_FORWARD_QUAT_XYZW)


def index_trigger_to_gripper_residual(index: float, threshold: float, deadband: float) -> float:
    threshold = float(np.clip(threshold, 0.0, 0.999))
    deadband = float(np.clip(deadband, 0.0, 0.999))
    index = float(np.clip(index, 0.0, 1.0))
    if index < threshold:
        return -1.0
    close_fraction = (index - threshold) / max(1.0 - threshold, 1e-6)
    return float(deadband + close_fraction * (1.0 - deadband))


def index_trigger_to_gripper_action(index: float, threshold: float) -> float:
    """Map the analog index trigger to 1=open, 0=closed continuously."""
    threshold = float(np.clip(threshold, 0.0, 0.999))
    index = float(np.clip(index, 0.0, 1.0))
    close_fraction = np.clip(
        (index - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    return float(1.0 - close_fraction)


class QuestResidualIntervention:
    def __init__(
        self,
        *,
        state_url: str,
        trigger_threshold: float,
        gripper_threshold: float,
        position_scale: float,
        residual_position_scale: float,
        residual_rotation_scale: float,
        residual_gripper_deadband: float = 0.5,
        rotation_gain: float = 1.0,
        rotation_mode: str = "full",
        timeout: float = 0.05,
        episode_button_hand: str = "right",
        success_button_index: int = 4,
        failure_button_index: int = 5,
        episode_button_threshold: float = 0.5,
        neutral_gripper_when_released: bool = True,
        verify_ssl: bool = False,
    ) -> None:
        self.state_url = state_url.rstrip("/")
        self.trigger_threshold = float(trigger_threshold)
        self.gripper_threshold = float(gripper_threshold)
        self.position_scale = float(position_scale)
        self.residual_position_scale = float(residual_position_scale)
        self.residual_rotation_scale = float(residual_rotation_scale)
        self.residual_gripper_deadband = float(residual_gripper_deadband)
        self.rotation_gain = float(rotation_gain)
        self.rotation_mode = str(rotation_mode).strip().lower() or "full"
        if self.rotation_mode not in {"full", "vertical_roll_only"}:
            raise ValueError(
                "Quest rotation_mode must be 'full' or 'vertical_roll_only', "
                f"got {self.rotation_mode!r}"
            )
        self.timeout = float(timeout)
        self.episode_button_hand = episode_button_hand
        self.success_button_index = int(success_button_index)
        self.failure_button_index = int(failure_button_index)
        self.episode_button_threshold = float(episode_button_threshold)
        self.neutral_gripper_when_released = bool(neutral_gripper_when_released)
        self.verify_ssl = bool(verify_ssl)
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.anchors = {"left": _HandAnchor(), "right": _HandAnchor()}
        self.last_snapshot: dict[str, Any] | None = None
        self.last_error: str = ""
        self._success_pressed = False
        self._failure_pressed = False
        self._pending_outcome: Literal["success", "failure"] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.state_url)

    def reset(self) -> None:
        self.anchors = {"left": _HandAnchor(), "right": _HandAnchor()}
        self.last_snapshot = None
        self.last_error = ""
        self._success_pressed = False
        self._failure_pressed = False
        self._pending_outcome = None

    def pop_episode_outcome(self) -> Literal["success", "failure"] | None:
        outcome = self._pending_outcome
        self._pending_outcome = None
        return outcome

    def get_residual_action(self) -> tuple[np.ndarray, bool, dict[str, Any]]:
        residual = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
        if not self.enabled:
            return residual, False, {}

        try:
            response = requests.get(self.state_url, timeout=self.timeout, verify=self.verify_ssl)
            response.raise_for_status()
            snapshot = response.json()
        except Exception as exc:
            self.last_error = str(exc)
            return residual, False, {"quest_error": self.last_error}

        self.last_snapshot = snapshot
        episode_debug = self._update_episode_buttons(snapshot)
        any_takeover = False
        hands = snapshot.get("hands", {})
        debug: dict[str, Any] = {
            "quest_timestamp": snapshot.get("server_time", time.time()),
            "episode_buttons": episode_debug,
        }
        for hand, offset in (("left", 0), ("right", 7)):
            hand_state = hands.get(hand, {})
            if not hand_state.get("valid", False):
                self.anchors[hand] = _HandAnchor()
                continue

            aligned = hand_state.get("aligned", {})
            position = np.asarray(aligned.get("position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            quat = webxr_aligned_quat_to_robot(aligned.get("quaternion", [0.0, 0.0, 0.0, 1.0]))
            if self.rotation_mode == "vertical_roll_only":
                quat = project_quat_to_vertical_roll_xyzw(quat)
            controller_quat_xyzw = (
                quest_controller_quat_to_robot_target(quat)
                if self.rotation_mode == "full"
                else quat
            )
            middle = float(hand_state.get("middle", 0.0))
            index = float(hand_state.get("index", 0.0))
            active = middle >= self.trigger_threshold
            anchor = self.anchors[hand]

            if active:
                any_takeover = True
                if not anchor.active or anchor.position is None or anchor.quat_xyzw is None:
                    anchor.position = position.copy()
                    anchor.quat_xyzw = controller_quat_xyzw.copy()
                    anchor.active = True

                delta_webxr = position - anchor.position
                controller_robot_delta = webxr_aligned_delta_to_robot(delta_webxr, 1.0)
                robot_delta = webxr_aligned_delta_to_robot(delta_webxr, self.position_scale)
                robot_pose = webxr_aligned_position_to_robot(position)
                anchor_pose = webxr_aligned_position_to_robot(anchor.position)
                absolute_rotvec = quat_xyzw_to_rotvec(controller_quat_xyzw)
                residual[offset : offset + 3] = robot_delta / max(self.residual_position_scale, 1e-6)

                delta_quat = quat_multiply_xyzw(quat_inverse_xyzw(anchor.quat_xyzw), controller_quat_xyzw)
                rotvec = quat_xyzw_to_rotvec(delta_quat)
                scaled_rotvec = rotvec * self.rotation_gain
                residual[offset + 3 : offset + 6] = scaled_rotvec / max(
                    self.residual_rotation_scale,
                    1e-6,
                )
                gripper_residual = index_trigger_to_gripper_residual(
                    index,
                    self.gripper_threshold,
                    self.residual_gripper_deadband,
                )
                residual[offset + 6] = gripper_residual
                debug[hand] = {
                    "active": True,
                    "middle": middle,
                    "index": index,
                    "gripper_active": gripper_residual > 0.0,
                    "gripper_residual": gripper_residual,
                    "pose_mode": "relative",
                    "relative_position": robot_delta.tolist(),
                    "absolute_position": robot_pose.tolist(),
                    "anchor_position": anchor_pose.tolist(),
                    "robot_delta": robot_delta.tolist(),
                    "relative_rotvec": rotvec.tolist(),
                    "controller_robot_delta": controller_robot_delta.tolist(),
                    "controller_quat_xyzw": controller_quat_xyzw.tolist(),
                    "controller_anchor_quat_xyzw": anchor.quat_xyzw.tolist(),
                    "rotvec": rotvec.tolist(),
                    "scaled_rotvec": scaled_rotvec.tolist(),
                    "absolute_rotvec": absolute_rotvec.tolist(),
                    "rotation_gain": self.rotation_gain,
                    "rotation_mode": self.rotation_mode,
                }
            else:
                anchor.active = False
                anchor.position = None
                anchor.quat_xyzw = None
                debug[hand] = {"active": False, "middle": middle, "index": index}

        return residual, any_takeover, debug

    def _update_episode_buttons(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        hand_state = snapshot.get("hands", {}).get(self.episode_button_hand, {})
        buttons = hand_state.get("buttons", []) if hand_state.get("valid", False) else []

        success_value = self._button_value(buttons, self.success_button_index)
        failure_value = self._button_value(buttons, self.failure_button_index)
        success_pressed = success_value >= self.episode_button_threshold
        failure_pressed = failure_value >= self.episode_button_threshold
        success_edge = success_pressed and not self._success_pressed
        failure_edge = failure_pressed and not self._failure_pressed

        self._success_pressed = success_pressed
        self._failure_pressed = failure_pressed

        if success_edge:
            self._pending_outcome = "success"
        elif failure_edge:
            self._pending_outcome = "failure"

        return {
            "hand": self.episode_button_hand,
            "success_button_index": self.success_button_index,
            "failure_button_index": self.failure_button_index,
            "success_value": success_value,
            "failure_value": failure_value,
            "success_edge": success_edge,
            "failure_edge": failure_edge,
            "pending_outcome": self._pending_outcome,
        }

    @staticmethod
    def _button_value(buttons: Any, index: int) -> float:
        if index < 0:
            return 0.0
        try:
            if index >= len(buttons):
                return 0.0
            return float(buttons[index])
        except Exception:
            return 0.0
