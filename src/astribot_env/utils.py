from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ACTION16_DIM = 16
RESIDUAL14_DIM = 14


@dataclass(frozen=True)
class DualArmAction16:
    left_pose: np.ndarray
    left_gripper: float
    right_pose: np.ndarray
    right_gripper: float


def normalize_quat_xyzw(quat_xyzw: Iterable[float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32, copy=False)


def quat_wxyz_to_xyzw(quat_wxyz: Iterable[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    return quat[[1, 2, 3, 0]].astype(np.float32, copy=False)


def quat_xyzw_to_wxyz(quat_xyzw: Iterable[float]) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    return quat[[3, 0, 1, 2]].astype(np.float32, copy=False)


def action_quat_to_sdk_xyzw(quat: Iterable[float], *, use_xyzw: bool) -> np.ndarray:
    if use_xyzw:
        return normalize_quat_xyzw(quat)
    return quat_wxyz_to_xyzw(quat)


def sdk_xyzw_to_action_quat(quat_xyzw: Iterable[float], *, use_xyzw: bool) -> np.ndarray:
    if use_xyzw:
        return normalize_quat_xyzw(quat_xyzw)
    return quat_xyzw_to_wxyz(quat_xyzw)


def quat_multiply_xyzw(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    ax, ay, az, aw = normalize_quat_xyzw(a)
    bx, by, bz, bw = normalize_quat_xyzw(b)
    return normalize_quat_xyzw(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_inverse_xyzw(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw)
    return np.asarray([-x, -y, -z, w], dtype=np.float32)


def rotvec_to_quat_xyzw(rotvec: Iterable[float]) -> np.ndarray:
    vec = np.asarray(rotvec, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(vec))
    if angle <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = vec / angle
    s = np.sin(angle * 0.5)
    return normalize_quat_xyzw([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)])


def quat_xyzw_to_rotvec(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8:
        return np.asarray([2.0 * x, 2.0 * y, 2.0 * z], dtype=np.float32)
    return np.asarray([x, y, z], dtype=np.float32) * (angle / sin_half)


def quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw).astype(np.float64)
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


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float32,
        )
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float32,
            )
        elif idx == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float32,
            )
    return normalize_quat_xyzw(quat)


def _local_axis_vector(axis_name: str) -> np.ndarray:
    sign = -1.0 if str(axis_name).startswith("-") else 1.0
    axis = str(axis_name)[-1]
    if axis == "x":
        return np.asarray([sign, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.asarray([0.0, sign, 0.0], dtype=np.float64)
    return np.asarray([0.0, 0.0, sign], dtype=np.float64)


def _line_plane_angle_deg(direction: Iterable[float]) -> float:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return float(np.degrees(np.arcsin(np.clip(abs(direction[2]), 0.0, 1.0))))


def _wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _axis_angle_rotation_matrix(axis: Iterable[float], angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_between_vectors(src: Iterable[float], dst: Iterable[float]) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64).reshape(3)
    dst = np.asarray(dst, dtype=np.float64).reshape(3)
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        axis = np.cross(src, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(src, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    axis = np.cross(src, dst)
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _twist_rotation_to_level_axis(rotation: np.ndarray, *, ray_axis: str, level_axis: str) -> np.ndarray:
    local_ray = _local_axis_vector(ray_axis)
    local_level = _local_axis_vector(level_axis)
    ray = rotation @ local_ray
    ray = ray / (np.linalg.norm(ray) + 1e-12)
    level = rotation @ local_level

    ray_dot_level = float(np.dot(ray, level))
    a = float(level[2] - ray[2] * ray_dot_level)
    b = float(np.cross(ray, level)[2])
    c = float(ray[2] * ray_dot_level)
    radius = float(np.hypot(a, b))
    if radius < 1e-9:
        return np.eye(3, dtype=np.float64)

    phi = float(np.arctan2(b, a))
    target = float(np.clip(-c / radius, -1.0, 1.0))
    delta = float(np.arccos(target))
    candidates = [_wrap_to_pi(phi + delta), _wrap_to_pi(phi - delta)]
    twist_angle = min(candidates, key=lambda value: abs(value))
    return _axis_angle_rotation_matrix(ray, twist_angle)


def adjust_quat_ray_angle_to_horizontal(
    quat_xyzw: Iterable[float],
    *,
    target_angle_deg: float,
    ray_axis: str,
    level_axis: str | None = None,
    keep_level_axis_horizontal: bool = True,
) -> np.ndarray:
    rotation = quat_xyzw_to_matrix(quat_xyzw)
    local_ray = _local_axis_vector(ray_axis)
    current_ray = rotation @ local_ray

    horizontal = current_ray.copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-8:
        horizontal = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-8:
            horizontal = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            horizontal_norm = 1.0
    horizontal /= horizontal_norm

    target_angle = np.deg2rad(float(target_angle_deg))
    vertical_sign = -1.0 if float(current_ray[2]) < 0.0 else 1.0
    target_ray = horizontal * np.cos(target_angle)
    target_ray[2] = vertical_sign * np.sin(target_angle)
    target_ray /= np.linalg.norm(target_ray) + 1e-12

    correction = _rotation_matrix_between_vectors(current_ray, target_ray)
    adjusted_rotation = correction @ rotation
    if keep_level_axis_horizontal and level_axis is not None:
        adjusted_rotation = _twist_rotation_to_level_axis(
            adjusted_rotation,
            ray_axis=ray_axis,
            level_axis=level_axis,
        ) @ adjusted_rotation
    return matrix_to_quat_xyzw(adjusted_rotation)


def convert_gripper_value_to_cmd_value(gripper_value: float) -> float:
    value = float(np.clip(gripper_value, 0.0, 1.0))
    return (1.0 - value) * 100.0


def convert_gripper_cmd_value_to_action_value(cmd_value: float) -> float:
    value = float(np.clip(cmd_value, 0.0, 100.0))
    return 1.0 - value / 100.0


def split_action16(action: Iterable[float]) -> DualArmAction16:
    action = np.asarray(action, dtype=np.float32).reshape(ACTION16_DIM)
    return DualArmAction16(
        left_pose=action[0:7].copy(),
        left_gripper=float(action[7]),
        right_pose=action[8:15].copy(),
        right_gripper=float(action[15]),
    )


def assemble_action16(
    left_pose: Iterable[float],
    left_gripper: float,
    right_pose: Iterable[float],
    right_gripper: float,
) -> np.ndarray:
    action = np.zeros((ACTION16_DIM,), dtype=np.float32)
    action[0:7] = np.asarray(left_pose, dtype=np.float32).reshape(7)
    action[7] = float(left_gripper)
    action[8:15] = np.asarray(right_pose, dtype=np.float32).reshape(7)
    action[15] = float(right_gripper)
    return action


def first_action_from_wam4d_response(action: np.ndarray) -> np.ndarray:
    return actions_from_wam4d_response(action, max_steps=1)[0]


def actions_from_wam4d_response(action: np.ndarray, max_steps: int | None = None) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape == (ACTION16_DIM,):
        return action.reshape(1, ACTION16_DIM).copy()
    if action.ndim == 3 and action.shape[0] == ACTION16_DIM:
        selected_actions = []
        for group_idx in range(action.shape[1]):
            for horizon_idx in range(action.shape[2]):
                selected_actions.append(action[:, group_idx, horizon_idx].astype(np.float32, copy=True))
                if max_steps is not None and len(selected_actions) >= int(max_steps):
                    return np.asarray(selected_actions, dtype=np.float32)
        return np.asarray(selected_actions, dtype=np.float32)
    if action.ndim == 3 and action.shape[-1] == ACTION16_DIM:
        selected_actions = []
        for group_idx in range(action.shape[0]):
            for horizon_idx in range(action.shape[1]):
                selected_actions.append(action[group_idx, horizon_idx, :].astype(np.float32, copy=True))
                if max_steps is not None and len(selected_actions) >= int(max_steps):
                    return np.asarray(selected_actions, dtype=np.float32)
        return np.asarray(selected_actions, dtype=np.float32)
    raise ValueError(f"Expected WAM4D action shape (16,), (16,G,H), or (G,H,16), got {action.shape}")


def fuse_wam4d_prior_with_residual(
    prior_action16: Iterable[float],
    residual_action14: Iterable[float],
    *,
    position_scale: float,
    rotation_scale: float,
    use_xyzw: bool,
    gripper_deadband: float = 0.5,
    clip_residual: bool = True,
) -> np.ndarray:
    prior = np.asarray(prior_action16, dtype=np.float32).reshape(ACTION16_DIM).copy()
    residual = np.asarray(residual_action14, dtype=np.float32).reshape(RESIDUAL14_DIM)
    if clip_residual:
        residual = np.clip(residual, -1.0, 1.0)

    out = prior.copy()
    for pose_start, grip_index, residual_start in ((0, 7, 0), (8, 15, 7)):
        out[pose_start : pose_start + 3] = (
            prior[pose_start : pose_start + 3]
            + residual[residual_start : residual_start + 3] * float(position_scale)
        )

        base_quat_xyzw = action_quat_to_sdk_xyzw(
            prior[pose_start + 3 : pose_start + 7],
            use_xyzw=use_xyzw,
        )
        delta_quat_xyzw = rotvec_to_quat_xyzw(
            residual[residual_start + 3 : residual_start + 6] * float(rotation_scale)
        )
        fused_quat_xyzw = quat_multiply_xyzw(delta_quat_xyzw, base_quat_xyzw)
        out[pose_start + 3 : pose_start + 7] = sdk_xyzw_to_action_quat(
            fused_quat_xyzw,
            use_xyzw=use_xyzw,
        )

        grip_residual = float(residual[residual_start + 6])
        if grip_residual <= -float(gripper_deadband):
            out[grip_index] = 1.0
        elif grip_residual >= float(gripper_deadband):
            close_fraction = (grip_residual - float(gripper_deadband)) / max(
                1.0 - float(gripper_deadband),
                1e-6,
            )
            out[grip_index] = 1.0 - float(np.clip(close_fraction, 0.0, 1.0))
        else:
            out[grip_index] = float(np.clip(out[grip_index], 0.0, 1.0))

    return out.astype(np.float32, copy=False)


def sac_label_from_executed_action16(
    executed_action16: Iterable[float],
    prior_action16: Iterable[float],
    *,
    position_scale: float,
    rotation_scale: float,
    use_xyzw: bool,
    action_bound: float = 1.0,
    bound_tolerance: float = 1e-5,
) -> dict:
    executed = np.asarray(executed_action16, dtype=np.float32).reshape(ACTION16_DIM)
    prior = np.asarray(prior_action16, dtype=np.float32).reshape(ACTION16_DIM)
    label = np.zeros((RESIDUAL14_DIM,), dtype=np.float32)
    pose_rotation_residuals = []
    pose_residuals = []
    rotation_residuals = []

    for pose_start, grip_index, residual_start in ((0, 7, 0), (8, 15, 7)):
        label[residual_start : residual_start + 3] = (
            executed[pose_start : pose_start + 3] - prior[pose_start : pose_start + 3]
        ) / max(float(position_scale), 1e-8)
        pose_residuals.append(label[residual_start : residual_start + 3])

        exec_quat_xyzw = action_quat_to_sdk_xyzw(
            executed[pose_start + 3 : pose_start + 7],
            use_xyzw=use_xyzw,
        )
        prior_quat_xyzw = action_quat_to_sdk_xyzw(
            prior[pose_start + 3 : pose_start + 7],
            use_xyzw=use_xyzw,
        )
        delta_quat_xyzw = quat_multiply_xyzw(exec_quat_xyzw, quat_inverse_xyzw(prior_quat_xyzw))
        label[residual_start + 3 : residual_start + 6] = quat_xyzw_to_rotvec(
            delta_quat_xyzw
        ) / max(float(rotation_scale), 1e-8)
        rotation_residuals.append(label[residual_start + 3 : residual_start + 6])

        label[residual_start + 6] = float(executed[grip_index])
        pose_rotation_residuals.append(label[residual_start : residual_start + 6])

    raw_pose_rotation_residual = np.concatenate(pose_rotation_residuals).astype(
        np.float32,
        copy=False,
    )
    max_abs_pose_rotation = float(np.max(np.abs(raw_pose_rotation_residual)))
    max_abs_pose = float(np.max(np.abs(np.concatenate(pose_residuals))))
    max_abs_rotation = float(np.max(np.abs(np.concatenate(rotation_residuals))))
    bound = float(action_bound)
    tolerance = float(bound_tolerance)
    pose_out_of_bounds = bool(max_abs_pose > bound + tolerance)
    rotation_out_of_bounds = bool(max_abs_rotation > bound + tolerance)
    pose_rotation_out_of_bounds = pose_out_of_bounds or rotation_out_of_bounds
    gripper_values = label[[6, 13]]
    gripper_out_of_bounds = bool(
        np.any(gripper_values < -bound - tolerance)
        or np.any(gripper_values > bound + tolerance)
    )
    drop_reasons = []
    if pose_out_of_bounds:
        drop_reasons.append("pose_residual_out_of_bounds")
    if rotation_out_of_bounds:
        drop_reasons.append("rotation_residual_out_of_bounds")
    if gripper_out_of_bounds:
        drop_reasons.append("gripper_label_out_of_bounds")

    return {
        "sac_action_label14": label.astype(np.float32, copy=False),
        "raw_pose_rotation_residual_label": raw_pose_rotation_residual,
        "sac_action_label_valid": not drop_reasons,
        "pose_rotation_residual_label_max_abs": max_abs_pose_rotation,
        "pose_residual_label_max_abs": max_abs_pose,
        "rotation_residual_label_max_abs": max_abs_rotation,
        "pose_rotation_residual_clipped_any": pose_rotation_out_of_bounds,
        "pose_residual_clipped_any": pose_out_of_bounds,
        "rotation_residual_clipped_any": rotation_out_of_bounds,
        "drop_reason": ",".join(drop_reasons) if drop_reasons else None,
    }


def action16_to_sdk_commands(action16: Iterable[float], *, use_xyzw: bool) -> tuple[list[list[float]], list[list[float]]]:
    action = split_action16(action16)
    left_pose = action.left_pose.astype(np.float32, copy=True)
    right_pose = action.right_pose.astype(np.float32, copy=True)
    left_pose[3:7] = action_quat_to_sdk_xyzw(left_pose[3:7], use_xyzw=use_xyzw)
    right_pose[3:7] = action_quat_to_sdk_xyzw(right_pose[3:7], use_xyzw=use_xyzw)
    arm_poses = [left_pose.tolist(), right_pose.tolist()]
    grippers = [
        [convert_gripper_value_to_cmd_value(action.left_gripper)],
        [convert_gripper_value_to_cmd_value(action.right_gripper)],
    ]
    return arm_poses, grippers


def apply_xyz_limits(
    action16: Iterable[float],
    *,
    left_low: Iterable[float] | None = None,
    left_high: Iterable[float] | None = None,
    right_low: Iterable[float] | None = None,
    right_high: Iterable[float] | None = None,
    right_min_z: float | None = None,
) -> np.ndarray:
    action = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM).copy()
    if left_low is not None and left_high is not None:
        action[0:3] = np.clip(action[0:3], np.asarray(left_low, dtype=np.float32), np.asarray(left_high, dtype=np.float32))
    if right_low is not None and right_high is not None:
        action[8:11] = np.clip(action[8:11], np.asarray(right_low, dtype=np.float32), np.asarray(right_high, dtype=np.float32))
    if right_min_z is not None:
        action[10] = max(float(action[10]), float(right_min_z))
    action[7] = float(np.clip(action[7], 0.0, 1.0))
    action[15] = float(np.clip(action[15], 0.0, 1.0))
    return action


def apply_right_gripper_orientation_constraint(
    action16: Iterable[float],
    *,
    enabled: bool,
    use_xyzw: bool,
    target_angle_deg: float = 45.0,
    ray_axis: str = "+z",
    level_axis: str = "+x",
    keep_level_axis_horizontal: bool = True,
) -> np.ndarray:
    action = np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM).copy()
    if not enabled:
        return action
    quat_xyzw = action_quat_to_sdk_xyzw(action[11:15], use_xyzw=use_xyzw)
    adjusted_quat_xyzw = adjust_quat_ray_angle_to_horizontal(
        quat_xyzw,
        target_angle_deg=target_angle_deg,
        ray_axis=ray_axis,
        level_axis=level_axis,
        keep_level_axis_horizontal=keep_level_axis_horizontal,
    )
    action[11:15] = sdk_xyzw_to_action_quat(adjusted_quat_xyzw, use_xyzw=use_xyzw)
    return action.astype(np.float32, copy=False)
