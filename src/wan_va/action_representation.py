"""Delta-EEF representation shared by training and inference.

The 16-D command layout is::

    left delta_xyz+relative_wxyz+absolute_gripper,
    right delta_xyz+relative_wxyz+absolute_gripper

The diffusion model keeps its historical 30-D layout. Delta EEF targets occupy
the execution channels and optional release-pose targets occupy
the otherwise unused 14 channels.
"""

from __future__ import annotations

import numpy as np


EXECUTION_CHANNEL_IDS = tuple(
    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
)
LEFT_RELEASE_CHANNEL_IDS = tuple(range(14, 21))
RIGHT_RELEASE_CHANNEL_IDS = tuple(range(21, 28))
RELEASE_CHANNEL_IDS = LEFT_RELEASE_CHANNEL_IDS + RIGHT_RELEASE_CHANNEL_IDS


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    identity = np.zeros_like(q)
    identity[..., 0] = 1.0
    q = np.where(norm > 1e-8, q / np.maximum(norm, 1e-8), identity)
    # q and -q are equivalent.  Canonicalization removes target discontinuities.
    return np.where(q[..., :1] < 0.0, -q, q).astype(np.float32, copy=False)


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = normalize_quaternion_wxyz(lhs)
    rhs = normalize_quaternion_wxyz(rhs)
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    result = np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )
    return normalize_quaternion_wxyz(result)


def quaternion_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_wxyz(quaternion).copy()
    q[..., 1:] *= -1.0
    return q


def relative_pose7(reference_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return world-frame delta xyz and local relative rotation.

    ``q_delta = inverse(q_reference) * q_target`` and therefore
    ``q_target = q_reference * q_delta``.
    """
    reference = np.asarray(reference_pose, dtype=np.float32)
    target = np.asarray(target_pose, dtype=np.float32)
    delta_position = target[..., :3] - reference[..., :3]
    delta_rotation = quaternion_multiply_wxyz(
        quaternion_inverse_wxyz(reference[..., 3:7]), target[..., 3:7]
    )
    return np.concatenate([delta_position, delta_rotation], axis=-1).astype(
        np.float32, copy=False
    )


def apply_relative_pose7(reference_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_pose, dtype=np.float32)
    delta = np.asarray(delta_pose, dtype=np.float32)
    position = reference[..., :3] + delta[..., :3]
    rotation = quaternion_multiply_wxyz(reference[..., 3:7], delta[..., 3:7])
    return np.concatenate([position, rotation], axis=-1).astype(np.float32, copy=False)


def delta16_to_model30(
    absolute_actions: np.ndarray,
    *,
    references: np.ndarray | None = None,
    release_poses: np.ndarray | None = None,
    release_references: np.ndarray | None = None,
    release_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode absolute robot commands as delta-EEF model targets and a loss mask."""
    absolute = np.asarray(absolute_actions, dtype=np.float32)
    if absolute.ndim != 2 or absolute.shape[1] != 16:
        raise ValueError(f"Expected absolute actions [T,16], got {absolute.shape}")
    encoded16 = absolute.copy()
    if references is None:
        references = np.concatenate([absolute[:1], absolute[:-1]], axis=0)
    references = np.asarray(references, dtype=np.float32)
    if references.shape != absolute.shape:
        raise ValueError(
            f"Delta references must match actions {absolute.shape}, got {references.shape}"
        )
    encoded16[:, 0:7] = relative_pose7(references[:, 0:7], absolute[:, 0:7])
    encoded16[:, 8:15] = relative_pose7(references[:, 8:15], absolute[:, 8:15])
    # Gripper targets remain absolute, matching delta.json statistics.

    model = np.zeros((absolute.shape[0], 30), dtype=np.float32)
    mask = np.zeros_like(model, dtype=bool)
    model[:, EXECUTION_CHANNEL_IDS] = encoded16
    mask[:, EXECUTION_CHANNEL_IDS] = True

    if release_poses is not None:
        release_poses = np.asarray(release_poses, dtype=np.float32)
        if release_poses.shape != absolute.shape:
            raise ValueError(
                f"Release poses must match actions {absolute.shape}, got {release_poses.shape}"
            )
        if release_valid is None:
            release_valid = np.ones((absolute.shape[0], 2), dtype=bool)
        release_valid = np.asarray(release_valid, dtype=bool)
        if release_valid.shape != (absolute.shape[0], 2):
            raise ValueError(f"Expected release_valid [T,2], got {release_valid.shape}")
        if release_references is None:
            release_references = absolute
        release_references = np.asarray(release_references, dtype=np.float32)
        if release_references.shape != absolute.shape:
            raise ValueError(
                "Release references must match actions "
                f"{absolute.shape}, got {release_references.shape}"
            )
        left_release = relative_pose7(
            release_references[:, 0:7], release_poses[:, 0:7]
        )
        right_release = relative_pose7(
            release_references[:, 8:15], release_poses[:, 8:15]
        )
        model[:, LEFT_RELEASE_CHANNEL_IDS] = left_release
        model[:, RIGHT_RELEASE_CHANNEL_IDS] = right_release
        mask[:, LEFT_RELEASE_CHANNEL_IDS] = release_valid[:, 0:1]
        mask[:, RIGHT_RELEASE_CHANNEL_IDS] = release_valid[:, 1:2]
    return model, mask


def model30_to_execution16(model_actions: np.ndarray) -> np.ndarray:
    model = np.asarray(model_actions, dtype=np.float32)
    return model[..., EXECUTION_CHANNEL_IDS].astype(np.float32, copy=False)


def decode_execution_sequence(
    execution_actions: np.ndarray,
    *,
    initial_absolute: np.ndarray,
) -> np.ndarray:
    """Decode a temporal 16-D model sequence into absolute robot commands."""
    encoded = np.asarray(execution_actions, dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[1] != 16:
        raise ValueError(f"Expected execution sequence [T,16], got {encoded.shape}")
    previous = np.asarray(initial_absolute, dtype=np.float32).reshape(16).copy()
    decoded = np.empty_like(encoded)
    for index, delta in enumerate(encoded):
        current = np.empty((16,), dtype=np.float32)
        current[0:7] = apply_relative_pose7(previous[0:7], delta[0:7])
        current[7] = delta[7]
        current[8:15] = apply_relative_pose7(previous[8:15], delta[8:15])
        current[15] = delta[15]
        decoded[index] = current
        previous = current
    return decoded


def absolute_history_to_delta(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 16:
        raise ValueError(f"Expected absolute history [T,16], got {history.shape}")
    if history.shape[0] == 0:
        return history.copy()
    references = np.concatenate([history[:1], history[:-1]], axis=0)
    model, _ = delta16_to_model30(history, references=references)
    return model30_to_execution16(model)
