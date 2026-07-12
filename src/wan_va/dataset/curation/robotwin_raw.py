from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    align_quaternion_sequence_wxyz,
    decode_h5_text,
    decode_object_jpeg_dataset,
    format_virtual_video_path,
    standardize_quaternion_wxyz,
)


def _task_slug_from_dir(task_dir: Path) -> str:
    name = str(task_dir.name).strip()
    if name in {
        "demo_clean_collect_200",
        "aloha-agilex_clean_50",
        "aloha-agilex_randomized_500",
    }:
        parent_name = str(task_dir.parent.name).strip()
        if parent_name and parent_name.lower() not in {"dataset", "data"}:
            return parent_name
    for suffix in [
        "-demo_clean_collect_200-50",
        "-aloha-agilex_clean_50-50",
        "-aloha-agilex_randomized_500-1000",
        "-aloha-agilex_clean_50",
        "-aloha-agilex_randomized_500",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _task_repo_name(task_dir: Path, task_slug: str | None = None) -> str:
    slug = str(task_slug or _task_slug_from_dir(task_dir)).strip()
    dir_name = str(task_dir.name).strip()
    if not slug:
        return dir_name
    if dir_name == slug or dir_name.startswith(f"{slug}-"):
        return dir_name
    return f"{slug}-{dir_name}"


ROBOTWIN_DEFAULT_REPO_DIR_NAMES = (
    "aloha-agilex_clean_50",
    "aloha-agilex_randomized_500",
)


def _iter_instruction_candidates(task_dir: Path, episode_stem: str) -> list[Path]:
    return [
        task_dir / "instructions.json",
        task_dir / "instructions" / f"{episode_stem}.json",
        task_dir / "instructions" / f"{episode_stem.replace('episode', 'episode_')}.json",
    ]


def _load_instruction(task_dir: Path, episode_stem: str, fallback_task: str) -> str:
    for path in _iter_instruction_candidates(task_dir, episode_stem):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            for key in ["instructions", "seen", "unseen"]:
                value = payload.get(key)
                if isinstance(value, list) and value:
                    text = str(value[0]).strip()
                    if text:
                        return text
            for key in ["instruction", "task", "prompt"]:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(payload, list) and payload:
            text = str(payload[0]).strip()
            if text:
                return text

    return fallback_task.replace("_", " ").replace("-", " ").strip()


def _ensure_2d_float(arr: Any, width: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {out.shape}")
    if width is not None and out.shape[1] != width:
        raise ValueError(f"Expected width {width}, got shape {out.shape}")
    return out.astype(np.float32, copy=False)


def _normalize_gripper_channel(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] <= 0:
        return arr.astype(np.float32)
    minimum = float(np.nanmin(arr))
    maximum = float(np.nanmax(arr))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Non-finite RobotWin gripper values")
    if maximum - minimum < 1e-8:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return np.clip((arr - minimum) / (maximum - minimum), 0.0, 1.0).astype(np.float32)


def _build_abs_eef_state_from_merged_hdf5(handle: h5py.File) -> np.ndarray:
    left_pose = _ensure_2d_float(handle["endpose/left_endpose"], width=7)
    right_pose = _ensure_2d_float(handle["endpose/right_endpose"], width=7)
    left_gripper = _normalize_gripper_channel(handle["endpose/left_gripper"])
    right_gripper = _normalize_gripper_channel(handle["endpose/right_gripper"])
    length = min(
        int(left_pose.shape[0]),
        int(right_pose.shape[0]),
        int(left_gripper.shape[0]),
        int(right_gripper.shape[0]),
    )
    if length <= 0:
        raise ValueError("Empty RobotWin merged HDF5 EEF state sequence")
    left_quat = align_quaternion_sequence_wxyz(
        standardize_quaternion_wxyz(left_pose[:length, 3:7])
    )
    right_quat = align_quaternion_sequence_wxyz(
        standardize_quaternion_wxyz(right_pose[:length, 3:7])
    )
    return np.concatenate(
        [
            left_pose[:length, :3],
            left_quat,
            left_gripper[:length],
            right_pose[:length, :3],
            right_quat,
            right_gripper[:length],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _build_qpos_from_merged_hdf5(handle: h5py.File) -> np.ndarray | None:
    if "joint_action/vector" in handle:
        return _ensure_2d_float(handle["joint_action/vector"])

    required = [
        "joint_action/left_arm",
        "joint_action/left_gripper",
        "joint_action/right_arm",
        "joint_action/right_gripper",
    ]
    if not all(key in handle for key in required):
        return None

    left_arm = _ensure_2d_float(handle["joint_action/left_arm"])
    left_gripper = _ensure_2d_float(handle["joint_action/left_gripper"])
    right_arm = _ensure_2d_float(handle["joint_action/right_arm"])
    right_gripper = _ensure_2d_float(handle["joint_action/right_gripper"])
    length = min(
        int(left_arm.shape[0]),
        int(left_gripper.shape[0]),
        int(right_arm.shape[0]),
        int(right_gripper.shape[0]),
    )
    if length <= 0:
        return None
    return np.concatenate(
        [
            left_arm[:length],
            left_gripper[:length],
            right_arm[:length],
            right_gripper[:length],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _load_processed_hdf5_episode(
    handle: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if "/observations/qpos" not in handle or "/action" not in handle:
        raise KeyError("Processed RobotWin HDF5 requires `/observations/qpos` and `/action`")
    states = _ensure_2d_float(handle["/observations/qpos"])
    actions = _ensure_2d_float(handle["/action"])
    length = min(int(states.shape[0]), int(actions.shape[0]))
    if length <= 0:
        raise ValueError("Empty RobotWin processed HDF5 sequence")
    return actions[:length], states[:length], states[:length]


def _load_merged_hdf5_episode(
    handle: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    states = _build_abs_eef_state_from_merged_hdf5(handle)
    if states.shape[0] <= 1:
        raise ValueError("RobotWin merged HDF5 needs at least 2 frames")
    actions = states[1:].astype(np.float32, copy=False)
    states = states[:-1].astype(np.float32, copy=False)
    qpos = _build_qpos_from_merged_hdf5(handle)
    if qpos is not None:
        qpos = qpos[: states.shape[0]].astype(np.float32, copy=False)
    return actions, states, qpos


def _load_episode_payload(
    episode_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(episode_path, "r") as handle:
        if "endpose" in handle:
            return _load_merged_hdf5_episode(handle)
        return _load_processed_hdf5_episode(handle)


def _build_virtual_video_frames(episode_path: Path) -> dict[str, str]:
    with h5py.File(episode_path, "r") as handle:
        if "observation/head_camera/rgb" in handle:
            return {
                "main": format_virtual_video_path(episode_path, "observation/head_camera/rgb"),
                "left_wrist": format_virtual_video_path(episode_path, "observation/left_camera/rgb"),
                "right_wrist": format_virtual_video_path(episode_path, "observation/right_camera/rgb"),
            }
        if "observations/images/cam_high" in handle:
            return {
                "main": format_virtual_video_path(episode_path, "observations/images/cam_high"),
                "left_wrist": format_virtual_video_path(episode_path, "observations/images/cam_left_wrist"),
                "right_wrist": format_virtual_video_path(episode_path, "observations/images/cam_right_wrist"),
            }
    raise KeyError(f"Unrecognized RobotWin image layout in {episode_path}")


def _count_hdf5_video_frames(handle: h5py.File, inner_path: str) -> int:
    dataset = handle[inner_path]
    size_key = f"{inner_path}_size"
    if size_key in handle:
        return int(np.asarray(handle[size_key]).reshape(-1).shape[0])
    if len(dataset.shape) <= 0:
        raise ValueError(f"HDF5 dataset has no frame axis: {inner_path}")
    return int(dataset.shape[0])


def _video_frame_counts(episode_path: Path) -> dict[str, int]:
    with h5py.File(episode_path, "r") as handle:
        if "observation/head_camera/rgb" in handle:
            return {
                "main": _count_hdf5_video_frames(handle, "observation/head_camera/rgb"),
                "left_wrist": _count_hdf5_video_frames(handle, "observation/left_camera/rgb"),
                "right_wrist": _count_hdf5_video_frames(handle, "observation/right_camera/rgb"),
            }
        if "observations/images/cam_high" in handle:
            return {
                "main": _count_hdf5_video_frames(handle, "observations/images/cam_high"),
                "left_wrist": _count_hdf5_video_frames(handle, "observations/images/cam_left_wrist"),
                "right_wrist": _count_hdf5_video_frames(handle, "observations/images/cam_right_wrist"),
            }
    raise KeyError(f"Unrecognized RobotWin image layout in {episode_path}")


def _decode_video_frames(episode_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(episode_path, "r") as handle:
        if "observation/head_camera/rgb" in handle:
            return {
                "main": decode_object_jpeg_dataset(handle["observation/head_camera/rgb"]),
                "left_wrist": decode_object_jpeg_dataset(handle["observation/left_camera/rgb"]),
                "right_wrist": decode_object_jpeg_dataset(handle["observation/right_camera/rgb"]),
            }
        if "observations/images/cam_high" in handle:
            return {
                "main": decode_object_jpeg_dataset(handle["observations/images/cam_high"]),
                "left_wrist": decode_object_jpeg_dataset(handle["observations/images/cam_left_wrist"]),
                "right_wrist": decode_object_jpeg_dataset(handle["observations/images/cam_right_wrist"]),
            }
    raise KeyError(f"Unrecognized RobotWin image layout in {episode_path}")


class RobotWinRawDataset(BaseRawCurationDataset):
    def __init__(
        self,
        *args: Any,
        allowed_repo_dir_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.allowed_repo_dir_names = {
            str(name).strip()
            for name in (allowed_repo_dir_names or ())
            if str(name).strip()
        }
        super().__init__(*args, **kwargs)

    def _build_index(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        episode_paths = sorted(self.input_root.rglob("episode*.hdf5"), key=lambda p: str(p))
        for episode_path in episode_paths:
            task_dir = episode_path.parent.parent if episode_path.parent.name == "data" else episode_path.parent
            if self.allowed_repo_dir_names and task_dir.name not in self.allowed_repo_dir_names:
                continue
            task_slug = _task_slug_from_dir(task_dir)
            task_repo_name = _task_repo_name(task_dir, task_slug)
            episode_stem = episode_path.stem
            instruction = _load_instruction(task_dir, episode_stem, task_slug)
            if not self._match_task_filters(task_dir.name, task_slug, instruction, episode_stem):
                continue
            records.append(
                EpisodeRecord(
                    episode_id=f"{task_dir.name}:{episode_stem}",
                    payload={
                        "episode_path": str(episode_path),
                        "instruction": instruction,
                        "task_slug": task_slug,
                        "task_repo_name": task_repo_name,
                        "task_dir_name": str(task_dir.name),
                    },
                )
            )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        episode_path = Path(record.payload["episode_path"])
        instruction = str(record.payload["instruction"])

        raw_absolute_actions, raw_states, raw_qpos = _load_episode_payload(episode_path)
        if self.return_video_path:
            video_frames = _build_virtual_video_frames(episode_path)
            video_lengths = _video_frame_counts(episode_path)
        else:
            video_frames = _decode_video_frames(episode_path)
            video_lengths = {
                key: int(frames.shape[0]) for key, frames in video_frames.items()
            }

        num_frames = min(
            int(raw_absolute_actions.shape[0]),
            int(raw_states.shape[0]),
            *(int(length) for length in video_lengths.values()),
        )
        if raw_qpos is not None:
            num_frames = min(num_frames, int(raw_qpos.shape[0]))
        if num_frames <= 0:
            raise ValueError(f"Empty RobotWin episode after alignment: {record.episode_id}")

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions[:num_frames],
            raw_states=raw_states[:num_frames],
            raw_qpos=None if raw_qpos is None else raw_qpos[:num_frames],
            instruction=instruction,
        )
