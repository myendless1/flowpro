from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    decode_h5_text,
    decode_object_jpeg_dataset,
    format_virtual_video_path,
    quaternion_xyzw_to_wxyz,
    shift_next,
    sort_paths_natural,
)


def _task_name_from_dir_name(task_dir_name: str) -> str:
    parts = str(task_dir_name).split("_")
    if len(parts) <= 4:
        return task_dir_name
    task_core = "_".join(parts[4:-2]).strip("_")
    return task_core or task_dir_name


def _task_prompt_from_hdf5(handle: h5py.File, fallback_task_slug: str) -> str:
    prompt = decode_h5_text(handle.get("prompts", b"")).strip()
    if prompt:
        marker = "described here:"
        lower_prompt = prompt.lower()
        marker_idx = lower_prompt.find(marker)
        if marker_idx >= 0:
            prompt = prompt[marker_idx + len(marker) :].strip()
        return prompt

    instruction = decode_h5_text(handle.get("metadata/language_instruction", b"")).strip()
    if instruction:
        return instruction
    return fallback_task_slug.replace("_", " ")


def _normalize_gripper_channel(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] == 0:
        return arr
    minimum = float(np.nanmin(arr))
    maximum = float(np.nanmax(arr))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Non-finite Songling gripper values")
    if maximum - minimum < 1e-8:
        return np.ones_like(arr, dtype=np.float32)
    return np.clip((arr - minimum) / (maximum - minimum), 0.0, 1.0).astype(np.float32)


def _extract_state(handle: h5py.File) -> np.ndarray:
    left_pose = np.asarray(handle["puppet/end_effector_left_pose_align/data"], dtype=np.float32).reshape(-1, 7)
    right_pose = np.asarray(handle["puppet/end_effector_right_pose_align/data"], dtype=np.float32).reshape(-1, 7)
    left_gripper = _normalize_gripper_channel(
        np.asarray(handle["puppet/end_effector_left_position_align/data"], dtype=np.float32)
    )
    right_gripper = _normalize_gripper_channel(
        np.asarray(handle["puppet/end_effector_right_position_align/data"], dtype=np.float32)
    )
    length = min(
        int(left_pose.shape[0]),
        int(right_pose.shape[0]),
        int(left_gripper.shape[0]),
        int(right_gripper.shape[0]),
    )
    if length <= 0:
        raise ValueError("Empty Songling state sequence")
    return np.concatenate(
        [
            left_pose[:length, :3].astype(np.float32),
            quaternion_xyzw_to_wxyz(left_pose[:length, 3:7]),
            left_gripper[:length],
            right_pose[:length, :3].astype(np.float32),
            quaternion_xyzw_to_wxyz(right_pose[:length, 3:7]),
            right_gripper[:length],
        ],
        axis=1,
    ).astype(np.float32)


class SonglingRawDataset(BaseRawCurationDataset):
    def _build_index(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        task_dirs = sort_paths_natural(path for path in self.input_root.iterdir() if path.is_dir())
        for task_dir in task_dirs:
            task_slug = _task_name_from_dir_name(task_dir.name)
            if not self._match_task_filters(task_dir.name, task_slug):
                continue
            for episode_path in sort_paths_natural(task_dir.glob("episode_*.hdf5")):
                records.append(
                    EpisodeRecord(
                        episode_id=f"{task_dir.name}:{episode_path.name}",
                        payload={
                            "episode_path": str(episode_path),
                            "task_slug": task_slug,
                        },
                    )
                )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        episode_path = Path(record.payload["episode_path"])
        task_slug = str(record.payload["task_slug"])
        with h5py.File(episode_path, "r") as handle:
            states = _extract_state(handle)
            raw_absolute_actions = shift_next(states)
            if self.return_video_path:
                video_frames = {
                    "main": format_virtual_video_path(
                        episode_path, "observations/images/cam_high"
                    ),
                    "left_wrist": format_virtual_video_path(
                        episode_path, "observations/images/cam_left"
                    ),
                    "right_wrist": format_virtual_video_path(
                        episode_path, "observations/images/cam_right"
                    ),
                }
            else:
                video_frames = {
                    "main": decode_object_jpeg_dataset(handle["observations/images/cam_high"]),
                    "left_wrist": decode_object_jpeg_dataset(handle["observations/images/cam_left"]),
                    "right_wrist": decode_object_jpeg_dataset(handle["observations/images/cam_right"]),
                }
            instruction = _task_prompt_from_hdf5(handle, task_slug)

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
