from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import decode_video_mp4, resolve_lerobot_video_chunk_dir
from .lerobot_repo_raw import LeRobotRepoRawDataset, _array_from_column, _maybe_array_from_column


DEFAULT_CAMERA_MAPPING = {
    "main": "observation.images.camera_front",
    "left_main": "observation.images.camera_left",
    "right_main": "observation.images.camera_right",
    "top_main": "observation.images.camera_top",
    "left_wrist": "observation.images.camera_wrist_left",
    "right_wrist": "observation.images.camera_wrist_right",
}


def _concat_columns(table_dict: dict[str, Any], keys: list[str]) -> np.ndarray:
    arrays = [_array_from_column(table_dict, key) for key in keys]
    length = min(int(arr.shape[0]) for arr in arrays)
    if length <= 0:
        raise ValueError(f"Empty arrays for keys={keys}")
    return np.concatenate([arr[:length] for arr in arrays], axis=1).astype(np.float32)


class RoboMindRawDataset(LeRobotRepoRawDataset):
    repo_kind = "robomind"
    default_camera_sources = DEFAULT_CAMERA_MAPPING

    def _resolve_camera_sources(self, repo_root: Path, episode_index: int) -> dict[str, Path]:
        episode_chunk = episode_index // 1000
        videos_root = resolve_lerobot_video_chunk_dir(repo_root, episode_chunk, must_exist=True)
        source_dirs = {
            "front": videos_root / "observation.images.camera_front",
            "left": videos_root / "observation.images.camera_left",
            "right": videos_root / "observation.images.camera_right",
            "top": videos_root / "observation.images.camera_top",
            "wrist_left": videos_root / "observation.images.camera_wrist_left",
            "wrist_right": videos_root / "observation.images.camera_wrist_right",
        }
        source_files = {
            name: directory / f"episode_{episode_index:06d}.mp4"
            for name, directory in source_dirs.items()
        }
        existing = {
            name: path
            for name, path in source_files.items()
            if path.exists()
        }

        resolved: dict[str, Path] = {}
        if "front" in existing:
            resolved["main"] = existing["front"]
        if "top" in existing:
            resolved["top_main"] = existing["top"]

        if "wrist_left" in existing:
            resolved["left_wrist"] = existing["wrist_left"]
            if "left" in existing:
                resolved["left_main"] = existing["left"]
        elif "left" in existing:
            resolved["left_wrist"] = existing["left"]

        if "wrist_right" in existing:
            resolved["right_wrist"] = existing["wrist_right"]
            if "right" in existing:
                resolved["right_main"] = existing["right"]
        elif "right" in existing:
            resolved["right_wrist"] = existing["right"]

        return resolved

    def _load_video_frames(self, *, repo_root: Path, episode_index: int) -> dict[str, np.ndarray]:
        video_frames: dict[str, np.ndarray] = {}
        for output_key, video_path in self._resolve_camera_sources(repo_root, episode_index).items():
            if self.return_video_path:
                video_frames[output_key] = str(video_path)
            else:
                video_frames[output_key] = decode_video_mp4(
                    video_path, backend=self.video_backend
                )
        return video_frames

    def _build_raw_absolute_actions(
        self,
        *,
        repo_root: Path,
        episode_index: int,
        table_dict: dict[str, Any],
    ) -> np.ndarray:
        left_pose = _maybe_array_from_column(table_dict, "action.eef_left_pose")
        right_pose = _maybe_array_from_column(table_dict, "action.eef_right_pose")
        if left_pose is not None and right_pose is not None:
            left_gripper = _maybe_array_from_column(table_dict, "action.eef_left_gripper")
            right_gripper = _maybe_array_from_column(table_dict, "action.eef_right_gripper")
            pieces = [
                left_pose,
                left_gripper if left_gripper is not None else np.zeros((left_pose.shape[0], 1), dtype=np.float32),
                right_pose,
                right_gripper if right_gripper is not None else np.zeros((right_pose.shape[0], 1), dtype=np.float32),
            ]
            length = min(int(arr.shape[0]) for arr in pieces)
            if length <= 0:
                raise ValueError(f"Empty RoboMIND pose episode: {repo_root.name}:{episode_index}")
            return np.concatenate([arr[:length] for arr in pieces], axis=1).astype(np.float32)

        raw_pose_keys = [
            "action.end_effector_left_pose_raw",
            "action.end_effector_left_position_raw",
            "action.end_effector_right_pose_raw",
            "action.end_effector_right_position_raw",
        ]
        if all(key in table_dict for key in raw_pose_keys):
            return _concat_columns(table_dict, raw_pose_keys)

        preferred_keys = [
            "action.arm_left",
            "action.arm_right",
            "action.chassis_pose",
            "action.chassis_twist",
            "action.head_position",
            "action.eef_left_gripper",
            "action.eef_right_gripper",
        ]
        available = [key for key in preferred_keys if key in table_dict]
        if available:
            return _concat_columns(table_dict, available)

        generic_action_keys = sorted(
            key
            for key in table_dict
            if key.startswith("action.")
        )
        if not generic_action_keys:
            raise KeyError(f"No RoboMIND action columns found for {repo_root.name}:{episode_index}")
        return _concat_columns(table_dict, generic_action_keys)


__all__ = ["RoboMindRawDataset"]
