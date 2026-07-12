from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import resolve_lerobot_video_chunk_dir
from .lerobot_repo_raw import LeRobotRepoRawDataset, _array_from_column, _maybe_array_from_column


DEFAULT_CAMERA_MAPPING = {
    "main": "observation.images.camera_head_rgb",
    "left_wrist": "observation.images.camera_left_wrist_rgb",
    "right_wrist": "observation.images.camera_right_wrist_rgb",
    "left_main": "observation.images.cam_third_view",
}

MAIN_CAMERA_CANDIDATES = (
    "observation.images.camera_head_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_front_rgb",
    "observation.images.cam_high_rgb",
)
LEFT_WRIST_CAMERA_CANDIDATES = (
    "observation.images.camera_left_wrist_rgb",
    "observation.images.cam_left_wrist_rgb",
)
RIGHT_WRIST_CAMERA_CANDIDATES = (
    "observation.images.camera_right_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)
THIRD_VIEW_CAMERA_CANDIDATES = (
    "observation.images.cam_third_view",
    "observation.images.camera_third_view",
)


def _pick_first_existing_camera(repo_root: Path, episode_chunk: int, episode_index: int, candidates: tuple[str, ...]) -> str | None:
    videos_root = resolve_lerobot_video_chunk_dir(repo_root, episode_chunk, must_exist=True)
    for source_name in candidates:
        video_path = (
            videos_root
            / source_name
            / f"episode_{episode_index:06d}.mp4"
        )
        if video_path.exists():
            return source_name
    return None


class RoboCoinRawDataset(LeRobotRepoRawDataset):
    repo_kind = "robocoin"
    default_camera_sources = DEFAULT_CAMERA_MAPPING

    def _load_video_frames(self, *, repo_root: Path, episode_index: int) -> dict[str, np.ndarray]:
        episode_chunk = episode_index // 1000
        effective_mapping = dict(self.default_camera_sources)
        effective_mapping.update(self.camera_mapping)

        if "main" not in self.camera_mapping:
            picked = _pick_first_existing_camera(
                repo_root, episode_chunk, episode_index, MAIN_CAMERA_CANDIDATES
            )
            if picked is not None:
                effective_mapping["main"] = picked
        if "left_wrist" not in self.camera_mapping:
            picked = _pick_first_existing_camera(
                repo_root, episode_chunk, episode_index, LEFT_WRIST_CAMERA_CANDIDATES
            )
            if picked is not None:
                effective_mapping["left_wrist"] = picked
        if "right_wrist" not in self.camera_mapping:
            picked = _pick_first_existing_camera(
                repo_root, episode_chunk, episode_index, RIGHT_WRIST_CAMERA_CANDIDATES
            )
            if picked is not None:
                effective_mapping["right_wrist"] = picked
        if "left_main" not in self.camera_mapping:
            picked = _pick_first_existing_camera(
                repo_root, episode_chunk, episode_index, THIRD_VIEW_CAMERA_CANDIDATES
            )
            if picked is not None:
                effective_mapping["left_main"] = picked

        old_mapping = self.effective_camera_mapping
        try:
            self.effective_camera_mapping = effective_mapping
            return super()._load_video_frames(repo_root=repo_root, episode_index=episode_index)
        finally:
            self.effective_camera_mapping = old_mapping

    def _build_raw_absolute_actions(
        self,
        *,
        repo_root: Path,
        episode_index: int,
        table_dict: dict[str, Any],
    ) -> np.ndarray:
        pose_action = _maybe_array_from_column(table_dict, "eef_sim_pose_action")
        gripper_action = _maybe_array_from_column(table_dict, "gripper_open_scale_action")

        if pose_action is not None:
            if gripper_action is None:
                gripper_action = np.zeros((pose_action.shape[0], 2), dtype=np.float32)
            pieces = [pose_action, gripper_action]
            length = min(int(arr.shape[0]) for arr in pieces)
            if length <= 0:
                raise ValueError(f"Empty RoboCOIN episode: {repo_root.name}:{episode_index}")
            return np.concatenate([arr[:length] for arr in pieces], axis=1).astype(np.float32)

        action = _array_from_column(table_dict, "action")
        if action.shape[0] <= 0:
            raise ValueError(f"Empty RoboCOIN action sequence: {repo_root.name}:{episode_index}")
        return action.astype(np.float32, copy=False)


__all__ = ["RoboCoinRawDataset"]
