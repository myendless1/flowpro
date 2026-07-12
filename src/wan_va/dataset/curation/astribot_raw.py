from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    GripperNormSpec,
    decode_concat_jpeg_stream,
    format_virtual_video_path,
    load_array,
    normalize_gripper_with_spec,
    quaternion_xyzw_to_wxyz,
    sort_paths_natural,
)


DEFAULT_CAMERA_MAPPING = {
    "main": "head",
    "left_wrist": "left",
    "right_wrist": "right",
}


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_episode_(\d+)\.hdf5$", path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


class AstribotRawDataset(BaseRawCurationDataset):
    def __init__(
        self,
        input_root: str | Path,
        split: str | tuple[str, ...] | None = None,
        task_filter: str | tuple[str, ...] | None = None,
        max_episodes: int = 0,
        video_backend: str = "opencv",
        camera_mapping: dict[str, str] | None = None,
        return_video_path: bool = False,
    ) -> None:
        super().__init__(
            input_root=input_root,
            split=split,
            task_filter=task_filter,
            max_episodes=max_episodes,
            video_backend=video_backend,
            camera_mapping=camera_mapping,
            return_video_path=return_video_path,
        )
        self.effective_camera_mapping = dict(DEFAULT_CAMERA_MAPPING)
        self.effective_camera_mapping.update(self.camera_mapping)
        self.left_gripper_spec = None
        self.right_gripper_spec = None

    def _build_index(self) -> list[EpisodeRecord]:
        episode_paths = sorted(self.input_root.glob("*.hdf5"), key=_episode_sort_key)
        records: list[EpisodeRecord] = []
        for episode_path in episode_paths:
            if not self._match_task_filters(self.input_root.name, episode_path.name):
                continue
            records.append(
                EpisodeRecord(
                    episode_id=episode_path.stem,
                    payload={"episode_path": str(episode_path)},
                )
            )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        episode_path = Path(record.payload["episode_path"])
        with h5py.File(episode_path, "r") as handle:
            if self.left_gripper_spec is None:
                left_state = load_array(
                    handle, "poses_dict/astribot_gripper_left", dtype=np.float32
                ).reshape(-1)
                left_command = load_array(
                    handle, "command_poses_dict/astribot_gripper_left", dtype=np.float32
                ).reshape(-1)
                left_merged = np.concatenate([left_state, left_command], axis=0)
                self.left_gripper_spec = GripperNormSpec(
                    minimum=float(np.min(left_merged)),
                    maximum=float(np.max(left_merged)),
                    larger_is_closed=True,
                    constant_output=1.0 if float(np.max(left_merged) - np.min(left_merged)) < 1e-6 else None,
                )
            if self.right_gripper_spec is None:
                right_state = load_array(
                    handle, "poses_dict/astribot_gripper_right", dtype=np.float32
                ).reshape(-1)
                right_command = load_array(
                    handle, "command_poses_dict/astribot_gripper_right", dtype=np.float32
                ).reshape(-1)
                right_merged = np.concatenate([right_state, right_command], axis=0)
                self.right_gripper_spec = GripperNormSpec(
                    minimum=float(np.min(right_merged)),
                    maximum=float(np.max(right_merged)),
                    larger_is_closed=True,
                    constant_output=1.0 if float(np.max(right_merged) - np.min(right_merged)) < 1e-6 else None,
                )
            video_frames = {}
            for output_key, source_name in self.effective_camera_mapping.items():
                rgb_key = f"images_dict/{source_name}/rgb"
                size_key = f"images_dict/{source_name}/rgb_size"
                if rgb_key not in handle or size_key not in handle:
                    continue
                if self.return_video_path:
                    video_frames[output_key] = format_virtual_video_path(
                        episode_path, rgb_key
                    )
                else:
                    video_frames[output_key] = decode_concat_jpeg_stream(
                        handle[rgb_key],
                        load_array(handle, size_key, dtype=np.int64),
                    )

            left_pose = load_array(
                handle, "command_poses_dict/astribot_arm_left", dtype=np.float32
            ).reshape(-1, 7)
            right_pose = load_array(
                handle, "command_poses_dict/astribot_arm_right", dtype=np.float32
            ).reshape(-1, 7)
            left_gripper = normalize_gripper_with_spec(
                load_array(
                    handle, "command_poses_dict/astribot_gripper_left", dtype=np.float32
                ).reshape(-1),
                self.left_gripper_spec,
            )
            right_gripper = normalize_gripper_with_spec(
                load_array(
                    handle, "command_poses_dict/astribot_gripper_right", dtype=np.float32
                ).reshape(-1),
                self.right_gripper_spec,
            )
            raw_absolute_actions = np.concatenate(
                [
                    left_pose[:, :3].astype(np.float32),
                    quaternion_xyzw_to_wxyz(left_pose[:, 3:7]),
                    left_gripper,
                    right_pose[:, :3].astype(np.float32),
                    quaternion_xyzw_to_wxyz(right_pose[:, 3:7]),
                    right_gripper,
                ],
                axis=1,
            ).astype(np.float32)

            task_name = str(handle.attrs.get("task_name", "") or "").strip()
            instruction = task_name or self.input_root.name.replace("_", " ").strip()

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
