from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    decode_video_mp4,
    find_hdf5_dataset_by_candidates,
    load_json,
    normalize_gripper,
    resolve_droid_video_paths,
    walk_hdf5_datasets,
    xyz_euler_xyz_to_xyz_quat_wxyz,
    xyz_rotvec_to_xyz_quat_wxyz,
)


STATE_CARTESIAN_CANDIDATES = [
    "observation/robot_state/cartesian_position",
    "observation/cartesian_position",
    "robot_state/cartesian_position",
    "cartesian_position",
]
STATE_GRIPPER_CANDIDATES = [
    "observation/robot_state/gripper_position",
    "observation/gripper_position",
    "robot_state/gripper_position",
    "gripper_position",
]
ACTION_CARTESIAN_CANDIDATES = [
    "action/cartesian_position",
    "action/target_cartesian_position",
    "action/cartesian_velocity",
    "target_cartesian_position",
    "cartesian_velocity",
]
ACTION_GRIPPER_CANDIDATES = [
    "action/gripper_position",
    "action/target_gripper_position",
    "action/gripper_velocity",
    "target_gripper_position",
    "gripper_velocity",
]


class DroidRawDataset(BaseRawCurationDataset):
    rotation_representation = "rotvec"

    def _build_index(self) -> list[EpisodeRecord]:
        splits = set(self.splits or ("success",))
        records: list[EpisodeRecord] = []
        for trajectory_path in sorted(self.input_root.rglob("trajectory.h5")):
            episode_dir = trajectory_path.parent
            if splits and not splits.intersection(episode_dir.parts):
                continue

            metadata_path = next(iter(sorted(episode_dir.glob("metadata_*.json"))), None)
            instruction = ""
            if metadata_path is not None:
                try:
                    instruction = str(load_json(metadata_path).get("current_task", "")).strip()
                except Exception:
                    instruction = ""

            rel_id = str(episode_dir.relative_to(self.input_root))
            if not self._match_task_filters(rel_id, instruction):
                continue

            records.append(
                EpisodeRecord(
                    episode_id=rel_id,
                    payload={
                        "episode_dir": str(episode_dir),
                        "metadata_path": str(metadata_path) if metadata_path else "",
                    },
                )
            )
        return records

    def _convert_pose_array(self, cartesian: np.ndarray) -> np.ndarray:
        if self.rotation_representation == "rotvec":
            return xyz_rotvec_to_xyz_quat_wxyz(cartesian)
        return xyz_euler_xyz_to_xyz_quat_wxyz(cartesian)

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        episode_dir = Path(record.payload["episode_dir"])
        metadata_path = Path(record.payload["metadata_path"]) if record.payload["metadata_path"] else None
        metadata = load_json(metadata_path) if metadata_path and metadata_path.exists() else None
        instruction = str((metadata or {}).get("current_task", "")).strip()

        with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
            dataset_map = walk_hdf5_datasets(handle)
            _, state_cartesian = find_hdf5_dataset_by_candidates(
                dataset_map, STATE_CARTESIAN_CANDIDATES
            )
            _, state_gripper = find_hdf5_dataset_by_candidates(
                dataset_map, STATE_GRIPPER_CANDIDATES
            )
            _, action_cartesian = find_hdf5_dataset_by_candidates(
                dataset_map, ACTION_CARTESIAN_CANDIDATES
            )
            _, action_gripper = find_hdf5_dataset_by_candidates(
                dataset_map, ACTION_GRIPPER_CANDIDATES
            )

            state_cartesian = np.asarray(state_cartesian, dtype=np.float32).reshape(-1, 6)
            state_gripper = normalize_gripper(
                np.asarray(state_gripper, dtype=np.float32).reshape(-1),
                invert=True,
            )
            action_cartesian = np.asarray(action_cartesian, dtype=np.float32).reshape(-1, 6)
            action_gripper = normalize_gripper(
                np.asarray(action_gripper, dtype=np.float32).reshape(-1),
                invert=True,
            )
            length = min(
                int(state_cartesian.shape[0]),
                int(state_gripper.shape[0]),
                int(action_cartesian.shape[0]),
                int(action_gripper.shape[0]),
            )
            if length <= 0:
                raise ValueError(f"Empty DROID episode: {record.episode_id}")

            raw_absolute_actions = np.concatenate(
                [
                    self._convert_pose_array(action_cartesian[:length]),
                    action_gripper[:length],
                ],
                axis=1,
            ).astype(np.float32)

            resolved_paths = resolve_droid_video_paths(
                dataset_root=self.input_root,
                episode_dir=episode_dir,
                metadata=metadata,
                dataset_map=dataset_map,
            )

        video_frames = {}
        if "wrist" in resolved_paths:
            if self.return_video_path:
                video_frames["left_wrist"] = str(resolved_paths["wrist"])
            else:
                video_frames["left_wrist"] = decode_video_mp4(
                    resolved_paths["wrist"], backend=self.video_backend
                )
        if "left" in resolved_paths:
            if self.return_video_path:
                video_frames["left_main"] = str(resolved_paths["left"])
            else:
                video_frames["left_main"] = decode_video_mp4(
                    resolved_paths["left"], backend=self.video_backend
                )
        if "right" in resolved_paths:
            if self.return_video_path:
                video_frames["right_main"] = str(resolved_paths["right"])
            else:
                video_frames["right_main"] = decode_video_mp4(
                    resolved_paths["right"], backend=self.video_backend
                )

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
