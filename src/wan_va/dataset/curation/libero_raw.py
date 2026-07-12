from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    format_virtual_video_path,
    quaternion_xyzw_to_wxyz,
    shift_next,
    sort_paths_natural,
)


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler_xyz, dtype=np.float32).reshape(-1, 3)
    half = euler * 0.5
    cx = np.cos(half[:, 0])
    sx = np.sin(half[:, 0])
    cy = np.cos(half[:, 1])
    sy = np.sin(half[:, 1])
    cz = np.cos(half[:, 2])
    sz = np.sin(half[:, 2])
    quat = np.stack(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        axis=1,
    ).astype(np.float32)
    return quaternion_xyzw_to_wxyz(quat)


def _build_state_array(obs_group: h5py.Group) -> np.ndarray:
    ee_pos = np.asarray(obs_group["ee_pos"], dtype=np.float32).reshape(-1, 3)
    ee_ori = np.asarray(obs_group["ee_ori"], dtype=np.float32).reshape(-1, 3)
    gripper_raw = np.asarray(obs_group["gripper_states"], dtype=np.float32)
    if gripper_raw.ndim == 2:
        gripper_raw = gripper_raw[:, 0]
    gripper = (gripper_raw.reshape(-1, 1) > 0.0).astype(np.float32)
    quat = _euler_xyz_to_quat_wxyz(ee_ori)
    return np.concatenate([ee_pos, quat, gripper], axis=1).astype(np.float32)


class LiberoRawDataset(BaseRawCurationDataset):
    def _iter_hdf5_files(self) -> list[Path]:
        direct_files = sort_paths_natural(self.input_root.glob("*.hdf5"))
        if direct_files:
            return direct_files

        split_tokens = set(self.splits)
        out = []
        for path in sorted(self.input_root.rglob("*.hdf5"), key=lambda p: str(p)):
            if split_tokens and not split_tokens.intersection(path.parts):
                continue
            out.append(path)
        return out

    def _build_index(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        for hdf5_path in self._iter_hdf5_files():
            task_text = hdf5_path.stem.replace("_demo", "").replace("_", " ").strip()
            if not self._match_task_filters(task_text, hdf5_path.name, str(hdf5_path.parent)):
                continue
            with h5py.File(hdf5_path, "r") as handle:
                if "data" not in handle:
                    continue
                for demo_name in sorted(handle["data"].keys()):
                    records.append(
                        EpisodeRecord(
                            episode_id=f"{hdf5_path.name}::{demo_name}",
                            payload={
                                "hdf5_path": str(hdf5_path),
                                "demo_name": str(demo_name),
                                "instruction": task_text,
                            },
                        )
                    )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        hdf5_path = Path(record.payload["hdf5_path"])
        demo_name = str(record.payload["demo_name"])
        instruction = str(record.payload["instruction"])

        with h5py.File(hdf5_path, "r") as handle:
            demo_group = handle["data"][demo_name]
            obs = demo_group["obs"]
            states = _build_state_array(obs)
            raw_absolute_actions = shift_next(states)
            if self.return_video_path:
                video_frames = {
                    "main": format_virtual_video_path(
                        hdf5_path, f"data/{demo_name}/obs/agentview_rgb"
                    ),
                    "left_wrist": format_virtual_video_path(
                        hdf5_path, f"data/{demo_name}/obs/eye_in_hand_rgb"
                    ),
                }
            else:
                video_frames = {
                    "main": np.asarray(obs["agentview_rgb"], dtype=np.uint8),
                    "left_wrist": np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8),
                }

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
