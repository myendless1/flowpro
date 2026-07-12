from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    decode_frame_dir,
    extract_nested_array,
    load_bridge_pickle,
    normalize_gripper,
    sort_paths_natural,
    transform_matrices_to_xyz_quat_wxyz,
)


class BridgeRawDataset(BaseRawCurationDataset):
    def _build_index(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        for root, dirs, files in os.walk(self.input_root):
            dirs.sort()
            if "obs_dict.pkl" not in files or "policy_out.pkl" not in files:
                continue
            traj_dir = Path(root)
            image_dirs = [
                child
                for child in sort_paths_natural(traj_dir.iterdir())
                if child.is_dir()
                and child.name.startswith("images")
                and any(child.glob("im_*.jpg"))
            ]
            if not image_dirs:
                continue

            instruction = ""
            instruction_path = traj_dir / "lang.txt"
            if instruction_path.exists():
                instruction = instruction_path.read_text(encoding="utf-8").strip()

            for image_dir in image_dirs:
                rel_id = f"{traj_dir.relative_to(self.input_root)}::{image_dir.name}"
                if not self._match_task_filters(rel_id, instruction):
                    continue
                records.append(
                    EpisodeRecord(
                        episode_id=rel_id,
                        payload={
                            "traj_dir": str(traj_dir),
                            "image_dir": str(image_dir),
                            "instruction": instruction,
                        },
                    )
                )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        traj_dir = Path(record.payload["traj_dir"])
        image_dir = Path(record.payload["image_dir"])
        instruction = str(record.payload["instruction"])

        obs_payload = load_bridge_pickle(traj_dir / "obs_dict.pkl")
        policy_payload = load_bridge_pickle(traj_dir / "policy_out.pkl")

        raw_states = extract_nested_array(obs_payload, "state")
        raw_actions = extract_nested_array(policy_payload, "actions")
        eef_transforms = extract_nested_array(obs_payload, "eef_transform").reshape(-1, 4, 4)
        new_robot_transforms = np.asarray(
            [item["new_robot_transform"] for item in policy_payload],
            dtype=np.float32,
        ).reshape(-1, 4, 4)

        length = min(
            int(raw_states.shape[0]),
            int(raw_actions.shape[0]),
            int(eef_transforms.shape[0]),
            int(new_robot_transforms.shape[0]),
        )
        if length <= 0:
            raise ValueError(f"Empty Bridge episode: {record.episode_id}")

        raw_absolute_actions = np.concatenate(
            [
                transform_matrices_to_xyz_quat_wxyz(new_robot_transforms[:length]),
                normalize_gripper(raw_actions[:length, -1], invert=False),
            ],
            axis=1,
        ).astype(np.float32)
        if self.return_video_path:
            video_frames = {"main": str(image_dir)}
        else:
            video_frames = {"main": decode_frame_dir(image_dir)}

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
