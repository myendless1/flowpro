from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    decode_video_mp4,
    load_json,
    normalize_gripper,
    xyz_euler_xyz_to_xyz_quat_wxyz,
)


class RT1RawDataset(BaseRawCurationDataset):
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
        self.dataset_root, self.annotation_root = self._resolve_roots(Path(input_root))
        super().__init__(
            input_root=self.dataset_root,
            split=split,
            task_filter=task_filter,
            max_episodes=max_episodes,
            video_backend=video_backend,
            camera_mapping=camera_mapping,
            return_video_path=return_video_path,
        )

    @staticmethod
    def _resolve_roots(input_root: Path) -> tuple[Path, Path]:
        root = input_root.expanduser().resolve()
        if (root / "annotation").exists():
            return root, root / "annotation"
        if root.name == "annotation":
            return root.parent, root
        if any(root.glob("*.json")) and root.parent.name == "annotation":
            return root.parents[1], root.parent
        raise FileNotFoundError(
            f"Could not resolve RT1 annotation root from {input_root}"
        )

    def _build_index(self) -> list[EpisodeRecord]:
        split_names = self.splits or ("train",)
        records: list[EpisodeRecord] = []
        for split_name in split_names:
            split_dir = self.annotation_root / split_name
            if not split_dir.exists():
                continue
            for annotation_path in sorted(split_dir.glob("*.json"), key=lambda path: int(path.stem)):
                payload = load_json(annotation_path)
                texts = payload.get("texts", [])
                instruction = str(texts[0]).strip() if texts else str(payload.get("task", "")).strip()
                if not self._match_task_filters(instruction, annotation_path.name):
                    continue
                records.append(
                    EpisodeRecord(
                        episode_id=str(payload.get("episode_id", annotation_path.stem)),
                        payload={
                            "annotation_path": str(annotation_path),
                            "instruction": instruction,
                        },
                    )
                )
        return records

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        annotation_path = Path(record.payload["annotation_path"])
        payload = load_json(annotation_path)
        instruction = str(record.payload["instruction"])

        raw_actions = np.asarray(payload["action"], dtype=np.float32)
        raw_states = np.asarray(payload["state"], dtype=np.float32)
        if raw_actions.ndim != 2 or raw_states.ndim != 2:
            raise ValueError(f"Unexpected RT1 shapes: action={raw_actions.shape}, state={raw_states.shape}")

        if raw_states.shape[0] >= raw_actions.shape[0] + 1:
            length = min(int(raw_actions.shape[0]), int(raw_states.shape[0] - 1))
            action_pose_source = raw_states[1 : length + 1, :6]
        else:
            length = min(int(raw_actions.shape[0]), int(raw_states.shape[0]))
            action_pose_source = raw_states[:length, :6]
        if length <= 0:
            raise ValueError(f"Empty RT1 episode: {record.episode_id}")

        continuous_gripper_state = payload.get("continuous_gripper_state")
        if continuous_gripper_state is not None:
            gripper = np.asarray(continuous_gripper_state, dtype=np.float32).reshape(-1)
            if gripper.shape[0] >= length + 1 and raw_states.shape[0] >= raw_actions.shape[0] + 1:
                action_gripper = normalize_gripper(gripper[1 : length + 1], invert=True)
            elif gripper.shape[0] >= length:
                action_gripper = normalize_gripper(gripper[:length], invert=True)
            else:
                action_gripper = normalize_gripper(raw_actions[:length, -1], invert=True)
        else:
            if raw_states.shape[0] >= raw_actions.shape[0] + 1 and raw_states.shape[0] >= length + 1:
                action_gripper = normalize_gripper(raw_states[1 : length + 1, -1], invert=True)
            else:
                action_gripper = normalize_gripper(raw_actions[:length, -1], invert=True)

        raw_absolute_actions = np.concatenate(
            [xyz_euler_xyz_to_xyz_quat_wxyz(action_pose_source), action_gripper],
            axis=1,
        ).astype(np.float32)

        videos = payload.get("videos", [])
        if not videos or "video_path" not in videos[0]:
            raise KeyError(f"Missing RT1 video path for {record.episode_id}")
        video_path = self.dataset_root / videos[0]["video_path"]
        if self.return_video_path:
            video_frames = {"main": str(video_path)}
        else:
            video_frames = {"main": decode_video_mp4(video_path, backend=self.video_backend)}

        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=instruction,
        )
