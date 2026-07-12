from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .base import (
    BaseRawCurationDataset,
    EpisodeRecord,
    decode_video_mp4,
    load_json,
    natural_sort_key,
    resolve_lerobot_video_chunk_dir,
    resolve_lerobot_videos_root,
)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _first_task_text(episode_row: dict[str, Any]) -> str:
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0]).strip()
    task = episode_row.get("task")
    if task is not None:
        return str(task).strip()
    return ""


def _array_from_column(table_dict: dict[str, Any], key: str) -> np.ndarray:
    if key not in table_dict:
        raise KeyError(f"Missing parquet column: {key}")
    arr = np.asarray(table_dict[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32, copy=False)


def _maybe_array_from_column(table_dict: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in table_dict:
        return None
    return _array_from_column(table_dict, key)


class LeRobotRepoRawDataset(BaseRawCurationDataset):
    repo_kind = "lerobot_repo_raw"
    default_camera_sources: dict[str, str] = {}

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
        self.effective_camera_mapping = dict(self.default_camera_sources)
        self.effective_camera_mapping.update(self.camera_mapping)

    def _discover_repo_roots(self) -> list[Path]:
        root = self.input_root
        direct_meta = root / "meta" / "episodes.jsonl"
        direct_data = root / "data"
        try:
            direct_videos = resolve_lerobot_videos_root(root, must_exist=True)
        except FileNotFoundError:
            direct_videos = None
        if direct_meta.exists() and direct_data.exists() and direct_videos is not None:
            return [root]
        repo_roots: set[Path] = set()
        for path in root.rglob("meta/episodes.jsonl"):
            repo_root = path.parent.parent
            if not repo_root.is_dir():
                continue
            try:
                resolve_lerobot_videos_root(repo_root, must_exist=True)
            except FileNotFoundError:
                continue
            repo_roots.add(repo_root)
        return sorted(repo_roots, key=lambda p: natural_sort_key(str(p)))

    def _build_index(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        repo_roots = self._discover_repo_roots()
        for repo_root in repo_roots:
            records.extend(self._build_repo_records(repo_root))
        return records

    def _build_repo_records(self, repo_root: Path) -> list[EpisodeRecord]:
        episodes_path = repo_root / "meta" / "episodes.jsonl"
        info_path = repo_root / "meta" / "info.json"
        if not episodes_path.exists() or not info_path.exists():
            return []

        info = load_json(info_path)
        split_range = self._resolve_split_range(info)
        records: list[EpisodeRecord] = []
        for episode_row in _load_jsonl_rows(episodes_path):
            episode_index = int(episode_row["episode_index"])
            if split_range is not None and not (split_range[0] <= episode_index < split_range[1]):
                continue
            instruction = self._get_instruction(repo_root, episode_row)
            repo_task_text = repo_root.name.replace("_", " ")
            if not self._match_task_filters(repo_root.name, repo_task_text, instruction):
                continue

            record_id = f"{repo_root.name}:episode_{episode_index:06d}"
            records.append(
                EpisodeRecord(
                    episode_id=record_id,
                    payload={
                        "repo_root": str(repo_root),
                        "episode_index": episode_index,
                        "instruction": instruction,
                    },
                )
            )
        return records

    def _resolve_split_range(self, info: dict[str, Any]) -> tuple[int, int] | None:
        splits = info.get("splits")
        if not isinstance(splits, dict) or not splits:
            return None
        split_names = self.splits or tuple(splits.keys())
        selected = [name for name in split_names if name in splits]
        if not selected:
            return None
        if len(selected) != 1:
            raise ValueError(
                f"{self.repo_kind} only supports one split at a time, got {selected}"
            )
        spec = str(splits[selected[0]])
        start_text, end_text = spec.split(":", 1)
        return (int(start_text), int(end_text))

    def _get_instruction(self, repo_root: Path, episode_row: dict[str, Any]) -> str:
        instruction = _first_task_text(episode_row)
        if instruction:
            return instruction
        tasks_path = repo_root / "meta" / "tasks.jsonl"
        task_index = episode_row.get("task_index")
        if tasks_path.exists() and task_index is not None:
            wanted = int(task_index)
            for row in _load_jsonl_rows(tasks_path):
                if int(row.get("task_index", -1)) == wanted:
                    task = str(row.get("task", "")).strip()
                    if task:
                        return task
        return repo_root.name.replace("_", " ").strip()

    def _load_record(self, record: EpisodeRecord) -> dict[str, object]:
        repo_root = Path(record.payload["repo_root"])
        episode_index = int(record.payload["episode_index"])
        parquet_path = self._get_episode_parquet_path(repo_root, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        table = pq.read_table(parquet_path)
        table_dict = table.to_pydict()
        raw_absolute_actions = self._build_raw_absolute_actions(
            repo_root=repo_root,
            episode_index=episode_index,
            table_dict=table_dict,
        )
        video_frames = self._load_video_frames(
            repo_root=repo_root,
            episode_index=episode_index,
        )
        return self._finalize_sample(
            video_frames=video_frames,
            raw_absolute_actions=raw_absolute_actions,
            instruction=str(record.payload["instruction"]),
        )

    def _get_episode_parquet_path(self, repo_root: Path, episode_index: int) -> Path:
        episode_chunk = episode_index // 1000
        return (
            repo_root
            / "data"
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

    def _load_video_frames(self, *, repo_root: Path, episode_index: int) -> dict[str, np.ndarray]:
        video_frames: dict[str, np.ndarray] = {}
        episode_chunk = episode_index // 1000
        video_chunk_root = resolve_lerobot_video_chunk_dir(repo_root, episode_chunk, must_exist=True)
        for output_key, source_name in self.effective_camera_mapping.items():
            video_path = (
                video_chunk_root
                / source_name
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.exists():
                continue
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
        raise NotImplementedError


__all__ = [
    "LeRobotRepoRawDataset",
    "_array_from_column",
    "_maybe_array_from_column",
]
