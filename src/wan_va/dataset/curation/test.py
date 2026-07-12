from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset.curation import (
    AstribotRawDataset,
    BridgeRawDataset,
    DroidRawDataset,
    LiberoRawDataset,
    RoboCoinRawDataset,
    RoboMindRawDataset,
    RT1RawDataset,
    SonglingRawDataset,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cls: type
    input_root: str
    kwargs: dict[str, Any]


DEFAULT_DATASET_SPECS = (
    DatasetSpec(
        name="astribot",
        cls=AstribotRawDataset,
        input_root="/media/damoxing/ckp/astribot_data/myendless/black_white_plate",
        kwargs={},
    ),
    DatasetSpec(
        name="bridge",
        cls=BridgeRawDataset,
        input_root="/media/damoxing/datasets/bridge-data/raw",
        kwargs={},
    ),
    DatasetSpec(
        name="droid",
        cls=DroidRawDataset,
        input_root="/media/damoxing/datasets/droid_raw/1.0.1/AUTOLab",
        kwargs={},
    ),
    DatasetSpec(
        name="libero",
        cls=LiberoRawDataset,
        input_root="/media/damoxing/datasets/libero/libero_spatial",
        kwargs={},
    ),
    DatasetSpec(
        name="robocoin",
        cls=RoboCoinRawDataset,
        input_root="/media/damoxing/datasets/RoboCOIN",
        kwargs={},
    ),
    DatasetSpec(
        name="robomind",
        cls=RoboMindRawDataset,
        input_root="/media/damoxing/datasets/RoboMIND2.0_LeRobot",
        kwargs={},
    ),
    DatasetSpec(
        name="rt1",
        cls=RT1RawDataset,
        input_root="/media/damoxing/datasets/rt1/opensource_robotdata/rt1",
        kwargs={"split": "train"},
    ),
    DatasetSpec(
        name="songling",
        cls=SonglingRawDataset,
        input_root="/media/damoxing/datasets/RoboTwin2_0_processed/songling_tasks/hdf5_from_mcap/processed",
        kwargs={},
    ),
)


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _build_video_strip(
    video_frames: dict[str, np.ndarray],
    camera_keys: list[str],
    action_step: int,
    num_action_steps: int,
    target_width: int,
) -> np.ndarray:
    tiles = []
    target_tile_h = None

    for camera_key in camera_keys:
        frames = np.asarray(video_frames[camera_key])
        num_video_frames = int(frames.shape[0])
        if num_video_frames <= 0:
            continue
        if num_action_steps <= 1:
            video_step = 0
        else:
            video_step = int(
                round(action_step * max(num_video_frames - 1, 0) / max(num_action_steps - 1, 1))
            )
        frame_rgb = frames[video_step]
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])

        if target_tile_h is None:
            target_tile_h = frame_bgr.shape[0]
        elif frame_bgr.shape[0] != target_tile_h:
            resized_w = max(
                1,
                int(round(frame_bgr.shape[1] * target_tile_h / frame_bgr.shape[0])),
            )
            frame_bgr = cv2.resize(
                frame_bgr,
                (resized_w, target_tile_h),
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.putText(
            frame_bgr,
            f"{camera_key}  f={video_step + 1}/{num_video_frames}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(frame_bgr)

    if not tiles:
        raise ValueError("No camera tiles available for visualization")

    strip = cv2.hconcat(tiles)
    if strip.shape[1] != target_width:
        resized_h = max(1, int(round(strip.shape[0] * target_width / strip.shape[1])))
        strip = cv2.resize(strip, (target_width, resized_h), interpolation=cv2.INTER_LINEAR)
    return strip


def _compute_action_ranges(actions_t: np.ndarray) -> list[tuple[float, float]]:
    ranges = []
    for dim_idx in range(actions_t.shape[0]):
        values = actions_t[dim_idx]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ranges.append((-1.0, 1.0))
            continue
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if abs(hi - lo) < 1e-6:
            margin = max(abs(lo) * 0.1, 1.0)
            ranges.append((lo - margin, hi + margin))
            continue
        margin = (hi - lo) * 0.1
        ranges.append((lo - margin, hi + margin))
    return ranges


def _draw_actions_panel(actions: np.ndarray, cursor_step: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T,D], got {actions.shape}")

    actions_t = actions.T
    action_dim, num_action_steps = actions_t.shape
    action_ranges = _compute_action_ranges(actions_t)

    num_cols = min(5, max(1, action_dim))
    num_rows = math.ceil(action_dim / num_cols)
    cell_w = 220
    cell_h = 110
    pad = 10
    title_h = 28
    width = num_cols * cell_w + (num_cols + 1) * pad
    height = num_rows * cell_h + (num_rows + 1) * pad + title_h
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Raw absolute actions",
        (pad, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    colors = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (218, 165, 32),
        (186, 85, 211),
        (70, 130, 180),
    ]

    def _x_of(step: int, left: int, plot_w: int) -> int:
        if num_action_steps <= 1:
            return left
        return left + int(round(step * (plot_w - 1) / (num_action_steps - 1)))

    def _y_of(value: float, top: int, plot_h: int, y_min: float, y_max: float) -> int:
        clipped = float(np.clip(value, y_min, y_max))
        ratio = (clipped - y_min) / max(y_max - y_min, 1e-6)
        return top + plot_h - 1 - int(round(ratio * (plot_h - 1)))

    for dim_idx in range(action_dim):
        row = dim_idx // num_cols
        col = dim_idx % num_cols
        cell_x = pad + col * (cell_w + pad)
        cell_y = title_h + pad + row * (cell_h + pad)
        plot_x = cell_x + 30
        plot_y = cell_y + 16
        plot_w = cell_w - 40
        plot_h = cell_h - 28
        color = colors[dim_idx % len(colors)]
        y_min, y_max = action_ranges[dim_idx]

        cv2.rectangle(
            canvas,
            (cell_x, cell_y),
            (cell_x + cell_w, cell_y + cell_h),
            (210, 210, 210),
            1,
        )
        cv2.putText(
            canvas,
            f"a{dim_idx:02d}",
            (cell_x + 6, cell_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"[{y_min:.2f}, {y_max:.2f}]",
            (cell_x + 60, cell_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

        if y_min <= 0.0 <= y_max:
            zero_y = _y_of(0.0, plot_y, plot_h, y_min, y_max)
            cv2.line(
                canvas,
                (plot_x, zero_y),
                (plot_x + plot_w, zero_y),
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            canvas,
            (plot_x, plot_y),
            (plot_x, plot_y + plot_h),
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cursor_x = _x_of(min(cursor_step, num_action_steps - 1), plot_x, plot_w)
        cv2.line(
            canvas,
            (cursor_x, plot_y),
            (cursor_x, plot_y + plot_h),
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        for step in range(num_action_steps - 1):
            p1 = (
                _x_of(step, plot_x, plot_w),
                _y_of(actions_t[dim_idx, step], plot_y, plot_h, y_min, y_max),
            )
            p2 = (
                _x_of(step + 1, plot_x, plot_w),
                _y_of(actions_t[dim_idx, step + 1], plot_y, plot_h, y_min, y_max),
            )
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        current_step = min(cursor_step, num_action_steps - 1)
        current_point = (
            _x_of(current_step, plot_x, plot_w),
            _y_of(actions_t[dim_idx, current_step], plot_y, plot_h, y_min, y_max),
        )
        cv2.circle(canvas, current_point, 3, color, -1, cv2.LINE_AA)

    return canvas


def _make_visualization_frames(
    *,
    dataset_name: str,
    sample: dict[str, Any],
    episode_id: str,
    sample_index: int,
) -> list[np.ndarray]:
    if sample["video_frames"] and isinstance(next(iter(sample["video_frames"].values())), str):
        raise ValueError("Visualization does not support return_video_path=True samples")
    video_frames = {
        key: np.asarray(value)
        for key, value in sample["video_frames"].items()
    }
    camera_keys = list(video_frames.keys())
    actions = np.asarray(sample["raw_absolute_actions"], dtype=np.float32)
    instruction = str(sample["instruction"])

    if actions.ndim != 2 or actions.shape[0] <= 0:
        raise ValueError(f"Invalid raw_absolute_actions shape: {actions.shape}")

    num_action_steps = int(actions.shape[0])
    action_panel = _draw_actions_panel(actions, cursor_step=0)
    frames = []

    for action_step in range(num_action_steps):
        top_strip = _build_video_strip(
            video_frames=video_frames,
            camera_keys=camera_keys,
            action_step=action_step,
            num_action_steps=num_action_steps,
            target_width=action_panel.shape[1],
        )
        bottom_panel = _draw_actions_panel(actions, cursor_step=action_step)
        header_h = 78
        header = np.full((header_h, action_panel.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            header,
            f"dataset={dataset_name} sample={sample_index} episode={episode_id}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"action_step={action_step + 1}/{num_action_steps} action_dim={actions.shape[1]} cameras={','.join(camera_keys)}",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"instruction={instruction[:160]}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        frames.append(np.concatenate([header, top_strip, bottom_panel], axis=0))

    return frames


def _write_video(output_path: Path, frames: list[np.ndarray], fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame[:, :, ::-1])


def _resolve_specs(dataset_names: list[str]) -> list[DatasetSpec]:
    if not dataset_names:
        return list(DEFAULT_DATASET_SPECS)
    selected = set(dataset_names)
    return [spec for spec in DEFAULT_DATASET_SPECS if spec.name in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize each raw curation dataset, load one sample, and save a full-episode visualization."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated subset from: astribot,bridge,droid,libero,robocoin,robomind,rt1,songling",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--video-backend", type=str, default="opencv")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "debug" / "curation_dataset_videos",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Default writes to <output-dir>/summary.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_specs = _resolve_specs(_parse_csv(args.datasets))
    if not selected_specs:
        raise ValueError("No datasets selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or (args.output_dir / "summary.json")
    summary_rows = []

    for spec in selected_specs:
        input_root = Path(spec.input_root)
        row: dict[str, Any] = {
            "dataset": spec.name,
            "input_root": str(input_root),
            "status": "pending",
        }
        try:
            if not input_root.exists():
                raise FileNotFoundError(f"input_root does not exist: {input_root}")

            print(f"[INFO] init dataset={spec.name} root={input_root}")
            dataset = spec.cls(
                input_root=str(input_root),
                max_episodes=int(args.max_episodes),
                video_backend=str(args.video_backend),
                **spec.kwargs,
            )
            if len(dataset) <= 0:
                raise RuntimeError(f"dataset={spec.name} is empty")

            sample_index = min(max(int(args.sample_index), 0), len(dataset) - 1)
            print(
                f"[INFO] load sample dataset={spec.name} sample_index={sample_index} total={len(dataset)}"
            )
            sample = dataset[sample_index]
            episode_id = dataset.records[sample_index].episode_id
            frames = _make_visualization_frames(
                dataset_name=spec.name,
                sample=sample,
                episode_id=episode_id,
                sample_index=sample_index,
            )
            output_path = (
                args.output_dir
                / f"{spec.name}_{sample_index:03d}_{_sanitize_name(episode_id)}.mp4"
            )
            _write_video(output_path, frames, fps=int(args.video_fps))

            video_shapes = {
                key: list(np.asarray(value).shape)
                for key, value in sample["video_frames"].items()
            }
            row.update(
                {
                    "status": "ok",
                    "sample_index": sample_index,
                    "episode_id": episode_id,
                    "instruction": str(sample["instruction"]),
                    "raw_absolute_actions_shape": list(np.asarray(sample["raw_absolute_actions"]).shape),
                    "video_shapes": video_shapes,
                    "output_video": str(output_path),
                }
            )
            print(f"[OK] dataset={spec.name} output={output_path}")
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] dataset={spec.name} error={exc}")
            if args.strict:
                summary_rows.append(row)
                summary_path.write_text(
                    json.dumps(summary_rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                raise

        summary_rows.append(row)

    summary_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
