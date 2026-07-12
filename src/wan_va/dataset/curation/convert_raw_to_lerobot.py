#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from contextlib import contextmanager
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
try:
    import pyarrow as pa
except Exception:
    pa = None
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset.curation import (  # noqa: E402
    AstribotRawDataset,
    BridgeRawDataset,
    DroidRawDataset,
    LiberoRawDataset,
    RoboCoinRawDataset,
    RoboMindRawDataset,
    RobotWinRawDataset,
    RT1RawDataset,
    SonglingRawDataset,
)
from wan_va.dataset.curation.robotwin_raw import ROBOTWIN_DEFAULT_REPO_DIR_NAMES  # noqa: E402
from wan_va.dataset.curation.base import (  # noqa: E402
    extract_nested_array,
    find_hdf5_dataset_by_candidates,
    format_virtual_video_path,
    get_lerobot_videos_dir_name,
    load_bridge_pickle,
    load_json,
    resolve_lerobot_videos_root,
    xyz_euler_xyz_to_xyz_quat_wxyz,
    quaternion_xyzw_to_wxyz,
    resolve_droid_video_paths,
    standardize_quaternion_xyzw,
    standardize_quaternion_wxyz,
    walk_hdf5_datasets,
)
FRAME_SAMPLE_STRIDE = 4
OUTPUT_ROOT_DEFAULT = Path("/media/damoxing/datasets/vae4d/lerobot-f1")

RT1_DEFAULT_SPLIT = "train"
ASTRIBOT_STATE_ACTION_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
]
DROID_TIMESTAMP_SECONDS_CANDIDATES = [
    "observation/timestamp/robot_state/robot_timestamp_seconds",
    "observation/timestamp/robot_timestamp_seconds",
    "timestamp/robot_timestamp_seconds",
    "robot_timestamp_seconds",
]
DROID_TIMESTAMP_NANOS_CANDIDATES = [
    "observation/timestamp/robot_state/robot_timestamp_nanos",
    "observation/timestamp/robot_timestamp_nanos",
    "timestamp/robot_timestamp_nanos",
    "robot_timestamp_nanos",
]
PENDING_WRIST_TASKS_FILENAME = "pending_wrist_exports.json"
VIDEOS_DIR_NAME_RE = re.compile(r"^videos_(\d+)x(\d+)_(\d+)x(\d+)$")


def _load_writer_symbols() -> dict[str, Any]:
    candidates = [
        REPO_ROOT.parent
        / "md4d"
        / "third_parties"
        / "lingbot-va"
        / "wan_wa"
        / "dataset"
        / "lerobot_v21_writer.py",
        Path(
            "/media/damoxing/fileset/md4d/third_parties/lingbot-va/wan_wa/dataset/lerobot_v21_writer.py"
        ),
    ]
    writer_path = next((path for path in candidates if path.exists()), None)
    if writer_path is None:
        raise FileNotFoundError(
            "Failed to locate lingbot-va lerobot_v21_writer.py. "
            f"Tried: {[str(path) for path in candidates]}"
        )
    module_name = "curation_lerobot_v21_writer"
    spec = importlib.util.spec_from_file_location(module_name, writer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load writer module from {writer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return {
        "dump_json": module.dump_json,
        "write_episode_parquet": module.write_episode_parquet,
        "ensure_dir": module.ensure_dir,
        "rebuild_lerobot_v21_meta": module.rebuild_lerobot_v21_meta,
    }


_WRITER = _load_writer_symbols()
dump_json = _WRITER["dump_json"]
write_episode_parquet = _WRITER["write_episode_parquet"]
ensure_dir = _WRITER["ensure_dir"]
rebuild_lerobot_v21_meta = _WRITER["rebuild_lerobot_v21_meta"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cls: type | None
    input_root: str
    kwargs: dict[str, Any]
    output_name: str
    source_suffix: str
    robot_type: str
    default_fps: float
    mode: str = "generic"


@dataclass(frozen=True)
class AstribotCameraSpec:
    camera_key: str
    source_name: str
    output_key: str
    image_height: int
    image_width: int


@dataclass(frozen=True)
class AstribotGripperNormSpec:
    minimum: float
    maximum: float
    larger_is_closed: bool
    constant_output: float | None = None


class AstribotSkipEpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolutionConfig:
    main_height: int
    main_width: int
    wrist_height: int
    wrist_width: int


@dataclass(frozen=True)
class PendingWristVideoTask:
    output_relpath: str
    source: str
    num_frames: int
    fps: float
    camera_key: str
    episode_index: int | None = None


DEFAULT_DATASET_SPECS = (
    DatasetSpec(
        name="astribot",
        cls=AstribotRawDataset,
        input_root="/media/damoxing/ckp/astribot_data/myendless/black_white_plate",
        kwargs={},
        output_name="astribot-lerobot",
        source_suffix="astribot_raw_v1",
        robot_type="astribot_dual_arm",
        default_fps=30.0,
        mode="astribot",
    ),
    DatasetSpec(
        name="bridge",
        cls=BridgeRawDataset,
        input_root="/media/damoxing/datasets/bridge-data/raw",
        kwargs={},
        output_name="bridge-lerobot",
        source_suffix="bridge_raw_v2",
        robot_type="bridge_widowx",
        default_fps=5.0,
    ),
    DatasetSpec(
        name="droid",
        cls=DroidRawDataset,
        input_root="/media/damoxing/datasets/droid_raw/1.0.1/AUTOLab",
        kwargs={},
        output_name="droid-lerobot",
        source_suffix="droid_raw_v1",
        robot_type="droid_franka",
        default_fps=15.0,
    ),
    DatasetSpec(
        name="libero",
        cls=LiberoRawDataset,
        input_root="/media/damoxing/datasets/libero/libero_spatial",
        kwargs={},
        output_name="libero-lerobot",
        source_suffix="libero_raw_v1",
        robot_type="libero_single_arm",
        default_fps=20.0,
    ),
    DatasetSpec(
        name="robocoin",
        cls=RoboCoinRawDataset,
        input_root="/media/damoxing/datasets/RoboCOIN",
        kwargs={},
        output_name="robocoin-lerobot",
        source_suffix="robocoin_raw_v1",
        robot_type="robocoin_mixed",
        default_fps=30.0,
    ),
    DatasetSpec(
        name="robomind",
        cls=RoboMindRawDataset,
        input_root="/media/damoxing/datasets/RoboMIND2.0_LeRobot",
        kwargs={},
        output_name="robomind-lerobot",
        source_suffix="robomind_raw_v1",
        robot_type="robomind_mixed",
        default_fps=20.0,
    ),
    DatasetSpec(
        name="robotwin",
        cls=RobotWinRawDataset,
        input_root="/media/damoxing/datasets/RoboTwin2_0/dataset",
        kwargs={"allowed_repo_dir_names": ROBOTWIN_DEFAULT_REPO_DIR_NAMES},
        output_name="robotwin-lerobot",
        source_suffix="robotwin_raw_v1",
        robot_type="aloha",
        default_fps=15.0,
    ),
    DatasetSpec(
        name="rt1",
        cls=RT1RawDataset,
        input_root="/media/damoxing/datasets/rt1/opensource_robotdata/rt1",
        kwargs={"split": RT1_DEFAULT_SPLIT},
        output_name="rt-1-lerobot",
        source_suffix="rt1_raw",
        robot_type="rt1_single_arm",
        default_fps=3.0,
    ),
    DatasetSpec(
        name="songling",
        cls=SonglingRawDataset,
        input_root="/media/damoxing/datasets/RoboTwin2_0_processed/songling_tasks/hdf5_from_mcap/processed",
        kwargs={},
        output_name="songling-lerobot",
        source_suffix="songling_hdf5_v1",
        robot_type="songling_dual_arm",
        default_fps=30.0,
    ),
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_specs(dataset_names: list[str]) -> list[DatasetSpec]:
    if not dataset_names:
        return list(DEFAULT_DATASET_SPECS)
    selected = set(dataset_names)
    specs = [spec for spec in DEFAULT_DATASET_SPECS if spec.name in selected]
    missing = sorted(selected - {spec.name for spec in DEFAULT_DATASET_SPECS})
    if missing:
        raise ValueError(f"Unknown dataset names: {missing}")
    return specs


def _parse_dataset_path_overrides(value: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(
                "input root overrides must use `dataset=/path` entries, "
                f"got `{item}`"
            )
        dataset_name, input_root = item.split("=", 1)
        dataset_name = dataset_name.strip()
        input_root = input_root.strip()
        if not dataset_name or not input_root:
            raise ValueError(
                "input root overrides must use non-empty `dataset=/path` entries, "
                f"got `{item}`"
            )
        overrides[dataset_name] = input_root
    return overrides


def _parse_dataset_string_overrides(value: str, *, option_name: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(
                f"{option_name} overrides must use `dataset=value` entries, got `{item}`"
            )
        dataset_name, override_value = item.split("=", 1)
        dataset_name = dataset_name.strip()
        override_value = override_value.strip()
        if not dataset_name or not override_value:
            raise ValueError(
                f"{option_name} overrides must use non-empty `dataset=value` entries, got `{item}`"
            )
        overrides[dataset_name] = override_value
    return overrides


def _apply_input_root_overrides(
    specs: Sequence[DatasetSpec],
    overrides: Mapping[str, str],
) -> list[DatasetSpec]:
    if not overrides:
        return list(specs)
    known = {spec.name for spec in DEFAULT_DATASET_SPECS}
    missing = sorted(set(overrides) - known)
    if missing:
        raise ValueError(f"Unknown dataset names in input root overrides: {missing}")
    return [
        replace(spec, input_root=overrides.get(spec.name, spec.input_root))
        for spec in specs
    ]


def _apply_output_name_overrides(
    specs: Sequence[DatasetSpec],
    overrides: Mapping[str, str],
) -> list[DatasetSpec]:
    if not overrides:
        return list(specs)
    known = {spec.name for spec in DEFAULT_DATASET_SPECS}
    missing = sorted(set(overrides) - known)
    if missing:
        raise ValueError(f"Unknown dataset names in output name overrides: {missing}")
    return [
        replace(spec, output_name=overrides.get(spec.name, spec.output_name))
        for spec in specs
    ]


def _sampled_frame_ids(num_frames: int) -> list[int]:
    return list(range(0, max(0, int(num_frames)), FRAME_SAMPLE_STRIDE))


def _camera_to_output_key(camera_key: str) -> str:
    return f"observation.images.cam_{camera_key}"


def _is_wrist_camera(camera_key: str) -> bool:
    return "wrist" in str(camera_key)


def _is_main_camera(camera_key: str) -> bool:
    return not _is_wrist_camera(camera_key)


def _main_resolution(
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None = None,
) -> tuple[int, int]:
    if resolution_config is not None:
        return int(resolution_config.main_height), int(resolution_config.main_width)
    return int(args.main_height), int(args.main_width)


def _wrist_resolution(
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None = None,
) -> tuple[int, int]:
    if resolution_config is not None:
        return int(resolution_config.wrist_height), int(resolution_config.wrist_width)
    return int(args.wrist_height), int(args.wrist_width)


def _videos_dir_name(
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None = None,
) -> str:
    main_height, main_width = _main_resolution(args, resolution_config)
    wrist_height, wrist_width = _wrist_resolution(args, resolution_config)
    return get_lerobot_videos_dir_name(
        main_height=main_height,
        main_width=main_width,
        wrist_height=wrist_height,
        wrist_width=wrist_width,
    )


def _resolution_config_from_args(args: argparse.Namespace) -> ResolutionConfig:
    return ResolutionConfig(
        main_height=int(args.main_height),
        main_width=int(args.main_width),
        wrist_height=int(args.wrist_height),
        wrist_width=int(args.wrist_width),
    )


def _parse_videos_dir_name(value: str) -> ResolutionConfig | None:
    match = VIDEOS_DIR_NAME_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    return ResolutionConfig(
        main_height=int(match.group(1)),
        main_width=int(match.group(2)),
        wrist_height=int(match.group(3)),
        wrist_width=int(match.group(4)),
    )


def _resolve_existing_repo_resolution_config(repo_dir: Path) -> ResolutionConfig | None:
    videos_root = resolve_lerobot_videos_root(repo_dir, must_exist=False)
    if not videos_root.exists():
        return None
    return _parse_videos_dir_name(videos_root.name)


def _camera_resolution(
    args: argparse.Namespace,
    camera_key: str,
    resolution_config: ResolutionConfig | None = None,
) -> tuple[int, int]:
    if _is_wrist_camera(camera_key):
        return _wrist_resolution(args, resolution_config)
    return _main_resolution(args, resolution_config)


def _pending_wrist_tasks_path(repo_dir: Path) -> Path:
    return repo_dir / PENDING_WRIST_TASKS_FILENAME


def _load_pending_wrist_tasks(repo_dir: Path) -> list[PendingWristVideoTask]:
    path = _pending_wrist_tasks_path(repo_dir)
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Invalid pending wrist task file: {path}")
    tasks: list[PendingWristVideoTask] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid pending wrist task row in {path}: {row!r}")
        tasks.append(
            PendingWristVideoTask(
                output_relpath=str(row["output_relpath"]),
                source=str(row["source"]),
                num_frames=int(row["num_frames"]),
                fps=float(row["fps"]),
                camera_key=str(row["camera_key"]),
                episode_index=None
                if row.get("episode_index") is None
                else int(row["episode_index"]),
            )
        )
    return tasks


def _save_pending_wrist_tasks(repo_dir: Path, tasks: Sequence[PendingWristVideoTask]) -> None:
    path = _pending_wrist_tasks_path(repo_dir)
    if not tasks:
        if path.exists():
            path.unlink()
        return
    dump_json(
        path,
        [
            {
                "output_relpath": str(task.output_relpath),
                "source": str(task.source),
                "num_frames": int(task.num_frames),
                "fps": float(task.fps),
                "camera_key": str(task.camera_key),
                "episode_index": task.episode_index,
            }
            for task in tasks
        ],
    )


def _probe_frame_dir_resolution(frame_dir: Path) -> tuple[int, int]:
    frame_paths = sorted(frame_dir.glob("im_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames matching im_*.jpg under {frame_dir}")
    if cv2 is None:
        raise RuntimeError("opencv-python is required to probe frame directory resolutions")
    frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    return int(frame.shape[0]), int(frame.shape[1])


def _probe_virtual_hdf5_resolution(container_path: Path, inner_path: str) -> tuple[int, int]:
    with h5py.File(container_path, "r") as handle:
        dataset = handle[inner_path]
        if len(dataset.shape) == 4 and int(dataset.shape[-1]) == 3:
            return int(dataset.shape[1]), int(dataset.shape[2])
        if inner_path.endswith("/rgb") and f"{inner_path}_size" in handle:
            sizes = np.asarray(handle[f"{inner_path}_size"], dtype=np.int64).reshape(-1)
            if sizes.shape[0] <= 0:
                raise ValueError(f"Empty JPEG size stream for {container_path}::{inner_path}")
            blob = np.asarray(handle[inner_path], dtype=np.uint8).reshape(-1)
            first_size = int(sizes[0])
            payload = blob[:first_size]
            if payload.size <= 0:
                raise ValueError(f"Empty first JPEG payload for {container_path}::{inner_path}")
            if cv2 is None:
                raise RuntimeError("opencv-python is required to probe JPEG stream resolutions")
            image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode first JPEG for {container_path}::{inner_path}")
            return int(image.shape[0]), int(image.shape[1])
        if len(dataset.shape) >= 1 and int(dataset.shape[0]) > 0:
            first_value = dataset[0]
            payload = _buffer_to_uint8_array(first_value)
            if payload.size > 0:
                if cv2 is None:
                    raise RuntimeError("opencv-python is required to probe JPEG stream resolutions")
                image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
                if image is not None:
                    return int(image.shape[0]), int(image.shape[1])
        if len(dataset.shape) >= 3 and int(dataset.shape[-1]) == 3:
            return int(dataset.shape[-3]), int(dataset.shape[-2])
        raise ValueError(f"Unable to infer source resolution for {container_path}::{inner_path}")


def _probe_source_resolution(source: str) -> tuple[int, int]:
    if "::" in str(source):
        container_path, inner_path = _parse_virtual_video_path(source)
        return _probe_virtual_hdf5_resolution(container_path, inner_path)
    path = Path(source)
    if path.is_dir():
        return _probe_frame_dir_resolution(path)
    info = _ffprobe_video_info(path)
    if info["height"] is None or info["width"] is None:
        raise RuntimeError(f"Failed to infer source resolution for {path}")
    return int(info["height"]), int(info["width"])


def _sanitize_instruction(text: str) -> str:
    return str(text).strip()


def _ffprobe_video_info(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    fps = _parse_ffmpeg_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = stream.get("duration")
    duration_f = float(duration) if duration not in (None, "", "N/A") else None
    nb_frames = _safe_int(stream.get("nb_frames"))
    if nb_frames is None and fps is not None and duration_f is not None:
        nb_frames = int(max(0, round(fps * duration_f)))
    return {
        "width": _safe_int(stream.get("width")),
        "height": _safe_int(stream.get("height")),
        "fps": fps,
        "duration": duration_f,
        "nb_frames": nb_frames,
    }


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_ffmpeg_rate(value: str | None) -> float | None:
    if value in (None, "", "0/0"):
        return None
    if "/" in str(value):
        num, den = str(value).split("/", 1)
        den_f = float(den)
        if abs(den_f) < 1e-12:
            return None
        return float(num) / den_f
    return float(value)


def _buffer_to_uint8_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.shape == ():
                return _buffer_to_uint8_array(value.item())
            if value.size == 1:
                return _buffer_to_uint8_array(value.reshape(-1)[0])
        return np.asarray(value, dtype=np.uint8).reshape(-1)
    if isinstance(value, np.void):
        return np.frombuffer(value.tobytes(), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8)
    return np.frombuffer(bytes(value), dtype=np.uint8)


def _run_ffmpeg_with_stdin(
    cmd: Sequence[str],
    chunks: Iterable[bytes],
) -> None:
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for chunk in chunks:
            proc.stdin.write(chunk)
        proc.stdin.close()
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n"
            f"{' '.join(str(part) for part in cmd)}\n"
            f"stdout:\n{stdout.decode('utf-8', errors='ignore')}\n"
            f"stderr:\n{stderr.decode('utf-8', errors='ignore')}"
        )


def _build_ffmpeg_output_args(
    *,
    output_path: Path,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    num_frames: int,
) -> list[str]:
    return [
        "-an",
        "-vf",
        f"scale={target_width}:{target_height}:flags=lanczos",
        "-frames:v",
        str(int(num_frames)),
        "-r",
        f"{float(fps):.6f}",
        "-c:v",
        str(video_codec),
        "-preset",
        str(video_preset),
        "-crf",
        str(int(video_crf)),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _export_mp4_to_mp4(
    *,
    source_path: Path,
    output_path: Path,
    num_frames: int,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
    ]
    cmd.extend(
        _build_ffmpeg_output_args(
            output_path=output_path,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            num_frames=num_frames,
        )
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source_path} -> {output_path}: {proc.stderr.strip()}"
        )


def _export_frame_dir_to_mp4(
    *,
    frame_dir: Path,
    output_path: Path,
    num_frames: int,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):.6f}",
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "im_%d.jpg"),
    ]
    cmd.extend(
        _build_ffmpeg_output_args(
            output_path=output_path,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            num_frames=num_frames,
        )
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for frame_dir={frame_dir} -> {output_path}: {proc.stderr.strip()}"
        )


def _export_rgb_frame_iter_to_mp4(
    *,
    frames: Iterable[np.ndarray],
    source_height: int,
    source_width: int,
    output_path: Path,
    num_frames: int,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{int(source_width)}x{int(source_height)}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "pipe:0",
    ]
    cmd.extend(
        _build_ffmpeg_output_args(
            output_path=output_path,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            num_frames=num_frames,
        )
    )

    def _iter_chunks() -> Iterator[bytes]:
        count = 0
        for frame in frames:
            if count >= int(num_frames):
                break
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected RGB frame [H,W,3], got {arr.shape}")
            if int(arr.shape[0]) != int(source_height) or int(arr.shape[1]) != int(source_width):
                raise ValueError(
                    f"Inconsistent frame size. Expected {(source_height, source_width)}, got {arr.shape[:2]}"
                )
            yield np.ascontiguousarray(arr).tobytes()
            count += 1
        if count != int(num_frames):
            raise RuntimeError(f"Expected {num_frames} RGB frames, only saw {count}")

    _run_ffmpeg_with_stdin(cmd, _iter_chunks())


def _export_jpeg_payload_iter_to_mp4(
    *,
    payloads: Iterable[Any],
    output_path: Path,
    num_frames: int,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mjpeg",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "pipe:0",
    ]
    cmd.extend(
        _build_ffmpeg_output_args(
            output_path=output_path,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            num_frames=num_frames,
        )
    )

    def _iter_chunks() -> Iterator[bytes]:
        count = 0
        for payload in payloads:
            if count >= int(num_frames):
                break
            yield _buffer_to_uint8_array(payload).tobytes()
            count += 1
        if count != int(num_frames):
            raise RuntimeError(f"Expected {num_frames} JPEG frames, only saw {count}")

    _run_ffmpeg_with_stdin(cmd, _iter_chunks())


def _parse_virtual_video_path(value: str) -> tuple[Path, str]:
    text = str(value)
    if "::" not in text:
        raise ValueError(f"Not a virtual video path: {value}")
    container_path, inner_path = text.split("::", 1)
    return Path(container_path), inner_path


def _count_virtual_hdf5_frames(container_path: Path, inner_path: str) -> int:
    with h5py.File(container_path, "r") as handle:
        dataset = handle[inner_path]
        if inner_path.endswith("/rgb"):
            size_key = f"{inner_path}_size"
            if size_key in handle:
                return int(np.asarray(handle[size_key]).reshape(-1).shape[0])
        if len(dataset.shape) <= 0:
            raise ValueError(f"HDF5 dataset has no frame axis: {inner_path}")
        return int(dataset.shape[0])


def _count_source_frames(source: str) -> int:
    if "::" in str(source):
        container_path, inner_path = _parse_virtual_video_path(source)
        return _count_virtual_hdf5_frames(container_path, inner_path)

    path = Path(source)
    if path.is_dir():
        return len(sorted(path.glob("im_*.jpg")))
    info = _ffprobe_video_info(path)
    if info["nb_frames"] is None:
        raise RuntimeError(f"Failed to infer frame count for {path}")
    return int(info["nb_frames"])


def _export_virtual_hdf5_source(
    *,
    source: str,
    output_path: Path,
    num_frames: int,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    container_path, inner_path = _parse_virtual_video_path(source)
    with h5py.File(container_path, "r") as handle:
        dataset = handle[inner_path]
        if (
            len(dataset.shape) == 4
            and int(dataset.shape[-1]) == 3
            and np.issubdtype(dataset.dtype, np.integer)
        ):
            source_height = int(dataset.shape[1])
            source_width = int(dataset.shape[2])

            def _iter_frames() -> Iterator[np.ndarray]:
                for frame_idx in range(int(num_frames)):
                    yield np.asarray(dataset[frame_idx], dtype=np.uint8)

            _export_rgb_frame_iter_to_mp4(
                frames=_iter_frames(),
                source_height=source_height,
                source_width=source_width,
                output_path=output_path,
                num_frames=num_frames,
                fps=fps,
                target_height=target_height,
                target_width=target_width,
                video_codec=video_codec,
                video_crf=video_crf,
                video_preset=video_preset,
            )
            return

        if inner_path.endswith("/rgb") and f"{inner_path}_size" in handle:
            blob = np.asarray(handle[inner_path], dtype=np.uint8).reshape(-1)
            sizes = np.asarray(handle[f"{inner_path}_size"], dtype=np.int64).reshape(-1)[:num_frames]
            offsets = np.zeros(sizes.shape[0] + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(sizes, dtype=np.int64)
            if int(offsets[-1]) > int(blob.shape[0]):
                raise ValueError(
                    f"Corrupted JPEG stream for {container_path}::{inner_path}: "
                    f"{int(offsets[-1])} > {int(blob.shape[0])}"
                )

            def _iter_payloads() -> Iterator[np.ndarray]:
                for frame_idx in range(int(num_frames)):
                    start = int(offsets[frame_idx])
                    end = int(offsets[frame_idx + 1])
                    yield blob[start:end]

            _export_jpeg_payload_iter_to_mp4(
                payloads=_iter_payloads(),
                output_path=output_path,
                num_frames=num_frames,
                fps=fps,
                target_height=target_height,
                target_width=target_width,
                video_codec=video_codec,
                video_crf=video_crf,
                video_preset=video_preset,
            )
            return

        def _iter_payloads() -> Iterator[Any]:
            for frame_idx in range(int(num_frames)):
                yield dataset[frame_idx]

        _export_jpeg_payload_iter_to_mp4(
            payloads=_iter_payloads(),
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
        )


def _export_source_to_output(
    *,
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None,
    source: str,
    output_path: Path,
    num_frames: int,
    fps: float,
    camera_key: str,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> None:
    target_height, target_width = _camera_resolution(
        args,
        camera_key,
        resolution_config,
    )
    if "::" in str(source):
        _export_virtual_hdf5_source(
            source=source,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
        )
        return

    path = Path(source)
    if path.is_dir():
        _export_frame_dir_to_mp4(
            frame_dir=path,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
            target_height=target_height,
            target_width=target_width,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
        )
        return

    _export_mp4_to_mp4(
        source_path=path,
        output_path=output_path,
        num_frames=num_frames,
        fps=fps,
        target_height=target_height,
        target_width=target_width,
        video_codec=video_codec,
        video_crf=video_crf,
        video_preset=video_preset,
    )


def _episode_parquet_rows(
    *,
    episode_index: int,
    task_index: int,
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    global_index_offset: int,
) -> dict[str, list[Any]]:
    num_frames = int(actions.shape[0])
    return {
        "observation.state": np.asarray(states, dtype=np.float32).tolist(),
        "action": np.asarray(actions, dtype=np.float32).tolist(),
        "timestamp": np.asarray(timestamps, dtype=np.float32).tolist(),
        "frame_index": list(range(num_frames)),
        "episode_index": [int(episode_index)] * num_frames,
        "index": list(range(global_index_offset, global_index_offset + num_frames)),
        "task_index": [int(task_index)] * num_frames,
    }


def _generic_feature_names(dim: int) -> list[str]:
    return [f"d{idx:02d}" for idx in range(int(dim))]


WXYZ_DUAL_ARM_STATE_ACTION_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
]


def _metadata_feature_names(
    repo_root: Path,
    feature_key: str,
) -> list[str] | None:
    info_path = repo_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = load_json(info_path)
    except Exception:
        return None
    feature = info.get("features", {}).get(feature_key)
    if not isinstance(feature, dict):
        return None
    names = feature.get("names")
    if not isinstance(names, list):
        return None
    return [str(name) for name in names]


def _pose12_rpy_to_pose16_wxyz(pose12: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose12, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected pose12 shape [T,12], got {arr.shape}")
    left = xyz_euler_xyz_to_xyz_quat_wxyz(arr[:, 0:6])
    right = xyz_euler_xyz_to_xyz_quat_wxyz(arr[:, 6:12])
    return np.concatenate([left, right], axis=1).astype(np.float32, copy=False)


def _pose16_xyzw_to_wxyz(pose16: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose16, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"Expected pose16 shape [T,16], got {arr.shape}")
    left_quat = quaternion_xyzw_to_wxyz(arr[:, 3:7])
    right_quat = quaternion_xyzw_to_wxyz(arr[:, 11:15])
    out = np.concatenate(
        [
            arr[:, 0:3],
            left_quat,
            arr[:, 7:8],
            arr[:, 8:11],
            right_quat,
            arr[:, 15:16],
        ],
        axis=1,
    )
    return out.astype(np.float32, copy=False)


def _pose16_wxyz_to_wxyz(pose16: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose16, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"Expected pose16 shape [T,16], got {arr.shape}")
    left_quat = standardize_quaternion_wxyz(arr[:, 3:7])
    right_quat = standardize_quaternion_wxyz(arr[:, 11:15])
    out = np.concatenate(
        [
            arr[:, 0:3],
            left_quat,
            arr[:, 7:8],
            arr[:, 8:11],
            right_quat,
            arr[:, 15:16],
        ],
        axis=1,
    )
    return out.astype(np.float32, copy=False)


def _normalize_repo_pose_actions(
    *,
    spec: DatasetSpec,
    record_payload: Mapping[str, Any],
    actions: np.ndarray,
    states: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if spec.name not in {"robomind", "robocoin"}:
        return actions, states, {}

    repo_root = Path(record_payload["repo_root"])
    notes: dict[str, Any] = {}

    if spec.name == "robocoin":
        action_names = _metadata_feature_names(repo_root, "eef_sim_pose_action") or []
        if len(action_names) < 12:
            return None, None, {
                "pose_filter_reason": "missing_eef_sim_pose_action_metadata",
            }
        if not any("ori_" in name for name in action_names):
            return None, None, {
                "pose_filter_reason": "unsupported_robocoin_pose_layout",
                "source_pose_names": action_names,
            }

        if int(actions.shape[1]) < 12:
            return None, None, {
                "pose_filter_reason": "robocoin_action_too_short",
                "source_action_dim": int(actions.shape[1]),
            }
        gripper = actions[:, 12:14] if int(actions.shape[1]) >= 14 else np.zeros((actions.shape[0], 2), dtype=np.float32)
        actions_out = _pose12_rpy_to_pose16_wxyz(actions[:, :12])
        actions_out = np.concatenate(
            [actions_out[:, :7], gripper[:, 0:1], actions_out[:, 7:14], gripper[:, 1:2]],
            axis=1,
        ).astype(np.float32, copy=False)

        states_out = None
        if states is not None:
            states_arr = np.asarray(states, dtype=np.float32)
            if int(states_arr.shape[1]) >= 12:
                state_gripper = states_arr[:, 12:14] if int(states_arr.shape[1]) >= 14 else gripper
                states_pose = _pose12_rpy_to_pose16_wxyz(states_arr[:, :12])
                states_out = np.concatenate(
                    [states_pose[:, :7], state_gripper[:, 0:1], states_pose[:, 7:14], state_gripper[:, 1:2]],
                    axis=1,
                ).astype(np.float32, copy=False)
        notes["source_pose_repr"] = "xyz+rpy"
        notes["output_pose_repr"] = "xyz+wxyz"
        return actions_out, states_out, notes

    info_path = repo_root / "meta" / "info.json"
    try:
        info = load_json(info_path)
    except Exception:
        info = {}
    features = info.get("features", {}) if isinstance(info, dict) else {}
    if not (
        isinstance(features, dict)
        and "action.eef_left_pose" in features
        and "action.eef_right_pose" in features
    ):
        return None, None, {
            "pose_filter_reason": "missing_robomind_eef_pose",
        }

    if int(actions.shape[1]) != 16:
        return None, None, {
            "pose_filter_reason": "unexpected_robomind_action_dim",
            "source_action_dim": int(actions.shape[1]),
        }

    actions_out = _pose16_xyzw_to_wxyz(actions)
    states_out = None
    if states is not None and int(states.shape[1]) == 16:
        states_out = _pose16_xyzw_to_wxyz(states)
    notes["source_pose_repr"] = "xyz+xyzw"
    notes["output_pose_repr"] = "xyz+wxyz"
    return actions_out, states_out, notes


def _uniform_timestamps(num_frames: int, fps: float) -> np.ndarray:
    return (np.arange(int(num_frames), dtype=np.float32) / float(fps)).astype(np.float32)


def _bridge_timestamps(record_payload: Mapping[str, Any], num_frames: int, fps: float) -> np.ndarray:
    traj_dir = Path(record_payload["traj_dir"])
    obs_payload = load_bridge_pickle(traj_dir / "obs_dict.pkl")
    if isinstance(obs_payload, dict) and "time_stamp" in obs_payload:
        timestamps = np.asarray(obs_payload["time_stamp"], dtype=np.float64).reshape(-1)
        if timestamps.shape[0] >= int(num_frames):
            timestamps = timestamps[: int(num_frames)]
            timestamps = timestamps - timestamps[0]
            return timestamps.astype(np.float32)
    return _uniform_timestamps(num_frames, fps)


def _droid_timestamps(record_payload: Mapping[str, Any], num_frames: int, fps: float) -> np.ndarray:
    episode_dir = Path(record_payload["episode_dir"])
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        dataset_map = walk_hdf5_datasets(handle)
        seconds = nanos = None
        for candidate in DROID_TIMESTAMP_SECONDS_CANDIDATES:
            if candidate in dataset_map:
                seconds = np.asarray(dataset_map[candidate]).reshape(-1)
                break
        for candidate in DROID_TIMESTAMP_NANOS_CANDIDATES:
            if candidate in dataset_map:
                nanos = np.asarray(dataset_map[candidate]).reshape(-1)
                break
        if (
            seconds is not None
            and nanos is not None
            and int(seconds.shape[0]) >= int(num_frames)
            and int(nanos.shape[0]) >= int(num_frames)
        ):
            timestamps = seconds[: int(num_frames)].astype(np.float64)
            timestamps = timestamps + nanos[: int(num_frames)].astype(np.float64) * 1e-9
            timestamps = timestamps - timestamps[0]
            return timestamps.astype(np.float32)
    return _uniform_timestamps(num_frames, fps)


def _repo_info_fps(record_payload: Mapping[str, Any], fallback_fps: float) -> float:
    repo_root = Path(record_payload["repo_root"])
    info_path = repo_root / "meta" / "info.json"
    if info_path.exists():
        try:
            info = load_json(info_path)
            fps = float(info.get("fps", fallback_fps))
            if np.isfinite(fps) and fps > 0:
                return fps
        except Exception:
            pass
    return float(fallback_fps)


def _repo_parquet_path(record_payload: Mapping[str, Any]) -> Path | None:
    repo_root = record_payload.get("repo_root")
    episode_index = record_payload.get("episode_index")
    if repo_root is None or episode_index is None:
        return None
    episode_chunk = int(episode_index) // 1000
    path = (
        Path(repo_root)
        / "data"
        / f"chunk-{episode_chunk:03d}"
        / f"episode_{int(episode_index):06d}.parquet"
    )
    return path if path.exists() else None


def _repo_parquet_timestamps(
    record_payload: Mapping[str, Any],
    num_frames: int,
) -> np.ndarray | None:
    parquet_path = _repo_parquet_path(record_payload)
    if parquet_path is None or pq is None:
        return None
    try:
        table = pq.read_table(parquet_path, columns=["timestamp"])
        values = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if values.shape[0] < int(num_frames):
        return None
    out = values[: int(num_frames)].astype(np.float32, copy=False)
    out = out - out[0]
    return out


def _episode_fps(spec: DatasetSpec, record_payload: Mapping[str, Any]) -> float:
    if spec.name in {"robocoin", "robomind"}:
        return _repo_info_fps(record_payload, spec.default_fps)
    return float(spec.default_fps)


def _episode_timestamps(
    *,
    spec: DatasetSpec,
    record_payload: Mapping[str, Any],
    num_frames: int,
    fps: float,
) -> np.ndarray:
    if spec.name == "bridge":
        return _bridge_timestamps(record_payload, num_frames, fps)
    if spec.name == "droid":
        return _droid_timestamps(record_payload, num_frames, fps)
    parquet_timestamps = _repo_parquet_timestamps(record_payload, num_frames)
    if parquet_timestamps is not None:
        return parquet_timestamps
    return _uniform_timestamps(num_frames, fps)


def _ensure_repo_dirs(
    repo_dir: Path,
    *,
    overwrite: bool,
    videos_dir_name: str,
) -> tuple[Path, Path, Path]:
    if repo_dir.exists():
        if not bool(overwrite):
            raise FileExistsError(f"Repo already exists: {repo_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(repo_dir)
    data_dir = repo_dir / "data" / "chunk-000"
    videos_root = repo_dir / str(videos_dir_name) / "chunk-000"
    source_meta_dir = repo_dir / "source_meta"
    ensure_dir(data_dir)
    ensure_dir(videos_root)
    ensure_dir(source_meta_dir)
    return data_dir, videos_root, source_meta_dir


def _dataset_family_root(output_root: Path, spec: DatasetSpec) -> Path:
    return Path(output_root) / spec.name


def _dataset_repo_root(output_root: Path, spec: DatasetSpec) -> Path:
    return _dataset_family_root(output_root, spec) / spec.output_name


def _summary_output_root(output_root: Path, spec: DatasetSpec) -> Path:
    if spec.name == "robotwin":
        return _dataset_family_root(output_root, spec)
    return _dataset_repo_root(output_root, spec)


@contextmanager
def _legacy_videos_compat(repo_dir: Path) -> Iterator[None]:
    legacy_videos_root = repo_dir / "videos"
    actual_videos_root = resolve_lerobot_videos_root(repo_dir, must_exist=True)
    created_legacy_link = False
    if not legacy_videos_root.exists():
        legacy_videos_root.symlink_to(actual_videos_root.name, target_is_directory=True)
        created_legacy_link = True
    try:
        yield
    finally:
        if created_legacy_link and legacy_videos_root.is_symlink():
            legacy_videos_root.unlink()


def _rewrite_repo_info_video_path(
    repo_dir: Path,
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None = None,
) -> None:
    info_path = repo_dir / "meta" / "info.json"
    if not info_path.exists():
        return
    info = load_json(info_path)
    video_path = str(info.get("video_path", ""))
    if video_path.startswith("videos/"):
        effective_resolution = (
            resolution_config
            or _resolve_existing_repo_resolution_config(repo_dir)
            or _resolution_config_from_args(args)
        )
        info["video_path"] = video_path.replace(
            "videos/",
            f"{_videos_dir_name(args, effective_resolution)}/",
            1,
        )
        dump_json(info_path, info)


def _rebuild_repo_meta(
    *,
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None,
    repo_dir: Path,
    robot_type: str,
    source_name: str,
    source_suffix: str,
    action_names: Sequence[str],
    state_names: Sequence[str],
    camera_keys: Sequence[str] | None = None,
    include_image_stats: bool = True,
):
    with _legacy_videos_compat(repo_dir):
        result = rebuild_lerobot_v21_meta(
            repo_dir=repo_dir,
            robot_type=robot_type,
            source_name=source_name,
            source_suffix=source_suffix,
            action_names=action_names,
            state_names=state_names,
            camera_keys=camera_keys,
            include_image_stats=include_image_stats,
        )
    _rewrite_repo_info_video_path(repo_dir, args, resolution_config)
    return result


def _generic_source_meta(
    *,
    spec: DatasetSpec,
    record: Any,
    instruction: str,
    camera_sources: Mapping[str, str],
    deferred_camera_sources: Mapping[str, str] | None,
    fps: float,
    num_frames: int,
    action_dim: int,
) -> dict[str, Any]:
    deferred = dict(deferred_camera_sources or {})
    return {
        "task": instruction,
        "source_name": spec.name,
        "source_episode_id": str(record.episode_id),
        "action_config": [
            {
                "start_frame": 0,
                "end_frame": int(num_frames),
                "action_text": instruction,
                "skill": "",
            }
        ],
        "source_meta": {
            "dataset_name": spec.name,
            "raw_record_payload": dict(record.payload),
            "camera_sources": dict(camera_sources),
            "camera_output_mapping": {
                _camera_to_output_key(key): value for key, value in camera_sources.items()
            },
            "deferred_camera_sources": deferred,
            "deferred_camera_output_mapping": {
                _camera_to_output_key(key): value for key, value in deferred.items()
            },
            "fps": float(fps),
            "frame_sample_stride": FRAME_SAMPLE_STRIDE,
            "sampled_frame_ids": _sampled_frame_ids(num_frames),
            "raw_absolute_action_dim": int(action_dim),
            "notes": [
                "This repo was generated from wan_va.dataset.curation raw dataset readers.",
                "observation.state is populated with raw_absolute_actions as a compatibility fallback.",
                "Videos are stored as full resized episodes; sampled_frame_ids are only recorded in source_meta.",
            ],
        },
    }


def _robotwin_task_repo_name(record: Any) -> str:
    payload = getattr(record, "payload", {})
    task_repo_name = str(payload.get("task_repo_name", "")).strip()
    if task_repo_name:
        return task_repo_name
    task_slug = str(payload.get("task_slug", "")).strip()
    task_dir_name = str(payload.get("task_dir_name", "")).strip()
    if task_slug and task_dir_name:
        if task_dir_name == task_slug or task_dir_name.startswith(f"{task_slug}-"):
            return task_dir_name
        return f"{task_slug}-{task_dir_name}"
    return str(getattr(record, "episode_id", "robotwin")).replace(":", "_")


@dataclass(frozen=True)
class PreparedEpisodeWrite:
    record: Any
    instruction: str
    camera_sources: dict[str, str]
    fps: float
    num_frames: int
    timestamps: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    raw_qpos: np.ndarray | None
    used_fallback_state: bool
    action_dim: int
    source_meta_updates: dict[str, Any]


def _prepare_generic_episode(
    *,
    spec: DatasetSpec,
    record: Any,
    sample: Mapping[str, Any],
    expected_action_dim: int | None,
) -> tuple[PreparedEpisodeWrite | None, int | None]:
    actions = np.asarray(sample["raw_absolute_actions"], dtype=np.float32)
    raw_states = sample.get("raw_states")
    raw_qpos = sample.get("raw_qpos")
    instruction = _sanitize_instruction(sample["instruction"])
    camera_sources = {key: str(value) for key, value in sample["video_frames"].items()}
    if not camera_sources:
        return None, expected_action_dim

    fps = _episode_fps(spec, record.payload)
    if not np.isfinite(fps) or fps <= 0:
        fps = float(spec.default_fps)

    source_counts = {
        key: _count_source_frames(source)
        for key, source in camera_sources.items()
    }
    num_frames = min(int(actions.shape[0]), *(int(count) for count in source_counts.values()))
    if num_frames <= 0:
        return None, expected_action_dim

    timestamps = _episode_timestamps(
        spec=spec,
        record_payload=record.payload,
        num_frames=num_frames,
        fps=fps,
    )
    num_frames = min(int(num_frames), int(timestamps.shape[0]))
    if num_frames <= 0:
        return None, expected_action_dim

    actions = actions[:num_frames].astype(np.float32, copy=False)
    if raw_states is None:
        states = actions.copy()
        used_fallback_state = True
    else:
        states = np.asarray(raw_states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError(
                f"dataset={spec.name} raw_states must have shape [T,D], got {states.shape}"
            )
        if int(states.shape[0]) < num_frames:
            raise ValueError(
                f"dataset={spec.name} raw_states is shorter than actions/video: "
                f"states={int(states.shape[0])}, required={num_frames}, "
                f"episode_id={record.episode_id}"
        )
        states = states[:num_frames].astype(np.float32, copy=False)
        used_fallback_state = False

    actions, maybe_states, source_meta_updates = _normalize_repo_pose_actions(
        spec=spec,
        record_payload=record.payload,
        actions=actions,
        states=states,
    )
    if actions is None:
        return None, expected_action_dim
    if maybe_states is not None:
        states = maybe_states

    action_dim = int(actions.shape[1])
    if expected_action_dim is None:
        expected_action_dim = action_dim
    elif action_dim != expected_action_dim:
        raise ValueError(
            f"dataset={spec.name} has inconsistent action dims: "
            f"expected {expected_action_dim}, got {action_dim} "
            f"episode_id={record.episode_id}"
        )
    timestamps = timestamps[:num_frames].astype(np.float32, copy=False)

    qpos = None
    if raw_qpos is not None:
        qpos = np.asarray(raw_qpos, dtype=np.float32)
        if qpos.ndim == 2 and int(qpos.shape[0]) >= num_frames:
            qpos = qpos[:num_frames].astype(np.float32, copy=False)
        else:
            qpos = None

    prepared = PreparedEpisodeWrite(
        record=record,
        instruction=instruction,
        camera_sources=camera_sources,
        fps=float(fps),
        num_frames=int(num_frames),
        timestamps=timestamps,
        actions=actions,
        states=states,
        raw_qpos=qpos,
        used_fallback_state=used_fallback_state,
        action_dim=action_dim,
        source_meta_updates=source_meta_updates,
    )
    return prepared, expected_action_dim


def _write_prepared_generic_episode(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None,
    prepared: PreparedEpisodeWrite,
    data_dir: Path,
    videos_root: Path,
    source_meta_dir: Path,
    task_index: int,
    global_index_offset: int,
    output_episode_index: int,
) -> tuple[int, set[str], list[PendingWristVideoTask]]:
    written_camera_keys: set[str] = set()
    pending_wrist_tasks: list[PendingWristVideoTask] = []
    written_camera_sources: dict[str, str] = {}
    deferred_camera_sources: dict[str, str] = {}
    for camera_key, source in prepared.camera_sources.items():
        output_camera_key = _camera_to_output_key(camera_key)
        output_path = (
            videos_root
            / output_camera_key
            / f"episode_{output_episode_index:06d}.mp4"
        )
        if bool(args.main_only) and _is_wrist_camera(camera_key):
            pending_wrist_tasks.append(
                PendingWristVideoTask(
                    output_relpath=str(output_path.relative_to(videos_root.parent.parent)),
                    source=str(source),
                    num_frames=int(prepared.num_frames),
                    fps=float(prepared.fps),
                    camera_key=str(camera_key),
                    episode_index=int(output_episode_index),
                )
            )
            deferred_camera_sources[camera_key] = str(source)
            continue
        _export_source_to_output(
            args=args,
            resolution_config=resolution_config,
            source=source,
            output_path=output_path,
            num_frames=prepared.num_frames,
            fps=prepared.fps,
            camera_key=camera_key,
            video_codec=str(args.video_codec),
            video_crf=int(args.video_crf),
            video_preset=str(args.video_preset),
        )
        written_camera_keys.add(output_camera_key)
        written_camera_sources[camera_key] = str(source)

    parquet_rows = _episode_parquet_rows(
        episode_index=output_episode_index,
        task_index=task_index,
        states=prepared.states,
        actions=prepared.actions,
        timestamps=prepared.timestamps,
        global_index_offset=global_index_offset,
    )
    parquet_path = data_dir / f"episode_{output_episode_index:06d}.parquet"
    write_episode_parquet(parquet_path, parquet_rows)

    source_meta = _generic_source_meta(
        spec=spec,
        record=prepared.record,
        instruction=prepared.instruction,
        camera_sources=written_camera_sources,
        deferred_camera_sources=deferred_camera_sources,
        fps=prepared.fps,
        num_frames=prepared.num_frames,
        action_dim=prepared.action_dim,
    )
    source_meta["source_meta"]["observation_state_source"] = (
        "raw_absolute_actions_fallback" if prepared.used_fallback_state else "raw_states"
    )
    if prepared.raw_qpos is not None:
        source_meta["source_meta"]["raw_qpos_dim"] = int(prepared.raw_qpos.shape[1])
    source_meta["source_meta"].update(prepared.source_meta_updates)
    dump_json(source_meta_dir / f"episode_{output_episode_index:06d}.json", source_meta)

    return prepared.num_frames, written_camera_keys, pending_wrist_tasks


def _cleanup_partial_generic_episode_output(
    *,
    data_dir: Path,
    videos_root: Path,
    source_meta_dir: Path,
    output_episode_index: int,
) -> None:
    stem = f"episode_{int(output_episode_index):06d}"
    parquet_path = data_dir / f"{stem}.parquet"
    meta_path = source_meta_dir / f"{stem}.json"
    for path in [parquet_path, meta_path]:
        if path.exists():
            path.unlink()
    for video_path in videos_root.glob(f"*/{stem}.mp4"):
        if video_path.exists():
            video_path.unlink()


def _merge_resolution_config(
    base: ResolutionConfig,
    scanned: Mapping[str, tuple[int, int]],
) -> ResolutionConfig:
    main_sizes: list[tuple[int, int]] = []
    wrist_sizes: list[tuple[int, int]] = []
    for camera_key, (height, width) in scanned.items():
        if _is_wrist_camera(camera_key):
            wrist_sizes.append((int(height), int(width)))
        else:
            main_sizes.append((int(height), int(width)))
    main_height = max((size[0] for size in main_sizes), default=int(base.main_height))
    main_width = max((size[1] for size in main_sizes), default=int(base.main_width))
    wrist_height = max((size[0] for size in wrist_sizes), default=int(base.wrist_height))
    wrist_width = max((size[1] for size in wrist_sizes), default=int(base.wrist_width))
    return ResolutionConfig(
        main_height=main_height,
        main_width=main_width,
        wrist_height=wrist_height,
        wrist_width=wrist_width,
    )


def _scan_max_camera_resolutions_for_samples(
    samples: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
    maxima: dict[str, tuple[int, int]] = {}
    for sample in samples:
        for camera_key, source in sample["video_frames"].items():
            height, width = _probe_source_resolution(str(source))
            prev = maxima.get(str(camera_key))
            if prev is None:
                maxima[str(camera_key)] = (int(height), int(width))
            else:
                maxima[str(camera_key)] = (
                    max(int(prev[0]), int(height)),
                    max(int(prev[1]), int(width)),
                )
    return maxima


def _scan_dataset_resolution_config(
    *,
    args: argparse.Namespace,
    dataset: Any,
    sample_indices: Sequence[int] | None = None,
) -> ResolutionConfig:
    base = _resolution_config_from_args(args)
    if not bool(args.keep_original_resolution):
        return base
    indices = (
        list(range(len(dataset)))
        if sample_indices is None
        else [int(index) for index in sample_indices]
    )
    scanned = _scan_max_camera_resolutions_for_samples(
        dataset[sample_index] for sample_index in indices
    )
    return _merge_resolution_config(base, scanned)


def _resume_pending_wrist_tasks(
    *,
    args: argparse.Namespace,
    repo_dir: Path,
    resolution_config: ResolutionConfig | None,
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    pending = _load_pending_wrist_tasks(repo_dir)
    if not pending:
        return {
            "repo_dir": str(repo_dir),
            "resumed_wrist_videos": 0,
            "camera_keys": sorted(set(str(key) for key in (camera_keys or []))),
        }
    effective_resolution = (
        resolution_config
        or _resolve_existing_repo_resolution_config(repo_dir)
        or _resolution_config_from_args(args)
    )
    resumed = 0
    for task in pending:
        output_path = repo_dir / task.output_relpath
        _export_source_to_output(
            args=args,
            resolution_config=effective_resolution,
            source=task.source,
            output_path=output_path,
            num_frames=int(task.num_frames),
            fps=float(task.fps),
            camera_key=str(task.camera_key),
            video_codec=str(args.video_codec),
            video_crf=int(args.video_crf),
            video_preset=str(args.video_preset),
        )
        resumed += 1
    _save_pending_wrist_tasks(repo_dir, [])
    return {
        "repo_dir": str(repo_dir),
        "resumed_wrist_videos": int(resumed),
        "camera_keys": sorted(set(str(key) for key in (camera_keys or []))),
    }


def _load_existing_repo_feature_names(
    repo_dir: Path,
) -> tuple[list[str], list[str]]:
    info_path = repo_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing repo metadata: {info_path}")
    info = load_json(info_path)
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    parsed_state = list(state_names[0]) if isinstance(state_names, list) and state_names else []
    parsed_action = list(action_names[0]) if isinstance(action_names, list) and action_names else []
    if not parsed_state or not parsed_action:
        raise ValueError(f"Failed to read existing feature names from {info_path}")
    return parsed_action, parsed_state


def _iter_existing_task_repo_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir()
            and (path / "meta" / "info.json").exists()
        ],
        key=lambda path: path.name,
    )


def _write_generic_episode(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    resolution_config: ResolutionConfig | None,
    record: Any,
    sample: Mapping[str, Any],
    data_dir: Path,
    videos_root: Path,
    source_meta_dir: Path,
    task_to_index: dict[str, int],
    global_index_offset: int,
    output_episode_index: int,
    written_camera_keys: set[str],
    expected_action_dim: int | None,
) -> tuple[int, int, int | None, list[PendingWristVideoTask]]:
    prepared, expected_action_dim = _prepare_generic_episode(
        spec=spec,
        record=record,
        sample=sample,
        expected_action_dim=expected_action_dim,
    )
    if prepared is None:
        return 0, global_index_offset, expected_action_dim, []

    task_index = task_to_index.setdefault(prepared.instruction, len(task_to_index))
    episode_frames, episode_camera_keys, pending_wrist_tasks = _write_prepared_generic_episode(
        spec=spec,
        args=args,
        resolution_config=resolution_config,
        prepared=prepared,
        data_dir=data_dir,
        videos_root=videos_root,
        source_meta_dir=source_meta_dir,
        task_index=task_index,
        global_index_offset=global_index_offset,
        output_episode_index=output_episode_index,
    )
    written_camera_keys.update(episode_camera_keys)
    return (
        int(episode_frames),
        global_index_offset + int(episode_frames),
        expected_action_dim,
        pending_wrist_tasks,
    )


def _convert_robotwin_dataset_per_task(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    assert spec.cls is not None
    dataset_root = _dataset_family_root(args.output_root, spec)
    ensure_dir(dataset_root)

    if bool(args.resume_wrist):
        repo_dirs: list[str] = []
        total_resumed_wrist = 0
        for repo_dir in _iter_existing_task_repo_dirs(dataset_root):
            resolution_config = (
                _resolve_existing_repo_resolution_config(repo_dir)
                or _resolution_config_from_args(args)
            )
            result = _resume_pending_wrist_tasks(
                args=args,
                repo_dir=repo_dir,
                resolution_config=resolution_config,
            )
            action_names, state_names = _load_existing_repo_feature_names(repo_dir)
            _rebuild_repo_meta(
                args=args,
                resolution_config=resolution_config,
                repo_dir=repo_dir,
                robot_type=str(spec.robot_type),
                source_name=str(spec.name),
                source_suffix=str(spec.source_suffix),
                action_names=action_names,
                state_names=state_names,
                camera_keys=result["camera_keys"] or None,
                include_image_stats=not bool(args.skip_image_stats),
            )
            repo_dirs.append(str(repo_dir))
            total_resumed_wrist += int(result["resumed_wrist_videos"])
        if not repo_dirs:
            raise RuntimeError(
                f"dataset={spec.name} found no existing task repos with pending wrist exports under {dataset_root}"
            )
        return {
            "dataset": spec.name,
            "repo_dir": str(dataset_root),
            "repo_dirs": repo_dirs,
            "episodes": 0,
            "frames": 0,
            "skipped": 0,
            "pending_wrist_videos": 0,
            "resumed_wrist_videos": int(total_resumed_wrist),
        }

    dataset = spec.cls(
        input_root=str(spec.input_root),
        max_episodes=int(args.max_episodes),
        video_backend=str(args.video_backend),
        return_video_path=True,
        **spec.kwargs,
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"dataset={spec.name} is empty")

    task_groups: dict[str, list[int]] = {}
    for sample_index, record in enumerate(dataset.records):
        task_repo_name = _robotwin_task_repo_name(record)
        task_groups.setdefault(task_repo_name, []).append(sample_index)

    total_frames = 0
    total_episodes = 0
    total_skipped = 0
    repo_dirs: list[str] = []
    total_pending_wrist = 0
    total_resumed_wrist = 0
    num_workers = max(1, int(args.num_workers))
    max_inflight = max(1, 2 * num_workers)

    for task_repo_name in sorted(task_groups):
        repo_dir = dataset_root / task_repo_name
        sample_indices = task_groups[task_repo_name]
        resolution_config = _scan_dataset_resolution_config(
            args=args,
            dataset=dataset,
            sample_indices=sample_indices,
        )
        data_dir, videos_root, source_meta_dir = _ensure_repo_dirs(
            repo_dir,
            overwrite=bool(args.overwrite),
            videos_dir_name=_videos_dir_name(args, resolution_config),
        )
        task_to_index: dict[str, int] = {}
        global_index_offset = 0
        output_episode_index = 0
        written_camera_keys: set[str] = set()
        action_dim: int | None = None
        skipped = 0
        pending_wrist_tasks: list[PendingWristVideoTask] = []
        pending: dict[int, Future[PreparedEpisodeWrite | None]] = {}
        next_submit = 0
        next_consume = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(sample_indices), desc=f"Convert {task_repo_name}", leave=True) as pbar:
                while next_consume < len(sample_indices):
                    while next_submit < len(sample_indices) and len(pending) < max_inflight:
                        sample_index = sample_indices[next_submit]
                        record = dataset.records[sample_index]
                        try:
                            sample = dataset[sample_index]
                            prepared, action_dim = _prepare_generic_episode(
                                spec=spec,
                                record=record,
                                sample=sample,
                                expected_action_dim=action_dim,
                            )
                        except Exception as exc:
                            print(
                                f"[WARN] skip robotwin episode={record.episode_id} "
                                f"repo={task_repo_name}: {exc}"
                            )
                            prepared = None
                        if prepared is None:
                            pending[next_submit] = executor.submit(lambda: None)
                        else:
                            pending[next_submit] = executor.submit(lambda item=prepared: item)
                        next_submit += 1

                    future = pending.get(next_consume)
                    if future is None:
                        raise RuntimeError(
                            f"Missing pending RobotWin episode future at index={next_consume} "
                            f"for repo={task_repo_name}"
                        )
                    if not future.done():
                        wait([future], return_when=FIRST_COMPLETED)

                    while next_consume < len(sample_indices):
                        ready = pending.get(next_consume)
                        if ready is None or not ready.done():
                            break
                        prepared = ready.result()
                        del pending[next_consume]
                        if prepared is None:
                            skipped += 1
                            pbar.update(1)
                            next_consume += 1
                            continue

                        task_index = task_to_index.setdefault(prepared.instruction, len(task_to_index))
                        try:
                            written_frames, episode_camera_keys, episode_pending_wrist = _write_prepared_generic_episode(
                                spec=spec,
                                args=args,
                                resolution_config=resolution_config,
                                prepared=prepared,
                                data_dir=data_dir,
                                videos_root=videos_root,
                                source_meta_dir=source_meta_dir,
                                task_index=task_index,
                                global_index_offset=global_index_offset,
                                output_episode_index=output_episode_index,
                            )
                        except Exception as exc:
                            _cleanup_partial_generic_episode_output(
                                data_dir=data_dir,
                                videos_root=videos_root,
                                source_meta_dir=source_meta_dir,
                                output_episode_index=output_episode_index,
                            )
                            print(
                                f"[WARN] skip robotwin write episode={prepared.record.episode_id} "
                                f"repo={task_repo_name}: {exc}"
                            )
                            skipped += 1
                            pbar.update(1)
                            next_consume += 1
                            continue
                        written_camera_keys.update(episode_camera_keys)
                        pending_wrist_tasks.extend(episode_pending_wrist)
                        global_index_offset += int(written_frames)
                        output_episode_index += 1
                        total_frames += int(written_frames)
                        total_episodes += 1
                        pbar.update(1)
                        next_consume += 1

        if output_episode_index <= 0:
            total_skipped += skipped
            continue

        assert action_dim is not None
        state_action_names = _generic_feature_names(int(action_dim))
        _rebuild_repo_meta(
            args=args,
            resolution_config=resolution_config,
            repo_dir=repo_dir,
            robot_type=str(spec.robot_type),
            source_name=str(spec.name),
            source_suffix=str(spec.source_suffix),
            action_names=state_action_names,
            state_names=state_action_names,
            camera_keys=sorted(written_camera_keys),
            include_image_stats=not bool(args.skip_image_stats),
        )
        _save_pending_wrist_tasks(repo_dir, pending_wrist_tasks)
        repo_dirs.append(str(repo_dir))
        total_pending_wrist += len(pending_wrist_tasks)
        total_skipped += skipped

    if not repo_dirs:
        raise RuntimeError(f"dataset={spec.name} produced no converted task repos")

    return {
        "dataset": spec.name,
        "repo_dir": str(dataset_root),
        "repo_dirs": repo_dirs,
        "episodes": int(total_episodes),
        "frames": int(total_frames),
        "skipped": int(total_skipped),
        "pending_wrist_videos": int(total_pending_wrist),
        "resumed_wrist_videos": int(total_resumed_wrist),
    }


def _convert_generic_dataset(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if spec.name == "robotwin":
        return _convert_robotwin_dataset_per_task(spec=spec, args=args)
    assert spec.cls is not None
    repo_dir = _dataset_repo_root(args.output_root, spec)
    if bool(args.resume_wrist):
        resolution_config = (
            _resolve_existing_repo_resolution_config(repo_dir)
            or _resolution_config_from_args(args)
        )
        result = _resume_pending_wrist_tasks(
            args=args,
            repo_dir=repo_dir,
            resolution_config=resolution_config,
        )
        action_names, state_names = _load_existing_repo_feature_names(repo_dir)
        rebuilt = _rebuild_repo_meta(
            args=args,
            resolution_config=resolution_config,
            repo_dir=repo_dir,
            robot_type=str(spec.robot_type),
            source_name=str(spec.name),
            source_suffix=str(spec.source_suffix),
            action_names=action_names,
            state_names=state_names,
            camera_keys=result["camera_keys"] or None,
            include_image_stats=not bool(args.skip_image_stats),
        )
        return {
            "dataset": spec.name,
            "repo_dir": str(rebuilt.repo_dir),
            "episodes": int(rebuilt.total_episodes),
            "frames": int(rebuilt.total_frames),
            "skipped": 0,
            "pending_wrist_videos": 0,
            "resumed_wrist_videos": int(result["resumed_wrist_videos"]),
        }

    dataset = spec.cls(
        input_root=str(spec.input_root),
        max_episodes=int(args.max_episodes),
        video_backend=str(args.video_backend),
        return_video_path=True,
        **spec.kwargs,
    )
    resolution_config = _scan_dataset_resolution_config(
        args=args,
        dataset=dataset,
    )
    data_dir, videos_root, source_meta_dir = _ensure_repo_dirs(
        repo_dir,
        overwrite=bool(args.overwrite),
        videos_dir_name=_videos_dir_name(args, resolution_config),
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"dataset={spec.name} is empty")

    task_to_index: dict[str, int] = {}
    global_index_offset = 0
    output_episode_index = 0
    written_camera_keys: set[str] = set()
    skipped = 0
    action_dim: int | None = None
    pending_wrist_tasks: list[PendingWristVideoTask] = []

    for sample_index in range(len(dataset)):
        record = dataset.records[sample_index]
        sample = dataset[sample_index]
        episode_frames, global_index_offset, action_dim, episode_pending_wrist = _write_generic_episode(
            spec=spec,
            args=args,
            resolution_config=resolution_config,
            record=record,
            sample=sample,
            data_dir=data_dir,
            videos_root=videos_root,
            source_meta_dir=source_meta_dir,
            task_to_index=task_to_index,
            global_index_offset=global_index_offset,
            output_episode_index=output_episode_index,
            written_camera_keys=written_camera_keys,
            expected_action_dim=action_dim,
        )
        if episode_frames <= 0:
            skipped += 1
            continue
        pending_wrist_tasks.extend(episode_pending_wrist)
        output_episode_index += 1

    if output_episode_index <= 0:
        raise RuntimeError(f"dataset={spec.name} produced no converted episodes")

    assert action_dim is not None
    state_action_names = (
        list(WXYZ_DUAL_ARM_STATE_ACTION_NAMES)
        if spec.name in {"robomind", "robocoin"} and int(action_dim) == len(WXYZ_DUAL_ARM_STATE_ACTION_NAMES)
        else _generic_feature_names(int(action_dim))
    )
    result = _rebuild_repo_meta(
        args=args,
        resolution_config=resolution_config,
        repo_dir=repo_dir,
        robot_type=str(spec.robot_type),
        source_name=str(spec.name),
        source_suffix=str(spec.source_suffix),
        action_names=state_action_names,
        state_names=state_action_names,
        camera_keys=sorted(written_camera_keys),
        include_image_stats=not bool(args.skip_image_stats),
    )
    _save_pending_wrist_tasks(repo_dir, pending_wrist_tasks)
    return {
        "dataset": spec.name,
        "repo_dir": str(result.repo_dir),
        "episodes": int(result.total_episodes),
        "frames": int(result.total_frames),
        "skipped": int(skipped),
        "pending_wrist_videos": int(len(pending_wrist_tasks)),
        "resumed_wrist_videos": 0,
    }


def _astribot_episode_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    tail = stem.rsplit("_episode_", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return (10**9, path.name)
    return (int(tail[1]), path.name)


def _astribot_iter_episode_paths(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*.hdf5"), key=_astribot_episode_sort_key)


def _read_astribot_episode_list(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    if not entries:
        raise ValueError(f"No astribot hdf5 entries found in {path}")
    return entries


def _build_astribot_hdf5_name_index(input_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in input_root.rglob("*.hdf5"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def _resolve_astribot_list_entry(
    entry: str,
    *,
    input_root: Path,
    name_index: Mapping[str, list[Path]],
) -> Path:
    raw_path = Path(entry)
    if raw_path.suffix.lower() not in {".hdf5", ".h5"}:
        raw_path = raw_path.with_suffix(".hdf5")
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(input_root / raw_path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.absolute()

    matches = list(name_index.get(raw_path.name, []))
    if len(matches) == 1:
        return matches[0].absolute()
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous astribot hdf5 entry `{entry}` matched multiple files under {input_root}: "
            f"{[str(path) for path in matches[:10]]}"
        )
    raise FileNotFoundError(f"Could not resolve astribot hdf5 entry `{entry}` under {input_root}")


def _astribot_iter_episode_paths_from_list(input_root: Path, list_path: Path) -> list[Path]:
    entries = _read_astribot_episode_list(list_path)
    name_index = _build_astribot_hdf5_name_index(input_root)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        path = _resolve_astribot_list_entry(entry, input_root=input_root, name_index=name_index)
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    return resolved


def _astribot_source_episode_key(path: Path) -> str:
    return path.resolve().name


def _astribot_source_episode_keys(path: Path) -> list[str]:
    keys = [path.name, path.resolve().name]
    return list(dict.fromkeys(key for key in keys if key))


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _safe_hardlink_symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    resolved_src = src.resolve()
    try:
        os.link(resolved_src, dst)
        return
    except OSError:
        pass
    try:
        dst.symlink_to(resolved_src)
    except OSError:
        shutil.copy2(resolved_src, dst)


def _rewrite_episode_parquet_indices(
    *,
    source_parquet: Path,
    target_parquet: Path,
    episode_index: int,
    global_index_offset: int,
    task_index: int | None = None,
) -> int:
    if pq is None or pa is None:
        raise RuntimeError("pyarrow and pyarrow.parquet are required to rewrite parquet indices")
    ensure_dir(target_parquet.parent)
    table = pq.read_table(source_parquet)
    num_rows = int(table.num_rows)
    replacements = {
        "episode_index": pa.array([int(episode_index)] * num_rows, type=pa.int64()),
        "index": pa.array(
            list(range(int(global_index_offset), int(global_index_offset) + num_rows)),
            type=pa.int64(),
        ),
    }
    if task_index is not None:
        replacements["task_index"] = pa.array([int(task_index)] * num_rows, type=pa.int64())
    for column_name, values in replacements.items():
        column_idx = table.schema.get_field_index(column_name)
        if column_idx < 0:
            raise RuntimeError(f"Missing `{column_name}` column in {source_parquet}")
        table = table.set_column(column_idx, column_name, values)
    pq.write_table(table, target_parquet)
    return num_rows


def _copy_or_link_astribot_stats(
    *,
    source_repo_dir: Path,
    target_repo_dir: Path,
    prefer_symlink: bool,
) -> None:
    source_meta_dir = source_repo_dir / "meta"
    target_meta_dir = target_repo_dir / "meta"
    for filename in ("stats.json", "norm_stat_quantiles_q01_q99.json"):
        source_path = source_meta_dir / filename
        if not source_path.exists():
            continue
        target_path = target_meta_dir / filename
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        if prefer_symlink:
            target_path.symlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, target_path)


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_reused_astribot_episode_stats(
    *,
    source_repo_dir: Path,
    target_repo_dir: Path,
    source_episode_indices: Sequence[int],
) -> None:
    source_path = source_repo_dir / "meta" / "episodes_stats.jsonl"
    if not source_path.exists():
        return
    rows_by_index = {
        int(row["episode_index"]): dict(row)
        for row in _load_jsonl(source_path)
        if isinstance(row, Mapping) and "episode_index" in row
    }
    target_rows = []
    for target_episode_index, source_episode_index in enumerate(source_episode_indices):
        row = rows_by_index.get(int(source_episode_index))
        if row is None:
            continue
        copied = dict(row)
        copied["episode_index"] = int(target_episode_index)
        target_rows.append(copied)
    if len(target_rows) == len(source_episode_indices):
        _write_jsonl(target_repo_dir / "meta" / "episodes_stats.jsonl", target_rows)


def _load_astribot_reuse_index(source_repo_dir: Path) -> dict[str, int]:
    source_meta_dir = source_repo_dir / "source_meta"
    if not source_meta_dir.exists():
        raise FileNotFoundError(f"Missing source_meta dir in reuse repo: {source_meta_dir}")
    index: dict[str, int] = {}
    for meta_path in sorted(source_meta_dir.glob("episode_*.json")):
        match = re.search(r"episode_(\d+)\.json$", meta_path.name)
        if match is None:
            continue
        payload = load_json(meta_path)
        source_meta = payload.get("source_meta", {}) if isinstance(payload, dict) else {}
        source_path = str(source_meta.get("source_path", "")).strip()
        source_episode_id = str(payload.get("source_episode_id", "")).strip()
        keys = []
        if source_path:
            keys.append(Path(source_path).name)
        if source_episode_id:
            keys.append(Path(source_episode_id).name)
        for key in keys:
            if key:
                index[key] = int(match.group(1))
    if not index:
        raise RuntimeError(f"No source episode mapping found in reuse repo: {source_repo_dir}")
    return index


def _reuse_astribot_subset_repo(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    episode_paths: Sequence[Path],
    repo_dir: Path,
    source_repo_dir: Path,
    resolution_config: ResolutionConfig,
    prefer_symlink: bool,
) -> dict[str, Any]:
    source_index = _load_astribot_reuse_index(source_repo_dir)
    data_dir, videos_root, source_meta_dir = _ensure_repo_dirs(
        repo_dir,
        overwrite=bool(args.overwrite),
        videos_dir_name=_videos_dir_name(args, resolution_config),
    )
    source_videos_root = resolve_lerobot_videos_root(source_repo_dir, must_exist=True) / "chunk-000"
    written_camera_keys: set[str] = set()
    missing: list[str] = []
    global_index_offset = 0
    reused_source_episode_indices: list[int] = []

    for output_episode_index, episode_path in enumerate(episode_paths):
        source_episode_index = None
        for key in _astribot_source_episode_keys(episode_path):
            source_episode_index = source_index.get(key)
            if source_episode_index is not None:
                break
        if source_episode_index is None:
            missing.append(str(episode_path))
            continue

        source_parquet = (
            source_repo_dir
            / "data"
            / "chunk-000"
            / f"episode_{source_episode_index:06d}.parquet"
        )
        source_meta = source_repo_dir / "source_meta" / f"episode_{source_episode_index:06d}.json"
        if not source_parquet.exists() or not source_meta.exists():
            missing.append(str(episode_path))
            continue

        num_frames = _rewrite_episode_parquet_indices(
            source_parquet=source_parquet,
            target_parquet=data_dir / f"episode_{output_episode_index:06d}.parquet",
            episode_index=output_episode_index,
            global_index_offset=global_index_offset,
        )
        global_index_offset += int(num_frames)
        _safe_symlink_or_copy(source_meta, source_meta_dir / f"episode_{output_episode_index:06d}.json")

        for source_camera_dir in sorted(source_videos_root.iterdir()):
            if not source_camera_dir.is_dir():
                continue
            source_video = source_camera_dir / f"episode_{source_episode_index:06d}.mp4"
            if not source_video.exists():
                continue
            target_video = videos_root / source_camera_dir.name / f"episode_{output_episode_index:06d}.mp4"
            _safe_hardlink_symlink_or_copy(source_video, target_video)
            written_camera_keys.add(source_camera_dir.name)
        reused_source_episode_indices.append(int(source_episode_index))

    if missing:
        raise RuntimeError(
            f"Cannot reuse {len(missing)} astribot episodes from {source_repo_dir}; "
            f"first missing: {missing[:5]}"
        )

    result = _rebuild_repo_meta(
        args=args,
        resolution_config=resolution_config,
        repo_dir=repo_dir,
        robot_type=str(spec.robot_type),
        source_name=str(spec.name),
        source_suffix=str(spec.source_suffix),
        action_names=list(ASTRIBOT_STATE_ACTION_NAMES),
        state_names=list(ASTRIBOT_STATE_ACTION_NAMES),
        camera_keys=sorted(written_camera_keys),
        include_image_stats=False,
    )
    _copy_or_link_astribot_stats(
        source_repo_dir=source_repo_dir,
        target_repo_dir=repo_dir,
        prefer_symlink=prefer_symlink,
    )
    _write_reused_astribot_episode_stats(
        source_repo_dir=source_repo_dir,
        target_repo_dir=repo_dir,
        source_episode_indices=reused_source_episode_indices,
    )
    _save_pending_wrist_tasks(repo_dir, [])
    return {
        "dataset": spec.name,
        "repo_dir": str(result.repo_dir),
        "episodes": int(result.total_episodes),
        "frames": int(result.total_frames),
        "skipped": 0,
        "pending_wrist_videos": 0,
        "resumed_wrist_videos": 0,
        "reused_from": str(source_repo_dir),
    }


def _astribot_repo_has_expected_outputs(
    *,
    repo_dir: Path,
    resolution_config: ResolutionConfig,
    args: argparse.Namespace,
    expected_episodes: int,
) -> bool:
    if expected_episodes <= 0:
        return False
    videos_root = repo_dir / _videos_dir_name(args, resolution_config) / "chunk-000"
    required_dirs = [
        repo_dir / "data" / "chunk-000",
        repo_dir / "source_meta",
        videos_root / _camera_to_output_key("main"),
    ]
    if not bool(args.main_only):
        required_dirs.extend(
            [
                videos_root / _camera_to_output_key("left_wrist"),
                videos_root / _camera_to_output_key("right_wrist"),
            ]
        )
    for directory in required_dirs:
        if not directory.exists():
            return False
        count = len(list(directory.glob("episode_*.*")))
        if count < expected_episodes:
            return False
    return True


def _astribot_load_array(
    handle: h5py.File,
    key: str,
    *,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"Missing dataset: {key}")
    arr = np.asarray(handle[key])
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _astribot_prepare_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    length = min(int(t.shape[0]), int(v.shape[0]))
    if length <= 0:
        raise ValueError("Empty time/value series")
    t = t[:length]
    v = v[:length]
    mask = np.isfinite(t) & np.all(np.isfinite(v), axis=1)
    if not np.any(mask):
        raise ValueError("No finite samples in time/value series")
    t = t[mask]
    v = v[mask]
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = v[order]
    unique_t, unique_idx = np.unique(t, return_index=True)
    if unique_t.size <= 0:
        raise ValueError("No unique timestamps after filtering")
    return unique_t, v[unique_idx]


def _astribot_interp_linear(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    t, v = _astribot_prepare_series(times, values)
    target = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if t.shape[0] == 1:
        return np.repeat(v[:1], repeats=target.shape[0], axis=0).astype(np.float32)
    out = np.stack([np.interp(target, t, v[:, dim]) for dim in range(v.shape[1])], axis=1)
    return out.astype(np.float32)


def _astribot_interp_quat_wxyz(
    times: np.ndarray,
    source_quats_xyzw: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    t, q = _astribot_prepare_series(times, source_quats_xyzw)
    if q.shape[1] != 4:
        raise ValueError(f"Quaternion series must have width 4, got {q.shape}")
    q = standardize_quaternion_xyzw(q.astype(np.float32)).astype(np.float64)
    target = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if t.shape[0] == 1:
        out = np.repeat(q[:1], repeats=target.shape[0], axis=0).astype(np.float32)
        return quaternion_xyzw_to_wxyz(out)
    clipped = np.clip(target, t[0], t[-1])
    slerp = Slerp(t, R.from_quat(q))
    out = slerp(clipped).as_quat().astype(np.float32)
    return quaternion_xyzw_to_wxyz(out)


def _astribot_nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=np.float64).reshape(-1)
    target = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if source.shape[0] <= 0:
        raise ValueError("Empty source times")
    source = np.maximum.accumulate(source)
    idx = np.searchsorted(source, target, side="left")
    idx = np.clip(idx, 0, source.shape[0] - 1)
    prev = np.clip(idx - 1, 0, source.shape[0] - 1)
    choose_prev = np.abs(target - source[prev]) <= np.abs(source[idx] - target)
    return np.where(choose_prev, prev, idx).astype(np.int64)


def _astribot_scan_gripper_spec(
    episode_paths: Sequence[Path],
    *,
    state_key: str,
    command_key: str,
    larger_is_closed: bool,
    constant_output: float,
) -> AstribotGripperNormSpec:
    del episode_paths, state_key, command_key, constant_output
    return AstribotGripperNormSpec(
        0.0,
        100.0,
        larger_is_closed=larger_is_closed,
        constant_output=None,
    )


def _astribot_normalize_gripper(
    values: np.ndarray,
    spec: AstribotGripperNormSpec,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if spec.constant_output is not None:
        return np.full_like(arr, fill_value=float(spec.constant_output), dtype=np.float32)
    arr = np.clip(arr, float(spec.minimum), float(spec.maximum))
    scale = max(float(spec.maximum - spec.minimum), 1e-6)
    out = (arr - float(spec.minimum)) / scale
    if spec.larger_is_closed:
        out = 1.0 - out
    return out.astype(np.float32)


def _astribot_build_target_times(
    *,
    common_start: float,
    common_end: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(common_start) or not np.isfinite(common_end) or common_end <= common_start:
        raise ValueError(f"Invalid shared time window: start={common_start}, end={common_end}")
    duration = max(0.0, float(common_end - common_start))
    num_frames = max(1, int(np.floor(duration * float(target_fps))) + 1)
    rel = (np.arange(num_frames, dtype=np.float64) / float(target_fps)).astype(np.float64)
    abs_times = (common_start + rel).astype(np.float64)
    return rel, abs_times


def _astribot_extract_task_text(handle: h5py.File, fallback: str) -> str:
    task_name = str(handle.attrs.get("task_name", "") or "").strip()
    return task_name or fallback


def _astribot_task_fallback_from_path(path: Path, default: str) -> str:
    stem = path.stem
    task_slug = stem.split("_episode_", 1)[0]
    task_map = {
        "centrifuge": "pick up the plate and put it on centrifuge",
        "multidrop": "pick up the plate and put it on multidrop",
    }
    return task_map.get(task_slug, default)


def _astribot_resample_state_action(
    *,
    handle: h5py.File,
    target_abs_times: np.ndarray,
    left_gripper_spec: AstribotGripperNormSpec,
    right_gripper_spec: AstribotGripperNormSpec,
) -> tuple[np.ndarray, np.ndarray]:
    state_times = _astribot_load_array(handle, "time", dtype=np.float64).reshape(-1)
    action_times = _astribot_load_array(handle, "command_poses_dict/timestamp", dtype=np.float64).reshape(-1)

    state_left_pose = _astribot_load_array(handle, "poses_dict/astribot_arm_left", dtype=np.float64)
    state_right_pose = _astribot_load_array(handle, "poses_dict/astribot_arm_right", dtype=np.float64)
    action_left_pose = _astribot_load_array(handle, "command_poses_dict/astribot_arm_left", dtype=np.float64)
    action_right_pose = _astribot_load_array(handle, "command_poses_dict/astribot_arm_right", dtype=np.float64)

    state_left_gripper_raw = _astribot_load_array(handle, "poses_dict/astribot_gripper_left", dtype=np.float64).reshape(-1)
    state_right_gripper_raw = _astribot_load_array(handle, "poses_dict/astribot_gripper_right", dtype=np.float64).reshape(-1)
    action_left_gripper_raw = _astribot_load_array(handle, "command_poses_dict/astribot_gripper_left", dtype=np.float64).reshape(-1)
    action_right_gripper_raw = _astribot_load_array(handle, "command_poses_dict/astribot_gripper_right", dtype=np.float64).reshape(-1)

    state_left_xyz = _astribot_interp_linear(state_times, state_left_pose[:, :3], target_abs_times)
    state_right_xyz = _astribot_interp_linear(state_times, state_right_pose[:, :3], target_abs_times)
    action_left_xyz = _astribot_interp_linear(action_times, action_left_pose[:, :3], target_abs_times)
    action_right_xyz = _astribot_interp_linear(action_times, action_right_pose[:, :3], target_abs_times)

    state_left_quat = _astribot_interp_quat_wxyz(
        state_times, state_left_pose[:, 3:7], target_abs_times
    )
    state_right_quat = _astribot_interp_quat_wxyz(
        state_times, state_right_pose[:, 3:7], target_abs_times
    )
    action_left_quat = _astribot_interp_quat_wxyz(
        action_times, action_left_pose[:, 3:7], target_abs_times
    )
    action_right_quat = _astribot_interp_quat_wxyz(
        action_times, action_right_pose[:, 3:7], target_abs_times
    )

    state_left_gripper = _astribot_normalize_gripper(
        _astribot_interp_linear(state_times, state_left_gripper_raw[:, None], target_abs_times)[:, 0],
        left_gripper_spec,
    )
    state_right_gripper = _astribot_normalize_gripper(
        _astribot_interp_linear(state_times, state_right_gripper_raw[:, None], target_abs_times)[:, 0],
        right_gripper_spec,
    )
    action_left_gripper = _astribot_normalize_gripper(
        _astribot_interp_linear(action_times, action_left_gripper_raw[:, None], target_abs_times)[:, 0],
        left_gripper_spec,
    )
    action_right_gripper = _astribot_normalize_gripper(
        _astribot_interp_linear(action_times, action_right_gripper_raw[:, None], target_abs_times)[:, 0],
        right_gripper_spec,
    )

    states = np.concatenate(
        [
            state_left_xyz,
            state_left_quat,
            state_left_gripper,
            state_right_xyz,
            state_right_quat,
            state_right_gripper,
        ],
        axis=1,
    ).astype(np.float32)
    actions = np.concatenate(
        [
            action_left_xyz,
            action_left_quat,
            action_left_gripper,
            action_right_xyz,
            action_right_quat,
            action_right_gripper,
        ],
        axis=1,
    ).astype(np.float32)
    return states, actions


def _astribot_stream_camera_to_mp4(
    *,
    handle: h5py.File,
    camera_name: str,
    target_abs_times: np.ndarray,
    output_path: Path,
    fps: float,
    target_height: int,
    target_width: int,
    video_codec: str,
    video_crf: int,
    video_preset: str,
) -> dict[str, Any]:
    blob = _astribot_load_array(handle, f"images_dict/{camera_name}/rgb", dtype=np.uint8).reshape(-1)
    raw_sizes = _astribot_load_array(handle, f"images_dict/{camera_name}/rgb_size", dtype=np.float64).reshape(-1)
    timestamps = _astribot_load_array(handle, f"images_dict/{camera_name}/rgb_timestamp", dtype=np.float64).reshape(-1)
    length = min(int(raw_sizes.shape[0]), int(timestamps.shape[0]))
    if length <= 0:
        raise AstribotSkipEpisodeError(f"No image frames for camera={camera_name}")
    sizes = raw_sizes[:length].astype(np.int64)
    timestamps = timestamps[:length]
    if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(timestamps)):
        raise AstribotSkipEpisodeError(f"Non-finite camera payload for camera={camera_name}")
    if np.any(sizes <= 0):
        raise AstribotSkipEpisodeError(f"Non-positive rgb_size for camera={camera_name}")
    offsets = np.zeros(length + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes, dtype=np.int64)
    if int(offsets[-1]) > int(blob.shape[0]):
        raise AstribotSkipEpisodeError(
            f"Corrupted JPEG stream for camera={camera_name}: "
            f"{int(offsets[-1])} > {int(blob.shape[0])}"
        )
    selected = _astribot_nearest_indices(timestamps, target_abs_times)

    def _iter_payloads() -> Iterator[np.ndarray]:
        for frame_idx in selected.tolist():
            start = int(offsets[frame_idx])
            end = int(offsets[frame_idx + 1])
            yield blob[start:end]

    _export_jpeg_payload_iter_to_mp4(
        payloads=_iter_payloads(),
        output_path=output_path,
        num_frames=int(selected.shape[0]),
        fps=fps,
        target_height=target_height,
        target_width=target_width,
        video_codec=video_codec,
        video_crf=video_crf,
        video_preset=video_preset,
    )
    return {
        "source_camera": camera_name,
        "source_num_frames": int(length),
        "output_num_frames": int(selected.shape[0]),
        "selected_source_frame_min": int(selected.min()),
        "selected_source_frame_max": int(selected.max()),
        "selected_unique_source_frames": int(np.unique(selected).shape[0]),
    }


def _astribot_process_episode_job(job: Mapping[str, Any]) -> dict[str, Any]:
    episode_path = Path(job["episode_path"])
    output_episode_index = int(job["output_episode_index"])
    camera_specs = list(job["camera_specs"])
    data_dir = Path(job["data_dir"])
    videos_root = Path(job["videos_root"])
    main_only = bool(job["main_only"])
    default_fps = float(job["default_fps"])
    left_gripper_spec = job["left_gripper_spec"]
    right_gripper_spec = job["right_gripper_spec"]

    with h5py.File(episode_path, "r") as handle:
        instruction = _astribot_extract_task_text(
            handle,
            fallback=_astribot_task_fallback_from_path(
                episode_path,
                str(job["input_root_name"]).replace("_", " ").strip(),
            ),
        )
        state_times = _astribot_load_array(handle, "time", dtype=np.float64).reshape(-1)
        action_times = _astribot_load_array(handle, "command_poses_dict/timestamp", dtype=np.float64).reshape(-1)
        camera_time_series = {
            camera_spec.output_key: _astribot_load_array(
                handle,
                f"images_dict/{camera_spec.source_name}/rgb_timestamp",
                dtype=np.float64,
            ).reshape(-1)
            for camera_spec in camera_specs
        }
        common_start = max(
            [float(state_times[0]), float(action_times[0])]
            + [float(series[0]) for series in camera_time_series.values()]
        )
        common_end = min(
            [float(state_times[-1]), float(action_times[-1])]
            + [float(series[-1]) for series in camera_time_series.values()]
        )
        timestamps, target_abs_times = _astribot_build_target_times(
            common_start=common_start,
            common_end=common_end,
            target_fps=default_fps,
        )
        if timestamps.shape[0] <= 1:
            raise AstribotSkipEpisodeError(
                f"Episode {episode_path.name} collapsed to <=1 frame after time alignment"
            )
        states, actions = _astribot_resample_state_action(
            handle=handle,
            target_abs_times=target_abs_times,
            left_gripper_spec=left_gripper_spec,
            right_gripper_spec=right_gripper_spec,
        )
        num_frames = int(min(states.shape[0], actions.shape[0], timestamps.shape[0]))
        if num_frames <= 0:
            raise AstribotSkipEpisodeError(f"Episode {episode_path.name} has no aligned frames")

        written_camera_keys: list[str] = []
        pending_wrist_tasks: list[PendingWristVideoTask] = []
        for camera_spec in camera_specs:
            output_path = (
                videos_root
                / camera_spec.output_key
                / f"episode_{output_episode_index:06d}.mp4"
            )
            if main_only and _is_wrist_camera(camera_spec.camera_key):
                pending_wrist_tasks.append(
                    PendingWristVideoTask(
                        output_relpath=str(output_path.relative_to(Path(job["repo_dir"]))),
                        source=format_virtual_video_path(
                            episode_path,
                            f"images_dict/{camera_spec.source_name}/rgb",
                        ),
                        num_frames=int(num_frames),
                        fps=default_fps,
                        camera_key=str(camera_spec.camera_key),
                        episode_index=output_episode_index,
                    )
                )
                continue
            _astribot_stream_camera_to_mp4(
                handle=handle,
                camera_name=camera_spec.source_name,
                target_abs_times=target_abs_times[:num_frames],
                output_path=output_path,
                fps=default_fps,
                target_height=int(camera_spec.image_height),
                target_width=int(camera_spec.image_width),
                video_codec=str(job["video_codec"]),
                video_crf=int(job["video_crf"]),
                video_preset=str(job["video_preset"]),
            )
            written_camera_keys.append(camera_spec.output_key)

        parquet_rows = _episode_parquet_rows(
            episode_index=output_episode_index,
            task_index=0,
            states=states[:num_frames],
            actions=actions[:num_frames],
            timestamps=timestamps[:num_frames].astype(np.float32),
            global_index_offset=0,
        )
        parquet_path = data_dir / f"episode_{output_episode_index:06d}.parquet"
        write_episode_parquet(parquet_path, parquet_rows)

        source_meta = {
            "task": instruction,
            "source_name": str(job["spec_name"]),
            "source_episode_id": str(episode_path.name),
            "action_config": [
                {
                    "start_frame": 0,
                    "end_frame": int(num_frames),
                    "action_text": instruction,
                    "skill": "",
                }
            ],
            "source_meta": {
                "dataset_name": str(job["spec_name"]),
                "source_path": str(episode_path),
                "camera_mapping": {
                    camera_spec.output_key: camera_spec.source_name
                    for camera_spec in camera_specs
                    if (not main_only) or _is_main_camera(camera_spec.camera_key)
                },
                "deferred_camera_mapping": {
                    camera_spec.output_key: camera_spec.source_name
                    for camera_spec in camera_specs
                    if main_only and _is_wrist_camera(camera_spec.camera_key)
                },
                "timestamp_alignment": {
                    "state": "time",
                    "action": "command_poses_dict/timestamp",
                    "videos": {
                        camera_spec.output_key: f"images_dict/{camera_spec.source_name}/rgb_timestamp"
                        for camera_spec in camera_specs
                    },
                    "common_start": float(common_start),
                    "common_end": float(common_end),
                    "target_fps": default_fps,
                },
                "gripper_normalization": {
                    "left": {
                        "minimum": float(left_gripper_spec.minimum),
                        "maximum": float(left_gripper_spec.maximum),
                        "larger_is_closed": bool(left_gripper_spec.larger_is_closed),
                        "constant_output": None
                        if left_gripper_spec.constant_output is None
                        else float(left_gripper_spec.constant_output),
                    },
                    "right": {
                        "minimum": float(right_gripper_spec.minimum),
                        "maximum": float(right_gripper_spec.maximum),
                        "larger_is_closed": bool(right_gripper_spec.larger_is_closed),
                        "constant_output": None
                        if right_gripper_spec.constant_output is None
                        else float(right_gripper_spec.constant_output),
                    },
                },
                "frame_sample_stride": FRAME_SAMPLE_STRIDE,
                "sampled_frame_ids": _sampled_frame_ids(num_frames),
                "state_action_layout": list(ASTRIBOT_STATE_ACTION_NAMES),
                "notes": [
                    "Astribot videos are aligned onto a shared target timeline using nearest camera frames.",
                    "Astribot states/actions are linearly interpolated in xyz and slerped in quaternion space.",
                ],
            },
        }

    return {
        "episode_index": output_episode_index,
        "num_frames": int(num_frames),
        "source_meta": source_meta,
        "written_camera_keys": written_camera_keys,
        "pending_wrist_tasks": pending_wrist_tasks,
    }


def _scan_astribot_resolution_config(
    *,
    args: argparse.Namespace,
    episode_paths: Sequence[Path],
    camera_specs: Sequence[AstribotCameraSpec],
) -> ResolutionConfig:
    base = _resolution_config_from_args(args)
    if not bool(args.keep_original_resolution):
        return base
    scanned: dict[str, tuple[int, int]] = {}
    for episode_path in episode_paths:
        with h5py.File(episode_path, "r") as handle:
            for camera_spec in camera_specs:
                rgb_key = f"images_dict/{camera_spec.source_name}/rgb"
                size_key = f"{rgb_key}_size"
                if rgb_key not in handle or size_key not in handle:
                    continue
                height, width = _probe_virtual_hdf5_resolution(episode_path, rgb_key)
                prev = scanned.get(camera_spec.camera_key)
                if prev is None:
                    scanned[camera_spec.camera_key] = (int(height), int(width))
                else:
                    scanned[camera_spec.camera_key] = (
                        max(int(prev[0]), int(height)),
                        max(int(prev[1]), int(width)),
                    )
    return _merge_resolution_config(base, scanned)


def _convert_astribot_dataset(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    repo_dir = _dataset_repo_root(args.output_root, spec)
    if bool(args.resume_wrist):
        resolution_config = (
            _resolve_existing_repo_resolution_config(repo_dir)
            or _resolution_config_from_args(args)
        )
        result = _resume_pending_wrist_tasks(
            args=args,
            repo_dir=repo_dir,
            resolution_config=resolution_config,
        )
        action_names, state_names = _load_existing_repo_feature_names(repo_dir)
        rebuilt = _rebuild_repo_meta(
            args=args,
            resolution_config=resolution_config,
            repo_dir=repo_dir,
            robot_type=str(spec.robot_type),
            source_name=str(spec.name),
            source_suffix=str(spec.source_suffix),
            action_names=action_names,
            state_names=state_names,
            camera_keys=result["camera_keys"] or None,
            include_image_stats=not bool(args.skip_image_stats),
        )
        return {
            "dataset": spec.name,
            "repo_dir": str(rebuilt.repo_dir),
            "episodes": int(rebuilt.total_episodes),
            "frames": int(rebuilt.total_frames),
            "skipped": 0,
            "pending_wrist_videos": 0,
            "resumed_wrist_videos": int(result["resumed_wrist_videos"]),
        }

    base_main_height, base_main_width = _main_resolution(args)
    base_wrist_height, base_wrist_width = _wrist_resolution(args)
    base_camera_specs = [
        AstribotCameraSpec(
            camera_key="main",
            source_name="head",
            output_key=_camera_to_output_key("main"),
            image_height=base_main_height,
            image_width=base_main_width,
        ),
        AstribotCameraSpec(
            camera_key="left_wrist",
            source_name="left",
            output_key=_camera_to_output_key("left_wrist"),
            image_height=base_wrist_height,
            image_width=base_wrist_width,
        ),
        AstribotCameraSpec(
            camera_key="right_wrist",
            source_name="right",
            output_key=_camera_to_output_key("right_wrist"),
            image_height=base_wrist_height,
            image_width=base_wrist_width,
        ),
    ]
    if args.astribot_episode_list is not None:
        episode_paths = _astribot_iter_episode_paths_from_list(
            Path(spec.input_root),
            Path(args.astribot_episode_list),
        )
    else:
        episode_paths = _astribot_iter_episode_paths(Path(spec.input_root))
    if int(args.max_episodes) > 0:
        episode_paths = episode_paths[: int(args.max_episodes)]
    if not episode_paths:
        raise RuntimeError(f"dataset={spec.name} found no .hdf5 episodes under {spec.input_root}")

    resolution_config = _scan_astribot_resolution_config(
        args=args,
        episode_paths=episode_paths,
        camera_specs=base_camera_specs,
    )
    if args.astribot_reuse_from is not None:
        return _reuse_astribot_subset_repo(
            spec=spec,
            args=args,
            episode_paths=episode_paths,
            repo_dir=repo_dir,
            source_repo_dir=Path(args.astribot_reuse_from).resolve(),
            resolution_config=resolution_config,
            prefer_symlink=bool(args.astribot_link_stats),
        )

    if not bool(args.overwrite) and _astribot_repo_has_expected_outputs(
        repo_dir=repo_dir,
        resolution_config=resolution_config,
        args=args,
        expected_episodes=len(episode_paths),
    ):
        result = _rebuild_repo_meta(
            args=args,
            resolution_config=resolution_config,
            repo_dir=repo_dir,
            robot_type=str(spec.robot_type),
            source_name=str(spec.name),
            source_suffix=str(spec.source_suffix),
            action_names=list(ASTRIBOT_STATE_ACTION_NAMES),
            state_names=list(ASTRIBOT_STATE_ACTION_NAMES),
            camera_keys=None,
            include_image_stats=not bool(args.skip_image_stats),
        )
        _save_pending_wrist_tasks(repo_dir, [])
        return {
            "dataset": spec.name,
            "repo_dir": str(result.repo_dir),
            "episodes": int(result.total_episodes),
            "frames": int(result.total_frames),
            "skipped": 0,
            "pending_wrist_videos": 0,
            "resumed_wrist_videos": 0,
            "reused_existing_outputs": True,
        }

    data_dir, videos_root, source_meta_dir = _ensure_repo_dirs(
        repo_dir,
        overwrite=bool(args.overwrite),
        videos_dir_name=_videos_dir_name(args, resolution_config),
    )
    main_height, main_width = _main_resolution(args, resolution_config)
    wrist_height, wrist_width = _wrist_resolution(args, resolution_config)

    camera_specs = [
        AstribotCameraSpec(
            camera_key="main",
            source_name="head",
            output_key=_camera_to_output_key("main"),
            image_height=main_height,
            image_width=main_width,
        ),
        AstribotCameraSpec(
            camera_key="left_wrist",
            source_name="left",
            output_key=_camera_to_output_key("left_wrist"),
            image_height=wrist_height,
            image_width=wrist_width,
        ),
        AstribotCameraSpec(
            camera_key="right_wrist",
            source_name="right",
            output_key=_camera_to_output_key("right_wrist"),
            image_height=wrist_height,
            image_width=wrist_width,
        ),
    ]

    left_gripper_spec = _astribot_scan_gripper_spec(
        episode_paths,
        state_key="poses_dict/astribot_gripper_left",
        command_key="command_poses_dict/astribot_gripper_left",
        larger_is_closed=True,
        constant_output=1.0,
    )
    right_gripper_spec = _astribot_scan_gripper_spec(
        episode_paths,
        state_key="poses_dict/astribot_gripper_right",
        command_key="command_poses_dict/astribot_gripper_right",
        larger_is_closed=True,
        constant_output=1.0,
    )

    if int(args.num_workers) > 1:
        written_camera_keys: set[str] = set()
        pending_wrist_tasks: list[PendingWristVideoTask] = []
        completed: dict[int, dict[str, Any]] = {}
        skipped = 0
        jobs = [
            {
                "episode_path": str(episode_path),
                "output_episode_index": int(index),
                "data_dir": str(data_dir),
                "videos_root": str(videos_root),
                "repo_dir": str(repo_dir),
                "input_root_name": Path(spec.input_root).name,
                "spec_name": spec.name,
                "default_fps": float(spec.default_fps),
                "camera_specs": camera_specs,
                "left_gripper_spec": left_gripper_spec,
                "right_gripper_spec": right_gripper_spec,
                "main_only": bool(args.main_only),
                "video_codec": str(args.video_codec),
                "video_crf": int(args.video_crf),
                "video_preset": str(args.video_preset),
            }
            for index, episode_path in enumerate(episode_paths)
        ]
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_path = {
                executor.submit(_astribot_process_episode_job, job): Path(job["episode_path"])
                for job in jobs
            }
            progress = tqdm(
                total=len(future_to_path),
                desc=f"Convert Astribot {spec.output_name}",
                unit="episode",
            )
            for future in as_completed(future_to_path):
                episode_path = future_to_path[future]
                try:
                    result_row = future.result()
                    completed[int(result_row["episode_index"])] = result_row
                except (AstribotSkipEpisodeError, KeyError, ValueError, IndexError) as exc:
                    skipped += 1
                    progress.write(f"[WARN] skip astribot episode={episode_path.name}: {exc}")
                finally:
                    progress.update(1)
            progress.close()

        if not completed:
            raise RuntimeError(f"dataset={spec.name} produced no converted episodes")

        global_index_offset = 0
        compact_episode_index = 0
        task_to_index: dict[str, int] = {}
        for original_episode_index in sorted(completed.keys()):
            result_row = completed[original_episode_index]
            source_meta = dict(result_row["source_meta"])
            instruction = str(source_meta.get("task", ""))
            task_index = task_to_index.setdefault(instruction, len(task_to_index))
            original_parquet = data_dir / f"episode_{original_episode_index:06d}.parquet"
            compact_parquet = data_dir / f"episode_{compact_episode_index:06d}.parquet"
            _rewrite_episode_parquet_indices(
                source_parquet=original_parquet,
                target_parquet=compact_parquet,
                episode_index=compact_episode_index,
                global_index_offset=global_index_offset,
                task_index=task_index,
            )
            if original_episode_index != compact_episode_index and original_parquet.exists():
                original_parquet.unlink()

            source_meta["source_meta"] = dict(source_meta.get("source_meta", {}))
            dump_json(source_meta_dir / f"episode_{compact_episode_index:06d}.json", source_meta)

            for camera_key in result_row["written_camera_keys"]:
                written_camera_keys.add(str(camera_key))
                original_video = videos_root / str(camera_key) / f"episode_{original_episode_index:06d}.mp4"
                compact_video = videos_root / str(camera_key) / f"episode_{compact_episode_index:06d}.mp4"
                if original_episode_index != compact_episode_index and original_video.exists():
                    original_video.rename(compact_video)

            for task in result_row["pending_wrist_tasks"]:
                pending_wrist_tasks.append(
                    replace(task, episode_index=int(compact_episode_index))
                )
            global_index_offset += int(result_row["num_frames"])
            compact_episode_index += 1

        result = _rebuild_repo_meta(
            args=args,
            resolution_config=resolution_config,
            repo_dir=repo_dir,
            robot_type=str(spec.robot_type),
            source_name=str(spec.name),
            source_suffix=str(spec.source_suffix),
            action_names=list(ASTRIBOT_STATE_ACTION_NAMES),
            state_names=list(ASTRIBOT_STATE_ACTION_NAMES),
            camera_keys=sorted(written_camera_keys),
            include_image_stats=not bool(args.skip_image_stats),
        )
        _save_pending_wrist_tasks(repo_dir, pending_wrist_tasks)
        return {
            "dataset": spec.name,
            "repo_dir": str(result.repo_dir),
            "episodes": int(result.total_episodes),
            "frames": int(result.total_frames),
            "skipped": int(skipped),
            "pending_wrist_videos": int(len(pending_wrist_tasks)),
            "resumed_wrist_videos": 0,
        }

    task_to_index: dict[str, int] = {}
    global_index_offset = 0
    output_episode_index = 0
    skipped = 0
    written_camera_keys: set[str] = set()
    pending_wrist_tasks: list[PendingWristVideoTask] = []

    for episode_path in episode_paths:
        try:
            with h5py.File(episode_path, "r") as handle:
                instruction = _astribot_extract_task_text(
                    handle,
                    fallback=_astribot_task_fallback_from_path(
                        episode_path,
                        Path(spec.input_root).name.replace("_", " ").strip(),
                    ),
                )
                state_times = _astribot_load_array(handle, "time", dtype=np.float64).reshape(-1)
                action_times = _astribot_load_array(handle, "command_poses_dict/timestamp", dtype=np.float64).reshape(-1)
                camera_time_series = {
                    camera_spec.output_key: _astribot_load_array(
                        handle,
                        f"images_dict/{camera_spec.source_name}/rgb_timestamp",
                        dtype=np.float64,
                    ).reshape(-1)
                    for camera_spec in camera_specs
                }
                common_start = max(
                    [float(state_times[0]), float(action_times[0])]
                    + [float(series[0]) for series in camera_time_series.values()]
                )
                common_end = min(
                    [float(state_times[-1]), float(action_times[-1])]
                    + [float(series[-1]) for series in camera_time_series.values()]
                )
                timestamps, target_abs_times = _astribot_build_target_times(
                    common_start=common_start,
                    common_end=common_end,
                    target_fps=float(spec.default_fps),
                )
                if timestamps.shape[0] <= 1:
                    raise AstribotSkipEpisodeError(
                        f"Episode {episode_path.name} collapsed to <=1 frame after time alignment"
                    )
                states, actions = _astribot_resample_state_action(
                    handle=handle,
                    target_abs_times=target_abs_times,
                    left_gripper_spec=left_gripper_spec,
                    right_gripper_spec=right_gripper_spec,
                )
                num_frames = int(min(states.shape[0], actions.shape[0], timestamps.shape[0]))
                if num_frames <= 0:
                    raise AstribotSkipEpisodeError(f"Episode {episode_path.name} has no aligned frames")

                for camera_spec in camera_specs:
                    output_path = (
                        videos_root
                        / camera_spec.output_key
                        / f"episode_{output_episode_index:06d}.mp4"
                    )
                    if bool(args.main_only) and _is_wrist_camera(camera_spec.camera_key):
                        pending_wrist_tasks.append(
                            PendingWristVideoTask(
                                output_relpath=str(output_path.relative_to(repo_dir)),
                                source=format_virtual_video_path(
                                    episode_path,
                                    f"images_dict/{camera_spec.source_name}/rgb",
                                ),
                                num_frames=int(num_frames),
                                fps=float(spec.default_fps),
                                camera_key=str(camera_spec.camera_key),
                                episode_index=int(output_episode_index),
                            )
                        )
                        continue
                    _astribot_stream_camera_to_mp4(
                        handle=handle,
                        camera_name=camera_spec.source_name,
                        target_abs_times=target_abs_times[:num_frames],
                        output_path=output_path,
                        fps=float(spec.default_fps),
                        target_height=int(camera_spec.image_height),
                        target_width=int(camera_spec.image_width),
                        video_codec=str(args.video_codec),
                        video_crf=int(args.video_crf),
                        video_preset=str(args.video_preset),
                    )
                    written_camera_keys.add(camera_spec.output_key)

                task_index = task_to_index.setdefault(instruction, len(task_to_index))
                parquet_rows = _episode_parquet_rows(
                    episode_index=output_episode_index,
                    task_index=task_index,
                    states=states[:num_frames],
                    actions=actions[:num_frames],
                    timestamps=timestamps[:num_frames].astype(np.float32),
                    global_index_offset=global_index_offset,
                )
                parquet_path = data_dir / f"episode_{output_episode_index:06d}.parquet"
                write_episode_parquet(parquet_path, parquet_rows)

                source_meta = {
                    "task": instruction,
                    "source_name": spec.name,
                    "source_episode_id": str(episode_path.name),
                    "action_config": [
                        {
                            "start_frame": 0,
                            "end_frame": int(num_frames),
                            "action_text": instruction,
                            "skill": "",
                        }
                    ],
                    "source_meta": {
                        "dataset_name": spec.name,
                        "source_path": str(episode_path),
                        "camera_mapping": {
                            camera_spec.output_key: camera_spec.source_name
                            for camera_spec in camera_specs
                            if (not bool(args.main_only)) or _is_main_camera(camera_spec.camera_key)
                        },
                        "deferred_camera_mapping": {
                            camera_spec.output_key: camera_spec.source_name
                            for camera_spec in camera_specs
                            if bool(args.main_only) and _is_wrist_camera(camera_spec.camera_key)
                        },
                        "timestamp_alignment": {
                            "state": "time",
                            "action": "command_poses_dict/timestamp",
                            "videos": {
                                camera_spec.output_key: f"images_dict/{camera_spec.source_name}/rgb_timestamp"
                                for camera_spec in camera_specs
                            },
                            "common_start": float(common_start),
                            "common_end": float(common_end),
                            "target_fps": float(spec.default_fps),
                        },
                        "gripper_normalization": {
                            "left": {
                                "minimum": float(left_gripper_spec.minimum),
                                "maximum": float(left_gripper_spec.maximum),
                                "larger_is_closed": bool(left_gripper_spec.larger_is_closed),
                                "constant_output": None
                                if left_gripper_spec.constant_output is None
                                else float(left_gripper_spec.constant_output),
                            },
                            "right": {
                                "minimum": float(right_gripper_spec.minimum),
                                "maximum": float(right_gripper_spec.maximum),
                                "larger_is_closed": bool(right_gripper_spec.larger_is_closed),
                                "constant_output": None
                                if right_gripper_spec.constant_output is None
                                else float(right_gripper_spec.constant_output),
                            },
                        },
                        "frame_sample_stride": FRAME_SAMPLE_STRIDE,
                        "sampled_frame_ids": _sampled_frame_ids(num_frames),
                        "state_action_layout": list(ASTRIBOT_STATE_ACTION_NAMES),
                        "notes": [
                            "Astribot videos are aligned onto a shared target timeline using nearest camera frames.",
                            "Astribot states/actions are linearly interpolated in xyz and slerped in quaternion space.",
                        ],
                    },
                }
                dump_json(source_meta_dir / f"episode_{output_episode_index:06d}.json", source_meta)

                global_index_offset += int(num_frames)
                output_episode_index += 1
        except (AstribotSkipEpisodeError, KeyError, ValueError, IndexError) as exc:
            skipped += 1
            print(f"[WARN] skip astribot episode={episode_path.name}: {exc}")

    if output_episode_index <= 0:
        raise RuntimeError(f"dataset={spec.name} produced no converted episodes")

    result = _rebuild_repo_meta(
        args=args,
        resolution_config=resolution_config,
        repo_dir=repo_dir,
        robot_type=str(spec.robot_type),
        source_name=str(spec.name),
        source_suffix=str(spec.source_suffix),
        action_names=list(ASTRIBOT_STATE_ACTION_NAMES),
        state_names=list(ASTRIBOT_STATE_ACTION_NAMES),
        camera_keys=sorted(written_camera_keys),
        include_image_stats=not bool(args.skip_image_stats),
    )
    _save_pending_wrist_tasks(repo_dir, pending_wrist_tasks)
    return {
        "dataset": spec.name,
        "repo_dir": str(result.repo_dir),
        "episodes": int(result.total_episodes),
        "frames": int(result.total_frames),
        "skipped": int(skipped),
        "pending_wrist_videos": int(len(pending_wrist_tasks)),
        "resumed_wrist_videos": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw curation datasets into simplified LeRobot repos with data/meta/videos only. "
            "Latent .pth generation is intentionally skipped."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated subset from: astribot,bridge,droid,libero,robocoin,robomind,robotwin,rt1,songling",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--input-roots",
        type=str,
        default="",
        help=(
            "Optional dataset input root overrides as comma-separated "
            "`dataset=/abs/path` pairs."
        ),
    )
    parser.add_argument(
        "--output-names",
        type=str,
        default="",
        help=(
            "Optional dataset output repo name overrides as comma-separated "
            "`dataset=repo_name` pairs."
        ),
    )
    parser.add_argument(
        "--astribot-episode-list",
        type=Path,
        default=None,
        help=(
            "Optional txt file selecting Astribot HDF5 episodes. Entries may be absolute "
            "paths, paths relative to the Astribot input root, or bare file names."
        ),
    )
    parser.add_argument(
        "--astribot-reuse-from",
        type=Path,
        default=None,
        help=(
            "Optional existing Astribot LeRobot repo to reuse for a subset conversion. "
            "Videos and source_meta are linked, parquet index columns are rewritten, "
            "and dataset-level stats are copied or linked from this repo."
        ),
    )
    parser.add_argument(
        "--astribot-link-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --astribot-reuse-from is used, symlink dataset-level stats from the source repo.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Per-dataset episode cap for debugging. 0 means all.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Episode-level worker count for datasets that support parallel conversion.",
    )
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", type=str, default="medium")
    parser.add_argument("--main-height", type=int, default=256)
    parser.add_argument("--main-width", type=int, default=320)
    parser.add_argument("--wrist-height", type=int, default=128)
    parser.add_argument("--wrist-width", type=int, default=160)
    parser.add_argument(
        "--keep-original-resolution",
        action="store_true",
        help=(
            "Use the maximum source resolution found per repo for main and wrist views "
            "instead of the fixed --main-* / --wrist-* defaults."
        ),
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help=(
            "Only export main-view videos now. Wrist-view exports are deferred into "
            "a pending task file under each output repo."
        ),
    )
    parser.add_argument(
        "--resume-wrist",
        action="store_true",
        help=(
            "Resume deferred wrist-view exports from the pending task files and rebuild repo metadata."
        ),
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="opencv",
        help="Only used for dataset initialization compatibility. Video conversion always uses ffmpeg.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-image-stats",
        action="store_true",
        help="Skip decoding output videos again while rebuilding metadata stats.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Default writes to <output-root>/curation_convert_summary.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    # import debugpy; debugpy.listen(5678); print("Waiting for debugger attach..."); debugpy.wait_for_client()
    args = parse_args()
    if bool(args.main_only) and bool(args.resume_wrist):
        raise ValueError("--main-only and --resume-wrist are mutually exclusive")
    for field_name in ("main_height", "main_width", "wrist_height", "wrist_width"):
        if int(getattr(args, field_name)) <= 0:
            raise ValueError(f"{field_name} must be positive, got {getattr(args, field_name)}")
    selected_specs = _resolve_specs(_parse_csv(args.datasets))
    selected_specs = _apply_input_root_overrides(
        selected_specs,
        _parse_dataset_path_overrides(args.input_roots),
    )
    selected_specs = _apply_output_name_overrides(
        selected_specs,
        _parse_dataset_string_overrides(args.output_names, option_name="--output-names"),
    )
    ensure_dir(Path(args.output_root))
    summary_path = args.summary_path or (Path(args.output_root) / "curation_convert_summary.json")

    summary_rows: list[dict[str, Any]] = []
    for spec in selected_specs:
        print(f"[INFO] convert dataset={spec.name} input_root={spec.input_root}")
        row: dict[str, Any] = {
            "dataset": spec.name,
            "input_root": str(spec.input_root),
            "output_root": str(_summary_output_root(args.output_root, spec)),
            "status": "pending",
        }
        try:
            if spec.mode == "astribot":
                result = _convert_astribot_dataset(spec=spec, args=args)
            else:
                result = _convert_generic_dataset(spec=spec, args=args)
            row.update({"status": "ok", **result})
            print(
                f"[OK] dataset={spec.name} repo={row['repo_dir']} "
                f"episodes={row['episodes']} frames={row['frames']}"
            )
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
            print(f"[ERROR] dataset={spec.name} error={exc}")
            if args.strict:
                summary_rows.append(row)
                dump_json(summary_path, summary_rows)
                raise
        summary_rows.append(row)

    dump_json(summary_path, summary_rows)
    print(f"[INFO] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
