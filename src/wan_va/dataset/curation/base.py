from __future__ import annotations

import json
import os
import pickle
import re
import sys
import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


CAMERA_KEY_ORDER = (
    "main",
    "left_wrist",
    "right_wrist",
    "left_main",
    "right_main",
    "top_main",
)
CAMERA_KEY_SET = set(CAMERA_KEY_ORDER)
LEGACY_VIDEOS_DIR_NAME = "videos"


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class GripperNormSpec:
    minimum: float
    maximum: float
    larger_is_closed: bool
    constant_output: float | None = None


class BaseRawCurationDataset(Dataset, ABC):
    def __init__(
        self,
        input_root: str | os.PathLike[str],
        split: str | Sequence[str] | None = None,
        task_filter: str | Sequence[str] | None = None,
        max_episodes: int = 0,
        video_backend: str = "opencv",
        camera_mapping: Mapping[str, str] | None = None,
        return_video_path: bool = False,
    ) -> None:
        self.input_root = Path(input_root).expanduser().resolve()
        if not self.input_root.exists():
            raise FileNotFoundError(f"input_root does not exist: {self.input_root}")

        self.splits = _normalize_tokens(split)
        self.task_filters = tuple(token.lower() for token in _normalize_tokens(task_filter))
        self.max_episodes = int(max_episodes)
        self.video_backend = str(video_backend)
        self.camera_mapping = dict(camera_mapping or {})
        self.return_video_path = bool(return_video_path)
        unknown_camera_keys = sorted(set(self.camera_mapping) - CAMERA_KEY_SET)
        if unknown_camera_keys:
            raise ValueError(
                f"Unknown camera_mapping keys: {unknown_camera_keys}. "
                f"Expected subset of {sorted(CAMERA_KEY_SET)}"
            )

        self.records = self._build_index()
        if self.max_episodes > 0:
            self.records = self.records[: self.max_episodes]

    @abstractmethod
    def _build_index(self) -> list[EpisodeRecord]:
        raise NotImplementedError

    @abstractmethod
    def _load_record(self, record: EpisodeRecord) -> dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._load_record(self.records[index])
        required_keys = {"instruction", "raw_absolute_actions", "video_frames"}
        optional_keys = {"raw_states", "raw_qpos"}
        sample_keys = set(sample.keys())
        if not required_keys.issubset(sample_keys):
            raise ValueError(
                "Each sample must contain at least: "
                "video_frames, raw_absolute_actions, instruction"
            )
        unexpected_keys = sample_keys - required_keys - optional_keys
        if unexpected_keys:
            raise ValueError(
                f"Unexpected sample keys: {sorted(unexpected_keys)}. "
                "Only `raw_states` and `raw_qpos` are allowed as optional extra keys."
            )
        return sample

    def _match_task_filters(self, *texts: str) -> bool:
        if not self.task_filters:
            return True
        lowered = [str(text).lower() for text in texts if str(text).strip()]
        if not lowered:
            return False
        return any(token in text for token in self.task_filters for text in lowered)

    def _finalize_sample(
        self,
        *,
        video_frames: Mapping[str, Any],
        raw_absolute_actions: np.ndarray,
        raw_states: np.ndarray | None = None,
        raw_qpos: np.ndarray | None = None,
        instruction: str,
    ) -> dict[str, Any]:
        ordered_frames: OrderedDict[str, Any] = OrderedDict()
        for key in CAMERA_KEY_ORDER:
            if key not in video_frames:
                continue
            if self.return_video_path:
                value = video_frames[key]
                if not isinstance(value, (str, os.PathLike)):
                    raise ValueError(
                        f"video_frames[{key}] must be a path string when "
                        f"return_video_path=True, got type={type(value).__name__}"
                    )
                text = str(value).strip()
                if not text:
                    raise ValueError(f"video_frames[{key}] is empty")
                ordered_frames[key] = text
                continue

            arr = np.asarray(video_frames[key])
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError(
                    f"video_frames[{key}] must have shape [T,H,W,3], got {arr.shape}"
                )
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            ordered_frames[key] = np.ascontiguousarray(arr)

        if not ordered_frames:
            raise ValueError("video_frames is empty")

        actions = np.asarray(raw_absolute_actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"raw_absolute_actions must have shape [T,D], got {actions.shape}"
            )

        states = None
        if raw_states is not None:
            states = np.asarray(raw_states, dtype=np.float32)
            if states.ndim != 2:
                raise ValueError(
                    f"raw_states must have shape [T,D], got {states.shape}"
                )
            if states.shape[0] != actions.shape[0]:
                raise ValueError(
                    "raw_states and raw_absolute_actions must share the same time dimension, "
                    f"got {states.shape[0]} and {actions.shape[0]}"
                )

        qpos = None
        if raw_qpos is not None:
            qpos = np.asarray(raw_qpos, dtype=np.float32)
            if qpos.ndim != 2:
                raise ValueError(
                    f"raw_qpos must have shape [T,D], got {qpos.shape}"
                )
            if qpos.shape[0] != actions.shape[0]:
                raise ValueError(
                    "raw_qpos and raw_absolute_actions must share the same time dimension, "
                    f"got {qpos.shape[0]} and {actions.shape[0]}"
                )

        sample = {
            "video_frames": ordered_frames,
            "raw_absolute_actions": np.ascontiguousarray(actions),
            "instruction": str(instruction).strip(),
        }
        if states is not None:
            sample["raw_states"] = np.ascontiguousarray(states)
        if qpos is not None:
            sample["raw_qpos"] = np.ascontiguousarray(qpos)
        return sample


def format_virtual_video_path(container_path: Path, inner_path: str) -> str:
    return f"{container_path}::{inner_path}"


def _normalize_tokens(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return tuple(item for item in items if item)


def natural_sort_key(value: str | os.PathLike[str]) -> tuple[Any, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def sort_paths_natural(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: natural_sort_key(path.name))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lerobot_videos_dir_name(
    *,
    main_height: int,
    main_width: int,
    wrist_height: int,
    wrist_width: int,
) -> str:
    return f"videos_{int(main_height)}x{int(main_width)}_{int(wrist_height)}x{int(wrist_width)}"


def _find_named_videos_dir(repo_root: Path) -> Path | None:
    candidates = sorted(
        (
            path for path in repo_root.iterdir()
            if path.is_dir() and path.name.startswith("videos_")
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        return None
    return candidates[0]


def resolve_lerobot_videos_root(
    repo_root: Path,
    *,
    must_exist: bool = True,
) -> Path:
    preferred = _find_named_videos_dir(repo_root)
    legacy = repo_root / LEGACY_VIDEOS_DIR_NAME
    if preferred is not None:
        return preferred
    if legacy.exists():
        return legacy
    if must_exist:
        raise FileNotFoundError(
            f"Could not find videos root under {repo_root}. "
            f"Tried `videos_*` and `{LEGACY_VIDEOS_DIR_NAME}`."
        )
    return repo_root / "videos"


def resolve_lerobot_video_chunk_dir(
    repo_root: Path,
    episode_chunk: int,
    *,
    must_exist: bool = True,
) -> Path:
    return resolve_lerobot_videos_root(repo_root, must_exist=must_exist) / f"chunk-{int(episode_chunk):03d}"


def decode_h5_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, h5py.Dataset):
        return decode_h5_text(value[()])
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_h5_text(value.item())
    return str(value)


def standardize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat_xyzw must have shape [N,4], got {quat.shape}")
    sign = np.where(quat[:, 3:4] < 0.0, -1.0, 1.0).astype(np.float32)
    return (quat * sign).astype(np.float32)


def standardize_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat_wxyz must have shape [N,4], got {quat.shape}")
    sign = np.where(quat[:, 0:1] < 0.0, -1.0, 1.0).astype(np.float32)
    return (quat * sign).astype(np.float32)


def quaternion_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = standardize_quaternion_xyzw(quat_xyzw)
    return standardize_quaternion_wxyz(quat[:, [3, 0, 1, 2]])


def align_quaternion_sequence_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = standardize_quaternion_wxyz(quat_wxyz)
    if quat.shape[0] <= 1:
        return quat.astype(np.float32, copy=False)
    out = quat.copy()
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] *= -1.0
    return out.astype(np.float32, copy=False)


def xyz_euler_xyz_to_xyz_quat_wxyz(poses_xyz_euler: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_xyz_euler, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"poses_xyz_euler must have shape [N,6], got {poses.shape}")
    quat = quaternion_xyzw_to_wxyz(
        R.from_euler("xyz", poses[:, 3:6]).as_quat().astype(np.float32)
    )
    return np.concatenate([poses[:, :3].astype(np.float32), quat], axis=1)


def xyz_euler_xyz_to_xyz_quat(poses_xyz_euler: np.ndarray) -> np.ndarray:
    return xyz_euler_xyz_to_xyz_quat_wxyz(poses_xyz_euler)


def xyz_rotvec_to_xyz_quat_wxyz(poses_xyz_rotvec: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_xyz_rotvec, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"poses_xyz_rotvec must have shape [N,6], got {poses.shape}")
    quat = quaternion_xyzw_to_wxyz(
        R.from_rotvec(poses[:, 3:6]).as_quat().astype(np.float32)
    )
    return np.concatenate([poses[:, :3].astype(np.float32), quat], axis=1)


def xyz_rotvec_to_xyz_quat(poses_xyz_rotvec: np.ndarray) -> np.ndarray:
    return xyz_rotvec_to_xyz_quat_wxyz(poses_xyz_rotvec)


def transform_matrices_to_xyz_quat_wxyz(transforms: np.ndarray) -> np.ndarray:
    mats = np.asarray(transforms, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"transforms must have shape [N,4,4], got {mats.shape}")
    quat = quaternion_xyzw_to_wxyz(
        R.from_matrix(mats[:, :3, :3]).as_quat().astype(np.float32)
    )
    xyz = mats[:, :3, 3].astype(np.float32)
    return np.concatenate([xyz, quat], axis=1)


def transform_matrices_to_xyz_quat(transforms: np.ndarray) -> np.ndarray:
    return transform_matrices_to_xyz_quat_wxyz(transforms)


def normalize_gripper(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    arr = np.clip(arr, 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    return arr.astype(np.float32)


def shift_next(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"values must have shape [T,D], got {arr.shape}")
    if arr.shape[0] == 0:
        return arr
    return np.concatenate([arr[1:], arr[-1:]], axis=0).astype(np.float32, copy=False)


def decode_video_mp4(video_path: Path, backend: str = "opencv") -> np.ndarray:
    backend = str(backend).lower()
    if backend in {"opencv", "cv2"}:
        capture = cv2.VideoCapture(str(video_path))
        frames = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if frames:
            return np.stack(frames, axis=0).astype(np.uint8, copy=False)
        if iio is not None:
            frames = np.asarray(iio.imread(video_path), dtype=np.uint8)
            if frames.ndim == 4 and frames.shape[-1] == 3:
                return np.ascontiguousarray(frames)
        raise ValueError(f"Failed to decode frames from {video_path}")

    if backend == "imageio":
        if iio is None:
            raise ImportError("imageio.v3 is unavailable")
        frames = np.asarray(iio.imread(video_path), dtype=np.uint8)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Unexpected decoded video shape from {video_path}: {frames.shape}")
        return np.ascontiguousarray(frames)

    raise ValueError(f"Unsupported video backend: {backend}")


def decode_frame_dir(frame_dir: Path, pattern: str = "im_*.jpg") -> np.ndarray:
    frame_paths = sort_paths_natural(frame_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No frames matching {pattern} under {frame_dir}")
    frames = []
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


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


def decode_jpeg_bytes(payload: Any) -> np.ndarray:
    arr = _buffer_to_uint8_array(payload)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode JPEG payload")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def decode_concat_jpeg_stream(blob: Any, sizes: np.ndarray) -> np.ndarray:
    flat_blob = _buffer_to_uint8_array(blob)
    sizes_arr = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if sizes_arr.size == 0:
        raise ValueError("sizes is empty")
    offsets = np.zeros(sizes_arr.shape[0] + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes_arr, dtype=np.int64)
    if int(offsets[-1]) > int(flat_blob.shape[0]):
        raise ValueError(
            f"Corrupted JPEG stream: sum(sizes)={int(offsets[-1])} > blob_len={int(flat_blob.shape[0])}"
        )
    frames = [
        decode_jpeg_bytes(flat_blob[int(offsets[idx]) : int(offsets[idx + 1])])
        for idx in range(sizes_arr.shape[0])
    ]
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def decode_object_jpeg_dataset(dataset: h5py.Dataset) -> np.ndarray:
    frames = [decode_jpeg_bytes(dataset[idx]) for idx in range(int(dataset.shape[0]))]
    if not frames:
        raise ValueError("dataset has no image frames")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def load_array(handle: h5py.File | h5py.Group, key: str, *, dtype: np.dtype | None = None) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"Missing dataset: {key}")
    arr = np.asarray(handle[key])
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def install_bridge_pickle_stubs() -> None:
    module_names = [
        "sensor_msgs",
        "sensor_msgs.msg",
        "std_msgs",
        "std_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
        "genpy",
        "genpy.rostime",
    ]
    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    class _RosStub:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    for cls_name in ["Image", "CompressedImage", "Header", "PoseStamped", "Time"]:
        cls = type(
            cls_name,
            (),
            {"__init__": _RosStub.__init__, "__setstate__": _RosStub.__setstate__},
        )
        for module_name in [
            "sensor_msgs.msg",
            "std_msgs.msg",
            "geometry_msgs.msg",
            "genpy.rostime",
        ]:
            setattr(sys.modules[module_name], cls_name, cls)


def load_bridge_pickle(path: Path) -> Any:
    install_bridge_pickle_stubs()
    with path.open("rb") as f:
        return pickle.load(f)


def extract_nested_array(container: Any, key: str) -> np.ndarray:
    def _as_float_matrix(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, None]
        return arr.astype(np.float32)

    if isinstance(container, dict):
        if key not in container:
            raise KeyError(f"Key {key} not found. keys={sorted(container.keys())}")
        return _as_float_matrix(container[key])

    if isinstance(container, (list, tuple)):
        if len(container) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        values = []
        for item in container:
            if isinstance(item, dict) and key in item:
                values.append(item[key])
            elif hasattr(item, key):
                values.append(getattr(item, key))
            else:
                raise KeyError(f"Key {key} not found in list entry type={type(item).__name__}")
        return _as_float_matrix(values)

    raise TypeError(f"Unsupported container type: {type(container).__name__}")


def walk_hdf5_datasets(handle: h5py.Group) -> dict[str, h5py.Dataset]:
    out: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out[name] = obj

    handle.visititems(visitor)
    return out


def find_hdf5_dataset_by_candidates(
    dataset_map: Mapping[str, h5py.Dataset],
    candidates: Sequence[str],
) -> tuple[str, np.ndarray]:
    for candidate in candidates:
        if candidate in dataset_map:
            return candidate, np.asarray(dataset_map[candidate])
    for candidate in candidates:
        for path, dataset in dataset_map.items():
            if path.endswith(candidate):
                return path, np.asarray(dataset)
    raise KeyError(f"Could not resolve dataset from candidates={list(candidates)}")


def infer_droid_camera_video_paths(
    dataset_map: Mapping[str, h5py.Dataset],
    mp4_paths: Sequence[Path],
) -> dict[str, Path]:
    serial_to_path = {path.stem: path for path in mp4_paths}
    wrist_serials = set()
    extrinsics_prefix = "observation/camera_extrinsics/"
    for key in dataset_map:
        if not key.startswith(extrinsics_prefix):
            continue
        tail = key[len(extrinsics_prefix) :]
        serial = tail.split("_", 1)[0]
        if tail.endswith("_gripper_offset"):
            wrist_serials.add(serial)

    available_serials = sorted(serial_to_path.keys())
    wrist_serial = next((serial for serial in available_serials if serial in wrist_serials), None)
    external_serials = [serial for serial in available_serials if serial != wrist_serial]

    serial_to_y: dict[str, float] = {}
    for serial in external_serials:
        key = f"observation/camera_extrinsics/{serial}_left"
        if key not in dataset_map:
            continue
        values = np.asarray(dataset_map[key], dtype=np.float32)
        if values.ndim == 2 and values.shape[1] >= 2:
            serial_to_y[serial] = float(np.nanmean(values[:, 1]))

    if len(external_serials) >= 2 and len(serial_to_y) >= 2:
        sorted_external = sorted(
            external_serials,
            key=lambda serial: serial_to_y.get(serial, float("-inf")),
            reverse=True,
        )
    else:
        sorted_external = sorted(external_serials)

    out: dict[str, Path] = {}
    if wrist_serial is not None:
        out["wrist"] = serial_to_path[wrist_serial]
    if len(sorted_external) >= 1:
        out["left"] = serial_to_path[sorted_external[0]]
    if len(sorted_external) >= 2:
        out["right"] = serial_to_path[sorted_external[1]]
    return out


def resolve_droid_video_paths(
    *,
    dataset_root: Path,
    episode_dir: Path,
    metadata: dict[str, Any] | None,
    dataset_map: Mapping[str, h5py.Dataset],
) -> dict[str, Path]:
    field_map = {
        "wrist": "wrist_mp4_path",
        "left": "left_mp4_path",
        "right": "right_mp4_path",
        "ext1": "ext1_mp4_path",
        "ext2": "ext2_mp4_path",
    }
    resolved: dict[str, Path] = {}
    if metadata is not None:
        for key, field_name in field_map.items():
            rel_path = metadata.get(field_name)
            if rel_path:
                path = dataset_root / rel_path
                if path.exists():
                    resolved[key] = path

    if {"wrist", "left", "right"}.issubset(resolved):
        return resolved

    mp4_paths = sorted((episode_dir / "recordings" / "MP4").glob("*.mp4"))
    inferred = infer_droid_camera_video_paths(dataset_map, mp4_paths)
    if "wrist" not in resolved and "wrist" in inferred:
        resolved["wrist"] = inferred["wrist"]
    if "left" not in resolved and "left" in inferred:
        resolved["left"] = inferred["left"]
    if "right" not in resolved and "right" in inferred:
        resolved["right"] = inferred["right"]
    return resolved


def scan_gripper_spec(
    episode_paths: Sequence[Path],
    *,
    state_key: str,
    command_key: str,
    larger_is_closed: bool,
    constant_output: float | None,
) -> GripperNormSpec:
    state_values: list[np.ndarray] = []
    command_values: list[np.ndarray] = []

    for episode_path in episode_paths:
        with h5py.File(episode_path, "r") as handle:
            if state_key in handle:
                arr = load_array(handle, state_key, dtype=np.float64).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    state_values.append(arr)
            if command_key in handle:
                arr = load_array(handle, command_key, dtype=np.float64).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    command_values.append(arr)

    if not state_values and not command_values:
        return GripperNormSpec(
            minimum=0.0,
            maximum=1.0,
            larger_is_closed=larger_is_closed,
            constant_output=constant_output,
        )

    merged = np.concatenate(state_values + command_values, axis=0)
    minimum = float(np.min(merged))
    maximum = float(np.max(merged))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum < 1e-6:
        return GripperNormSpec(
            minimum=minimum,
            maximum=maximum,
            larger_is_closed=larger_is_closed,
            constant_output=constant_output,
        )

    return GripperNormSpec(
        minimum=minimum,
        maximum=maximum,
        larger_is_closed=larger_is_closed,
        constant_output=None,
    )


def normalize_gripper_with_spec(values: np.ndarray, spec: GripperNormSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if spec.constant_output is not None:
        return np.full_like(arr, fill_value=float(spec.constant_output), dtype=np.float32)
    scale = max(float(spec.maximum - spec.minimum), 1e-6)
    out = np.clip((arr - float(spec.minimum)) / scale, 0.0, 1.0)
    if spec.larger_is_closed:
        out = 1.0 - out
    return out.astype(np.float32)
