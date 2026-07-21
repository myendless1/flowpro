from __future__ import annotations

import io
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
from typing import Any

import numpy as np

from .types import Frame, TrajectoryPair


STREAM_FILE_NAME = "trajectories.h5"
_STREAM_STOP = object()


class PairStore:
    """Crash-safe preference store supporting legacy NPZ and streaming HDF5 pairs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def begin_stream(
        self,
        *,
        pair_id: str,
        loser: list[Frame],
        rollback_index: int,
        round_id: int,
        metadata: dict[str, Any],
    ) -> StreamingPairWriter:
        try:
            import h5py  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("流式保存需要安装 h5py") from exc
        return StreamingPairWriter(
            self,
            pair_id=pair_id,
            loser=loser,
            rollback_index=rollback_index,
            round_id=round_id,
            metadata=metadata,
        )

    def completed_pairs(self) -> list[Path]:
        pairs = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            if not (path / "metadata.json").is_file():
                continue
            legacy_complete = (path / "loser.npz").is_file() and (path / "winner.npz").is_file()
            stream_complete = (path / STREAM_FILE_NAME).is_file()
            if legacy_complete or stream_complete:
                pairs.append(path)
        return sorted(pairs)

    def completed_count(self) -> int:
        return len(self.completed_pairs())

    def save(self, pair: TrajectoryPair) -> Path:
        pair.validate()
        target = self.root / pair.pair_id
        if target.exists():
            raise FileExistsError(target)
        tmp = Path(tempfile.mkdtemp(prefix=f".{pair.pair_id}-", dir=self.root))
        try:
            for name, frames in (("loser", pair.loser), ("winner", pair.winner)):
                np.savez_compressed(
                    tmp / f"{name}.npz",
                    actions=np.stack([f.action for f in frames]),
                    timestamps=np.asarray([f.timestamp for f in frames]),
                    sources=np.asarray([f.source for f in frames]),
                )
                
                # Extract arrays and JSON metadata
                obs_arrays = {}
                obs_json_list = []
                for i, f in enumerate(frames):
                    json_meta, arrays_meta = _split_arrays(f.observation, f"frame_{i}")
                    obs_json_list.append(json_meta)
                    obs_arrays.update(arrays_meta)
                    
                np.savez_compressed(tmp / f"{name}_obs_arrays.npz", **obs_arrays)
                
                with (tmp / f"{name}_observations.jsonl").open("w") as stream:
                    for meta in obs_json_list:
                        stream.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        
            (tmp / "metadata.json").write_text(json.dumps({
                "pair_id": pair.pair_id, "rollback_index": pair.rollback_index,
                "round_id": pair.round_id, **pair.metadata,
            }, ensure_ascii=False, indent=2))
            os.replace(tmp, target)
        except Exception:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return target

    def load(self, target: str | Path) -> TrajectoryPair:
        target = Path(target)
        if not target.is_absolute(): target = self.root / target
        if (target / STREAM_FILE_NAME).is_file():
            return self._load_stream(target)
        return self._load_legacy(target)

    def _load_stream(self, target: Path) -> TrajectoryPair:
        import h5py

        meta = json.loads((target / "metadata.json").read_text())
        trajectories = {}
        with h5py.File(target / STREAM_FILE_NAME, "r") as handle:
            for name in ("loser", "winner"):
                group = handle[name]
                actions = np.asarray(group["actions"], dtype=np.float32)
                timestamps = np.asarray(group["timestamps"], dtype=np.float64)
                sources = [_decode_h5_string(value) for value in group["sources"]]
                observations = [
                    _read_h5_observation(group["observations"][f"{index:08d}"])
                    for index in range(len(actions))
                ]
                trajectories[name] = [
                    Frame(observation, action, float(timestamp), source)
                    for observation, action, timestamp, source in zip(
                        observations, actions, timestamps, sources
                    )
                ]
        return TrajectoryPair(
            meta.pop("pair_id"),
            trajectories["loser"],
            trajectories["winner"],
            int(meta.pop("rollback_index")),
            int(meta.pop("round_id", 1)),
            meta,
        )

    def _load_legacy(self, target: Path) -> TrajectoryPair:
        meta = json.loads((target / "metadata.json").read_text())
        trajectories = {}
        for name in ("loser", "winner"):
            arrays = np.load(target / f"{name}.npz")
            observations_meta = [
                json.loads(line)
                for line in (target / f"{name}_observations.jsonl").read_text().splitlines()
            ]
            sidecar = target / f"{name}_obs_arrays.npz"
            if sidecar.exists():
                obs_arrays = np.load(sidecar)
                restored_obs = [_restore_split(meta, obs_arrays) for meta in observations_meta]
            else:
                # Backward compatibility with the initial JSON-only format.
                restored_obs = [_restore_legacy(meta) for meta in observations_meta]
            
            trajectories[name] = [Frame(obs, action, float(ts), str(source)) for obs, action, ts, source in zip(
                restored_obs, arrays["actions"], arrays["timestamps"], arrays["sources"])]
        return TrajectoryPair(meta.pop("pair_id"), trajectories["loser"], trajectories["winner"],
                              int(meta.pop("rollback_index")), int(meta.pop("round_id", 1)), meta)


class StreamingPairWriter:
    """Append frames on one background thread and atomically publish on commit."""

    def __init__(
        self,
        store: PairStore,
        *,
        pair_id: str,
        loser: list[Frame],
        rollback_index: int,
        round_id: int,
        metadata: dict[str, Any],
    ) -> None:
        if not loser:
            raise ValueError("流式偏好样本至少需要一帧 loser 数据")
        self.store = store
        self.pair_id = str(pair_id)
        self.target = store.root / self.pair_id
        if self.target.exists():
            raise FileExistsError(self.target)
        self.tmp = Path(tempfile.mkdtemp(prefix=f".{self.pair_id}-", dir=store.root))
        self.rollback_index = int(rollback_index)
        self.round_id = int(round_id)
        self.metadata = dict(metadata)
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._finish_lock = threading.Lock()
        self._finished = False
        self._committed = False
        self._error: BaseException | None = None
        self._winner_enqueued = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"flowpro-writer-{self.pair_id}",
            daemon=True,
        )
        self._thread.start()
        for frame in loser:
            self._queue.put(("loser", frame))

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()

    def append_winner(self, frame: Frame) -> None:
        if self._finished:
            raise RuntimeError("流式保存会话已经结束")
        self._winner_enqueued += 1
        self._queue.put(("winner", frame))

    def commit(self) -> Path:
        if self._winner_enqueued <= 0:
            raise ValueError("偏好样本至少需要一帧 winner 数据")
        self._finish(cancel=False)
        if self._error is not None:
            raise RuntimeError("后台流式写入失败") from self._error
        metadata = {
            "pair_id": self.pair_id,
            "rollback_index": self.rollback_index,
            "round_id": self.round_id,
            "storage_format": "hdf5-stream-v1",
            **self.metadata,
        }
        (self.tmp / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        os.replace(self.tmp, self.target)
        self._committed = True
        return self.target

    def abort(self) -> None:
        if not self._finished:
            self._finish(cancel=True)
        if not self._committed:
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _finish(self, *, cancel: bool) -> None:
        with self._finish_lock:
            if self._finished:
                return
            if cancel:
                self._cancel.set()
            self._queue.put(_STREAM_STOP)
            self._queue.join()
            self._thread.join()
            self._finished = True

    def _run(self) -> None:
        stop_seen = False
        try:
            import h5py

            with h5py.File(self.tmp / STREAM_FILE_NAME, "w") as handle:
                handle.attrs["storage_format"] = "hdf5-stream-v1"
                groups = {
                    name: _create_h5_trajectory_group(handle, name)
                    for name in ("loser", "winner")
                }
                written_since_flush = 0
                while True:
                    item = self._queue.get()
                    try:
                        if item is _STREAM_STOP:
                            stop_seen = True
                            break
                        if self._cancel.is_set():
                            continue
                        name, frame = item
                        _append_h5_frame(groups[name], frame)
                        written_since_flush += 1
                        if written_since_flush >= 10:
                            handle.flush()
                            written_since_flush = 0
                    finally:
                        self._queue.task_done()
                handle.flush()
        except BaseException as exc:
            self._error = exc
            if not stop_seen:
                while True:
                    item = self._queue.get()
                    self._queue.task_done()
                    if item is _STREAM_STOP:
                        break


def _create_h5_trajectory_group(handle, name: str):
    import h5py

    group = handle.create_group(name)
    group.create_dataset(
        "actions",
        shape=(0, 16),
        maxshape=(None, 16),
        chunks=(64, 16),
        dtype=np.float32,
    )
    group.create_dataset(
        "timestamps",
        shape=(0,),
        maxshape=(None,),
        chunks=(64,),
        dtype=np.float64,
    )
    group.create_dataset(
        "sources",
        shape=(0,),
        maxshape=(None,),
        chunks=(64,),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    group.create_group("observations")
    return group


def _append_h5_frame(group, frame: Frame) -> None:
    index = int(group["actions"].shape[0])
    for key in ("actions", "timestamps", "sources"):
        dataset = group[key]
        dataset.resize((index + 1,) + dataset.shape[1:])
    group["actions"][index] = frame.action
    group["timestamps"][index] = frame.timestamp
    group["sources"][index] = frame.source

    observation_group = group["observations"].create_group(f"{index:08d}")
    arrays_group = observation_group.create_group("arrays")
    arrays: dict[str, tuple[np.ndarray, bool]] = {}
    encoded = _split_h5_arrays(frame.observation, arrays, {})
    observation_group.attrs["json"] = json.dumps(encoded, ensure_ascii=False)
    for key, (array, allow_pickle) in arrays.items():
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=allow_pickle)
        payload = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        dataset = arrays_group.create_dataset(key, data=payload, compression="lzf")
        dataset.attrs["allow_pickle"] = bool(allow_pickle)


def _split_h5_arrays(value, arrays, memo):
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in memo:
            return {"__h5_ref__": memo[identity]}
        key = f"array_{len(arrays):04d}"
        memo[identity] = key
        allow_pickle = value.dtype.hasobject
        arrays[key] = (np.asarray(value), allow_pickle)
        return {"__h5_ref__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _split_h5_arrays(inner, arrays, memo) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_split_h5_arrays(inner, arrays, memo) for inner in value]
    return value


def _read_h5_observation(group):
    encoded = json.loads(_decode_h5_string(group.attrs["json"]))
    arrays = {}
    for key, dataset in group["arrays"].items():
        payload = np.asarray(dataset, dtype=np.uint8).tobytes()
        arrays[key] = np.load(
            io.BytesIO(payload),
            allow_pickle=bool(dataset.attrs.get("allow_pickle", False)),
        )
    return _restore_h5_arrays(encoded, arrays)


def _restore_h5_arrays(value, arrays):
    if isinstance(value, dict) and "__h5_ref__" in value:
        return np.asarray(arrays[value["__h5_ref__"]]).copy()
    if isinstance(value, dict):
        return {key: _restore_h5_arrays(inner, arrays) for key, inner in value.items()}
    if isinstance(value, list):
        return [_restore_h5_arrays(inner, arrays) for inner in value]
    return value


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _split_arrays(value, prefix):
    if isinstance(value, np.ndarray):
        return {"__npz_ref__": prefix}, {prefix: value}
    if isinstance(value, np.generic):
        return value.item(), {}
    if isinstance(value, dict):
        json_dict, arrays = {}, {}
        for k, v in value.items():
            j, a = _split_arrays(v, f"{prefix}_{k}")
            json_dict[str(k)] = j
            arrays.update(a)
        return json_dict, arrays
    if isinstance(value, (list, tuple)):
        json_list, arrays = [], {}
        for i, v in enumerate(value):
            j, a = _split_arrays(v, f"{prefix}_{i}")
            json_list.append(j)
            arrays.update(a)
        return json_list, arrays
    return value, {}


def _restore_split(value, arrays):
    if isinstance(value, dict) and "__npz_ref__" in value:
        return np.asarray(arrays[value["__npz_ref__"]]).copy()
    if isinstance(value, dict):
        return {k: _restore_split(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_split(v, arrays) for v in value]
    return value


def _restore_legacy(value):
    if isinstance(value, dict) and "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value.get("dtype"))
    if isinstance(value, dict):
        return {key: _restore_legacy(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_restore_legacy(inner) for inner in value]
    return value
