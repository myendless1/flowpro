from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
import numpy as np

from .types import Frame, TrajectoryPair


class PairStore:
    """Crash-safe, dependency-free preference store (NPZ arrays + JSON metadata)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

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
        meta = json.loads((target / "metadata.json").read_text())
        trajectories = {}
        for name in ("loser", "winner"):
            arrays = np.load(target / f"{name}.npz")
            observations_meta = [json.loads(line) for line in (target / f"{name}_observations.jsonl").read_text().splitlines()]
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
