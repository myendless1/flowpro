from __future__ import annotations

from pathlib import Path

import numpy as np


def load_init_joint_target_from_hdf5(hdf5_path: str, frame_idx: int = 0) -> list[list[float]]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading Astribot initial joints from HDF5 requires h5py.") from exc

    path = Path(hdf5_path).expanduser()
    with h5py.File(path, "r") as handle:
        joints_dict = handle["joints_dict"]
        source_key = (
            "joints_position_command"
            if "joints_position_command" in joints_dict
            else "joints_position_state"
        )
        joints = np.asarray(joints_dict[source_key], dtype=np.float32)[frame_idx].reshape(-1)[-22:]

    if joints.shape[0] != 22:
        raise ValueError(f"Expected 22 non-chassis joints from {path}, got {joints.shape[0]}")
    return [
        joints[:4].tolist(),
        joints[4:11].tolist(),
        joints[11:12].tolist(),
        joints[12:19].tolist(),
        joints[19:20].tolist(),
        joints[20:22].tolist(),
    ]
