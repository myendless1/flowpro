"""JSON experiment configuration loader used by both training and inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from easydict import EasyDict


def _merge(target: EasyDict, values: dict) -> EasyDict:
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), (dict, EasyDict)):
            nested = EasyDict(copy.deepcopy(dict(target[key])))
            target[key] = _merge(nested, value)
        else:
            target[key] = EasyDict(value) if isinstance(value, dict) else value
    return target


def _read_with_inheritance(path: Path, seen: set[Path]) -> dict:
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"Experiment config inheritance cycle at {path}")
    seen.add(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent_name = payload.pop("base_experiment", None)
    if parent_name is None:
        return payload
    parent = _read_with_inheritance(path.parent / parent_name, seen)
    parent.update(payload)
    return parent


def load_experiment_config(path: str | Path, config_registry: dict) -> EasyDict:
    path = Path(path).expanduser().resolve()
    payload = _read_with_inheritance(path, set())
    base_name = payload.pop("base_config", None)
    if base_name is None:
        raise ValueError(f"Experiment config {path} requires 'base_config'")
    if base_name not in config_registry:
        raise KeyError(
            f"Unknown base_config '{base_name}'; available={sorted(config_registry)}"
        )
    config = EasyDict(copy.deepcopy(dict(config_registry[base_name])))
    # State/action history is always absolute cmd action, even when the model
    # predicts delta actions. Preserve the task config's absolute statistics
    # before an experiment overrides norm_stat with delta target statistics.
    if "state_norm_stat" not in payload and "norm_stat" in config:
        config.state_norm_stat = copy.deepcopy(config.norm_stat)
    config = _merge(config, payload)
    config.state_action_representation = str(
        getattr(config, "state_action_representation", "absolute")
    )
    if config.state_action_representation != "absolute":
        raise ValueError("state_action_representation must be 'absolute'")
    config.experiment_config_path = str(path)
    config.experiment_name = str(config.get("experiment_name", path.stem))
    return config
