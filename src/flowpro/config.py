from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        value = Path(self.data.get("project_root", "."))
        return (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve()

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing config section: {name}")
        return value

    def path_for(self, dotted_key: str, *, create: bool = False) -> Path:
        value: Any = self.data
        for part in dotted_key.split("."):
            value = value[part]
        path = Path(str(value))
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def round_dir(self, round_id: int, *, create: bool = False) -> Path:
        path = self.path_for("paths.rounds") / f"round_{round_id:02d}"
        if create: path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    path = path if path.is_absolute() else (PROJECT_ROOT / path)
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.pop("base_config", None)
    if parent:
        base = load_config(path.parent / parent).data
        payload = _merge(base, payload)
    return ProjectConfig(path, payload)

