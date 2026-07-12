from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys
import types
from typing import Any


DEFAULT_ASTRIBOT_SDK_ROOT = Path("/opt/astribot_sdk")


def _prepend_python_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        return
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _make_package_alias(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def ensure_astribot_sdk_path(sdk_root: str | os.PathLike[str] | None = None) -> Path:
    root = Path(sdk_root or os.environ.get("ASTRIBOT_SDK_ROOT", DEFAULT_ASTRIBOT_SDK_ROOT)).expanduser().resolve()
    if not (root / "core").is_dir():
        raise RuntimeError(f"Astribot SDK path does not look valid: {root}")

    _prepend_python_path(root)
    _prepend_python_path(root / "astribot_msgs" / "build" / "devel" / "lib" / "python3" / "dist-packages")
    _prepend_python_path(root / "core" / "common")
    _prepend_python_path(root / "core" / "common" / "util")
    _prepend_python_path(root / "core" / "common" / "whole_body_control" / "script")
    _prepend_python_path(root / "core" / "common" / "whole_body_control" / "third_party")

    os.environ.setdefault("ASTRIBOT_SDK_ROOT", str(root))
    os.environ.setdefault("ASTRIBOT_COMMON_ROOT", str(root))
    os.environ.setdefault("ROBOT_TYPE", "S1")
    if not hasattr(builtins, "MetaBase"):
        builtins.MetaBase = object

    common_root = root / "core" / "common"
    wbc_root = common_root / "whole_body_control"
    wbc_script = wbc_root / "script"
    if wbc_script.exists():
        _make_package_alias("meta", common_root)
        _make_package_alias("meta.whole_body_control", wbc_root)
        _make_package_alias("meta.whole_body_control.script", wbc_script)
    return root


def load_astribot_class(sdk_root: str | os.PathLike[str] | None = None) -> Any:
    ensure_astribot_sdk_path(sdk_root)
    from core.astribot_api.astribot_client import Astribot

    return Astribot
