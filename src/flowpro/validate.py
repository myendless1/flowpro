from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import os

from .config import ProjectConfig


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def validate_project(cfg: ProjectConfig, *, require_hardware: bool = False) -> list[Check]:
    checks: list[Check] = []
    collection = cfg.section("collection")
    augmentation = cfg.section("augmentation")
    horizon = int(augmentation["horizon"])
    rollback = int(collection["rollback_horizon"])
    capacity = int(collection["rollback_capacity"])
    checks.append(Check("rollback_vs_chunk", rollback >= horizon,
                        f"rollback_horizon={rollback}, action_horizon={horizon}"))
    checks.append(Check("rollback_capacity", capacity >= rollback,
                        f"capacity={capacity}, horizon={rollback}"))
    base = cfg.path_for("model.base_checkpoint")
    for component in ("vae", "tokenizer", "text_encoder", "transformer"):
        path = base / component
        checks.append(Check(f"checkpoint/{component}", path.is_dir(), str(path)))
    sft = cfg.path_for("paths.sft_dataset")
    checks.append(Check("sft_dataset", sft.is_dir(), str(sft)))
    checks.append(Check("sft_text_embedding", (sft / "empty_emb.pt").is_file(), str(sft / "empty_emb.pt")))
    distributed = cfg.path_for("distributed.config")
    checks.append(Check("accelerate_config", distributed.is_file(), str(distributed)))
    for module in ("numpy", "torch", "accelerate", "diffusers", "transformers", "einops", "easydict"):
        checks.append(Check(f"python/{module}", importlib.util.find_spec(module) is not None, module))
    if require_hardware:
        prompt = str(cfg.section("collection").get("prompt", "")).strip()
        checks.append(Check("task_prompt", bool(prompt), prompt or "set collection.prompt"))
        for module in ("cv2", "requests", "websockets", "msgpack"):
            checks.append(Check(f"robot/{module}", importlib.util.find_spec(module) is not None, module))
        quest = str(cfg.section("collection").get("quest_state_url", ""))
        checks.append(Check("quest_state_url", bool(quest), quest or "not configured"))
        sdk_value = str(cfg.section("collection").get("sdk_root", "")).strip() or os.getenv("ASTRIBOT_SDK_ROOT", "")
        sdk = Path(sdk_value).expanduser() if sdk_value else Path("/opt/astribot_sdk")
        checks.append(Check("astribot_sdk", (sdk / "core").is_dir(), str(sdk)))
        init_hdf5 = str(cfg.section("collection").get("init_hdf5", "")).strip()
        if init_hdf5:
            init_path = cfg.path_for("collection.init_hdf5")
            checks.append(Check("initial_pose", init_path.is_file(), str(init_path)))
    return checks
