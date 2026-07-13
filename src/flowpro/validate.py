from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import os

from astribot_env.initial_pose import normalize_init_joint_action

from .config import ProjectConfig


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _experiment_action_representation(path: Path) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("base_experiment")
    if "action_representation" in payload:
        return str(payload["action_representation"])
    if parent:
        return _experiment_action_representation(path.parent / str(parent))
    return None


def validate_project(cfg: ProjectConfig, *, require_hardware: bool = False) -> list[Check]:
    checks: list[Check] = []
    collection = cfg.section("collection")
    augmentation = cfg.section("augmentation")
    action_representation = str(
        cfg.section("model").get("action_representation", "delta")
    )
    checks.append(Check(
        "action_representation",
        action_representation in {"absolute", "delta"},
        action_representation,
    ))
    experiment_path = cfg.path_for("model.experiment_config")
    try:
        experiment_representation = _experiment_action_representation(experiment_path)
    except (OSError, ValueError, TypeError) as exc:
        checks.append(Check("experiment_representation", False, str(exc)))
    else:
        checks.append(Check(
            "experiment_representation",
            experiment_representation == action_representation,
            f"project={action_representation}, experiment={experiment_representation}",
        ))
    horizon = int(augmentation["horizon"])
    rollback = int(collection["rollback_horizon"])
    capacity = int(collection["rollback_capacity"])
    checks.append(Check("rollback_vs_chunk", rollback >= horizon,
                        f"rollback_horizon={rollback}, action_horizon={horizon}"))
    checks.append(Check("rollback_capacity", capacity >= rollback,
                        f"capacity={capacity}, horizon={rollback}"))
    try:
        normalize_init_joint_action(collection["init_joint_action"])
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(Check("initial_joint_action", False, str(exc)))
    else:
        checks.append(Check("initial_joint_action", True, "six Astribot non-chassis joint groups"))
    base = cfg.path_for("model.base_checkpoint")
    for component in ("vae", "tokenizer", "text_encoder", "transformer"):
        path = base / component
        checks.append(Check(f"checkpoint/{component}", path.is_dir(), str(path)))
    sft = cfg.path_for("paths.sft_dataset")
    checks.append(Check("sft_dataset", sft.is_dir(), str(sft)))
    checks.append(Check("sft_text_embedding", (sft / "empty_emb.pt").is_file(), str(sft / "empty_emb.pt")))
    transformer = cfg.path_for("paths.pretrained_transformer_dir")
    transformer_ok = (
        (transformer / "config.json").is_file()
        or (transformer / "transformer" / "config.json").is_file()
        or (transformer / "checkpoints" / "last" / "transformer" / "config.json").is_file()
    )
    checks.append(Check("pretrained_transformer_dir", transformer_ok, str(transformer)))
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
    return checks
