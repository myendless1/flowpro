from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import socket

from .config import ProjectConfig, load_config
from .validate import validate_project


def _subprocess_env(cfg: ProjectConfig) -> dict[str, str]:
    src_path = str(cfg.root / "src")
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = src_path if not existing else src_path + os.pathsep + existing
    return {**os.environ, "PYTHONPATH": pythonpath}


def _training_prefix(cfg: ProjectConfig) -> list[str]:
    distributed = cfg.section("distributed")
    if not distributed.get("enabled", True):
        return [sys.executable]
    return [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
            str(cfg.path_for("distributed.config"))]


def _write_manifest(cfg: ProjectConfig, stage: str, command: list[str], outputs: dict, round_id: int | None):
    root = cfg.path_for("paths.manifests", create=True)
    record = {"stage": stage, "round": round_id, "time": time.time(), "config": str(cfg.path),
              "command": command, "outputs": {k: str(v) for k, v in outputs.items()}}
    target = root / f"{stage}{'' if round_id is None else f'_r{round_id:02d}'}.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def _run(cfg, stage, command, outputs, round_id, dry_run):
    print(f"[{stage}] cwd={cfg.root}")
    print(shlex.join(command))
    if dry_run: return
    for value in outputs.values(): Path(value).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=cfg.root, env=_subprocess_env(cfg), check=True)
    _write_manifest(cfg, stage, command, outputs, round_id)


def pretrain(cfg, round_id, dry_run):
    s=cfg.section("pretrain"); out=cfg.path_for("paths.pretrain_save_dir", create=not dry_run)
    cmd=[*_training_prefix(cfg),"-m","wan_va.train","--config-name",s["config_name"],"--experiment-config",str(cfg.path_for("model.experiment_config")),
         "--save-root",str(out),"--dataset-paths",repr([str(cfg.path_for("paths.sft_dataset"))]),"--pretrained-model-path",str(cfg.path_for("model.base_checkpoint")),
         "--batch-size",str(s["batch_size"]),"--num-steps",str(s["num_steps"]),"--enable-wandb",str(s.get("enable_wandb",False)).lower()]
    _run(cfg,"pretrain",cmd,{"checkpoint":out},None,dry_run)


def validate(cfg, round_id, dry_run, *, require_hardware: bool = False):
    checks = validate_project(cfg, require_hardware=require_hardware)
    for check in checks:
        print(f"{'OK' if check.ok else 'MISSING':7} {check.name:28} {check.detail}")
    if not all(check.ok for check in checks):
        raise RuntimeError("FlowPRO validation failed")


def _has_transformer_checkpoint(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        or (path / "transformer" / "config.json").is_file()
        or (path / "checkpoints" / "last" / "transformer" / "config.json").is_file()
    )


def _inference_checkpoint(cfg: ProjectConfig, round_id: int) -> Path:
    pretrained = cfg.path_for("paths.pretrained_transformer_dir")
    mode = str(cfg.section("inference").get("checkpoint_source", "auto"))
    if mode not in {"auto", "pretrained", "previous_round"}:
        raise ValueError(
            "inference.checkpoint_source must be auto, pretrained, or previous_round"
        )
    if round_id <= 1 or mode == "pretrained":
        return pretrained
    previous = cfg.round_dir(round_id - 1) / "offline_rl"
    if mode == "previous_round" or _has_transformer_checkpoint(previous):
        return previous
    print(
        f"WARNING: previous-round checkpoint not found at {previous}; "
        f"using pretrained transformer {pretrained}",
        flush=True,
    )
    return pretrained


def infer(cfg, round_id, dry_run):
    rid=round_id or 1; s=cfg.section("inference"); out=cfg.round_dir(rid,create=not dry_run)/"inference"
    checkpoint=_inference_checkpoint(cfg,rid)
    cmd=[sys.executable,"-m","wan_va.wan_va_server","--config-name",s["config_name"],"--experiment-config",str(cfg.path_for("model.experiment_config")),
         "--port",str(s["port"]),"--transformer-source",str(checkpoint),
         "--pretrained-model-path",str(cfg.path_for("model.base_checkpoint")),"--state-history-len",str(s["state_history_len"]),
         "--action-num-inference-steps",str(s["action_num_inference_steps"])]
    _run(cfg,"inference",cmd,{"runtime":out},rid,dry_run)


def collect(cfg, round_id, dry_run):
    rid=round_id or 1; s=cfg.section("collection"); out=cfg.round_dir(rid,create=not dry_run)/"raw_pairs"
    cmd=[sys.executable,"-m","flowpro.cli.collect","--output",str(out),"--quest-state-url",s["quest_state_url"],
         "--rollback-horizon",str(s["rollback_horizon"]),"--rollback-capacity",str(s["rollback_capacity"]),
         "--rollback-rate-hz",str(s.get("rollback_rate_hz",20)),
         "--trigger-threshold",str(s["middle_trigger_threshold"]),"--control-rate-hz",str(s["control_rate_hz"]),
         "--policy-rate-hz",str(s["policy_rate_hz"]),"--record-rate-hz",str(s.get("record_rate_hz",s["policy_rate_hz"])),
         "--takeover-rate-hz",str(s.get("takeover_rate_hz",50)),
         "--host",str(cfg.section("inference").get("host","127.0.0.1")),
         "--port",str(cfg.section("inference")["port"]),"--round",str(rid),
         "--prompt",str(s.get("prompt","perform the task")),"--replan-steps",str(s.get("replan_steps",8)),
         "--state-history-len",str(cfg.section("inference")["state_history_len"]),
         "--obs-history-len",str(s.get("obs_history_len",9)),
         "--camera-sync-slop-s",str(s.get("camera_sync_slop_s",0.05)),
         "--camera-sync-rate-hz",str(s.get("camera_sync_rate_hz",40)),
         "--video-guidance-scale",str(s.get("video_guidance_scale",1)),
         "--action-guidance-scale",str(s.get("action_guidance_scale",1)),
         "--action-representation",str(cfg.section("model").get("action_representation","delta")),
         "--max-translation-step-m",str(s.get("max_translation_step_m",0.06)),
         "--takeover-max-translation-step-m",str(s.get("takeover_max_translation_step_m",0.01)),
         "--takeover-max-rotation-step-deg",str(s.get("takeover_max_rotation_step_deg",2.5)),
         "--takeover-max-gripper-step",str(s.get("takeover_max_gripper_step",0.02)),
         "--right-gripper-target-angle-deg",str(s.get("right_gripper_target_angle_deg",45.0)),
         "--right-gripper-ray-axis",str(s.get("right_gripper_ray_axis","+z")),
         "--right-gripper-level-axis",str(s.get("right_gripper_level_axis","+x")),
         "--gripper-trigger-threshold",str(s.get("gripper_trigger_threshold",0.2)),
         "--first-policy-waypoint-duration",str(s.get("first_policy_waypoint_duration",0.6)),
         "--policy-waypoint-duration",str(s.get("policy_waypoint_duration",0.1)),
         "--policy-waypoint-batch-actions",str(s.get("policy_waypoint_batch_actions",8)),
         "--reset-prelift-height-m",str(s.get("reset_prelift_height_m",0.10)),
         "--reset-prelift-duration",str(s.get("reset_prelift_duration",1.0))]
    if int(s.get("target_pairs",0)) > 0:
        cmd.extend(["--target-pairs",str(s["target_pairs"])])
    if s.get("sdk_root"):
        cmd.extend(["--sdk-root",str(Path(s["sdk_root"]).expanduser())])
    if s.get("init_joint_action") is not None:
        cmd.extend(["--init-joint-action", json.dumps(s["init_joint_action"], separators=(",", ":"))])
    if bool(s.get("image_from_s1_topic", True)):
        cmd.append("--image-from-s1-topic")
    else:
        cmd.append("--sdk-image-polling")
    if not bool(s.get("policy_control_left_arm", True)):
        cmd.append("--disable-policy-left-arm")
    if not bool(s.get("right_gripper_angle_constraint_during_takeover", True)):
        cmd.append("--disable-right-gripper-angle-constraint-during-takeover")
    if not bool(s.get("right_gripper_twist_level_constraint", True)):
        cmd.append("--disable-right-gripper-twist-level-constraint")
    for key, flag in (("left_xyz_low","--left-xyz-low"),("left_xyz_high","--left-xyz-high"),
                      ("right_xyz_low","--right-xyz-low"),("right_xyz_high","--right-xyz-high")):
        if s.get(key) is not None:
            cmd.extend([flag,*[str(value) for value in s[key]]])
    if s.get("right_min_z") is not None:
        cmd.extend(["--right-min-z",str(s["right_min_z"])])
    _run(cfg,"collect",cmd,{"raw_pairs":out},rid,dry_run)


def augment(cfg, round_id, dry_run):
    rid=round_id or 1; inp=cfg.round_dir(rid)/"raw_pairs"; out=cfg.round_dir(rid,create=not dry_run)/"preference_dataset"
    cmd=[sys.executable,"-m","flowpro.cli.augment","--input",str(inp),"--output",str(out),"--config",str(cfg.path),"--round",str(rid)]
    _run(cfg,"augment",cmd,{"preference_dataset":out},rid,dry_run)


def offline_rl(cfg, round_id, dry_run):
    rid=round_id or 1; s=cfg.section("offline_rl"); out=cfg.round_dir(rid,create=not dry_run)/"offline_rl"
    ref=cfg.path_for("paths.pretrained_transformer_dir") if rid == 1 else cfg.round_dir(rid-1)/"offline_rl"
    cmd=[*_training_prefix(cfg),"-m","flowpro.cli.offline_rl","--config",str(cfg.path),"--round",str(rid),"--reference",str(ref),"--output",str(out),
         "--steps",str(s["num_steps"]),"--batch-size",str(s["batch_size"])]
    _run(cfg,"offline_rl",cmd,{"checkpoint":out},rid,dry_run)


STAGES={"validate":validate,"pretrain":pretrain,"infer":infer,"collect":collect,"augment":augment,"offline-rl":offline_rl}


def inference_collect(cfg: ProjectConfig, round_id: int, dry_run: bool):
    if dry_run:
        infer(cfg, round_id, True); collect(cfg, round_id, True); return
    s=cfg.section("inference"); checkpoint=_inference_checkpoint(cfg,round_id)
    server=[sys.executable,"-m","wan_va.wan_va_server","--config-name",s["config_name"],"--experiment-config",str(cfg.path_for("model.experiment_config")),
            "--port",str(s["port"]),"--transformer-source",str(checkpoint),
            "--pretrained-model-path",str(cfg.path_for("model.base_checkpoint")),"--state-history-len",str(s["state_history_len"]),
            "--action-num-inference-steps",str(s["action_num_inference_steps"])]
    env=_subprocess_env(cfg); process=subprocess.Popen(server,cwd=cfg.root,env=env)
    try:
        deadline=time.monotonic()+float(s.get("startup_timeout_seconds",600))
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError(f"Inference server exited with {process.returncode}")
            try:
                with socket.create_connection((s.get("host","127.0.0.1"),int(s["port"])),timeout=1): break
            except OSError: time.sleep(1)
        else: raise TimeoutError("Inference server did not become ready")
        collect(cfg,round_id,False)
    finally:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=[*STAGES,"round","all"]); p.add_argument("--config",default="configs/flowpro.json")
    p.add_argument("--round",type=int,default=1); p.add_argument("--dry-run",action="store_true"); p.add_argument("--hardware",action="store_true"); a=p.parse_args(); cfg=load_config(a.config)
    if not a.dry_run and a.stage in {"round", "all"}:
        failed = [check for check in validate_project(cfg, require_hardware=True) if not check.ok]
        if failed:
            details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
            raise RuntimeError(f"FlowPRO preflight failed:\n{details}")
    if a.stage=="round":
        inference_collect(cfg,a.round,a.dry_run)
        for name in ("augment","offline-rl"): STAGES[name](cfg,a.round,a.dry_run)
    elif a.stage=="all":
        pretrain(cfg,None,a.dry_run)
        for rid in range(1,int(cfg.section("pipeline")["rounds"])+1):
            inference_collect(cfg,rid,a.dry_run)
            for name in ("augment","offline-rl"): STAGES[name](cfg,rid,a.dry_run)
    elif a.stage=="validate":
        validate(cfg,a.round,a.dry_run,require_hardware=a.hardware)
    else: STAGES[a.stage](cfg,a.round,a.dry_run)

if __name__ == "__main__": main()
