from pathlib import Path
import json
from flowpro.config import load_config, PROJECT_ROOT
from flowpro.workflow import _inference_checkpoint


def test_unified_config_paths_are_project_relative():
    cfg=load_config("configs/flowpro.json")
    assert cfg.root==PROJECT_ROOT
    assert cfg.path_for("paths.pretrain_save_dir")==PROJECT_ROOT/"outputs/pretrain"
    assert cfg.path_for("paths.pretrained_transformer_dir")==PROJECT_ROOT/"outputs/pretrain/no4d-abl-delta-3500"
    assert cfg.round_dir(2)==PROJECT_ROOT/"outputs/rounds/round_02"


def test_mode_configs_inherit_common_settings_and_isolate_outputs():
    delta = load_config("configs/flowpro.delta.json")
    absolute = load_config("configs/flowpro.absolute.json")

    assert delta.section("collection")["prompt"] == absolute.section("collection")["prompt"]
    assert delta.section("model")["action_representation"] == "delta"
    assert absolute.section("model")["action_representation"] == "absolute"
    assert delta.path_for("paths.pretrained_transformer_dir").name == "no4d-abl-delta-3500"
    assert absolute.path_for("paths.pretrained_transformer_dir").name == "no4d-abl-abs-3500"
    assert delta.path_for("paths.rounds") != absolute.path_for("paths.rounds")


def test_absolute_rl_config_only_replaces_inference_checkpoint():
    absolute = load_config("configs/flowpro.absolute.json")
    absolute_rl = load_config("configs/flowpro.absolute-rl.json")

    assert absolute_rl.section("model") == absolute.section("model")
    assert absolute_rl.section("collection") == absolute.section("collection")
    assert absolute_rl.path_for("paths.rounds") == absolute.path_for("paths.rounds")
    assert absolute_rl.path_for("paths.manifests") == absolute.path_for("paths.manifests")
    assert absolute_rl.path_for("paths.pretrained_transformer_dir") == (
        PROJECT_ROOT / "outputs/rl-finetune/no4d-abl-abs-rl"
    )
    assert absolute_rl.section("inference")["checkpoint_source"] == "pretrained"
    assert _inference_checkpoint(absolute_rl, 10) == absolute_rl.path_for(
        "paths.pretrained_transformer_dir"
    )


def test_absolute_centrifuge_config_only_replaces_task_prompt():
    absolute = load_config("configs/flowpro.absolute.json")
    centrifuge = load_config("configs/flowpro.absolute.centrifuge.json")

    assert centrifuge.section("model") == absolute.section("model")
    assert centrifuge.section("inference") == absolute.section("inference")
    assert centrifuge.path_for("paths.pretrained_transformer_dir") == (
        absolute.path_for("paths.pretrained_transformer_dir")
    )
    assert centrifuge.section("collection")["prompt"] == (
        "pick up the plate and put it on centrifuge"
    )


def test_later_round_without_rpro_checkpoint_falls_back_to_mode_pretrained(tmp_path, capsys):
    config_path = tmp_path / "absolute.json"
    config_path.write_text(json.dumps({
        "base_config": str(PROJECT_ROOT / "configs/flowpro.absolute.json"),
        "paths": {"rounds": str(tmp_path / "rounds")},
    }))
    absolute = load_config(config_path)

    checkpoint = _inference_checkpoint(absolute, 10)

    assert checkpoint.name == "no4d-abl-abs-3500"
    assert "previous-round checkpoint not found" in capsys.readouterr().out
