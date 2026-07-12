from pathlib import Path
from flowpro.config import load_config, PROJECT_ROOT


def test_unified_config_paths_are_project_relative():
    cfg=load_config("configs/flowpro.json")
    assert cfg.root==PROJECT_ROOT
    assert cfg.path_for("paths.pretrain")==PROJECT_ROOT/"outputs/pretrain"
    assert cfg.round_dir(2)==PROJECT_ROOT/"outputs/rounds/round_02"
