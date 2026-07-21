from astribot_env.initial_pose import default_init_joint_action, normalize_init_joint_action
from flowpro.config import load_config
from flowpro.validate import validate_project
from flowpro.workflow import collect


def test_default_initial_joint_action_has_astribot_sdk_grouping():
    target = default_init_joint_action()

    assert [len(group) for group in target] == [4, 7, 1, 7, 1, 2]
    assert target == [
        [0.5863, -1.1816, 0.5947, -0.0006],
        [0.2850, -0.3639, -1.2369, 1.6490, -0.3651, -0.0550, -0.2365],
        [0.0],
        [-0.9200, -0.4720, 1.6216, 1.9225, 0.4911, 0.0491, 0.4409],
        [0.0],
        [-0.0064, 0.8870],
    ]
    assert normalize_init_joint_action(target) == target


def test_collection_workflow_passes_configured_joint_action(capsys):
    config = load_config("configs/flowpro.json")
    assert next(check for check in validate_project(config) if check.name == "initial_joint_action").ok

    collect(config, round_id=1, dry_run=True)

    command = capsys.readouterr().out
    assert "--init-joint-action" in command
    assert "--right-gripper-target-angle-deg 45.0" in command
    assert "--init-hdf5" not in command
