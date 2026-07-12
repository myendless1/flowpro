import json
from pathlib import Path

import numpy as np

from flowpro.collection.astribot_runtime import FakeAstribotRobotIO
from wan_va.action_representation import (
    EXECUTION_CHANNEL_IDS,
    decode_execution_sequence,
    delta16_to_model30,
    model30_to_execution16,
)


def _pose16():
    value = np.zeros(16, np.float32)
    value[[3, 11]] = 1.0
    return value


def test_delta_model_encoding_and_sequential_decoding_round_trip():
    initial = _pose16()
    targets = np.stack([initial.copy(), initial.copy()])
    targets[0, 0] = 0.01
    targets[1, 0] = 0.03
    targets[:, [7, 15]] = [[0.2, 0.8], [0.3, 0.7]]
    references = np.stack([initial, targets[0]])

    model, mask = delta16_to_model30(targets, references=references)
    deltas = model30_to_execution16(model)

    np.testing.assert_allclose(deltas[:, 0], [0.01, 0.02], atol=1e-7)
    np.testing.assert_allclose(decode_execution_sequence(deltas, initial_absolute=initial), targets)
    assert mask[:, EXECUTION_CHANNEL_IDS].all()


def test_fake_robot_applies_each_delta_against_live_state():
    robot = FakeAstribotRobotIO()
    delta = _pose16()
    delta[0] = 0.01
    robot.execute(delta)
    robot.execute(delta)
    np.testing.assert_allclose(robot.state_action16()[0], 0.02, atol=1e-7)


def test_flowpro_uses_reference_delta_experiment_without_representation_switch():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/flowpro.json").read_text())
    assert config["model"]["experiment_config"].endswith("/delta.json")
    assert "action_representation" not in config["model"]
