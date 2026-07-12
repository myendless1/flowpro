import json
import numpy as np

from flowpro.data.store import _split_arrays
from flowpro.training.rpro_trainer import PreferencePool


def test_augmented_observation_sidecar_roundtrip(tmp_path):
    observation = {"image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3), "state": np.zeros(16)}
    encoded, arrays = _split_arrays(observation, "observation")
    stem = tmp_path / "sample"
    action = np.zeros((16, 16), np.float32)
    np.savez_compressed(stem.with_suffix(".npz"), winner=action, loser=action, **arrays)
    stem.with_suffix(".json").write_text(json.dumps({"observation": encoded}))
    winner, loser, restored = PreferencePool([tmp_path]).sample()
    assert np.array_equal(winner, loser)
    assert np.array_equal(restored["image"], observation["image"])
