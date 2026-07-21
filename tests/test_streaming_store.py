import json
import threading
import time

import numpy as np

from flowpro.data import Frame, PairStore, TrajectoryPair
from flowpro.data.store import STREAM_FILE_NAME


def _action(value):
    action = np.zeros(16, np.float32)
    action[[3, 11]] = 1.0
    action[0] = value
    action[8] = value
    return action


def _frame(index, source="policy"):
    image = np.full((8, 8, 3), index, np.uint8)
    payload = {
        "state_action16": _action(index),
        "wam4d": {"image": image, "history": np.asarray([_action(index)])},
        "wam4d_history": [{"image": image}],
        "step": index,
    }
    return Frame(payload, _action(index + 0.5), timestamp=1000 + index, source=source)


def test_streaming_pair_roundtrip_and_atomic_commit(tmp_path):
    store = PairStore(tmp_path)
    loser = [_frame(0), _frame(1)]
    writer = store.begin_stream(
        pair_id="pair-1",
        loser=loser,
        rollback_index=0,
        round_id=3,
        metadata={"mode": "stream"},
    )
    writer.append_winner(_frame(2, "human"))
    writer.append_winner(_frame(3, "human"))

    target = writer.commit()

    assert target == tmp_path / "pair-1"
    assert (target / STREAM_FILE_NAME).is_file()
    assert not list(tmp_path.glob(".pair-1-*"))
    pair = store.load(target)
    assert pair.round_id == 3
    assert pair.metadata["storage_format"] == "hdf5-stream-v1"
    assert len(pair.loser) == 2
    assert len(pair.winner) == 2
    np.testing.assert_array_equal(
        pair.winner[0].observation["wam4d"]["image"],
        np.full((8, 8, 3), 2, np.uint8),
    )
    np.testing.assert_array_equal(
        pair.winner[0].observation["wam4d_history"][0]["image"],
        pair.winner[0].observation["wam4d"]["image"],
    )


def test_streaming_abort_removes_unpublished_data(tmp_path):
    store = PairStore(tmp_path)
    writer = store.begin_stream(
        pair_id="pair-abort",
        loser=[_frame(0)],
        rollback_index=0,
        round_id=1,
        metadata={},
    )
    writer.append_winner(_frame(1, "human"))

    writer.abort()

    assert list(tmp_path.iterdir()) == []


def test_streaming_enqueue_does_not_run_hdf5_writes_on_control_thread(tmp_path, monkeypatch):
    from flowpro.data import store as store_module

    original = store_module._append_h5_frame
    writer_threads = []

    def slow_write(group, frame):
        writer_threads.append(threading.current_thread().name)
        time.sleep(0.02)
        original(group, frame)

    monkeypatch.setattr(store_module, "_append_h5_frame", slow_write)
    store = PairStore(tmp_path)
    writer = store.begin_stream(
        pair_id="pair-thread",
        loser=[_frame(0)],
        rollback_index=0,
        round_id=1,
        metadata={},
    )
    started = time.monotonic()
    for index in range(1, 11):
        writer.append_winner(_frame(index, "human"))
    enqueue_elapsed = time.monotonic() - started
    writer.commit()

    assert enqueue_elapsed < 0.1
    assert writer_threads
    assert all(name.startswith("flowpro-writer-") for name in writer_threads)


def test_completed_pairs_count_legacy_and_stream_formats(tmp_path):
    store = PairStore(tmp_path)
    legacy = TrajectoryPair("legacy", [_frame(0)], [_frame(1, "human")], 0)
    store.save(legacy)
    writer = store.begin_stream(
        pair_id="stream",
        loser=[_frame(2)],
        rollback_index=0,
        round_id=1,
        metadata={},
    )
    writer.append_winner(_frame(3, "human"))
    writer.commit()
    incomplete = tmp_path / ".incomplete"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text(json.dumps({"pair_id": "incomplete"}))

    assert [path.name for path in store.completed_pairs()] == ["legacy", "stream"]
    assert store.completed_count() == 2
    legacy_loaded = store.load("legacy")
    assert legacy_loaded.pair_id == "legacy"
    assert len(legacy_loaded.loser) == 1
    assert len(legacy_loaded.winner) == 1
