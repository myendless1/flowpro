import threading
import time

import cv2
import numpy as np
import pytest

from astribot_env.rgbd import RGBDReader


class _Stamp:
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


class _Message:
    def __init__(self, image, timestamp):
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        self.data = encoded.tobytes()
        self.header = type("Header", (), {"stamp": _Stamp(timestamp)})()


def _reader():
    reader = RGBDReader.__new__(RGBDReader)
    reader.camera_timeout = 0.05
    reader.use_topic = True
    reader.use_sdk_callback = False
    reader.sync_min_interval_s = 0.05
    reader._lock = threading.Lock()
    reader._images = {"Bolt": None, "left_D405": None, "right_D405": None}
    reader._times = {"Bolt": 0.0, "left_D405": 0.0, "right_D405": 0.0}
    reader._capture_info = {}
    return reader


def test_synced_camera_callback_records_ros_timestamps_and_triplet():
    reader = _reader()
    images = [np.full((12, 16, 3), value, np.uint8) for value in (20, 80, 140)]

    reader._synced_camera_callback(
        _Message(images[0], 100.00),
        _Message(images[1], 100.02),
        _Message(images[2], 100.04),
    )
    captured, timing = reader.get_bgr_images_snapshot()

    assert set(captured) == {"Bolt", "left_D405", "right_D405"}
    assert all(image.shape == (12, 16, 3) for image in captured.values())
    assert timing["image_source"] == "ros_approximate_sync"
    assert timing["image_timestamp"] == pytest.approx(100.04)
    assert timing["camera_skew_s"] == pytest.approx(0.04)
    assert timing["camera_timestamps"]["cam_left_wrist"] == pytest.approx(100.02)


def test_synced_camera_snapshot_rejects_stale_frames():
    reader = _reader()
    reader.camera_timeout = 0.01
    with reader._lock:
        reader._images = {
            "Bolt": np.zeros((2, 2, 3), np.uint8),
            "left_D405": np.zeros((2, 2, 3), np.uint8),
            "right_D405": np.zeros((2, 2, 3), np.uint8),
        }
        reader._times = {key: time.time() - 1.0 for key in reader._images}

    with pytest.raises(RuntimeError, match="missing synchronized frames"):
        reader.get_bgr_images_snapshot()
