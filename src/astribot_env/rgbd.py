from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

import cv2
import numpy as np

from astribot_env.utils import convert_gripper_cmd_value_to_action_value, sdk_xyzw_to_action_quat

from astribot_env.utils import ACTION16_DIM


CAMERA_NAME_MAP = {
    "cam_high": "Bolt",
    "cam_left_wrist": "left_D405",
    "cam_right_wrist": "right_D405",
}


class RGBDReader:
    def __init__(
        self,
        astribot: Any,
        *,
        camera_timeout: float = 0.3,
        cameras_info: dict[str, Any] | None = None,
        use_topic: bool = False,
        sync_slop_s: float = 0.05,
        sync_rate_hz: float = 40.0,
    ) -> None:
        self.astribot = astribot
        self.camera_timeout = float(camera_timeout)
        self.use_topic = bool(use_topic)
        self.sync_slop_s = float(sync_slop_s)
        self.sync_min_interval_s = 1.0 / max(float(sync_rate_hz), 1e-6)
        self.use_sdk_callback = False
        self._lock = threading.Lock()
        self._images = {"Bolt": None, "left_D405": None, "right_D405": None}
        self._times = {"Bolt": 0.0, "left_D405": 0.0, "right_D405": 0.0}
        self._capture_info: dict[str, Any] = {}
        self._subscribers = []
        self._synchronizer = None

        if self.use_topic:
            self.astribot.activate_camera(cameras_info or {})
            self._init_ros_topics()
        else:
            self.astribot.activate_camera(cameras_info or {})
            if not hasattr(self.astribot, "get_images_dict") and hasattr(self.astribot, "register_image_callback"):
                self.use_sdk_callback = True
                self._register_sdk_camera_callbacks()

    def _init_ros_topics(self) -> None:
        try:
            import message_filters
            from sensor_msgs.msg import CompressedImage
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROS topic camera mode requires message_filters and sensor_msgs."
            ) from exc

        topics = (
            "/astribot_camera/head_rgbd/color_compress/compressed",
            "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
            "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
        )
        self._subscribers = [
            message_filters.Subscriber(topic, CompressedImage)
            for topic in topics
        ]
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            self._subscribers,
            queue_size=20,
            slop=self.sync_slop_s,
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._synced_camera_callback)

    def _synced_camera_callback(self, head_msg: Any, left_msg: Any, right_msg: Any) -> None:
        messages = (head_msg, left_msg, right_msg)
        stamps = [float(msg.header.stamp.to_sec()) for msg in messages]
        image_timestamp = max(stamps)
        with self._lock:
            previous_timestamp = self._capture_info.get("image_timestamp")
        if (
            previous_timestamp is not None
            and image_timestamp - float(previous_timestamp) < self.sync_min_interval_s
        ):
            return
        images = {
            key: cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            for key, msg in zip(("Bolt", "left_D405", "right_D405"), messages)
        }
        if any(image is None for image in images.values()):
            return
        received_at = time.time()
        with self._lock:
            self._images.update(images)
            self._times.update({key: received_at for key in images})
            self._capture_info = {
                "image_timestamp": image_timestamp,
                "camera_timestamps": {
                    key: stamp
                    for key, stamp in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), stamps)
                },
                "camera_skew_s": max(stamps) - min(stamps),
                "image_received_at": received_at,
                "image_source": "ros_approximate_sync",
            }

    def _register_sdk_camera_callbacks(self) -> None:
        def _cb(topic_name, _msg, _width, _height, array):
            if not isinstance(array, np.ndarray) or array.ndim != 3:
                return
            parts = str(topic_name).split("/")
            camera_name = parts[2] if len(parts) > 2 else ""
            key = {
                "head_rgbd": "Bolt",
                "left_wrist_rgbd": "left_D405",
                "right_wrist_rgbd": "right_D405",
            }.get(camera_name)
            if key is not None:
                captured_at = time.time()
                with self._lock:
                    self._images[key] = array
                    self._times[key] = captured_at
                    self._capture_info = {
                        "image_timestamp": captured_at,
                        "camera_skew_s": max(self._times.values()) - min(self._times.values()),
                        "image_received_at": captured_at,
                        "image_source": "sdk_callback",
                    }

        self.astribot.register_image_callback("head_rgbd", "color", _cb, need_decode=True)
        self.astribot.register_image_callback("left_wrist_rgbd", "color", _cb, need_decode=True)
        self.astribot.register_image_callback("right_wrist_rgbd", "color", _cb, need_decode=True)

    def _set_compressed_image(self, key: str, msg: Any) -> None:
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        captured_at = float(msg.header.stamp.to_sec())
        with self._lock:
            self._images[key] = image
            self._times[key] = captured_at

    def _head_callback(self, msg: Any) -> None:
        self._set_compressed_image("Bolt", msg)

    def _left_callback(self, msg: Any) -> None:
        self._set_compressed_image("left_D405", msg)

    def _right_callback(self, msg: Any) -> None:
        self._set_compressed_image("right_D405", msg)

    def get_bgr_images_dict(self) -> dict[str, np.ndarray]:
        images, _ = self.get_bgr_images_snapshot()
        return images

    def get_bgr_images_snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.use_topic or self.use_sdk_callback:
            deadline = time.monotonic() + self.camera_timeout
            while True:
                now = time.time()
                with self._lock:
                    missing = [
                        key for key in CAMERA_NAME_MAP.values()
                        if self._images[key] is None or now - self._times[key] > self.camera_timeout
                    ]
                    if not missing:
                        images = {
                            key: np.asarray(value)
                            for key, value in self._images.items()
                            if value is not None
                        }
                        info = dict(self._capture_info)
                        info["image_age_s"] = max(
                            0.0, now - float(info.get("image_received_at", now))
                        )
                        return images, info
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Astribot camera timeout or missing synchronized frames: {missing}"
                    )
                time.sleep(0.005)

        rgb_dict, _, _, _ = self.astribot.get_images_dict()
        missing = [camera for camera in CAMERA_NAME_MAP.values() if camera not in rgb_dict]
        if missing:
            raise RuntimeError(f"Astribot SDK image dict missing cameras: {missing}")
        captured_at = time.time()
        return rgb_dict, {
            "image_timestamp": captured_at,
            "camera_skew_s": None,
            "image_received_at": captured_at,
            "image_age_s": 0.0,
            "image_source": "sdk_get_images_dict",
        }


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC color image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_rgb(image_rgb: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel image_shape, got {image_shape}")
    image_rgb = np.asarray(image_rgb)
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized.astype(np.uint8, copy=False))


def get_current_eef_state(astribot: Any, *, use_xyzw: bool = False, frame: str | None = None) -> np.ndarray:
    names = [astribot.arm_left_name, astribot.arm_right_name]
    if frame:
        try:
            cart_state = astribot.get_current_cartesian_pose(names=names, frame=frame)
        except TypeError:
            cart_state = astribot.get_current_cartesian_pose(names=names)
    else:
        cart_state = astribot.get_current_cartesian_pose(names=names)
    joint_state = astribot.get_current_joints_position(
        names=[astribot.effector_left_name, astribot.effector_right_name]
    )
    left_pose = np.asarray(cart_state[0], dtype=np.float32).reshape(-1)
    right_pose = np.asarray(cart_state[1], dtype=np.float32).reshape(-1)
    left_pose[3:7] = sdk_xyzw_to_action_quat(left_pose[3:7], use_xyzw=use_xyzw)
    right_pose[3:7] = sdk_xyzw_to_action_quat(right_pose[3:7], use_xyzw=use_xyzw)
    left_gripper = convert_gripper_cmd_value_to_action_value(
        float(np.asarray(joint_state[0], dtype=np.float32).reshape(-1)[0])
    )
    right_gripper = convert_gripper_cmd_value_to_action_value(
        float(np.asarray(joint_state[1], dtype=np.float32).reshape(-1)[0])
    )
    return np.concatenate(
        [left_pose[:7], [left_gripper], right_pose[:7], [right_gripper]]
    ).astype(np.float32)


def build_history_state(action_history: deque[np.ndarray], history_len: int) -> np.ndarray:
    valid_len = min(len(action_history), int(history_len))
    if valid_len == 0:
        return np.zeros((0, ACTION16_DIM), dtype=np.float32)
    history = np.asarray(list(action_history)[-valid_len:], dtype=np.float32)
    return history.reshape(valid_len, ACTION16_DIM)


def build_wam4d_observation_payload(
    *,
    bgr_images: dict[str, np.ndarray],
    prompt: str,
    action_history: deque[np.ndarray],
    history_len: int,
) -> dict[str, Any]:
    executed_action_history = build_history_state(action_history, history_len).astype(np.float32)
    return {
        "observation.images.cam_high": bgr_to_rgb(bgr_images["Bolt"]),
        "observation.images.cam_left_wrist": bgr_to_rgb(bgr_images["left_D405"]),
        "observation.images.cam_right_wrist": bgr_to_rgb(bgr_images["right_D405"]),
        "observation.state": executed_action_history,
        "observation.executed_action_history": executed_action_history,
        "task": prompt,
    }


def build_serl_images(
    *,
    bgr_images: dict[str, np.ndarray],
    image_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    return {
        serl_key: resize_rgb(bgr_to_rgb(bgr_images[sdk_key]), image_shape)
        for serl_key, sdk_key in CAMERA_NAME_MAP.items()
    }
