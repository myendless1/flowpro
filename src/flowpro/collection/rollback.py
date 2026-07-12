from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from .protocol import RobotIO
from flowpro.data.types import Frame
from wan_va.action_representation import relative_pose7


@dataclass
class RollbackConfig:
    capacity: int = 200
    default_horizon: int = 20
    step_interval_s: float = 0.0


class RollbackBuffer:
    def __init__(self, config: RollbackConfig | None = None) -> None:
        self.config = config or RollbackConfig()
        self.frames: deque[Frame] = deque(maxlen=self.config.capacity)

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def segment(self, horizon: int | None = None) -> list[Frame]:
        n = min(horizon or self.config.default_horizon, len(self.frames))
        return list(self.frames)[-n:]

    def execute(self, robot: RobotIO, frames: list[Frame]) -> None:
        # Roll back to recorded pre-action states using one-step delta commands.
        for frame in reversed(frames[:-1]):
            target = observation_state16(frame)
            if target is not None:
                robot.execute(delta_to_target(robot.state_action16(), target))
            if self.config.step_interval_s > 0:
                time.sleep(self.config.step_interval_s)
        start_state = observation_state16(frames[0]) if frames else None
        if start_state is not None:
            robot.execute(delta_to_target(robot.state_action16(), start_state))
            if self.config.step_interval_s > 0:
                time.sleep(self.config.step_interval_s)


def observation_state16(frame: Frame):
    observation = frame.observation
    if not isinstance(observation, dict):
        return None
    state = observation.get("state_action16")
    if state is None:
        history = observation.get("wam4d", {}).get("observation.state", [])
        if len(history):
            state = history[-1]
    if state is None:
        return None
    import numpy as np
    return np.asarray(state, np.float32).reshape(16)


def delta_to_target(current, target):
    import numpy as np
    current = np.asarray(current, np.float32).reshape(16)
    target = np.asarray(target, np.float32).reshape(16)
    delta = target.copy()
    delta[0:7] = relative_pose7(current[0:7], target[0:7])
    delta[8:15] = relative_pose7(current[8:15], target[8:15])
    return delta
