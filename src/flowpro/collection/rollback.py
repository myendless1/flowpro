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
        targets = [rollback_target16(frame) for frame in frames]
        start_state = frame_start16(frames[0]) if frames else None
        execute_waypoints = getattr(robot, "execute_rollback_waypoints", None)
        if (
            callable(execute_waypoints)
            and start_state is not None
            and targets
            and all(target is not None for target in targets)
        ):
            reverse_targets = [target for target in reversed(targets[:-1])]
            reverse_targets.append(start_state)
            import numpy as np
            execute_waypoints(
                np.asarray(reverse_targets, np.float32),
                step_duration_s=max(float(self.config.step_interval_s), 1e-3),
            )
            return

        # Compatibility path for RobotIO implementations without waypoint support.
        for frame in reversed(frames[:-1]):
            target = rollback_target16(frame)
            if target is not None:
                execute_absolute = getattr(robot, "execute_absolute", None)
                if callable(execute_absolute):
                    execute_absolute(target)
                else:
                    robot.execute(delta_to_target(robot.state_action16(), target))
            if self.config.step_interval_s > 0:
                time.sleep(self.config.step_interval_s)
        if start_state is not None:
            execute_absolute = getattr(robot, "execute_absolute", None)
            if callable(execute_absolute):
                execute_absolute(start_state)
            else:
                robot.execute(delta_to_target(robot.state_action16(), start_state))
            if self.config.step_interval_s > 0:
                time.sleep(self.config.step_interval_s)


def rollback_target16(frame: Frame):
    observation = frame.observation
    if not isinstance(observation, dict):
        return None
    state = observation.get("_flowpro_rollback_target16")
    if state is None:
        state = observation.get("state_action16")
    if state is None:
        history = observation.get("wam4d", {}).get("observation.state", [])
        if len(history):
            state = history[-1]
    if state is None:
        return None
    import numpy as np
    return np.asarray(state, np.float32).reshape(16)


def frame_start16(frame: Frame):
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
