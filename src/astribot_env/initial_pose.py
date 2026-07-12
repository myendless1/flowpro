from __future__ import annotations

from math import isfinite
from typing import Sequence


# The six groups match Astribot.move_joints_position(non_chassis_names, targets):
# torso, left arm, left gripper, right arm, right gripper, head.
INITIAL_JOINT_GROUP_SIZES = (4, 7, 1, 7, 1, 2)
DEFAULT_INIT_JOINT_ACTION = (
    (0.5863, -1.1816, 0.5947, -0.0006),
    (0.2850, -0.3639, -1.2369, 1.6490, -0.3651, -0.0550, -0.2365),
    (0.0,),
    (-0.9200, -0.4720, 1.6216, 1.9225, 0.4911, 0.0491, 0.4409),
    (0.0,),
    (-0.0064, 0.8870),
)


def normalize_init_joint_action(value: Sequence[Sequence[float]]) -> list[list[float]]:
    """Validate and copy a non-chassis Astribot joint target."""
    if isinstance(value, (str, bytes)) or len(value) != len(INITIAL_JOINT_GROUP_SIZES):
        raise ValueError(
            "init_joint_action must contain six groups: "
            "torso(4), left_arm(7), left_gripper(1), right_arm(7), "
            "right_gripper(1), head(2)."
        )

    target: list[list[float]] = []
    for index, (group, expected_size) in enumerate(zip(value, INITIAL_JOINT_GROUP_SIZES)):
        if isinstance(group, (str, bytes)) or len(group) != expected_size:
            actual_size = len(group) if not isinstance(group, (str, bytes)) else "a string"
            raise ValueError(
                f"init_joint_action group {index} must contain {expected_size} values, got {actual_size}."
            )
        normalized = [float(item) for item in group]
        if not all(isfinite(item) for item in normalized):
            raise ValueError(f"init_joint_action group {index} contains a non-finite value.")
        target.append(normalized)
    return target


def default_init_joint_action() -> list[list[float]]:
    """Return a fresh copy of the shared, cross-task reset pose."""
    return normalize_init_joint_action(DEFAULT_INIT_JOINT_ACTION)
