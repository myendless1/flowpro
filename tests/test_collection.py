import numpy as np
import pytest
from flowpro.collection import InputState, InterventionCollector, Phase
from flowpro.collection.rollback import RollbackConfig
from flowpro.data import PairStore


class Robot:
    def __init__(self): self.action = np.r_[np.zeros(3), [1,0,0,0], 0, np.zeros(3), [1,0,0,0], 0].astype(np.float32); self.n=0
    def observe(self): self.n += 1; return {"step": self.n}
    def state_action16(self): return self.action.copy()
    def execute(self, action): self.action = np.asarray(action).copy()


class Policy:
    def reset(self, observation): pass
    def infer(self, observation): return np.tile(Robot().action, (2,1))


def test_b_middle_a_commits_atomic_pair(tmp_path):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=2))
    c.tick(InputState()); c.tick(InputState())
    assert c.tick(InputState(b=True)) is Phase.ROLLED_BACK
    assert c.tick(InputState(middle=1, expert_action=robot.action)) is Phase.TAKEOVER
    assert c.tick(InputState(a=True)) is Phase.POLICY
    target = next(tmp_path.iterdir())
    assert (target / "winner.npz").exists() and (target / "loser.npz").exists()


def test_a_before_takeover_is_rejected(tmp_path):
    robot = Robot(); c = InterventionCollector(robot, Policy(), PairStore(tmp_path), rollback=RollbackConfig(default_horizon=1))
    c.tick(InputState()); c.tick(InputState(b=True)); c.tick(InputState())
    with pytest.raises(RuntimeError, match="middle-trigger"):
        c.tick(InputState(a=True))
