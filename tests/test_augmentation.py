import numpy as np
from flowpro.augmentation import InterpolationConfig, augment_pair
from flowpro.data import Frame, TrajectoryPair


def action(x):
    a=np.zeros(16,np.float32); a[[3,11]]=1; a[0]=x; a[8]=x; return a


def test_dense_pairs_and_identical_winner_samples():
    loser=[Frame({"i":i},action(i*.1)) for i in range(3)]
    winner=[Frame({"i":i},action(i*.05)) for i in range(4)]
    out=augment_pair(TrajectoryPair("x",loser,winner,0),InterpolationConfig(horizon=3))
    # Only loser states with a complete, unpadded H-step negative chunk are
    # eligible; the dangerous tail is excluded (paper Appendix E).
    assert len(out)==5
    assert all(x.winner.shape==(3,16) for x in out)
    assert all(np.array_equal(x.winner,x.loser) for x in out if x.source=="positive")
