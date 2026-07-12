from __future__ import annotations
import numpy as np


def batch_counts(batch_size: int, round_id: int) -> dict[str, int]:
    ratios = {"current": .8, "sft": .2} if round_id == 1 else {"current": .7, "history": .15, "sft": .15}
    raw = {k: batch_size*v for k, v in ratios.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda x: raw[x]-counts[x], reverse=True)[:batch_size-sum(counts.values())]: counts[k] += 1
    return counts


class MixedBatchSampler:
    def __init__(self, current, history, sft, *, round_id: int, seed: int = 0):
        self.sources = {"current": current, "history": history, "sft": sft}
        self.round_id, self.rng = round_id, np.random.default_rng(seed)

    def sample(self, batch_size: int):
        result = []
        for key, count in batch_counts(batch_size, self.round_id).items():
            source = self.sources[key]
            if not source: raise ValueError(f"Required batch source {key!r} is empty")
            result.extend(source[int(i)] for i in self.rng.integers(0, len(source), count))
        self.rng.shuffle(result)
        return result

