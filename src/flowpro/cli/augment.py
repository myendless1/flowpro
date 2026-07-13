from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from flowpro.augmentation import InterpolationConfig, augment_pair
from flowpro.config import load_config
from flowpro.data import PairStore
from flowpro.data.store import _split_arrays


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--config",required=True); p.add_argument("--round",type=int,required=True); a=p.parse_args()
    cfg=load_config(a.config); c=InterpolationConfig(**cfg.section("augmentation")); source=PairStore(a.input)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); count=0
    for pair_dir in sorted(Path(a.input).iterdir() if Path(a.input).exists() else []):
        if not pair_dir.is_dir() or not (pair_dir/"metadata.json").exists(): continue
        for i,sample in enumerate(augment_pair(source.load(pair_dir),c)):
            observation, observation_arrays = _split_arrays(sample.observation, "observation")
            np.savez_compressed(out/f"{pair_dir.name}-{i:05d}.npz",winner=sample.winner,loser=sample.loser,
                                **observation_arrays)
            (out/f"{pair_dir.name}-{i:05d}.json").write_text(json.dumps({"source":sample.source,"pair_id":sample.pair_id,
                "observation":observation},default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
            count+=1
    (out/"manifest.json").write_text(json.dumps({
        "round": a.round,
        "samples": count,
        "config": str(cfg.path),
        "action_representation": cfg.section("model").get("action_representation", "delta"),
        "history_action_representation": "absolute",
    }, indent=2))
    print(f"wrote {count} preference samples to {out}")

if __name__=="__main__": main()
