from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from flowpro.config import load_config
from flowpro.training.mixer import batch_counts


def _validate(spec):
    missing=[]
    current=Path(spec["current_preferences"])
    if not any(current.glob("*.npz")): missing.append(f"preference samples: {current}")
    for path in spec["historical_preferences"]:
        if not any(Path(path).glob("*.npz")): missing.append(f"historical samples: {path}")
    sft=Path(spec["sft_dataset"])
    if not sft.is_dir(): missing.append(f"SFT dataset: {sft}")
    if not (sft/"empty_emb.pt").is_file(): missing.append(f"SFT text embedding: {sft/'empty_emb.pt'}")
    base=Path(spec["base_checkpoint"])
    for name in ("vae","tokenizer","text_encoder"):
        if not (base/name).is_dir(): missing.append(f"base checkpoint component: {base/name}")
    ref=Path(spec["reference_checkpoint"])
    if not any((p/"transformer").is_dir() for p in (ref,ref/"checkpoints"/"last")):
        missing.append(f"reference transformer: {ref}")
    if missing: raise FileNotFoundError("RPRO preflight failed:\n- " + "\n- ".join(missing))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--round",type=int,required=True)
    p.add_argument("--reference",required=True); p.add_argument("--output",required=True); p.add_argument("--steps",type=int,required=True); p.add_argument("--batch-size",type=int,required=True)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); cfg=load_config(a.config); current=cfg.round_dir(a.round)/"preference_dataset"
    if not current.exists(): raise FileNotFoundError(f"Run augmentation first: {current}")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    spec={"round":a.round,"reference_checkpoint":str(Path(a.reference).resolve()),"current_preferences":str(current),
          "base_checkpoint":str(cfg.path_for("model.base_checkpoint")),
          "historical_preferences":[str(cfg.round_dir(i)/"preference_dataset") for i in range(1,a.round)],
          "sft_dataset":str(cfg.path_for("paths.sft_dataset")),"output":str(out.resolve()),"steps":a.steps,
          "batch_size":a.batch_size,"batch_counts":batch_counts(a.batch_size,a.round),**cfg.section("offline_rl")}
    spec_file = out/"training_spec.json"
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        spec_file.write_text(json.dumps(spec,indent=2))
    else:
        deadline = time.monotonic() + 60
        while not spec_file.is_file() and time.monotonic() < deadline:
            time.sleep(.1)
        if not spec_file.is_file():
            raise TimeoutError(f"Timed out waiting for rank 0 to write {spec_file}")
    _validate(spec)
    
    if a.dry_run:
        print(f"validated RPRO training specification: {spec_file}")
        return
    from flowpro.training.rpro_trainer import run_rpro
    run_rpro(spec_file, config_name=cfg.section("pretrain")["config_name"],
             experiment_config=str(cfg.path_for("model.experiment_config")))

if __name__=="__main__": main()
