# no4d ablation configurations

The same experiment JSON is consumed by training and the inference server.
The FlowPRO project config selects the matching client execution semantics.
Relative paths are resolved from the repository working directory.

| Config | Action target | Action backbone | Auxiliary target | Views |
|---|---|---|---|---|
| `absolute.json` | absolute xyz+wxyz | shared video/action backbone | none | 3 |
| `delta.json` | delta xyz and `q_ref^-1 * q_target` | shared | none | 3 |
| `delta_action_head.json` | delta | separate 30-layer, 768-wide head | none | 3 |
| `delta_action_head_release.json` | delta | separate head | left/right release-relative pose in channels 14:28 | 3 |
| `crop_view.json` | absolute | shared | none | 5 |

Gripper targets remain absolute for every action representation. Robot-facing
commands remain the historical 16-D layout. Release-pose channels are enabled
only in the training loss and are removed by server execution postprocessing.
Executed-action history is always encoded as absolute cmd action in both modes.
`state_norm_stat` therefore retains the task's absolute statistics while a
delta experiment's `norm_stat` applies only to predicted action targets.

## Training

```bash
EXPERIMENT_CONFIG=wan_va/experiment_configs/delta.json \
  bash script/run_no4d_ablation_train.sh
```

For multi-node training, this launcher follows the existing repository
convention that `WORLD_SIZE` means node count:

```bash
WORLD_SIZE=2 RANK=0 NGPU=8 MASTER_ADDR=<node0-ip> \
  EXPERIMENT_CONFIG=wan_va/experiment_configs/delta.json \
  bash script/run_no4d_ablation_train.sh
```

Use `RANK=1` on the second node. `NUM_MACHINES` and `NNODES` take precedence
over `WORLD_SIZE`; `NPROC_PER_NODE` takes precedence over `NGPU`.

## Server and Astribot inference

```bash
EXPERIMENT_CONFIG=wan_va/experiment_configs/delta.json \
  bash script/run_launch_va_server_sync.sh

python evaluation/astribot/eval_astribot_openpi.py \
  --experiment-config wan_va/experiment_configs/delta.json \
  --task centrifuge --obs_history_len 1 <robot/init arguments>
```

The client automatically enables projected crop observations when the selected
config declares `crop_view_keys`, and automatically performs sequential delta
integration for delta configs. Do not pass `--use-xyzw`; dataset/model
quaternions use wxyz while the Astribot SDK conversion remains internal.

## Recomputing action normalization statistics

```bash
python script/compute_no4d_action_stats.py <dataset-root> \
  --representation delta --release-pose-aux
```

The checked-in delta statistics were computed from the current Astribot
keyframe manifests and dataset using q01/q99.
