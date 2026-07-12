#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

source /media/damoxing/fileset/conda/etc/profile.d/conda.sh
conda activate da3

apt update -y
apt-get install -y ffmpeg libgl1-mesa-glx
python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("h5py") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py"])
PY

DATASET_ROOT=${DATASET_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org}
OUTPUT_ROOT=${OUTPUT_ROOT:-/media/damoxing/datasets/vae4d/da3-vae4d-org}
MODEL_PATH=${MODEL_PATH:-/media/damoxing/ckp/Depth-Anything-3/DA3NESTED-GIANT-LARGE-1.1}
SCRIPT_PATH=${SCRIPT_PATH:-wan_va/dataset/curation/build_3d_data_da3.py}

DEVICE=${DEVICE:-cuda}
PROCESS_RES=${PROCESS_RES:-504}
PROCESS_RES_METHOD=${PROCESS_RES_METHOD:-upper_bound_resize}
REF_VIEW_STRATEGY=${REF_VIEW_STRATEGY:-middle}
MAX_REPOS=${MAX_REPOS:-0}
MAX_EPISODES=${MAX_EPISODES:-0}
FAMILIES=${FAMILIES:-}
OVERWRITE=${OVERWRITE:-1}
NUM_GPUS=${NUM_GPUS:-}
TOTAL_MACHINES=${TOTAL_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "missing script: $SCRIPT_PATH" >&2
  exit 1
fi

if [[ -z "$FAMILIES" ]]; then
  echo "missing FAMILIES, expected comma-separated first-level directories under DATASET_ROOT" >&2
  exit 1
fi

if ! [[ "$TOTAL_MACHINES" =~ ^[0-9]+$ ]] || [[ "$TOTAL_MACHINES" -le 0 ]]; then
  echo "invalid TOTAL_MACHINES=$TOTAL_MACHINES, expected positive integer" >&2
  exit 1
fi

if ! [[ "$MACHINE_RANK" =~ ^[0-9]+$ ]] || [[ "$MACHINE_RANK" -lt 0 ]] || [[ "$MACHINE_RANK" -ge "$TOTAL_MACHINES" ]]; then
  echo "invalid MACHINE_RANK=$MACHINE_RANK for TOTAL_MACHINES=$TOTAL_MACHINES, expected 0 <= MACHINE_RANK < TOTAL_MACHINES" >&2
  exit 1
fi

if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
fi

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --model-path "$MODEL_PATH"
  --device "$DEVICE"
  --process-res "$PROCESS_RES"
  --process-res-method "$PROCESS_RES_METHOD"
  --ref-view-strategy "$REF_VIEW_STRATEGY"
  --max-repos "$MAX_REPOS"
  --max-episodes "$MAX_EPISODES"
  --num-gpus "$NUM_GPUS"
  --families "$FAMILIES"
  --total-machines "$TOTAL_MACHINES"
  --machine-rank "$MACHINE_RANK"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "$SCRIPT_PATH" "${ARGS[@]}"
