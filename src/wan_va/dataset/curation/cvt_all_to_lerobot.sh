#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
source /media/damoxing/fileset/conda/etc/profile.d/conda.sh && conda activate lingbot-va

if ! command -v ffmpeg >/dev/null 2>&1; then
  cp /etc/apt/sources.list /etc/apt/sources.list.bak && sed -i 's/archive.ubuntu.com/mirrors.baidubce.com/g' /etc/apt/sources.list && sed -i 's/security.ubuntu.com/mirrors.baidubce.com/g' /etc/apt/sources.list && apt update -y
  apt-get install -y ffmpeg
fi

MAIN_HEIGHT=${MAIN_HEIGHT:-480}
MAIN_WIDTH=${MAIN_WIDTH:-720}
WRIST_HEIGHT=${WRIST_HEIGHT:-240}
WRIST_WIDTH=${WRIST_WIDTH:-360}
ASTRIBOT_HDF5_ROOT=${ASTRIBOT_HDF5_ROOT:-/media/damoxing/datasets/astribot_tasks/myendless}
ASTRIBOT_MP4_ROOT=${ASTRIBOT_MP4_ROOT:-${ASTRIBOT_HDF5_ROOT}}
ASTRIBOT_OUTPUT_ROOT=${ASTRIBOT_OUTPUT_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org}
ASTRIBOT_FILTER_ROOT=${ASTRIBOT_FILTER_ROOT:-/media/damoxing/fileset/data_quality_check/outputs}
ASTRIBOT_MAX_FILTER=${ASTRIBOT_MAX_FILTER:-800}
ASTRIBOT_FILTERS=${ASTRIBOT_FILTERS:-800 500 300}
ASTRIBOT_ENABLE_FILTER_REUSE=${ASTRIBOT_ENABLE_FILTER_REUSE:-0}
ASTRIBOT_RUN_MP4_TO_HDF5=${ASTRIBOT_RUN_MP4_TO_HDF5:-0}
ASTRIBOT_HDF5_WORKERS=${ASTRIBOT_HDF5_WORKERS:-64}
ASTRIBOT_HDF5_EXTRA_ARGS=${ASTRIBOT_HDF5_EXTRA_ARGS:-}
ASTRIBOT_REPO_PREFIX=${ASTRIBOT_REPO_PREFIX:-astribot_filter}

EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --keep-original-resolution|--main-only|--resume-wrist)
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

max_list="${ASTRIBOT_FILTER_ROOT}/astribot_filter_${ASTRIBOT_MAX_FILTER}/step2_fps_sampled_hdf5_files.txt"
max_repo="${ASTRIBOT_OUTPUT_ROOT}/astribot/${ASTRIBOT_REPO_PREFIX}_${ASTRIBOT_MAX_FILTER}"

if [[ "${ASTRIBOT_RUN_MP4_TO_HDF5}" == "1" ]]; then
  hdf5_extra_args=()
  if [[ -n "${ASTRIBOT_HDF5_EXTRA_ARGS}" ]]; then
    read -r -a hdf5_extra_args <<< "${ASTRIBOT_HDF5_EXTRA_ARGS}"
  fi
  python wan_va/dataset/curation/convert_mp4_to_hdf5.py \
    --input-root "${ASTRIBOT_MP4_ROOT}" \
    --output-root "${ASTRIBOT_HDF5_ROOT}" \
    --input-list "${max_list}" \
    --workers "${ASTRIBOT_HDF5_WORKERS}" \
    "${hdf5_extra_args[@]}"
fi

for filter_size in ${ASTRIBOT_FILTERS}; do
  list_path="${ASTRIBOT_FILTER_ROOT}/astribot_filter_${filter_size}/step2_fps_sampled_hdf5_files.txt"
  repo_name="${ASTRIBOT_REPO_PREFIX}_${filter_size}"
  reuse_args=()
  if [[ "${ASTRIBOT_ENABLE_FILTER_REUSE}" == "1" && "${filter_size}" != "${ASTRIBOT_MAX_FILTER}" ]]; then
    reuse_args+=(--astribot-reuse-from "${max_repo}" --astribot-link-stats)
  fi

  python wan_va/dataset/curation/convert_raw_to_lerobot.py \
    --max-episodes 0 \
    --output-root "${ASTRIBOT_OUTPUT_ROOT}" \
    --datasets astribot \
    --input-roots "astribot=${ASTRIBOT_HDF5_ROOT}" \
    --output-names "astribot=${repo_name}" \
    --astribot-episode-list "${list_path}" \
    --main-height "${MAIN_HEIGHT}" \
    --main-width "${MAIN_WIDTH}" \
    --wrist-height "${WRIST_HEIGHT}" \
    --wrist-width "${WRIST_WIDTH}" \
    --overwrite \
    --skip-image-stats \
    "${reuse_args[@]}" \
    "${EXTRA_ARGS[@]}"
done

# astribot,bridge,droid,libero,robotwin,rt1,songling
