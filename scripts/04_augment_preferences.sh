#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
flowpro_run_stage "${FLOWPRO_TRAIN_PYTHON_DEFAULT}" augment "$@"
