#!/usr/bin/env bash
# Shared shell helpers for FlowPRO stage entry points.

set -euo pipefail

FLOWPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FLOWPRO_CONFIG_DEFAULT="${FLOWPRO_CONFIG:-${FLOWPRO_ROOT}/configs/flowpro.json}"
FLOWPRO_TRAIN_PYTHON_DEFAULT="${FLOWPRO_TRAIN_PYTHON:-/home/xddex05/miniconda3/envs/lingbot/bin/python}"
FLOWPRO_ROBOT_PYTHON_DEFAULT="${FLOWPRO_ROBOT_PYTHON:-/home/xddex05/miniconda3/envs/astribot/bin/python}"
FLOWPRO_QUEST_ROOT_DEFAULT="${FLOWPRO_QUEST_ROOT:-/home/xddex05/repo/quest-server}"

flowpro_require_file() {
    local path=$1
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        return 1
    fi
}

flowpro_has_config_arg() {
    local arg
    for arg in "$@"; do
        [[ "${arg}" == "--config" || "${arg}" == --config=* ]] && return 0
    done
    return 1
}

flowpro_config_path_from_args() {
    local arg
    local next_is_config=0
    for arg in "$@"; do
        if ((next_is_config)); then
            printf '%s\n' "${arg}"
            return 0
        fi
        case "${arg}" in
            --config) next_is_config=1 ;;
            --config=*) printf '%s\n' "${arg#--config=}"; return 0 ;;
        esac
    done
    if ((next_is_config)); then
        echo "--config requires a path" >&2
        return 2
    fi
    printf '%s\n' "${FLOWPRO_CONFIG_DEFAULT}"
}

flowpro_run_stage() {
    local python=$1
    local stage=$2
    shift 2
    flowpro_require_file "${python}"
    flowpro_require_file "${FLOWPRO_ROOT}/src"
    export PYTHONPATH="${FLOWPRO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${FLOWPRO_ROOT}"
    if flowpro_has_config_arg "$@"; then
        exec "${python}" -m flowpro.workflow "${stage}" "$@"
    fi
    exec "${python}" -m flowpro.workflow "${stage}" --config "${FLOWPRO_CONFIG_DEFAULT}" "$@"
}

flowpro_source_robot_environment() {
    local sdk_root=${ASTRIBOT_SDK_ROOT:-/home/xddex05/repo/astribot_sdk}
    flowpro_require_file "/opt/ros/noetic/setup.bash"
    flowpro_require_file "${sdk_root}/env.sh"
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash
    # Astribot's env.sh tests $ZSH_VERSION and $ROBOT_TYPE without defaults.
    # It is meant to be sourced by an interactive shell, so temporarily relax
    # nounset while importing it into these strict-mode scripts.
    set +u
    # shellcheck disable=SC1090
    source "${sdk_root}/env.sh"
    set -u
}

flowpro_config_value() {
    local dotted_key=$1
    local config_path=${2:-${FLOWPRO_CONFIG_DEFAULT}}
    local python=${3:-${FLOWPRO_TRAIN_PYTHON_DEFAULT}}
    flowpro_require_file "${python}"
    "${python}" - "${config_path}" "${dotted_key}" <<'PY'
import json
import sys

config_path, dotted_key = sys.argv[1:]
with open(config_path, encoding="utf-8") as fh:
    value = json.load(fh)
for part in dotted_key.split("."):
    value = value[part]
print(value)
PY
}
