#!/usr/bin/env bash
# Run the hardware collector after bringing up the Quest state bridge.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/_common.sh"

lock_file=${FLOWPRO_COLLECT_LOCK_FILE:-/tmp/flowpro-collector.lock}
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "Another FlowPRO collector is already running (lock: ${lock_file}). Stop it before starting a second collector." >&2
    exit 1
fi

"${script_dir}/quest_webxr.sh" --background
config_path=$(flowpro_config_path_from_args "$@")
quest_state_url=$(flowpro_config_value collection.quest_state_url "${config_path}")
for _ in {1..40}; do
    if curl --fail --silent --show-error --insecure --max-time 1 "${quest_state_url}" >/dev/null; then
        break
    fi
    sleep 0.25
done
if ! curl --fail --silent --show-error --insecure --max-time 2 "${quest_state_url}" >/dev/null; then
    echo "Quest WebXR state endpoint is not ready: ${quest_state_url}. See ${FLOWPRO_QUEST_LOG:-/tmp/flowpro-quest-webxr.log}" >&2
    exit 1
fi

flowpro_source_robot_environment
flowpro_run_stage "${FLOWPRO_ROBOT_PYTHON_DEFAULT}" collect "$@"
