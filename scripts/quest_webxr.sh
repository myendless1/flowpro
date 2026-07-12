#!/usr/bin/env bash
# Start the Quest WebXR state bridge used by the FlowPRO collector.
# The bridge runs in --test mode: only the collector may command the robot.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

background=0
while (($#)); do
    case "$1" in
        --background) background=1; shift ;;
        --help)
            cat <<'EOF'
Usage: scripts/quest_webxr.sh [--background]

Checks the USB-debugging connection, maps Quest localhost:8443 to the host,
then runs the HTTPS WebXR bridge in dry-run mode. Set ADB_SERIAL when more
than one device is connected. Use FLOWPRO_QUEST_PYTHON to override Python.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

adb_bin=${ADB:-adb}
quest_port=${QUEST_HOST_PORT:-8443}
device_port=${QUEST_DEVICE_PORT:-8443}
quest_python=${FLOWPRO_QUEST_PYTHON:-${FLOWPRO_ROBOT_PYTHON_DEFAULT}}
log_file=${FLOWPRO_QUEST_LOG:-/tmp/flowpro-quest-webxr.log}
pid_file=${FLOWPRO_QUEST_PID_FILE:-/tmp/flowpro-quest-webxr.pid}

flowpro_require_file "${quest_python}"
flowpro_require_file "${FLOWPRO_QUEST_ROOT_DEFAULT}/quest_server/server.py"

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
    adb_args=(-s "${ADB_SERIAL}")
fi
if ! "${adb_bin}" "${adb_args[@]}" get-state 2>/dev/null | grep -qx 'device'; then
    echo "Quest ADB device is unavailable. Enable Developer Mode and USB debugging, accept the RSA prompt in the headset, then run sudo ${FLOWPRO_ROOT}/scripts/setup_quest_adb.sh if ADB reports 'no permissions'. Current devices:" >&2
    "${adb_bin}" devices -l >&2 || true
    exit 1
fi
"${adb_bin}" "${adb_args[@]}" reverse "tcp:${device_port}" "tcp:${quest_port}"
echo "ADB reverse active: Quest https://localhost:${device_port} -> host tcp:${quest_port}"

flowpro_source_robot_environment
export PYTHONPATH="${FLOWPRO_QUEST_ROOT_DEFAULT}${PYTHONPATH:+:${PYTHONPATH}}"
command=("${quest_python}" -m quest_server.server
    --host 0.0.0.0 --port "${quest_port}"
    --adb "${adb_bin}" --adb-device-port "${device_port}"
    --astribot-sdk "${ASTRIBOT_SDK_ROOT:-/home/xddex05/repo/astribot_sdk}"
    --test --body-realign-button-index 99)
if [[ -n "${ADB_SERIAL:-}" ]]; then
    command+=(--adb-serial "${ADB_SERIAL}")
fi

if ((background)); then
    if [[ -r "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        echo "Quest WebXR bridge already running (pid $(<"${pid_file}"), log ${log_file})"
        exit 0
    fi
    nohup "${command[@]}" >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
    echo "Quest WebXR bridge started (pid $!, log ${log_file})"
else
    exec "${command[@]}"
fi
