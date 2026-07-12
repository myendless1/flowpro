#!/usr/bin/env bash
# Install the one-time udev rule that allows the plugdev group to use Quest ADB.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Installing the Quest ADB udev rule requires administrator permission; re-running with sudo." >&2
    exec sudo -- "$0" "$@"
fi

rule_path=/etc/udev/rules.d/51-quest-adb.rules
printf '%s\n' \
    'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \
    >"${rule_path}"
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=2833
echo "Installed ${rule_path}. Reconnect the Quest USB cable, accept its USB-debugging RSA prompt, then run: adb devices -l"
