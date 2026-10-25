#!/usr/bin/env bash
# Lock the overhead Arducam AR0234's exposure / white balance / gain.
# UVC settings are volatile — run this before EVERY run (the orchestrator does it in INIT).
# Same numbers as the Mac version (2_lock_camera.sh) and camera_settings.json.
#
#   ./vision-tools/2_lock_camera_linux.sh            # /dev/video0
#   ./vision-tools/2_lock_camera_linux.sh /dev/video2
#
# Newer kernels renamed the UVC controls (exposure_auto -> auto_exposure,
# exposure_absolute -> exposure_time_absolute); both spellings are tried.
set -u
DEV="${1:-/dev/video0}"

if ! command -v v4l2-ctl >/dev/null; then
  echo "v4l2-ctl missing: sudo apt install v4l-utils" >&2
  exit 1
fi

set_ctl() {  # set_ctl <value> <name> [<alt-name>...]
  local value="$1"; shift
  for name in "$@"; do
    if v4l2-ctl -d "$DEV" -c "${name}=${value}" 2>/dev/null; then
      return 0
    fi
  done
  echo "WARN: none of [$*] accepted on $DEV" >&2
  return 1
}

# 1 = manual exposure (legacy name) ; 1 is also "manual" for auto_exposure on most UVC cams
set_ctl 1    exposure_auto auto_exposure
set_ctl 83   exposure_absolute exposure_time_absolute
set_ctl 0    white_balance_temperature_auto white_balance_automatic
set_ctl 5000 white_balance_temperature
set_ctl 168  gain

echo "locked $DEV:"
v4l2-ctl -d "$DEV" --get-ctrl=exposure_absolute 2>/dev/null \
  || v4l2-ctl -d "$DEV" --get-ctrl=exposure_time_absolute
v4l2-ctl -d "$DEV" --get-ctrl=white_balance_temperature
v4l2-ctl -d "$DEV" --get-ctrl=gain
