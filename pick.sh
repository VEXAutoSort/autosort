#!/bin/bash
# One pick per Enter. Run from any SSH session on the GPU box:
#     ~/autosort/pick.sh              (step mode: pick -> home -> wait for Enter)
#     ~/autosort/pick.sh --continuous (clear the zone, then exit)
# Ctrl-C at any time: the arm returns home and disconnects.
cd "$(dirname "$0")" || exit 1
if pgrep -f "python.*run.py" >/dev/null || pgrep -f "tools/teach.py" >/dev/null; then
  echo "another AutoSort run/teach is still holding the arm - stop it first (tmux attach -t autosort)"; exit 1
fi
source ~/autosort-train/.venv/bin/activate
mkdir -p /tmp/autosort_debug
mode=--step; [ "$1" = "--continuous" ] && mode=--continuous
exec python -u run.py $mode 2>&1 | grep --line-buffered -v Warning | tee "/tmp/autosort_debug/run_$(date +%H%M%S).log"
