#!/usr/bin/env bash
# Find the USB serial port of a device (the arm, or the router controller).
# Follow the prompt: unplug it, press Enter, then plug it back in.
# Put the port it reports into config.yaml (arm.port / router.port).
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
lerobot-find-port
