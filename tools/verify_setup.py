"""Capture a standardized evidence pack for verifying the whole camera setup.

Run:  python tools/verify_setup.py          (LeLab quit, arm powered, table set
                                             up as for a real run, pieces optional)

Drives the arm through TAUGHT poses only and saves raw frames to /tmp/verify/:

  top_home_parked_{1..3}.png   top camera, arm at home, fingers parked closed
                               (the exact state detection runs in)
  top_home_open.png            same but claw at approach width - shows whether
                               the wide-open jaw intrudes into pile_roi
  wrist_inspect_empty_{1..5}.png  wrist camera at inspect, fingers closed, EMPTY
                               (false-positive hunting: all five must read 0)

Claude can read the saved files and do the analysis; it cannot open cameras.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.arm import Arm            # noqa: E402
from autosort.config import Config      # noqa: E402
from autosort.motion import move_smooth  # noqa: E402

OUT = Path("/tmp/verify")


def snap(arm: Arm, cam: str, path: Path) -> None:
    frame = arm.frame(cam)
    if frame is None:
        print(f"  !! no frame from '{cam}' for {path.name}")
        return
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # frames are RGB
    print(f"  saved {path.name}")


def main() -> None:
    cfg = Config.load()
    OUT.mkdir(exist_ok=True)
    arm = Arm(cfg.arm, cfg.cameras, dry_run=False)
    arm.connect()
    try:
        print("homing (fingers park closed)...")
        arm.home()
        time.sleep(1.0)
        for i in range(1, 4):
            snap(arm, "top", OUT / f"top_home_parked_{i}.png")
            time.sleep(0.3)

        print("opening claw to approach width at home...")
        move_smooth(arm.robot, {"gripper": arm.taught.gripper_open}, duration_s=0.4)
        time.sleep(0.5)
        snap(arm, "top", OUT / "top_home_open.png")
        move_smooth(arm.robot, {"gripper": arm.taught.gripper_closed}, duration_s=0.3)

        print("moving to inspect (fingers closed, empty)...")
        arm.move_to("inspect")
        time.sleep(1.0)
        for i in range(1, 6):
            snap(arm, "wrist", OUT / f"wrist_inspect_empty_{i}.png")
            time.sleep(0.25)

        print("returning home...")
        arm.home()
    finally:
        arm.disconnect()
    print(f"\ndone - evidence pack in {OUT}. Hand it to Claude for analysis.")


if __name__ == "__main__":
    main()
