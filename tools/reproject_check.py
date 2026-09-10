"""Where do OLD taught poses land in the NEW camera? (after a rig move)

Run:  python tools/reproject_check.py taught_backup_X.json 2,11,22,29,38 [--go]

For each listed grid point (1-based, as printed by validate_taught.py) the arm
goes home -> hover above the recorded pose -> descends to the recorded grasp
pose in steps, watching joint tracking error (a table at a different height
stalls the servos: abort + lift) -> snaps the top camera with the RECORDED
pixel drawn as a red cross -> lifts. Frames go to /tmp/autosort_debug/reproj_*.png.
Compare the fingertip position in the image to the cross: a consistent
shift/scale means the old grid can be re-registered instead of re-taught.

Without --go it only prints the plan (no motion).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.analytic import AnalyticSolver, UnsafePoseError  # noqa: E402
from autosort.config import Config                             # noqa: E402
from autosort.motion import JOINTS, move_smooth, read_joints   # noqa: E402
from autosort.taught import Taught                             # noqa: E402

OUT = Path("/tmp/autosort_debug")
TRACK_ERR_DEG = 5.0   # goal-vs-present on shoulder_lift/elbow_flex/wrist_flex above this = contact


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    go = "--go" in sys.argv
    data = json.loads(Path(args[0]).read_text())
    idx = [int(i) for i in args[1].split(",")]
    cfg = Config.load()
    taught = Taught(data)
    solver = AnalyticSolver(taught, urdf_path=cfg.arm.urdf_path)
    plan = []
    for i in idx:
        g = data["grid"][i - 1]
        grasp = dict(g["joints"])
        hover = solver.hover_for(grasp, cfg.arm.hover_lift_m)   # guarded
        plan.append((i, g["pixel"], grasp, hover))
        print(f"#{i}: recorded pixel ({g['pixel'][0]:.0f},{g['pixel'][1]:.0f})  "
              f"grasp {[round(grasp[j]) for j in JOINTS[:4]]}  hover {[round(hover[j]) for j in JOINTS[:4]]}")
    if not go:
        print("(plan only - add --go to move the arm)")
        return

    OUT.mkdir(exist_ok=True)
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    robot = SOFollower(SO101FollowerConfig(port=cfg.arm.port, id=cfg.arm.id))
    robot.connect(calibrate=False)
    cap = cfg.cameras["top"].verify_open("top")

    def snap(name):
        for _ in range(6):
            ok, f = cap.read()
        return f if ok else None

    def descend(grasp, steps=4):
        """hover -> grasp in steps; abort on tracking error (= hit something)."""
        cur = read_joints(robot)
        for s in range(1, steps + 1):
            a = s / steps
            tgt = {j: cur[j] + a * (grasp[j] - cur[j]) for j in JOINTS[:5]}
            move_smooth(robot, tgt, duration_s=0.5)
            time.sleep(0.25)
            now = read_joints(robot)
            err = {j: abs(now[j] - tgt[j]) for j in ("shoulder_lift", "elbow_flex", "wrist_flex")}
            if max(err.values()) > TRACK_ERR_DEG:
                print(f"   CONTACT? tracking error {err} at step {s}/{steps} - stopping descent here")
                return False
        return True

    home = data["poses"]["home"]
    open_g = data["gripper_open"]
    try:
        print("home...")
        move_smooth(robot, {**{j: home[j] for j in JOINTS[:5]}, "gripper": open_g}, duration_s=2.0)
        for i, px, grasp, hover in plan:
            print(f"#{i}: hover")
            move_smooth(robot, {**{j: hover[j] for j in JOINTS[:5]}, "gripper": open_g}, duration_s=1.8)
            reached = descend(grasp)
            time.sleep(0.5)
            f = snap(i)
            j = read_joints(robot)
            if f is not None:
                vis = f.copy()
                cv2.drawMarker(vis, (int(px[0]), int(px[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
                cv2.putText(vis, f"#{i} recorded px ({px[0]:.0f},{px[1]:.0f}) {'' if reached else 'STOPPED EARLY'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(str(OUT / f"reproj_{i:02d}.png"), vis)
            print(f"   at {[round(j[k], 1) for k in JOINTS[:4]]}  saved reproj_{i:02d}.png")
            move_smooth(robot, {j: hover[j] for j in JOINTS[:5]}, duration_s=1.2)
        print("home...")
        move_smooth(robot, {j: home[j] for j in JOINTS[:5]}, duration_s=2.0)
        move_smooth(robot, {"gripper": data["gripper_closed"] + 3.0}, duration_s=0.4)
    finally:
        cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
