"""Wrist-counter diagnostic: see what it thinks, and capture labeled frames.

Run:  python tools/wrist_diag.py

Live window shows the ROI box, every counted blob (green + area), and the count.
Capture labeled frames so the ROI can be placed from real evidence:

    O = open fingers (to insert/remove a gear)
    C = CLOSE fingers - this is the state a real run is in at the inspect pose
    E = capture with gripper closed and EMPTY
    H = capture with gripper closed on a GEAR
    Q = quit

Saves raw + annotated frames to /tmp/wrist_diag/. Take a few of each (move the
arm a little between captures so we see the range, not one lucky frame).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.config import Config          # noqa: E402
from autosort.motion import move_smooth, read_joints  # noqa: E402
from autosort.perception import Perception  # noqa: E402

OUTDIR = Path("/tmp/wrist_diag")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    for old in OUTDIR.glob("*.png"):
        old.unlink()
    cfg = Config.load()
    perception = Perception(cfg.perception, dry_run=False)

    # drive the gripper from here so the camera sees the SAME states a real run
    # produces: at the inspect pose the fingers are CLOSED (on a gear or empty).
    import json
    taught = json.loads((ROOT / "taught.json").read_text())
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
    robot = SO101Follower(SO101FollowerConfig(port=cfg.arm.port, id=cfg.arm.id,
                                              disable_torque_on_disconnect=False))
    for attempt in range(4):
        try:
            robot.connect(calibrate=False)
            break
        except ConnectionError:
            if attempt == 3:
                raise
            time.sleep(2)
    move_smooth(robot, {"gripper": taught["gripper_closed"]}, duration_s=0.6)

    wrist = cfg.cameras["wrist"]
    cap = wrist.verify_open("wrist")
    roi = cfg.perception.gripper_roi
    print(f"ROI={roi}  contrast_margin={cfg.perception.contrast_margin}  "
          f"min_area={cfg.perception.min_piece_area}")
    print("E = capture EMPTY,  H = capture HOLDING a gear,  Q = quit", flush=True)

    counts = {"empty": 0, "held": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blobs = perception._blobs(rgb, roi)
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        vis = frame.copy()
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 150, 0), 2)
        for area, cx, cy in blobs:
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 255, 0), 3)
            cv2.putText(vis, str(int(area)), (int(cx) + 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"count: {len(blobs)}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        try:
            gpos = read_joints(robot)["gripper"]
        except Exception:
            gpos = float("nan")
        cv2.putText(vis, f"gripper {gpos:5.1f}   saved empty:{counts['empty']} held:{counts['held']}   O/C/E/H/Q",
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("wrist diagnostic (E=empty  H=held  Q=quit)", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("o"):
            move_smooth(robot, {"gripper": taught["gripper_open"]}, duration_s=0.5)
            print(f"fingers opened to {taught['gripper_open']}", flush=True)
        elif key == ord("c"):
            move_smooth(robot, {"gripper": taught["gripper_closed"]}, duration_s=0.6)
            time.sleep(0.4)
            print(f"fingers closed (commanded {taught['gripper_closed']}, "
                  f"actual {read_joints(robot)['gripper']:.1f})", flush=True)
        elif key in (ord("e"), ord("h")):
            label = "empty" if key == ord("e") else "held"
            counts[label] += 1
            n = counts[label]
            cv2.imwrite(str(OUTDIR / f"{label}_{n:02d}_raw.png"), frame)
            cv2.imwrite(str(OUTDIR / f"{label}_{n:02d}_annotated.png"), vis)
            print(f"captured {label} #{n}  (count {len(blobs)}, gripper {gpos:.1f})", flush=True)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()
    print(f"\nsaved {counts['empty']} empty + {counts['held']} held frames to {OUTDIR}")


if __name__ == "__main__":
    main()
