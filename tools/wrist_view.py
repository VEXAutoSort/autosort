"""Live view of what the wrist camera counts as "pieces in the gripper".

Run:  python tools/wrist_view.py
No arm connection needed — hold the arm roughly at the inspect pose by hand.

Shows the wrist feed with:
  - the gripper_roi count region (orange box)
  - every counted blob (green circle + area)
  - the resulting count, big, top-left

Goal: empty gripper -> count 0, one held piece -> count 1.
If the gripper's own fingertips are being counted, shrink/move gripper_roi in
config.yaml (fractions [left, top, right, bottom] of the WRIST frame) until the
box only covers the space BETWEEN the fingertips. Restart this tool after each
config edit to see the new box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.config import Config          # noqa: E402
from autosort.perception import Perception  # noqa: E402


def main() -> None:
    cfg = Config.load()
    perception = Perception(cfg.perception, dry_run=False)
    wrist = cfg.cameras["wrist"]
    try:
        cap = wrist.verify_open("wrist")
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    print("Q quits. Edit perception.gripper_roi in config.yaml, restart to see changes.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blobs = perception._blobs(rgb, cfg.perception.gripper_roi)
        vis = frame.copy()
        h, w = vis.shape[:2]
        x0, y0, x1, y1 = cfg.perception.gripper_roi
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 150, 0), 2)
        for area, cx, cy in blobs:
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 255, 0), 3)
            cv2.putText(vis, f"{int(area)}", (int(cx) + 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"count: {len(blobs)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.imshow("wrist count debug (Q quits)", vis)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
