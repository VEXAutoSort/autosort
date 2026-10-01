#!/usr/bin/env python3
"""Calibrate the TOP-cam pixel -> table(mm) homography for geometric grasping.

Place 4 markers on the table at known positions (measure them in the arm's frame),
put those millimetre coordinates in TABLE_POINTS below, run this, and click the 4
markers in the SAME order in the image window. It prints the 3x3 matrix to paste
into config.yaml under grasp.homography.

Usage:  python scripts/calibrate_homography.py            # uses cameras.top from config.yaml
        python scripts/calibrate_homography.py --index 0  # or force a camera index
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

# EDIT THESE: the 4 markers' table coordinates (mm), in the click order you'll use.
TABLE_POINTS = [
    (0.0, 0.0),
    (200.0, 0.0),
    (200.0, 150.0),
    (0.0, 150.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="camera index (default: cameras.top from config)")
    args = ap.parse_args()

    index = args.index
    if index is None:
        from autosort.config import Config
        index = Config.load().cameras["top"].index

    cap = cv2.VideoCapture(index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read from camera index {index}")

    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(len(clicks)), (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print(f"Click the 4 markers in this order: {TABLE_POINTS}")
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while len(clicks) < 4:
        cv2.imshow("calibrate", frame)
        if cv2.waitKey(20) == 27:  # Esc
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()

    H = cv2.getPerspectiveTransform(
        np.array(clicks, dtype=np.float32), np.array(TABLE_POINTS, dtype=np.float32)
    )
    print("\nPaste into config.yaml under grasp.homography:\n")
    print("  homography: [", ", ".join("[%.8g, %.8g, %.8g]" % tuple(row) for row in H), "]")


if __name__ == "__main__":
    main()
