#!/usr/bin/env python3
"""Test the geometric grasp algorithm with NO arm — draws the grasp it computes.

    python scripts/preview_grasp.py                 # live top camera (cameras.top from config)
    python scripts/preview_grasp.py --image top.jpg # a saved still
    python scripts/preview_grasp.py --index 0       # force a camera index

Green line = piece's long axis. Red line = where the gripper jaws close.
Red dot = grasp centroid. Press any key (image) or Esc (live) to quit.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from autosort.config import Config


def annotate(frame, min_area: int):
    """Segment the largest piece, compute the grasp, and draw it. Returns the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        cv2.putText(frame, "no piece found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    mean, eig = cv2.PCACompute(pts, mean=None)
    cx, cy = mean[0]
    long_ax, short_ax = eig[0], eig[1]
    proj_l = (pts - mean) @ long_ax
    proj_s = (pts - mean) @ short_ax
    hl, hs = (proj_l.max() - proj_l.min()) / 2, (proj_s.max() - proj_s.min()) / 2

    cv2.drawContours(frame, [c], -1, (0, 255, 255), 1)
    p = np.array([cx, cy])
    _line(frame, p - long_ax * hl, p + long_ax * hl, (0, 255, 0))    # long axis
    _line(frame, p - short_ax * hs, p + short_ax * hs, (0, 0, 255))  # jaw-close axis
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"grasp width ~{2 * hs:.0f}px", (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def _line(img, a, b, color):
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="annotate a saved still instead of the camera")
    ap.add_argument("--index", type=int, help="camera index (default: cameras.top)")
    args = ap.parse_args()
    min_area = Config.load().grasp.min_piece_area

    if args.image:
        img = cv2.imread(args.image)
        cv2.imshow("grasp", annotate(img, min_area))
        cv2.waitKey(0)
        return

    index = args.index if args.index is not None else Config.load().cameras["top"].index
    cap = cv2.VideoCapture(index)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("grasp (Esc to quit)", annotate(frame, min_area))
        if cv2.waitKey(20) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
