#!/usr/bin/env python3
"""List the cameras OpenCV can see and the resolutions each one really gives you.

A camera that cannot do the size you ask for silently hands back the nearest one
it can, so this probes by setting a size and reading the frame back.

    python scripts/list_cameras.py
    python scripts/list_cameras.py --max-index 6

Put the index and the largest accepted size into config.yaml under cameras.
"""
from __future__ import annotations

import argparse

import cv2

PROBE = [(1920, 1200), (1920, 1080), (1600, 1200), (1280, 800),
         (1280, 720), (960, 600), (640, 480)]


def sizes_for(cap, fourcc: str | None):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    accepted = []
    for w, h in PROBE:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, f = cap.read()
        if ok and (f.shape[1], f.shape[0]) == (w, h):
            accepted.append((w, h))
    return accepted


def inspect(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None

    default = (frame.shape[1], frame.shape[0])
    # the default pixel format often hides a camera's larger modes; MJPG may expose them
    by_format = {None: sizes_for(cap, None), "MJPG": sizes_for(cap, "MJPG")}
    cap.release()
    return default, by_format


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=4, help="highest camera index to try")
    args = ap.parse_args()

    found = 0
    for index in range(args.max_index + 1):
        result = inspect(index)
        if result is None:
            continue
        found += 1
        default, by_format = result
        print(f"\nindex {index}")
        print(f"  opens at  {default[0]}x{default[1]}")
        for fourcc, accepted in by_format.items():
            sizes = ", ".join(f"{w}x{h}" for w, h in accepted) or "none of the probed sizes"
            print(f"  {fourcc or 'default':<7}   {sizes}")

        best = max((s for sizes in by_format.values() for s in sizes), default=None)
        if best:
            w, h = best
            fmt = next(f for f, sizes in by_format.items() if best in sizes)
            note = "" if fmt is None else f"   # needs CAP_PROP_FOURCC {fmt}"
            print(f"  config:   {{index: {index}, width: {w}, height: {h}, fps: 30}}{note}")

    if not found:
        print("No cameras found. On macOS, check Settings > Privacy & Security > Camera.")


if __name__ == "__main__":
    main()
