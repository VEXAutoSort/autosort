#!/usr/bin/env python3
"""Measure the box camera's px/mm, so the classifier can work in millimetres.

Put a part whose long dimension you know on the stage — a 2" standoff is 50.8 mm.
Every number the classifier uses depends on this, so redo it if the camera moves.

Usage:  python scripts/calibrate_scale.py                 # assumes a 2" standoff
        python scripts/calibrate_scale.py --length 25.4   # or any known length (mm)
        python scripts/calibrate_scale.py --index 0       # force a camera index
"""
from __future__ import annotations

import argparse

from autosort.classification import BoxCamera, measure, segment
from autosort.config import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=50.8,
                    help="long dimension of the reference part in mm (default: 2\" standoff)")
    ap.add_argument("--index", type=int, default=None, help="camera index (default: cameras.box)")
    args = ap.parse_args()

    cam_cfg = Config.load().cameras["box"]
    if args.index is not None:
        cam_cfg.index = args.index

    cam = BoxCamera(cam_cfg)
    try:
        input("Clear the stage, then press Enter... ")
        cam.open()
        input(f"Place the {args.length:.1f} mm part on the stage, then press Enter... ")

        shape = measure(segment(cam.grab(), cam.background), px_per_mm=1.0)
        if shape is None:
            raise SystemExit("nothing found on the stage — check lighting and camera index")

        px_per_mm = shape.length_mm / args.length   # measured in px, since px_per_mm was 1
        print(f"\nmeasured {shape.length_mm:.1f} px long, {shape.width_mm:.1f} px wide")
        print(f"\nPaste into config.yaml under classifier:\n\n  px_per_mm: {px_per_mm:.2f}\n")
        print(f"that is {1000 / px_per_mm:.0f} um per pixel")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
