#!/usr/bin/env python3
"""Watch the classifier work, with no arm and no sorting — the tool for tuning it.

    python scripts/preview_classify.py            # live box camera
    python scripts/preview_classify.py --index 0  # force a camera index

Keys:  b  re-learn the empty stage (do this if the lighting changed)
       r  print a catalog row for the piece on the stage
       Esc  quit

A piece reading 'unknown' is telling you one of two things: nothing in the catalog
is that shape, or two labels fit equally well. The distance and margin say which.
"""
from __future__ import annotations

import argparse

import cv2

from autosort.classification import BoxCamera, Catalog, measure, outline, segment
from autosort.config import Config

GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)


def row_for(shape) -> str:
    return (f"- {{label: ?, name: ?, length_mm: {shape.length_mm:.2f}, "
            f"width_mm: {shape.width_mm:.2f}, extent: {shape.extent:.2f}, "
            f"circularity: {shape.circularity:.2f}, solidity: {shape.solidity:.2f}, "
            f"holes: {shape.holes}}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="camera index (default: cameras.box)")
    args = ap.parse_args()

    cfg = Config.load()
    cam_cfg = cfg.cameras["box"]
    if args.index is not None:
        cam_cfg.index = args.index
    catalog = Catalog.load(cfg.classifier.catalog)

    cam = BoxCamera(cam_cfg)
    print("Clear the stage — the first frame becomes the reference.")
    cam.open()
    print(f"{len(catalog.entries)} catalog entries, {cfg.classifier.px_per_mm:.2f} px/mm")

    try:
        while True:
            frame = cam.grab()
            mask = segment(frame, cam.background)
            shape = measure(mask, cfg.classifier.px_per_mm)

            if shape is None:
                cv2.putText(frame, "stage empty", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
            else:
                found = outline(mask)
                if found:
                    cv2.drawContours(frame, [found[0]], -1, GREEN, 2)
                m = catalog.match(shape)
                ok = m.distance <= cfg.classifier.max_distance and m.margin >= cfg.classifier.min_confidence
                lines = [
                    str(shape),
                    f"{m.entry.label}  ({m.entry.name})" if ok else "unknown",
                    f"d={m.distance:.2f}  margin={m.margin:.2f}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (20, 32 + 28 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN if ok else RED, 2)

            cv2.imshow("classify  (b background, r row, Esc quit)", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord("b"):
                cam.learn_background()
            if key == ord("r") and shape is not None:
                print(row_for(shape))
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
