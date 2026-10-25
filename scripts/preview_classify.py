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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import autosort/ without installing

import argparse
import statistics

import cv2

from autosort.classification import BoxCamera, Catalog, measure, outline, segment
from autosort.config import Config

GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)


def row_for(shape, label: str = "?", name: str = "?") -> str:
    return (f"- {{label: {label}, name: {name}, length_mm: {shape.length_mm:.2f}, "
            f"width_mm: {shape.width_mm:.2f}, extent: {shape.extent:.2f}, "
            f"circularity: {shape.circularity:.2f}, solidity: {shape.solidity:.2f}, "
            f"holes: {shape.holes}}}")


def append_row(path, label: str, shape, seen: dict) -> str:
    seen[label] = seen.get(label, 0) + 1
    row = row_for(shape, label, f"{label} {seen[label]}")
    with open(path, "a") as f:
        f.write(row + "\n")
    return row


def summarise(truth: str, trials: list) -> None:
    """Report what fraction of the drops came back right, and how steady they were."""
    if not trials:
        print("\nno trials recorded")
        return

    got = [t[0] for t in trials]
    right = sum(1 for g in got if g == truth)
    print(f"\n--- {truth}: {right}/{len(trials)} correct ({100 * right / len(trials):.0f}%) ---")

    for field in ("length_mm", "width_mm", "circularity", "solidity"):
        vals = [getattr(t[1], field) for t in trials]
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {field:<13} {statistics.mean(vals):7.2f}  +/- {spread:.2f}")

    ds = [t[2].distance for t in trials]
    ms = [t[2].margin for t in trials]
    print(f"  distance      {statistics.mean(ds):7.2f}  worst {max(ds):.2f}")
    print(f"  margin        {statistics.mean(ms):7.2f}  worst {min(ms):.2f}")

    wrong = sorted({g for g in got if g != truth})
    if wrong:
        print(f"  MISREAD AS: {', '.join(wrong)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="camera index (default: cameras.box)")
    ap.add_argument("--add", metavar="LABEL",
                    help="pressing 'r' appends a row with this label straight to the catalog")
    ap.add_argument("--test", metavar="LABEL",
                    help="what the part really is; space records a trial, Esc prints the report")
    args = ap.parse_args()

    cfg = Config.load()
    cam_cfg = cfg.cameras["box"]
    if args.index is not None:
        cam_cfg.index = args.index
    catalog = Catalog.load(cfg.classifier.catalog)

    cam = BoxCamera(cam_cfg, cfg.classifier.roi, cfg.classifier.average_frames)
    print("Clear the stage — the first frame becomes the reference.")
    cam.open()
    print(f"{len(catalog.entries)} catalog entries, {cfg.classifier.px_per_mm:.2f} px/mm")
    seen: dict[str, int] = {}
    trials: list = []
    if args.test:
        print(f"testing '{args.test}' — drop it, let it settle, press SPACE. Esc for the report.")
    if args.add:
        if args.add not in cfg.classifier.labels:
            raise SystemExit(f"'{args.add}' is not in classifier.labels — add it to config.yaml first")
        print(f"press 'r' to add a '{args.add}' row to {cfg.classifier.catalog}")

    try:
        while True:
            frame = cam.grab()
            mask = segment(frame, cam.background, cfg.classifier.diff_threshold)
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
                verdict = m.entry.label if ok else "unknown"
                lines = [
                    str(shape),
                    f"{m.entry.label}  ({m.entry.name})" if ok else "unknown",
                    f"d={m.distance:.2f}  margin={m.margin:.2f}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (20, 32 + 28 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN if ok else RED, 2)

            caption = "classify  (b background, r row, space trial, Esc quit)"
            cv2.imshow(caption, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord("b"):
                cam.learn_background()
            if key == ord(" ") and args.test and shape is not None:
                trials.append((verdict, shape, m))
                print(f"  trial {len(trials):2}: {verdict:<9} d={m.distance:.2f} margin={m.margin:.2f}")
            if key == ord("r") and shape is not None:
                if args.add:
                    print("added:", append_row(cfg.classifier.catalog, args.add, shape, seen))
                    catalog = Catalog.load(cfg.classifier.catalog)
                else:
                    print(row_for(shape))
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if args.test:
            summarise(args.test, trials)


if __name__ == "__main__":
    main()
