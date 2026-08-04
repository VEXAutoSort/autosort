"""Timed wrist-counter diagnostic. Runs a protocol, writes results to a file.

Run:  python tools/wrist_check.py
On-screen countdown tells you what to do:
  Phase 1 (12s): EMPTY gripper - hands away
  Phase 2 (18s): hold a GEAR between the fingertips
Writes /tmp/wrist_check.json with per-phase counts and blob areas.
Q quits early.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.config import Config          # noqa: E402
from autosort.perception import Perception  # noqa: E402

OUT = Path("/tmp/wrist_check.json")
EMPTY_S, HELD_S = 12, 18


def main() -> None:
    cfg = Config.load()
    perception = Perception(cfg.perception, dry_run=False)
    w = cfg.cameras["wrist"]
    cap = w.verify_open("wrist")
    if False:
        print("wrist camera failed to open - is LeLab or another tool holding it?")
        sys.exit(1)

    roi = cfg.perception.gripper_roi
    samples = {"empty": [], "held": []}
    snaps_taken = set()
    fails = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 60:
                print("camera stopped returning frames - aborting", flush=True)
                break
            time.sleep(0.05)
            continue
        fails = 0
        el = time.time() - t0
        phase = "empty" if el < EMPTY_S else ("held" if el < EMPTY_S + HELD_S else None)
        if phase is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blobs = perception._blobs(rgb, roi)
        if el > 2:  # skip the first seconds while the user reads the prompt
            samples[phase].append({"n": len(blobs), "areas": [round(b[0]) for b in blobs]})
        sec = int(el)
        if sec not in snaps_taken:
            snaps_taken.add(sec)
            print(f"  t={sec:>3}s  phase={phase:<5}  count={len(blobs)}", flush=True)
            # raw snapshots (no overlay) for offline ROI placement
            if sec in (5, 9, EMPTY_S + 6, EMPTY_S + 12):
                cv2.imwrite(f"/tmp/wrist_snap_{phase}_{sec}.png", frame)

        vis = frame.copy()
        h, wd = vis.shape[:2]
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (int(x0 * wd), int(y0 * h)), (int(x1 * wd), int(y1 * h)), (255, 150, 0), 2)
        for area, cx, cy in blobs:
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 255, 0), 3)
            cv2.putText(vis, str(int(area)), (int(cx) + 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        left = (EMPTY_S - el) if phase == "empty" else (EMPTY_S + HELD_S - el)
        msg = ("PHASE 1: EMPTY gripper - hands away" if phase == "empty"
               else "PHASE 2: hold a GEAR in the fingertips")
        cv2.putText(vis, msg, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"{left:4.0f}s left     count: {len(blobs)}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("wrist check", vis)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    def summarize(rows):
        if not rows:
            return {"frames": 0}
        counts = [r["n"] for r in rows]
        areas = [a for r in rows for a in r["areas"]]
        hist = {}
        for c in counts:
            hist[c] = hist.get(c, 0) + 1
        return {"frames": len(counts), "count_histogram": hist,
                "median_count": sorted(counts)[len(counts) // 2],
                "area_min": min(areas) if areas else None,
                "area_median": sorted(areas)[len(areas) // 2] if areas else None,
                "area_max": max(areas) if areas else None}

    result = {"roi": roi, "contrast_margin": cfg.perception.contrast_margin,
              "min_piece_area": cfg.perception.min_piece_area,
              "empty": summarize(samples["empty"]), "held": summarize(samples["held"])}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
