"""Capture the ArUco reference for camera-drift auto-correction.

Run:  python tools/capture_markers.py

Run this ONLY when picking is known accurate (right after teaching, or after
verifying a good grab): it freezes "where the markers sit when calibration is
right" into markers_ref.json. From then on the pipeline detects the markers
every cycle and cancels any camera drift relative to this reference.

Shows the top camera with detected markers outlined. SPACE saves once all
expected markers have been steadily visible; Q quits without saving.
Re-run any time picking has been re-verified (e.g. after a re-teach).
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
from autosort.recal import detect_markers   # noqa: E402

EXPECTED_IDS = {0, 1, 2, 3}   # the make_markers.py sheet
OUT = ROOT / "markers_ref.json"


def main() -> None:
    cfg = Config.load()
    try:
        cap = cfg.cameras["top"].verify_open("top")
    except RuntimeError as e:
        sys.exit(str(e))

    print("SPACE saves the reference (all 4 markers must be visible), Q quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("top camera stopped returning frames")
        seen = detect_markers(frame)
        vis = frame.copy()
        for mid, c in seen.items():
            pts = c.astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(vis, str(mid), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        missing = EXPECTED_IDS - set(seen)
        msg = ("all markers visible - SPACE to save" if not missing
               else f"missing ids {sorted(missing)} - check placement/glare/occlusion")
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if not missing else (0, 0, 255), 2)
        cv2.imshow("capture_markers", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("not saved")
            break
        if key == ord(" "):
            if missing:
                print(f"refusing to save: missing marker ids {sorted(missing)}")
                continue
            OUT.write_text(json.dumps({
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "markers": {str(k): v.tolist() for k, v in seen.items()},
            }, indent=2))
            print(f"saved {len(seen)} markers -> {OUT}")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
