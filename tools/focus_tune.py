"""Live sharpness meter for physically focusing a camera lens.

Run:  python tools/focus_tune.py [top|wrist]        (default: top)

The Arducam (and most M12-lens cameras) have NO focus motor — focus is set by
rotating the lens barrel by hand. This tool replaces "looks sharp to me" with a
number: it shows the live feed and a big sharpness score (variance of the
Laplacian over the measured region). Rotate the lens slowly; the score rises,
peaks, and falls again — leave it at the peak.

  - put a piece (or the ArUco sheet) in the pick zone first: sharpness is only
    meaningful on real detail at the REAL working distance, not an empty table
  - for 'top' the score is measured over pile_roi (what detection actually sees)
  - PEAK score is remembered and shown for reference; R resets it (press after
    moving the target), Q quits

The score is relative — only compare readings of the same scene. Anything
within ~5% of the peak is in the flat top of the focus curve and fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.config import Config  # noqa: E402


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "top"
    cfg = Config.load()
    if role not in cfg.cameras:
        sys.exit(f"unknown camera role '{role}' (have: {', '.join(cfg.cameras)})")
    try:
        cap = cfg.cameras[role].verify_open(role)
    except RuntimeError as e:
        sys.exit(str(e))

    roi = cfg.perception.pile_roi if role == "top" else cfg.perception.gripper_roi
    peak = 0.0
    ema = None
    print("Rotate the lens until the score peaks. R resets the peak, Q quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped returning frames")
            break
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h))
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        # per-frame scores jitter with sensor noise; a light EMA makes the
        # rise/peak/fall readable while you turn the lens
        ema = score if ema is None else 0.7 * ema + 0.3 * score
        peak = max(peak, ema)

        vis = frame.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 165, 255), 2)
        pct = 100.0 * ema / peak if peak > 0 else 0.0
        color = (0, 255, 0) if pct >= 95.0 else (0, 200, 255)
        cv2.putText(vis, f"sharpness {ema:7.1f}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(vis, f"peak {peak:7.1f}  ({pct:4.1f}% of peak)", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(f"focus_tune [{role}]", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            peak = 0.0
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
