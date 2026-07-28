"""Assign camera roles by looking at them (macOS shuffles USB camera indices).

Run:  python tools/select_cameras.py
For each camera it opens, you say which role it is:
    T = top (overhead)    W = wrist    B = box (classifier enclosure)
    N = skip this camera  Q = quit
Writes cameras_override.json next to config.yaml; Config.load applies it on
top of config.yaml automatically. Re-run any time the cameras re-shuffle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cameras_override.json"
ROLES = {"t": "top", "w": "wrist", "b": "box"}


def main() -> None:
    assigned: dict[str, int] = {}
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        decision = None
        while decision is None:
            ok, frame = cap.read()
            if not ok:
                decision = "n"
                break
            vis = cv2.resize(frame, (800, 500))
            cv2.putText(vis, f"index {idx}   T=top  W=wrist  B=box  N=skip  Q=quit",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(vis, f"assigned so far: {assigned}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("select cameras", vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("t"), ord("w"), ord("b"), ord("n"), ord("q")):
                decision = chr(key)
        cap.release()
        if decision == "q":
            break
        if decision in ROLES:
            assigned[ROLES[decision]] = idx
    cv2.destroyAllWindows()

    if not assigned:
        print("nothing assigned — is another app (LeLab?) holding the cameras?")
        sys.exit(1)
    OUT.write_text(json.dumps(assigned, indent=2))
    print(f"saved {OUT}: {assigned}")
    missing = [r for r in ("top", "wrist") if r not in assigned]
    if missing:
        print(f"WARNING: still unassigned: {missing} — the pipeline needs them")


if __name__ == "__main__":
    main()
