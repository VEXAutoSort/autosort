"""Assign camera roles by looking at them (USB camera numbering shuffles).

Run:  python tools/select_cameras.py [--web]
For each camera it opens, you say which role it is:
    T = top (overhead)    W = wrist    B = box (classifier enclosure)
    N = skip this camera  Q = quit
Writes cameras_override.json next to config.yaml; Config.load applies it on
top of config.yaml automatically. Re-run any time the cameras re-shuffle.

Linux: cameras are listed by their /dev/v4l/by-id/ path (built from the
device's burned-in serial), which never moves between boots - the override
stores that path. macOS: falls back to numeric indices 0..5 as before.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.webui import make_display, say  # noqa: E402

OUT = ROOT / "cameras_override.json"
ROLES = {"t": "top", "w": "wrist", "b": "box"}


def candidates() -> list[int | str]:
    by_id = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    return by_id if by_id else list(range(6))


def main() -> None:
    assigned: dict[str, int | str] = {}
    ui = make_display("select cameras")
    for dev in candidates():
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            continue
        say(f"showing {dev}")
        decision = None
        fails = 0
        while decision is None:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    decision = "n"
                    break
                continue
            vis = cv2.resize(frame, (800, 500))
            cv2.putText(vis, f"{Path(str(dev)).name[:60]}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, "T=top  W=wrist  B=box  N=skip  Q=quit", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"assigned so far: {list(assigned)}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ui.status(f"{dev}\nT=top  W=wrist  B=box  N=skip  Q=quit    assigned: {list(assigned)}")
            ui.show(vis)
            key = ui.waitkey(30) & 0xFF
            if key in (ord("t"), ord("w"), ord("b"), ord("n"), ord("q")):
                decision = chr(key)
        cap.release()
        if decision == "q":
            break
        if decision in ROLES:
            assigned[ROLES[decision]] = dev
            say(f"{ROLES[decision]} = {dev}")
    ui.close()

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
