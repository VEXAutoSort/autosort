"""Teach every pose the system needs BY HAND (~15 minutes, one time).

Run:  python tools/teach.py
The arm's torque is released — physically move it. A camera window shows the
top view with live piece detection; click the window so it gets your keys.

Teach, in order:
  G  x9  grid: put a piece somewhere new in the pile zone, wait for the green
          circle, move the gripper onto it (open fingers around the piece), press G.
          Spread the 9 spots: corners, edge midpoints, center.
  H       hover: from the arm's position after your last G, lift straight up
          ~10 cm, press H.
  M       home: arm fully OUT of the camera's view of the pile zone, press M.
          (All detection happens at home — the arm must not block the zone.)
  I       inspect: hold the gripper up where the wrist camera sees its contents
          clearly, press I.
  B       box_drop: over the box/placeholder where pieces get released, press B.
  O / C   gripper open (O) then squeezed closed on a piece (C).
  S       save taught.json     Q  quit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.config import Config          # noqa: E402
from autosort.motion import read_joints     # noqa: E402
from autosort.perception import Perception  # noqa: E402

OUT = ROOT / "taught.json"


def main() -> None:
    override = ROOT / "cameras_override.json"
    if not override.exists():
        print("Camera roles have never been assigned on this machine.")
        print("Run this first (1 minute):")
        print("  ~/.local/share/uv/tools/lelab/bin/python "
              f"{ROOT}/tools/select_cameras.py")
        print("macOS shuffles USB camera numbering - teaching against the wrong")
        print("camera would corrupt every taught position.")
        sys.exit(1)
    cfg = Config.load()
    cfg.run.dry_run = False
    perception = Perception(cfg.perception, dry_run=False)

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower

    robot = SO101Follower(SO101FollowerConfig(port=cfg.arm.port, id=cfg.arm.id))
    # the servo bus occasionally corrupts a packet; retry instead of dying
    import time as _t
    for attempt in range(4):
        try:
            robot.connect(calibrate=False)
            break
        except ConnectionError as e:
            if attempt == 3:
                raise
            print(f"bus glitch on connect ({e}); retrying in 2s [{attempt + 1}/3]")
            try:
                robot.bus.disconnect()
            except Exception:
                pass
            _t.sleep(2)
    robot.bus.disable_torque()
    print("TORQUE RELEASED — move the arm by hand. Keys: G H M I B O C S Q")

    top = cfg.cameras["top"]
    cap = cv2.VideoCapture(top.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, top.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, top.height)

    def safe_joints():
        """read_joints, but a bus glitch returns None instead of crashing the session."""
        try:
            return read_joints(robot)
        except ConnectionError as e:
            print(f"bus glitch while reading the arm ({e}) — nothing recorded, press the key again")
            return None

    if OUT.exists():
        data = json.loads(OUT.read_text())
        print(f"resuming from existing taught.json: {len(data['grid'])} grid points already "
              "recorded. New G presses ADD points; H/M/I/B/O/C overwrite just that item.")
    else:
        data = {"grid": [], "hover_delta": None, "poses": {},
                "gripper_open": None, "gripper_closed": None}
    hover_ref = None
    last_key = "none yet"
    pending_pixel = None   # grid teaching is two-phase: lock pixel first, then record joints

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target = perception.largest_piece_px(rgb)
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        rx0, ry0, rx1, ry1 = cfg.perception.pile_roi
        cv2.rectangle(vis, (int(rx0 * fw), int(ry0 * fh)), (int(rx1 * fw), int(ry1 * fh)),
                      (255, 150, 0), 2)
        cv2.putText(vis, "detection zone - claw must stay OUTSIDE this box at home",
                    (int(rx0 * fw) + 5, int(ry0 * fh) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)
        if target and pending_pixel is None:
            cv2.circle(vis, (int(target[0]), int(target[1])), 12, (0, 255, 0), 3)
        if pending_pixel:
            cv2.circle(vis, (int(pending_pixel[0]), int(pending_pixel[1])), 16, (255, 0, 255), 3)
            cv2.putText(vis, "TARGET LOCKED - move gripper onto it, press G again (X cancels)",
                        (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        for i, pt in enumerate(data["grid"]):
            cv2.putText(vis, str(i + 1), (int(pt["pixel"][0]), int(pt["pixel"][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        done = [k for k in ("home", "inspect", "box_drop") if k in data["poses"]]
        status = (f"grid:{len(data['grid'])}  hover:{'Y' if data['hover_delta'] else '-'}  "
                  f"poses:{','.join(done) or '-'}  "
                  f"open:{'Y' if data['gripper_open'] is not None else '-'}  "
                  f"closed:{'Y' if data['gripper_closed'] is not None else '-'}")
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, f"last key seen: {last_key}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("teach (G H M I B O C S Q)", vis)
        raw = cv2.waitKey(30)
        key = raw & 0xFF
        if raw != -1 and 32 <= key < 127:
            key = ord(chr(key).lower())
            last_key = chr(key)
            print(f"[key] {last_key!r} (code {raw})")
        elif raw != -1:
            last_key = f"code {raw}"
            print(f"[key] non-printable code {raw}")

        if key == ord("g"):
            if pending_pixel is None:
                # phase 1: arm must be CLEAR of the zone so the piece is visible
                if not target:
                    print("no piece detected — keep the arm out of view and check placement")
                    continue
                pending_pixel = (target[0], target[1])
                print(f"target locked at pixel ({target[0]:.0f},{target[1]:.0f}) — "
                      "now move the gripper onto the piece and press G again")
            else:
                # phase 2: arm is on the piece; pair the locked pixel with these joints
                joints = safe_joints()
                if joints is None:
                    continue
                data["grid"].append({"pixel": list(pending_pixel), "joints": joints})
                hover_ref = joints
                print(f"grid point recorded ({len(data['grid'])} total)")
                pending_pixel = None
        elif key == ord("x"):
            if pending_pixel is not None:
                pending_pixel = None
                print("locked target cancelled")
        elif key == ord("h"):
            if hover_ref is None:
                print("teach a grid point first, then lift from it")
                continue
            now = safe_joints()
            if now is None:
                continue
            data["hover_delta"] = {j: now[j] - hover_ref[j] for j in now}
            print("hover delta recorded")
        elif key == ord("m"):
            _j = safe_joints()
            if _j is None:
                continue
            data["poses"]["home"] = _j
            print("home recorded (make sure the arm is OUT of the pile view!)")
        elif key == ord("i"):
            _j = safe_joints()
            if _j is None:
                continue
            data["poses"]["inspect"] = _j
            print("inspect recorded")
        elif key == ord("b"):
            _j = safe_joints()
            if _j is None:
                continue
            data["poses"]["box_drop"] = _j
            print("box_drop recorded")
        elif key == ord("o"):
            _j = safe_joints()
            if _j is None:
                continue
            data["gripper_open"] = _j["gripper"]
            print(f"gripper open = {data['gripper_open']:.1f}")
        elif key == ord("c"):
            _j = safe_joints()
            if _j is None:
                continue
            data["gripper_closed"] = _j["gripper"]
            print(f"gripper closed = {data['gripper_closed']:.1f}")
        elif key == ord("s"):
            missing = [k for k, v in data.items() if v in (None, [], {})]
            if len(data["grid"]) < 4:
                missing.append(f"grid has {len(data['grid'])}/4 minimum")
            if any(p not in data["poses"] for p in ("home", "inspect", "box_drop")):
                missing.append("poses incomplete")
            if missing:
                print(f"not saved — missing: {missing}")
                continue
            OUT.write_text(json.dumps(data, indent=2))
            print(f"saved {OUT}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()


if __name__ == "__main__":
    main()
