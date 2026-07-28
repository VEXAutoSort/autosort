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
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print("TORQUE RELEASED — move the arm by hand. Keys: G H M I B O C S Q")

    top = cfg.cameras["top"]
    cap = cv2.VideoCapture(top.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, top.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, top.height)

    data = {"grid": [], "hover_delta": None, "poses": {},
            "gripper_open": None, "gripper_closed": None}
    hover_ref = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target = perception.largest_piece_px(rgb)
        vis = frame.copy()
        if target:
            cv2.circle(vis, (int(target[0]), int(target[1])), 12, (0, 255, 0), 3)
        for i, pt in enumerate(data["grid"]):
            cv2.putText(vis, str(i + 1), (int(pt["pixel"][0]), int(pt["pixel"][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        done = [k for k in ("home", "inspect", "box_drop") if k in data["poses"]]
        status = (f"grid:{len(data['grid'])}/9  hover:{'Y' if data['hover_delta'] else '-'}  "
                  f"poses:{','.join(done) or '-'}  "
                  f"open:{'Y' if data['gripper_open'] is not None else '-'}  "
                  f"closed:{'Y' if data['gripper_closed'] is not None else '-'}")
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("teach (G H M I B O C S Q)", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("g"):
            if not target:
                print("no piece detected — check placement/lighting")
                continue
            joints = read_joints(robot)
            data["grid"].append({"pixel": [target[0], target[1]], "joints": joints})
            hover_ref = joints
            print(f"grid {len(data['grid'])}/9 at pixel ({target[0]:.0f},{target[1]:.0f})")
        elif key == ord("h"):
            if hover_ref is None:
                print("teach a grid point first, then lift from it")
                continue
            now = read_joints(robot)
            data["hover_delta"] = {j: now[j] - hover_ref[j] for j in now}
            print("hover delta recorded")
        elif key == ord("m"):
            data["poses"]["home"] = read_joints(robot)
            print("home recorded (make sure the arm is OUT of the pile view!)")
        elif key == ord("i"):
            data["poses"]["inspect"] = read_joints(robot)
            print("inspect recorded")
        elif key == ord("b"):
            data["poses"]["box_drop"] = read_joints(robot)
            print("box_drop recorded")
        elif key == ord("o"):
            data["gripper_open"] = read_joints(robot)["gripper"]
            print(f"gripper open = {data['gripper_open']:.1f}")
        elif key == ord("c"):
            data["gripper_closed"] = read_joints(robot)["gripper"]
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
