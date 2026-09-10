"""Measure the table height vs the taught z-plane by touching down.

Run:  python tools/touchdown_probe.py [--go] [--points 3]

After a rig move the taught JOINTS still encode the OLD table height. XY can be
re-registered from the camera, but Z cannot - so measure it: at a few taught
points the arm goes hover -> taught grasp height -> down in 2 mm steps until the
servos can no longer track the command (fingertips on the table). The gap
between that contact height and the z-plane's prediction is the shift the
solver must apply (taught.json "z_shift_m"). Torque stays at the arm's normal
limit; the descent stops within one step of contact and never goes more than
MAX_BELOW below the taught height.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.analytic import AnalyticSolver, UnsafePoseError  # noqa: E402
from autosort.config import Config                             # noqa: E402
from autosort.motion import JOINTS, move_smooth, read_joints   # noqa: E402
from autosort.taught import Taught                             # noqa: E402

STEP_M = 0.002
MAX_BELOW_M = 0.035
CONTACT_DEG = 2.5     # tracking error on lift/elbow/wrist above this = touching
FREE_DEG = 1.2        # what the arm tracks to when nothing is in the way (measured ~1 deg)


def main() -> None:
    go = "--go" in sys.argv
    npts = int(sys.argv[sys.argv.index("--points") + 1]) if "--points" in sys.argv else 3
    cfg = Config.load()
    data = json.loads((ROOT / "taught.json").read_text())
    taught = Taught(data)
    solver = AnalyticSolver(taught, urdf_path=cfg.arm.urdf_path)
    # spread: leftmost, rightmost, and the ones nearest the grid centre
    px = solver.pixels
    order = [int(px[:, 0].argmin()), int(px[:, 0].argmax())]
    c = px.mean(0)
    for k in np.argsort(np.linalg.norm(px - c, axis=1)):
        if len(order) >= npts:
            break
        if int(k) not in order:
            order.append(int(k))
    plan = []
    for i in order:
        g = data["grid"][i]
        q0 = np.array([g["joints"][j] for j in JOINTS], float)
        T = solver.kin.forward_kinematics(q0)
        x, y, z = T[:3, 3]
        hover = solver.hover_for({j: float(q0[k]) for k, j in enumerate(JOINTS)}, cfg.arm.hover_lift_m)
        plan.append((i + 1, g["pixel"], q0, hover, (x, y, z)))
        print(f"#{i+1}: pixel ({g['pixel'][0]:.0f},{g['pixel'][1]:.0f}) taught z {z*1000:.1f} mm, "
              f"plane z {solver.grasp_z_at(x, y)*1000:.1f} mm")
    if not go:
        print("(plan only - add --go to move the arm)")
        return

    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    robot = SOFollower(SO101FollowerConfig(port=cfg.arm.port, id=cfg.arm.id))
    robot.connect(calibrate=False)
    home = data["poses"]["home"]
    results = []
    try:
        move_smooth(robot, {**{j: home[j] for j in JOINTS[:5]}, "gripper": data["gripper_open"]}, duration_s=2.0)
        for i, pix, q0, hover, (x, y, z0) in plan:
            print(f"#{i}: hover -> taught height")
            move_smooth(robot, {j: hover[j] for j in JOINTS[:5]}, duration_s=1.6)
            q = q0.copy()
            move_smooth(robot, {j: float(q[k]) for k, j in enumerate(JOINTS[:5])}, duration_s=1.2)
            time.sleep(0.4)
            contact_z, last_free_z = None, z0
            for n in range(int(MAX_BELOW_M / STEP_M) + 1):
                zc = z0 - n * STEP_M
                qn = solver._solve_dls(np.array([x, y, zc]), q)
                qn[JOINTS.index("wrist_roll")] = q0[JOINTS.index("wrist_roll")]
                try:
                    solver._sanity_check(qn, f"probe z={zc*1000:.0f}mm")
                except UnsafePoseError as e:
                    print(f"   stopped by guard: {e}")
                    break
                cmd = {j: float(qn[k]) for k, j in enumerate(JOINTS[:5])}
                move_smooth(robot, cmd, duration_s=0.35)
                time.sleep(0.3)
                now = read_joints(robot)
                err = {j: abs(now[j] - cmd[j]) for j in ("shoulder_lift", "elbow_flex", "wrist_flex")}
                worst = max(err.values())
                print(f"   z {zc*1000:6.1f} mm  track err {worst:4.1f} deg")
                if worst > CONTACT_DEG:
                    contact_z = zc
                    break
                last_free_z = zc
                q = qn
            plane = solver.grasp_z_at(x, y)
            if contact_z is None:
                print(f"   no contact within {MAX_BELOW_M*1000:.0f} mm below the taught height")
            else:
                est = (contact_z + last_free_z) / 2
                results.append((i, est - plane))
                print(f"   CONTACT between {last_free_z*1000:.1f} and {contact_z*1000:.1f} mm  "
                      f"-> table at ~{est*1000:.1f} mm, plane predicts {plane*1000:.1f} mm, "
                      f"shift {1000*(est-plane):+.1f} mm")
            move_smooth(robot, {j: hover[j] for j in JOINTS[:5]}, duration_s=1.2)
        move_smooth(robot, {j: home[j] for j in JOINTS[:5]}, duration_s=2.0)
        move_smooth(robot, {"gripper": data["gripper_closed"] + 3.0}, duration_s=0.4)
    finally:
        robot.disconnect()
    if results:
        s = np.array([r[1] for r in results])
        print(f"\nz shift vs plane: {[f'{v*1000:+.1f}' for v in s]} mm -> mean {s.mean()*1000:+.1f} mm "
              f"(spread {1000*(s.max()-s.min()):.1f} mm)")
        print("Note: the taught grasp height sat a few mm ABOVE the old table (gear between the tips),")
        print("so z_shift_m should be (this mean) + the same few mm; start with mean + 3 mm.")


if __name__ == "__main__":
    main()
