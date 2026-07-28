"""Safe arm motion: interpolate to targets instead of jumping.

send_action() alone commands the full move in one step, which makes the arm
lurch at maximum speed. Every scripted motion in AutoSort goes through
move_smooth() so speed is bounded and predictable near people and hardware.
"""
from __future__ import annotations

import time

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def read_joints(robot) -> dict[str, float]:
    """Current joint positions as a {name: degrees} dict (no camera reads).

    num_retry absorbs the occasional corrupted packet on the servo bus.
    """
    obs = robot.bus.sync_read("Present_Position", num_retry=4)
    return {j: float(obs[j]) for j in JOINTS}


def move_smooth(robot, target: dict[str, float], duration_s: float = 1.2, hz: int = 50) -> None:
    """Linearly interpolate all joints from the current pose to `target`.

    `target` maps joint name -> degrees; joints missing from it hold position.
    """
    cur = read_joints(robot)
    full_target = {j: float(target.get(j, cur[j])) for j in JOINTS}
    steps = max(2, int(duration_s * hz))
    for i in range(1, steps + 1):
        a = i / steps
        cmd = {f"{j}.pos": cur[j] + a * (full_target[j] - cur[j]) for j in JOINTS}
        robot.send_action(cmd)
        time.sleep(1.0 / hz)
