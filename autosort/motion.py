"""Safe arm motion: interpolate to targets instead of jumping.

send_action() alone commands the full move in one step, which makes the arm
lurch at maximum speed. Every scripted motion in AutoSort goes through
move_smooth() so speed is bounded and predictable near people and hardware.
"""
from __future__ import annotations

import logging
import time

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

log = logging.getLogger("autosort.motion")

# Errors seen when the USB serial device disappears mid-run (loose cable, power
# blip): the OS invalidates the fd, so every subsequent call raises.
_LINK_ERRORS = (ConnectionError, OSError)


def _recover(robot, exc):
    """Try to bring a dropped serial link back.

    Returns the NEW robot object on success, None on failure. The old object's
    serial fd is invalid after a drop, so callers MUST retry on the returned
    object — retrying on the original would fail again immediately.
    """
    hook = getattr(robot, "_autosort_reconnect", None)
    if hook is None:
        return None
    log.warning("serial link dropped (%s) - reconnecting", type(exc).__name__)
    for attempt in range(3):
        time.sleep(1.5)
        try:
            new_robot = hook()
            if new_robot is not None:
                log.warning("serial link restored")
                return new_robot
        except Exception:
            pass
    log.error("could not restore the serial link after 3 attempts")
    return None


def read_joints(robot) -> dict[str, float]:
    """Current joint positions as a {name: degrees} dict (no camera reads).

    num_retry absorbs corrupted packets; a full link drop triggers a reconnect.
    """
    try:
        obs = robot.bus.sync_read("Present_Position", num_retry=4)
    except _LINK_ERRORS as e:
        robot = _recover(robot, e)
        if robot is None:
            raise
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
        try:
            robot.send_action(cmd)
        except _LINK_ERRORS as e:
            new_robot = _recover(robot, e)
            if new_robot is None:
                raise
            robot = new_robot  # rest of the move continues on the live link
            robot.send_action(cmd)
        time.sleep(1.0 / hz)
