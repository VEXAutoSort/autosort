"""SO-ARM101 follower: executes either an ACT policy or a geometric grasp pose.

- pick_act()      : run the trained ACT policy for one grasp (used by ACTGrasper).
- execute_grasp() : drive to a table-frame GraspPose via top-down IK (used by
                    GeometricGrasper), then descend, close, and lift.

Heavy imports (lerobot, torch) are lazy so dry-run works without them.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from .config import ArmCfg, CameraCfg

if TYPE_CHECKING:
    from .grasp import GraspPose

log = logging.getLogger("autosort.arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IK_JOINTS = JOINTS[:5]  # everything except the gripper


class Arm:
    def __init__(self, cfg: ArmCfg, cameras: dict[str, CameraCfg], dry_run: bool = False):
        self.cfg = cfg
        self.cameras = cameras
        self.dry_run = dry_run
        self.robot = None
        self.policy = None

    # --- lifecycle ----------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            log.info("[dry-run] arm connected")
            return
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # older layout
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.cameras.opencv import OpenCVCameraConfig

        cams = {
            name: OpenCVCameraConfig(index_or_path=c.index, width=c.width, height=c.height, fps=c.fps)
            for name, c in self.cameras.items()
            if name in ("top", "wrist")
        }
        self.robot = SO101Follower(
            SO101FollowerConfig(port=self.cfg.port, id=self.cfg.id, cameras=cams)
        )
        self.robot.connect()
        log.info("arm connected on %s", self.cfg.port)

    def load_policy(self) -> None:
        """Load the ACT policy (only needed for grasp.mode == act)."""
        if self.dry_run or self.policy is not None:
            return
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.policy = ACTPolicy.from_pretrained(self.cfg.policy)
        self.policy.eval()
        log.info("ACT policy loaded: %s", self.cfg.policy)

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()

    # --- observation --------------------------------------------------
    def observe(self) -> dict[str, Any]:
        return {} if self.dry_run else self.robot.get_observation()

    def frame(self, name: str):
        return self.observe().get(name)

    def gripper_pos(self) -> float:
        return 50.0 if self.dry_run else float(self.observe().get("gripper.pos", 0.0))

    def gripper_holding(self) -> bool:
        return self.gripper_pos() > self.cfg.gripper_empty_pos

    # --- ACT grasp ----------------------------------------------------
    def pick_act(self) -> None:
        """Run the ACT policy for one grasp, then hold at the inspect pose."""
        if self.dry_run:
            log.info("[dry-run] pick_act()")
            return
        import torch

        self.load_policy()
        self.policy.reset()
        t0 = time.time()
        while time.time() - t0 < self.cfg.pick_timeout_s:
            obs = self.robot.get_observation()
            with torch.no_grad():
                # NOTE: some LeRobot versions need make_pre_post_processors() here.
                action = self.policy.select_action(obs)
            self.robot.send_action(action)
        self.move_to("inspect", hold_gripper=True)

    # --- geometric grasp ----------------------------------------------
    def execute_grasp(self, pose: "GraspPose") -> None:
        """Descend on a table-frame grasp pose, close, and lift to inspect."""
        if self.dry_run:
            log.info("[dry-run] execute_grasp(%.0f, %.0f)", pose.x, pose.y)
            return
        ik = self.cfg.ik
        above = self._ik(pose.x, pose.y, ik["surface_z"] + ik["approach_clearance"], pose.theta)
        at = self._ik(pose.x, pose.y, ik["surface_z"], pose.theta)
        self._set_gripper(open_=True)
        self._send(above); time.sleep(0.8)   # over the piece
        self._send(at);    time.sleep(0.6)    # down to it
        self._set_gripper(open_=False); time.sleep(0.4)  # grasp
        self._send(above); time.sleep(0.6)    # lift straight up
        self.move_to("inspect", hold_gripper=True)

    def _ik(self, x: float, y: float, z: float, theta: float) -> dict[str, float]:
        """Top-down IK: table point (mm) + gripper yaw (rad) -> joint commands (deg).

        Standard base-yaw + planar 2-link solution. The link lengths and the
        per-joint calibration (`cmd = sign*angle + offset`) live in cfg.ik and
        MUST be measured/verified on your arm — the geometry is correct but the
        joint zero/sign conventions depend on your calibration.
        """
        ik = self.cfg.ik
        l1, l2, tool, base_h = ik["link_1"], ik["link_2"], ik["tool_len"], ik["base_height"]
        pan = math.atan2(y, x)
        dx = math.hypot(x, y)
        dy = (z + tool) - base_h  # wrist pivot sits tool_len above the tip (gripper points down)
        d = min(math.hypot(dx, dy), l1 + l2 - 1e-3)
        elbow = math.acos(max(-1.0, min(1.0, (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2))))
        shoulder = math.atan2(dy, dx) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
        wrist_flex = -math.pi / 2 - (shoulder + elbow)  # keep the tool pointing straight down
        wrist_roll = theta - pan
        geom = dict(zip(IK_JOINTS, (pan, shoulder, elbow, wrist_flex, wrist_roll)))
        calib = ik.get("calib", {})
        return {f"{j}.pos": calib.get(j, [0.0, 1.0])[1] * math.degrees(a) + calib.get(j, [0.0, 1.0])[0]
                for j, a in geom.items()}

    # --- shared motions -----------------------------------------------
    def place_in_box(self) -> None:
        if self.dry_run:
            log.info("[dry-run] place_in_box()")
            return
        self.move_to("box_drop", hold_gripper=True)  # carry the piece over the box
        self._set_gripper(open_=True)
        time.sleep(0.4)
        self.move_to("home")

    def drop_back(self) -> None:
        if self.dry_run:
            log.info("[dry-run] drop_back()")
            return
        self.move_to("home", hold_gripper=True)
        self._set_gripper(open_=True)
        time.sleep(0.3)

    def home(self) -> None:
        if self.dry_run:
            log.info("[dry-run] home()")
            return
        self.move_to("home")

    # --- low level ----------------------------------------------------
    def move_to(self, pose_name: str, hold_gripper: bool = False) -> None:
        target = self.cfg.poses[pose_name]
        joints = {f"{j}.pos": float(target[j]) for j in JOINTS
                  if j in target and not (hold_gripper and j == "gripper")}
        self._send(joints)
        time.sleep(0.6)

    def _send(self, joints: dict[str, float]) -> None:
        self.robot.send_action(joints)

    def _set_gripper(self, open_: bool) -> None:
        self.robot.send_action({"gripper.pos": 100.0 if open_ else 0.0})
