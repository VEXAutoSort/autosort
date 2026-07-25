"""SO-ARM101 follower: ACT pick + scripted place, with gripper feedback.

Split of responsibility:
  - the ACT policy does the hard part — grasp ONE piece from the pile and lift it.
  - everything after (move to the box, release) is scripted from fixed joint
    poses, because a fixed target doesn't need a learned policy.

Heavy imports (lerobot, torch) are loaded lazily inside connect()/pick() so the
rest of the system — and dry-run mode — work without them installed.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import ArmCfg, CameraCfg

log = logging.getLogger("autosort.arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


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
        from lerobot.policies.act.modeling_act import ACTPolicy

        cams = {
            name: OpenCVCameraConfig(index_or_path=c.index, width=c.width, height=c.height, fps=c.fps)
            for name, c in self.cameras.items()
            if name in ("top", "wrist")  # only policy cameras go on the robot
        }
        self.robot = SO101Follower(
            SO101FollowerConfig(port=self.cfg.port, id=self.cfg.id, cameras=cams)
        )
        self.robot.connect()
        self.policy = ACTPolicy.from_pretrained(self.cfg.policy)
        self.policy.eval()
        log.info("arm connected on %s (policy=%s)", self.cfg.port, self.cfg.policy)

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()

    # --- observation --------------------------------------------------
    def observe(self) -> dict[str, Any]:
        return {} if self.dry_run else self.robot.get_observation()

    def frame(self, name: str):
        """Latest image from a policy camera ('top' or 'wrist'); None in dry-run."""
        return self.observe().get(name)

    def gripper_pos(self) -> float:
        return 50.0 if self.dry_run else float(self.observe().get("gripper.pos", 0.0))

    def gripper_holding(self) -> bool:
        """True if the gripper stopped on something; False if it closed on nothing."""
        return self.gripper_pos() > self.cfg.gripper_empty_pos

    # --- motions ------------------------------------------------------
    def pick(self) -> None:
        """Run the ACT policy for one grasp, then hold at the 'inspect' pose."""
        if self.dry_run:
            log.info("[dry-run] pick()")
            return
        import torch

        self.policy.reset()
        t0 = time.time()
        while time.time() - t0 < self.cfg.pick_timeout_s:
            obs = self.robot.get_observation()
            with torch.no_grad():
                # NOTE: some LeRobot versions need make_pre_post_processors() around
                # this call — wire your policy's preprocessing here if select_action
                # doesn't accept the raw observation dict.
                action = self.policy.select_action(obs)
            self.robot.send_action(action)
        self.move_to("inspect")  # standardize the pose for the single-piece check

    def place_in_box(self) -> None:
        """Carry the held piece to the enclosure and release it."""
        if self.dry_run:
            log.info("[dry-run] place_in_box()")
            return
        self.move_to("box_drop")
        self._set_gripper(open_=True)
        time.sleep(0.4)
        self.move_to("home")

    def drop_back(self) -> None:
        """Release a multi-grab back over the pile."""
        if self.dry_run:
            log.info("[dry-run] drop_back()")
            return
        self.move_to("home")
        self._set_gripper(open_=True)
        time.sleep(0.3)

    def home(self) -> None:
        if self.dry_run:
            log.info("[dry-run] home()")
            return
        self.move_to("home")

    # --- low level ----------------------------------------------------
    def move_to(self, pose_name: str) -> None:
        target = self.cfg.poses[pose_name]
        self.robot.send_action({f"{j}.pos": float(target[j]) for j in JOINTS if j in target})
        time.sleep(0.6)  # let the move settle

    def _set_gripper(self, open_: bool) -> None:
        self.robot.send_action({"gripper.pos": 100.0 if open_ else 0.0})
