"""SO-ARM101 follower with two interchangeable pick backends.

  pick_mode: classical  — detect the piece from the top camera while the arm is
      at home (clear view), blend hand-taught grid poses into a grasp, execute a
      scripted hover -> descend -> close -> lift. No trained model. Teach once
      with tools/teach.py.

  pick_mode: act        — run a trained ACT policy for the grasp (experimental
      until a pick-from-pile policy is trained). Observations go through the
      policy's own pre/post processor pipelines — required in lerobot >= 0.6,
      raw observations are NOT accepted by select_action.

Everything after the grasp (inspect, box drop, home) is scripted from taught
poses for both modes, and every scripted move is interpolated (motion.py) so
the arm never lurches.

Heavy imports (lerobot, torch) stay lazy so dry-run works without them.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import ArmCfg, CameraCfg
from .motion import JOINTS, move_smooth, read_joints
from .taught import Taught

log = logging.getLogger("autosort.arm")


class Arm:
    def __init__(self, cfg: ArmCfg, cameras: dict[str, CameraCfg], dry_run: bool = False):
        self.cfg = cfg
        self.cameras = cameras
        self.dry_run = dry_run
        self.robot = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.taught: Taught | None = None

    # --- lifecycle ----------------------------------------------------
    def connect(self) -> None:
        if not self.dry_run:
            self.taught = Taught.load(self.cfg.taught_file)
        if self.dry_run:
            log.info("[dry-run] arm connected (pick_mode=%s)", self.cfg.pick_mode)
            return
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # 0.6.x layout
            from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
            from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cams = {
            name: OpenCVCameraConfig(index_or_path=c.index, width=c.width, height=c.height, fps=c.fps)
            for name, c in self.cameras.items()
            if name in ("top", "wrist")  # only arm cameras; 'box' belongs to the classifier
        }
        self.robot = SO101Follower(
            SO101FollowerConfig(port=self.cfg.port, id=self.cfg.id, cameras=cams)
        )
        # the servo bus occasionally corrupts a packet; retry instead of dying
        for attempt in range(4):
            try:
                self.robot.connect(calibrate=False)
                break
            except ConnectionError as e:
                if attempt == 3:
                    raise
                log.warning("bus glitch on connect (%s); retrying in 2s [%d/3]", e, attempt + 1)
                try:
                    self.robot.bus.disconnect()
                except Exception:
                    pass
                time.sleep(2)
        if self.cfg.pick_mode == "act":
            self._load_policy()
        log.info("arm connected on %s (pick_mode=%s)", self.cfg.port, self.cfg.pick_mode)

    def _load_policy(self) -> None:
        """ACT policy + its processor pipelines (normalization lives there, not in the policy)."""
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.policy = ACTPolicy.from_pretrained(self.cfg.policy)
        self.policy.config.device = self.cfg.device
        self.policy.to(self.cfg.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.cfg.policy,
            preprocessor_overrides={"device_processor": {"device": self.cfg.device}},
        )
        log.info("ACT policy loaded: %s on %s", self.cfg.policy, self.cfg.device)

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()

    # --- observation --------------------------------------------------
    def observe(self) -> dict[str, Any]:
        return {} if self.dry_run else self.robot.get_observation()

    def frame(self, name: str):
        """Latest image from an arm camera ('top' or 'wrist'); None in dry-run."""
        return self.observe().get(name)

    def gripper_pos(self) -> float:
        return 50.0 if self.dry_run else read_joints(self.robot)["gripper"]

    def gripper_holding(self) -> bool:
        """True if the fingers stalled ABOVE the commanded closed position.

        NOTE: pieces with holes/spokes (gears!) can swallow the fingertips, so a
        real hold can read near-closed. The pipeline therefore treats the wrist
        camera as the deciding vote — this check alone is advisory.
        """
        if self.dry_run:
            return True
        pos = self.gripper_pos()
        gap = pos - self.taught.gripper_closed
        holding = gap > self.cfg.gripper_holding_margin
        log.info("grip check: present=%.1f commanded=%.1f gap=%.1f margin=%.1f -> %s",
                 pos, self.taught.gripper_closed, gap, self.cfg.gripper_holding_margin,
                 "holding" if holding else "empty")
        return holding

    def open_gripper(self) -> None:
        """Park the fingers at the approach width so picks start pre-opened."""
        if self.dry_run:
            return
        move_smooth(self.robot, {"gripper": self.taught.gripper_open}, duration_s=0.4)

    # --- pick backends ------------------------------------------------
    def pick(self, target_px: tuple[float, float] | None = None) -> None:
        """Grasp one piece, end holding at the 'inspect' pose."""
        if self.dry_run:
            log.info("[dry-run] pick(%s)", target_px)
            return
        if self.cfg.pick_mode == "classical":
            self._pick_classical(target_px)
        else:
            self._pick_act()
        self.move_to("inspect")  # standardize the pose for the single-piece check

    def _pick_classical(self, target_px: tuple[float, float] | None) -> None:
        if target_px is None:
            log.warning("classical pick called with no target — skipping")
            return
        grasp = self.taught.grasp_for_pixel(*target_px)
        for joint, delta in (self.cfg.pick_joint_offsets or {}).items():
            grasp[joint] = grasp.get(joint, 0.0) + delta
        hover = self.taught.hover_for(grasp)
        open_g, closed_g = self.taught.gripper_open, self.taught.gripper_closed

        # set the gripper width FIRST, in place, so it stays constant through
        # the whole approach instead of interpolating from wherever it was
        move_smooth(self.robot, {"gripper": open_g}, duration_s=0.4)
        move_smooth(self.robot, {**hover, "gripper": open_g}, duration_s=1.3)
        move_smooth(self.robot, {**grasp, "gripper": open_g}, duration_s=0.9)
        move_smooth(self.robot, {**grasp, "gripper": closed_g}, duration_s=0.5)
        time.sleep(0.3)
        move_smooth(self.robot, {**hover, "gripper": closed_g}, duration_s=0.9)

    def _pick_act(self) -> None:
        import torch

        self.policy.reset()
        period = 1.0 / self.cfg.act_fps
        t0 = time.time()
        while time.time() - t0 < self.cfg.pick_timeout_s:
            t_step = time.time()
            obs = self._policy_observation()
            batch = self.preprocessor(obs)
            with torch.no_grad():
                action_t = self.policy.select_action(batch)
            action_t = self.postprocessor(action_t)
            action = {f"{j}.pos": float(action_t[0][i]) for i, j in enumerate(JOINTS)}
            self.robot.send_action(action)
            time.sleep(max(0.0, period - (time.time() - t_step)))

    def _policy_observation(self) -> dict[str, Any]:
        """Robot observation -> the keys the trained policy expects.

        Camera names on the robot ('top'/'wrist') rarely match the dataset's
        camera keys (e.g. 'Top view'); arm.policy_camera_names maps them.
        """
        import numpy as np

        raw = self.robot.get_observation()
        obs: dict[str, Any] = {}
        state = [raw[f"{j}.pos"] for j in JOINTS]
        obs["observation.state"] = np.asarray(state, dtype=np.float32)
        for robot_name, dataset_name in self.cfg.policy_camera_names.items():
            obs[f"observation.images.{dataset_name}"] = raw[robot_name]
        return obs

    # --- scripted motions ---------------------------------------------
    def place_in_box(self) -> None:
        """Carry the held piece to the enclosure/box and release it."""
        if self.dry_run:
            log.info("[dry-run] place_in_box()")
            return
        closed = self.taught.gripper_closed
        pose = self.taught.poses["box_drop"]
        move_smooth(self.robot, {**pose, "gripper": closed}, duration_s=1.5)
        move_smooth(self.robot, {**pose, "gripper": self.taught.gripper_open}, duration_s=0.4)
        time.sleep(0.4)
        self.home()

    def drop_back(self) -> None:
        """Release a multi-grab back over the pile."""
        if self.dry_run:
            log.info("[dry-run] drop_back()")
            return
        pose = self.taught.poses["home"]
        move_smooth(self.robot, {**pose, "gripper": self.taught.gripper_closed}, duration_s=1.2)
        move_smooth(self.robot, {**pose, "gripper": self.taught.gripper_open}, duration_s=0.4)
        time.sleep(0.3)

    def home(self) -> None:
        if self.dry_run:
            log.info("[dry-run] home()")
            return
        self.move_to("home")

    def move_to(self, pose_name: str) -> None:
        """Move to a taught pose (falls back to config.yaml poses if not taught).

        The gripper is deliberately EXCLUDED: named poses only position the arm,
        and the fingers hold their current width. Gripper changes happen only in
        the explicit pick/place steps, so the width never drifts mid-travel.
        """
        if self.taught and pose_name in self.taught.poses:
            target = self.taught.poses[pose_name]
        else:
            target = self.cfg.poses[pose_name]
        target = {k: v for k, v in target.items() if k != "gripper"}
        move_smooth(self.robot, target, duration_s=1.4)
