"""The arm layer: SO-101 follower driven by a LeRobot ACT policy.

This is Henry's layer. The cycle only ever calls four things:

    connect() / disconnect()
    home()            scripted move to the HOME pose
    is_home()         every joint within home_tolerance_deg of HOME
    pick(timeout_s)   run the policy until it parks back at HOME (success) or
                      the timeout hits (failure -> scripted HOME, returns False)
    frame(name)       a camera frame from the robot's own cameras (the policy owns them)

Swap the policy by changing arm.policy in config.yaml; swap the inference recipe
by editing `_policy_step()` only. LeRobot pinned to 0.5.2 (what Henry trains with).
Heavy imports (lerobot, torch) are lazy so nothing here loads under --sim.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import Arm as ArmCfg
from .config import Cameras as CamerasCfg

log = logging.getLogger("orchestrator.arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class PolicyArm:
    def __init__(self, port: str, cfg: ArmCfg, cameras: CamerasCfg):
        self.port = port
        self.cfg = cfg
        self.cameras = cameras
        self.robot = None
        self.policy = None
        self._pre = self._post = None
        self._features = None

    # --- lifecycle ----------------------------------------------------
    def connect(self) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        cams = {}
        for obs_name, cam_name in self.cfg.policy_cameras.items():
            index = getattr(self.cameras, cam_name)
            if index is None:
                raise ValueError(f"arm.policy_cameras maps {obs_name} -> cameras.{cam_name}, which is null")
            cams[obs_name] = OpenCVCameraConfig(
                index_or_path=index, width=self.cameras.width, height=self.cameras.height, fps=30)
        self.robot = SO101Follower(SO101FollowerConfig(port=self.port, id=self.cfg.id, cameras=cams))
        self.robot.connect()
        log.info("arm connected on %s (cameras %s)", self.port, list(cams))

    def load_policy(self) -> None:
        if self.policy is not None:
            return
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.policy = ACTPolicy.from_pretrained(self.cfg.policy)
        self.policy.to(self.cfg.device).eval()
        try:  # lerobot >= 0.4: normalisation lives in pre/post processors
            from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
            from lerobot.policies.factory import make_pre_post_processors

            self._pre, self._post = make_pre_post_processors(
                policy_cfg=self.policy.config, pretrained_path=self.cfg.policy,
                preprocessor_overrides={"device_processor": {"device": self.cfg.device}})
            self._features = combine_feature_dicts(
                hw_to_dataset_features(self.robot.observation_features, "observation"),
                hw_to_dataset_features(self.robot.action_features, "action"))
        except ImportError:
            log.warning("lerobot pre/post processors unavailable — using raw select_action()")
        log.info("ACT policy loaded: %s on %s", self.cfg.policy, self.cfg.device)

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None

    def torque_off(self) -> None:
        """Best-effort: free every joint (used by ESTOP)."""
        if self.robot is None:
            return
        try:
            self.robot.bus.disable_torque()
        except Exception as e:  # noqa: BLE001 — ESTOP path, never raise
            log.warning("arm torque_off failed: %s", e)

    # --- observation --------------------------------------------------
    def observe(self) -> dict[str, Any]:
        return self.robot.get_observation()

    def frame(self, obs_name: str):
        return self.observe().get(obs_name)

    def joint_positions(self) -> dict[str, float]:
        obs = self.observe()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS if f"{j}.pos" in obs}

    def is_home(self) -> bool:
        now = self.joint_positions()
        return all(abs(now.get(j, 1e9) - float(target)) <= self.cfg.home_tolerance_deg
                   for j, target in self.cfg.home_pose.items())

    # --- motions ------------------------------------------------------
    def home(self, settle_s: float = 1.0) -> None:
        self.robot.send_action({f"{j}.pos": float(v) for j, v in self.cfg.home_pose.items()})
        time.sleep(settle_s)

    def pick(self, timeout_s: float) -> bool:
        """One policy episode. Success = the arm is back at HOME after min_policy_s."""
        import torch

        self.load_policy()
        self.policy.reset()
        if self._pre is not None:
            self._pre.reset(); self._post.reset()
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout_s:
                log.warning("pick timed out after %.0fs — scripted HOME", timeout_s)
                self.home()
                return self.is_home() and False
            obs = self.observe()
            with torch.no_grad():
                action = self._policy_step(obs)
            self.robot.send_action(action)
            if elapsed > self.cfg.min_policy_s and self.is_home():
                return True

    def _policy_step(self, obs: dict[str, Any]) -> dict[str, float]:
        """obs dict from the robot -> action dict for the robot. The one place
        that knows the LeRobot inference recipe (mirror of lerobot_record.py)."""
        if self._pre is not None:
            from lerobot.datasets.utils import build_dataset_frame
            from lerobot.utils.control_utils import predict_action
            from lerobot.utils.robot_utils import make_robot_action

            frame = build_dataset_frame(self._features, obs, prefix="observation")
            values = predict_action(frame, self.policy, self.cfg.device, self._pre, self._post,
                                    use_amp=self.policy.config.use_amp, task=self.cfg.task,
                                    robot_type=self.robot.robot_type)
            return make_robot_action(values, self._features)
        return self.policy.select_action(obs)
