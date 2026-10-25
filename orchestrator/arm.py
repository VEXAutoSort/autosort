"""The arm layer: SO-101 follower doing one pick from the table into the cup.

The cycle only ever calls:

    connect() / disconnect() / torque_off()
    home()            scripted move to the HOME pose
    is_home()         every joint within home_tolerance_deg of HOME
    pick(timeout_s)   one attempt: pick a part off the table, drop it in the
                      cup (the box on top of the distributor), return HOME.
                      True on success; False means "nothing in the cup" and the
                      arm is back at HOME (the cycle retries / re-shakes).
    frame(name)       a frame from one of the arm-owned cameras (RGB, LeRobot order)

Two implementations behind `make_arm()`:

    GeometricArm  (arm.mode: geometric — the supported path)
        Reuses autosort/arm.py + autosort/grasp.py: segment the top-camera
        frame, PCA for centroid + grasp axis, homography to table mm, top-down
        IK to the grasp, lift, scripted move to the taught `drop` pose over the
        cup, open, HOME. Needs the homography (scripts/calibrate_homography.py),
        the IK link lengths, and the three poses — all in config.yaml `arm:`.

    PolicyArm     (arm.mode: act — kept for reference, NOT exercised on this rig)
        Runs a LeRobot ACT policy until the arm parks back at HOME.

Both drive the robot through LeRobot's SO101Follower; the arm owns the cameras
listed in `arm.cameras` (one process per V4L2 device), and hw.py pulls frames
for counting/classifying from here when the overhead camera is one of them.
lerobot / torch / cv2 are imported lazily so nothing here loads under --sim.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import Arm as ArmCfg
from .config import Cameras as CamerasCfg

log = logging.getLogger("orchestrator.arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LEGACY_DROP_POSE = "box_drop"  # the name autosort/arm.py uses for the pose over the cup


# --- pure config mapping (unit-tested without lerobot) ----------------------

def legacy_arm_cfg(port: str, cfg: ArmCfg):
    """orchestrator `arm:` section -> the autosort.config.ArmCfg that autosort/arm.py expects."""
    from autosort.config import ArmCfg as LegacyArmCfg

    poses = {name: dict(p) for name, p in cfg.poses.items()}
    if "drop" in poses and LEGACY_DROP_POSE not in poses:
        poses[LEGACY_DROP_POSE] = poses["drop"]
    for required in ("home", "inspect", LEGACY_DROP_POSE):
        if required not in poses:
            raise ValueError(f"arm.poses needs '{required}' (home / inspect / drop)")
    return LegacyArmCfg(
        port=port, id=cfg.id, device=cfg.device, policy=cfg.policy or "",
        pick_timeout_s=0.0, gripper_empty_pos=float(cfg.gripper_empty_pos),
        poses=poses, ik=dict(cfg.ik),
    )


def legacy_camera_cfgs(cfg: ArmCfg, cameras: CamerasCfg):
    """arm.cameras {lerobot_name: cameras.*} -> {lerobot_name: autosort CameraCfg}."""
    from autosort.config import CameraCfg

    out = {}
    for lerobot_name, cam_name in cfg.cameras.items():
        index = getattr(cameras, cam_name, None)
        if index is None:
            raise ValueError(f"arm.cameras maps {lerobot_name} -> cameras.{cam_name}, which is null/unknown")
        w, h = cameras.size(cam_name)
        out[lerobot_name] = CameraCfg(index=index, width=w, height=h, fps=30)
    return out


def make_arm(port: str, cfg: ArmCfg, cameras: CamerasCfg):
    if cfg.mode == "geometric":
        return GeometricArm(port, cfg, cameras)
    if cfg.mode == "act":
        return PolicyArm(port, cfg, cameras)
    raise ValueError(f"arm.mode must be geometric | act, not {cfg.mode!r}")


# --- shared LeRobot plumbing --------------------------------------------------

class _BaseArm:
    def __init__(self, port: str, cfg: ArmCfg, cameras: CamerasCfg):
        self.port = port
        self.cfg = cfg
        self.cameras = cameras
        self.arm = None  # autosort.arm.Arm

    def connect(self) -> None:
        from autosort.arm import Arm

        self.arm = Arm(legacy_arm_cfg(self.port, self.cfg), legacy_camera_cfgs(self.cfg, self.cameras))
        self.arm.connect()

    def disconnect(self) -> None:
        if self.arm is not None:
            try:
                self.arm.disconnect()
            finally:
                self.arm = None

    def torque_off(self) -> None:
        """Best-effort: free every joint (ESTOP path, never raises)."""
        robot = getattr(self.arm, "robot", None)
        if robot is None:
            return
        try:
            robot.bus.disable_torque()
        except Exception as e:  # noqa: BLE001
            log.warning("arm torque_off failed: %s", e)

    def observe(self) -> dict[str, Any]:
        return self.arm.observe()

    def frame(self, lerobot_name: str):
        return self.arm.frame(lerobot_name)

    def joint_positions(self) -> dict[str, float]:
        obs = self.observe()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS if f"{j}.pos" in obs}

    def is_home(self) -> bool:
        now = self.joint_positions()
        home = self.cfg.poses["home"]
        return all(abs(now.get(j, 1e9) - float(target)) <= self.cfg.home_tolerance_deg
                   for j, target in home.items() if j != "gripper")

    def wait_home(self, timeout_s: float = 4.0) -> bool:
        """autosort.arm.Arm.move_to() only sleeps 0.6 s; a long move may still be
        under way, so poll the joints instead of trusting the sleep."""
        t0 = time.monotonic()
        while True:
            if self.is_home():
                return True
            if time.monotonic() - t0 > timeout_s:
                log.warning("arm not at HOME after %.1fs: %s", timeout_s, self.joint_positions())
                return False
            time.sleep(0.05)

    def home(self) -> None:
        self.arm.home()
        self.wait_home()


# --- geometric (IK) pick ------------------------------------------------------

class GeometricArm(_BaseArm):
    def connect(self) -> None:
        from autosort.config import GraspCfg
        from autosort.grasp import GeometricGrasper

        super().connect()
        g = self.cfg.grasp
        self.grasper = GeometricGrasper(
            self.arm, GraspCfg(mode="geometric", min_piece_area=int(g.get("min_piece_area", 300)),
                               homography=g["homography"]))
        log.info("geometric arm connected on %s (cameras %s)", self.port, list(self.cfg.cameras))

    def pick(self, timeout_s: float) -> bool:
        t0 = time.monotonic()
        top = self.cfg.top_camera
        pose = self.grasper.plan(self.frame(top))  # expects RGB — LeRobot's default
        if pose is None:
            log.info("no graspable piece in the %s frame", top)
            return False
        log.info("grasp @ table (%.0f, %.0f) mm, w=%.0f mm", pose.x, pose.y, pose.width)
        self.arm.execute_grasp(pose)                # descend, close, lift to inspect
        if not self.arm.gripper_holding():
            log.info("empty grasp — back to HOME")
            self.arm.drop_back()
            self.arm.home()
            return False
        self.arm.place_in_box()                     # taught drop pose over the cup, open, HOME
        home = self.wait_home(max(1.0, timeout_s - (time.monotonic() - t0)))
        if time.monotonic() - t0 > timeout_s:
            log.warning("pick took %.1fs (> %.0fs timeout) — tune the scripted dwell times",
                        time.monotonic() - t0, timeout_s)
        return home


# --- ACT policy pick (reference only) ----------------------------------------

class PolicyArm(_BaseArm):
    def __init__(self, port: str, cfg: ArmCfg, cameras: CamerasCfg):
        super().__init__(port, cfg, cameras)
        self.policy = None
        self._pre = self._post = None
        self._features = None

    def connect(self) -> None:
        super().connect()
        self.load_policy()

    def load_policy(self) -> None:
        if self.policy is not None:
            return
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.policy = ACTPolicy.from_pretrained(self.cfg.policy)
        self.policy.to(self.cfg.device).eval()
        try:  # lerobot >= 0.4: normalisation lives in pre/post processors
            from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
            from lerobot.policies.factory import make_pre_post_processors

            robot = self.arm.robot
            self._pre, self._post = make_pre_post_processors(
                policy_cfg=self.policy.config, pretrained_path=self.cfg.policy,
                preprocessor_overrides={"device_processor": {"device": self.cfg.device}})
            self._features = combine_feature_dicts(
                hw_to_dataset_features(robot.observation_features, "observation"),
                hw_to_dataset_features(robot.action_features, "action"))
        except ImportError:
            log.warning("lerobot pre/post processors unavailable — using raw select_action()")
        log.info("ACT policy loaded: %s on %s", self.cfg.policy, self.cfg.device)

    def pick(self, timeout_s: float) -> bool:
        """One policy episode. Success = the arm is back at HOME after min_policy_s."""
        import torch

        self.policy.reset()
        if self._pre is not None:
            self._pre.reset(); self._post.reset()
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout_s:
                log.warning("pick timed out after %.0fs — scripted HOME", timeout_s)
                self.home()
                return False
            obs = self.observe()
            with torch.no_grad():
                action = self._policy_step(obs)
            self.arm.robot.send_action(action)
            if elapsed > self.cfg.min_policy_s and self.is_home():
                return True

    def _policy_step(self, obs: dict[str, Any]) -> dict[str, float]:
        if self._pre is not None:
            from lerobot.datasets.utils import build_dataset_frame
            from lerobot.utils.control_utils import predict_action
            from lerobot.utils.robot_utils import make_robot_action

            frame = build_dataset_frame(self._features, obs, prefix="observation")
            values = predict_action(frame, self.policy, self.cfg.device, self._pre, self._post,
                                    use_amp=self.policy.config.use_amp, task=self.cfg.task,
                                    robot_type=self.arm.robot.robot_type)
            return make_robot_action(values, self._features)
        return self.policy.select_action(obs)
