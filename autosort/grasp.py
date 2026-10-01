"""Grasp strategies — switch between a geometric algorithm and the ACT policy.

Both implement the same `Grasper.grasp()` interface, so the pipeline doesn't care
which is in use. Pick the mode in config.yaml (`grasp.mode: geometric | act`).

- GeometricGrasper: segment the piece in the top view, find its centroid + principal
  axis (PCA), and grasp across its narrow dimension. No training data. Ideal for
  isolated rigid parts on a plain surface. (Inspired by Ambi's learning-free pose
  estimation: segment -> centroid + principal axes -> pick.)
- ACTGrasper: run the trained ACT policy. Better for clutter / contact-rich grasps.

Heavy imports (cv2, numpy) are lazy so dry-run needs neither.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

from .config import Config, GraspCfg

log = logging.getLogger("autosort.grasp")


@dataclass
class GraspPose:
    """A top-down grasp in table coordinates."""
    x: float        # table position, mm
    y: float        # table position, mm
    theta: float    # gripper yaw so the jaws close across the piece, rad
    width: float    # piece width across the grasp axis, mm (informational)


class Grasper(Protocol):
    def grasp(self) -> bool:
        """Attempt one pick; on success the piece is held at the inspect pose.
        Returns False if no graspable piece was found."""
        ...


class GeometricGrasper:
    def __init__(self, arm, cfg: GraspCfg, dry_run: bool = False):
        self.arm = arm
        self.cfg = cfg
        self.dry_run = dry_run

    def grasp(self) -> bool:
        if self.dry_run:
            log.info("[dry-run] geometric grasp()")
            return True
        pose = self.plan(self.arm.frame("top"))
        if pose is None:
            log.info("no graspable piece in view")
            return False
        log.info(
            "geometric grasp @ table (%.0f, %.0f) mm  theta=%.0f deg  w=%.0f mm",
            pose.x, pose.y, math.degrees(pose.theta), pose.width,
        )
        self.arm.execute_grasp(pose)
        return True

    def plan(self, top_frame) -> GraspPose | None:
        """Segment the largest isolated piece and compute its grasp pose."""
        import cv2
        import numpy as np

        if top_frame is None:
            return None
        gray = cv2.cvtColor(top_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.cfg.min_piece_area]
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        mean, eigvecs = cv2.PCACompute(c, mean=None)  # eigvecs[0]=long axis, [1]=short axis
        centroid = mean[0]
        long_axis, short_axis = eigvecs[0], eigvecs[1]
        # jaws close across the NARROW dimension -> gripper axis == short axis
        proj = (c - centroid) @ short_axis
        width_px = float(proj.max() - proj.min())

        # pixels -> table (mm) via the calibrated homography
        H = np.asarray(self.cfg.homography, dtype=np.float64)
        (tx, ty), = cv2.perspectiveTransform(centroid[None, None, :], H)[0]
        # transform the two grasp-contact points to table frame -> width (mm) + gripper yaw
        e0 = centroid + short_axis * (width_px / 2)
        e1 = centroid - short_axis * (width_px / 2)
        pts_mm = cv2.perspectiveTransform(np.array([e0, e1], np.float32)[None], H)[0]
        d_mm = pts_mm[1] - pts_mm[0]
        width_mm = float(np.linalg.norm(d_mm))
        theta = math.atan2(float(d_mm[1]), float(d_mm[0]))
        return GraspPose(float(tx), float(ty), theta, width_mm)


class ACTGrasper:
    def __init__(self, arm, dry_run: bool = False):
        self.arm = arm
        self.dry_run = dry_run

    def grasp(self) -> bool:
        self.arm.pick_act()  # dry-run safe
        return True


def make_grasper(cfg: Config, arm) -> Grasper:
    if cfg.grasp.mode == "geometric":
        return GeometricGrasper(arm, cfg.grasp, cfg.run.dry_run)
    return ACTGrasper(arm, cfg.run.dry_run)
