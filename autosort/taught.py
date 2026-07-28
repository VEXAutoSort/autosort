"""Taught poses: everything the robot learned from being moved by hand.

tools/teach.py writes taught.json; this module loads it and turns "piece at
pixel (x, y)" into a grasp pose by inverse-distance-weighted blending of the
taught grid points. No IK, no camera calibration — the grid IS the calibration.

taught.json:
  {
    "grid":        [{"pixel": [x, y], "joints": {joint: deg, ...}}, ...],
    "hover_delta": {joint: deg_offset, ...},
    "poses":       {"home": {...}, "inspect": {...}, "box_drop": {...}},
    "gripper_open":   float,
    "gripper_closed": float
  }
"""
from __future__ import annotations

import json
from pathlib import Path

from .motion import JOINTS

DEFAULT_TAUGHT_PATH = Path(__file__).resolve().parent.parent / "taught.json"


class Taught:
    def __init__(self, data: dict):
        self.grid: list = data["grid"]
        self.hover_delta: dict = data["hover_delta"]
        self.poses: dict = data.get("poses", {})
        self.gripper_open: float = data["gripper_open"]
        self.gripper_closed: float = data["gripper_closed"]

    @staticmethod
    def load(path: str | Path | None = None) -> "Taught":
        p = Path(path) if path else DEFAULT_TAUGHT_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python tools/teach.py` first (15 min, one-time)."
            )
        return Taught(json.loads(p.read_text()))

    def grasp_for_pixel(self, px: float, py: float, power: float = 2.0) -> dict[str, float]:
        """IDW blend of taught grid poses, weighted by pixel distance."""
        weights, poses = [], []
        for pt in self.grid:
            gx, gy = pt["pixel"]
            d2 = (px - gx) ** 2 + (py - gy) ** 2
            if d2 < 1e-6:
                return dict(pt["joints"])
            weights.append(1.0 / (d2 ** (power / 2)))
            poses.append(pt["joints"])
        wsum = sum(weights)
        return {j: sum(w * p[j] for w, p in zip(weights, poses)) / wsum for j in JOINTS}

    def hover_for(self, grasp: dict[str, float]) -> dict[str, float]:
        return {j: grasp[j] + self.hover_delta.get(j, 0.0) for j in JOINTS}
