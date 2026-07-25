"""Camera-based checks — classical CV, no trained model.

  - pieces_on_tray(top_frame)     -> how many pieces remain (empty-pile detection)
  - pieces_in_gripper(wrist_frame)-> 0 / 1 / 2+  (single-grasp verification)

Both are blob counts inside a region of interest. They assume the pieces are
darker than a fairly uniform background (tray / gripper). If your lighting is
reversed, flip THRESH_BINARY_INV below.
"""
from __future__ import annotations

import logging

from .config import PerceptionCfg

log = logging.getLogger("autosort.perception")


class Perception:
    def __init__(self, cfg: PerceptionCfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._sim_remaining = 5  # dry-run: pretend the pile drains over a few cycles

    def pieces_on_tray(self, top_frame) -> int:
        if self.dry_run:
            self._sim_remaining = max(0, self._sim_remaining - 1)
            return self._sim_remaining
        return self._count_blobs(top_frame, self.cfg.pile_roi)

    def pieces_in_gripper(self, wrist_frame) -> int:
        if self.dry_run:
            return 1
        return self._count_blobs(wrist_frame, self.cfg.gripper_roi)

    # --- shared blob counter -----------------------------------------
    def _count_blobs(self, frame, roi) -> int:
        import cv2
        import numpy as np

        if frame is None:
            return 0
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if cv2.contourArea(c) >= self.cfg.min_piece_area)
