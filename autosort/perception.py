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
        """0 / 1 / 2+ pieces held, judged by TOTAL dark area, not blob count.

        Blob counting is fragile here: a single gear gets split into two blobs by
        a fingertip crossing it or by its own spokes, which used to read as "2
        pieces" and trigger a needless drop-back. Total area doesn't care how the
        region is carved up - two pieces genuinely cover about twice the area.
        """
        if self.dry_run:
            return 1
        blobs = self._blobs(wrist_frame, self.cfg.gripper_roi)
        total = sum(b[0] for b in blobs)
        if total < self.cfg.min_piece_area:
            return 0
        if total >= self.cfg.multi_piece_area:
            return 2
        return 1

    def largest_piece_px(self, top_frame) -> tuple[float, float] | None:
        """Full-frame pixel centroid of the biggest piece in the pile ROI.

        This is the classical pick target: detection happens while the arm is
        at home (view unobstructed), so the arm never occludes what it measures.
        """
        if self.dry_run:
            return (480.0, 300.0)
        blobs = self._blobs(top_frame, self.cfg.pile_roi)
        if not blobs:
            return None
        area, cx, cy = max(blobs)
        return (cx, cy)

    # --- shared blob detector ----------------------------------------
    def _blobs(self, frame, roi) -> list[tuple[float, float, float]]:
        """Pieces in `roi` as (area, cx, cy) with centroids in FULL-frame pixels.

        Assumes pieces darker than a light background (per config decision).
        If your surface is dark and pieces are light, drop THRESH_BINARY_INV.
        """
        import cv2
        import numpy as np

        if frame is None:
            return []
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        ox, oy = int(x0 * w), int(y0 * h)
        crop = frame[oy:int(y1 * h), ox:int(x1 * w)]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold relative to the background brightness instead of Otsu: Otsu
        # ALWAYS splits the image, so on an empty crop it promotes soft shadows
        # into phantom pieces. Real pieces are far darker than the surface;
        # anything within contrast_margin of the median is background/shadow.
        bg = float(np.median(gray))
        thr = max(1.0, bg - self.cfg.contrast_margin)
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # bridge thin gaps (a fingertip or gear spoke splitting one piece in two)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_piece_area or area > self.cfg.max_piece_area:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            out.append((area, ox + m["m10"] / m["m00"], oy + m["m01"] / m["m00"]))
        return out

    def _count_blobs(self, frame, roi) -> int:
        return len(self._blobs(frame, roi))
