"""Camera-based checks — classical CV, no trained model.

  - pieces_on_tray(top_frame)     -> how many pieces remain (empty-pile detection)
  - pieces_in_gripper(wrist_frame)-> 0 / 1 / 2+  (single-grasp verification)

Both are blob counts inside a region of interest. They assume the pieces are
darker than a fairly uniform background (tray / gripper). If your lighting is
reversed, flip THRESH_BINARY_INV below.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import PerceptionCfg

log = logging.getLogger("autosort.perception")


class Perception:
    def __init__(self, cfg: PerceptionCfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._sim_remaining = 5  # dry-run: pretend the pile drains over a few cycles
        self._static_marker_zones: list[tuple[float, float, float, float]] = []

    def set_static_marker_zones(self, corner_arrays) -> None:
        """Permanent marker exclusion from the captured ArUco reference.

        Live per-frame detection can miss a marker (glare, a shadow edge), and
        a missed marker would instantly become a pick candidate again. The
        reference positions never lie by more than the drift, so they are
        excluded unconditionally, with extra padding to absorb that drift.
        """
        self._static_marker_zones = []
        for c in corner_arrays:
            x0, y0 = c.min(axis=0)
            x1, y1 = c.max(axis=0)
            pad = 0.5 * max(x1 - x0, y1 - y0)
            self._static_marker_zones.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))

    def pieces_on_tray(self, top_frame) -> int:
        if self.dry_run:
            self._sim_remaining = max(0, self._sim_remaining - 1)
            return self._sim_remaining
        return len(self._pile_blobs(top_frame))

    def pieces_in_gripper(self, wrist_frame) -> int:
        """0 or 1: is anything held? Judged by TOTAL dark area in the ROI.

        This deliberately does NOT try to count 2+: one gear already covers
        ~67% of the tight fingertip ROI (measured 13350 of 20000 px), so area
        saturates and normal grip-pose variance pushed single gears over any
        multi-piece threshold (field-observed false "2"s). Telling one piece
        from two is the gripper STALL WIDTH's job (pipeline) - two side-by-side
        pieces stop the fingers much wider than one.
        """
        if self.dry_run:
            return 1
        import cv2
        import numpy as np

        h, w = wrist_frame.shape[:2]
        x0, y0, x1, y1 = self.cfg.gripper_roi
        crop = cv2.cvtColor(wrist_frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)],
                            cv2.COLOR_BGR2GRAY)
        # Background is the 90th percentile, NOT the median: a held piece fills
        # most of this tight ROI, so the median IS the piece and "darker than
        # median" finds nothing (measured 2026-08-05: held median 64 vs true
        # background 166). The bright background stays visible around the piece.
        bg = float(np.percentile(crop, 90))
        mask = (crop < bg - self.cfg.contrast_margin).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return 1 if int(mask.sum()) >= self.cfg.min_piece_area else 0

    def pile_blobs_sorted(self, top_frame) -> list[tuple[float, float, float]]:
        """All pile blobs, biggest first — lets the caller apply its own filters
        (e.g. reachability) and fall through to the next candidate."""
        return sorted(self._pile_blobs(top_frame), key=lambda b: -b[0])

    def largest_piece_px(self, top_frame) -> tuple[float, float] | None:
        """Full-frame pixel centroid of the biggest piece in the pile ROI.

        This is the classical pick target: detection happens while the arm is
        at home (view unobstructed), so the arm never occludes what it measures.
        """
        if self.dry_run:
            return (480.0, 300.0)
        blobs = self._pile_blobs(top_frame)
        if not blobs:
            return None
        area, cx, cy, *_ = max(blobs)
        return (cx, cy)

    # --- shared blob detector ----------------------------------------
    def _pile_blobs(self, top_frame) -> list[tuple[float, float, float]]:
        """Pile-ROI blobs with anything that is an ArUco marker excluded.

        The drift-correction markers are taped in view of the pick zone and
        their black squares are perfect "dark piece" blobs — without this
        filter the pile never reads empty and a marker can win 'largest piece'
        (the arm then tries to pick up a fiducial). Detection is per-frame, so
        exclusion keeps working even as the camera drifts.
        """
        zones = self._marker_zones(top_frame) + self._static_marker_zones
        return [b for b in self._blobs(top_frame, self.cfg.pile_roi)
                if not any(x0 <= b[1] <= x1 and y0 <= b[2] <= y1
                           for x0, y0, x1, y1 in zones)]

    def _marker_zones(self, frame) -> list[tuple[float, float, float, float]]:
        """Padded pixel bounding boxes of every ArUco marker seen in `frame`."""
        from .recal import detect_markers

        zones = []
        for c in detect_markers(frame).values():
            x0, y0 = c.min(axis=0)
            x1, y1 = c.max(axis=0)
            pad = 0.25 * max(x1 - x0, y1 - y0)
            zones.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        return zones

    def save_debug_frame(self, frame, target_px=None, tag="refused") -> str | None:
        """Annotated snapshot to /tmp for post-mortem (Claude can read files but
        cannot open cameras, so refused picks save what the camera saw)."""
        import time as _time

        import cv2

        if frame is None:
            return None
        vis = frame.copy()
        h, w = vis.shape[:2]
        x0, y0, x1, y1 = self.cfg.pile_roi
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                      (0, 165, 255), 2)
        for x0m, y0m, x1m, y1m in self._marker_zones(frame):
            cv2.rectangle(vis, (int(x0m), int(y0m)), (int(x1m), int(y1m)),
                          (160, 160, 160), 2)
        import numpy as np
        for area, cx, cy, angle, aspect, color in self._pile_blobs(frame):
            cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.putText(vis, f"{int(area)} {color}", (int(cx) + 12, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if aspect >= 1.4:  # draw the long axis of elongated pieces
                dx, dy = np.cos(np.radians(angle)) * 24, np.sin(np.radians(angle)) * 24
                cv2.line(vis, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                         (255, 0, 255), 2)
        if target_px is not None:
            cv2.drawMarker(vis, (int(target_px[0]), int(target_px[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 44, 3)
        out = Path("/tmp/autosort_debug")
        out.mkdir(exist_ok=True)
        path = str(out / f"{tag}_{_time.strftime('%H%M%S')}.png")
        # frames are RGB (lerobot default); imwrite expects BGR - convert so
        # saved files show true colors (matters now that color is a feature)
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        return path

    def _blobs(self, frame, roi) -> list[tuple[float, float, float, float, float]]:
        """Pieces in `roi` as (area, cx, cy, angle_deg, aspect), centroids in
        FULL-frame pixels. angle_deg is the long-axis direction in IMAGE
        coordinates, [0,180); aspect >= 1 is long/short side of the minAreaRect
        (aspect ~1 = round piece, its angle is noise - callers must gate on it).

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
        # into phantom pieces. Anything within the margin of the median is
        # background/shadow. TWO-SIDED: white pieces (white spacers, plastic
        # screws) are LIGHTER than the surface and were invisible to the old
        # darker-only threshold. The light margin is separate because specular
        # glints skew bright; None disables the light side entirely.
        bg = float(np.median(gray))
        mask = (gray < max(1.0, bg - self.cfg.contrast_margin)).astype(np.uint8) * 255
        if self.cfg.contrast_margin_light is not None:
            mask |= (gray > min(254.0, bg + self.cfg.contrast_margin_light)).astype(np.uint8) * 255
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
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)
            if rw >= rh:
                angle = ang % 180.0
            else:
                angle = (ang + 90.0) % 180.0
            aspect = (max(rw, rh) / min(rw, rh)) if min(rw, rh) > 1e-6 else 1.0
            color = self._blob_color(crop, c) if crop.ndim == 3 else "unknown"
            out.append((area, ox + m["m10"] / m["m00"], oy + m["m01"] / m["m00"],
                        angle, aspect, color))
        return out

    @staticmethod
    def _blob_color(crop_rgb, contour) -> str:
        """Coarse color name for a blob: dark / white / gray / red / colored.

        Half the inventory separates on color, not shape: red sprocket vs
        black gear (same size+roundness), black vs white spacers. Coarse HSV
        buckets are robust to lighting; fine hue is not.
        """
        import cv2
        import numpy as np

        m = np.zeros(crop_rgb.shape[:2], np.uint8)
        cv2.drawContours(m, [contour], -1, 255, -1)
        mean = cv2.mean(crop_rgb, mask=m)[:3]
        hsv = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if s < 60:                     # achromatic
            if v < 90:
                return "dark"
            return "white" if v > 180 else "gray"
        if h < 12 or h > 168:          # OpenCV hue wraps at 180
            return "red"
        return "colored"

    def _count_blobs(self, frame, roi) -> int:
        return len(self._blobs(frame, roi))
