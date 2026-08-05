"""Camera-drift auto-correction from ArUco markers.

The homography in analytic.py maps top-camera pixels to table coordinates and
is only valid for the camera pose it was taught under - a 1 degree camera tilt
is ~8 mm of reach error, 2 degrees is a missed grasp. Rather than re-teaching
after every bump, four ArUco markers taped permanently around the pick zone
give the drift away:

  capture (tools/capture_markers.py, run when picking is KNOWN accurate):
      detect markers, save their pixel corners to markers_ref.json
  every pick cycle:
      detect markers again, fit a homography mapping CURRENT pixels to
      REFERENCE pixels, and push each detected target through it before the
      solver. The solver then always sees reference-era pixels, so camera
      drift cancels without touching taught.json.

Physical marker positions are never measured - only their pixel positions at
capture time matter.

Safety: correction is refused (RecalError) if fewer than min_markers are
found or if the implied shift exceeds max_correction_px - a huge shift means
the camera was outright moved, and silently "correcting" through it would put
picks far outside anything validated. The caller decides whether to proceed
uncorrected or stop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("autosort.recal")


class RecalError(RuntimeError):
    """Correction unavailable or implausible - caller must not trust this frame."""


def detect_markers(frame) -> dict[int, np.ndarray]:
    """Marker id -> 4x2 float pixel corners for every marker seen in `frame`."""
    import cv2

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dic).detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.flatten(), corners)}


class Recalibrator:
    def __init__(self, ref_file: str | Path, min_markers: int = 3,
                 max_correction_px: float = 40.0):
        self.ref_file = Path(ref_file)
        self.min_markers = min_markers
        self.max_correction_px = max_correction_px
        ref = json.loads(self.ref_file.read_text())
        self.ref = {int(k): np.asarray(v, dtype=float) for k, v in ref["markers"].items()}
        if len(self.ref) < min_markers:
            raise RecalError(f"reference {self.ref_file} has only {len(self.ref)} markers")

    def correction(self, frame) -> np.ndarray:
        """3x3 homography mapping CURRENT frame pixels to REFERENCE-era pixels."""
        import cv2

        seen = detect_markers(frame)
        common = sorted(set(seen) & set(self.ref))
        if len(common) < self.min_markers:
            raise RecalError(
                f"only {len(common)} of {len(self.ref)} reference markers visible "
                f"(ids {common}) - occluded, glared, or the camera moved a lot")
        cur = np.vstack([seen[i] for i in common])
        ref = np.vstack([self.ref[i] for i in common])
        H, inliers = cv2.findHomography(cur, ref, cv2.RANSAC, 3.0)
        if H is None or inliers.sum() < 0.75 * len(cur):
            raise RecalError("marker correspondence would not fit a homography")
        shift = float(np.linalg.norm(ref - cur, axis=1).mean())
        if shift > self.max_correction_px:
            raise RecalError(
                f"markers moved {shift:.0f} px on average (limit {self.max_correction_px:.0f}) - "
                "the camera was moved, not nudged. Verify picking, then re-capture the "
                "reference with tools/capture_markers.py")
        if shift > 2.0:
            log.info("camera drift detected: %.1f px average - correcting", shift)
        return H

    @staticmethod
    def apply(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
        v = H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])
