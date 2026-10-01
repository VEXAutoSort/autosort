"""Turn a frame of the stage into a millimetre measurement of the piece on it.

Segmentation is a difference against a reference shot of the empty stage, not a
threshold on the frame itself — a steel screw is as bright as white paper, but it
still differs from bare paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DIFF_THRESHOLD = 20      # grey levels a pixel must differ from the empty stage
MIN_AREA_PX = 150        # smaller blobs are noise, not pieces
MIN_HOLE_FRACTION = 0.005  # a hole must be this fraction of the piece to count


@dataclass
class Shape:
    """One piece, measured. Lengths in mm, ratios dimensionless."""

    length_mm: float
    width_mm: float
    area_mm2: float
    extent: float        # area / bounding-box area — low for a thin shank in a wide box
    circularity: float   # 4*pi*area / perimeter^2 — 1.0 is a circle
    solidity: float      # area / convex-hull area — below 1 for gear teeth
    holes: int

    def __str__(self) -> str:
        return (f"{self.length_mm:.2f} x {self.width_mm:.2f} mm  "
                f"extent {self.extent:.2f}  circ {self.circularity:.2f}  "
                f"sol {self.solidity:.2f}  holes {self.holes}")


def segment(frame, background):
    """Binary mask of whatever is on the stage that was not there before."""
    import cv2
    import numpy as np

    diff = cv2.absdiff(frame, background)
    if diff.ndim == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def outline(mask):
    """Largest outer contour plus its hole count, or None if the stage is empty."""
    import cv2

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    outer = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    if not outer:
        return None
    i = max(outer, key=lambda j: cv2.contourArea(contours[j]))
    area = cv2.contourArea(contours[i])
    if area < MIN_AREA_PX:
        return None
    holes = sum(
        1 for j in range(len(contours))
        if hierarchy[0][j][3] == i and cv2.contourArea(contours[j]) >= area * MIN_HOLE_FRACTION
    )
    return contours[i], holes


def measure(mask, px_per_mm: float) -> Shape | None:
    """Measure the piece in the mask. None if there is nothing to measure."""
    import cv2

    found = outline(mask)
    if found is None:
        return None
    contour, holes = found

    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    length = max(w, h) / px_per_mm
    width = min(w, h) / px_per_mm
    area = cv2.contourArea(contour) / px_per_mm**2
    perimeter = cv2.arcLength(contour, True) / px_per_mm
    hull = cv2.contourArea(cv2.convexHull(contour)) / px_per_mm**2

    return Shape(
        length_mm=length,
        width_mm=width,
        area_mm2=area,
        extent=area / (length * width) if length and width else 0.0,
        circularity=4 * math.pi * area / perimeter**2 if perimeter else 0.0,
        solidity=area / hull if hull else 0.0,
        holes=holes,
    )
