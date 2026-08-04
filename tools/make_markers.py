"""Generate a printable ArUco marker sheet for camera auto-recalibration.

Run:  python tools/make_markers.py
Writes /tmp/aruco_markers.png (US Letter, 300 dpi). Print at 100% scale -
"fit to page" will resize the markers and is fine here, because the system
learns the markers' pixel positions rather than assuming a physical size.

Markers: DICT_4X4_50, IDs 0-3, 40 mm squares with a white quiet zone.
4x4 is deliberately low-resolution: fewer bits means far more reliable
detection at distance and under poor lighting than denser dictionaries.
"""
from __future__ import annotations

import cv2
import numpy as np

DPI = 300
MM = DPI / 25.4
MARKER_MM = 40
QUIET_MM = 8          # white border; ArUco needs one to detect the marker edge
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)
IDS = [0, 1, 2, 3]
OUT = "/tmp/aruco_markers.png"


def main() -> None:
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m_px = int(MARKER_MM * MM)
    q_px = int(QUIET_MM * MM)
    tile = m_px + 2 * q_px

    page = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    cols, rows = 2, 2
    gap = int(20 * MM)
    total_w = cols * tile + (cols - 1) * gap
    total_h = rows * tile + (rows - 1) * gap
    x0 = (PAGE_W - total_w) // 2
    y0 = (PAGE_H - total_h) // 2 + int(15 * MM)

    for i, mid in enumerate(IDS):
        r, c = divmod(i, cols)
        img = cv2.aruco.generateImageMarker(dic, mid, m_px)
        tx = x0 + c * (tile + gap)
        ty = y0 + r * (tile + gap)
        page[ty:ty + tile, tx:tx + tile] = 255
        page[ty + q_px:ty + q_px + m_px, tx + q_px:tx + q_px + m_px] = img
        # cut guide + label
        cv2.rectangle(page, (tx, ty), (tx + tile, ty + tile), 160, 2)
        cv2.putText(page, f"ID {mid}", (tx + 6, ty + tile + int(9 * MM)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)

    head = [
        "AutoSort camera reference markers - DICT_4X4_50, 40 mm",
        "Cut along the grey lines. Keep the WHITE BORDER - it is required for detection.",
        "Tape all four flat and permanently around the pick zone, fully in the camera view,",
        "OUTSIDE the orange detection box so they are never mistaken for parts.",
    ]
    for j, line in enumerate(head):
        cv2.putText(page, line, (int(12 * MM), int(14 * MM) + j * int(7 * MM)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, 0, 2)

    cv2.imwrite(OUT, page)
    print(f"wrote {OUT}  ({MARKER_MM} mm markers, IDs {IDS}, {DPI} dpi US Letter)")


if __name__ == "__main__":
    main()
