"""The Arducam looking down at the piece on the white stage."""
from __future__ import annotations

import logging

from ..config import CameraCfg

log = logging.getLogger("autosort.classification")

FLUSH_FRAMES = 5   # the driver buffers a few frames; read past the stale ones


class BoxCamera:
    """Capture device, a reference shot of the empty stage, and how to frame it.

    Averaging several frames cuts sensor noise by roughly the square root of the
    count, which buys back edge precision when the image is grainy. The region of
    interest trims the view to the stage so nothing on the table around it can ever
    become the largest blob.
    """

    def __init__(self, cfg: CameraCfg, roi: list[float] | None = None, average: int = 1):
        self.cfg = cfg
        self.roi = roi
        self.average = max(1, average)
        self.background = None
        self._cap = None

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.cfg.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"box camera index {self.cfg.index} would not open")
        self.learn_background()

    def learn_background(self) -> None:
        """Snapshot the empty stage. The stage must actually be empty right now."""
        self.background = self.grab()
        log.info("stage reference captured (it must be empty at startup)")

    def grab(self):
        import numpy as np

        frames = []
        for _ in range(FLUSH_FRAMES + self.average):
            ok, frame = self._cap.read()
            if ok:
                frames.append(frame)
        if not frames:
            raise RuntimeError("box camera read failed")

        keep = frames[-self.average:]
        merged = keep[0] if len(keep) == 1 else np.mean(keep, axis=0).astype(np.uint8)
        return self._crop(merged)

    def _crop(self, frame):
        if not self.roi:
            return frame
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.roi
        return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
