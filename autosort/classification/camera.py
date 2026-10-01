"""The Arducam looking down at the piece on the white stage."""
from __future__ import annotations

import logging

from ..config import CameraCfg

log = logging.getLogger("autosort.classification")

FLUSH_FRAMES = 5   # the driver buffers a few frames; read past the stale ones


class BoxCamera:
    """Holds the capture device and a reference shot of the empty stage."""

    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
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
        ok, frame = False, None
        for _ in range(FLUSH_FRAMES):
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("box camera read failed")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
