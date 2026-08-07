"""Classify the piece sitting in the enclosure, from the Arducam ('box' camera).

Plug in your trained model at models/classifier.pt (a TorchScript module that maps
a 224x224 RGB image -> logits over `labels`). If the file is missing, everything is
labelled 'unknown' so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from .config import CameraCfg, ClassifierCfg

log = logging.getLogger("autosort.classifier")


class Classifier:
    def __init__(self, cfg: ClassifierCfg, box_cam: CameraCfg | None, dry_run: bool = False):
        self.cfg = cfg
        self.box_cam = box_cam
        self.dry_run = dry_run
        self._cap = None
        self._model = None

    def connect(self) -> None:
        if self.cfg.enabled and self.box_cam is None:
            raise ValueError("classifier.enabled is true but cameras.box is not configured")
        if self.dry_run or not self.cfg.enabled:
            if not self.cfg.enabled:
                log.info("classifier disabled (no enclosure hardware yet) — everything routes as 'unknown'")
            return
        import cv2

        self._cap = cv2.VideoCapture(self.box_cam.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.box_cam.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.box_cam.height)

        if Path(self.cfg.model).exists():
            import torch

            self._model = torch.jit.load(self.cfg.model)
            self._model.eval()
            log.info("classifier loaded: %s", self.cfg.model)
        else:
            log.warning("no classifier at %s — labelling everything 'unknown'", self.cfg.model)

    @staticmethod
    def classify_geometry(area: float, aspect: float, pieces: dict,
                          color: str = "unknown") -> str:
        """Label a piece from its TOP-VIEW footprint - no model, no enclosure.

        First profile (config order) whose area/aspect window contains the
        blob AND whose color requirement (if any) matches wins; put the most
        specific windows first. No match = 'unknown' (which also keeps every
        grasp value at its proven default).
        """
        for name, p in pieces.items():
            if (p.min_area <= area <= p.max_area
                    and p.min_aspect <= aspect <= p.max_aspect
                    and (p.color is None or p.color == color)):
                return name
        return "unknown"

    def classify(self) -> tuple[str, float]:
        """Grab a frame from the box cam and return (label, confidence)."""
        if not self.cfg.enabled:
            return "unknown", 0.0
        time.sleep(self.cfg.settle_s)  # let the piece settle after the drop
        if self.dry_run:
            return random.choice(self.cfg.labels), 0.99
        ok, frame = self._cap.read()
        if not ok or self._model is None:
            return "unknown", 0.0
        label, conf = self._infer(frame)
        return (label, conf) if conf >= self.cfg.min_confidence else ("unknown", conf)

    def _infer(self, frame) -> tuple[str, float]:
        import cv2
        import torch

        img = cv2.resize(frame, (224, 224))[:, :, ::-1]  # BGR -> RGB
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with torch.no_grad():
            probs = self._model(x).softmax(dim=1)[0]
        i = int(probs.argmax())
        return self.cfg.labels[i], float(probs[i])

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
