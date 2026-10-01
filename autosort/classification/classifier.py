"""Identify the piece on the stage: measure it, then match the VEX part table.

Two ways to come back 'unknown', and they mean different things:
  distance  > max_distance    nothing in the table is this shape (odd part, odd pose)
  margin    < min_confidence  two different labels fit about equally well
Both belong in the reject bin — a slow bin beats a wrong bin.
"""
from __future__ import annotations

import logging
import random
import time

from ..config import CameraCfg, ClassifierCfg
from .camera import BoxCamera
from .catalog import Catalog
from .vision import Shape, measure, segment

log = logging.getLogger("autosort.classification")

UNKNOWN = "unknown"


class Classifier:
    def __init__(self, cfg: ClassifierCfg, box_cam: CameraCfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.camera = BoxCamera(box_cam)
        self.catalog: Catalog | None = None

    def connect(self) -> None:
        if self.dry_run:
            return
        self.catalog = Catalog.load(self.cfg.catalog)
        missing = sorted(self.catalog.labels - set(self.cfg.labels))
        if missing:
            raise ValueError(f"catalog uses labels absent from classifier.labels: {missing}")
        log.info("%d catalog entries, %.2f px/mm", len(self.catalog.entries), self.cfg.px_per_mm)
        self.camera.open()

    def classify(self) -> tuple[str, float]:
        """Wait for the piece to settle, image it, and name it."""
        time.sleep(self.cfg.settle_s)
        if self.dry_run:
            return random.choice(self.cfg.labels), 0.99
        return self.identify(self.camera.grab())

    def identify(self, frame) -> tuple[str, float]:
        shape = measure(segment(frame, self.camera.background), self.cfg.px_per_mm)
        if shape is None:
            log.info("stage looks empty")
            return UNKNOWN, 0.0
        return self.name(shape)

    def name(self, shape: Shape) -> tuple[str, float]:
        match = self.catalog.match(shape)
        log.debug("%s -> %s (d=%.2f)", shape, match.entry.name, match.distance)

        if match.distance > self.cfg.max_distance:
            log.info("no match: %s (nearest %s, d=%.2f)", shape, match.entry.name, match.distance)
            return UNKNOWN, 0.0
        if match.margin < self.cfg.min_confidence:
            log.info("ambiguous: %s vs another label (margin %.2f)", match.entry.name, match.margin)
            return UNKNOWN, match.margin
        log.info("%s -> %s (d=%.2f, margin=%.2f)",
                 match.entry.name, match.entry.label, match.distance, match.margin)
        return match.entry.label, match.margin

    def disconnect(self) -> None:
        self.camera.close()
