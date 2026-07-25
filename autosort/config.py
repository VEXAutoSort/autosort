"""Load and validate config.yaml into typed objects.

One file (config.yaml) drives the whole system. This turns it into dataclasses
and checks the couple of things that would otherwise fail deep in a run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class CameraCfg:
    index: int
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class RunCfg:
    mode: str = "continuous"          # continuous | step
    dry_run: bool = True
    max_consecutive_fails: int = 8


@dataclass
class ArmCfg:
    port: str
    id: str
    device: str
    policy: str
    pick_timeout_s: float
    gripper_empty_pos: float
    poses: dict[str, dict[str, float]]


@dataclass
class PerceptionCfg:
    pile_roi: list[float]
    gripper_roi: list[float]
    min_piece_area: int
    empty_frames: int


@dataclass
class ClassifierCfg:
    model: str
    labels: list[str]
    min_confidence: float
    settle_s: float


@dataclass
class RouterCfg:
    port: str
    baud: int
    drop_dwell_s: float
    bins: dict[str, float]


@dataclass
class Config:
    run: RunCfg
    arm: ArmCfg
    cameras: dict[str, CameraCfg]
    perception: PerceptionCfg
    classifier: ClassifierCfg
    router: RouterCfg

    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        import yaml

        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw: dict[str, Any] = yaml.safe_load(p.read_text())

        cfg = Config(
            run=RunCfg(**raw.get("run", {})),
            arm=ArmCfg(**raw["arm"]),
            cameras={name: CameraCfg(**c) for name, c in raw["cameras"].items()},
            perception=PerceptionCfg(**raw["perception"]),
            classifier=ClassifierCfg(**raw["classifier"]),
            router=RouterCfg(**raw["router"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.run.mode not in ("continuous", "step"):
            raise ValueError("run.mode must be 'continuous' or 'step'")
        # every class label must have a bin, or a piece could be unroutable mid-run
        missing = [lbl for lbl in self.classifier.labels if lbl not in self.router.bins]
        if missing:
            raise ValueError(f"These labels have no router.bins angle: {missing}")
        for name in ("top", "wrist", "box"):
            if name not in self.cameras:
                raise ValueError(f"cameras.{name} is required")
