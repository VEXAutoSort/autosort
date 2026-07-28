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
    poses: dict[str, dict[str, float]]
    pick_mode: str = "classical"              # classical | act
    taught_file: str | None = None            # taught.json path (None = repo root)
    gripper_holding_margin: float = 4.0       # deg short of taught-closed that means "holding a piece"
    pick_joint_offsets: dict[str, float] | None = None  # global reach correction, deg added to every computed grasp
    act_fps: float = 30.0                     # control rate for the ACT pick loop
    policy_camera_names: dict[str, str] | None = None  # robot cam name -> dataset camera key


@dataclass
class PerceptionCfg:
    pile_roi: list[float]
    gripper_roi: list[float]
    min_piece_area: int
    empty_frames: int
    max_piece_area: int = 10000   # px^2 - ignore blobs bigger than a piece (the drop box, a hand, shadows)
    contrast_margin: int = 45     # how much darker than the surface a pixel must be to count as piece (shadows are ~15-25)


@dataclass
class ClassifierCfg:
    model: str
    labels: list[str]
    min_confidence: float
    settle_s: float
    enabled: bool = True    # false = hardware not built yet; label everything 'unknown'


@dataclass
class RouterCfg:
    port: str
    baud: int
    drop_dwell_s: float
    bins: dict[str, float]
    enabled: bool = True    # false = hardware not built yet; log the routing decision only


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

        # tools/select_cameras.py writes camera indices here so nobody hand-edits
        # (macOS shuffles USB camera numbering between reboots/replugs)
        override = p.parent / "cameras_override.json"
        if override.exists():
            import json
            for name, idx in json.loads(override.read_text()).items():
                if name in raw.get("cameras", {}):
                    raw["cameras"][name]["index"] = idx

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
        if self.arm.pick_mode not in ("classical", "act"):
            raise ValueError("arm.pick_mode must be 'classical' or 'act'")
        if self.router.enabled:
            # every class label must have a bin, or a piece could be unroutable mid-run
            missing = [lbl for lbl in self.classifier.labels if lbl not in self.router.bins]
            if missing:
                raise ValueError(f"These labels have no router.bins angle: {missing}")
        required_cams = ("top", "wrist", "box") if self.classifier.enabled else ("top", "wrist")
        for name in required_cams:
            if name not in self.cameras:
                raise ValueError(f"cameras.{name} is required")
