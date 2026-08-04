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

    def verify_open(self, role: str):
        """Open the camera and assert it looks like the RIGHT one.

        macOS renumbers USB cameras on replug, so a saved index can silently point
        at the wrong camera - which corrupts detection and any taught calibration.
        Frame size is a cheap fingerprint: the overhead and wrist cameras have
        different native resolutions.
        """
        import cv2

        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"camera '{role}' (index {self.index}) would not open - "
                               "is LeLab or another tool holding it?")
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"camera '{role}' (index {self.index}) opened but returned no frame")
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            cap.release()
            raise RuntimeError(
                f"camera '{role}' (index {self.index}) returned {w}x{h}, expected "
                f"{self.width}x{self.height}. The USB camera order almost certainly changed - "
                f"re-run tools/select_cameras.py before doing anything else."
            )
        return cap


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
    solver: str = "analytic"                  # analytic (homography+IK) | interpolate (taught-pose blending)
    urdf_path: str | None = None              # SO-101 URDF for the analytic solver
    hover_lift_m: float = 0.08                # how far straight up the hover pose sits (analytic only)
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
    multi_piece_area: int = 3000  # total dark px^2 in the gripper ROI that means 2+ pieces (one gear is well under this)


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
