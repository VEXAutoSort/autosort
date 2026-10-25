"""Load orchestrator/config.yaml into typed dataclasses and check the VERIFY values."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"
REJECT = "REJECT"


@dataclass
class Ports:
    arm: str = "/dev/arm"
    machine: str = "/dev/machine"


@dataclass
class Cameras:
    overhead: int = 0
    wrist: int | None = None
    cup: int | None = None
    width: int = 960
    height: int = 600
    table_roi: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.95, 0.95])
    classify_camera: str = "overhead"
    cup_roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class Servo:
    baud: int = 1_000_000
    bucket_id: int = 7
    distributor_id: int = 8
    cup_id: int = 9


@dataclass
class Bucket:
    rest_pos: int | None = None
    pour_pos: int | None = None
    dwell_s: float = 1.5


@dataclass
class Distributor:
    ratio: float = 3.26
    bins: int = 7
    bin0_servo_pos: int | None = None
    tolerance_counts: int = 20
    settle_s: float = 0.3
    max_excursion_turns: float = 1.5


@dataclass
class Cup:
    hold_pos: int | None = None
    dump_pos: int | None = None
    dwell_s: float = 0.8


@dataclass
class Shaker:
    ip: str = "192.168.1.38"
    user: str = ""
    password: str = ""
    burst_s: float = 1.0


@dataclass
class Timing:
    settle_s: float = 0.7
    pick_timeout_s: float = 25.0
    max_doses_per_fill: int = 3
    pick_retries: int = 2


@dataclass
class Vision:
    min_part_area: int = 150
    dark_parts: bool = True


@dataclass
class Classifier:
    backend: str = "onnx"
    model: str = "models/autosort_cls.onnx"
    labels: list[str] = field(default_factory=list)
    threshold: float = 0.85
    class_to_bin: dict[str, int] = field(default_factory=dict)
    catalog: dict[str, Any] = field(default_factory=dict)


@dataclass
class Arm:
    id: str = "autosort_follower"
    device: str = "cuda"
    policy: str = ""
    task: str = ""
    policy_cameras: dict[str, str] = field(default_factory=lambda: {"wrist": "wrist"})
    home_pose: dict[str, float] = field(default_factory=dict)
    home_tolerance_deg: float = 6.0
    min_policy_s: float = 4.0


@dataclass
class Config:
    ports: Ports = field(default_factory=Ports)
    cameras: Cameras = field(default_factory=Cameras)
    servo: Servo = field(default_factory=Servo)
    bucket: Bucket = field(default_factory=Bucket)
    distributor: Distributor = field(default_factory=Distributor)
    cup: Cup = field(default_factory=Cup)
    shaker: Shaker = field(default_factory=Shaker)
    timing: Timing = field(default_factory=Timing)
    vision: Vision = field(default_factory=Vision)
    classifier: Classifier = field(default_factory=Classifier)
    arm: Arm = field(default_factory=Arm)

    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        import yaml

        p = Path(path) if path else DEFAULT_PATH
        raw = _expand_env(yaml.safe_load(p.read_text()) or {})
        sections = {
            "ports": Ports, "cameras": Cameras, "servo": Servo, "bucket": Bucket,
            "distributor": Distributor, "cup": Cup, "shaker": Shaker, "timing": Timing,
            "vision": Vision, "classifier": Classifier, "arm": Arm,
        }
        cfg = Config(**{name: cls(**(raw.get(name) or {})) for name, cls in sections.items()})
        cfg.validate_static()
        return cfg

    # -- checks that hold in every mode --
    def validate_static(self) -> None:
        if self.distributor.bins < 2:
            raise ValueError("distributor.bins must be >= 2")
        c2b = self.classifier.class_to_bin
        if REJECT not in c2b:
            raise ValueError(f"classifier.class_to_bin needs a '{REJECT}' bin")
        for label in self.classifier.labels:
            if label not in c2b:
                raise ValueError(f"label '{label}' has no bin in classifier.class_to_bin")
        bad = {k: v for k, v in c2b.items() if not 0 <= v < self.distributor.bins}
        if bad:
            raise ValueError(f"class_to_bin entries outside 0..{self.distributor.bins - 1}: {bad}")
        if self.classifier.backend not in ("onnx", "catalog"):
            raise ValueError("classifier.backend must be 'onnx' or 'catalog'")
        if self.cameras.classify_camera not in ("overhead", "wrist", "cup"):
            raise ValueError("cameras.classify_camera must be overhead | wrist | cup")

    # -- the VERIFY list: only a real run needs these filled in --
    def missing_hardware_values(self, skip_arm: bool = False) -> list[str]:
        missing = []
        for name, value in (
            ("bucket.rest_pos", self.bucket.rest_pos),
            ("bucket.pour_pos", self.bucket.pour_pos),
            ("distributor.bin0_servo_pos", self.distributor.bin0_servo_pos),
            ("cup.hold_pos", self.cup.hold_pos),
            ("cup.dump_pos", self.cup.dump_pos),
        ):
            if value is None:
                missing.append(name)
        if not self.shaker.user or not self.shaker.password:
            missing.append("shaker.user/password (KASA_USER / KASA_PASS)")
        if self.cameras.classify_camera == "cup" and self.cameras.cup is None:
            missing.append("cameras.cup")
        if not skip_arm and not self.arm.home_pose:
            missing.append("arm.home_pose")
        return missing

    def bin_for(self, label: str) -> int:
        return self.classifier.class_to_bin.get(label, self.classifier.class_to_bin[REJECT])


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(node: Any) -> Any:
    """Replace ${VAR} inside string values with the environment (missing -> '')."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), node)
    return node
