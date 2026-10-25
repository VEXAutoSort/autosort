"""Rig: the interlocked hardware façade the cycle drives.

`Rig` is the abstract base. Its PUBLIC methods are what cycle.py calls, and
they are where every global interlock is enforced — in code, for the real
machine and the simulator alike:

    * the shaker is never ON while the arm is away from HOME, or during IMAGE / CLASSIFY
    * the bucket (ID 7) never moves unless the last table count was 0 and the arm is at HOME
    * the distributor (ID 8) never moves while the arm is away from HOME
    * the arm never starts a pick while the shaker is ON
    * estop() torques off every servo and kills the shaker, swallowing errors

Subclasses implement the `_do_*` primitives: RealRig (below) with machine/,
the cameras and the LeRobot arm; SimRig (sim.py) with stubs.
"""
from __future__ import annotations

import logging
import time

from .config import Config

log = logging.getLogger("orchestrator.hw")


class InterlockError(RuntimeError):
    """A public Rig call that would violate a safety interlock."""


class Rig:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.arm_at_home = False
        self.shaker_on = False
        self.table_count: int | None = None  # None = unknown (table changed since last IMAGE)
        self.current_bin: int | None = None

    # --- public API (interlocked) --------------------------------------
    def init(self) -> None:
        """Lock cameras, open buses, home every actuator, shaker OFF, arm HOME."""
        self._do_init()
        self._do_shaker(False)
        self.shaker_on = False
        self.arm_at_home = self._do_arm_is_home()
        if not self.arm_at_home:
            raise InterlockError("arm did not reach HOME during INIT")
        self.current_bin = 0

    def dose(self) -> None:
        self._require_arm_home("bucket")
        if self.table_count != 0:
            raise InterlockError(f"bucket may only tip onto an empty table (last count: {self.table_count})")
        self._do_bucket_pour()
        self.table_count = None

    def shake(self, seconds: float) -> None:
        self._require_arm_home("shaker")
        self._do_shaker(True)
        self.shaker_on = True
        try:
            self._sleep(seconds)
        finally:
            self._do_shaker(False)
            self.shaker_on = False
        self.table_count = None

    def count_table(self) -> int:
        if self.shaker_on:
            raise InterlockError("IMAGE while the shaker is on")
        n = self._do_count_table()
        self.table_count = n
        return n

    def pick(self, timeout_s: float) -> bool:
        if self.shaker_on:
            raise InterlockError("pick while the shaker is on")
        self._require_arm_home("pick start")
        self.arm_at_home = False
        self.table_count = None
        try:
            ok = self._do_arm_pick(timeout_s)
        finally:
            self.arm_at_home = self._do_arm_is_home()
        if not self.arm_at_home:
            raise InterlockError("arm is not at HOME after the pick — refusing to continue")
        return ok

    def classify(self) -> tuple[str, float]:
        if self.shaker_on:
            raise InterlockError("CLASSIFY while the shaker is on")
        self._require_arm_home("classify")
        return self._do_classify()

    def aim(self, bin_index: int) -> None:
        self._require_arm_home("distributor")
        self._do_aim(bin_index)
        self.current_bin = bin_index

    def dump(self) -> None:
        self._require_arm_home("cup")
        self._do_cup_dump()

    def estop(self) -> None:
        """Everything off, errors swallowed — safe to call from any state, twice."""
        for name, fn in (("shaker", lambda: self._do_shaker(False)), ("actuators", self._do_all_off)):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.error("ESTOP: %s off failed: %s", name, e)
        self.shaker_on = False

    def close(self) -> None:
        self._do_close()

    # --- helpers ---------------------------------------------------------
    def _require_arm_home(self, what: str) -> None:
        if not self.arm_at_home:
            raise InterlockError(f"{what}: arm is not at HOME")

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    # --- primitives for subclasses --------------------------------------
    def _do_init(self) -> None: raise NotImplementedError
    def _do_bucket_pour(self) -> None: raise NotImplementedError
    def _do_shaker(self, on: bool) -> None: raise NotImplementedError
    def _do_count_table(self) -> int: raise NotImplementedError
    def _do_arm_pick(self, timeout_s: float) -> bool: raise NotImplementedError
    def _do_arm_is_home(self) -> bool: raise NotImplementedError
    def _do_classify(self) -> tuple[str, float]: raise NotImplementedError
    def _do_aim(self, bin_index: int) -> None: raise NotImplementedError
    def _do_cup_dump(self) -> None: raise NotImplementedError
    def _do_all_off(self) -> None: raise NotImplementedError
    def _do_close(self) -> None: raise NotImplementedError


# ===========================================================================

class RealRig(Rig):
    """The actual machine. One process owns /dev/machine and /dev/arm — this one.

    skip_arm     no arm at all: pick() succeeds instantly (feeder/distributor bring-up)
    manual_pick  a human moves the part to the cup and presses Enter
    """

    def __init__(self, cfg: Config, skip_arm: bool = False, manual_pick: bool = False):
        super().__init__(cfg)
        self.skip_arm = skip_arm
        self.manual_pick = manual_pick
        self.bus = None
        self.bucket = self.distributor = self.cup = None
        self.shaker = None
        self.arm = None
        self._cams: dict[str, object] = {}
        self._classifier = None

    def _do_init(self) -> None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "machine"))
        from machine import Bucket, Bus, Cup, Distributor, Shaker  # noqa: E402

        from . import vision

        cfg = self.cfg
        # 1. cameras — lock the overhead before anything opens it
        vision.lock_camera(cfg.cameras.overhead)
        arm_owns = set(cfg.arm.cameras.values()) if not self.skip_arm else set()
        for name in ("overhead", "wrist", "cup"):
            index = getattr(cfg.cameras, name)
            if index is None or name in arm_owns:
                continue
            if name != "overhead" and name != cfg.cameras.classify_camera:
                continue
            cam = vision.Camera(index, *cfg.cameras.size(name))
            cam.open()
            self._cams[name] = cam
        self._arm_owns = arm_owns

        # 2. machine bus
        self.bus = Bus(cfg.ports.machine, cfg.servo.baud)
        self.bus.open()
        self.bucket = Bucket(self.bus, cfg.servo.bucket_id, cfg.bucket.rest_pos, cfg.bucket.pour_pos,
                             cfg.bucket.dwell_s)
        self.distributor = Distributor(
            self.bus, cfg.servo.distributor_id, ratio=cfg.distributor.ratio, bins=cfg.distributor.bins,
            bin0_servo_pos=cfg.distributor.bin0_servo_pos, tolerance_counts=cfg.distributor.tolerance_counts,
            settle_s=cfg.distributor.settle_s, max_excursion_turns=cfg.distributor.max_excursion_turns)
        self.cup = Cup(self.bus, cfg.servo.cup_id, cfg.cup.hold_pos, cfg.cup.dump_pos, cfg.cup.dwell_s)
        for s in (self.bucket, self.distributor, self.cup):
            s.setup()
        self.distributor.home()
        self.bucket.rest()
        self.cup.hold()

        # 3. shaker
        self.shaker = Shaker(cfg.shaker.ip, cfg.shaker.user, cfg.shaker.password)

        # 4. arm (also the owner of the cameras in arm.cameras)
        if not self.skip_arm:
            from .arm import make_arm

            self.arm = make_arm(cfg.ports.arm, cfg.arm, cfg.cameras)
            self.arm.connect()
            self.arm.home()
        # some UVC drivers re-enable auto exposure when a device is (re)opened — lock again now
        vision.lock_camera(cfg.cameras.overhead)

        # 5. classifier
        if cfg.classifier.backend == "onnx":
            self._classifier = vision.OnnxClassifier(cfg.classifier.model, cfg.classifier.labels)
            self._classifier.open()
        else:
            self._classifier = vision.CatalogClassifier(cfg.classifier.catalog, cfg.classifier.labels)
            self._classifier.open()
            self._classifier.set_background(self._cup_frame())  # cup must be empty now

    def _frame(self, name: str):
        """A BGR frame from camera `name`, whoever owns the device.

        LeRobot's OpenCV cameras hand back RGB; our own captures are BGR. Everything
        downstream (count_parts, the classifiers) is written for BGR, so arm-owned
        frames are converted here, once."""
        if name in self._arm_owns:
            import cv2

            lerobot_name = next(k for k, v in self.cfg.arm.cameras.items() if v == name)
            return cv2.cvtColor(self.arm.frame(lerobot_name), cv2.COLOR_RGB2BGR)
        return self._cams[name].grab()

    def _cup_frame(self):
        from .vision import crop_roi

        return crop_roi(self._frame(self.cfg.cameras.classify_camera), self.cfg.cameras.cup_roi)

    def _do_bucket_pour(self) -> None:
        self.bucket.pour()

    def _do_shaker(self, on: bool) -> None:
        if self.shaker is not None:
            self.shaker.on() if on else self.shaker.off()

    def _do_count_table(self) -> int:
        from .vision import count_parts

        return count_parts(self._frame("overhead"), self.cfg.cameras.table_roi,
                           self.cfg.vision.min_part_area, self.cfg.vision.dark_parts)

    def _do_arm_pick(self, timeout_s: float) -> bool:
        if self.skip_arm:
            return True
        if self.manual_pick:
            input("  MANUAL PICK: move one part from the table into the cup, then press Enter... ")
            return True
        return self.arm.pick(timeout_s)

    def _do_arm_is_home(self) -> bool:
        return True if (self.skip_arm or self.arm is None) else self.arm.is_home()

    def _do_classify(self) -> tuple[str, float]:
        return self._classifier.classify(self._cup_frame())

    def _do_aim(self, bin_index: int) -> None:
        self.distributor.aim(bin_index)

    def _do_cup_dump(self) -> None:
        self.cup.dump()

    def _do_all_off(self) -> None:
        for s in (self.bucket, self.distributor, self.cup):
            if s is not None:
                try:
                    s.torque_off()
                except Exception as e:  # noqa: BLE001
                    log.error("torque off id %s failed: %s", getattr(s, "sid", "?"), e)
        if self.arm is not None:
            self.arm.torque_off()

    def _do_close(self) -> None:
        for cam in self._cams.values():
            cam.close()
        if self._classifier is not None:
            self._classifier.close()
        if self.arm is not None:
            self.arm.disconnect()
        if self.bus is not None:
            self.bus.close()
