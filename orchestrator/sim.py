"""SimRig: every actuator and camera stubbed, with realistic delays.

    python -m orchestrator.run --sim                 # feels like the machine (~10 s / part)
    SimRig(cfg, time_scale=0)                        # instant, for the tests

The sim keeps a tiny physical model: parts in the bucket, parts on the table,
whether the cup holds a part, which bin the ring faces. Every primitive call is
appended to `calls` so tests can assert ordering (e.g. DUMP before the next PICK).
"""
from __future__ import annotations

import logging
import random
import time

from .config import Config
from .hw import Rig

log = logging.getLogger("orchestrator.sim")


class SimRig(Rig):
    def __init__(self, cfg: Config, bucket_parts: int = 12, time_scale: float = 1.0,
                 seed: int = 0, fail_picks: frozenset[int] | set[int] = frozenset(),
                 parts_per_dose: tuple[int, int] = (2, 5), empty_doses: int = 0):
        super().__init__(cfg)
        self.time_scale = time_scale
        self.rng = random.Random(seed)
        self.fail_picks = set(fail_picks)       # 1-based indices of pick attempts that fail
        self.parts_per_dose = parts_per_dose
        self.empty_doses = empty_doses          # first N doses pour nothing (sticky bucket)
        self.bucket_parts = bucket_parts
        self.table_parts = 0
        self.cup_has_part = False
        self.ring_bin = 0
        self.arm_home = False
        self.shaker_state = False
        self.pick_attempts = 0
        self.doses = 0
        self.dropped: list[tuple[str, int]] = []  # (label, bin) of every dumped part
        self.calls: list[str] = []
        self._last_label = "REJECT"

    def _wait(self, seconds: float) -> None:
        if self.time_scale > 0:
            time.sleep(seconds * self.time_scale)

    def _sleep(self, seconds: float) -> None:  # the shaker burst, scaled like everything else
        self._wait(seconds)

    # --- primitives ------------------------------------------------------
    def _do_init(self) -> None:
        self.calls.append("init")
        self._wait(1.0)
        self.arm_home = True

    def _do_bucket_pour(self) -> None:
        self.calls.append("dose")
        self.doses += 1
        self._wait(self.cfg.bucket.dwell_s + 0.8)
        if self.doses <= self.empty_doses:
            return
        n = min(self.bucket_parts, self.rng.randint(*self.parts_per_dose))
        self.bucket_parts -= n
        self.table_parts += n
        log.info("[sim] poured %d parts (%d left in bucket)", n, self.bucket_parts)

    def _do_shaker(self, on: bool) -> None:
        self.calls.append("shaker_on" if on else "shaker_off")
        self._wait(0.2)  # plug latency
        self.shaker_state = on

    def _do_count_table(self) -> int:
        self.calls.append("count")
        self._wait(0.15)
        return self.table_parts

    def _do_arm_pick(self, timeout_s: float) -> bool:
        self.calls.append("pick")
        self.pick_attempts += 1
        self.arm_home = False
        if self.pick_attempts in self.fail_picks:
            self._wait(min(timeout_s, 3.0))
            self.arm_home = True
            return False
        self._wait(6.0)
        if self.table_parts > 0:
            self.table_parts -= 1
            self.cup_has_part = True
        self.arm_home = True
        return True

    def _do_arm_is_home(self) -> bool:
        return self.arm_home

    def _do_classify(self) -> tuple[str, float]:
        self.calls.append("classify")
        self._wait(0.1)
        if not self.cup_has_part:
            return "REJECT", 0.0
        self._last_label = self.rng.choice(self.cfg.classifier.labels)
        return self._last_label, round(self.rng.uniform(0.7, 1.0), 3)

    def _do_aim(self, bin_index: int) -> None:
        self.calls.append(f"aim:{bin_index}")
        steps = min(abs(bin_index - self.ring_bin), self.cfg.distributor.bins - abs(bin_index - self.ring_bin))
        self._wait(0.4 + 0.35 * steps + self.cfg.distributor.settle_s)
        self.ring_bin = bin_index

    def _do_cup_dump(self) -> None:
        self.calls.append("dump")
        self._wait(self.cfg.cup.dwell_s + 0.6)
        if self.cup_has_part:
            self.dropped.append((self._last_label, self.ring_bin))
            self.cup_has_part = False

    def _do_all_off(self) -> None:
        self.calls.append("all_off")
        self.shaker_state = False

    def _do_close(self) -> None:
        self.calls.append("close")
