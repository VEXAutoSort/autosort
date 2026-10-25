"""The sorting cycle as an explicit state machine.

    INIT -> IMAGE -> (DOSE -> SHAKE -> SETTLE -> IMAGE)* -> PICK -> CLASSIFY -> AIM -> DUMP -> LOG -> IMAGE ...

Every transition is logged with a timestamp and kept in `trace`. The machine
knows nothing about hardware: it drives a `Rig` (hw.py), which enforces the
interlocks. The one rule that matters most: a new PICK starts only after the
previous part's DUMP has finished — the chain PICK -> CLASSIFY -> AIM -> DUMP
-> LOG has no branch that leads back to PICK early.

INIT goes to IMAGE rather than straight to DOSE because the bucket interlock
("never tip unless the table count is 0") needs a count first; on an empty
table IMAGE doses immediately, so the operator sees the same thing.

Stopping: Ctrl-C, or an `ESTOP` file appearing in the run directory, torques
everything off before exit. Any exception does the same and exits non-zero.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import REJECT, Config
from .hw import Rig

log = logging.getLogger("orchestrator.cycle")


class State(Enum):
    INIT = "INIT"
    DOSE = "DOSE"
    SHAKE = "SHAKE"
    SETTLE = "SETTLE"
    IMAGE = "IMAGE"
    PICK = "PICK"
    PICK_RETRY = "PICK_RETRY"
    CLASSIFY = "CLASSIFY"
    AIM = "AIM"
    DUMP = "DUMP"
    LOG = "LOG"
    EMPTY = "EMPTY"
    DONE = "DONE"
    ESTOP = "ESTOP"


class EStop(Exception):
    """Operator stop (Ctrl-C or ESTOP file). Not an error."""


EXIT_OK, EXIT_ERROR, EXIT_ESTOP = 0, 1, 130
CSV_FIELDS = ["time", "class", "confidence", "bin", "pick_attempts", "cycle_time_s"]


@dataclass
class Stats:
    parts_sorted: int = 0
    doses: int = 0
    pick_failures: int = 0
    rejects: int = 0
    final_state: State | None = None
    per_class: dict[str, int] = field(default_factory=dict)


class Orchestrator:
    def __init__(self, cfg: Config, rig: Rig, run_dir: Path, max_parts: int | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.rig = rig
        self.run_dir = Path(run_dir)
        self.max_parts = max_parts
        self.sleep = sleep
        self.clock = clock
        self.trace: list[tuple[float, State]] = []
        self.stats = Stats()
        self.state = State.INIT
        self._csv = None

    # --- public ----------------------------------------------------------
    def run(self) -> int:
        """Run to completion. Returns a process exit code; the rig is always shut down."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        code = EXIT_OK
        try:
            self._open_csv()
            while self.state not in (State.DONE,):
                self._check_estop()
                self._step()
        except (KeyboardInterrupt, EStop) as e:
            self._enter(State.ESTOP)
            log.warning("STOP (%s) in %s — all actuators off", e.__class__.__name__, self._prev_state())
            self.rig.estop()
            code = EXIT_ESTOP
        except Exception:
            self._enter(State.ESTOP)
            log.exception("FAULT in %s — all actuators off", self._prev_state())
            self.rig.estop()
            code = EXIT_ERROR
        finally:
            self.stats.final_state = self.state
            try:
                self.rig.close()
            except Exception as e:  # noqa: BLE001
                log.error("close failed: %s", e)
            if self._csv:
                self._csv[0].close()
            log.info("summary: %d sorted, %d doses, %d pick failures, %d rejects, per class %s",
                     self.stats.parts_sorted, self.stats.doses, self.stats.pick_failures,
                     self.stats.rejects, self.stats.per_class)
        return code

    # --- the machine -----------------------------------------------------
    def _step(self) -> None:
        cfg, rig, s = self.cfg, self.rig, self.state

        if s is State.INIT:
            self._doses_this_fill = 0
            self._pick_attempts = 0
            rig.init()
            self._enter(State.IMAGE)

        elif s is State.DOSE:
            rig.dose()
            self._doses_this_fill += 1
            self.stats.doses += 1
            self._enter(State.SHAKE)

        elif s is State.SHAKE:
            rig.shake(cfg.shaker.burst_s)
            self._enter(State.SETTLE)

        elif s is State.SETTLE:
            self.sleep(cfg.timing.settle_s)
            self._enter(State.IMAGE)

        elif s is State.IMAGE:
            n = rig.count_table()
            log.info("table count = %d", n)
            if n == 0:
                if self._doses_this_fill < cfg.timing.max_doses_per_fill:
                    self._enter(State.DOSE)
                else:
                    self._enter(State.EMPTY)
            else:
                self._part_t0 = self.clock()
                self._enter(State.PICK)

        elif s is State.PICK:
            self._pick_attempts += 1
            ok = rig.pick(cfg.timing.pick_timeout_s)
            if ok:
                self._doses_this_fill = 0  # the table demonstrably had parts
                self._enter(State.CLASSIFY)
            else:
                self.stats.pick_failures += 1
                self._enter(State.PICK_RETRY)

        elif s is State.PICK_RETRY:
            if self._pick_attempts <= cfg.timing.pick_retries:
                log.info("pick retry %d/%d", self._pick_attempts, cfg.timing.pick_retries)
                self._enter(State.PICK)
            else:
                log.info("pick failed %d times — re-spreading", self._pick_attempts)
                self._pick_attempts = 0
                self._enter(State.SHAKE)

        elif s is State.CLASSIFY:
            label, conf = rig.classify()
            if conf < cfg.classifier.threshold or label not in cfg.classifier.class_to_bin:
                log.info("classified %s @ %.3f < %.2f -> %s", label, conf, cfg.classifier.threshold, REJECT)
                label = REJECT
            else:
                log.info("classified %s @ %.3f", label, conf)
            self._label, self._conf = label, conf
            self._bin = cfg.bin_for(label)
            self._enter(State.AIM)

        elif s is State.AIM:
            rig.aim(self._bin)
            self._enter(State.DUMP)

        elif s is State.DUMP:
            rig.dump()
            self._enter(State.LOG)

        elif s is State.LOG:
            cycle_time = self.clock() - self._part_t0
            self._write_row(self._label, self._conf, self._bin, self._pick_attempts, cycle_time)
            self.stats.parts_sorted += 1
            self.stats.per_class[self._label] = self.stats.per_class.get(self._label, 0) + 1
            if self._label == REJECT:
                self.stats.rejects += 1
            self._pick_attempts = 0
            if self.max_parts is not None and self.stats.parts_sorted >= self.max_parts:
                log.info("max parts (%d) reached", self.max_parts)
                self._enter(State.DONE)
            else:
                self._enter(State.IMAGE)

        elif s is State.EMPTY:
            log.info("bucket empty: %d doses in a row produced nothing. %d parts sorted.",
                     self._doses_this_fill, self.stats.parts_sorted)
            self._enter(State.DONE)

        else:
            raise RuntimeError(f"no handler for state {s}")

    # --- bookkeeping -------------------------------------------------------
    def _enter(self, new: State) -> None:
        now = self.clock()
        self.trace.append((now, new))
        log.info("%-10s -> %s", self.state.value, new.value)
        self.state = new

    def _prev_state(self) -> str:
        return self.trace[-2][1].value if len(self.trace) >= 2 else self.state.value

    def _check_estop(self) -> None:
        if (self.run_dir / "ESTOP").exists():
            raise EStop("ESTOP file present")

    def _open_csv(self) -> None:
        path = self.run_dir / "log.csv"
        new = not path.exists()
        f = path.open("a", newline="")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        self._csv = (f, w)

    def _write_row(self, label: str, conf: float, bin_index: int, attempts: int, cycle_time: float) -> None:
        if not self._csv:
            return
        self._csv[1].writerow({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "class": label, "confidence": f"{conf:.3f}", "bin": bin_index,
            "pick_attempts": attempts, "cycle_time_s": f"{cycle_time:.2f}",
        })
        self._csv[0].flush()
