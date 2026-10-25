"""Machine servo bus + shaker — the hardware layer for everything that is not the arm.

Three Feetech ST3215 servos on the second Waveshare adapter (/dev/machine):

    ID 7  bucket tilt        position mode   REST <-> POUR
    ID 8  distributor swivel step mode       drives the ring gear through a ~3.26:1 pinion
    ID 9  cup tipper         position mode   HOLD <-> DUMP

plus the vibrating-table shaker behind a TP-Link Kasa HS103 smart plug
(software can only switch it ON/OFF; the inline knob sets intensity).

The distributor is the tricky one: one ring revolution is ~3.26 servo
revolutions, so the servo's 0-4095 encoder wraps ~3 times per ring turn.
We home once at startup against a known bin (the ring MUST physically sit at
bin 0 when the orchestrator starts) and from then on track multi-turn position
in software, moving the servo in STEP mode (mode 3: signed relative moves, so
the 0/4095 boundary is never a wall). All the multi-turn arithmetic lives in
pure functions at the top of this file so it can be unit-tested with no
hardware or scservo_sdk installed.

CLI (VERIFY values are read with `where` while moving a torque-off servo by hand):

    python machine/machine.py scan
    python machine/machine.py where 8            # live position read (Ctrl-C to stop)
    python machine/machine.py torque 8 off       # free a servo to move it by hand
    python machine/machine.py move 7 1024        # position-mode goal (bucket / cup)
    python machine/machine.py aim 3              # distributor -> bin 3 (homes on first call)
    python machine/machine.py shake 1.0          # one shaker burst
    python machine/machine.py rhythm 0.8 0.4 5   # on/off/n bursts

scservo_sdk (pip install feetech-servo-sdk) and python-kasa are imported
lazily so the math is importable anywhere.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("machine")

# --- STS/ST3215 register map (protocol 0) ---------------------------------
REG_ID = 5
REG_MODE = 33            # 0 = position servo, 1 = wheel, 2 = PWM, 3 = step (relative)
REG_TORQUE_ENABLE = 40
REG_GOAL_POSITION = 42   # 2 bytes
REG_GOAL_SPEED = 46      # 2 bytes
REG_LOCK = 55            # EEPROM lock
REG_PRESENT_POSITION = 56  # 2 bytes

RING = 4096              # encoder counts per servo turn, and ring counts per ring turn


# ===========================================================================
# Pure multi-turn math (unit-tested in tests/test_distributor.py)
# ===========================================================================

def wrap_delta(new_raw: int, old_raw: int) -> int:
    """Smallest signed step new-old on a 0..4095 circle (assumes |true step| < 2048)."""
    return (new_raw - old_raw + RING // 2) % RING - RING // 2


class MultiTurnTracker:
    """Software multi-turn position: feed it raw 0-4095 reads often enough that
    the servo never moves half a turn between reads, and `.abs` stays true."""

    def __init__(self, raw: int):
        self.raw = int(raw)
        self.abs = 0  # counts of travel since construction

    def update(self, raw: int) -> int:
        raw = int(raw)
        self.abs += wrap_delta(raw, self.raw)
        self.raw = raw
        return self.abs


def bin_ring_pos(bin_index: int, bins: int) -> float:
    """Ring position (counts on the ring's own 0..4095 circle) of a bin center."""
    return (bin_index % bins) * RING / bins


def ring_pos_to_bin(ring_pos: float, bins: int, tol: float) -> int | None:
    """Inverse of bin_ring_pos: which bin a ring position sits at (None = between bins)."""
    per = RING / bins
    n = round((ring_pos % RING) / per) % bins
    err = (ring_pos - n * per + RING / 2) % RING - RING / 2
    return n if abs(err) <= tol else None


def plan_ring_delta(ring_abs: float, target_bin: int, bins: int,
                    max_excursion: float = 1.5 * RING) -> float:
    """Signed ring travel (counts) from the current position to `target_bin`.

    `ring_abs` is total ring travel since homing at bin 0 (so ring position is
    ring_abs mod 4096). Picks the short way around unless that would push
    |ring_abs| past `max_excursion` — the servo-side wind-up bound that keeps
    the software multi-turn count from drifting off to infinity when the same
    direction keeps winning. Going the long way is always available on a ring.
    """
    current = ring_abs % RING
    target = bin_ring_pos(target_bin, bins)
    delta = (target - current + RING / 2) % RING - RING / 2
    if abs(ring_abs + delta) > max_excursion:
        delta -= RING if delta > 0 else -RING
    return delta


def encode_signed(value: int) -> int:
    """Feetech sign-magnitude: bit 15 is the direction bit (for step mode / speed)."""
    return (abs(value) | 0x8000) if value < 0 else value


# ===========================================================================
# Bus + servos
# ===========================================================================

class Bus:
    """One serial bus of Feetech STS-protocol servos (protocol 0)."""

    def __init__(self, port: str, baud: int = 1_000_000):
        self.port_name = port
        self.baud = baud
        self._port = None
        self._packet = None

    def open(self) -> None:
        import scservo_sdk as scs

        self._scs = scs
        self._port = scs.PortHandler(self.port_name)
        if not self._port.openPort():
            raise RuntimeError(f"could not open {self.port_name}")
        if not self._port.setBaudRate(self.baud):
            raise RuntimeError(f"could not set baud {self.baud} on {self.port_name}")
        self._packet = scs.PacketHandler(0)  # protocol_end=0 for STS/ST
        log.info("machine bus open: %s @ %d", self.port_name, self.baud)

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None

    # -- raw register access --
    def _check(self, what: str, result: int, error: int) -> None:
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(f"{what}: {self._packet.getTxRxResult(result)}")
        if error != 0:
            log.warning("%s: servo error %s", what, self._packet.getRxPacketError(error))

    def ping(self, sid: int) -> bool:
        _, result, _ = self._packet.ping(self._port, sid)
        return result == self._scs.COMM_SUCCESS

    def read2(self, sid: int, reg: int) -> int:
        value, result, error = self._packet.read2ByteTxRx(self._port, sid, reg)
        self._check(f"read2 id={sid} reg={reg}", result, error)
        return value

    def write1(self, sid: int, reg: int, value: int) -> None:
        result, error = self._packet.write1ByteTxRx(self._port, sid, reg, value)
        self._check(f"write1 id={sid} reg={reg}", result, error)

    def write2(self, sid: int, reg: int, value: int) -> None:
        result, error = self._packet.write2ByteTxRx(self._port, sid, reg, value)
        self._check(f"write2 id={sid} reg={reg}", result, error)

    # -- servo-level helpers --
    def position(self, sid: int) -> int:
        return self.read2(sid, REG_PRESENT_POSITION) & 0x0FFF

    def set_mode(self, sid: int, mode: int) -> None:
        self.write1(sid, REG_MODE, mode)

    def torque(self, sid: int, on: bool) -> None:
        self.write1(sid, REG_TORQUE_ENABLE, 1 if on else 0)

    def set_speed(self, sid: int, speed: int) -> None:
        self.write2(sid, REG_GOAL_SPEED, encode_signed(speed))

    def goal_position(self, sid: int, pos: int) -> None:
        self.write2(sid, REG_GOAL_POSITION, pos & 0x0FFF)

    def goal_step(self, sid: int, delta: int) -> None:
        """Step-mode (mode 3) relative move: signed counts, sign-magnitude encoded."""
        self.write2(sid, REG_GOAL_POSITION, encode_signed(int(delta)) & 0xFFFF)


class PositionServo:
    """A plain single-turn position-mode servo (bucket tilt, cup tipper)."""

    def __init__(self, bus: Bus, sid: int, speed: int = 800, tolerance: int = 25):
        self.bus = bus
        self.sid = sid
        self.speed = speed
        self.tolerance = tolerance

    def setup(self) -> None:
        if not self.bus.ping(self.sid):
            raise RuntimeError(f"servo id {self.sid} did not answer ping")
        self.bus.set_mode(self.sid, 0)
        self.bus.set_speed(self.sid, self.speed)
        self.bus.torque(self.sid, True)

    def where(self) -> int:
        return self.bus.position(self.sid)

    def move_to(self, pos: int, timeout_s: float = 4.0) -> None:
        self.bus.goal_position(self.sid, int(pos))
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if abs(wrap_delta(self.where(), int(pos))) <= self.tolerance:
                return
            time.sleep(0.02)
        raise TimeoutError(f"servo {self.sid} never reached {pos} (at {self.where()})")

    def torque_off(self) -> None:
        self.bus.torque(self.sid, False)


class Bucket(PositionServo):
    """ID 7: tips Prahlad's bucket to pour parts onto the vibrating table.
    rest_pos / pour_pos are VERIFY-on-hardware (see CALIBRATION.md)."""

    def __init__(self, bus: Bus, sid: int, rest_pos: int, pour_pos: int,
                 dwell_s: float = 1.5, speed: int = 500):
        super().__init__(bus, sid, speed=speed)
        self.rest_pos = rest_pos
        self.pour_pos = pour_pos
        self.dwell_s = dwell_s

    def rest(self) -> None:
        self.move_to(self.rest_pos)

    def pour(self) -> None:
        self.move_to(self.pour_pos)
        time.sleep(self.dwell_s)
        self.move_to(self.rest_pos)


class Cup(PositionServo):
    """ID 9: tips the cup on top of the distributor so the part slides into the bin."""

    def __init__(self, bus: Bus, sid: int, hold_pos: int, dump_pos: int,
                 dwell_s: float = 0.8, speed: int = 700):
        super().__init__(bus, sid, speed=speed)
        self.hold_pos = hold_pos
        self.dump_pos = dump_pos
        self.dwell_s = dwell_s

    def hold(self) -> None:
        self.move_to(self.hold_pos)

    def dump(self) -> None:
        self.move_to(self.dump_pos)
        time.sleep(self.dwell_s)
        self.move_to(self.hold_pos)


class Distributor:
    """ID 8: swivels the bin ring through a ~3.26:1 pinion. Multi-turn aware.

    Homing contract: when `home()` is called the ring must PHYSICALLY sit at
    bin 0. We verify the servo reads near `bin0_servo_pos` (recorded during
    calibration), but because the servo wraps ~3.26x per ring turn the same
    raw reading recurs at ~3 other ring angles — the check catches a nudged
    ring, not a ring parked at the wrong bin. Leave the ring at bin 0.
    """

    STEP_MODE = 3

    def __init__(self, bus: Bus, sid: int, ratio: float = 3.26, bins: int = 7,
                 bin0_servo_pos: int | None = None, tolerance_counts: int = 20,
                 settle_s: float = 0.3, speed: int = 1200,
                 max_excursion_turns: float = 1.5):
        self.bus = bus
        self.sid = sid
        self.ratio = ratio
        self.bins = bins
        self.bin0_servo_pos = bin0_servo_pos
        self.tolerance = tolerance_counts
        self.settle_s = settle_s
        self.speed = speed
        self.max_excursion = max_excursion_turns * RING
        self._tracker: MultiTurnTracker | None = None
        self._ring_abs = 0.0  # ring travel (counts) since homing; position = mod RING
        self.current_bin: int | None = None

    def setup(self) -> None:
        if not self.bus.ping(self.sid):
            raise RuntimeError(f"distributor servo id {self.sid} did not answer ping")
        self.bus.set_mode(self.sid, self.STEP_MODE)
        self.bus.set_speed(self.sid, self.speed)
        self.bus.torque(self.sid, True)

    def home(self) -> None:
        raw = self.bus.position(self.sid)
        if self.bin0_servo_pos is not None and \
                abs(wrap_delta(raw, self.bin0_servo_pos)) > self.tolerance * 3:
            raise RuntimeError(
                f"distributor reads {raw}, expected ~{self.bin0_servo_pos} for bin 0 — "
                "put the ring back at bin 0 (or re-record bin0_servo_pos, see CALIBRATION.md)")
        self._tracker = MultiTurnTracker(raw)
        self._ring_abs = 0.0
        self.current_bin = 0
        log.info("distributor homed at bin 0 (servo raw %d)", raw)

    def where(self) -> int:
        return self.bus.position(self.sid)

    def aim(self, bin_index: int, timeout_s: float = 8.0) -> None:
        """Move the ring to `bin_index` and block until within tolerance + settle."""
        if self._tracker is None:
            raise RuntimeError("distributor.aim() before home()")
        if not 0 <= bin_index < self.bins:
            raise ValueError(f"bin {bin_index} out of range 0..{self.bins - 1}")

        ring_delta = plan_ring_delta(self._ring_abs, bin_index, self.bins, self.max_excursion)
        servo_delta = round(ring_delta * self.ratio)
        abs_before = self._tracker.abs
        target_abs = abs_before + servo_delta
        log.info("aim bin %d: ring Δ%+.0f -> servo Δ%+d", bin_index, ring_delta, servo_delta)

        self.bus.goal_step(self.sid, servo_delta)
        t0 = time.monotonic()
        while True:
            abs_now = self._tracker.update(self.bus.position(self.sid))
            if abs(abs_now - target_abs) <= self.tolerance:
                break
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(
                    f"distributor: {abs(abs_now - target_abs)} counts from bin {bin_index} "
                    f"after {timeout_s}s — jammed?")
            time.sleep(0.02)
        # book travel from the servo's actual motion, not the commanded delta
        self._ring_abs += (self._tracker.abs - abs_before) / self.ratio
        self.current_bin = bin_index
        time.sleep(self.settle_s)

    def torque_off(self) -> None:
        self.bus.torque(self.sid, False)


# ===========================================================================
# Shaker — vibrating table behind the Kasa HS103 smart plug
# ===========================================================================

class Shaker:
    """ON/OFF only; the inline knob sets intensity. Plug latency ~0.2 s, so
    bursts shorter than 0.5 s are not meaningful. KLAP auth needs the Kasa
    account login (config / env, never hardcoded here)."""

    def __init__(self, ip: str, username: str, password: str):
        self.ip = ip
        self.username = username
        self.password = password

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    async def _set(self, on: bool) -> None:
        from kasa import Credentials, Discover

        dev = await Discover.discover_single(
            self.ip, credentials=Credentials(self.username, self.password))
        try:
            await dev.update()
            await (dev.turn_on() if on else dev.turn_off())
        finally:
            await dev.disconnect()

    def on(self) -> None:
        self._run(self._set(True))

    def off(self) -> None:
        self._run(self._set(False))

    def burst(self, seconds: float) -> None:
        if seconds < 0.5:
            log.warning("burst %.2fs < plug latency floor; rounding up to 0.5s", seconds)
            seconds = 0.5
        self.on()
        try:
            time.sleep(seconds)
        finally:
            self.off()

    def rhythm(self, on_s: float, off_s: float, n: int) -> None:
        for _ in range(n):
            self.burst(on_s)
            time.sleep(off_s)


# ===========================================================================
# CLI
# ===========================================================================

def _cli() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="AutoSort machine bus / shaker CLI")
    ap.add_argument("--port", default="/dev/machine")
    ap.add_argument("--baud", type=int, default=1_000_000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    p = sub.add_parser("where"); p.add_argument("id", type=int)
    p = sub.add_parser("torque"); p.add_argument("id", type=int); p.add_argument("state", choices=["on", "off"])
    p = sub.add_parser("move"); p.add_argument("id", type=int); p.add_argument("pos", type=int)
    p = sub.add_parser("aim"); p.add_argument("bin", type=int)
    p.add_argument("--bin0", type=int, required=True, help="servo raw position at bin 0 (from CALIBRATION.md)")
    p = sub.add_parser("shake"); p.add_argument("seconds", type=float)
    p = sub.add_parser("rhythm"); p.add_argument("on_s", type=float); p.add_argument("off_s", type=float); p.add_argument("n", type=int)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd in ("shake", "rhythm"):
        user, pw = os.environ.get("KASA_USER"), os.environ.get("KASA_PASS")
        ip = os.environ.get("KASA_IP", "192.168.1.38")
        if not (user and pw):
            raise SystemExit("set KASA_USER and KASA_PASS (and optionally KASA_IP)")
        shaker = Shaker(ip, user, pw)
        if args.cmd == "shake":
            shaker.burst(args.seconds)
        else:
            shaker.rhythm(args.on_s, args.off_s, args.n)
        return

    bus = Bus(args.port, args.baud)
    bus.open()
    try:
        if args.cmd == "scan":
            found = [sid for sid in range(1, 31) if bus.ping(sid)]
            print(f"servos answering on {args.port}: {found or 'none'}")
        elif args.cmd == "where":
            print(f"reading id {args.id} — Ctrl-C to stop")
            try:
                while True:
                    print(f"\r  position: {bus.position(args.id):4d}", end="", flush=True)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print()
        elif args.cmd == "torque":
            bus.torque(args.id, args.state == "on")
            print(f"id {args.id} torque {args.state}")
        elif args.cmd == "move":
            s = PositionServo(bus, args.id)
            s.setup()
            s.move_to(args.pos)
            print(f"id {args.id} at {s.where()}")
        elif args.cmd == "aim":
            d = Distributor(bus, 8, bin0_servo_pos=args.bin0)
            d.setup()
            d.home()
            d.aim(args.bin)
            print(f"at bin {d.current_bin} (servo raw {d.where()})")
    finally:
        bus.close()


if __name__ == "__main__":
    _cli()
