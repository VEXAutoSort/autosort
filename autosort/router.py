"""Rotating-arm router: turn to the bin for a label, then let the piece drop.

Firmware protocol to implement on the Arduino / servo controller:
    host -> "G<angle>\\n"    rotate to <angle> degrees
    ctrl -> "OK\\n"          movement complete
"""
from __future__ import annotations

import logging
import time

from .config import RouterCfg

log = logging.getLogger("autosort.router")


class Router:
    def __init__(self, cfg: RouterCfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._ser = None

    def connect(self) -> None:
        if self.dry_run:
            return
        import serial

        self._ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=5)
        time.sleep(2)  # controller resets when the port opens

    def route_to(self, label: str) -> None:
        angle = self.cfg.bins.get(label, self.cfg.bins["unknown"])
        log.info("route '%s' -> %.0f deg", label, angle)
        if self.dry_run:
            return
        self._command(f"G{angle:.0f}")
        time.sleep(self.cfg.drop_dwell_s)  # hold while the piece falls into the bin

    def home(self) -> None:
        if self.dry_run:
            return
        self._command("G0")

    def _command(self, cmd: str) -> None:
        self._ser.write((cmd + "\n").encode())
        self._ser.flush()
        ack = self._ser.readline().decode().strip()
        if ack != "OK":
            log.warning("router: expected 'OK', got %r", ack)

    def disconnect(self) -> None:
        if self._ser is not None:
            self._ser.close()
