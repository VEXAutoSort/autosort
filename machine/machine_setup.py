"""Set a Feetech ST3215's bus ID. Plug in ONE servo at a time.

    python machine/machine_setup.py --port /dev/machine 7      # bucket
    python machine/machine_setup.py --port /dev/machine 8      # distributor
    python machine/machine_setup.py --port /dev/machine 9      # cup tipper

Already done for 7/8/9 on the current rig — keep this for replacements.
Factory ID is 1. Scans the bus first and refuses if more than one servo answers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine import REG_ID, REG_LOCK, Bus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_id", type=int)
    ap.add_argument("--port", default="/dev/machine")
    ap.add_argument("--baud", type=int, default=1_000_000)
    args = ap.parse_args()
    if not 1 <= args.new_id <= 253:
        raise SystemExit("id must be 1..253")

    bus = Bus(args.port, args.baud)
    bus.open()
    try:
        found = [sid for sid in range(1, 31) if bus.ping(sid)]
        if len(found) != 1:
            raise SystemExit(f"expected exactly one servo on the bus, found {found}")
        old = found[0]
        if old == args.new_id:
            print(f"servo already has id {old}")
            return
        bus.write1(old, REG_LOCK, 0)          # unlock EEPROM
        bus.write1(old, REG_ID, args.new_id)
        time.sleep(0.05)
        bus.write1(args.new_id, REG_LOCK, 1)  # lock it again
        time.sleep(0.05)
        if not bus.ping(args.new_id):
            raise SystemExit("servo did not answer on the new id — power-cycle and rescan")
        print(f"servo id {old} -> {args.new_id}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
