"""AutoSort orchestrator CLI.

    python -m orchestrator.run --sim                    # no hardware, realistic timing
    python -m orchestrator.run --dry-run                # sim with no delays: print the transitions
    python -m orchestrator.run --skip-arm               # feeder + distributor only, no /dev/arm
    python -m orchestrator.run --manual-pick            # you move the part to the cup
    python -m orchestrator.run --max-parts 5            # stop after 5 parts

Each run writes runs/<timestamp>/{run.log,log.csv}. Touch runs/<timestamp>/ESTOP
(or press Ctrl-C) to stop with every actuator off.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Config
from .cycle import Orchestrator


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AutoSort orchestrator — full sorting cycle")
    ap.add_argument("-c", "--config", default=None, help="path to config.yaml (default: orchestrator/config.yaml)")
    ap.add_argument("--sim", action="store_true", help="stub every actuator/camera with realistic delays")
    ap.add_argument("--dry-run", action="store_true", help="like --sim but with zero delays")
    ap.add_argument("--max-parts", type=int, default=None, help="stop after N parts")
    ap.add_argument("--skip-arm", action="store_true", help="run feeder/distributor without the arm")
    ap.add_argument("--manual-pick", action="store_true", help="a human moves each part to the cup")
    ap.add_argument("--runs-dir", default="runs", help="where run directories go")
    ap.add_argument("--sim-parts", type=int, default=12, help="(sim) parts in the bucket")
    ap.add_argument("--sim-seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)-20s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")],
    )
    log = logging.getLogger("orchestrator")

    cfg = Config.load(args.config)
    simulated = args.sim or args.dry_run
    if simulated:
        from .sim import SimRig

        rig = SimRig(cfg, bucket_parts=args.sim_parts, time_scale=0.0 if args.dry_run else 1.0,
                     seed=args.sim_seed)
        sleep = (lambda s: None) if args.dry_run else time.sleep
    else:
        missing = cfg.missing_hardware_values(skip_arm=args.skip_arm)
        if missing:
            log.error("config has VERIFY values still unset: %s — see CALIBRATION.md", ", ".join(missing))
            return 2
        from .hw import RealRig

        rig = RealRig(cfg, skip_arm=args.skip_arm, manual_pick=args.manual_pick)
        sleep = time.sleep

    log.info("run dir %s  (sim=%s skip_arm=%s manual_pick=%s max_parts=%s)",
             run_dir, simulated, args.skip_arm, args.manual_pick, args.max_parts)
    log.info("to stop safely: Ctrl-C or `touch %s/ESTOP`", run_dir)
    orch = Orchestrator(cfg, rig, run_dir, max_parts=args.max_parts, sleep=sleep)
    return orch.run()


if __name__ == "__main__":
    sys.exit(main())
