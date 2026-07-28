"""The AutoSort pipeline.

One cycle:  pick -> verify exactly one -> place in box -> classify -> route to bin.
Repeats until the pile is empty (or too many picks fail in a row).

    setup()  connect arm, classifier, router; home everything
    run()    the loop above, with graceful shutdown on Ctrl-C
"""
from __future__ import annotations

import argparse
import logging

from .arm import Arm
from .classifier import Classifier
from .config import Config
from .perception import Perception
from .router import Router

log = logging.getLogger("autosort")


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        dry = cfg.run.dry_run
        self.arm = Arm(cfg.arm, cfg.cameras, dry)
        self.perception = Perception(cfg.perception, dry)
        self.classifier = Classifier(cfg.classifier, cfg.cameras.get("box"), dry)
        self.router = Router(cfg.router, dry)

    def setup(self) -> None:
        self.arm.connect()
        self.classifier.connect()
        self.router.connect()
        self.arm.home()
        self.router.home()

    def run(self) -> None:
        self.setup()
        sorted_count = 0
        fails = 0
        empty_reads = 0
        try:
            while True:
                # 1. done? pile empty (confirmed over several frames) or too many failures
                if self.perception.pieces_on_tray(self.arm.frame("top")) == 0:
                    empty_reads += 1
                else:
                    empty_reads = 0
                if empty_reads >= self.cfg.perception.empty_frames:
                    log.info("pile empty — %d pieces sorted. done.", sorted_count)
                    break
                if fails >= self.cfg.run.max_consecutive_fails:
                    log.warning("%d failed picks in a row — stopping.", fails)
                    break

                # 2. pick one, then confirm it really is exactly one.
                # Target is measured NOW, while the arm is at home and the view
                # of the pile is unobstructed (classical mode uses it; ACT ignores it).
                target_px = self.perception.largest_piece_px(self.arm.frame("top"))
                if target_px is None and not self.cfg.run.dry_run:
                    log.info("no pick target found — retrying")
                    fails += 1
                    continue
                self.arm.pick(target_px)
                if not self.arm.gripper_holding():
                    log.info("empty grasp — retrying")
                    fails += 1
                    continue
                n = self.perception.pieces_in_gripper(self.arm.frame("wrist"))
                if n == 0:
                    log.info("nothing seen in gripper — retrying")
                    fails += 1
                    continue
                if n >= 2:
                    log.info("grabbed %d pieces — dropping back", n)
                    self.arm.drop_back()
                    fails += 1
                    continue

                # 3. place -> classify -> route
                self.arm.place_in_box()
                label, conf = self.classifier.classify()
                log.info("classified: %s (%.2f)", label, conf)
                self.router.route_to(label)

                fails = 0
                sorted_count += 1
                log.info("cycle %d done", sorted_count)

                # 4. in step mode, wait for a go-ahead before the next piece
                if self.cfg.run.mode == "step":
                    input("  press Enter for the next piece... ")
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.arm.home()
        finally:
            self.arm.disconnect()
            self.classifier.disconnect()
            self.router.disconnect()
        log.info("shut down cleanly")


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoSort — autonomous VEX hardware sorter")
    ap.add_argument("-c", "--config", default=None, help="path to config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="simulate everything (no hardware/models)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.load(args.config)
    if args.dry_run:
        cfg.run.dry_run = True
    log.info("AutoSort starting (dry_run=%s, mode=%s)", cfg.run.dry_run, cfg.run.mode)
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
