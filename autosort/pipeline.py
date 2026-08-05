"""The AutoSort pipeline.

One cycle:  pick -> verify exactly one -> place in box -> classify -> route to bin.
Repeats until the pile is empty (or too many picks fail in a row).

    setup()  connect arm, classifier, router; home everything
    run()    the loop above, with graceful shutdown on Ctrl-C
"""
from __future__ import annotations

import argparse
import logging
import time

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

        # ArUco drift correction: active only when enabled AND a reference was
        # captured (tools/capture_markers.py). Absent reference = feature off.
        self.recal = None
        if cfg.recal.enabled and not dry:
            from pathlib import Path

            from .recal import Recalibrator
            ref = Path(cfg.recal.ref_file) if cfg.recal.ref_file else Path("markers_ref.json")
            if ref.exists():
                self.recal = Recalibrator(ref, cfg.recal.min_markers,
                                          cfg.recal.max_correction_px)
                log.info("ArUco drift correction active (%d reference markers)",
                         len(self.recal.ref))
            else:
                log.info("no ArUco reference (%s) - drift correction off. Capture one "
                         "with tools/capture_markers.py when picking is accurate.", ref)

    def setup(self) -> None:
        self.arm.connect()
        self.classifier.connect()
        self.router.connect()
        self.arm.home()
        self.arm.open_gripper() if hasattr(self.arm, "open_gripper") else None
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
                top_frame = self.arm.frame("top")
                if top_frame is None and not self.cfg.run.dry_run:
                    log.error("top camera unavailable - cannot locate a piece; retrying")
                    fails += 1
                    time.sleep(1.0)
                    continue
                target_px = self.perception.largest_piece_px(top_frame)
                if target_px is None and not self.cfg.run.dry_run:
                    log.info("no pick target found — retrying")
                    fails += 1
                    continue
                if self.recal is not None and target_px is not None:
                    from .recal import RecalError
                    try:
                        H = self.recal.correction(top_frame)
                        target_px = self.recal.apply(H, *target_px)
                    except RecalError as e:
                        # picking uncorrected after a failed check could aim a
                        # drifted camera's pixels at the wrong table spot - skip
                        log.error("drift correction refused (%s) - skipping cycle", e)
                        fails += 1
                        time.sleep(1.0)
                        continue
                self.arm.pick(target_px)
                # Two holding signals: gripper position (fooled by gears' spokes)
                # and the wrist camera (the deciding vote). Empty only if BOTH say so.
                pos_holding = self.arm.gripper_holding()
                wrist_frame = self.arm.frame("wrist")
                if wrist_frame is None:
                    # camera dropped out: trust the gripper position alone rather
                    # than reading "no frame" as "no piece"
                    n = 1 if pos_holding else 0
                    log.warning("wrist camera unavailable - using gripper position only (%s)",
                                "holding" if pos_holding else "empty")
                else:
                    n = self.perception.pieces_in_gripper(wrist_frame)
                    log.info("hold check: position says %s, wrist camera sees %d piece(s)",
                             "holding" if pos_holding else "empty", n)
                if n >= 2:
                    log.info("grabbed %d pieces — dropping back", n)
                    self.arm.drop_back()
                    fails += 1
                    continue
                if n == 0 and not pos_holding:
                    log.info("empty grasp — retrying")
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
        except Exception as e:
            log.warning("could not return home during shutdown: %s", e)
        finally:
            try:
                self.arm.disconnect()
            except Exception:
                pass
            for closer in (self.classifier.disconnect, self.router.disconnect):
                try:
                    closer()
                except Exception:
                    pass
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
    if not cfg.run.dry_run:
        from pathlib import Path as _P
        override = _P(args.config).parent if args.config else _P(__file__).resolve().parent.parent
        if not (override / "cameras_override.json").exists():
            raise SystemExit(
                "cameras_override.json missing - run tools/select_cameras.py before a real run "
                "(macOS shuffles USB camera indices; hardcoded ones are not trustworthy)."
            )
    log.info("AutoSort starting (dry_run=%s, mode=%s)", cfg.run.dry_run, cfg.run.mode)
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
