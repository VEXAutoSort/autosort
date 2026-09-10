"""Offline validation of taught.json BEFORE anything touches the arm.

Run:  python tools/validate_taught.py [taught.json] [--drop 3,17]

Prints, from the taught grid alone (no hardware):
  1. fit summary: points, distortion k, homography residual, z-plane tilt,
     reachable sector
  2. leave-one-out table: for every point, refit WITHOUT it and predict its
     table position from its pixel -> error vector (mm). Error VECTORS are
     the outlier tool: a bad point (strained teach, wrist rotated, hand in
     frame at lock) shows as a large vector that points the same way as its
     neighbours' don't. Flags > 2.5x median or > 8 mm.
  3. configuration fidelity: solve every taught pixel and compare solved
     joints to the RECORDED joints (catches null-space wander that position
     checks alone hide)
  4. guard regression: 100% of taught pixels must pass grasp + hover guards;
     absurd / out-of-frame inputs must be REFUSED
--drop lets you preview the fit without suspect points (1-based indices as
printed) before actually removing them from taught.json.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autosort.analytic import AnalyticSolver, UnsafePoseError  # noqa: E402
from autosort.config import Config                             # noqa: E402
from autosort.motion import JOINTS                             # noqa: E402
from autosort.taught import Taught                             # noqa: E402

import logging
logging.getLogger("autosort.analytic").setLevel(logging.WARNING)


def make_solver(data: dict, cfg) -> AnalyticSolver:
    return AnalyticSolver(Taught(data), urdf_path=cfg.arm.urdf_path,
                          orient_min_aspect=cfg.arm.orient_min_aspect,
                          orient_roll_offset_deg=cfg.arm.orient_roll_offset_deg,
                          orient_roll_extra_deg=cfg.arm.orient_roll_extra_deg)


def main() -> None:
    drop: set[int] = set()
    argv = list(sys.argv[1:])
    if "--drop" in argv:
        k = argv.index("--drop")
        drop = {int(x) for x in argv[k + 1].split(",")}
        del argv[k:k + 2]
    args = [a for a in argv if not a.startswith("--")]
    path = Path(args[0]) if args else ROOT / "taught.json"
    cfg = Config.load()
    data = json.loads(path.read_text())
    if drop:
        data["grid"] = [g for i, g in enumerate(data["grid"], 1) if i not in drop]
        print(f"(previewing without points {sorted(drop)})")
    n = len(data["grid"])
    if n < 4:
        sys.exit(f"only {n} grid points - need at least 4 (aim for 40)")

    full = make_solver(data, cfg)
    print(f"== {path.name}: {n} grid points ==")
    print(f"distortion k = {full.k:+.3f}   homography residual mean {full.fit_residual_mm:.1f} mm")
    xy = full.table_xy
    zc = [full.grasp_z_at(x, y) for x, y in
          [(xy[:, 0].min(), xy[:, 1].min()), (xy[:, 0].max(), xy[:, 1].max()),
           (xy[:, 0].min(), xy[:, 1].max()), (xy[:, 0].max(), xy[:, 1].min())]]
    print(f"z-plane: grasp z {min(zc)*1000:.1f}..{max(zc)*1000:.1f} mm over the hull "
          f"(tilt {1000*(max(zc)-min(zc)):.1f} mm corner-to-corner)")
    print(f"reach sector: r {full._r_bounds[0]*1000:.0f}..{full._r_bounds[1]*1000:.0f} mm, "
          f"pan {full._ang_bounds[0]:.0f}..{full._ang_bounds[1]:.0f} deg")
    print(f"table extent: x {xy[:,0].min()*1000:.0f}..{xy[:,0].max()*1000:.0f} mm, "
          f"y {xy[:,1].min()*1000:.0f}..{xy[:,1].max()*1000:.0f} mm")

    # ---- leave-one-out ----------------------------------------------------
    print("\n== leave-one-out (refit without the point, predict it) ==")
    print(f"{'#':>3} {'pixel':>12} {'err':>6} {'dx':>6} {'dy':>6}  flag")
    errs, vecs = [], []
    for i in range(n):
        sub = copy.deepcopy(data)
        held = sub["grid"].pop(i)
        s = make_solver(sub, cfg)
        pred = s.table_xy_for_pixel(*held["pixel"])
        true = full.table_xy[i]
        v = (pred - true) * 1000
        vecs.append(v); errs.append(float(np.linalg.norm(v)))
    errs = np.array(errs); med = float(np.median(errs))
    for i in range(n):
        flag = ""
        if errs[i] > max(2.5 * med, 8.0):
            flag = "<-- OUTLIER"
        elif errs[i] > 2.0 * med:
            flag = "(watch)"
        px = data["grid"][i]["pixel"]
        print(f"{i+1:>3} {px[0]:6.0f},{px[1]:<5.0f} {errs[i]:6.1f} {vecs[i][0]:+6.1f} {vecs[i][1]:+6.1f}  {flag}")
    print(f"LOOCV: mean {errs.mean():.1f} mm  median {med:.1f} mm  max {errs.max():.1f} mm  "
          f"(#{int(errs.argmax())+1})   [previous rig: ~4.0 mean / 11.5 max on 42 pts]")

    # ---- configuration fidelity + guards on every taught pixel --------------
    print("\n== solve every taught pixel: position miss + joint drift vs RECORDED joints ==")
    misses, drifts, refused = [], [], []
    for i, g in enumerate(data["grid"]):
        try:
            q = full.grasp_for_pixel(*g["pixel"])
            full.hover_for(q, cfg.arm.hover_lift_m)
        except UnsafePoseError as e:
            refused.append((i + 1, str(e)))
            continue
        qv = np.array([q[j] for j in JOINTS])
        rec = np.array([g["joints"][j] for j in JOINTS])
        p = full.kin.forward_kinematics(qv)[:3, 3]
        misses.append(float(np.linalg.norm(p - full.kin.forward_kinematics(rec)[:3, 3]) * 1000))
        drifts.append(float(np.abs(qv[:4] - rec[:4]).max()))
    misses, drifts = np.array(misses), np.array(drifts)
    if len(misses):
        print(f"fingertip position vs recorded: mean {misses.mean():.1f} mm, max {misses.max():.1f} mm")
        print(f"max arm-joint drift from recorded: mean {drifts.mean():.1f} deg, max {drifts.max():.1f} deg "
              f"({'OK' if drifts.max() < 8 else 'HIGH - inspect configuration'})")
    print(f"guards: {n - len(refused)}/{n} taught pixels pass grasp+hover "
          f"({'PASS' if not refused else 'FAIL'})")
    for i, e in refused:
        print(f"   refused #{i}: {e}")

    # ---- absurd inputs must be refused -------------------------------------
    print("\n== absurd inputs (all must be REFUSED) ==")
    ok = True
    for px in [(-100, -100), (5000, 300), (480, 10000), (-1, 300), (961, 300)]:
        try:
            full.grasp_for_pixel(*px)
            print(f"   {px}: SOLVED - GUARD FAILURE"); ok = False
        except UnsafePoseError as e:
            print(f"   {px}: refused ({str(e)[:60]})")
    # z-offset clamp: a config typo must not drive the tips into the table
    c = data["grid"][n // 2]["pixel"]
    qa = full.grasp_for_pixel(*c, z_offset=-0.5)
    qb = full.grasp_for_pixel(*c, z_offset=-0.010)
    same = all(abs(qa[j] - qb[j]) < 1e-6 for j in JOINTS)
    print(f"   z_offset -500 mm clamps to -10 mm: {'yes' if same else 'NO - GUARD FAILURE'}")
    ok = ok and same
    print(f"\nVERDICT: {'guards PASS' if ok and not refused else 'guards FAIL'}; LOOCV mean {errs.mean():.1f} mm")


if __name__ == "__main__":
    main()
