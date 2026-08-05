"""Analytic pixel -> joint-angle solver (homography + inverse kinematics).

The taught-pose blending in taught.py is an interpolation: accurate inside the
convex hull of the taught points and increasingly wrong outside it. This module
solves the same problem from geometry instead, so accuracy no longer depends on
how close the target is to a taught sample:

    pixel (px, py)
      --[ homography H, exact for a planar surface ]-->  table (x, y)
      --[ inverse kinematics on the arm's URDF ]------>  joint angles

Calibration reuses the SAME taught.json: forward kinematics turns each taught
grasp pose into the gripper's real (x, y, z), which pairs with that point's
pixel coordinates to fit H. Four points determine a homography; more are fitted
by least squares.

Requires: placo (pip install placo) and an SO-101 URDF.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .motion import JOINTS

log = logging.getLogger("autosort.analytic")


class UnsafePoseError(RuntimeError):
    """Raised when a computed pose is rejected before it can be sent to the arm."""

DEFAULT_URDF = Path("/tmp/so101_urdf/so101_nomesh.urdf")


class AnalyticSolver:
    def __init__(self, taught, urdf_path: str | Path | None = None,
                 target_frame: str = "gripper_frame_link"):
        from lerobot.model.kinematics import RobotKinematics

        self.urdf_path = Path(urdf_path or DEFAULT_URDF)
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF not found at {self.urdf_path}. Fetch the SO-101 URDF and strip its mesh "
                "refs, or set arm.urdf_path in config.yaml."
            )
        self.kin = RobotKinematics(urdf_path=str(self.urdf_path),
                                   target_frame_name=target_frame, joint_names=JOINTS)
        self.taught = taught
        self.joint_margin_deg = 25.0   # how far outside the taught envelope a solve may go
        self.max_ik_error_mm = 15.0    # reject solutions that miss the requested point
        self._fit(taught)

    # --- calibration --------------------------------------------------
    def _fit(self, taught) -> None:
        """Fit pixel->table homography from the taught points via forward kinematics.

        Also fitted from the same data:
        - a single division-model distortion parameter k (Fitzgibbon): mild
          barrel warp, cross-validated worst-case gain ~3 mm on a 40-pt grid
        - a grasp-height PLANE z(x,y): the table is measurably tilted vs the
          arm's base (~16 mm corner-to-corner) - a constant median z was
          systematically ~5 mm wrong along the far edge
        """
        px, xy, z = [], [], []
        self._ref_poses = []
        for g in taught.grid:
            q = np.array([g["joints"][j] for j in JOINTS], dtype=float)
            T = self.kin.forward_kinematics(q)
            p = T[:3, 3]
            px.append(g["pixel"])
            xy.append(p[:2])
            z.append(p[2])
            self._ref_poses.append((q, T))
        self.pixels = np.asarray(px, dtype=float)
        self.table_xy = np.asarray(xy, dtype=float)
        self.grasp_z = float(np.median(z))

        # distortion: 1-D search over k, refitting H each time (fast, stable
        # with a spread grid; with few clustered points it just finds ~0 gain)
        best = (0.0, None, np.inf)
        for k in np.linspace(-0.5, 2.0, 126):
            up = self._undistort(self.pixels, k)
            H = _fit_homography(up, self.table_xy)
            proj = np.hstack([up, np.ones((len(up), 1))]) @ H.T
            r = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - self.table_xy, axis=1).mean()
            if r < best[2]:
                best = (k, H, r)
        self.k, self.H, _ = best

        # grasp-height plane z(x, y), least squares on table coords
        A = np.hstack([self.table_xy, np.ones((len(xy), 1))])
        self._z_plane, *_ = np.linalg.lstsq(A, np.asarray(z), rcond=None)

        # reachable sector: the arm sweeps an annular sector, not a rectangle.
        # The taught points trace it empirically (teach the fence where you
        # want it); margins let pieces slightly past the last taught point in.
        r = np.linalg.norm(self.table_xy, axis=1)
        ang = np.degrees(np.arctan2(self.table_xy[:, 1], self.table_xy[:, 0]))
        self._r_bounds = (float(r.min()) - 0.015, float(r.max()) + 0.015)
        self._ang_bounds = (float(ang.min()) - 5.0, float(ang.max()) + 5.0)

        # orientation: reuse the median taught grasp orientation (all are
        # top-down grasps; IK weights orientation weakly anyway)
        self._ref_R = self._ref_poses[len(self._ref_poses) // 2][1][:3, :3].copy()
        self._ref_q = np.median(np.stack([q for q, _ in self._ref_poses]), axis=0)

        resid = np.linalg.norm(self._apply_H(self.pixels) - self.table_xy, axis=1)
        log.info("homography fitted on %d points (k=%+.3f): mean residual %.1f mm, max %.1f mm",
                 len(px), self.k, resid.mean() * 1000, resid.max() * 1000)
        self.fit_residual_mm = float(resid.mean() * 1000)

    @staticmethod
    def _undistort(pixels: np.ndarray, k: float,
                   center=(480.0, 300.0), scale=566.0) -> np.ndarray:
        """Division-model undistortion; k=0 is the identity."""
        d = np.atleast_2d(pixels) - center
        r2 = (d ** 2).sum(1, keepdims=True) / scale ** 2
        return np.asarray(center + d / (1 + k * r2))

    def grasp_z_at(self, x: float, y: float) -> float:
        return float(np.array([x, y, 1.0]) @ self._z_plane)

    def reach_ok(self, px: float, py: float) -> bool:
        """Is the piece at this pixel inside the arm's taught reachable sector?"""
        x, y = self.table_xy_for_pixel(px, py)
        r = float(np.hypot(x, y))
        ang = float(np.degrees(np.arctan2(y, x)))
        return (self._r_bounds[0] <= r <= self._r_bounds[1]
                and self._ang_bounds[0] <= ang <= self._ang_bounds[1])

    def _apply_H(self, pixels: np.ndarray) -> np.ndarray:
        pts = self._undistort(np.atleast_2d(pixels).astype(float), self.k)
        ones = np.ones((len(pts), 1))
        hom = np.hstack([pts, ones]) @ self.H.T
        return hom[:, :2] / hom[:, 2:3]

    # --- solving ------------------------------------------------------
    def table_xy_for_pixel(self, px: float, py: float) -> np.ndarray:
        return self._apply_H(np.array([[px, py]]))[0]

    def _solve_ik(self, T_target: np.ndarray, q_seed: np.ndarray,
                  iters: int = 12, tol_m: float = 0.0005,
                  orientation_weight: float = 0.02) -> np.ndarray:
        """Iterate placo's single-step solver to convergence.

        inverse_kinematics() performs one local optimization step, so a seed far
        from the solution lands tens of mm away. Feeding the result back in
        converges; a good seed (the interpolated pose) makes it fast and stable.
        """
        q = np.asarray(q_seed, dtype=float).copy()
        for _ in range(iters):
            q = np.asarray(self.kin.inverse_kinematics(
                q, T_target, position_weight=1.0,
                orientation_weight=orientation_weight), dtype=float)
            err = np.linalg.norm(self.kin.forward_kinematics(q)[:3, 3] - T_target[:3, 3])
            if err < tol_m:
                break
        return q

    def _sanity_check(self, q: np.ndarray, label: str) -> np.ndarray:
        """Reject IK solutions that are physically implausible for this task.

        An IK solver can return a mathematically valid pose that swings the arm
        somewhere absurd (reaching the same point from a mirrored configuration).
        Every taught grasp is a top-down pick in a small workspace, so any joint
        far outside the taught envelope means the solve went wrong - we refuse it
        rather than driving the arm there and hitting something.
        """
        taught_q = np.stack([[g["joints"][j] for j in JOINTS] for g in self.taught.grid])
        lo = taught_q.min(axis=0) - self.joint_margin_deg
        hi = taught_q.max(axis=0) + self.joint_margin_deg
        bad = [(JOINTS[i], round(float(q[i]), 1), round(float(lo[i]), 1), round(float(hi[i]), 1))
               for i in range(len(JOINTS) - 1)  # gripper is commanded separately
               if q[i] < lo[i] or q[i] > hi[i]]
        if bad:
            raise UnsafePoseError(
                f"IK solution for {label} is outside the taught envelope: "
                + "; ".join(f"{n}={v} not in [{a},{b}]" for n, v, a, b in bad)
            )
        return q

    def grasp_for_pixel(self, px: float, py: float,
                        z_offset: float = 0.0) -> dict[str, float]:
        """Joint angles that put the gripper on the piece seen at (px, py).

        The taught-pose interpolation supplies the seed (close, but only accurate
        near taught samples); the IK then solves the geometry exactly, which is
        what makes this valid across the whole workspace.
        """
        x, y = self.table_xy_for_pixel(px, py)
        T = np.eye(4)
        T[:3, :3] = self._ref_R
        T[:3, 3] = [x, y, self.grasp_z_at(x, y) + z_offset]
        seed = np.array([self.taught.grasp_for_pixel(px, py)[j] for j in JOINTS], dtype=float)

        # The interpolated seed can land in the wrong IK basin (observed: one
        # taught point missing by 16 mm from its own seed). Retry from the
        # nearest taught poses before refusing - each is a known-good grasp
        # configuration, so its basin usually contains the solution.
        roll = JOINTS.index("wrist_roll")
        near = np.argsort(np.linalg.norm(self.pixels - [px, py], axis=1))[:3]
        seeds = [seed] + [self._ref_poses[i][0] for i in near]
        best_q, best_miss = None, np.inf
        # Two rounds: normal orientation weight, then position-first. Near the
        # zone edge the median grasp orientation can be slightly infeasible and
        # the orientation term drags the solve ~16 mm off the point (measured);
        # a near-zero weight nails position, and the wrist_roll pin plus the
        # envelope guard still keep the pose sane.
        for ow in (0.02, 0.001):
            for s0 in seeds:
                q = self._solve_ik(T, s0, orientation_weight=ow)
                q[roll] = seed[roll]
                miss = float(np.linalg.norm(self.kin.forward_kinematics(q)[:3, 3] - T[:3, 3]) * 1000)
                if miss < best_miss:
                    best_q, best_miss = q, miss
                if miss <= self.max_ik_error_mm:
                    break
            if best_miss <= self.max_ik_error_mm:
                break
        q, miss_mm = best_q, best_miss
        # (wrist_roll stays pinned to the interpolated seed: the gripper origin
        # sits on the roll axis, and free roll drifts ~28 deg and trips the
        # envelope guard. Orientation-aware grasping will set it explicitly.)

        if miss_mm > self.max_ik_error_mm:
            raise UnsafePoseError(
                f"IK did not converge for pixel ({px:.0f},{py:.0f}): off by {miss_mm:.0f} mm"
            )
        q = self._sanity_check(q, f"pixel ({px:.0f},{py:.0f})")
        out = {j: float(q[i]) for i, j in enumerate(JOINTS)}
        # the gripper joint is commanded separately by the pick sequence
        out["gripper"] = float(seed[JOINTS.index("gripper")])
        return out

    def hover_for(self, grasp: dict[str, float], lift_m: float = 0.08) -> dict[str, float]:
        """A pose above `grasp`, aiming lift_m straight up.

        The arm has 5 DOF, so "same orientation, lift_m higher" is usually NOT
        exactly reachable — the solve projects onto the feasible manifold
        (measured: ~50 mm of an 80 mm request, <11 mm lateral). The guards
        therefore check what hover actually needs to be safe:
          - gained at least half the requested lift (clears the pile),
          - stayed laterally above the grasp (descent is near-vertical),
          - every joint inside the taught envelope (the incident guard).
        """
        q = np.array([grasp[j] for j in JOINTS], dtype=float)
        T_g = self.kin.forward_kinematics(q)
        T = T_g.copy()
        T[2, 3] += lift_m
        qh = self._solve_ik(T, q)
        qh[JOINTS.index("wrist_roll")] = q[JOINTS.index("wrist_roll")]  # same pin as the grasp
        p_h, p_g = self.kin.forward_kinematics(qh)[:3, 3], T_g[:3, 3]
        dz_mm = float((p_h[2] - p_g[2]) * 1000)
        lat_mm = float(np.linalg.norm(p_h[:2] - p_g[:2]) * 1000)
        if dz_mm < 0.5 * lift_m * 1000:
            raise UnsafePoseError(
                f"hover only gained {dz_mm:.0f} mm of the requested {lift_m*1000:.0f} mm lift")
        if lat_mm > 25.0:
            raise UnsafePoseError(f"hover drifted {lat_mm:.0f} mm sideways from the grasp point")
        qh = self._sanity_check(qh, "hover pose")
        out = {j: float(qh[i]) for i, j in enumerate(JOINTS)}
        out["gripper"] = grasp["gripper"]
        return out


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography mapping src (pixels) -> dst (table metres)."""
    n = len(src)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        sx, sy = src[i]
        dx, dy = dst[i]
        A[2 * i] = [sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy, -dx]
        A[2 * i + 1] = [0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy, -dy]
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]
