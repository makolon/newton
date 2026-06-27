# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental Franka FR3 + Robotiq 2F-85 cube-stacking example.

Validates that the three solver configurations (``mujoco``, ``kamino`` and the
boundary-impulse ``hybrid``) build and run a few frames without blowing up, that
the hybrid's split arm/gripper models partition the combined bodies exactly and
its weld stays tight, and that the CUDA-graph-captured substep loop replays
correctly. SolverKamino is CUDA-only, so these tests skip without a CUDA device.

NOTE: these tests download the Franka and Robotiq 2F-85 assets on first run
(cached afterwards) and compile several solvers, so they are slow.
"""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.experimental.example_franka_2f85_stacking import Example, HybridExample, make_example


def _cuda_device() -> bool:
    return wp.get_cuda_device_count() > 0


def _args(solver: str) -> argparse.Namespace:
    # Use the per-solver default substep counts: the stiff PD-held arm needs a
    # small enough substep dt to integrate stably (a smaller override diverges).
    return argparse.Namespace(solver=solver, verbose=False, no_graph=False, substeps=None)


@unittest.skipIf(not _cuda_device(), "SolverKamino requires a CUDA device")
class TestFranka2f85Stacking(unittest.TestCase):
    def test_hybrid_body_split_and_weld(self):
        """The hybrid dispatches to HybridExample, partitions bodies exactly, runs
        finite, and holds the flange<->gripper-base weld tight."""
        viewer = newton.viewer.ViewerNull(num_frames=12)
        ex = make_example(viewer, _args("hybrid"))
        self.assertIsInstance(ex, HybridExample)
        # Split arm + (gripper+cubes) partition the combined bookkeeping model.
        self.assertEqual(ex.n_arm + ex.grip_model.body_count, ex.num_bodies_per_world)

        for _ in range(8):
            ex.step()

        self.assertTrue(np.all(np.isfinite(ex.state_0.body_q.numpy())))
        # The boundary impulse welds the gripper base onto the arm flange; the gap
        # and the reported pose error should both stay well under a centimeter.
        flange = ex.arm_0.body_q.numpy()[ex.arm_flange][:3]
        gbase = ex.grip_0.body_q.numpy()[ex.grip_base][:3]
        self.assertLess(float(np.linalg.norm(flange - gbase)), 2.0e-2)
        self.assertLess(float(np.linalg.norm(ex.solver.boundary_pose_error)), 5.0e-2)

    def test_hybrid_overload_is_clamped(self):
        """Under a dynamic overload the boundary coupling clamps the transmitted
        wrench and the reduced velocity correction, so it under-corrects instead of
        whipping the arm or blasting the gripper. Removing either clamp makes the
        unbounded impulse exceed these caps."""
        viewer = newton.viewer.ViewerNull(num_frames=4)
        args = _args("hybrid")
        args.no_graph = True
        ex = make_example(viewer, args)
        fmax, tmax, dvmax = 25.0, 5.0, 0.05
        ex.solver.config.max_boundary_force = fmax
        ex.solver.config.max_boundary_torque = tmax
        ex.solver.config.max_reduced_correction = dvmax

        # Inject a gross boundary velocity error: race the arm joints so the welded
        # flange shoots ahead of the (held) gripper base. The bare impulse would be
        # huge here; the caps must bound it.
        qd = ex.arm_0.joint_qd.numpy()
        qd[:] = 30.0
        ex.arm_0.joint_qd.assign(qd)
        ex.arm_0.clear_forces()
        ex.grip_0.clear_forces()
        ex.solver.step(ex.arm_0, ex.arm_1, ex.grip_0, ex.grip_1, ex.arm_ctrl, ex.grip_ctrl, ex.sim_dt, ex.grip_contacts)

        dt = ex.sim_dt
        lam = ex.solver.boundary_impulse  # impulse; wrench = lambda / dt
        self.assertLessEqual(float(np.linalg.norm(lam[:3])) / dt, fmax * 1.05)
        self.assertLessEqual(float(np.linalg.norm(lam[3:])) / dt, tmax * 1.05)
        # The per-step reduced (arm) velocity correction is bounded directly.
        self.assertLessEqual(float(np.linalg.norm(ex.solver._dv.numpy())), dvmax * 1.05)
        self.assertTrue(np.all(np.isfinite(ex.arm_0.body_q.numpy())))

    def test_hybrid_effective_inertia_probed(self):
        """The effective-inertia probe runs on-device under graph capture, yields a
        finite SPD operator, and captures the loaded angular inertia that the free
        base under-estimates, so the angular weld stays tight with only the light
        velocity-Baumgarte crutch."""
        viewer = newton.viewer.ViewerNull(num_frames=12)
        ex = make_example(viewer, _args("hybrid"))
        self.assertTrue(ex.solver.config.use_effective_inertia)
        for _ in range(8):
            ex.step()

        self.assertEqual(int(ex.solver._probe_ok.numpy()[0]), 1)
        A = ex.solver._A_m_eff.numpy()
        self.assertTrue(np.all(np.isfinite(A)))
        # Symmetric positive definite: the boundary Cholesky factorization needs it.
        eig = np.linalg.eigvalsh(0.5 * (A + A.T))
        self.assertGreater(float(eig.min()), 0.0)
        # The probe sees the loaded angular inertia: its (rotation-invariant) inverse
        # trace is markedly below the bare base's -- the root cause of the weak weld.
        free_ang_tr = float(np.trace(np.asarray(ex.solver._base_inv_inertia_local).reshape(3, 3)))
        probed_ang_tr = float(np.trace(A[3:, 3:]))
        self.assertLess(probed_ang_tr, free_ang_tr)
        # The angular weld stays tight (< 5 deg) with the relaxed crutch.
        self.assertLess(float(np.linalg.norm(ex.solver.boundary_pose_error[3:])), np.deg2rad(5.0))

    def test_kamino_runs_finite(self):
        """The full-Kamino config builds (unified mesh collision) and stays finite."""
        viewer = newton.viewer.ViewerNull(num_frames=12)
        ex = make_example(viewer, _args("kamino"))
        self.assertNotIsInstance(ex, HybridExample)
        for _ in range(8):
            ex.step()
        ex.test_final()

    def test_mujoco_runs_finite(self):
        """The reference MuJoCo config builds and stays finite."""
        viewer = newton.viewer.ViewerNull(num_frames=12)
        ex = make_example(viewer, _args("mujoco"))
        self.assertIsInstance(ex, Example)
        for _ in range(8):
            ex.step()
        ex.test_final()


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
