# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental Franka FR3 + Robotiq 2F-85 benchmark.

Validates that the three solver configurations (HYBRID, FULL-KAMINO,
FULL-MUJOCO) build and run a few steps without blowing up, that the hybrid
boundary coupling stays bounded, and that the benchmark harness produces a
finite speed number per config. SolverKamino is CUDA-only, so these tests skip
without a CUDA device.

NOTE: these tests download the Franka and Robotiq 2F-85 assets on first run
(cached afterwards) and compile several solvers, so they are slow.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton.examples.experimental import franka_2f85 as F
from newton.examples.experimental.example_franka_2f85_benchmark import _HybridConfig, run_benchmark


def _cuda_device() -> str | None:
    return "cuda:0" if wp.get_cuda_device_count() > 0 else None


@unittest.skipIf(_cuda_device() is None, "SolverKamino requires a CUDA device")
class TestFranka2f85Benchmark(unittest.TestCase):
    def test_models_build(self):
        """All three model shapes build from the cached assets with sane sizes."""
        device = _cuda_device()
        arm, flange = F.build_franka_arm(device)
        self.assertEqual(arm.joint_dof_count, 7)  # 7 revolute arm DOFs
        self.assertTrue(arm.body_label[flange].endswith("/fr3_link8"))

        grip, base, drivers = F.build_2f85_gripper(device, floating=True)
        self.assertGreaterEqual(grip.body_count, 14)
        self.assertEqual(len(drivers), 2)  # right + left driver joints
        self.assertGreater(float(grip.body_mass.numpy()[base]), 0.0)  # base must carry mass

        combined, _c_flange, arm_dofs, c_drivers = F.build_combined(device)
        self.assertEqual(combined.articulation_count, 1)  # single welded articulation
        self.assertEqual(len(arm_dofs), 7)
        self.assertEqual(len(c_drivers), 2)
        self.assertGreaterEqual(combined.body_count, 24)

    def test_hybrid_runs_and_stays_welded(self):
        """The hybrid Franka+2F-85 runs stably and the weld stays tight."""
        cfg = _HybridConfig(_cuda_device(), dt=1.0e-3)
        for _ in range(15):
            cfg._substep()
            self.assertEqual(cfg.solver.boundary_impulse.shape, (6,))
            self.assertTrue(np.all(np.isfinite(cfg.solver.boundary_impulse)))
        q = cfg.quality()
        self.assertTrue(q["finite"])
        # The rigid weld should hold to well under a centimeter.
        self.assertLess(q["boundary_pose_err"], 5.0e-2)
        self.assertLess(q["max_qd"], 50.0)  # no blow-up

    def test_benchmark_three_configs(self):
        """The benchmark harness runs all three configs and times each one."""
        results = run_benchmark(
            _cuda_device(), dt=1.0e-3, warmup=3, timed=4, configs=["FULL-MUJOCO", "FULL-KAMINO", "HYBRID"]
        )
        for name in ("FULL-MUJOCO", "FULL-KAMINO", "HYBRID"):
            self.assertIn(name, results)
            r = results[name]
            self.assertNotIn("error", r, msg=f"{name} failed: {r.get('error')}")
            self.assertGreater(r["ms_per_step"], 0.0)
            self.assertTrue(np.isfinite(r["ms_per_step"]))
            self.assertTrue(r["finite"], msg=f"{name} produced non-finite state")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
