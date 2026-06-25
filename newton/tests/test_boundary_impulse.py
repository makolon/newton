# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental SolverBoundaryImpulse (reduced<->maximal coupling).

SolverKamino (the maximal subsystem) currently requires a CUDA device, so these
tests skip when no CUDA device is available.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.experimental.example_boundary_impulse_toy import BoundaryImpulseToy


def _cuda_device() -> str | None:
    return "cuda:0" if wp.get_cuda_device_count() > 0 else None


@unittest.skipIf(_cuda_device() is None, "SolverKamino requires a CUDA device")
class TestSolverBoundaryImpulse(unittest.TestCase):
    """Smoke + sanity tests for the boundary-impulse coupling solver."""

    def _make_toy(self, baumgarte: float = 0.2):
        return BoundaryImpulseToy(device=_cuda_device(), baumgarte=baumgarte)

    def test_runs_and_stays_finite(self):
        """A few coupled steps run; all diagnostics are finite and well-shaped."""
        toy = self._make_toy()
        history = toy.run(num_steps=12, verbose=False)
        self.assertEqual(len(history), 12)

        for d in history:
            # Boundary impulse / errors have the right shape (6D wrench-impulse).
            self.assertEqual(d["boundary_impulse"].shape, (6,))
            self.assertEqual(d["boundary_pos_error"].shape, (6,))
            self.assertEqual(d["boundary_vel_error_pre"].shape, (6,))
            self.assertEqual(d["boundary_vel_error_post"].shape, (6,))
            # No NaN / Inf anywhere in the reported state.
            for key, val in d.items():
                self.assertTrue(np.all(np.isfinite(val)), msg=f"non-finite values in '{key}': {val}")
            # Boundary errors remain finite and bounded (not diverging).
            self.assertLess(float(np.linalg.norm(d["boundary_pos_error"])), 1.0)
            self.assertLess(float(np.linalg.norm(d["boundary_vel_error_pre"])), 1.0e2)

    def test_impulse_is_nontrivial_and_two_way(self):
        """The coupling is genuinely two-way: a real impulse drags the base."""
        toy = self._make_toy()
        base = toy.maximal_base_body
        v_before = np.linalg.norm(toy.maximal_state_0.body_qd.numpy()[base])
        history = toy.run(num_steps=12, verbose=False)
        v_after = np.linalg.norm(toy.maximal_state_0.body_qd.numpy()[base])

        # A non-zero boundary impulse was computed (not a no-op / one-way copy).
        max_lambda = max(float(np.linalg.norm(d["boundary_impulse"])) for d in history)
        self.assertGreater(max_lambda, 1.0e-3)

        # The four-bar base, initially (nearly) at rest, is accelerated by the
        # reaction -J_m^T lambda fed back from the arm.
        self.assertGreater(v_after, v_before + 1.0e-3)

    def test_velocity_error_is_controlled(self):
        """The post-correction boundary velocity error does not grow without bound."""
        toy = self._make_toy()
        history = toy.run(num_steps=20, verbose=False)
        errs = [float(np.linalg.norm(d["boundary_vel_error_post"])) for d in history]
        # The staggered scheme with an approximate maximal inertia does not drive
        # the error to zero, but it must stay bounded (no blow-up).
        self.assertLess(max(errs), 5.0)
        self.assertLess(errs[-1], 5.0)

    def test_impulse_shape_independent_of_steps(self):
        """boundary_impulse is always a 6-vector regardless of when it is read."""
        toy = self._make_toy(baumgarte=0.0)  # pure velocity-level coupling
        toy.step()
        self.assertEqual(toy.solver.boundary_impulse.shape, (6,))
        self.assertTrue(np.all(np.isfinite(toy.solver.boundary_impulse)))

    def test_existing_solvers_untouched(self):
        """The new solver does not perturb the public Featherstone path."""
        device = _cuda_device()
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        link = builder.add_link(mass=1.0)
        builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        builder.add_articulation([joint], label="pendulum")
        model = builder.finalize(device=device)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = newton.solvers.SolverFeatherstone(model)
        qd = state_0.joint_qd.numpy()
        qd[0] = 1.0
        state_0.joint_qd.assign(qd)
        for _ in range(5):
            solver.step(state_0, state_1, control, None, 0.01)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.all(np.isfinite(state_0.joint_qd.numpy())))


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
