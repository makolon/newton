# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Toy example for SolverBoundaryImpulse (experimental)
#
# Couples a tiny REDUCED-coordinate serial arm (2R, SolverFeatherstone) to a
# tiny MAXIMAL-coordinate closed chain (four-bar linkage, SolverKamino) through
# a single rigid 6D boundary impulse. The arm end-effector is welded to the
# four-bar base body; the arm is given an initial joint velocity and drags the
# four-bar along through the boundary impulse (not by copying its pose).
#
# Command: python -m newton.examples experimental.example_boundary_impulse_toy
#   (or run this file directly)
#
# NOTE: SolverKamino currently requires a CUDA device, so this toy does too.
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
from newton.tests.utils import basics


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if wp.get_cuda_device_count() > 0:
        return "cuda:0"
    return "cpu"


class BoundaryImpulseToy:
    """Builds the coupled arm + four-bar toy and steps it.

    The class is deliberately viewer-free so it can be driven head-less from a
    unit test; ``main`` adds the command-line/printing wrapper.
    """

    def __init__(self, device: str | None = None, baumgarte: float = 0.2, dt: float = 2.5e-3):
        self.device = _pick_device(device)
        self.dt = dt

        # ---- Reduced subsystem: 2R planar arm under SolverFeatherstone -----
        # Gravity off so the demonstration is a clean horizontal momentum
        # transfer driven purely by the initial arm motion + boundary impulse.
        arm = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        link1 = arm.add_link(mass=1.0)
        link2 = arm.add_link(mass=1.0)
        arm.add_shape_box(link1, hx=0.25, hy=0.05, hz=0.05)
        arm.add_shape_box(link2, hx=0.25, hy=0.05, hz=0.05)
        # Revolute about Y so the arm sweeps in the X-Z plane, sharing the
        # four-bar linkage's plane of motion (its hinges are about Y too). This
        # keeps the rigid weld in-plane and avoids exciting the four-bar base's
        # tiny out-of-plane rotational inertia.
        j1 = arm.add_joint_revolute(
            parent=-1,
            child=link1,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        )
        j2 = arm.add_joint_revolute(
            parent=link1,
            child=link2,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        )
        arm.add_articulation([j1, j2], label="arm")
        self.reduced_model = arm.finalize(device=self.device)
        self.reduced_ee_body = link2  # end-effector welded to the four-bar base

        self.reduced_solver = newton.solvers.SolverFeatherstone(self.reduced_model)
        self.reduced_state_0 = self.reduced_model.state()
        self.reduced_state_1 = self.reduced_model.state()
        self.reduced_control = self.reduced_model.control()

        # Initial arm motion: both joints spinning, so the EE sweeps and drags
        # the four-bar through the weld.
        qd = self.reduced_state_0.joint_qd.numpy()
        qd[:] = np.array([1.5, 0.8], dtype=qd.dtype)
        self.reduced_state_0.joint_qd.assign(qd)
        newton.eval_fk(
            self.reduced_model, self.reduced_state_0.joint_q, self.reduced_state_0.joint_qd, self.reduced_state_0
        )

        # ---- Maximal subsystem: four-bar linkage under SolverKamino --------
        rb = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverKamino.register_custom_attributes(rb)
        rb.default_shape_cfg.margin = 0.0
        rb.default_shape_cfg.gap = 0.0
        basics.build_boxes_fourbar(builder=rb, ground=False)
        wrapper = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        wrapper.add_world(rb)
        self.maximal_model = wrapper.finalize(skip_validation_joints=True, device=self.device)
        # Soft joint PD gains, matching the stock four-bar example.
        self.maximal_model.joint_target_ke.fill_(1.0)
        self.maximal_model.joint_target_kd.fill_(0.001)
        self.maximal_base_body = 0  # link_1 is added first -> body index 0

        cfg = newton.solvers.SolverKamino.Config.from_model(self.maximal_model)
        cfg.use_collision_detector = False  # toy: no contacts inside the mechanism
        cfg.use_fk_solver = True
        self.maximal_solver = newton.solvers.SolverKamino(model=self.maximal_model, config=cfg)
        self.maximal_state_0 = self.maximal_model.state()
        self.maximal_state_1 = self.maximal_model.state()
        self.maximal_control = self.maximal_model.control()

        # Warm-start Kamino, then place the four-bar base at the arm's
        # end-effector so the two subsystems start (nearly) coincident.
        self.maximal_solver.step(self.maximal_state_0, self.maximal_state_1, self.maximal_control, None, self.dt)
        self.maximal_solver.reset(self.maximal_state_0)
        ee_pos = self.reduced_state_0.body_q.numpy()[self.reduced_ee_body][0:3]
        base_q = wp.array(
            [wp.transform(wp.vec3(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])), wp.quat_identity())],
            dtype=wp.transformf,
            device=self.device,
        )
        self.maximal_solver.reset(state=self.maximal_state_0, base_q=base_q)

        # ---- The coupling solver -------------------------------------------
        self.solver = newton.solvers.SolverBoundaryImpulse(
            reduced_solver=self.reduced_solver,
            maximal_solver=self.maximal_solver,
            reduced_ee_body=self.reduced_ee_body,
            maximal_base_body=self.maximal_base_body,
            config=newton.solvers.SolverBoundaryImpulse.Config(baumgarte=baumgarte),
        )

    def step(self) -> dict:
        """Advance one coupled step and return a diagnostics dict."""
        # The boundary wrench is accumulated into the maximal input body_f, so
        # clear stale forces first.
        self.maximal_state_0.clear_forces()
        self.solver.step(
            self.reduced_state_0,
            self.reduced_state_1,
            self.maximal_state_0,
            self.maximal_state_1,
            self.reduced_control,
            self.maximal_control,
            self.dt,
        )
        self.reduced_state_0, self.reduced_state_1 = self.reduced_state_1, self.reduced_state_0
        self.maximal_state_0, self.maximal_state_1 = self.maximal_state_1, self.maximal_state_0

        return {
            "boundary_pos_error": self.solver.boundary_pose_error.copy(),
            "boundary_vel_error_pre": self.solver.boundary_velocity_error_pre.copy(),
            "boundary_vel_error_post": self.solver.boundary_velocity_error_post.copy(),
            "boundary_impulse": self.solver.boundary_impulse.copy(),
            "arm_joint_q": self.reduced_state_0.joint_q.numpy().copy(),
            "arm_joint_qd": self.reduced_state_0.joint_qd.numpy().copy(),
            "base_pose": self.maximal_state_0.body_q.numpy()[self.maximal_base_body].copy(),
            "base_twist": self.maximal_state_0.body_qd.numpy()[self.maximal_base_body].copy(),
        }

    def run(self, num_steps: int = 60, verbose: bool = True) -> list[dict]:
        history = []
        for k in range(num_steps):
            d = self.step()
            history.append(d)
            if verbose:
                print(
                    f"step {k:3d} | "
                    f"|pose_err|={np.linalg.norm(d['boundary_pos_error']):.3e} "
                    f"|vel_err_pre|={np.linalg.norm(d['boundary_vel_error_pre']):.3e} "
                    f"|vel_err_post|={np.linalg.norm(d['boundary_vel_error_post']):.3e} "
                    f"|lambda|={np.linalg.norm(d['boundary_impulse']):.3e} | "
                    f"q={np.array2string(d['arm_joint_q'], precision=3)} "
                    f"base_p={np.array2string(d['base_pose'][0:3], precision=3)} "
                    f"base_v={np.array2string(d['base_twist'][0:3], precision=3)}"
                )
        return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None, help="Warp device (default: cuda:0 if available).")
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--baumgarte", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    toy = BoundaryImpulseToy(device=args.device, baumgarte=args.baumgarte)
    print(
        f"Coupled arm(2R, Featherstone) + four-bar(Kamino) on {toy.device}; "
        f"reduced EE body={toy.reduced_ee_body}, maximal base body={toy.maximal_base_body}"
    )
    history = toy.run(num_steps=args.num_steps, verbose=True)

    last = history[-1]
    print("\nFinal boundary impulse lambda (lin | ang):", last["boundary_impulse"])
    print("Final base twist (lin | ang):", last["base_twist"])
    print("Final arm joint state q / qd:", last["arm_joint_q"], "/", last["arm_joint_qd"])


if __name__ == "__main__":
    main()
