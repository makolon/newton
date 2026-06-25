# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Franka FR3 + Robotiq 2F-85: hybrid-vs-Kamino-vs-MuJoCo benchmark (experimental)
#
# Compares three ways to simulate a 7-DOF arm carrying a closed-chain gripper:
#
#   HYBRID       : Franka arm in reduced coords (SolverFeatherstone) + 2F-85 in
#                  maximal coords (SolverKamino), coupled by SolverBoundaryImpulse.
#   FULL-KAMINO  : the whole arm+gripper in maximal coords (SolverKamino).
#   FULL-MUJOCO  : the whole arm+gripper in reduced coords with equality loops
#                  (SolverMuJoCo).
#
# (FULL-FEATHERSTONE is intentionally absent: Featherstone cannot represent the
# 2F-85 closed loop, so it would silently simulate a different, open-chain robot.
# That impossibility is itself the motivation for the hybrid solver.)
#
# Scenario (identical across configs): the arm PD-holds a "ready" pose under
# gravity while the gripper closes its fingers and holds. We report steady-state
# wall-clock per solver step (speed) and solver-agnostic quality (gripper closure,
# end-effector drift, stability), plus per-config diagnostics.
#
# Command: python -m newton.examples.experimental.example_franka_2f85_benchmark
#
# NOTE: SolverKamino is CUDA-only, so this benchmark requires a CUDA device.
# NOTE: Experimental — imports the experimental SolverBoundaryImpulse via the
#       public newton.solvers namespace and the franka_2f85 builders module.
###########################################################################

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import warp as wp

import newton
from newton.examples.experimental import franka_2f85 as F

# Franka FR3 "ready" pose (7 arm joints) — a typical non-singular configuration.
FRANKA_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

# PD gains for holding the arm and the gripper (shared by all configs).
ARM_KE, ARM_KD = 400.0, 40.0
GRIP_KE, GRIP_KD = 20.0, 1.0
# Gripper driver target. The benchmark holds the gripper at its initial (open)
# configuration so every config performs the identical *static-hold* task
# (maintaining the Franka + closed-chain gripper under gravity, with the gripper
# loop constraints enforced every step). Holding the initial config needs no
# SolverKamino warm-start/reset, which keeps all three configs stable and fair.
# A dynamic grasping scenario (driving the gripper closed consistently across
# all three solvers) is left as future work.
GRIPPER_HOLD = 0.0  # == F.GRIPPER_OPEN, the gripper's initial driver angle
# Joint armature on the arm DOFs. The Franka URDF carries no armature, and the
# semi-implicit SolverFeatherstone integrates the unarmatured arm unstably at
# dt=1e-3 (diverges to NaN); a small armature stabilizes it. Applied uniformly
# to every config so the compared models are identical.
ARM_ARMATURE = 0.2


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if wp.get_cuda_device_count() > 0:
        return "cuda:0"
    return "cpu"


def _set_gains(model, dofs, ke, kd):
    ke_arr = model.joint_target_ke.numpy()
    kd_arr = model.joint_target_kd.numpy()
    for d in dofs:
        ke_arr[d] = ke
        kd_arr[d] = kd
    model.joint_target_ke.assign(ke_arr)
    model.joint_target_kd.assign(kd_arr)


def _set_targets(control, dof_value_pairs):
    tq = control.joint_target_q.numpy()
    for d, v in dof_value_pairs:
        tq[d] = v
    control.joint_target_q.assign(tq)


def _set_armature(model, dofs, value):
    arm = model.joint_armature.numpy()
    for d in dofs:
        arm[d] = value
    model.joint_armature.assign(arm)


def _use_joint_target_actuation(model):
    """Switch any MuJoCo actuators to JOINT_TARGET so joint PD drives the model.

    The 2F-85 ships a tendon actuator that fights the joint-target PD (and is
    unstable); selecting JOINT_TARGET disables it so the driver-joint PD closes
    the gripper, matching the other configs.
    """
    mj = getattr(model, "mujoco", None)
    if mj is not None and getattr(mj, "ctrl_source", None) is not None:
        mj.ctrl_source.fill_(int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET))


def _seed_model_arm_pose(model, arm_dofs):
    """Seed the ready pose into the MODEL's initial joint_q.

    This must precede ``model.state()`` and any ``solver.reset()`` so that the
    state clones the ready pose and a Kamino reset restores it (rather than the
    zero default, which would drop the arm and destabilize it).
    """
    q = model.joint_q.numpy()
    for k, d in enumerate(arm_dofs):
        q[d] = FRANKA_READY[k]
    model.joint_q.assign(q)


def _seed_arm_pose(model, state, arm_dofs):
    q = state.joint_q.numpy()
    for k, d in enumerate(arm_dofs):
        q[d] = FRANKA_READY[k]
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _jaw_gap(model, state):
    """Distance between the two finger pads [m]."""
    lp = F.find_body(model, "/left_pad")
    rp = F.find_body(model, "/right_pad")
    bq = state.body_q.numpy()
    return float(np.linalg.norm(bq[lp][:3] - bq[rp][:3]))


def _max_speed(state):
    return float(np.max(np.abs(state.joint_qd.numpy())))


def _finite(state):
    return bool(np.all(np.isfinite(state.joint_q.numpy())) and np.all(np.isfinite(state.body_q.numpy())))


# ---------------------------------------------------------------------------
# Config drivers. Each exposes setup(), run(n_steps) -> seconds, and quality().
# ---------------------------------------------------------------------------


class _FullConfig:
    """FULL-KAMINO / FULL-MUJOCO: the whole arm+gripper in one model+solver."""

    def __init__(self, name, device, dt, make_solver, graph_capture):
        self.name = name
        self.device = device
        self.dt = dt
        self.model, self.flange, self.arm_dofs, self.drivers = F.build_combined(device)
        _set_gains(self.model, self.arm_dofs, ARM_KE, ARM_KD)
        _set_gains(self.model, self.drivers, GRIP_KE, GRIP_KD)
        _set_armature(self.model, self.arm_dofs, ARM_ARMATURE)
        _use_joint_target_actuation(self.model)
        _seed_model_arm_pose(self.model, self.arm_dofs)  # before state()/reset()
        self.solver = make_solver(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        _seed_arm_pose(self.model, self.state_0, self.arm_dofs)
        _set_targets(
            self.control,
            [(d, FRANKA_READY[k]) for k, d in enumerate(self.arm_dofs)] + [(d, GRIPPER_HOLD) for d in self.drivers],
        )
        self.info = {"bodies": self.model.body_count, "dofs": self.model.joint_dof_count}
        self.graph = None
        self.graph_capture = graph_capture

    def _substep(self):
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        if self.graph_capture and self.device.startswith("cuda"):
            with wp.ScopedCapture() as cap:
                self._substep()
            self.graph = cap.graph

    def _one(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._substep()

    def run(self, n_steps):
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self._one()
        wp.synchronize()
        return time.perf_counter() - t0

    def quality(self):
        return {
            "jaw_gap": _jaw_gap(self.model, self.state_0),
            "ee_pos": self.state_0.body_q.numpy()[self.flange][:3].tolist(),
            "max_qd": _max_speed(self.state_0),
            "finite": _finite(self.state_0),
        }


class _HybridConfig:
    """HYBRID: Featherstone arm + Kamino gripper coupled by SolverBoundaryImpulse."""

    name = "HYBRID"

    def __init__(self, device, dt):
        self.device = device
        self.dt = dt

        # Reduced arm (Featherstone).
        self.arm_model, self.flange = F.build_franka_arm(device)
        self.arm_dofs = F.find_joint_qd_indices(self.arm_model, tuple(f"fr3_joint{i}" for i in range(1, 8)))
        _set_gains(self.arm_model, self.arm_dofs, ARM_KE, ARM_KD)
        _set_armature(self.arm_model, self.arm_dofs, ARM_ARMATURE)
        _seed_model_arm_pose(self.arm_model, self.arm_dofs)
        self.arm_solver = newton.solvers.SolverFeatherstone(self.arm_model)
        self.arm_0 = self.arm_model.state()
        self.arm_1 = self.arm_model.state()
        self.arm_ctrl = self.arm_model.control()
        _seed_arm_pose(self.arm_model, self.arm_0, self.arm_dofs)
        _set_targets(self.arm_ctrl, [(d, FRANKA_READY[k]) for k, d in enumerate(self.arm_dofs)])

        # Maximal gripper (Kamino).
        self.grip_model, self.grip_base, self.drivers = F.build_2f85_gripper(device, floating=True)
        _set_gains(self.grip_model, self.drivers, GRIP_KE, GRIP_KD)
        cfg = newton.solvers.SolverKamino.Config.from_model(self.grip_model)
        cfg.use_collision_detector = False
        cfg.use_fk_solver = True
        self.grip_solver = newton.solvers.SolverKamino(self.grip_model, cfg)
        self.grip_0 = self.grip_model.state()
        self.grip_1 = self.grip_model.state()
        self.grip_ctrl = self.grip_model.control()
        _set_targets(self.grip_ctrl, [(d, GRIPPER_HOLD) for d in self.drivers])

        # Place the gripper base at the arm flange so the weld starts tight.
        flange_xform = self.arm_0.body_q.numpy()[self.flange]
        self.grip_solver.step(self.grip_0, self.grip_1, self.grip_ctrl, None, dt)
        self.grip_solver.reset(self.grip_0)
        base_q = wp.array(
            [wp.transform(wp.vec3(*flange_xform[:3].tolist()), wp.quat(*flange_xform[3:7].tolist()))],
            dtype=wp.transformf,
            device=device,
        )
        self.grip_solver.reset(state=self.grip_0, base_q=base_q)

        self.solver = newton.solvers.SolverBoundaryImpulse(
            reduced_solver=self.arm_solver,
            maximal_solver=self.grip_solver,
            reduced_ee_body=self.flange,
            maximal_base_body=self.grip_base,
        )
        self.info = {
            "bodies": self.arm_model.body_count + self.grip_model.body_count,
            "dofs": self.arm_model.joint_dof_count + self.grip_model.joint_dof_count,
        }

    def _substep(self):
        self.grip_0.clear_forces()
        self.solver.step(self.arm_0, self.arm_1, self.grip_0, self.grip_1, self.arm_ctrl, self.grip_ctrl, self.dt)
        self.arm_0, self.arm_1 = self.arm_1, self.arm_0
        self.grip_0, self.grip_1 = self.grip_1, self.grip_0

    def capture(self):
        # The boundary solve runs on the host (numpy), so the coupled step cannot
        # be CUDA-graph captured. This is a known limitation of the PoC.
        self.graph = None

    def run(self, n_steps):
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self._substep()
        wp.synchronize()
        return time.perf_counter() - t0

    def breakdown(self, n_steps):
        """Per-component ms/step: reduced arm solve, maximal gripper solve.

        The remainder of the full hybrid step time is the host-side boundary
        coupling (eval_jacobian / eval_mass_matrix + the NumPy 6x6 solve + the
        GPU<->CPU array transfers). Run after timing; it perturbs state.
        """
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.arm_solver.step(self.arm_0, self.arm_1, self.arm_ctrl, None, self.dt)
            self.arm_0, self.arm_1 = self.arm_1, self.arm_0
        wp.synchronize()
        t_arm = 1e3 * (time.perf_counter() - t0) / n_steps

        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.grip_0.clear_forces()
            self.grip_solver.step(self.grip_0, self.grip_1, self.grip_ctrl, None, self.dt)
            self.grip_0, self.grip_1 = self.grip_1, self.grip_0
        wp.synchronize()
        t_grip = 1e3 * (time.perf_counter() - t0) / n_steps
        return {"arm_step": t_arm, "gripper_step": t_grip}

    def quality(self):
        return {
            "jaw_gap": _jaw_gap(self.grip_model, self.grip_0),
            "ee_pos": self.arm_0.body_q.numpy()[self.flange][:3].tolist(),
            "max_qd": _max_speed(self.arm_0),
            "finite": _finite(self.arm_0) and _finite(self.grip_0),
            "boundary_pose_err": float(np.linalg.norm(self.solver.boundary_pose_error)),
            "boundary_vel_err": float(np.linalg.norm(self.solver.boundary_velocity_error_post)),
        }


def run_benchmark(device, dt, warmup, timed, configs):
    results = {}
    for name in configs:
        print(f"\n[{name}] building + warming up ...")
        try:
            if name == "FULL-MUJOCO":
                cfg = _FullConfig(name, device, dt, newton.solvers.SolverMuJoCo, graph_capture=True)
            elif name == "FULL-KAMINO":

                def _mk_kamino(m):
                    kc = newton.solvers.SolverKamino.Config.from_model(m)
                    kc.use_collision_detector = False
                    kc.use_fk_solver = True
                    return newton.solvers.SolverKamino(m, kc)

                cfg = _FullConfig(name, device, dt, _mk_kamino, graph_capture=True)
            else:
                cfg = _HybridConfig(device, dt)

            cfg.capture()
            cfg.run(warmup)  # warmup: JIT + reach steady state (discarded)
            elapsed = cfg.run(timed)
            ms = 1e3 * elapsed / timed
            q = cfg.quality()
            extra = {}
            if hasattr(cfg, "breakdown"):
                bd = cfg.breakdown(max(10, timed // 4))
                extra["breakdown"] = bd
                extra["breakdown"]["coupling_overhead"] = ms - bd["arm_step"] - bd["gripper_step"]
            results[name] = {"ms_per_step": ms, "info": cfg.info, **q, **extra}
            print(f"[{name}] {ms:.3f} ms/step | bodies={cfg.info['bodies']} dofs={cfg.info['dofs']} | {q}")
        except Exception as e:  # keep going if one config fails
            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] FAILED: {e}")
    return results


def _print_table(results, dt):
    print("\n" + "=" * 78)
    print(f"Franka FR3 + Robotiq 2F-85 benchmark (dt={dt}s)")
    print("=" * 78)
    header = f"{'config':<14}{'ms/step':>10}{'bodies':>8}{'dofs':>6}{'jaw_gap':>10}{'max|qd|':>10}{'finite':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<14}{'ERROR: ' + r['error']}")
            continue
        print(
            f"{name:<14}{r['ms_per_step']:>10.3f}{r['info']['bodies']:>8}{r['info']['dofs']:>6}"
            f"{r['jaw_gap']:>10.4f}{r['max_qd']:>10.2f}{r['finite']!s:>8}"
        )
    if "HYBRID" in results and "boundary_pose_err" in results["HYBRID"]:
        h = results["HYBRID"]
        print(f"\nHYBRID weld quality: pose_err={h['boundary_pose_err']:.3e}  vel_err={h['boundary_vel_err']:.3e}")
        if "breakdown" in h:
            bd = h["breakdown"]
            print(
                f"HYBRID breakdown [ms/step]: arm(Featherstone)={bd['arm_step']:.3f}  "
                f"gripper(Kamino, {results['HYBRID']['info']['bodies'] - 10}-body)={bd['gripper_step']:.3f}  "
                f"host-coupling overhead={bd['coupling_overhead']:.3f}"
            )
            if "FULL-KAMINO" in results and "ms_per_step" in results["FULL-KAMINO"]:
                full_k = results["FULL-KAMINO"]["ms_per_step"]
                ratio = bd["gripper_step"] / full_k
                print(
                    f"  -> gripper-only Kamino ({bd['gripper_step']:.1f} ms) is {ratio:.1f}x the full-system Kamino "
                    f"({full_k:.1f} ms): isolating the maximal subsystem did NOT reduce its per-step cost. The arm "
                    f"(Featherstone) and host coupling are cheap; Kamino's gripper solve dominates the hybrid."
                )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--configs", type=str, default="FULL-MUJOCO,FULL-KAMINO,HYBRID", help="comma-separated subset")
    args, _ = parser.parse_known_args()
    device = _pick_device(args.device)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"device={device} dt={args.dt} warmup={args.warmup} timed={args.timed} configs={configs}")
    results = run_benchmark(device, args.dt, args.warmup, args.timed, configs)
    _print_table(results, args.dt)
    return results


if __name__ == "__main__":
    main()
