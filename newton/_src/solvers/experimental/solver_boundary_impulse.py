# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Boundary-impulse coupling of a reduced- and a maximal-coordinate subsystem.

This is an experimental proof-of-concept. See
``docs/experimental_boundary_impulse.md`` for the full design note and the
mathematical derivation. The short version:

A reduced-coordinate subsystem (e.g. a serial arm under
:class:`~newton.solvers.SolverFeatherstone`) and a maximal-coordinate subsystem
(e.g. a closed-chain mechanism under :class:`~newton.solvers.SolverKamino`) are
welded together at a single rigid 6D boundary. Each subsystem is advanced by its
own solver; the two are coupled by a boundary impulse ``lambda`` obtained from
the saddle-point (KKT) system::

    [ M_r    0    -J_r^T ] [ dv     ]   [ 0   ]
    [ 0      M_m   J_m^T ] [ dV     ] = [ 0   ]
    [ J_r   -J_m   0     ] [ lambda ]   [ r_c ]

which collapses (Schur complement) to the 6x6 solve::

    A lambda = r_c ,   A = J_r M_r^-1 J_r^T + J_m M_m^-1 J_m^T

The reaction ``+J_r^T lambda`` is fed back to the reduced side and
``-J_m^T lambda`` to the maximal side, so the coupling conserves momentum across
the seam (Newton's third law) rather than one-way copying the end-effector pose
onto the mechanism base.
"""

from __future__ import annotations

import numpy as np

from ...sim import Contacts, Control, State, eval_fk, eval_jacobian, eval_mass_matrix
from ..solver import SolverBase

__all__ = ["SolverBoundaryImpulse"]


# ---------------------------------------------------------------------------
# Small host-side helpers. The boundary system is tiny (6x6); we assemble and
# solve it in NumPy for readability rather than in Warp kernels. Performance is
# explicitly not a goal of this proof-of-concept.
# ---------------------------------------------------------------------------


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a Warp-convention quaternion ``(x, y, z, w)``."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1.0e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix such that ``skew(v) @ w == cross(v, w)``."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (``so(3)`` log) of a rotation matrix, in world axes."""
    # Robust extraction of the rotation vector from a rotation matrix.
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1.0e-7:
        # Small angle: vee(R - I) is accurate and avoids the 1/sin(theta) blow-up.
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (theta / (2.0 * np.sin(theta)))


def _transform_point(xform: np.ndarray, local: np.ndarray) -> np.ndarray:
    """World position of a body-local point given a ``(p, quat)`` transform (7,)."""
    p = xform[0:3]
    R = _quat_to_matrix(xform[3:7])
    return p + R @ local


def _point_transport(r: np.ndarray) -> np.ndarray:
    """6x6 twist transport from a body COM to a point offset ``r`` (world).

    Maps a COM twist ``(v_com, omega)`` to the velocity of the material point at
    ``r = p_point - p_com``: ``v_point = v_com + omega x r``. In Newton's public
    ``(linear, angular)`` ordering this is ``[[I, -skew(r)], [0, I]]``. Its
    transpose performs the dual wrench transport (boundary -> COM), so building
    the boundary Jacobians with this factor makes ``J^T`` carry the moment-arm
    shift automatically.
    """
    J = np.eye(6)
    J[0:3, 3:6] = -_skew(r)
    return J


class SolverBoundaryImpulse(SolverBase):
    """Couple a reduced and a maximal subsystem through a rigid 6D boundary impulse.

    .. experimental::

        ``SolverBoundaryImpulse`` is a proof-of-concept. Its API, math, defaults,
        and even its file location may change without prior notice.

    The solver owns two child solvers (one per subsystem) and their two
    :class:`~newton.Model` objects. Each :meth:`step` advances both subsystems
    and exchanges a boundary impulse so that the reduced end-effector body and
    the maximal base body stay rigidly welded (velocity-level constraint, with
    optional Baumgarte position stabilization).

    Args:
        reduced_solver: Solver integrating the reduced-coordinate subsystem
            (e.g. :class:`~newton.solvers.SolverFeatherstone`). Its
            :attr:`~newton.solvers.SolverBase.model` must expose generalized
            coordinates so that :func:`~newton.eval_jacobian` /
            :func:`~newton.eval_mass_matrix` apply.
        maximal_solver: Solver integrating the maximal-coordinate subsystem
            (e.g. :class:`~newton.solvers.SolverKamino`). It must consume
            ``State.body_f`` as an external per-body wrench.
        reduced_ee_body: Global body index of the reduced end-effector that is
            welded to the maximal base.
        maximal_base_body: Global body index of the maximal base body that is
            welded to the reduced end-effector.
        config: Optional :class:`Config`. Defaults are used when omitted.
    """

    class Config:
        """Tunables for the boundary coupling.

        Args:
            baumgarte: Position-stabilization coefficient (dimensionless). The
                target boundary velocity is biased by ``(baumgarte / dt) * pose_error``
                so the weld pose error decays over steps. ``0`` gives a pure
                velocity-level constraint.
            regularization: Tikhonov term added to the boundary inverse-inertia
                ``A`` for numerical robustness near singular configurations
                [kg^-1-ish; small].
            max_velocity_error: If the pre-correction boundary velocity error
                norm exceeds this, the impulse is clamped to avoid blow-up in
                pathological steps. ``None`` disables clamping.
        """

        def __init__(
            self,
            baumgarte: float = 0.2,
            regularization: float = 1.0e-6,
            max_velocity_error: float | None = 1.0e3,
        ):
            self.baumgarte = baumgarte
            self.regularization = regularization
            self.max_velocity_error = max_velocity_error

    def __init__(
        self,
        reduced_solver: SolverBase,
        maximal_solver: SolverBase,
        reduced_ee_body: int,
        maximal_base_body: int,
        config: Config | None = None,
    ):
        # The base class only needs *a* model for .device etc.; the reduced model
        # is the natural choice. We keep explicit references to both subsystems.
        super().__init__(reduced_solver.model)

        self.reduced_solver = reduced_solver
        self.maximal_solver = maximal_solver
        self.reduced_model = reduced_solver.model
        self.maximal_model = maximal_solver.model
        self.reduced_ee_body = int(reduced_ee_body)
        self.maximal_base_body = int(maximal_base_body)
        self.config = config or SolverBoundaryImpulse.Config()

        # Locate the reduced end-effector within the articulation Jacobian:
        # eval_jacobian rows are blocked by joint-within-articulation index, and
        # joint k's child body is row block [6k : 6k+6].
        self._reduced_art, self._reduced_ee_row = self._locate_ee_rows()
        # Reduced articulation DOF slice (columns of J / rows&cols of M).
        qd_start = self.reduced_model.joint_qd_start.numpy()
        art_start = self.reduced_model.articulation_start.numpy()
        a = self._reduced_art
        self._reduced_dof0 = int(qd_start[int(art_start[a])])
        self._reduced_dof1 = int(qd_start[int(art_start[a + 1])])
        self._reduced_ndof = self._reduced_dof1 - self._reduced_dof0

        # Cached free-body inertia of the maximal base body (body-frame, about
        # COM). M_m is intentionally the *base body's free inertia*, not the
        # rigid-composite of the whole subsystem: a closed-chain mechanism's
        # links move relative to the base, so a composite-rigid estimate
        # OVER-states the effective inertia and makes the staggered impulse
        # overshoot and diverge. Under-estimating (base-only) under-corrects,
        # which is the stable failure mode. Choose a base body that carries
        # meaningful mass (e.g. the gripper's main `base` link, not a massless
        # mounting frame). See the design note.
        self._base_inv_mass = float(self.maximal_model.body_inv_mass.numpy()[self.maximal_base_body])
        self._base_inv_inertia_local = np.array(
            self.maximal_model.body_inv_inertia.numpy()[self.maximal_base_body], dtype=np.float64
        ).reshape(3, 3)
        self._base_com_local = np.array(self.maximal_model.body_com.numpy()[self.maximal_base_body], dtype=np.float64)
        self._ee_com_local = np.array(self.reduced_model.body_com.numpy()[self.reduced_ee_body], dtype=np.float64)

        # Rigid weld offset, captured lazily on the first step from the initial
        # relative configuration of the two subsystems (so they need not be
        # co-located by the caller). The boundary point is the maximal base COM;
        # we store that point in the end-effector frame at weld time.
        self._weld_offset_in_ee: np.ndarray | None = None  # base-COM in EE frame
        self._weld_rot_rel: np.ndarray | None = None  # R_ee^T @ R_base at weld time

        # Diagnostics from the most recent step (read by examples/tests).
        self.boundary_pose_error: np.ndarray | None = None  # (6,) (lin, ang) world
        self.boundary_velocity_error_pre: np.ndarray | None = None  # (6,) before correction
        self.boundary_velocity_error_post: np.ndarray | None = None  # (6,) after correction
        self.boundary_impulse: np.ndarray | None = None  # (6,) lambda (lin, ang) at boundary point
        self.boundary_twist_reduced: np.ndarray | None = None  # (6,) V_e before correction
        self.boundary_twist_maximal: np.ndarray | None = None  # (6,) V_b before maximal advance

    # -- setup helpers ------------------------------------------------------

    def _locate_ee_rows(self) -> tuple[int, int]:
        """Return ``(articulation_index, jacobian_row_offset)`` for the EE body."""
        joint_child = self.reduced_model.joint_child.numpy()
        joint_art = self.reduced_model.joint_articulation.numpy()
        art_start = self.reduced_model.articulation_start.numpy()
        matches = np.where(joint_child == self.reduced_ee_body)[0]
        if len(matches) == 0:
            raise ValueError(
                f"reduced_ee_body {self.reduced_ee_body} is not the child of any joint; it must be an articulated link."
            )
        j = int(matches[0])
        a = int(joint_art[j])
        if a < 0:
            raise ValueError(f"reduced_ee_body {self.reduced_ee_body} is not part of an articulation.")
        local = j - int(art_start[a])
        return a, 6 * local

    # -- the coupled step ---------------------------------------------------

    def step(
        self,
        reduced_state_in: State,
        reduced_state_out: State,
        maximal_state_in: State,
        maximal_state_out: State,
        reduced_control: Control | None,
        maximal_control: Control | None,
        dt: float,
        maximal_contacts: Contacts | None = None,
    ) -> None:
        """Advance both subsystems one step and exchange the boundary impulse.

        The experimental multi-state signature is intentional: the two
        subsystems own separate :class:`~newton.State` objects. See the design
        note for the per-step sequence; the short version is advance-reduced,
        solve-boundary-impulse, apply-to-both, advance-maximal.

        Args:
            reduced_state_in: Input state of the reduced subsystem.
            reduced_state_out: Output state of the reduced subsystem.
            maximal_state_in: Input state of the maximal subsystem. The boundary
                wrench is accumulated into its ``body_f`` before the maximal
                advance; call ``maximal_state_in.clear_forces()`` beforehand if
                you want only the boundary wrench applied.
            maximal_state_out: Output state of the maximal subsystem.
            reduced_control: Control for the reduced subsystem (may be ``None``).
            maximal_control: Control for the maximal subsystem (may be ``None``).
            dt: Time step [s].
            maximal_contacts: Optional contacts for the maximal subsystem.
        """
        # 1. Advance the reduced subsystem unconstrained by the boundary. After
        #    this, reduced_state_out carries q+, v* and the (public-convention)
        #    body_q / body_qd of the end-effector.
        self.reduced_solver.step(reduced_state_in, reduced_state_out, reduced_control, None, dt)

        # 2. Reduced-side boundary quantities. eval_* read body_q/joint_q, which
        #    the reduced solver already refreshed; we re-run FK defensively in
        #    case a child solver does not populate body_qd to the public contract.
        eval_fk(self.reduced_model, reduced_state_out.joint_q, reduced_state_out.joint_qd, reduced_state_out)
        J_full = eval_jacobian(self.reduced_model, reduced_state_out).numpy()
        H_full = eval_mass_matrix(self.reduced_model, reduced_state_out).numpy()
        a = self._reduced_art
        r0 = self._reduced_ee_row
        ndof = self._reduced_ndof
        J_r_com = J_full[a, r0 : r0 + 6, 0:ndof].astype(np.float64)  # (6, ndof) at the EE COM
        M_r = H_full[a, 0:ndof, 0:ndof].astype(np.float64)  # (ndof, ndof)

        v_r = reduced_state_out.joint_qd.numpy()[self._reduced_dof0 : self._reduced_dof1].astype(np.float64)
        ee_q = reduced_state_out.body_q.numpy()[self.reduced_ee_body].astype(np.float64)
        p_ee_com = _transform_point(ee_q, self._ee_com_local)
        R_ee = _quat_to_matrix(ee_q[3:7])

        # 3. Maximal-side boundary quantities, read from the *input* (pre-advance)
        #    state. We anchor the boundary frame at the maximal base COM, so the
        #    maximal Jacobian is the identity selection (J_m = I). This is
        #    deliberate: it keeps the base's (often tiny, e.g. a thin link)
        #    off-axis rotational inertia from being amplified by an offset moment
        #    arm into a spurious spin; the well-conditioned reduced arm carries
        #    the moment-arm transport instead.
        base_q = maximal_state_in.body_q.numpy()[self.maximal_base_body].astype(np.float64)
        base_qd = maximal_state_in.body_qd.numpy()[self.maximal_base_body].astype(np.float64)
        p_base_com = _transform_point(base_q, self._base_com_local)  # boundary point (world)
        R_base = _quat_to_matrix(base_q[3:7])

        # Transport the reduced Jacobian from the EE COM to the boundary point.
        J_r = _point_transport(p_base_com - p_ee_com) @ J_r_com  # (6, ndof)
        V_e = J_r @ v_r  # reduced boundary twist at the base COM

        J_m = np.eye(6)
        V_b = base_qd.copy()  # maximal boundary twist (boundary point == base COM)

        # Maximal inverse spatial inertia at COM (world frame), block-diagonal.
        # Free-body inertia of the base (an intentional under-estimate; see the
        # cached-data comment in __init__ and the design note).
        Minv_m = np.zeros((6, 6))
        Minv_m[0:3, 0:3] = self._base_inv_mass * np.eye(3)
        Minv_m[3:6, 3:6] = R_base @ self._base_inv_inertia_local @ R_base.T

        # Capture the rigid weld on the first step: the base COM, expressed in the
        # current end-effector frame (so the two subsystems need not be co-located).
        if self._weld_offset_in_ee is None:
            self._weld_offset_in_ee = R_ee.T @ (p_base_com - p_ee_com)
            self._weld_rot_rel = R_ee.T @ R_base

        # 4. Boundary pose error: deviation of the maximal base from where the
        #    reduced end-effector says it should be (world axes, at the base COM).
        desired_p_base = p_ee_com + R_ee @ self._weld_offset_in_ee
        x_err = p_base_com - desired_p_base
        R_des_base = R_ee @ self._weld_rot_rel
        rot_err = _rotation_log(R_base @ R_des_base.T)
        e_pose = np.concatenate([x_err, rot_err])  # (6,) base deviation, (lin, ang)

        # 5. Solve the boundary impulse (Schur complement of the KKT system).
        #    With e_pose the base-minus-desired deviation, d/dt(e_pose) ~ V_b - V_e,
        #    so the Baumgarte target on C = V_e - V_b is +(beta/dt) * e_pose.
        c0 = V_e - V_b  # current boundary velocity mismatch
        beta = self.config.baumgarte
        c_target = (beta / dt) * e_pose if beta != 0.0 else np.zeros(6)
        r_c = c_target - c0

        Minv_r = np.linalg.inv(M_r)
        A = J_r @ Minv_r @ J_r.T + J_m @ Minv_m @ J_m.T
        A += self.config.regularization * np.eye(6)
        lam = np.linalg.solve(A, r_c)

        # Optional safety clamp for pathological steps.
        if self.config.max_velocity_error is not None:
            err_norm = float(np.linalg.norm(c0))
            if err_norm > self.config.max_velocity_error:
                lam *= self.config.max_velocity_error / err_norm

        # 6. Apply +J_r^T lambda to the reduced side as a velocity correction
        #    dv = M_r^-1 J_r^T lambda (the reduced solver already advanced, so the
        #    impulse is applied as a velocity projection on the output state).
        tau_boundary = J_r.T @ lam  # generalized impulse on the arm DOFs
        dv = Minv_r @ tau_boundary
        qd_out = reduced_state_out.joint_qd.numpy()
        qd_out[self._reduced_dof0 : self._reduced_dof1] += dv.astype(qd_out.dtype)
        reduced_state_out.joint_qd.assign(qd_out)
        # Refresh body_qd so post-step reads see the corrected end-effector twist.
        eval_fk(self.reduced_model, reduced_state_out.joint_q, reduced_state_out.joint_qd, reduced_state_out)

        # 7. Apply -J_m^T lambda to the maximal side as an external wrench at the
        #    base COM, integrated by the maximal solver's own constraint solve.
        #    J_m^T performs the boundary -> COM moment transport automatically.
        f_boundary = -(J_m.T @ lam)  # spatial impulse at base COM (lin, ang), world
        wrench = f_boundary / dt  # convert impulse to a force over the step
        bf = maximal_state_in.body_f.numpy()
        bf[self.maximal_base_body] += wrench.astype(bf.dtype)
        maximal_state_in.body_f.assign(bf)
        self.maximal_solver.step(maximal_state_in, maximal_state_out, maximal_control, maximal_contacts, dt)

        # 8. Diagnostics.
        v_r_corr = v_r + dv
        V_e_post = J_r @ v_r_corr
        # Boundary point is the base COM, so V_b_post is just the base twist.
        V_b_post = maximal_state_out.body_qd.numpy()[self.maximal_base_body].astype(np.float64)
        self.boundary_pose_error = e_pose
        self.boundary_velocity_error_pre = c0
        self.boundary_velocity_error_post = V_e_post - V_b_post
        self.boundary_impulse = lam
        self.boundary_twist_reduced = V_e
        self.boundary_twist_maximal = V_b

    def notify_model_changed(self, flags: int) -> None:
        """Forward change notifications to both child solvers."""
        self.reduced_solver.notify_model_changed(flags)
        self.maximal_solver.notify_model_changed(flags)
