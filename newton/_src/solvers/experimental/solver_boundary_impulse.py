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

The boundary system is tiny (a 6x6 Schur solve plus an ``ndof`` Cholesky of the
reduced mass matrix). It is assembled and solved **on-device** in a single
Warp kernel so the whole coupled :meth:`SolverBoundaryImpulse.step` contains no
host round-trips and can be captured into a CUDA graph alongside the two child
solvers -- the only way the staggered scheme is competitive, since an
un-captured step forces each child solver onto its slow host-synchronized
iteration path.
"""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Control, State, eval_fk, eval_jacobian, eval_mass_matrix
from ..solver import SolverBase

__all__ = ["SolverBoundaryImpulse"]


# ---------------------------------------------------------------------------
# Device-side dense linear algebra. The boundary solve needs a small symmetric
# positive-definite Cholesky factorization (the reduced mass matrix, ndof x ndof)
# and a 6x6 Schur solve. Both run in a single thread inside the boundary kernel;
# the problem is microscopic, so a serial host-free solve is more than fast
# enough and -- crucially -- keeps the step graph-capturable.
# ---------------------------------------------------------------------------


@wp.func
def _chol_factor(n: int, A: wp.array2d[float], L: wp.array2d[float]):
    """Lower-triangular Cholesky factor ``L`` of an SPD matrix ``A`` (``A = L Lᵀ``).

    Only the lower triangle of ``L`` is written/read; ``A`` may be larger than
    ``n`` (only its leading ``n x n`` block is used).
    """
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        s = wp.sqrt(s)
        L[j, j] = s
        inv = 1.0 / s
        for i in range(j + 1, n):
            t = A[i, j]
            for k in range(j):
                t = t - L[i, k] * L[j, k]
            L[i, j] = t * inv


@wp.func
def _chol_solve(n: int, L: wp.array2d[float], b: wp.array[float], x: wp.array[float]):
    """Solve ``L Lᵀ x = b`` in place-safe form (``x`` may alias ``b``)."""
    # Forward substitution: L y = b.
    for i in range(n):
        s = b[i]
        for k in range(i):
            s = s - L[i, k] * x[k]
        x[i] = s / L[i, i]
    # Back substitution: Lᵀ x = y.
    for ii in range(n):
        i = n - 1 - ii
        s = x[i]
        for k in range(i + 1, n):
            s = s - L[k, i] * x[k]
        x[i] = s / L[i, i]


@wp.func
def _rotation_log(R: wp.mat33) -> wp.vec3:
    """Axis-angle vector (``so(3)`` log) of a rotation matrix, in world axes."""
    cos_theta = wp.clamp((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5, -1.0, 1.0)
    theta = wp.acos(cos_theta)
    axis = wp.vec3(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    if theta < 1.0e-7:
        # Small angle: vee(R - I) is accurate and avoids the 1/sin(theta) blow-up.
        return axis * 0.5
    return axis * (theta / (2.0 * wp.sin(theta)))


@wp.kernel
def _capture_weld_kernel(
    ee: int,
    base: int,
    ee_com_local: wp.vec3,
    base_com_local: wp.vec3,
    body_q_red: wp.array[wp.transform],
    body_q_max: wp.array[wp.transform],
    # outputs
    weld_offset: wp.array[wp.vec3],  # base-COM in EE frame
    weld_rot: wp.array[wp.mat33],  # R_ee^T @ R_base
):
    """Capture the rigid weld offset from the current relative configuration."""
    ee_tf = body_q_red[ee]
    R_ee = wp.quat_to_matrix(wp.transform_get_rotation(ee_tf))
    p_ee_com = wp.transform_point(ee_tf, ee_com_local)

    base_tf = body_q_max[base]
    R_base = wp.quat_to_matrix(wp.transform_get_rotation(base_tf))
    p_base_com = wp.transform_point(base_tf, base_com_local)

    weld_offset[0] = wp.transpose(R_ee) * (p_base_com - p_ee_com)
    weld_rot[0] = wp.transpose(R_ee) * R_base


@wp.kernel
def _boundary_solve_kernel(
    # layout / config
    r0: int,
    dof0: int,
    ndof: int,
    ee: int,
    base: int,
    dt: float,
    beta: float,
    reg: float,
    maxvel: float,
    base_inv_mass: float,
    ee_com_local: wp.vec3,
    base_com_local: wp.vec3,
    base_inv_inertia_local: wp.mat33,
    # reduced-side device state
    J_slice: wp.array2d[float],  # articulation Jacobian rows [6*joints, dofs]
    H_slice: wp.array2d[float],  # articulation mass matrix [dofs, dofs]
    body_q_red: wp.array[wp.transform],
    joint_qd: wp.array[float],
    # maximal-side device state
    body_q_max: wp.array[wp.transform],
    body_qd_max: wp.array[wp.spatial_vector],
    body_f_max: wp.array[wp.spatial_vector],
    # cached weld
    weld_offset: wp.array[wp.vec3],
    weld_rot: wp.array[wp.mat33],
    # scratch
    Jr: wp.array2d[float],  # [6, ndof] EE-COM Jacobian transported to base COM
    Lr: wp.array2d[float],  # [ndof, ndof] Cholesky of M_r
    X: wp.array2d[float],  # [ndof, 6] = M_r^-1 J_r^T
    A: wp.array2d[float],  # [6, 6] boundary inverse inertia
    LA: wp.array2d[float],  # [6, 6] Cholesky of A
    vrhs: wp.array[float],  # [ndof] column solve scratch
    vec6: wp.array[float],  # [6] r_c / lambda
    dv: wp.array[float],  # [ndof] reduced velocity correction
    # diagnostics
    d_pose: wp.array[float],
    d_c0: wp.array[float],
    d_lam: wp.array[float],
    d_Ve: wp.array[float],
    d_Vb: wp.array[float],
):
    # Single-thread solve (launched with dim=1); the boundary system is tiny.

    # --- reduced/maximal boundary geometry (boundary point = maximal base COM) ---
    ee_tf = body_q_red[ee]
    R_ee = wp.quat_to_matrix(wp.transform_get_rotation(ee_tf))
    p_ee_com = wp.transform_point(ee_tf, ee_com_local)

    base_tf = body_q_max[base]
    R_base = wp.quat_to_matrix(wp.transform_get_rotation(base_tf))
    p_base_com = wp.transform_point(base_tf, base_com_local)

    r = p_base_com - p_ee_com  # moment arm EE-COM -> boundary point

    # Transport the reduced Jacobian from the EE COM to the boundary point:
    # v_point = v_com + omega x r, i.e. lin' = lin - r x ang, ang' = ang.
    for c in range(ndof):
        lin = wp.vec3(J_slice[r0 + 0, c], J_slice[r0 + 1, c], J_slice[r0 + 2, c])
        ang = wp.vec3(J_slice[r0 + 3, c], J_slice[r0 + 4, c], J_slice[r0 + 5, c])
        lint = lin - wp.cross(r, ang)
        Jr[0, c] = lint[0]
        Jr[1, c] = lint[1]
        Jr[2, c] = lint[2]
        Jr[3, c] = ang[0]
        Jr[4, c] = ang[1]
        Jr[5, c] = ang[2]

    # Reduced boundary twist V_e = J_r v_r and maximal boundary twist V_b.
    for row in range(6):
        s = float(0.0)
        for c in range(ndof):
            s = s + Jr[row, c] * joint_qd[dof0 + c]
        d_Ve[row] = s
    qd_b = body_qd_max[base]
    for k in range(6):
        d_Vb[k] = qd_b[k]
    for k in range(6):
        d_c0[k] = d_Ve[k] - d_Vb[k]

    # Boundary pose error: deviation of the maximal base from where the reduced
    # end-effector says it should be (world axes, at the base COM).
    desired_p_base = p_ee_com + R_ee * weld_offset[0]
    x_err = p_base_com - desired_p_base
    R_des_base = R_ee * weld_rot[0]
    rot_err = _rotation_log(R_base * wp.transpose(R_des_base))
    d_pose[0] = x_err[0]
    d_pose[1] = x_err[1]
    d_pose[2] = x_err[2]
    d_pose[3] = rot_err[0]
    d_pose[4] = rot_err[1]
    d_pose[5] = rot_err[2]

    # r_c = c_target - c0, with the Baumgarte target +(beta/dt) e_pose on C = V_e - V_b.
    bgain = float(0.0)
    if beta != 0.0:
        bgain = beta / dt
    for k in range(6):
        vec6[k] = bgain * d_pose[k] - d_c0[k]

    # A = J_r M_r^-1 J_r^T + M_m^-1 (+ reg I).  Solve M_r X = J_r^T first.
    _chol_factor(ndof, H_slice, Lr)
    for c in range(6):
        for i in range(ndof):
            vrhs[i] = Jr[c, i]  # (J_r^T)[:, c] == row c of J_r
        _chol_solve(ndof, Lr, vrhs, vrhs)
        for i in range(ndof):
            X[i, c] = vrhs[i]
    for p in range(6):
        for q in range(6):
            s = float(0.0)
            for i in range(ndof):
                s = s + Jr[p, i] * X[i, q]
            A[p, q] = s

    # Maximal inverse spatial inertia at the base COM (world), block-diagonal.
    # The base's free-body inertia is an intentional under-estimate (see note).
    Iinv = R_base * base_inv_inertia_local * wp.transpose(R_base)
    for i in range(3):
        A[i, i] = A[i, i] + base_inv_mass
        for j in range(3):
            A[3 + i, 3 + j] = A[3 + i, 3 + j] + Iinv[i, j]
    for k in range(6):
        A[k, k] = A[k, k] + reg

    # lambda = A^-1 r_c.
    _chol_factor(6, A, LA)
    _chol_solve(6, LA, vec6, vec6)

    # Optional safety clamp for pathological steps.
    if maxvel >= 0.0:
        cn = float(0.0)
        for k in range(6):
            cn = cn + d_c0[k] * d_c0[k]
        cn = wp.sqrt(cn)
        if cn > maxvel:
            f = maxvel / cn
            for k in range(6):
                vec6[k] = vec6[k] * f
    for k in range(6):
        d_lam[k] = vec6[k]

    # Apply +J_r^T lambda to the reduced side as dv = M_r^-1 J_r^T lambda = X lambda.
    for i in range(ndof):
        s = float(0.0)
        for q in range(6):
            s = s + X[i, q] * vec6[q]
        dv[i] = s
        joint_qd[dof0 + i] = joint_qd[dof0 + i] + s

    # Apply -J_m^T lambda (= -lambda, J_m = I) to the maximal side as an external
    # wrench at the base COM (impulse / dt).
    fb = body_f_max[base]
    body_f_max[base] = fb + wp.spatial_vector(
        -vec6[0] / dt, -vec6[1] / dt, -vec6[2] / dt, -vec6[3] / dt, -vec6[4] / dt, -vec6[5] / dt
    )


@wp.kernel
def _post_diag_kernel(
    dof0: int,
    ndof: int,
    base: int,
    Jr: wp.array2d[float],
    joint_qd: wp.array[float],
    body_qd_max: wp.array[wp.spatial_vector],
    # outputs
    d_velpost: wp.array[float],
):
    """Post-correction boundary velocity error V_e_post - V_b_post (diagnostic)."""
    qd_b = body_qd_max[base]
    for row in range(6):
        s = float(0.0)
        for c in range(ndof):
            s = s + Jr[row, c] * joint_qd[dof0 + c]
        d_velpost[row] = s - qd_b[row]


class SolverBoundaryImpulse(SolverBase):
    """Couple a reduced and a maximal subsystem through a rigid 6D boundary impulse.

    .. experimental::

        ``SolverBoundaryImpulse`` is a proof-of-concept. Its API, math, defaults,
        and even its file location may change without prior notice.

    The solver owns two child solvers (one per subsystem) and their two
    :class:`~newton.Model` objects. Each :meth:`step` advances both subsystems
    and exchanges a boundary impulse so that the reduced end-effector body and
    the maximal base body stay rigidly welded (velocity-level constraint, with
    optional Baumgarte position stabilization). The boundary solve runs entirely
    on-device, so a coupled step contains no host synchronization and can be
    CUDA-graph captured together with the two child solvers.

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
        base_inv_mass = float(self.maximal_model.body_inv_mass.numpy()[self.maximal_base_body])
        base_inv_inertia_local = self.maximal_model.body_inv_inertia.numpy()[self.maximal_base_body].reshape(3, 3)
        base_com_local = self.maximal_model.body_com.numpy()[self.maximal_base_body]
        ee_com_local = self.reduced_model.body_com.numpy()[self.reduced_ee_body]
        self._base_inv_mass = base_inv_mass
        self._base_inv_inertia_local = wp.mat33(*[float(v) for v in base_inv_inertia_local.reshape(-1)])
        self._base_com_local = wp.vec3(*[float(v) for v in base_com_local])
        self._ee_com_local = wp.vec3(*[float(v) for v in ee_com_local])

        # Rigid weld offset, captured lazily on the first step from the initial
        # relative configuration of the two subsystems (so they need not be
        # co-located by the caller). Stored on-device; populated before any graph
        # capture (which only ever happens after at least one warmup step).
        self._weld_captured = False
        self._weld_offset = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._weld_rot = wp.zeros(1, dtype=wp.mat33, device=self.device)

        self._stepped = False
        self._alloc_scratch()

    # -- setup helpers ------------------------------------------------------

    def _locate_ee_rows(self) -> tuple[int, int]:
        """Return ``(articulation_index, jacobian_row_offset)`` for the EE body."""
        joint_child = self.reduced_model.joint_child.numpy()
        joint_art = self.reduced_model.joint_articulation.numpy()
        art_start = self.reduced_model.articulation_start.numpy()
        matches = (joint_child == self.reduced_ee_body).nonzero()[0]
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

    def _alloc_scratch(self) -> None:
        """Preallocate all device buffers so :meth:`step` allocates nothing.

        Reusing eval_jacobian / eval_mass_matrix output and temp buffers (and the
        boundary scratch) keeps the coupled step free of per-step allocation, a
        prerequisite for CUDA-graph capture.
        """
        m = self.reduced_model
        dev = self.device
        ndof = self._reduced_ndof
        max_links = m.max_joints_per_articulation
        max_dofs = m.max_dofs_per_articulation
        self._J = wp.empty((m.articulation_count, max_links * 6, max_dofs), dtype=float, device=dev)
        self._H = wp.empty((m.articulation_count, max_dofs, max_dofs), dtype=float, device=dev)
        self._joint_S_s = wp.zeros(m.joint_dof_count, dtype=wp.spatial_vector, device=dev)
        self._body_I_s = wp.zeros(m.body_count, dtype=wp.spatial_matrix, device=dev)

        z = lambda *shape: wp.zeros(shape, dtype=float, device=dev)  # noqa: E731
        self._Jr = z(6, ndof)
        self._Lr = z(ndof, ndof)
        self._X = z(ndof, 6)
        self._A = z(6, 6)
        self._LA = z(6, 6)
        self._vrhs = z(ndof)
        self._vec6 = z(6)
        self._dv = z(ndof)
        self._d_pose = z(6)
        self._d_c0 = z(6)
        self._d_lam = z(6)
        self._d_Ve = z(6)
        self._d_Vb = z(6)
        self._d_velpost = z(6)

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
        solve-boundary-impulse, apply-to-both, advance-maximal. The boundary
        solve runs on-device, so the whole step is CUDA-graph capturable.

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
        # 1. Advance the reduced subsystem unconstrained by the boundary.
        self.reduced_solver.step(reduced_state_in, reduced_state_out, reduced_control, None, dt)

        # 2. Reduced-side boundary quantities. eval_* read body_q/joint_q, which
        #    the reduced solver already refreshed; we re-run FK defensively in
        #    case a child solver does not populate body_qd to the public contract.
        eval_fk(self.reduced_model, reduced_state_out.joint_q, reduced_state_out.joint_qd, reduced_state_out)
        eval_jacobian(self.reduced_model, reduced_state_out, J=self._J, joint_S_s=self._joint_S_s)
        eval_mass_matrix(self.reduced_model, reduced_state_out, H=self._H, J=self._J, body_I_s=self._body_I_s)

        a = self._reduced_art

        # 3. Capture the rigid weld on the first step (before any graph capture).
        if not self._weld_captured:
            wp.launch(
                _capture_weld_kernel,
                dim=1,
                inputs=[
                    self.reduced_ee_body,
                    self.maximal_base_body,
                    self._ee_com_local,
                    self._base_com_local,
                    reduced_state_out.body_q,
                    maximal_state_in.body_q,
                ],
                outputs=[self._weld_offset, self._weld_rot],
                device=self.device,
            )
            self._weld_captured = True

        # 4. Assemble + solve the boundary impulse on-device, apply +J_r^T lambda
        #    to the reduced output velocity and -J_m^T lambda to the maximal input
        #    body_f. The boundary point is anchored at the maximal base COM, so the
        #    maximal Jacobian is the identity selection (J_m = I).
        maxvel = -1.0 if self.config.max_velocity_error is None else float(self.config.max_velocity_error)
        wp.launch(
            _boundary_solve_kernel,
            dim=1,
            inputs=[
                self._reduced_ee_row,
                self._reduced_dof0,
                self._reduced_ndof,
                self.reduced_ee_body,
                self.maximal_base_body,
                float(dt),
                float(self.config.baumgarte),
                float(self.config.regularization),
                maxvel,
                self._base_inv_mass,
                self._ee_com_local,
                self._base_com_local,
                self._base_inv_inertia_local,
                self._J[a],
                self._H[a],
                reduced_state_out.body_q,
                reduced_state_out.joint_qd,
                maximal_state_in.body_q,
                maximal_state_in.body_qd,
                maximal_state_in.body_f,
                self._weld_offset,
                self._weld_rot,
                self._Jr,
                self._Lr,
                self._X,
                self._A,
                self._LA,
                self._vrhs,
                self._vec6,
                self._dv,
                self._d_pose,
                self._d_c0,
                self._d_lam,
                self._d_Ve,
                self._d_Vb,
            ],
            device=self.device,
        )

        # 5. Refresh body_qd so post-step reads see the corrected end-effector twist.
        eval_fk(self.reduced_model, reduced_state_out.joint_q, reduced_state_out.joint_qd, reduced_state_out)

        # 6. Advance the maximal subsystem with the boundary wrench applied.
        self.maximal_solver.step(maximal_state_in, maximal_state_out, maximal_control, maximal_contacts, dt)

        # 7. Diagnostics (post-correction boundary velocity error).
        wp.launch(
            _post_diag_kernel,
            dim=1,
            inputs=[
                self._reduced_dof0,
                self._reduced_ndof,
                self.maximal_base_body,
                self._Jr,
                reduced_state_out.joint_qd,
                maximal_state_out.body_qd,
            ],
            outputs=[self._d_velpost],
            device=self.device,
        )
        self._stepped = True

    # -- diagnostics (read by examples/tests; host copies, do not call under capture) --

    @property
    def boundary_pose_error(self):
        """Boundary pose error ``(lin, ang)`` in world axes [m, rad], shape ``(6,)``."""
        return self._d_pose.numpy() if self._stepped else None

    @property
    def boundary_velocity_error_pre(self):
        """Pre-correction boundary velocity mismatch ``V_e - V_b``, shape ``(6,)``."""
        return self._d_c0.numpy() if self._stepped else None

    @property
    def boundary_velocity_error_post(self):
        """Post-correction boundary velocity mismatch, shape ``(6,)``."""
        return self._d_velpost.numpy() if self._stepped else None

    @property
    def boundary_impulse(self):
        """Boundary impulse ``lambda`` ``(lin, ang)`` at the boundary point, shape ``(6,)``."""
        return self._d_lam.numpy() if self._stepped else None

    @property
    def boundary_twist_reduced(self):
        """Reduced boundary twist ``V_e`` before correction, shape ``(6,)``."""
        return self._d_Ve.numpy() if self._stepped else None

    @property
    def boundary_twist_maximal(self):
        """Maximal boundary twist ``V_b`` before the maximal advance, shape ``(6,)``."""
        return self._d_Vb.numpy() if self._stepped else None

    def notify_model_changed(self, flags: int) -> None:
        """Forward change notifications to both child solvers."""
        self.reduced_solver.notify_model_changed(flags)
        self.maximal_solver.notify_model_changed(flags)
