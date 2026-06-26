# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Franka FR3 + Robotiq 2F-85 cube stacking, three solvers (experimental)
#
# Adapts newton/examples/ik/example_ik_cube_stacking.py from the Franka Hand to
# the Robotiq 2F-85 closed-chain gripper, and runs the same IK-driven pick-and-
# stack task under three solver configurations for a speed comparison:
#
#   mujoco : whole arm+gripper+cubes in SolverMuJoCo (reference).
#   kamino : whole arm+gripper+cubes in SolverKamino (maximal coordinates).
#   hybrid : arm in SolverFeatherstone + gripper+cubes in SolverKamino, coupled
#            by SolverBoundaryImpulse.
#
# The grasp is a *real* friction grasp (the gripper physically closes on the
# cubes). IK + task scheduling are solver-agnostic; only the physics step
# differs, and all three share one *combined* bookkeeping model+state for IK,
# task scheduling and rendering. The hybrid additionally runs physics on a split
# reduced-arm + maximal-gripper pair and syncs the result back into that combined
# state once per frame (see HybridExample), so everything downstream is unchanged.
#
# The per-frame substep loop is kept host-round-trip-free and CUDA-graph captured
# (~10x for Kamino, whose PADMM iteration otherwise host-syncs every step). The
# stiff PD-held Franka needs a small enough substep dt to integrate stably, so the
# substep count is per-solver (MuJoCo's implicit integrator 10; the hybrid 40 so
# the contact-rich grasp stays stable through the boundary weld; full-Kamino the
# most).
#
# Accuracy across solvers is not directly compared (the non-MuJoCo arms lack
# gravity compensation, so they track the IK targets less precisely); the goal is
# wall-clock speed plus a recorded MP4 per solver. The hybrid grasps and lifts the
# cubes; full-Kamino, running the stiff arm in maximal coordinates, is both slower
# and the least stable here — which is itself the motivation for the hybrid.
#
# Command:
#   python -m newton.examples.experimental.example_franka_2f85_stacking --solver mujoco
#
# NOTE: requires a CUDA device for kamino/hybrid. Experimental.
###########################################################################

from __future__ import annotations

import enum
import time

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.examples.experimental import franka_2f85 as F


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP_OFF = 4
    REFINE_DROP_OFF = 5
    RELEASE = 6
    RETRACT = 7
    HOME = 8


# 2F-85 driver angle: 0 = open, ~0.78 = fully closed.
GRIPPER_CLOSE_ANGLE = 0.72

# Grasp ("pinch") point expressed in the 2F-85 base body frame (+z is the
# gripper approach axis, matching the downward-orientation IK target).
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.145)


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_object: wp.array[int],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    task_offset_approach: wp.vec3,
    task_offset_lift: wp.vec3,
    task_offset_retract: wp.vec3,
    task_drop_off_pos: wp.vec3,
    cube_size: float,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    robot_body_count: int,
    num_bodies_per_world: int,
    close_angle: float,
    tcp_offset: wp.vec3,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    task_time_soft_limit = task_time_soft_limits[idx]
    cube_body_index = task_object[idx]
    cube_index = cube_body_index - robot_body_count

    task_time_elapsed[tid] += task_dt
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_point(task_init_body_q[ee_body_id], tcp_offset)
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    obj_body_id = tid * num_bodies_per_world + cube_body_index
    obj_pos_current = wp.transform_get_translation(body_q[obj_body_id])
    obj_quat_current = wp.transform_get_rotation(body_q[obj_body_id])
    cube_offset = wp.float(cube_index) * cube_size * wp.vec3(0.0, 0.0, 1.0)

    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        ee_pos_target[tid] = obj_pos_current + task_offset_approach
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi) * wp.quat_inverse(obj_quat_current)
    elif task == TaskType.REFINE_APPROACH.value:
        ee_pos_target[tid] = obj_pos_current
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_DROP_OFF.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset + task_offset_approach
        t_gripper = 1.0
    elif task == TaskType.REFINE_DROP_OFF.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset
        t_gripper = 1.0 - t
    elif task == TaskType.RETRACT.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset + task_offset_retract
    elif task == TaskType.HOME.value:
        ee_pos_target[tid] = home_pos
    else:
        ee_pos_target[tid] = home_pos

    ee_pos_target_interpolated[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target[tid] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[tid] = ee_quat_target[:4]
    ee_rot_target_interpolated[tid] = ee_quat_interpolated[:4]

    gripper_target[tid] = close_angle * t_gripper


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    num_bodies_per_world: int,
    ee_index: int,
    tcp_offset: wp.vec3,
    pos_tol: float,
    rot_tol: float,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    task_time_soft_limit = task_time_soft_limits[idx]

    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_current = wp.transform_point(body_q[ee_body_id], tcp_offset)
    ee_quat_current = wp.transform_get_rotation(body_q[ee_body_id])

    pos_err = wp.length(ee_pos_target[tid] - ee_pos_current)
    ee_quat_target = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_target)
    rot_err = wp.abs(wp.degrees(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3])))

    if (
        task_time_elapsed[tid] >= task_time_soft_limit
        and pos_err < pos_tol
        and rot_err < rot_tol
        and task_idx[tid] < wp.len(task_time_soft_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0
        body_id_start = tid * num_bodies_per_world
        for i in range(num_bodies_per_world):
            body_id = body_id_start + i
            task_init_body_q[body_id] = body_q[body_id]


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.solver_kind = args.solver
        # Substeps per frame (must be even for graph capture). The reduced/maximal
        # arms need a small enough dt to integrate the stiff PD-held Franka stably:
        # SolverMuJoCo's implicit integrator is fine at 10; SolverKamino driving the
        # full stiff arm in maximal coordinates needs the smallest dt. The hybrid
        # uses 40: the free arm sweep alone integrates at 20, but the contact-rich
        # grasp couples the gripper's reaction torque back through the boundary
        # weld, and that needs a finer dt to keep the angular weld + gripper
        # contacts stable (see HybridExample's boundary-weld config).
        default_substeps = {"mujoco": 10, "kamino": 32, "hybrid": 40}
        self.sim_substeps = getattr(args, "substeps", None) or default_substeps[self.solver_kind]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = 1
        self.verbose = getattr(args, "verbose", False)
        self.viewer = viewer

        # CUDA-graph capture the substep loop (kept host-round-trip-free) so the
        # device-side Kamino/MuJoCo iteration runs entirely on the GPU; without it
        # each step pays the per-launch host sync (~10x for Kamino). Requires an
        # even substep count (the ping-pong swap must net to the original buffers).
        self.use_graph = wp.get_device().is_cuda and not getattr(args, "no_graph", False)
        assert not self.use_graph or self.sim_substeps % 2 == 0, "graph capture needs an even sim_substeps"
        self._graph = None
        # Set by _make_kamino_solver; mujoco uses Newton contacts (model.collide).
        self._kamino_internal = False

        self.cube_count = 3
        self.cube_size = 0.05
        self.table_height = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        self.task_offset_approach = wp.vec3(0.0, 0.0, 1.0 * self.cube_size)
        self.task_offset_lift = wp.vec3(0.0, 0.0, 4.0 * self.cube_size)
        self.task_offset_retract = wp.vec3(0.0, 0.0, 2.0 * self.cube_size)
        self.task_drop_off_pos = self.table_top_center + wp.vec3(0.0, -0.15, 0.5 * self.cube_size)

        # Looser IK-tracking tolerances for the non-MuJoCo solvers (which lack
        # MuJoCo's gravity compensation, so the arm tracks less precisely).
        self.pos_tol = 0.002 if self.solver_kind == "mujoco" else 0.02
        self.rot_tol = 1.0 if self.solver_kind == "mujoco" else 5.0

        # Build the scene (single combined model for all configs; the hybrid
        # additionally builds split arm/gripper models below).
        self.robot_builder = self.build_robot()
        self.robot_body_count = self.robot_builder.body_count
        self.model_single = self.build_robot().finalize()
        self.model = self.build_scene().finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        self._build_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.setup_ik()
        self.setup_tasks()

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False

        self.episode_steps = 0
        self.sim_elapsed = 0.0

    # -- scene construction --------------------------------------------------

    def build_robot(self):
        """Franka FR3 arm + Robotiq 2F-85 gripper + a TCP frame, welded as one."""
        builder = newton.ModelBuilder()
        if self.solver_kind == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        newton.solvers.SolverKamino.register_custom_attributes(builder)

        builder.add_urdf(
            F.franka_arm_urdf(),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        flange = F.find_body(builder, "/fr3_link8")
        builder.add_mjcf(
            F.robotiq_2f85_mjcf(),
            xform=F.GRIPPER_MOUNT_XFORM,
            parent_body=flange,
            convert_mjc_equality_constraints=True,
            enable_self_collisions=False,
        )
        # The IK end-effector is the gripper base body + a fixed TCP_OFFSET to the
        # pinch point (see TCP_OFFSET); no separate TCP body is needed.

        # Keep collisions ONLY on the gripper pads (vs cubes/table). The 2F-85's
        # many mesh shapes and the Franka arm meshes otherwise generate thousands
        # of contacts (slow, memory-heavy), and the open fingers would knock cubes
        # during the approach. The pads are what actually grasp.
        pad_bodies = {
            F.find_body(builder, s) for s in ("/right_pad", "/left_pad", "/right_silicone_pad", "/left_silicone_pad")
        }
        for si in range(builder.shape_count):
            b = builder.shape_body[si]
            if b >= 0 and b not in pad_bodies:
                builder.shape_collision_group[si] = 0

        # Initial arm pose (ready) + PD gains + armature on the 7 arm joints.
        arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i in range(7):
            builder.joint_q[i] = arm_q[i]
            builder.joint_target_q[i] = arm_q[i]
        builder.joint_target_ke[:7] = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
        builder.joint_target_kd[:7] = [450, 450, 350, 350, 200, 200, 200]
        builder.joint_armature[:7] = [0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11]
        # Gripper driver gains (coords 7=right, 11=left driver). High stiffness so
        # the closed gripper clamps the cube firmly enough to hold it against
        # gravity. The other 2F-85 joints (coupler/spring/follower) are passive
        # (loop-driven), so their PD gains are zeroed.
        #
        # The 2F-85's native (tendon) actuator is dropped on import, leaving the
        # driver joints with joint_target_mode NONE. SolverMuJoCo then builds no
        # actuator for them, so under JOINT_TARGET the gripper is uncontrolled and
        # its return spring (springref) slowly closes it. Forcing POSITION mode
        # makes MuJoCo build a position actuator driven by joint_target_q. Kamino
        # already drives the drivers from their PD gains regardless of the mode, and
        # forcing POSITION there destabilizes its loop solve, so this is MuJoCo-only.
        for d in (7, 11):
            builder.joint_target_ke[d] = 800.0
            builder.joint_target_kd[d] = 40.0
            if self.solver_kind == "mujoco":
                builder.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
        for d in (8, 9, 10, 12, 13, 14):
            builder.joint_target_ke[d] = 0.0
            builder.joint_target_kd[d] = 0.0

        # Friction + table.
        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        shape_cfg.ke, shape_cfg.kd, shape_cfg.kf, shape_cfg.mu = 5.0e4, 5.0e2, 1.0e3, 1.0
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg,
        )

        if self.solver_kind == "mujoco":
            self._enable_mujoco_gravcomp(builder)
        return builder

    def _enable_mujoco_gravcomp(self, builder):
        # Cancel gravity on every robot body so IK tracks precisely (matches the
        # stock cube-stacking example). Cubes are added later and are NOT comped.
        gc_body = builder.custom_attributes["mujoco:gravcomp"]
        if gc_body.values is None:
            gc_body.values = {}
        for b in range(1, builder.body_count):
            gc_body.values[b] = 1.0
        gc_joint = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gc_joint.values is None:
            gc_joint.values = {}
        for d in range(7):
            gc_joint.values[d] = True

    def build_scene(self):
        rng = np.random.default_rng(42)
        scene = newton.ModelBuilder()
        if self.solver_kind == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        newton.solvers.SolverKamino.register_custom_attributes(scene)
        scene.begin_world()
        scene.add_builder(self.build_robot())
        self.add_cubes(scene, rng)
        scene.end_world()
        scene.add_ground_plane()
        return scene

    def add_cubes(self, scene, rng):
        # Deterministic, axis-aligned, well-separated cubes: the 2F-85's arc-
        # closing fingers grasp much more reliably than from a random yaw, and
        # fixed placement makes the run reproducible. Heavier cubes resist being
        # knocked during the approach.
        default_cube_offset = wp.transform(wp.vec3(0.0, 0.15, 0.5 * self.cube_size))
        origin = wp.transform(self.table_top_center) * default_cube_offset
        colors = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]
        local = [wp.vec3(-0.13, 0.0, 0.0), wp.vec3(0.0, 0.05, 0.0), wp.vec3(0.13, 0.0, 0.0)]
        for i in range(self.cube_count):
            cfg = newton.ModelBuilder.ShapeConfig(density=800.0, margin=0.0)
            cfg.mu = 1.2
            body = scene.add_body(xform=origin * wp.transform(local[i], wp.quat_identity()))
            scene.add_shape_box(
                body=body,
                hx=0.5 * self.cube_size,
                hy=0.5 * self.cube_size,
                hz=0.5 * self.cube_size,
                cfg=cfg,
                label=f"cube_{i}",
                color=colors[i],
            )

    # -- solver --------------------------------------------------------------

    def _use_joint_target_actuation(self):
        # The 2F-85 ships a tendon actuator that otherwise overrides the driver
        # joint-target PD (leaving the gripper open). Selecting JOINT_TARGET makes
        # MuJoCo drive the gripper from joint_target_q like the arm.
        mj = getattr(self.model, "mujoco", None)
        if mj is not None and getattr(mj, "ctrl_source", None) is not None:
            mj.ctrl_source.fill_(int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET))

    def _make_kamino_solver(self, model):
        """Build a SolverKamino using its own UNIFIED collision pipeline.

        Unlike the ``primitive`` pipeline, ``unified`` supports the gripper's
        convex-mesh shapes, and detecting/resolving contacts natively (rather than
        converting Newton's contacts) avoids a capacity-truncation mismatch that
        let the descending gripper launch the cubes.
        """
        self._kamino_internal = True
        cfg = newton.solvers.SolverKamino.Config.from_model(model)
        cfg.use_fk_solver = True
        cfg.use_collision_detector = True
        cfg.collision_detector.pipeline = "unified"
        cfg.collision_detector.max_contacts = 1024
        return newton.solvers.SolverKamino(model, cfg)

    def _build_solver(self):
        if self.solver_kind == "mujoco":
            self._use_joint_target_actuation()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver="newton",
                integrator="implicitfast",
                iterations=20,
                ls_iterations=100,
                nconmax=2000,
                njmax=4000,
                cone="elliptic",
                impratio=1000.0,
                use_mujoco_contacts=False,
            )
        elif self.solver_kind == "kamino":
            self.solver = self._make_kamino_solver(self.model)
        else:
            raise ValueError(f"solver '{self.solver_kind}' is built in the hybrid subclass")

    # -- IK + tasks (reused, solver-agnostic) --------------------------------

    def setup_ik(self):
        self.ee_index = F.find_body(self.model_single, "/base_mount/base")
        body_q_np = self.state.body_q.numpy()
        base_tf = wp.transform(*body_q_np[self.ee_index])
        self.ee_tf = base_tf
        grasp_pos = wp.transform_point(base_tf, TCP_OFFSET)
        self.home_pos = wp.vec3(grasp_pos)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=TCP_OFFSET,
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.transform_get_rotation(self.ee_tf)[:4]] * self.world_count, dtype=wp.vec4),
        )
        ik_dofs = self.model_single.joint_coord_count
        self.joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, :ik_dofs])
        self.joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.joint_limit_lower.flatten(),
            joint_limit_upper=self.joint_limit_upper.flatten(),
        )
        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, :ik_dofs])
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        task_per_object = 9
        task_schedule = []
        for _ in range(self.cube_count):
            task_schedule.extend(list(TaskType))
        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array([int(t) for t in task_schedule], shape=(self.task_counter,), dtype=wp.int32)
        # Slower per-task motion so the EE approaches cubes gently (the 2F-85 is
        # easy to knock cubes with at speed).
        self.task_time_soft_limits = wp.array([1.5] * self.task_counter, dtype=float)
        task_object = []
        for i in range(self.cube_count):
            task_object.extend([self.robot_body_count + i] * task_per_object)
        self.task_object = wp.array(task_object, shape=(self.task_counter,), dtype=wp.int32)
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_dt = self.frame_dt
        self.task_time_elapsed = wp.zeros(self.world_count, dtype=wp.float32)
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_pos_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_rot_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target_interpolated = wp.zeros(self.world_count, dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_object,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.task_offset_retract,
                self.task_drop_off_pos,
                self.cube_size,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.robot_body_count,
                self.num_bodies_per_world,
                GRIPPER_CLOSE_ANGLE,
                TCP_OFFSET,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target_interpolated,
            ],
        )
        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        joint_target_q_view = self.control.joint_target_q.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_q_view[:, :7], src=self.joint_q_ik[:, :7])
        # Drive both 2F-85 driver joints (coords 7 and 11) symmetrically.
        wp.copy(dest=joint_target_q_view[:, 7:8], src=self.gripper_target_interpolated.reshape((self.world_count, 1)))
        wp.copy(dest=joint_target_q_view[:, 11:12], src=self.gripper_target_interpolated.reshape((self.world_count, 1)))

        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
                TCP_OFFSET,
                self.pos_tol,
                self.rot_tol,
            ],
            outputs=[self.task_idx, self.task_time_elapsed, self.task_init_body_q],
        )

    # -- sim loop ------------------------------------------------------------

    def _collide(self):
        # Kamino's internal detector runs inside solver.step; only Newton-contact
        # solvers (MuJoCo here) need an explicit collide pass.
        if not self._kamino_internal:
            self.model.collide(self.state_0, self.contacts)

    def _run_substeps(self):
        contacts = None if self._kamino_internal else self.contacts
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _post_substeps(self):
        pass

    def simulate(self):
        # Collision detection and any host-side bookkeeping run eagerly each frame
        # (outside the captured region); only the pure-device substep loop is
        # captured and replayed.
        self._collide()
        if self.use_graph:
            if self._graph is None:
                self._run_substeps()  # warm: load modules + capture the rigid weld
                with wp.ScopedCapture() as cap:
                    self._run_substeps()
                self._graph = cap.graph
            else:
                wp.capture_launch(self._graph)
        else:
            self._run_substeps()
        self._post_substeps()

    def step(self):
        self.set_joint_targets()
        t0 = time.perf_counter()
        self.simulate()
        wp.synchronize()
        self.sim_elapsed += time.perf_counter() - t0
        self.sim_time += self.frame_dt
        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @property
    def ms_per_substep(self):
        n = max(1, self.episode_steps * self.sim_substeps)
        return 1e3 * self.sim_elapsed / n

    def test_final(self):
        # This example compares solver wall-clock speed and behaviour rather than
        # stacking accuracy (the non-MuJoCo arms lack gravity compensation, so they
        # track the IK targets less precisely; full-Kamino additionally tends to
        # knock the cubes during the contact-rich grasp). The invariant that holds
        # across all three is that the coupled simulation stays finite and bounded.
        body_q = self.state_0.body_q.numpy()
        if not (np.all(np.isfinite(body_q)) and np.all(np.isfinite(self.state_0.body_qd.numpy()))):
            raise ValueError(f"[{self.solver_kind}] simulation produced a non-finite state")
        # Surface (but don't fail on) the peak body speed: full-Kamino knocks the
        # cubes during the contact-rich grasp and so shows large but bounded
        # transients, whereas mujoco/hybrid stay slow.
        max_speed = float(np.max(np.abs(self.state_0.body_qd.numpy())))
        advanced = int(self.task_idx.numpy()[0])
        print(f"[{self.solver_kind}] finite, max|v|={max_speed:.2f}, task_idx={advanced}/{self.task_counter}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", type=str, default="mujoco", choices=["mujoco", "kamino", "hybrid"])
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--no-graph", action="store_true", help="Disable CUDA-graph capture of the substep loop.")
        parser.add_argument("--substeps", type=int, default=None, help="Override the per-solver substep count (even).")
        return parser


class HybridExample(Example):
    """Boundary-impulse hybrid: Franka arm in :class:`~newton.solvers.SolverFeatherstone`
    (reduced coordinates) + the 2F-85 gripper *and* the cubes in
    :class:`~newton.solvers.SolverKamino` (maximal coordinates), welded by
    :class:`~newton.solvers.SolverBoundaryImpulse` (flange ``fr3_link8`` <-> gripper
    ``base``).

    Only the physics step differs from the ``mujoco``/``kamino`` configs. IK, task
    scheduling and rendering all run off the *combined* bookkeeping model+state
    built by :class:`Example`; physics runs on the two split models and is synced
    back into ``state_0`` once per frame, so everything downstream is unchanged.
    """

    DRIVER_JOINTS = ("right_driver_joint", "left_driver_joint")
    # The 2F-85's passive four-bar links (coupler/spring/follower on each finger).
    PASSIVE_FINGER_JOINTS = (
        "right_coupler_joint",
        "left_coupler_joint",
        "right_spring_link_joint",
        "left_spring_link_joint",
        "right_follower_joint",
        "left_follower_joint",
    )
    # Passive-finger joint damping [N·m·s/rad]. Each 2F-85 finger is a 1-DOF four-bar
    # whose passive links are *meant* to be carried by the loop closure, but
    # SolverKamino consumes ``joint_target_kd`` (PD damping), not the model's passive
    # ``joint_damping``, and the MJCF spring (``springref``) is dropped on import — so
    # with zero target_kd these links are undamped free hinges held only by the soft
    # loop solve and flop (droop/swing) when the arm accelerates the gripper. A modest
    # target_kd damps them so they track the driver-commanded pose. ``target_ke`` stays
    # zero so the damping does not fight the four-bar as the gripper opens/closes.
    FINGER_DAMPING = 2.0

    def _build_solver(self):
        device = self.model.device

        # -- reduced arm (Featherstone) --
        self.arm_model = self.build_arm().finalize(device=device)
        self.arm_flange = F.find_body(self.arm_model, "/fr3_link8")
        self.n_arm = self.arm_model.body_count
        self.arm_solver = newton.solvers.SolverFeatherstone(self.arm_model)
        self.arm_0 = self.arm_model.state()
        self.arm_1 = self.arm_model.state()
        self.arm_ctrl = self.arm_model.control()
        wp.copy(self.arm_ctrl.joint_target_q, self.arm_model.joint_q)
        newton.eval_fk(self.arm_model, self.arm_model.joint_q, self.arm_model.joint_qd, self.arm_0)
        flange_tf = self.arm_0.body_q.numpy()[self.arm_flange]

        # -- maximal gripper + cubes (Kamino) --
        self.grip_model = self.build_gripper_cubes(flange_tf).finalize(device=device, skip_validation_joints=True)
        self.grip_base = F.find_body(self.grip_model, "/base")
        # The 2F-85's coupling mimic is dropped on import, so both drivers are
        # driven symmetrically. Gains index DOFs; targets index coords — these
        # differ once a floating base is present (free joint: 7 coords / 6 DOFs).
        self.grip_driver_dofs = F.find_joint_qd_indices(self.grip_model, self.DRIVER_JOINTS)
        self.grip_driver_coords = self._driver_q_indices()
        self._set_gripper_gains()
        self.grip_solver = self._make_kamino_solver(self.grip_model)
        self.grip_0 = self.grip_model.state()
        self.grip_1 = self.grip_model.state()
        self.grip_ctrl = self.grip_model.control()
        self.grip_contacts = None if self._kamino_internal else self.grip_model.contacts()

        assert self.n_arm + self.grip_model.body_count == self.num_bodies_per_world, (
            f"hybrid body split {self.n_arm}+{self.grip_model.body_count} != combined {self.num_bodies_per_world}"
        )

        # Weld the gripper base onto the flange: warm one Kamino step, then reset
        # the floating base pose to the flange (mirrors the benchmark's hybrid).
        self.grip_solver.step(self.grip_0, self.grip_1, self.grip_ctrl, None, self.sim_dt)
        self.grip_solver.reset(self.grip_0)
        base_q = wp.array(
            [wp.transform(wp.vec3(*flange_tf[:3].tolist()), wp.quat(*flange_tf[3:7].tolist()))],
            dtype=wp.transformf,
            device=device,
        )
        self.grip_solver.reset(state=self.grip_0, base_q=base_q)

        # The 2F-85 base is light (~0.78 kg, tiny rotational inertia), so the bare
        # boundary impulse cannot weld the *angular* DOFs: a staggered impulse needs
        # the base's small free inertia as M_m, leaving the angular impulse
        # negligible, so a grasp reaction torque spins the welded base and the
        # gripper jitters. The velocity-Baumgarte weld projection pulls the base
        # twist onto the end-effector twist after the maximal solve (dissipative,
        # independent of M_m), which holds the angular weld; combined with the finer
        # grasp substep dt it keeps the coupled gripper stable.
        #
        # The benchmark only validated a *static* hold; this example is a *dynamic*
        # grasp, where the gripper picks up a cube and the arm accelerates the loaded
        # gripper. There the free-base M_m under-estimates the loaded effective
        # inertia by ~50x, so the staggered impulse cannot hold the weld through the
        # lift/place transitions: the Baumgarte bias winds up (the maximal side
        # under-responds to the wrench it predicts), ramping the boundary force, and
        # near arm singularities the operational inverse-inertia amplifies even a
        # small impulse into a joint-velocity whip on the arm. The clamps below bound
        # the transmitted wrench (so the gripper is not blasted) and the arm velocity
        # correction (so the arm is not whipped), turning the dynamic overload into a
        # graceful under-correction instead of a destructive transient (peak arm
        # joint speed ~25 -> ~1 rad/s, peak boundary force ~310 -> ~50 N).
        self.solver = newton.solvers.SolverBoundaryImpulse(
            reduced_solver=self.arm_solver,
            maximal_solver=self.grip_solver,
            reduced_ee_body=self.arm_flange,
            maximal_base_body=self.grip_base,
            config=newton.solvers.SolverBoundaryImpulse.Config(
                weld_velocity_projection=0.4,
                max_boundary_force=50.0,
                max_boundary_torque=10.0,
                max_reduced_correction=0.1,
            ),
        )

    # -- split-model construction --

    def build_arm(self):
        """Franka FR3 arm only (7-DOF, reduced), with the same ready pose, PD gains
        and armature as :meth:`Example.build_robot`'s arm. Armature keeps the
        semi-implicit Featherstone integrator stable at this dt (the URDF has none).
        """
        builder = newton.ModelBuilder()
        builder.add_urdf(
            F.franka_arm_urdf(),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i in range(7):
            builder.joint_q[i] = arm_q[i]
            builder.joint_target_q[i] = arm_q[i]
        builder.joint_target_ke[:7] = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
        builder.joint_target_kd[:7] = [450, 450, 350, 350, 200, 200, 200]
        builder.joint_armature[:7] = [0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11]
        return builder

    def build_gripper_cubes(self, flange_tf):
        """2F-85 gripper (floating, maximal) + static table + cubes + ground.

        The gripper bodies precede the cube bodies, matching the combined model's
        ordering after the arm so :meth:`_sync_combined_state` is a contiguous copy.
        """
        rng = np.random.default_rng(42)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        grip_xform = wp.transform(wp.vec3(*flange_tf[:3].tolist()), wp.quat(*flange_tf[3:7].tolist()))
        builder.add_mjcf(
            F.robotiq_2f85_mjcf(),
            xform=grip_xform,
            floating=True,
            convert_mjc_equality_constraints=True,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
        )
        # Keep collisions only on the gripper pads (vs cubes/table); see build_robot.
        pad_bodies = {
            F.find_body(builder, s) for s in ("/right_pad", "/left_pad", "/right_silicone_pad", "/left_silicone_pad")
        }
        for si in range(builder.shape_count):
            b = builder.shape_body[si]
            if b >= 0 and b not in pad_bodies:
                builder.shape_collision_group[si] = 0
        # Table (static).
        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        shape_cfg.ke, shape_cfg.kd, shape_cfg.kf, shape_cfg.mu = 5.0e4, 5.0e2, 1.0e3, 1.0
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg,
        )
        self.add_cubes(builder, rng)
        builder.add_ground_plane()
        return builder

    def _driver_q_indices(self):
        labels = self.grip_model.joint_label
        q_start = self.grip_model.joint_q_start.numpy()
        out = []
        for name in self.DRIVER_JOINTS:
            j = next(i for i, label in enumerate(labels) if label.endswith(name))
            out.append(int(q_start[j]))
        return out

    def _set_gripper_gains(self):
        # Drive only the two driver joints (stiffness + damping); damp the passive
        # four-bar links so they track the loop instead of flopping (see
        # FINGER_DAMPING). target_ke stays zero on the passive links so it cannot
        # fight the four-bar kinematics. The free floating base is left untouched so
        # its damping does not drag the boundary weld.
        ke = self.grip_model.joint_target_ke.numpy()
        kd = self.grip_model.joint_target_kd.numpy()
        ke[:] = 0.0
        kd[:] = 0.0
        for d in self.grip_driver_dofs:
            ke[d] = 800.0
            kd[d] = 40.0
        for d in F.find_joint_qd_indices(self.grip_model, self.PASSIVE_FINGER_JOINTS):
            kd[d] = self.FINGER_DAMPING
        self.grip_model.joint_target_ke.assign(ke)
        self.grip_model.joint_target_kd.assign(kd)

    # -- sim loop (split physics + sync into the combined bookkeeping state) --

    def _apply_split_targets(self):
        # Arm joint targets from IK (the first 7 coords are the arm joints).
        wp.copy(dest=self.arm_ctrl.joint_target_q[:7], src=self.joint_q_ik[0, :7])
        # Both gripper drivers, symmetrically, at their coord indices.
        g = float(self.gripper_target_interpolated.numpy()[0])
        tq = self.grip_ctrl.joint_target_q.numpy()
        for c in self.grip_driver_coords:
            tq[c] = g
        self.grip_ctrl.joint_target_q.assign(tq)

    def _sync_combined_state(self):
        wp.copy(dest=self.state_0.body_q[: self.n_arm], src=self.arm_0.body_q)
        wp.copy(dest=self.state_0.body_qd[: self.n_arm], src=self.arm_0.body_qd)
        wp.copy(dest=self.state_0.body_q[self.n_arm :], src=self.grip_0.body_q)
        wp.copy(dest=self.state_0.body_qd[self.n_arm :], src=self.grip_0.body_qd)

    def _collide(self):
        # Per-frame, outside the captured region: push the new IK targets onto the
        # split controls. Kamino detects the gripper contacts internally (the
        # _kamino_internal default); the Newton-contact fallback collides here.
        self._apply_split_targets()
        if not self._kamino_internal:
            self.grip_model.collide(self.grip_0, self.grip_contacts)

    def _run_substeps(self):
        for _ in range(self.sim_substeps):
            self.arm_0.clear_forces()
            self.grip_0.clear_forces()
            self.solver.step(
                self.arm_0,
                self.arm_1,
                self.grip_0,
                self.grip_1,
                self.arm_ctrl,
                self.grip_ctrl,
                self.sim_dt,
                maximal_contacts=self.grip_contacts,
            )
            self.arm_0, self.arm_1 = self.arm_1, self.arm_0
            self.grip_0, self.grip_1 = self.grip_1, self.grip_0

    def _post_substeps(self):
        self._sync_combined_state()

    def render(self):
        # The combined contacts are never populated in the hybrid (physics runs on
        # the split models), so render the synced state only.
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def make_example(viewer, args):
    cls = HybridExample if args.solver == "hybrid" else Example
    return cls(viewer, args)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(make_example(viewer, args), args)
