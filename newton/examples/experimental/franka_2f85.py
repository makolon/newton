# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model builders for the Franka FR3 + Robotiq 2F-85 benchmark (experimental).

Shared by ``example_franka_2f85_benchmark.py`` and the benchmark test. Three
model shapes are produced from the same cached assets:

* :func:`build_franka_arm` — arm only (7-DOF, reduced coordinates), for the
  reduced side of the HYBRID config.
* :func:`build_2f85_gripper` — gripper only (closed-chain, maximal coordinates),
  for the Kamino side of the HYBRID config.
* :func:`build_combined` — Franka + 2F-85 welded into one articulation, for the
  FULL-KAMINO and FULL-MUJOCO baselines.

NOTE: This experimental module imports a menagerie download helper from
``newton._src`` (the helper is not part of the public API). The Franka asset
uses the public ``newton.utils.download_asset``.
"""

from __future__ import annotations

import warp as wp

import newton

# Menagerie (Robotiq 2F-85) fetch helper is internal-only; see module note.
from newton._src.utils.download_assets import MENAGERIE_REF, MENAGERIE_URL, download_git_folder

# Names from the cached assets.
_FR3_FLANGE = "/fr3_link8"  # mechanical tool flange the gripper mounts to
_GRIPPER_BASE = "/base"  # the massive 2F-85 base link (0.78 kg); NOT the massless base_mount
_DRIVER_JOINTS = ("right_driver_joint", "left_driver_joint")  # symmetric finger drive

# 2F-85 closed (fingers together) driver angle [rad]; the driver range is [0, 0.8].
GRIPPER_CLOSED = 0.78
GRIPPER_OPEN = 0.0

# Mount transform of the gripper base_mount on the arm flange (fr3_link8). The
# flange +z is the tool direction; mount the gripper a few mm out along +z.
GRIPPER_MOUNT_XFORM = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())


def franka_arm_urdf() -> str:
    """Path to the cached Franka FR3 arm-only URDF."""
    return str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3.urdf")


def robotiq_2f85_mjcf() -> str:
    """Path to the cached Robotiq 2F-85 MJCF (from the MuJoCo Menagerie)."""
    folder = download_git_folder(MENAGERIE_URL, "robotiq_2f85", ref=MENAGERIE_REF)
    return str(folder / "2f85.xml")


def find_body(builder_or_model, suffix: str) -> int:
    """Index of the (first) body whose label ends with ``suffix``."""
    for i, label in enumerate(builder_or_model.body_label):
        if label.endswith(suffix):
            return i
    raise ValueError(f"no body label ending with {suffix!r}")


def find_joint_qd_indices(model, joint_names) -> list[int]:
    """First DOF index of each joint whose label ends with one of ``joint_names``."""
    labels = model.joint_label
    qd_start = model.joint_qd_start.numpy()
    out = []
    for name in joint_names:
        idx = next(j for j, label in enumerate(labels) if label.endswith(name))
        out.append(int(qd_start[idx]))
    return out


def build_franka_arm(device, fixed_base: bool = True):
    """Build the arm-only Franka FR3 reduced-coordinate model.

    Returns:
        Tuple ``(model, flange_body_index)``.
    """
    builder = newton.ModelBuilder()
    builder.add_urdf(franka_arm_urdf(), floating=not fixed_base, enable_self_collisions=False)
    flange = find_body(builder, _FR3_FLANGE)
    model = builder.finalize(device=device)
    return model, flange


def build_2f85_gripper(device, floating: bool = True, register_kamino: bool = True):
    """Build the gripper-only Robotiq 2F-85 maximal-coordinate model.

    The MuJoCo ``connect`` equalities are converted to Newton ball loop joints
    (which Kamino handles natively); the driver-coupling mimic is dropped by
    Kamino, so callers must drive both driver joints symmetrically.

    Returns:
        Tuple ``(model, base_body_index, driver_qd_indices)``.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if register_kamino:
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
    builder.add_mjcf(
        robotiq_2f85_mjcf(),
        floating=floating,
        convert_mjc_equality_constraints=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
    )
    model = builder.finalize(device=device, skip_validation_joints=True)
    base = find_body(model, _GRIPPER_BASE)
    drivers = find_joint_qd_indices(model, _DRIVER_JOINTS)
    return model, base, drivers


def build_combined(device, register_kamino: bool = True):
    """Build the Franka FR3 + Robotiq 2F-85 welded into a single articulation.

    Used for the FULL-KAMINO and FULL-MUJOCO baselines.

    Returns:
        Tuple ``(model, ee_body_index, arm_qd_indices, gripper_driver_qd_indices)``.
    """
    builder = newton.ModelBuilder()
    if register_kamino:
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
    builder.add_urdf(franka_arm_urdf(), floating=False, enable_self_collisions=False)
    flange = find_body(builder, _FR3_FLANGE)
    builder.add_mjcf(
        robotiq_2f85_mjcf(),
        xform=GRIPPER_MOUNT_XFORM,
        parent_body=flange,
        convert_mjc_equality_constraints=True,
        enable_self_collisions=False,
    )
    model = builder.finalize(device=device, skip_validation_joints=True)
    # Arm DOFs are the 7 revolute joints fr3_joint1..7.
    arm_dofs = find_joint_qd_indices(model, tuple(f"fr3_joint{i}" for i in range(1, 8)))
    drivers = find_joint_qd_indices(model, _DRIVER_JOINTS)
    return model, flange, arm_dofs, drivers
