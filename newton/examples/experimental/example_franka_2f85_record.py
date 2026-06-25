# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record the HYBRID Franka FR3 + Robotiq 2F-85 simulation to an MP4
#
# Runs the boundary-impulse hybrid (Franka arm in SolverFeatherstone + 2F-85
# gripper in SolverKamino, coupled by SolverBoundaryImpulse) while the arm
# sweeps and the gripper closes, and records the scene to an MP4. The arm and
# the gripper are SEPARATE models drawn as two viewer "layers"; because the
# gripper base is welded to the arm flange by the boundary impulse, the gripper
# visibly rides rigidly with the moving arm.
#
# Headless: uses pyglet's EGL backend (no X display) + ffmpeg for encoding.
#
# Command:
#   python -m newton.examples.experimental.example_franka_2f85_record --out out.mp4
#
# NOTE: requires a CUDA device (SolverKamino) and ffmpeg on PATH. Experimental.
###########################################################################

from __future__ import annotations

import pyglet

# Select pyglet's headless EGL backend before any window/context is created.
pyglet.options["headless"] = True

import argparse
import math
import subprocess

import numpy as np
import warp as wp

import newton
from newton.examples.experimental import franka_2f85 as F
from newton.examples.experimental.example_franka_2f85_benchmark import FRANKA_READY, _HybridConfig


def _ffmpeg(out_path, width, height, fps):
    """Open an ffmpeg process that encodes piped raw RGB frames to MP4."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "26",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def record(
    out_path: str,
    device: str = "cuda:0",
    dt: float = 1.0e-3,
    num_steps: int = 1600,
    render_every: int = 8,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
):
    """Run the hybrid sim with an arm sweep + gripper close and write an MP4."""
    cfg = _HybridConfig(device, dt)

    # Drive the gripper closed so it visibly grasps during the clip.
    tq = cfg.grip_ctrl.joint_target_q.numpy()
    for d in cfg.drivers:
        tq[d] = F.GRIPPER_CLOSED
    cfg.grip_ctrl.joint_target_q.assign(tq)

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    # Two layers: the reduced arm and the maximal gripper, drawn together.
    viewer.activate("arm")
    viewer.set_model(cfg.arm_model)
    viewer.activate("gripper")
    viewer.set_model(cfg.grip_model)
    viewer.set_camera(wp.vec3(1.4, 1.1, 0.85), -12.0, 215.0)

    ff = _ffmpeg(out_path, width, height, fps)
    base = np.array(FRANKA_READY, dtype=np.float32)
    arm_tq = cfg.arm_ctrl.joint_target_q.numpy()
    try:
        for step in range(num_steps):
            t = step * dt
            # Gentle multi-joint sweep so the whole arm (and the welded gripper)
            # swings through a visible arc.
            phase = 2.0 * math.pi * t / 1.6
            arm_tq[cfg.arm_dofs[0]] = base[0] + 0.6 * math.sin(phase)
            arm_tq[cfg.arm_dofs[3]] = base[3] + 0.35 * math.sin(phase + 0.7)
            arm_tq[cfg.arm_dofs[5]] = base[5] + 0.4 * math.sin(phase + 1.4)
            cfg.arm_ctrl.joint_target_q.assign(arm_tq)

            cfg._substep()

            if step % render_every == 0:
                viewer.begin_frame(t)
                viewer.activate("arm")
                viewer.log_state(cfg.arm_0)
                viewer.activate("gripper")
                viewer.log_state(cfg.grip_0)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()  # (h, w, 3) uint8
                ff.stdin.write(frame.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    print(f"wrote {out_path} ({num_steps // render_every} frames at {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="franka_2f85_hybrid.mp4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=1600)
    parser.add_argument("--render-every", type=int, default=8)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args, _ = parser.parse_known_args()
    record(
        args.out,
        device=args.device,
        num_steps=args.num_steps,
        render_every=args.render_every,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
