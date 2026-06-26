# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record the Franka FR3 + Robotiq 2F-85 cube-stacking run to an MP4
#
# Drives example_franka_2f85_stacking.Example with the chosen solver and records
# the scene headless (pyglet EGL + ffmpeg). One MP4 per solver lets the three
# solver runs be compared side by side.
#
# Command:
#   python -m newton.examples.experimental.example_franka_2f85_stacking_record \
#       --solver mujoco --out stack_mujoco.mp4
#
# NOTE: requires a CUDA device and ffmpeg on PATH. Experimental.
###########################################################################

from __future__ import annotations

import pyglet

pyglet.options["headless"] = True

import argparse
import subprocess

import warp as wp

import newton
from newton.examples.experimental.example_franka_2f85_stacking import make_example


def _ffmpeg(out_path, width, height, fps):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out_path,
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def record(solver, out_path, num_frames=2200, render_every=1, width=960, height=540, fps=60):
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    ex = make_example(viewer, argparse.Namespace(solver=solver, verbose=False))
    # Frame the table + robot + cubes (scene centered near (-0.1, -0.45, 0.3)).
    viewer.set_camera(wp.vec3(0.7, 0.35, 0.95), -22.0, 232.0)
    ff = _ffmpeg(out_path, width, height, fps)
    try:
        for k in range(num_frames):
            ex.step()
            if k % render_every == 0:
                ex.render()
                ff.stdin.write(viewer.get_frame().numpy().tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    print(f"wrote {out_path}: {num_frames // render_every} frames, final task_idx={ex.task_idx.numpy()[0]}/{ex.task_counter}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=str, default="mujoco", choices=["mujoco", "kamino", "hybrid"])
    parser.add_argument("--out", type=str, default="stack.mp4")
    parser.add_argument("--num-frames", type=int, default=2200)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args, _ = parser.parse_known_args()
    record(args.solver, args.out, num_frames=args.num_frames, render_every=args.render_every,
           width=args.width, height=args.height)  # fmt: skip


if __name__ == "__main__":
    main()
