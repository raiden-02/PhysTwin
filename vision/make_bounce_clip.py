"""Write a short bouncing-ball clip for tracker and pipeline tests.

This is generated local footage, not a phone recording. Physics here is only
for drawing pixels. The C++ fitter reconstructs motion from tracked centroids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="samples/generated_bounce.mp4")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--x0", type=float, default=80.0)
    parser.add_argument("--y0", type=float, default=40.0)
    parser.add_argument("--vx", type=float, default=140.0)
    parser.add_argument("--vy", type=float, default=20.0)
    parser.add_argument("--g", type=float, default=1800.0)
    parser.add_argument("--e", type=float, default=0.72)
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--ground-inset", type=int, default=40)
    parser.add_argument(
        "--walls",
        action="store_true",
        help="bounce off left/right walls. Off by default so the clip matches the V1 ground-only model.",
    )
    args = parser.parse_args()

    n = int(args.fps * args.seconds)
    ground = args.height - args.ground_inset
    radius = args.radius
    x = float(args.x0)
    y = float(args.y0)
    vx = float(args.vx)
    vy = float(args.vy)
    g = float(args.g)
    e = float(args.e)
    dt = 1.0 / args.fps

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {path}")

    rng = np.random.default_rng(0)
    for _ in range(n):
        frame = np.full((args.height, args.width, 3), 32, dtype=np.uint8)
        frame[:, :] = (36, 42, 48)
        noise = rng.integers(0, 12, size=frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        cv2.rectangle(frame, (0, ground), (args.width, args.height), (70, 78, 86), -1)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, (48, 96, 220), -1)
        cv2.circle(frame, (int(round(x - 5)), int(round(y - 6))), 4, (180, 210, 255), -1)
        writer.write(frame)

        vy += g * dt
        x += vx * dt
        y += vy * dt
        if y + radius >= ground and vy > 0:
            y = ground - radius
            vy = -e * vy
        if args.walls and (x + radius >= args.width - 8 or x - radius <= 8):
            vx = -vx

    writer.release()
    click_x = int(round(args.x0))
    click_y = int(round(args.y0))
    print(
        f"wrote {path} frames={n} fps={args.fps} start=({click_x},{click_y}) "
        f"vx={args.vx} vy={args.vy} g={args.g} e={args.e} walls={args.walls}"
    )
    print(f"click --point {click_x},{click_y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
