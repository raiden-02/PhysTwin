"""Write a short bouncing-ball clip for tracker smoke tests.

This is generated local footage, not a phone recording. Put a real clip in
samples/ and rerun track.py when you have one.
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
    args = parser.parse_args()

    n = int(args.fps * args.seconds)
    ground = args.height - 40
    radius = 18
    x = 80.0
    y = 40.0
    vx = 140.0
    vy = 20.0
    g = 1800.0
    e = 0.72
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
        vy += g * dt
        x += vx * dt
        y += vy * dt
        if y + radius >= ground and vy > 0:
            y = ground - radius
            vy = -e * vy
        if x + radius >= args.width - 8 or x - radius <= 8:
            vx = -vx

        frame = np.full((args.height, args.width, 3), 32, dtype=np.uint8)
        frame[:, :] = (36, 42, 48)
        noise = rng.integers(0, 12, size=frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        cv2.rectangle(frame, (0, ground), (args.width, args.height), (70, 78, 86), -1)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, (48, 96, 220), -1)
        cv2.circle(frame, (int(round(x - 5)), int(round(y - 6))), 4, (180, 210, 255), -1)
        writer.write(frame)

    writer.release()
    print(f"wrote {path} frames={n} fps={args.fps} start=({80},{40})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
