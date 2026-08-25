#!/usr/bin/env python3
"""SAM 2 video tracking worker.

Checkpoint 0: CLI shape only. Checkpoint 2 implements GPU inference.
"""

from __future__ import annotations

import argparse
import sys


def parse_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected x,y as numbers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track one object in a video and write tracking.json"
    )
    parser.add_argument("video", help="input video path")
    parser.add_argument(
        "--point",
        required=True,
        type=parse_point,
        help="initial click in pixel coordinates, x,y",
    )
    parser.add_argument("--output", default="tracking.json", help="tracking.json path")
    args = parser.parse_args()
    print(
        "track.py is a Checkpoint 2 stub. "
        f"video={args.video} point={args.point[0]},{args.point[1]} output={args.output}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
