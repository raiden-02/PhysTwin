"""Run one executable PhysicalScene and write SimulatedWorldState JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics3d.newton_runtime import simulate_physical_scene  # noqa: E402
from vision.reconstruction.contracts import load_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        help="run the same scene twice and compare body transforms",
    )
    args = parser.parse_args()

    scene = dict(load_contract(args.physical_scene))
    rollout = simulate_physical_scene(scene, repeat_check=args.repeat_check)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "simulated_world_state.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rollout, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)

    validation = rollout["validation"]
    execution = rollout["execution"]
    repeat = rollout["reproducibility"]["repeat_run"]
    print(f"output: {destination}")
    print(f"backend: Newton {rollout['simulator']['backend_version']} / Warp {rollout['simulator']['warp_version']}")
    print(f"device: {rollout['simulator']['device_name']} ({rollout['simulator']['device']})")
    print(f"steps: {execution['steps']}")
    print(f"wall_seconds: {execution['wall_seconds']:.6f}")
    print(f"peak_gpu_memory_bytes: {execution['peak_gpu_memory_bytes']}")
    print(f"max_tether_error_m: {validation['tether_error_m']['maximum']:.9f}")
    print(f"rms_tether_error_m: {validation['tether_error_m']['rms']:.9f}")
    ranges = validation["body_position_range_m"]
    print(f"position_range_m: x={ranges['x']:.6f} y={ranges['y']:.6f} z={ranges['z']:.6f}")
    if repeat["performed"]:
        print(f"repeat_max_abs_transform_delta: {repeat['max_abs_transform_delta']:.9g}")
        print(f"repeat_within_tolerance: {str(repeat['within_tolerance']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
