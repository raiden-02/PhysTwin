#!/usr/bin/env python3
"""Write the compact falling-ball demo payload from saved local results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positions_from_rollout(path: Path) -> list[list[float]]:
    rollout = json.loads(path.read_text(encoding="utf-8"))
    return [
        [
            float(sample["T_world_body"][3]),
            float(sample["T_world_body"][7]),
            float(sample["T_world_body"][11]),
        ]
        for sample in rollout["bodies"][0]["samples"]
    ]


def times_from_rollout(path: Path) -> list[float]:
    rollout = json.loads(path.read_text(encoding="utf-8"))
    return [float(sample["timestamp_s"]) for sample in rollout["timeline"]["samples"]]


def main() -> int:
    fit_dir = ROOT / "results" / "physics3d" / "p5r-falling-ball"
    motion = json.loads((fit_dir / "target_motion_observation.json").read_text(encoding="utf-8"))
    evaluation = json.loads((fit_dir / "falling_ball_evaluation.json").read_text(encoding="utf-8"))
    counterfactual = json.loads((fit_dir / "counterfactual_moon.json").read_text(encoding="utf-8"))
    fit = json.loads((fit_dir / "inverse_physics_fit.json").read_text(encoding="utf-8"))
    fitted_times = times_from_rollout(fit_dir / "simulated_world_state.json")
    moon_times = times_from_rollout(fit_dir / "counterfactual_moon_simulated_world_state.json")
    payload = {
        "title": "IRIS Falling Ball",
        "dataset": evaluation["dataset"],
        "relative_video": evaluation["relative_video"],
        "video_start_s": 1.4,
        "video_duration_s": 0.5,
        "gpu": "NVIDIA GeForce RTX 4080 SUPER",
        "metrics": {
            "recovered_gravity_m_s2": evaluation["recovered_gravity_m_s2"],
            "iris_gravity_m_s2": evaluation["gravity_truth_m_s2"],
            "gravity_percent_error": evaluation["gravity_percent_error"],
            "rmse_m": evaluation["rmse_m"],
            "normalized_rmse": evaluation["normalized_rmse"],
        },
        "observed": {
            "times_s": [float(sample["timestamp_s"]) for sample in motion["track"]["samples"]],
            "positions_m": [sample["position_m"] for sample in motion["track"]["samples"]],
        },
        "fitted": {
            "times_s": fitted_times,
            "positions_m": positions_from_rollout(fit_dir / "simulated_world_state.json"),
        },
        "moon": {
            "times_s": moon_times,
            "positions_m": positions_from_rollout(
                fit_dir / "counterfactual_moon_simulated_world_state.json"
            ),
            "gravity_m_s2": counterfactual["counterfactual_value"],
            "source_fitted_scene_sha256": counterfactual["source_fitted_scene_sha256"],
            "rollout_sha256": counterfactual["counterfactual_rollout_sha256"],
        },
        "hashes": {
            "video_sha256": evaluation["video_sha256"],
            "fitted_scene_sha256": fit["outputs"]["fitted_physical_scene"]["sha256"],
            "fitted_rollout_sha256": fit["outputs"]["simulated_world_state"]["sha256"],
            "moon_scene_sha256": counterfactual["counterfactual_scene_sha256"],
            "moon_rollout_sha256": counterfactual["counterfactual_rollout_sha256"],
        },
    }
    out = ROOT / "docs" / "evaluation" / "iris-p5r-falling-ball-demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
