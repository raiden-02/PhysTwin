#!/usr/bin/env python3
"""Compare a completed free-fall fit to IRIS gravity. Gravity is unread until now."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-dir",
        type=Path,
        default=ROOT / "results" / "physics3d" / "p5r-falling-ball",
    )
    return parser.parse_args()


def main() -> int:
    from vision.reconstruction.iris_falling import (
        evaluation_gravity_m_s2,
        load_iris_falling_ball_benchmark,
    )

    args = parse_args()
    fit_dir = args.fit_dir.resolve()
    fit = json.loads((fit_dir / "inverse_physics_fit.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (fit_dir / "iris_p5r_falling_ball_evidence.json").read_text(encoding="utf-8")
    )
    benchmark = load_iris_falling_ball_benchmark(ROOT)
    truth = evaluation_gravity_m_s2(benchmark)
    recovered = next(
        item["fitted"]
        for item in fit["parameters"]
        if item["id"] == "gravity_magnitude_m_s2"
    )
    velocity = next(
        item["fitted"]
        for item in fit["parameters"]
        if item["id"] == "initial_velocity_y_m_s"
    )
    abs_error = abs(float(recovered) - truth)
    percent_error = 100.0 * abs_error / truth
    bound_hits = [
        item["id"]
        for item in fit["parameters"]
        if math.isclose(item["fitted"], item["lower_bound"], abs_tol=1e-6)
        or math.isclose(item["fitted"], item["upper_bound"], abs_tol=1e-6)
    ]
    objective = fit["objective"]
    gate = {
        "ideal": percent_error <= 10.0 and objective["normalized_rmse"] <= 0.10,
        "acceptable": percent_error <= 20.0 and objective["normalized_rmse"] <= 0.20,
        "bound_hit": bool(bound_hits),
    }
    report = {
        "dataset": evidence["dataset"],
        "relative_video": evidence["relative_video"],
        "video_sha256": evidence["video_sha256"],
        "ball_radius_m": evidence["ball_radius_m"],
        "reconstruction_method": evidence["reconstruction_method"],
        "accepted_frames": evidence["accepted_frames"],
        "rejected_frames": evidence["rejected_frames"],
        "gravity_truth_m_s2": truth,
        "recovered_gravity_m_s2": float(recovered),
        "gravity_abs_error_m_s2": abs_error,
        "gravity_percent_error": percent_error,
        "fitted_initial_velocity_y_m_s": float(velocity),
        "rmse_m": objective["rmse_m"],
        "normalized_rmse": objective["normalized_rmse"],
        "trajectory_extent_m": objective["trajectory_extent_m"],
        "initial_mse_m2": objective["initial_mse_m2"],
        "final_mse_m2": objective["mse_m2"],
        "improvement_ratio": objective["improvement_ratio"],
        "objective_evaluations": fit["optimizer"]["objective_evaluations"],
        "wall_seconds": fit["execution"]["wall_seconds"],
        "bound_hits": bound_hits,
        "gate": gate,
        "gravity_used_during_fit": False,
    }
    out = fit_dir / "falling_ball_evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if gate["ideal"] or (gate["acceptable"] and not gate["bound_hit"]):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
