"""IRIS falling-ball clip access. Gravity is evaluation ground truth only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .iris import (
    EVIDENCE_KIND,
    IRIS_REPO_ID,
    IRIS_SOURCE_URL,
    iris_dataset_root,
    load_iris_parameters,
    measured_mean_m,
    setting_parameters,
)


CLIP_CONFIG_NAME = "p5r_iris_falling_ball_big_01.json"


def load_falling_ball_clip_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "contracts" / "3d" / "v1" / "examples" / CLIP_CONFIG_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def load_iris_falling_ball_benchmark(project_root: Path) -> dict[str, Any]:
    """Resolve the IRIS falling_ball/big/01 clip and its measured ball radius.

    Gravity is returned only as evaluation ground truth. Do not pass it into
    the fitter initial value or bounds.
    """

    config = load_falling_ball_clip_config(project_root)
    dataset_root = iris_dataset_root(project_root)
    parameters = load_iris_parameters(dataset_root)
    setting = setting_parameters(parameters, config["class_key"], config["setting_key"])
    ball_radius_m = measured_mean_m(setting, "ball_radius")
    drop_height_m = measured_mean_m(setting, "drop_height")
    gravity_truth_m_s2 = measured_mean_m(setting, "gravity")
    video = dataset_root / config["relative_video"]
    if not video.is_file():
        raise FileNotFoundError(f"IRIS clip is missing: {video}")
    return {
        "dataset": "IRIS",
        "evidence_kind": EVIDENCE_KIND,
        "repo_id": IRIS_REPO_ID,
        "source_url": IRIS_SOURCE_URL,
        "class_key": config["class_key"],
        "setting_key": config["setting_key"],
        "relative_video": config["relative_video"],
        "video": video,
        "parameters": setting,
        "ball_radius_m": ball_radius_m,
        "drop_height_m": drop_height_m,
        "gravity_truth_m_s2": gravity_truth_m_s2,
        "measurement_source": (
            f"IRIS parameters.json {config['class_key']}.{config['setting_key']}"
            f".ball_radius mean {ball_radius_m:.2f} m"
        ),
        "target_xy": [float(config["target_xy"][0]), float(config["target_xy"][1])],
        "start_s": float(config["start_s"]),
        "duration_s": float(config["duration_s"]),
        "max_frames": int(config["max_frames"]),
        "up_mode": config["up_mode"],
        "up_source": config["up_source"],
        "static_camera": bool(config.get("static_camera", True)),
        "seed_notes": config.get("seed_notes"),
    }


def evaluation_gravity_m_s2(benchmark: dict[str, Any]) -> float:
    """Read IRIS gravity after fitting. Never feed this into optimizer setup."""

    return float(benchmark["gravity_truth_m_s2"])


def calibration_provenance(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_kind": EVIDENCE_KIND,
        "dataset": benchmark["dataset"],
        "repo_id": benchmark["repo_id"],
        "source_url": benchmark["source_url"],
        "class_key": benchmark["class_key"],
        "setting_key": benchmark["setting_key"],
        "relative_video": benchmark["relative_video"],
        "iris_parameters_used_for_metric": {
            "ball_radius": benchmark["parameters"]["ball_radius"],
            "drop_height": benchmark["parameters"]["drop_height"],
        },
        "gravity_used_during_fit": False,
        "metric_value_m": benchmark["ball_radius_m"],
        "metric_name": "ball_radius",
    }
