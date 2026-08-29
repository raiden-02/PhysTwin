"""IRIS falling-ball clip access.

Preparation sees video, ball radius, and drop-height context.
IRIS gravity is loaded only by evaluation, and only after a fit artifact exists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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


def _metric_slice(setting: dict[str, Any], name: str) -> dict[str, Any]:
    item = setting.get(name)
    if not isinstance(item, dict):
        raise KeyError(f"IRIS setting is missing {name}")
    return {key: item[key] for key in item if key != "gravity"}


def load_iris_falling_ball_input(project_root: Path) -> dict[str, Any]:
    """Resolve the IRIS falling_ball/big/01 clip for reconstruction and fitting.

    Returns video, measured ball radius, drop height, and clip window.
    Does not read or return IRIS gravity.
    """

    config = load_falling_ball_clip_config(project_root)
    dataset_root = iris_dataset_root(project_root)
    parameters = load_iris_parameters(dataset_root)
    setting = setting_parameters(parameters, config["class_key"], config["setting_key"])
    ball_radius_m = measured_mean_m(setting, "ball_radius")
    drop_height_m = measured_mean_m(setting, "drop_height")
    video = dataset_root / config["relative_video"]
    if not video.is_file():
        raise FileNotFoundError(f"IRIS clip is missing: {video}")
    document = {
        "dataset": "IRIS",
        "evidence_kind": EVIDENCE_KIND,
        "repo_id": IRIS_REPO_ID,
        "source_url": IRIS_SOURCE_URL,
        "class_key": config["class_key"],
        "setting_key": config["setting_key"],
        "relative_video": config["relative_video"],
        "video": video,
        "ball_radius": _metric_slice(setting, "ball_radius"),
        "drop_height": _metric_slice(setting, "drop_height"),
        "ball_radius_m": ball_radius_m,
        "drop_height_m": drop_height_m,
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
    assert_no_iris_gravity_truth(document)
    return document


def load_iris_gravity_for_evaluation(project_root: Path, *, fit_artifact: Path) -> float:
    """Read IRIS gravity after a fit artifact exists. Never used to set up the search."""

    if not fit_artifact.is_file():
        raise FileNotFoundError(
            "IRIS gravity is evaluation-only and requires an existing fit artifact: "
            f"{fit_artifact}"
        )
    config = load_falling_ball_clip_config(project_root)
    parameters = load_iris_parameters(iris_dataset_root(project_root))
    setting = setting_parameters(parameters, config["class_key"], config["setting_key"])
    return measured_mean_m(setting, "gravity")


def evaluation_gravity_m_s2(project_root: Path, *, fit_artifact: Path) -> float:
    return load_iris_gravity_for_evaluation(project_root, fit_artifact=fit_artifact)


def calibration_provenance(clip_input: dict[str, Any]) -> dict[str, Any]:
    provenance = {
        "evidence_kind": EVIDENCE_KIND,
        "dataset": clip_input["dataset"],
        "repo_id": clip_input["repo_id"],
        "source_url": clip_input["source_url"],
        "class_key": clip_input["class_key"],
        "setting_key": clip_input["setting_key"],
        "relative_video": clip_input["relative_video"],
        "iris_parameters_used_for_metric": {
            "ball_radius": clip_input["ball_radius"],
            "drop_height": clip_input["drop_height"],
        },
        "gravity_used_during_fit": False,
        "metric_value_m": clip_input["ball_radius_m"],
        "metric_name": "ball_radius",
    }
    assert_no_iris_gravity_truth(provenance)
    return provenance


def assert_no_iris_gravity_truth(document: Any) -> None:
    """Fail if a preparation or fit-input payload carries IRIS gravity truth."""

    if isinstance(document, Mapping):
        for key, value in document.items():
            if key == "gravity_truth_m_s2":
                raise AssertionError(f"IRIS gravity truth leaked through {key!r}")
            if key == "gravity" and isinstance(value, Mapping) and "mean" in value:
                raise AssertionError("IRIS gravity measurement leaked into fit preparation")
            assert_no_iris_gravity_truth(value)
        return
    if isinstance(document, (list, tuple)):
        for item in document:
            assert_no_iris_gravity_truth(item)
