"""IRIS external-dataset access for one P5R pendulum clip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IRIS_REPO_ID = "rasulkhanbayov/IRIS"
IRIS_SOURCE_URL = "https://huggingface.co/datasets/rasulkhanbayov/IRIS"
IRIS_DIRNAME = "IRIS"
CLIP_CONFIG_NAME = "p5r_iris_pendulum_45_01.json"
EVIDENCE_KIND = "external_dataset"


def iris_dataset_root(project_root: Path) -> Path:
    return project_root / "datasets" / IRIS_DIRNAME


def load_clip_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "contracts" / "3d" / "v1" / "examples" / CLIP_CONFIG_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def load_iris_parameters(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "parameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"IRIS parameters.json is missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def setting_parameters(
    parameters: dict[str, Any],
    class_key: str,
    setting_key: str,
) -> dict[str, Any]:
    classes = parameters.get(class_key)
    if not isinstance(classes, dict) or setting_key not in classes:
        raise KeyError(f"IRIS parameters.json has no {class_key}.{setting_key}")
    raw = classes[setting_key]
    if not isinstance(raw, dict):
        raise KeyError(f"IRIS {class_key}.{setting_key} is not an object")
    return raw


def measured_mean_m(setting: dict[str, Any], name: str) -> float:
    item = setting.get(name)
    if not isinstance(item, dict) or item.get("mean") is None:
        raise KeyError(f"IRIS setting is missing {name}.mean")
    value = float(item["mean"])
    if value <= 0.0:
        raise ValueError(f"IRIS {name}.mean must be > 0")
    return value


def load_iris_pendulum_benchmark(project_root: Path) -> dict[str, Any]:
    """Resolve the first supported IRIS pendulum clip and its measured rope length."""

    config = load_clip_config(project_root)
    dataset_root = iris_dataset_root(project_root)
    parameters = load_iris_parameters(dataset_root)
    setting = setting_parameters(parameters, config["class_key"], config["setting_key"])
    rope_length_m = measured_mean_m(setting, "rope_length")
    video = dataset_root / config["relative_video"]
    if not video.is_file():
        raise FileNotFoundError(f"IRIS clip is missing: {video}")
    measurement_source = (
        f"IRIS parameters.json {config['class_key']}.{config['setting_key']}"
        f".rope_length mean {rope_length_m:.2f} m"
    )
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
        "rope_length_m": rope_length_m,
        "angle_deg": float(setting["angle"]["mean"]) if "angle" in setting else None,
        "camera_to_cable_m": (
            float(setting["camera_to_cable"]["mean"])
            if "camera_to_cable" in setting
            else None
        ),
        "measurement_source": measurement_source,
        "held_fixed_parameter": "rest_length_m",
        "circular_with": config["circular_with"],
        "from_physical_point": config["from_physical_point"],
        "to_physical_point": config["to_physical_point"],
        "target_xy": [float(config["target_xy"][0]), float(config["target_xy"][1])],
        "anchor_xy": [float(config["anchor_xy"][0]), float(config["anchor_xy"][1])],
        "start_s": float(config["start_s"]),
        "duration_s": float(config["duration_s"]),
        "max_frames": int(config["max_frames"]),
        "up_mode": config["up_mode"],
        "up_source": config["up_source"],
        "seed_notes": config.get("seed_notes"),
    }


def iris_benchmark_review(project_root: Path) -> dict[str, Any]:
    """Classify the IRIS benchmark clip for footage inspect. Does not invent scale."""

    config = load_clip_config(project_root)
    dataset_root = iris_dataset_root(project_root)
    video = dataset_root / config["relative_video"]
    params_path = dataset_root / "parameters.json"
    present = video.is_file() and params_path.is_file()
    known = None
    reason = (
        f"IRIS {config['relative_video']} with parameters.json rope_length. "
        "external_dataset, not recorded_real."
    )
    if not present:
        missing = []
        if not params_path.is_file():
            missing.append("parameters.json")
        if not video.is_file():
            missing.append(config["relative_video"])
        reason = "IRIS files not on disk: " + ", ".join(missing)
    else:
        setting = setting_parameters(
            load_iris_parameters(dataset_root),
            config["class_key"],
            config["setting_key"],
        )
        known = measured_mean_m(setting, "rope_length")
    return {
        "id": "iris_pendulum_45_01",
        "path": str(video),
        "present": present,
        "kind": EVIDENCE_KIND,
        "tethered": True,
        "known_length_m": known,
        "eligible": bool(present and known is not None),
        "reason": reason,
    }


def calibration_provenance(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_kind": EVIDENCE_KIND,
        "dataset": benchmark["dataset"],
        "repo_id": benchmark["repo_id"],
        "source_url": benchmark["source_url"],
        "class_key": benchmark["class_key"],
        "setting_key": benchmark["setting_key"],
        "relative_video": benchmark["relative_video"],
        "iris_parameters": benchmark["parameters"],
        "held_fixed_because_it_established_scale": benchmark["held_fixed_parameter"],
    }
