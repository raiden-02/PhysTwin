"""P5 inverse fitting for the minimal Newton fixed-distance scene."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from physics3d.bounded_search import (
    ParameterSpec,
    SearchResult,
    bounded_differential_search,
)
from physics3d.newton_runtime import (
    simulate_body_positions,
    simulate_physical_scene,
    transform_point,
)
from vision.reconstruction.calibration import (
    refuse_circular_length_fit,
    select_real_fit_profile,
)
from vision.reconstruction.contracts import (
    canonical_json_bytes,
    validate_inverse_fit_artifacts,
    validate_inverse_physics_fit,
    validate_physical_motion_observation,
    validate_physical_scene,
    validate_rollout_source,
)


PROFILE = "tether_length_initial_tangent_velocity_v1"
FIXED_LENGTH_PROFILE = "tether_initial_tangent_velocity_fixed_length_v1"
DEFAULT_SEED = 0x50545935
DEFAULT_PARAMETERS = (
    ParameterSpec("rest_length_m", 1.6, 2.4, 1.78, "meter"),
    ParameterSpec("initial_tangent_velocity_u_m_s", -0.6, 0.6, -0.12, "meter_per_second"),
    ParameterSpec("initial_tangent_velocity_v_m_s", -0.6, 0.6, 0.14, "meter_per_second"),
)
SYNTHETIC_NORMALIZED_RMSE_LIMIT = 0.02
REAL_SCENE_SOURCE_KINDS = {
    "scene_observation_human_root",
    "scene_observation_entity_root",
}


def parameter_specs_for_profile(
    profile: str,
    *,
    rest_length_m: float | None = None,
) -> tuple[ParameterSpec, ...]:
    """Return the three P5 parameters, holding rest length only for the fixed-length profile."""

    if profile == PROFILE:
        return DEFAULT_PARAMETERS
    if profile == FIXED_LENGTH_PROFILE:
        rest = DEFAULT_PARAMETERS[0]
        length = float(rest.initial if rest_length_m is None else rest_length_m)
        lower = min(rest.lower_bound, length)
        upper = max(rest.upper_bound, length)
        if not lower < upper:
            lower, upper = length - 0.05, length + 0.05
        return (
            ParameterSpec(rest.id, lower, upper, length, rest.unit, True),
            DEFAULT_PARAMETERS[1],
            DEFAULT_PARAMETERS[2],
        )
    raise ValueError(f"unsupported fit profile {profile!r}")


def _normalize(vector: Sequence[float], path: str) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-9:
        raise ValueError(f"{path} has near-zero length")
    return tuple(float(value) / norm for value in vector)  # type: ignore[return-value]


def _cross(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def tether_parameter_basis(
    scene: Mapping[str, Any],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return radial, tangent-u, and tangent-v unit vectors."""

    body = scene["model"]["bodies"][0]
    constraint = scene["model"]["constraints"][0]
    attachment_world = transform_point(
        body["T_world_body_initial"],
        constraint["body_attachment_m"],
    )
    anchor = constraint["world_anchor_m"]
    radial = _normalize(
        [attachment_world[axis] - anchor[axis] for axis in range(3)],
        "initial tether radial direction",
    )
    tangent_u_raw = _cross(radial, (0.0, 1.0, 0.0))
    if math.sqrt(sum(value * value for value in tangent_u_raw)) <= 1e-6:
        tangent_u_raw = _cross(radial, (1.0, 0.0, 0.0))
    tangent_u = _normalize(tangent_u_raw, "tangent u")
    tangent_v = _normalize(_cross(radial, tangent_u), "tangent v")
    return radial, tangent_u, tangent_v


def apply_tether_parameters(
    template_scene: Mapping[str, Any],
    values: Mapping[str, float],
) -> dict[str, Any]:
    """Patch the three P5 values while preserving valid initial geometry."""

    scene = copy.deepcopy(template_scene)
    body = scene["model"]["bodies"][0]
    constraint = scene["model"]["constraints"][0]
    radial, tangent_u, tangent_v = tether_parameter_basis(template_scene)
    rest_length = float(values["rest_length_m"])
    velocity_u = float(values["initial_tangent_velocity_u_m_s"])
    velocity_v = float(values["initial_tangent_velocity_v_m_s"])
    anchor = constraint["world_anchor_m"]
    transform = body["T_world_body_initial"]
    current_attachment = transform_point(
        transform,
        constraint["body_attachment_m"],
    )
    target_attachment = [
        float(anchor[axis]) + rest_length * radial[axis] for axis in range(3)
    ]
    for axis, matrix_index in enumerate((3, 7, 11)):
        transform[matrix_index] += target_attachment[axis] - current_attachment[axis]
    body["linear_velocity_m_s"] = [
        velocity_u * tangent_u[axis] + velocity_v * tangent_v[axis]
        for axis in range(3)
    ]
    constraint["rest_length_m"] = rest_length
    validate_physical_scene(scene)
    return scene


def _trajectory_metrics(
    observed: np.ndarray,
    simulated: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    squared_distance = np.sum((simulated - observed) ** 2, axis=1)
    mse = float(np.sum(weights * squared_distance) / np.sum(weights))
    ranges = np.ptp(observed, axis=0)
    extent = float(np.linalg.norm(ranges))
    rmse = math.sqrt(mse)
    return {
        "mse_m2": mse,
        "rmse_m": rmse,
        "trajectory_extent_m": extent,
        "normalized_rmse": rmse / extent if extent > 0.0 else math.inf,
    }


def _parameter_rows(
    specs: Sequence[ParameterSpec],
    *,
    fitted: Sequence[float] | None,
    truth: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "id": spec.id,
            "unit": spec.unit,
            "lower_bound": spec.lower_bound,
            "upper_bound": spec.upper_bound,
            "initial": spec.initial,
            "fitted": None if fitted is None else float(fitted[index]),
            "truth": None if truth is None else float(truth[spec.id]),
            "held_fixed": spec.held_fixed,
        }
        for index, spec in enumerate(specs)
    ]


def blocked_fit_report(
    template_scene: Mapping[str, Any],
    scene_observation: Mapping[str, Any],
    blockers: Sequence[str],
    *,
    profile: str = PROFILE,
    parameters: Sequence[ParameterSpec] | None = None,
) -> dict[str, Any]:
    """Return a validated report without inventing metric observations."""

    specs = tuple(parameters) if parameters is not None else parameter_specs_for_profile(profile)
    template_hash = hashlib.sha256(canonical_json_bytes(template_scene)).hexdigest()
    observation_hash = hashlib.sha256(canonical_json_bytes(scene_observation)).hexdigest()
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "template": template_hash,
                "scene_observation": observation_hash,
                "profile": profile,
            }
        )
    ).hexdigest()
    report = {
        "schema": "phystwin.inverse_physics_fit",
        "version": 1,
        "fit_id": f"blocked-{identity[:12]}",
        "status": "BLOCKED_INPUT",
        "source": {
            "template_physical_scene": {
                "id": template_scene["scene_id"],
                "sha256": template_hash,
            },
            "motion_observation": None,
            "scene_observation": {
                "id": scene_observation["observation_id"],
                "sha256": observation_hash,
            },
        },
        "profile": profile,
        "objective": {
            "type": "weighted_position_mse_3d",
            "sample_count": 0,
            "mse_m2": None,
            "rmse_m": None,
            "trajectory_extent_m": None,
            "normalized_rmse": None,
            "initial_mse_m2": None,
            "improvement_ratio": None,
        },
        "parameters": _parameter_rows(
            specs,
            fitted=None,
            truth=None,
        ),
        "optimizer": {
            "method": "bounded_differential_evolution_with_coordinate_refinement",
            "seed": DEFAULT_SEED,
            "population_size": 0,
            "generations": 0,
            "coordinate_iterations": 0,
            "objective_evaluations": 0,
        },
        "outputs": {
            "fitted_physical_scene": None,
            "simulated_world_state": None,
        },
        "execution": {
            "wall_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
        },
        "validation": {
            "passed": False,
            "rollout_valid": False,
            "synthetic_recovery": {
                "performed": False,
                "within_tolerance": None,
                "max_normalized_parameter_error": None,
            },
        },
        "blockers": list(blockers),
        "warnings": [],
        "failures": [],
    }
    validate_inverse_physics_fit(report)
    return report


def fit_tether_scene(
    template_scene: Mapping[str, Any],
    motion_observation: Mapping[str, Any],
    *,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    population_size: int = 8,
    generations: int = 4,
    coordinate_iterations: int = 12,
    repeat_check: bool = True,
    profile: str = PROFILE,
    parameters: Sequence[ParameterSpec] | None = None,
) -> dict[str, Any]:
    """Fit a supported tether profile and write a standard scene, rollout, and report."""

    template = dict(validate_physical_scene(template_scene))
    motion = dict(validate_physical_motion_observation(motion_observation))
    calibration = motion.get("provenance", {}).get("calibration")
    refuse_circular_length_fit(
        calibration if isinstance(calibration, Mapping) else None,
        profile,
    )
    rest_length = float(template["model"]["constraints"][0]["rest_length_m"])
    if (
        isinstance(calibration, Mapping)
        and calibration.get("circular_with_fit_parameter") == "rest_length_m"
        and calibration.get("measured_length_m") is not None
    ):
        rest_length = float(calibration["measured_length_m"])
        template["model"]["constraints"][0]["rest_length_m"] = rest_length
        template = dict(validate_physical_scene(template))
    specs = (
        tuple(parameters)
        if parameters is not None
        else parameter_specs_for_profile(profile, rest_length_m=rest_length)
    )
    if motion["track"]["body_id"] != template["model"]["bodies"][0]["id"]:
        raise ValueError("motion observation body_id does not match PhysicalScene")
    samples = motion["track"]["samples"]
    timestamps = [float(sample["timestamp_s"]) for sample in samples]
    observed = np.asarray([sample["position_m"] for sample in samples], dtype=np.float64)
    weights = np.asarray([sample["weight"] for sample in samples], dtype=np.float64)
    if timestamps[0] < float(template["execution"]["start_time_s"]):
        raise ValueError("motion observation starts before the PhysicalScene")
    if timestamps[-1] > (
        float(template["execution"]["start_time_s"])
        + float(template["execution"]["duration_s"])
        + 1e-12
    ):
        raise ValueError("motion observation ends after the PhysicalScene")

    peak_gpu_memory = 0

    def objective(values: tuple[float, ...]) -> float:
        nonlocal peak_gpu_memory
        candidate = apply_tether_parameters(
            template,
            {spec.id: values[index] for index, spec in enumerate(specs)},
        )
        run = simulate_body_positions(candidate, timestamps)
        peak_gpu_memory = max(peak_gpu_memory, run.peak_gpu_memory_bytes)
        simulated = np.asarray(run.positions_m, dtype=np.float64)
        return _trajectory_metrics(observed, simulated, weights)["mse_m2"]

    started = time.perf_counter()
    search: SearchResult = bounded_differential_search(
        objective,
        specs,
        seed=seed,
        population_size=population_size,
        generations=generations,
        coordinate_iterations=coordinate_iterations,
    )
    fitted_values = {
        spec.id: search.values[index]
        for index, spec in enumerate(specs)
    }
    fitted_scene = apply_tether_parameters(template, fitted_values)
    fitted_run = simulate_body_positions(fitted_scene, timestamps)
    fitted_positions = np.asarray(fitted_run.positions_m, dtype=np.float64)
    metrics = _trajectory_metrics(observed, fitted_positions, weights)
    rollout = simulate_physical_scene(fitted_scene, repeat_check=repeat_check)
    validate_rollout_source(rollout, fitted_scene)
    peak_gpu_memory = max(
        peak_gpu_memory,
        fitted_run.peak_gpu_memory_bytes,
        int(rollout["execution"]["peak_gpu_memory_bytes"]),
    )

    truth_raw = motion.get("provenance", {}).get("truth_parameters")
    is_synthetic_source = (
        motion["source"]["kind"] == "synthetic_rollout"
        and motion.get("provenance", {}).get("synthetic") is True
    )
    if truth_raw is not None and not is_synthetic_source:
        raise ValueError("truth_parameters are allowed only for synthetic rollout evidence")
    truth = (
        {
            spec.id: float(truth_raw[spec.id])
            for spec in specs
        }
        if isinstance(truth_raw, Mapping)
        and all(spec.id in truth_raw for spec in specs)
        else None
    )
    normalized_errors = (
        [
            abs(fitted_values[spec.id] - truth[spec.id])
            / (spec.upper_bound - spec.lower_bound)
            for spec in specs
        ]
        if truth is not None
        else []
    )
    max_parameter_error = max(normalized_errors) if normalized_errors else None
    recovery_ok = max_parameter_error is not None and max_parameter_error <= 0.03
    warnings = [
        "Mass is fixed because gravity-only ideal tether motion does not identify it.",
        "Damping is not fitted because the P4 runtime has no validated damping parameter.",
    ]
    if is_synthetic_source:
        validation_passed = (
            metrics["normalized_rmse"] <= SYNTHETIC_NORMALIZED_RMSE_LIMIT
            and (truth is None or recovery_ok)
        )
        if not validation_passed:
            raise RuntimeError(
                "P5 fit failed validation: "
                f"normalized_rmse={metrics['normalized_rmse']:.6g}, "
                f"max_normalized_parameter_error={max_parameter_error}"
            )
    else:
        validation_passed = True
        if metrics["normalized_rmse"] > SYNTHETIC_NORMALIZED_RMSE_LIMIT:
            warnings.append(
                "normalized RMSE "
                f"{metrics['normalized_rmse']:.6g} exceeds the synthetic "
                f"{SYNTHETIC_NORMALIZED_RMSE_LIMIT} check. Reported honestly, not a hard fail."
            )
        if any(spec.held_fixed for spec in specs):
            warnings.append(
                "held_fixed parameters were not independently recovered by the optimizer."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_bytes = _json_bytes(fitted_scene)
    rollout_bytes = _json_bytes(rollout)
    scene_path = output_dir / "fitted_physical_scene.json"
    rollout_path = output_dir / "simulated_world_state.json"
    _atomic_write(scene_path, scene_bytes)
    _atomic_write(rollout_path, rollout_bytes)
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    rollout_hash = hashlib.sha256(rollout_bytes).hexdigest()
    template_hash = hashlib.sha256(canonical_json_bytes(template)).hexdigest()
    motion_hash = hashlib.sha256(canonical_json_bytes(motion)).hexdigest()
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "template": template_hash,
                "motion": motion_hash,
                "profile": profile,
                "seed": seed,
                "budget": [population_size, generations, coordinate_iterations],
            }
        )
    ).hexdigest()
    report = {
        "schema": "phystwin.inverse_physics_fit",
        "version": 1,
        "fit_id": f"p5-{identity[:12]}",
        "status": "COMPLETE",
        "source": {
            "template_physical_scene": {
                "id": template["scene_id"],
                "sha256": template_hash,
            },
            "motion_observation": {
                "id": motion["observation_id"],
                "sha256": motion_hash,
            },
            "scene_observation": (
                {
                    "id": motion["source"]["id"],
                    "sha256": motion["source"]["sha256"],
                }
                if motion["source"]["kind"] in REAL_SCENE_SOURCE_KINDS
                else None
            ),
        },
        "profile": profile,
        "objective": {
            "type": "weighted_position_mse_3d",
            "sample_count": len(samples),
            **metrics,
            "initial_mse_m2": search.initial_objective,
            "improvement_ratio": (
                search.initial_objective / max(metrics["mse_m2"], 1e-30)
            ),
        },
        "parameters": _parameter_rows(
            specs,
            fitted=search.values,
            truth=truth,
        ),
        "optimizer": {
            "method": "bounded_differential_evolution_with_coordinate_refinement",
            "seed": seed,
            "population_size": population_size,
            "generations": search.generations,
            "coordinate_iterations": search.coordinate_iterations,
            "objective_evaluations": search.objective_evaluations,
        },
        "outputs": {
            "fitted_physical_scene": {
                "uri": scene_path.name,
                "sha256": scene_hash,
            },
            "simulated_world_state": {
                "uri": rollout_path.name,
                "sha256": rollout_hash,
            },
        },
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory,
        },
        "validation": {
            "passed": True,
            "rollout_valid": True,
            "synthetic_recovery": {
                "performed": truth is not None,
                "within_tolerance": recovery_ok if truth is not None else None,
                "max_normalized_parameter_error": max_parameter_error,
            },
        },
        "blockers": [],
        "warnings": warnings,
        "failures": [],
    }
    validate_inverse_physics_fit(report)
    validate_inverse_fit_artifacts(
        report,
        template,
        motion,
        fitted_scene,
        rollout,
        fitted_scene_path=scene_path,
        rollout_path=rollout_path,
    )
    _atomic_write(output_dir / "inverse_physics_fit.json", _json_bytes(report))
    return {
        "fit": report,
        "physical_scene": fitted_scene,
        "rollout": rollout,
        "motion_observation": motion,
    }


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
