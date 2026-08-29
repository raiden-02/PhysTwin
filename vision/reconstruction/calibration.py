"""Known-distance metric calibration. Guessed scales are rejected."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .contracts import (
    CONTRACT_VERSION,
    ContractError,
    _finite,
    _mapping,
    _require_fields,
    canonical_json_bytes,
)
from .entities import ENTITIES_EXTENSION, find_entity, validate_entities_v1

METRIC_CALIBRATION_SCHEMA = "phystwin.metric_calibration"
LENGTH_FIT_PROFILE = "tether_length_initial_tangent_velocity_v1"
FIXED_LENGTH_PROFILE = "tether_initial_tangent_velocity_fixed_length_v1"
ALLOWED_METHODS = {"known_scene_distance"}
ALLOWED_STATISTICS = {"median_distance"}
REJECTED_SOURCES = {
    "estimator",
    "guessed",
    "assumed",
    "diameter_guess",
    "random_object",
    "unknown",
    "fabricated",
}


def validate_metric_calibration(document: Any) -> Mapping[str, Any]:
    """Validate an external measured-distance scale document."""

    root = _mapping(document, "MetricCalibration")
    _require_fields(
        root,
        "MetricCalibration",
        {
            "schema",
            "version",
            "calibration_id",
            "method",
            "measured_length_m",
            "observed_length_world_units",
            "meters_per_world_unit",
            "pair",
            "measurement_source",
            "circular_with_fit_parameter",
            "provenance",
        },
        exact=True,
    )
    if root["schema"] != METRIC_CALIBRATION_SCHEMA or root["version"] != CONTRACT_VERSION:
        raise ContractError("MetricCalibration: unsupported schema or version")
    if not isinstance(root["calibration_id"], str) or not root["calibration_id"]:
        raise ContractError("MetricCalibration.calibration_id: must be a string")
    if root["method"] not in ALLOWED_METHODS:
        raise ContractError("MetricCalibration.method: unsupported")
    measured = _finite(root["measured_length_m"], "measured_length_m")
    observed = _finite(root["observed_length_world_units"], "observed_length_world_units")
    scale = _finite(root["meters_per_world_unit"], "meters_per_world_unit")
    if measured <= 0.0 or observed <= 0.0 or scale <= 0.0:
        raise ContractError("calibration lengths and scale must be > 0")
    expected = measured / observed
    if not math.isclose(scale, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("meters_per_world_unit must equal measured_length_m / observed length")
    pair = _mapping(root["pair"], "MetricCalibration.pair")
    _require_fields(pair, "MetricCalibration.pair", {"from_id", "to_id", "statistic"}, exact=True)
    if pair["statistic"] not in ALLOWED_STATISTICS:
        raise ContractError("calibration statistic is unsupported")
    if not isinstance(pair.get("from_id"), str) or not pair["from_id"]:
        raise ContractError("calibration pair.from_id: must be a string")
    if not isinstance(pair.get("to_id"), str) or not pair["to_id"]:
        raise ContractError("calibration pair.to_id: must be a string")
    if pair["from_id"] == pair["to_id"]:
        raise ContractError("calibration pair must connect two different entities")
    source = root["measurement_source"]
    if not isinstance(source, str) or not source.strip():
        raise ContractError("measurement_source must name the external measurement")
    if source.strip().lower() in REJECTED_SOURCES:
        raise ContractError(
            f"measurement_source {source!r} cannot produce metric_measured scale"
        )
    circular = root["circular_with_fit_parameter"]
    if circular not in {None, "rest_length_m"}:
        raise ContractError("circular_with_fit_parameter: unsupported")
    _mapping(root["provenance"], "MetricCalibration.provenance")
    canonical_json_bytes(root)
    return root


def median_entity_distance(
    entities: Mapping[str, Any],
    from_id: str,
    to_id: str,
) -> float:
    """Median 3D distance between two visible entity roots, in observation units."""

    payload = validate_entities_v1(entities)
    left = find_entity(payload, from_id)
    right = find_entity(payload, to_id)
    by_index = {
        int(sample["sample_index"]): sample
        for sample in right["samples"]
        if sample.get("visible", True)
    }
    distances: list[float] = []
    for sample in left["samples"]:
        if not sample.get("visible", True):
            continue
        other = by_index.get(int(sample["sample_index"]))
        if other is None:
            continue
        delta = [
            float(sample["root"][axis]) - float(other["root"][axis])
            for axis in range(3)
        ]
        distances.append(math.sqrt(sum(value * value for value in delta)))
    if len(distances) < 2:
        raise ContractError("calibration pair needs at least two overlapping visible samples")
    ordered = sorted(distances)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_known_distance_calibration(
    *,
    calibration_id: str,
    entities: Mapping[str, Any],
    from_id: str,
    to_id: str,
    measured_length_m: float,
    measurement_source: str,
    circular_with_fit_parameter: str | None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a measured-distance calibration from entity tracks plus a tape value."""

    if measured_length_m <= 0.0 or not math.isfinite(measured_length_m):
        raise ContractError("measured_length_m must be a finite value > 0")
    observed = median_entity_distance(entities, from_id, to_id)
    document = {
        "schema": METRIC_CALIBRATION_SCHEMA,
        "version": 1,
        "calibration_id": calibration_id,
        "method": "known_scene_distance",
        "measured_length_m": float(measured_length_m),
        "observed_length_world_units": float(observed),
        "meters_per_world_unit": float(measured_length_m) / float(observed),
        "pair": {
            "from_id": from_id,
            "to_id": to_id,
            "statistic": "median_distance",
        },
        "measurement_source": measurement_source,
        "circular_with_fit_parameter": circular_with_fit_parameter,
        "provenance": dict(provenance or {}),
    }
    return dict(validate_metric_calibration(document))


def apply_measured_scale(
    observation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Stamp metric_measured scale only from a validated external measurement."""

    validated = validate_metric_calibration(calibration)
    if observation["coordinates"]["scale"].get("status") == "metric_measured":
        current = observation["coordinates"]["scale"].get("meters_per_world_unit")
        if current is not None and not math.isclose(
            float(current),
            float(validated["meters_per_world_unit"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContractError("observation already has a different metric_measured scale")
    if ENTITIES_EXTENSION in observation.get("extensions", {}):
        expected = median_entity_distance(
            observation["extensions"][ENTITIES_EXTENSION],
            validated["pair"]["from_id"],
            validated["pair"]["to_id"],
        )
        if not math.isclose(
            expected,
            float(validated["observed_length_world_units"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError("calibration observed length does not match the entity tracks")
    document = dict(observation)
    coordinates = dict(document["coordinates"])
    coordinates["scale"] = {
        "status": "metric_measured",
        "meters_per_world_unit": float(validated["meters_per_world_unit"]),
        "source": "known_scene_distance",
    }
    document["coordinates"] = coordinates
    provenance = dict(document.get("provenance") or {})
    provenance["metric_calibration"] = {
        "calibration_id": validated["calibration_id"],
        "method": validated["method"],
        "measurement_source": validated["measurement_source"],
        "circular_with_fit_parameter": validated["circular_with_fit_parameter"],
        "measured_length_m": validated["measured_length_m"],
    }
    document["provenance"] = provenance
    return document


def reject_direct_metric_scale(status: str, source: str | None) -> None:
    """Refuse unmarked or guessed metric claims."""

    if status != "metric_measured":
        return
    if source is None or source.strip().lower() in REJECTED_SOURCES:
        raise ContractError(
            "metric_measured requires an external known-distance measurement_source"
        )


def circular_length_is_calibrator(calibration: Mapping[str, Any]) -> bool:
    """True when tether length established scale and must stay fixed in the fit."""

    if calibration.get("schema") == METRIC_CALIBRATION_SCHEMA:
        return validate_metric_calibration(calibration)["circular_with_fit_parameter"] == "rest_length_m"
    return calibration.get("circular_with_fit_parameter") == "rest_length_m"


def select_real_fit_profile(calibration: Mapping[str, Any] | None) -> str:
    """Use the fixed-length profile when tether length established metric scale."""

    if calibration is None:
        return LENGTH_FIT_PROFILE
    if circular_length_is_calibrator(calibration):
        return FIXED_LENGTH_PROFILE
    return LENGTH_FIT_PROFILE


def refuse_circular_length_fit(
    calibration: Mapping[str, Any] | None,
    profile: str,
) -> None:
    """Refuse claiming an independent rest-length recovery after length calibration."""

    if select_real_fit_profile(calibration) == FIXED_LENGTH_PROFILE and profile == LENGTH_FIT_PROFILE:
        raise ValueError(
            "tether length used for metric scale cannot be fitted as an independent parameter"
        )
