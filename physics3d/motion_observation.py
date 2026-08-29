"""Build P5 metric point evidence from rollouts or eligible SceneObservation data."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from vision.reconstruction.contracts import (
    canonical_json_bytes,
    validate_physical_motion_observation,
    validate_physical_scene,
    validate_scene_observation,
)
from vision.reconstruction.calibration import REJECTED_SOURCES
from vision.reconstruction.entities import (
    ENTITIES_EXTENSION,
    find_entity,
    validate_entities_v1,
)
from vision.reconstruction.humans import HUMANS_EXTENSION, validate_humans_v1
from vision.reconstruction.transforms import transform_point

MIN_SPATIAL_EXTENT_M = 0.02


class FitInputBlocked(RuntimeError):
    """The available observation cannot support an honest metric P5 fit."""

    def __init__(self, blockers: list[str]) -> None:
        super().__init__("; ".join(blockers))
        self.blockers = tuple(blockers)


def axis_ranges_m(samples: list[Mapping[str, Any]]) -> list[float]:
    """AABB side lengths of body-origin samples, in meters."""

    return [
        max(sample["position_m"][axis] for sample in samples)
        - min(sample["position_m"][axis] for sample in samples)
        for axis in range(3)
    ]


def spatial_extent_m(samples: list[Mapping[str, Any]]) -> float:
    """AABB diagonal. Full 3D state. Planar travel still counts."""

    ranges = axis_ranges_m(samples)
    return math.sqrt(sum(value * value for value in ranges))


def require_nontrivial_spatial_motion(
    samples: list[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Block near-stationary tracks. Do not require travel on every axis."""

    ranges = axis_ranges_m(samples)
    extent = math.sqrt(sum(value * value for value in ranges))
    if extent < MIN_SPATIAL_EXTENT_M:
        raise FitInputBlocked(
            [
                f"{label} requires at least {MIN_SPATIAL_EXTENT_M} m of spatial travel "
                f"(AABB diagonal). Planar motion is allowed. Degenerate tracks are not. "
                f"got extent={extent:.6g} ranges={ranges}"
            ]
        )


def motion_observation_from_rollout(
    rollout: Mapping[str, Any],
    *,
    stride: int = 8,
    truth_parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Decimate one validated synthetic rollout into metric body-origin evidence."""

    if stride <= 0:
        raise ValueError("stride must be > 0")
    body = rollout["bodies"][0]
    timeline = rollout["timeline"]["samples"]
    body_samples = body["samples"]
    indices = list(range(0, len(timeline), stride))
    if indices[-1] != len(timeline) - 1:
        indices.append(len(timeline) - 1)
    rollout_hash = hashlib.sha256(canonical_json_bytes(rollout)).hexdigest()
    document = {
        "schema": "phystwin.physical_motion_observation",
        "version": 1,
        "observation_id": f"{rollout['rollout_id']}-motion",
        "source": {
            "kind": "synthetic_rollout",
            "id": rollout["rollout_id"],
            "sha256": rollout_hash,
        },
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Y",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
        },
        "units": {"length": "meter", "time": "second"},
        "track": {
            "body_id": body["id"],
            "point": "body_origin",
            "samples": [
                {
                    "sample_index": output_index,
                    "timestamp_s": float(timeline[source_index]["timestamp_s"]),
                    "position_m": [
                        float(body_samples[source_index]["T_world_body"][3]),
                        float(body_samples[source_index]["T_world_body"][7]),
                        float(body_samples[source_index]["T_world_body"][11]),
                    ],
                    "weight": 1.0,
                }
                for output_index, source_index in enumerate(indices)
            ],
        },
        "provenance": {
            "synthetic": True,
            "generator": "Newton/Warp SimulatedWorldState decimation",
            "sample_stride": stride,
            "noise_sigma_m": 0.0,
            "truth_parameters": (
                None
                if truth_parameters is None
                else {
                    key: float(value)
                    for key, value in truth_parameters.items()
                }
            ),
        },
        "warnings": [],
    }
    validate_physical_motion_observation(document)
    return document


def scene_observation_blockers(
    observation: Mapping[str, Any],
    template_scene: Mapping[str, Any],
    *,
    person_id: str | None = None,
) -> list[str]:
    """Return every reason a P1/P2 observation is ineligible for P5 fitting."""

    blockers: list[str] = []
    try:
        validate_scene_observation(observation)
    except Exception as error:
        return [f"invalid SceneObservation: {error}"]
    try:
        validate_physical_scene(template_scene)
    except Exception as error:
        return [f"invalid PhysicalScene template: {error}"]

    scale = observation["coordinates"]["scale"]
    if scale.get("status") != "metric_measured":
        blockers.append(
            "SceneObservation scale must be metric_measured; "
            f"got {scale.get('status')}"
        )
    meters = scale.get("meters_per_world_unit")
    if not isinstance(meters, (int, float)) or isinstance(meters, bool) or not math.isfinite(float(meters)):
        blockers.append("SceneObservation has no finite meters_per_world_unit")

    alignment = template_scene["observation_alignment"]
    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    if alignment.get("observation_sha256") != observation_hash:
        blockers.append("PhysicalScene observation_sha256 does not match SceneObservation")
    if alignment.get("scale_source") != "measured":
        blockers.append("PhysicalScene scale_source must be measured")
    if alignment.get("alignment_source") != "measured":
        blockers.append("PhysicalScene alignment_source must be measured")
    if alignment.get("T_scene_observation_m") is None:
        blockers.append("PhysicalScene has no measured T_scene_observation_m")
    if isinstance(meters, (int, float)) and not isinstance(meters, bool):
        scene_meters = alignment.get("meters_per_observation_unit")
        if not isinstance(scene_meters, (int, float)) or not math.isclose(
            float(scene_meters),
            float(meters),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("PhysicalScene and SceneObservation metric scales do not match")

    humans_raw = observation.get("extensions", {}).get(HUMANS_EXTENSION)
    if humans_raw is None:
        blockers.append("SceneObservation has no humans.v1 body track")
        return blockers
    try:
        humans = validate_humans_v1(
            humans_raw,
            sample_count=len(observation["timeline"]["samples"]),
        )
    except Exception as error:
        blockers.append(f"invalid humans.v1 evidence: {error}")
        return blockers
    person = next(
        (
            value
            for value in humans["people"]
            if person_id is None or value["id"] == person_id
        ),
        None,
    )
    if person is None:
        blockers.append(f"humans.v1 person {person_id!r} was not found")
        return blockers
    visible = sorted(
        (
            sample
            for sample in person["samples"]
            if sample.get("visible", True)
        ),
        key=lambda sample: int(sample["sample_index"]),
    )
    if len(visible) < 12:
        blockers.append("P5 requires at least 12 visible human-root samples")
    timeline_by_index = {
        int(sample["sample_index"]): sample for sample in observation["timeline"]["samples"]
    }
    times = [
        float(timeline_by_index[int(sample["sample_index"])]["timestamp_s"])
        for sample in visible
        if int(sample["sample_index"]) in timeline_by_index
    ]
    if len(times) >= 2 and times[-1] - times[0] < 0.5:
        blockers.append("P5 requires at least 0.5 seconds of visible motion")
    if any(times[index] >= times[index + 1] for index in range(len(times) - 1)):
        blockers.append("P5 human-root timestamps must be strictly increasing")
    scene_start = float(template_scene["execution"]["start_time_s"])
    scene_end = scene_start + float(template_scene["execution"]["duration_s"])
    if times and (
        times[0] < scene_start - 1e-12
        or times[-1] > scene_end + 1e-12
    ):
        blockers.append(
            "P5 human-root timestamps must lie inside the PhysicalScene timeline"
        )
    attachment = template_scene["model"]["constraints"][0]["body_attachment_m"]
    if any(abs(float(value)) > 1e-12 for value in attachment):
        blockers.append("human-root fitting requires body_attachment_m at the body origin")
    return blockers


def entity_observation_blockers(
    observation: Mapping[str, Any],
    template_scene: Mapping[str, Any],
    *,
    entity_id: str | None = None,
) -> list[str]:
    """Return every reason a P5R entity observation is ineligible for fitting."""

    blockers: list[str] = []
    try:
        validate_scene_observation(observation)
    except Exception as error:
        return [f"invalid SceneObservation: {error}"]
    try:
        validate_physical_scene(template_scene)
    except Exception as error:
        return [f"invalid PhysicalScene template: {error}"]

    scale = observation["coordinates"]["scale"]
    if scale.get("status") != "metric_measured":
        blockers.append(
            "SceneObservation scale must be metric_measured; "
            f"got {scale.get('status')}"
        )
    source = scale.get("source")
    if source is None or str(source).strip().lower() in REJECTED_SOURCES:
        blockers.append(
            "metric_measured scale must come from a known-distance measurement, "
            f"got source {source!r}"
        )
    meters = scale.get("meters_per_world_unit")
    if not isinstance(meters, (int, float)) or isinstance(meters, bool) or not math.isfinite(float(meters)):
        blockers.append("SceneObservation has no finite meters_per_world_unit")

    alignment = template_scene["observation_alignment"]
    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    if alignment.get("observation_sha256") != observation_hash:
        blockers.append("PhysicalScene observation_sha256 does not match SceneObservation")
    if alignment.get("scale_source") != "measured":
        blockers.append("PhysicalScene scale_source must be measured")
    if alignment.get("alignment_source") not in {"measured", "assumed"}:
        blockers.append("PhysicalScene alignment_source must be measured or assumed")
    if alignment.get("up_mode") not in {"level_camera", "supplied_vector"}:
        blockers.append(
            "PhysicalScene must declare up_mode level_camera or supplied_vector. "
            "First-camera +Y is not measured gravity."
        )
    if alignment.get("up_source") not in {"assumed", "measured"}:
        blockers.append("PhysicalScene must declare up_source assumed or measured")
    if alignment.get("up_mode") == "level_camera" and alignment.get("up_source") == "measured":
        blockers.append("level_camera cannot be claimed as measured gravity")
    if (
        alignment.get("alignment_source") == "measured"
        and alignment.get("up_source") != "measured"
    ):
        blockers.append("measured alignment_source requires a measured physical-up vector")
    if alignment.get("T_scene_observation_m") is None:
        blockers.append("PhysicalScene has no T_scene_observation_m")
    if isinstance(meters, (int, float)) and not isinstance(meters, bool):
        scene_meters = alignment.get("meters_per_observation_unit")
        if not isinstance(scene_meters, (int, float)) or not math.isclose(
            float(scene_meters),
            float(meters),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("PhysicalScene and SceneObservation metric scales do not match")

    entities_raw = observation.get("extensions", {}).get(ENTITIES_EXTENSION)
    if entities_raw is None:
        blockers.append("SceneObservation has no entities.v1 object track")
        return blockers
    try:
        entities = validate_entities_v1(
            entities_raw,
            sample_count=len(observation["timeline"]["samples"]),
        )
    except Exception as error:
        blockers.append(f"invalid entities.v1 evidence: {error}")
        return blockers
    try:
        entity = find_entity(entities, entity_id)
    except KeyError:
        blockers.append(f"entities.v1 entity {entity_id!r} was not found")
        return blockers
    visible = sorted(
        (sample for sample in entity["samples"] if sample.get("visible", True)),
        key=lambda sample: int(sample["sample_index"]),
    )
    if len(visible) < 12:
        blockers.append("P5R requires at least 12 visible entity-root samples")
    timeline_by_index = {
        int(sample["sample_index"]): sample for sample in observation["timeline"]["samples"]
    }
    times = [
        float(timeline_by_index[int(sample["sample_index"])]["timestamp_s"])
        for sample in visible
        if int(sample["sample_index"]) in timeline_by_index
    ]
    if len(times) >= 2 and times[-1] - times[0] < 0.5:
        blockers.append("P5R requires at least 0.5 seconds of visible motion")
    if any(times[index] >= times[index + 1] for index in range(len(times) - 1)):
        blockers.append("P5R entity timestamps must be strictly increasing")
    scene_start = float(template_scene["execution"]["start_time_s"])
    scene_end = scene_start + float(template_scene["execution"]["duration_s"])
    if times and (
        times[0] < scene_start - 1e-12
        or times[-1] > scene_end + 1e-12
    ):
        blockers.append(
            "P5R entity timestamps must lie inside the PhysicalScene timeline"
        )
    attachment = template_scene["model"]["constraints"][0]["body_attachment_m"]
    if any(abs(float(value)) > 1e-12 for value in attachment):
        blockers.append("entity-root fitting requires body_attachment_m at the body origin")
    return blockers


def motion_observation_from_scene_observation(
    observation: Mapping[str, Any],
    template_scene: Mapping[str, Any],
    *,
    person_id: str | None = None,
) -> dict[str, Any]:
    """Convert an eligible P2 pelvis track into physical-scene meters."""

    blockers = scene_observation_blockers(
        observation,
        template_scene,
        person_id=person_id,
    )
    if blockers:
        raise FitInputBlocked(blockers)

    humans = validate_humans_v1(
        observation["extensions"][HUMANS_EXTENSION],
        sample_count=len(observation["timeline"]["samples"]),
    )
    person = next(
        value
        for value in humans["people"]
        if person_id is None or value["id"] == person_id
    )
    timeline_by_index = {
        int(sample["sample_index"]): sample for sample in observation["timeline"]["samples"]
    }
    scale = float(observation["coordinates"]["scale"]["meters_per_world_unit"])
    transform = template_scene["observation_alignment"]["T_scene_observation_m"]
    samples = []
    for human_sample in sorted(
        person["samples"],
        key=lambda sample: int(sample["sample_index"]),
    ):
        if not human_sample.get("visible", True):
            continue
        sample_index = int(human_sample["sample_index"])
        timestamp = float(timeline_by_index[sample_index]["timestamp_s"])
        scaled = [scale * float(value) for value in human_sample["root"]]
        position = transform_point(transform, scaled)
        samples.append(
            {
                "sample_index": len(samples),
                "timestamp_s": timestamp,
                "position_m": list(position),
                "weight": 1.0,
            }
        )
    ranges = [
        max(sample["position_m"][axis] for sample in samples)
        - min(sample["position_m"][axis] for sample in samples)
        for axis in range(3)
    ]
    if any(value < 0.02 for value in ranges):
        raise FitInputBlocked(
            [
                "P5 requires at least 0.02 m observed variation on X, Y, and Z; "
                f"got {ranges}"
            ]
        )

    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    document = {
        "schema": "phystwin.physical_motion_observation",
        "version": 1,
        "observation_id": f"{observation['observation_id']}-{person['id']}-root-motion",
        "source": {
            "kind": "scene_observation_human_root",
            "id": observation["observation_id"],
            "sha256": observation_hash,
        },
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Y",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
        },
        "units": {"length": "meter", "time": "second"},
        "track": {
            "body_id": template_scene["model"]["bodies"][0]["id"],
            "point": "body_origin",
            "samples": samples,
        },
        "provenance": {
            "synthetic": False,
            "source_extension": HUMANS_EXTENSION,
            "person_id": person["id"],
            "joint": "pelvis",
            "scale_status": "metric_measured",
            "alignment_source": "measured",
        },
        "warnings": [
            "The P5 body is a passive rigid proxy for the reconstructed pelvis. "
            "Articulated control is outside P5."
        ],
    }
    validate_physical_motion_observation(document)
    return document


def motion_observation_from_entities(
    observation: Mapping[str, Any],
    template_scene: Mapping[str, Any],
    *,
    entity_id: str | None = None,
) -> dict[str, Any]:
    """Convert an eligible entity root track into physical-scene meters."""

    blockers = entity_observation_blockers(
        observation,
        template_scene,
        entity_id=entity_id,
    )
    if blockers:
        raise FitInputBlocked(blockers)

    entities = validate_entities_v1(
        observation["extensions"][ENTITIES_EXTENSION],
        sample_count=len(observation["timeline"]["samples"]),
    )
    entity = find_entity(entities, entity_id)
    timeline_by_index = {
        int(sample["sample_index"]): sample for sample in observation["timeline"]["samples"]
    }
    scale = float(observation["coordinates"]["scale"]["meters_per_world_unit"])
    transform = template_scene["observation_alignment"]["T_scene_observation_m"]
    samples = []
    for entity_sample in sorted(entity["samples"], key=lambda item: int(item["sample_index"])):
        if not entity_sample.get("visible", True):
            continue
        sample_index = int(entity_sample["sample_index"])
        timestamp = float(timeline_by_index[sample_index]["timestamp_s"])
        scaled = [scale * float(value) for value in entity_sample["root"]]
        position = transform_point(transform, scaled)
        samples.append(
            {
                "sample_index": len(samples),
                "timestamp_s": timestamp,
                "position_m": list(position),
                "weight": 1.0,
            }
        )
    require_nontrivial_spatial_motion(samples, label="P5R")

    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    document = {
        "schema": "phystwin.physical_motion_observation",
        "version": 1,
        "observation_id": f"{observation['observation_id']}-{entity['id']}-root-motion",
        "source": {
            "kind": "scene_observation_entity_root",
            "id": observation["observation_id"],
            "sha256": observation_hash,
        },
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Y",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
        },
        "units": {"length": "meter", "time": "second"},
        "track": {
            "body_id": template_scene["model"]["bodies"][0]["id"],
            "point": "body_origin",
            "samples": samples,
        },
        "provenance": {
            "synthetic": False,
            "source_extension": ENTITIES_EXTENSION,
            "entity_id": entity["id"],
            "scale_status": "metric_measured",
            "alignment_source": template_scene["observation_alignment"].get(
                "alignment_source"
            ),
            "calibration": observation.get("provenance", {}).get("metric_calibration"),
            "evidence_kind": observation.get("provenance", {}).get("evidence_kind"),
            "dataset": observation.get("provenance", {}).get("dataset"),
        },
        "warnings": [
            "The P5R body is a passive rigid proxy for the lifted object centroid."
        ],
    }
    validate_physical_motion_observation(document)
    return document
