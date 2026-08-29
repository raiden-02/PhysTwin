"""Assemble entity tracks and a metric PhysicalMotionObservation for P5R."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractError, canonical_json_bytes, validate_scene_observation
from .entities import (
    ENTITIES_EXTENSION,
    attach_entities,
    entities_payload,
    entity_payload,
    find_entity,
    validate_entities_v1,
)
from .lift import (
    camera_pose_for_sample,
    lift_mask_to_world,
    load_da3_depth_artifact,
    sample_intrinsics_for_frame,
)
from .transforms import transform_point


def lift_entities_from_masks(
    observation: Mapping[str, Any],
    *,
    target_masks: Mapping[int, np.ndarray],
    anchor_masks: Mapping[int, np.ndarray] | None,
    depth_artifact: Path,
    target_id: str = "target",
    anchor_id: str = "anchor",
    confidence_floor: float | None = None,
) -> dict[str, Any]:
    """Attach observation-world entity tracks using per-frame depth and K."""

    validate_scene_observation(observation)
    camera = observation["cameras"][0]
    da3 = observation.get("extensions", {}).get("phystwin.da3.v1")
    bundle = load_da3_depth_artifact(depth_artifact)
    depth = np.asarray(bundle["depth"])
    conf = np.asarray(bundle["conf"]) if "conf" in bundle else None
    source_size = tuple(int(value) for value in bundle["source_size"])
    accepted = 0
    rejected = 0
    entities = []
    tracks = [("object", target_id, target_masks)]
    if anchor_masks is not None:
        tracks.append(("anchor", anchor_id, anchor_masks))
    for kind, entity_id, masks in tracks:
        samples = []
        for sample in observation["timeline"]["samples"]:
            index = int(sample["sample_index"])
            mask = masks.get(index)
            if mask is None or index >= depth.shape[0]:
                rejected += 1
                samples.append(_rejected_sample(index))
                continue
            lifted = lift_mask_to_world(
                mask,
                depth[index],
                T_world_camera=camera_pose_for_sample(camera, index),
                intrinsics=sample_intrinsics_for_frame(camera, index, da3),
                confidence=None if conf is None else conf[index],
                confidence_floor=confidence_floor,
                source_size_px=(source_size[0], source_size[1]),
            )
            if lifted is None:
                rejected += 1
                samples.append(_rejected_sample(index))
                continue
            accepted += 1
            samples.append(
                {
                    "sample_index": index,
                    "root": lifted["root"],
                    "visible": True,
                    "pixel": lifted["pixel"],
                    "mask_pixels": lifted["used_pixels"],
                    "depth": lifted["depth"],
                }
            )
        entities.append(entity_payload(entity_id, kind, samples))
    attached = attach_entities(
        observation,
        entities_payload(entities),
        provenance_extra={
            "lift": "robust_3d_center",
            "accepted_lifts": accepted,
            "rejected_lifts": rejected,
        },
    )
    return {
        "observation": attached,
        "accepted_lifts": accepted,
        "rejected_lifts": rejected,
    }


ALLOWED_UP_MODES = {"level_camera", "supplied_vector"}
ALLOWED_UP_SOURCES = {"assumed", "measured"}


def validate_physical_up(physical_up: Mapping[str, Any] | None) -> dict[str, Any]:
    """Require an explicit physical-up declaration. First-camera +Y is not measured gravity."""

    from .contracts import _vector3

    if physical_up is None:
        raise ContractError(
            "P5R requires an explicit physical_up. Do not silently treat first-camera +Y "
            "as measured gravity. Use mode=level_camera with source=assumed, or supply a "
            "measured/assumed up vector."
        )
    mode = physical_up.get("mode")
    source = physical_up.get("source")
    if mode not in ALLOWED_UP_MODES:
        raise ContractError("physical_up.mode must be level_camera or supplied_vector")
    if source not in ALLOWED_UP_SOURCES:
        raise ContractError("physical_up.source must be assumed or measured")
    if mode == "level_camera" and source == "measured":
        raise ContractError(
            "A leveled camera is not a measured gravity vector. "
            "Use source=assumed, or supply a measured up vector."
        )
    if mode == "supplied_vector":
        vector = _vector3(physical_up.get("vector_observation"), "physical_up.vector_observation")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm <= 1e-9:
            raise ContractError("physical_up.vector_observation has near-zero length")
        return {
            "mode": mode,
            "source": source,
            "vector_observation": [float(value) / norm for value in vector],
        }
    return {"mode": mode, "source": source}


def _rotation_mapping_up_to_plus_y(up: Sequence[float]) -> list[float]:
    """Rotate observation so the given unit vector becomes physical +Y."""

    ux, uy, uz = (float(up[0]), float(up[1]), float(up[2]))
    target = (0.0, 1.0, 0.0)
    axis = (
        uy * target[2] - uz * target[1],
        uz * target[0] - ux * target[2],
        ux * target[1] - uy * target[0],
    )
    cosine = ux * target[0] + uy * target[1] + uz * target[2]
    axis_norm = math.sqrt(sum(value * value for value in axis))
    if axis_norm <= 1e-9:
        if cosine > 0.0:
            return [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]
        return [
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
        ]
    ax, ay, az = (value / axis_norm for value in axis)
    sine = axis_norm
    # Normalize (axis, cosine) into a proper Rodrigues angle.
    angle_cos = max(-1.0, min(1.0, cosine))
    one_minus = 1.0 - angle_cos
    return [
        angle_cos + ax * ax * one_minus,
        ax * ay * one_minus - az * sine,
        ax * az * one_minus + ay * sine,
        ay * ax * one_minus + az * sine,
        angle_cos + ay * ay * one_minus,
        ay * az * one_minus - ax * sine,
        az * ax * one_minus - ay * sine,
        az * ay * one_minus + ax * sine,
        angle_cos + az * az * one_minus,
    ]


def _rotate_point(rotation9: Sequence[float], point: Sequence[float]) -> list[float]:
    return [
        rotation9[0] * point[0] + rotation9[1] * point[1] + rotation9[2] * point[2],
        rotation9[3] * point[0] + rotation9[4] * point[1] + rotation9[5] * point[2],
        rotation9[6] * point[0] + rotation9[7] * point[1] + rotation9[8] * point[2],
    ]


def observation_alignment_transform(
    observation: Mapping[str, Any],
    *,
    entity_id: str,
    anchor_id: str | None,
    world_anchor_m: Sequence[float],
    physical_up: Mapping[str, Any],
) -> list[float]:
    """Scale, apply declared physical up, then translate the first anchor."""

    up = validate_physical_up(physical_up)
    scale = float(observation["coordinates"]["scale"]["meters_per_world_unit"])
    entities = validate_entities_v1(observation["extensions"][ENTITIES_EXTENSION])
    if anchor_id is None:
        source_obs = _first_visible(find_entity(entities, entity_id))
    else:
        source_obs = _first_visible(find_entity(entities, anchor_id))
    scaled = [scale * float(value) for value in source_obs]
    if up["mode"] == "supplied_vector":
        rotation = _rotation_mapping_up_to_plus_y(up["vector_observation"])
    else:
        rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    rotated = _rotate_point(rotation, scaled)
    destination = list(world_anchor_m)
    translation = [destination[axis] - rotated[axis] for axis in range(3)]
    return [
        rotation[0], rotation[1], rotation[2], translation[0],
        rotation[3], rotation[4], rotation[5], translation[1],
        rotation[6], rotation[7], rotation[8], translation[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def stamp_observation_alignment(
    template: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    entity_id: str,
    anchor_id: str | None,
    physical_up: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write measured scale and an explicitly declared physical-up alignment."""

    up = validate_physical_up(physical_up)
    scene = copy.deepcopy(dict(template))
    scale = observation["coordinates"]["scale"]
    meters = float(scale["meters_per_world_unit"])
    transform = observation_alignment_transform(
        observation,
        entity_id=entity_id,
        anchor_id=anchor_id,
        world_anchor_m=scene["model"]["constraints"][0]["world_anchor_m"],
        physical_up=up,
    )
    scene["observation_alignment"] = {
        "observation_uri": f"{observation['observation_id']}.json",
        "observation_sha256": hashlib.sha256(canonical_json_bytes(observation)).hexdigest(),
        "meters_per_observation_unit": meters,
        "scale_source": "measured",
        "alignment_source": up["source"],
        "up_mode": up["mode"],
        "up_source": up["source"],
        "T_scene_observation_m": transform,
    }
    entities = validate_entities_v1(observation["extensions"][ENTITIES_EXTENSION])
    constraint = scene["model"]["constraints"][0]
    constraint["body_attachment_m"] = [0.0, 0.0, 0.0]
    target_obs = _first_visible(find_entity(entities, entity_id))
    target_m = transform_point(
        transform,
        [meters * float(value) for value in target_obs],
    )
    anchor = [float(value) for value in constraint["world_anchor_m"]]
    direction = [target_m[axis] - anchor[axis] for axis in range(3)]
    norm = math.sqrt(sum(value * value for value in direction))
    if norm <= 1e-9:
        raise ValueError("first target coincides with the world anchor")
    rest = float(constraint["rest_length_m"])
    placed = [anchor[axis] + rest * direction[axis] / norm for axis in range(3)]
    body_transform = list(scene["model"]["bodies"][0]["T_world_body_initial"])
    body_transform[3] = placed[0]
    body_transform[7] = placed[1]
    body_transform[11] = placed[2]
    scene["model"]["bodies"][0]["T_world_body_initial"] = body_transform
    samples = observation["timeline"]["samples"]
    start = float(scene["execution"]["start_time_s"])
    last = float(samples[-1]["timestamp_s"]) if samples else start
    step = float(scene["execution"]["fixed_step_s"])
    needed = max(float(scene["execution"]["duration_s"]), last - start)
    steps = max(1, math.ceil(needed / step - 1e-12))
    scene["execution"]["duration_s"] = steps * step
    return scene


def _first_visible(entity: Mapping[str, Any]) -> list[float]:
    for sample in sorted(entity["samples"], key=lambda item: int(item["sample_index"])):
        if sample.get("visible", True):
            return [float(value) for value in sample["root"]]
    raise ValueError(f"entity {entity['id']} has no visible sample")


def _rejected_sample(sample_index: int) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "root": [0.0, 0.0, 0.0],
        "visible": False,
        "pixel": [0.0, 0.0],
        "mask_pixels": 0,
    }
