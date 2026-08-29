"""Assemble entity tracks and a metric PhysicalMotionObservation for P5R."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import canonical_json_bytes, validate_scene_observation
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
            "lift": "sam2_mask_median_da3_depth",
            "accepted_lifts": accepted,
            "rejected_lifts": rejected,
        },
    )
    return {
        "observation": attached,
        "accepted_lifts": accepted,
        "rejected_lifts": rejected,
    }


def assumed_first_camera_alignment(
    observation: Mapping[str, Any],
    *,
    entity_id: str,
    anchor_id: str | None,
    world_anchor_m: Sequence[float],
) -> list[float]:
    """Translate the first scaled pair so the anchor lands on the physical anchor.

    Rotation stays identity: first-camera +Y is treated as physical up. That is
    an assumption, not a measured gravity direction.
    """

    scale = float(observation["coordinates"]["scale"]["meters_per_world_unit"])
    entities = validate_entities_v1(observation["extensions"][ENTITIES_EXTENSION])
    target = _first_visible(find_entity(entities, entity_id))
    if anchor_id is None:
        source = [scale * float(value) for value in target]
        destination = list(world_anchor_m)
    else:
        source = [scale * float(value) for value in _first_visible(find_entity(entities, anchor_id))]
        destination = list(world_anchor_m)
    return [
        1.0, 0.0, 0.0, destination[0] - source[0],
        0.0, 1.0, 0.0, destination[1] - source[1],
        0.0, 0.0, 1.0, destination[2] - source[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def stamp_observation_alignment(
    template: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    entity_id: str,
    anchor_id: str | None,
) -> dict[str, Any]:
    """Write measured scale and assumed first-camera gravity alignment."""

    scene = copy.deepcopy(dict(template))
    scale = observation["coordinates"]["scale"]
    meters = float(scale["meters_per_world_unit"])
    transform = assumed_first_camera_alignment(
        observation,
        entity_id=entity_id,
        anchor_id=anchor_id,
        world_anchor_m=scene["model"]["constraints"][0]["world_anchor_m"],
    )
    scene["observation_alignment"] = {
        "observation_uri": f"{observation['observation_id']}.json",
        "observation_sha256": hashlib.sha256(canonical_json_bytes(observation)).hexdigest(),
        "meters_per_observation_unit": meters,
        "scale_source": "measured",
        "alignment_source": "assumed",
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
