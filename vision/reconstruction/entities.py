"""Generic object/entity 3D tracks on SceneObservation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    CONTRACT_VERSION,
    ContractError,
    _finite,
    _integer,
    _mapping,
    _require_fields,
    _sequence,
    _vector3,
    canonical_json_bytes,
)

ENTITIES_EXTENSION = "phystwin.entities.v1"
COORDINATE_FRAME = "observation_world"


def validate_entities_v1(
    payload: Any,
    *,
    sample_count: int | None = None,
) -> Mapping[str, Any]:
    """Validate namespaced object tracks. Extra keys stay allowed."""

    root = _mapping(payload, ENTITIES_EXTENSION)
    _require_fields(root, ENTITIES_EXTENSION, {"coordinate_frame", "entities"})
    if root.get("coordinate_frame") != COORDINATE_FRAME:
        raise ContractError(f"{ENTITIES_EXTENSION}.coordinate_frame: must be {COORDINATE_FRAME}")
    entities = _sequence(root.get("entities"), f"{ENTITIES_EXTENSION}.entities")
    if not entities:
        raise ContractError(f"{ENTITIES_EXTENSION}.entities: must not be empty")
    seen_ids: set[str] = set()
    for index, raw in enumerate(entities):
        entity = _mapping(raw, f"{ENTITIES_EXTENSION}.entities[{index}]")
        _require_fields(entity, f"entities[{index}]", {"id", "kind", "samples"})
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ContractError(f"entities[{index}].id: must be a string")
        if entity_id in seen_ids:
            raise ContractError(f"{ENTITIES_EXTENSION}: duplicate entity id {entity_id}")
        seen_ids.add(entity_id)
        if entity.get("kind") not in {"object", "anchor", "attachment"}:
            raise ContractError(
                f"entities[{entity_id}].kind: must be object, anchor, or attachment"
            )
        samples = _sequence(entity.get("samples"), f"entities[{entity_id}].samples")
        if not samples:
            raise ContractError(f"entities[{entity_id}].samples: must not be empty")
        seen_samples: set[int] = set()
        for raw_sample in samples:
            sample = _mapping(raw_sample, f"{entity_id}.sample")
            _require_fields(
                sample,
                f"{entity_id}.sample",
                {"sample_index", "root", "visible"},
            )
            sample_index = _integer(sample.get("sample_index"), f"{entity_id}.sample_index")
            if sample_index in seen_samples:
                raise ContractError(f"{entity_id}: duplicate sample_index {sample_index}")
            if sample_count is not None and not 0 <= sample_index < sample_count:
                raise ContractError(f"{entity_id}: sample_index {sample_index} is out of range")
            seen_samples.add(sample_index)
            _vector3(sample.get("root"), f"{entity_id}[{sample_index}].root")
            if not isinstance(sample.get("visible"), bool):
                raise ContractError(f"{entity_id}[{sample_index}].visible: must be a boolean")
            if "pixel" in sample:
                pixel = _sequence(sample.get("pixel"), f"{entity_id}[{sample_index}].pixel")
                if len(pixel) != 2:
                    raise ContractError(f"{entity_id}[{sample_index}].pixel: must be [u, v]")
                _finite(pixel[0], f"{entity_id}[{sample_index}].pixel[0]")
                _finite(pixel[1], f"{entity_id}[{sample_index}].pixel[1]")
            if "mask_pixels" in sample and _integer(
                sample.get("mask_pixels"),
                f"{entity_id}[{sample_index}].mask_pixels",
            ) < 0:
                raise ContractError(f"{entity_id}[{sample_index}].mask_pixels: must be >= 0")
    canonical_json_bytes(root)
    return root


def attach_entities(
    observation: Mapping[str, Any],
    entities: Mapping[str, Any],
    *,
    provenance_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a SceneObservation and write entities.v1 without changing core fields."""

    sample_count = len(observation["timeline"]["samples"])
    validate_entities_v1(entities, sample_count=sample_count)
    document = dict(observation)
    extensions = dict(document.get("extensions") or {})
    extensions[ENTITIES_EXTENSION] = dict(entities)
    document["extensions"] = extensions
    if provenance_extra:
        provenance = dict(document.get("provenance") or {})
        extras = dict(provenance.get("entities") or {})
        extras.update(dict(provenance_extra))
        provenance["entities"] = extras
        document["provenance"] = provenance
    return document


def entities_payload(entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "coordinate_frame": COORDINATE_FRAME,
        "entities": [dict(entity) for entity in entities],
    }


def entity_payload(
    entity_id: str,
    kind: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "kind": kind,
        "samples": [dict(sample) for sample in samples],
    }


def find_entity(entities: Mapping[str, Any], entity_id: str | None) -> Mapping[str, Any]:
    people = entities["entities"]
    if entity_id is None:
        return people[0]
    for entity in people:
        if entity["id"] == entity_id:
            return entity
    raise KeyError(entity_id)
