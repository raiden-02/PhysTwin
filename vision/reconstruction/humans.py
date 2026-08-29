"""Narrow human-reconstruction boundary and humans.v1 validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .adapter import EstimatorDescriptor
from .contracts import (
    CONTRACT_VERSION,
    ContractError,
    _integer,
    _mapping,
    _require_fields,
    _sequence,
    _vector3,
    canonical_json_bytes,
)
from .transforms import transform_point

HUMANS_EXTENSION = "phystwin.humans.v1"
JOINT_LAYOUT = "smpl24"
COORDINATE_FRAME = "observation_world"
SMPL24_COUNT = 24

SMPL24_NAMES = (
    "pelvis",
    "l_hip",
    "r_hip",
    "spine1",
    "l_knee",
    "r_knee",
    "spine2",
    "l_ankle",
    "r_ankle",
    "spine3",
    "l_foot",
    "r_foot",
    "neck",
    "l_collar",
    "r_collar",
    "head",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
)

# Parent-child pairs for a stick figure. Indices match SMPL24_NAMES.
SMPL24_BONES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
)


class HumanReconstructionError(RuntimeError):
    """The human adapter cannot produce a valid observation payload."""


@dataclass(frozen=True)
class HumanReconstructionRequest:
    """Estimator-independent human reconstruction request."""

    options: Mapping[str, Any]
    parent_observation: Mapping[str, Any] | None = None
    video_sha256: str | None = None


@dataclass(frozen=True)
class HumanReconstructionOutput:
    """Canonical observation plus runtime facts."""

    observation: Mapping[str, Any]
    runtime: Mapping[str, Any]


class HumanReconstructionAdapter(Protocol):
    """Convert estimator-native body output into SceneObservation humans.v1."""

    @property
    def descriptor(self) -> EstimatorDescriptor:
        """Return the exact adapter, model, revision, and weights identity."""

    def reconstruct_humans(
        self,
        request: HumanReconstructionRequest,
        work_dir: Path,
    ) -> HumanReconstructionOutput:
        """Write optional artifacts below work_dir and return a SceneObservation."""


def human_cache_key(
    descriptor: EstimatorDescriptor,
    request: HumanReconstructionRequest,
) -> str:
    """Return a path-independent SHA-256 identity for a humans overlay."""

    identity = {
        "schema": HUMANS_EXTENSION,
        "version": CONTRACT_VERSION,
        "adapter": {
            "name": descriptor.adapter,
            "version": descriptor.adapter_version,
        },
        "estimator": {
            "model": descriptor.model,
            "revision": descriptor.model_revision,
            "weights_sha256": descriptor.weights_sha256,
        },
        "parent_observation_id": (
            None
            if request.parent_observation is None
            else request.parent_observation.get("observation_id")
        ),
        "parent_observation_sha256": (
            None
            if request.parent_observation is None
            else hashlib.sha256(canonical_json_bytes(request.parent_observation)).hexdigest()
        ),
        "video_sha256": request.video_sha256,
        "options": request.options,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def joints_as_vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
    """Parse one SMPL24 joint array."""

    rows = _sequence(value, path)
    if len(rows) != SMPL24_COUNT:
        raise ContractError(f"{path}: must contain {SMPL24_COUNT} joints")
    return [_vector3(row, f"{path}[{index}]") for index, row in enumerate(rows)]


def lift_joints_to_world(
    joints_camera: Sequence[Sequence[float]],
    T_world_camera: Sequence[float],
) -> list[list[float]]:
    """Map OpenCV camera-space joints through one observation-world pose."""

    world = [list(transform_point(T_world_camera, joint)) for joint in joints_camera]
    if len(world) != SMPL24_COUNT:
        raise ContractError("lifted joints must be SMPL24")
    return world


def validate_humans_v1(
    payload: Any,
    *,
    sample_count: int | None = None,
) -> Mapping[str, Any]:
    """Validate namespaced body evidence. Unknown extra keys stay allowed."""

    root = _mapping(payload, HUMANS_EXTENSION)
    _require_fields(root, HUMANS_EXTENSION, {"joint_layout", "coordinate_frame", "people"})
    if root.get("joint_layout") != JOINT_LAYOUT:
        raise ContractError(f"{HUMANS_EXTENSION}.joint_layout: must be {JOINT_LAYOUT}")
    if root.get("coordinate_frame") != COORDINATE_FRAME:
        raise ContractError(f"{HUMANS_EXTENSION}.coordinate_frame: must be {COORDINATE_FRAME}")

    people = _sequence(root.get("people"), f"{HUMANS_EXTENSION}.people")
    if not people:
        raise ContractError(f"{HUMANS_EXTENSION}.people: must not be empty")
    seen_ids: set[str] = set()
    for person_index, raw_person in enumerate(people):
        person = _mapping(raw_person, f"{HUMANS_EXTENSION}.people[{person_index}]")
        _require_fields(person, f"people[{person_index}]", {"id", "samples"})
        person_id = person.get("id")
        if not isinstance(person_id, str) or not person_id:
            raise ContractError(f"people[{person_index}].id: must be a string")
        if person_id in seen_ids:
            raise ContractError(f"{HUMANS_EXTENSION}: duplicate person id {person_id}")
        seen_ids.add(person_id)
        if "track_id" in person:
            _integer(person.get("track_id"), f"people[{person_index}].track_id")
        samples = _sequence(person.get("samples"), f"people[{person_index}].samples")
        if not samples:
            raise ContractError(f"people[{person_index}].samples: must not be empty")
        seen_samples: set[int] = set()
        for raw_sample in samples:
            sample = _mapping(raw_sample, f"{person_id}.sample")
            _require_fields(sample, f"{person_id}.sample", {"sample_index", "root", "joints"})
            sample_index = _integer(sample.get("sample_index"), f"{person_id}.sample_index")
            if sample_index in seen_samples:
                raise ContractError(f"{person_id}: duplicate sample_index {sample_index}")
            if sample_count is not None and not 0 <= sample_index < sample_count:
                raise ContractError(f"{person_id}: sample_index {sample_index} is out of range")
            seen_samples.add(sample_index)
            joints = joints_as_vectors(sample.get("joints"), f"{person_id}[{sample_index}].joints")
            root_joint = _vector3(sample.get("root"), f"{person_id}[{sample_index}].root")
            if any(not _close(root_joint[axis], joints[0][axis]) for axis in range(3)):
                raise ContractError(f"{person_id}[{sample_index}].root: must equal joints[0] (pelvis)")
            if "visible" in sample and not isinstance(sample.get("visible"), bool):
                raise ContractError(f"{person_id}[{sample_index}].visible: must be a boolean")
    canonical_json_bytes(root)
    return root


def attach_humans(
    observation: Mapping[str, Any],
    humans: Mapping[str, Any],
    *,
    provenance_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a SceneObservation and write humans.v1 without changing core fields."""

    sample_count = len(observation["timeline"]["samples"])
    validate_humans_v1(humans, sample_count=sample_count)
    document = dict(observation)
    extensions = dict(document.get("extensions") or {})
    extensions[HUMANS_EXTENSION] = dict(humans)
    document["extensions"] = extensions
    if provenance_extra:
        provenance = dict(document.get("provenance") or {})
        extras = dict(provenance.get("humans") or {})
        extras.update(dict(provenance_extra))
        provenance["humans"] = extras
        document["provenance"] = provenance
    return document


def person_payload(
    person_id: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    track_id: int = 0,
) -> dict[str, Any]:
    return {
        "id": person_id,
        "track_id": track_id,
        "samples": [dict(sample) for sample in samples],
    }


def humans_payload(people: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "joint_layout": JOINT_LAYOUT,
        "coordinate_frame": COORDINATE_FRAME,
        "people": [dict(person) for person in people],
    }


def sample_payload(
    sample_index: int,
    joints_world: Sequence[Sequence[float]],
    *,
    visible: bool = True,
) -> dict[str, Any]:
    joints = [list(map(float, joint)) for joint in joints_world]
    if len(joints) != SMPL24_COUNT:
        raise ContractError("sample joints must be SMPL24")
    return {
        "sample_index": sample_index,
        "root": list(joints[0]),
        "joints": joints,
        "visible": visible,
    }


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6
