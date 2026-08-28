"""Narrow interface for third-party 3D reconstruction adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import CONTRACT_VERSION, SCENE_OBSERVATION_SCHEMA, canonical_json_bytes


@dataclass(frozen=True)
class EstimatorDescriptor:
    """Immutable estimator identity used for provenance and cache keys."""

    adapter: str
    adapter_version: str
    model: str
    model_revision: str
    weights_sha256: str | None


@dataclass(frozen=True)
class VideoInput:
    """One local source plus its content identity."""

    id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ReconstructionRequest:
    """Estimator-independent reconstruction request."""

    inputs: tuple[VideoInput, ...]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class ReconstructionOutput:
    """Canonical JSON data plus runtime facts not used as observations."""

    observation: Mapping[str, Any]
    runtime: Mapping[str, Any]


class ReconstructionAdapter(Protocol):
    """Convert source videos into one canonical SceneObservation."""

    @property
    def descriptor(self) -> EstimatorDescriptor:
        """Return the exact adapter, model, revision, and weights identity."""

    def reconstruct(
        self,
        request: ReconstructionRequest,
        work_dir: Path,
    ) -> ReconstructionOutput:
        """Write heavy artifacts below work_dir and return canonical metadata."""


def reconstruction_cache_key(
    descriptor: EstimatorDescriptor,
    request: ReconstructionRequest,
) -> str:
    """Return a path-independent SHA-256 identity for reconstruction output."""

    if not request.inputs:
        raise ValueError("reconstruction requires at least one video input")
    if len({source.id for source in request.inputs}) != len(request.inputs):
        raise ValueError("video input ids must be unique")

    identity = {
        "schema": SCENE_OBSERVATION_SCHEMA,
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
        "inputs": [
            {
                "id": source.id,
                "sha256": source.sha256,
            }
            for source in request.inputs
        ],
        "options": request.options,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
