"""Estimator-independent 3D reconstruction boundaries."""

from .adapter import (
    EstimatorDescriptor,
    ReconstructionAdapter,
    ReconstructionOutput,
    ReconstructionRequest,
    VideoInput,
    reconstruction_cache_key,
)
from .contracts import (
    ContractError,
    load_contract,
    require_motion_matches_scene_alignment,
    validate_inverse_fit_artifacts,
    validate_inverse_physics_fit,
    validate_physical_motion_observation,
    validate_physical_scene,
    validate_rollout_source,
    validate_scene_observation,
    validate_simulated_world_state,
)
from .calibration import validate_metric_calibration
from .entities import ENTITIES_EXTENSION, validate_entities_v1
from .footage import REQUESTED_CLIP, inspect_local_footage
from .humans import HUMANS_EXTENSION, validate_humans_v1


def __getattr__(name: str):
    """Load the optional DA3 adapter only when a caller requests it."""

    if name in {"ADAPTER_VERSION", "MODEL_ID", "MODEL_REVISION", "PACKAGE_REVISION", "Da3ReconstructionAdapter"}:
        from . import da3

        return getattr(da3, name)
    raise AttributeError(name)

__all__ = [
    "ADAPTER_VERSION",
    "ContractError",
    "Da3ReconstructionAdapter",
    "EstimatorDescriptor",
    "ENTITIES_EXTENSION",
    "HUMANS_EXTENSION",
    "REQUESTED_CLIP",
    "inspect_local_footage",
    "validate_entities_v1",
    "validate_metric_calibration",
    "MODEL_ID",
    "MODEL_REVISION",
    "PACKAGE_REVISION",
    "ReconstructionAdapter",
    "ReconstructionOutput",
    "ReconstructionRequest",
    "VideoInput",
    "load_contract",
    "reconstruction_cache_key",
    "require_motion_matches_scene_alignment",
    "validate_humans_v1",
    "validate_inverse_fit_artifacts",
    "validate_inverse_physics_fit",
    "validate_physical_motion_observation",
    "validate_physical_scene",
    "validate_rollout_source",
    "validate_scene_observation",
    "validate_simulated_world_state",
]
