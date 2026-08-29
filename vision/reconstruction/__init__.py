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
    validate_physical_scene,
    validate_rollout_source,
    validate_scene_observation,
    validate_simulated_world_state,
)
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
    "HUMANS_EXTENSION",
    "MODEL_ID",
    "MODEL_REVISION",
    "PACKAGE_REVISION",
    "ReconstructionAdapter",
    "ReconstructionOutput",
    "ReconstructionRequest",
    "VideoInput",
    "load_contract",
    "reconstruction_cache_key",
    "validate_humans_v1",
    "validate_physical_scene",
    "validate_rollout_source",
    "validate_scene_observation",
    "validate_simulated_world_state",
]
