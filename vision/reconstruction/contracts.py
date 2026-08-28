"""Core validation and coordinates for PhysTwin's 3D JSON boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCENE_OBSERVATION_SCHEMA = "phystwin.scene_observation"
PHYSICAL_SCENE_SCHEMA = "phystwin.physical_scene"
CONTRACT_VERSION = 1

# OpenCV camera (+X right, +Y down, +Z forward) to the first-camera
# graphics world (+X right, +Y up, +Z backward).
FIRST_CAMERA_WORLD_FROM_OPENCV = (
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


class ContractError(ValueError):
    """A document violates a canonical contract invariant."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{path}: must be an array")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path}: must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{path}: must be finite")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path}: must be an integer")
    return value


def _require_fields(
    value: Mapping[str, Any], path: str, fields: set[str], *, exact: bool = False
) -> None:
    missing = fields - set(value)
    if missing:
        raise ContractError(f"{path}: missing {', '.join(sorted(missing))}")
    if exact and set(value) != fields:
        unknown = set(value) - fields
        raise ContractError(f"{path}: unknown {', '.join(sorted(unknown))}")


def _items_by_id(value: Any, path: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_sequence(value, path)):
        item = _mapping(raw, f"{path}[{index}]")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ContractError(f"{path}[{index}].id: must be a string")
        if item_id in result:
            raise ContractError(f"{path}: duplicate id {item_id}")
        result[item_id] = item
    return result


def _sha256(value: Any, path: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ContractError(f"{path}: must be a lowercase SHA-256")


def _vector3(value: Any, path: str) -> tuple[float, float, float]:
    values = _sequence(value, path)
    if len(values) != 3:
        raise ContractError(f"{path}: must contain 3 numbers")
    return (
        _finite(values[0], f"{path}[0]"),
        _finite(values[1], f"{path}[1]"),
        _finite(values[2], f"{path}[2]"),
    )


def validate_rigid_transform(value: Any, path: str = "transform") -> tuple[float, ...]:
    """Validate a row-major rigid 4x4 transform."""

    values = _sequence(value, path)
    if len(values) != 16:
        raise ContractError(f"{path}: must contain 16 row-major values")
    matrix = tuple(_finite(item, f"{path}[{i}]") for i, item in enumerate(values))
    last_row = zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0))
    if any(not math.isclose(matrix[i], expected, abs_tol=1e-8) for i, expected in last_row):
        raise ContractError(f"{path}: last row must be [0, 0, 0, 1]")

    rows = (matrix[0:3], matrix[4:7], matrix[8:11])
    for row in rows:
        if not math.isclose(sum(x * x for x in row), 1.0, abs_tol=1e-6):
            raise ContractError(f"{path}: rotation row is not unit length")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if not math.isclose(sum(rows[a][i] * rows[b][i] for i in range(3)), 0.0, abs_tol=1e-6):
            raise ContractError(f"{path}: rotation rows are not orthogonal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-6):
        raise ContractError(f"{path}: rotation determinant must be +1")
    return matrix


def _multiply_4x4(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 4 + k] * b[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    )


def opencv_camera_to_three_world(T_world_camera: Any) -> tuple[float, ...]:
    """Return the Three.js camera-object transform for an OpenCV pose."""

    pose = validate_rigid_transform(T_world_camera, "T_world_camera")
    return _multiply_4x4(pose, FIRST_CAMERA_WORLD_FROM_OPENCV)


def project_world_point(
    point_world: Any,
    T_world_camera: Any,
    intrinsics: Mapping[str, Any],
) -> tuple[float, float]:
    """Project one canonical world point through a pinhole camera."""

    point = _vector3(point_world, "point_world")
    pose = validate_rigid_transform(T_world_camera, "T_world_camera")
    rotation = (pose[0:3], pose[4:7], pose[8:11])
    translation = (pose[3], pose[7], pose[11])
    delta = tuple(point[i] - translation[i] for i in range(3))
    camera = tuple(
        sum(rotation[column][row] * delta[column] for column in range(3))
        for row in range(3)
    )
    if camera[2] <= 0.0:
        raise ContractError("point_world: projects behind the camera")
    fx = _finite(intrinsics.get("fx_px"), "intrinsics.fx_px")
    fy = _finite(intrinsics.get("fy_px"), "intrinsics.fy_px")
    cx = _finite(intrinsics.get("cx_px"), "intrinsics.cx_px")
    cy = _finite(intrinsics.get("cy_px"), "intrinsics.cy_px")
    skew = _finite(intrinsics.get("skew_px", 0.0), "intrinsics.skew_px")
    return (
        fx * camera[0] / camera[2] + skew * camera[1] / camera[2] + cx,
        fy * camera[1] / camera[2] + cy,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize deterministic cache identity and reject NaN or infinity."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"cannot serialize canonical JSON: {error}") from error
    return text.encode("utf-8")


def validate_scene_observation(document: Any) -> Mapping[str, Any]:
    """Validate the minimal estimator-independent observation envelope."""

    root = _mapping(document, "SceneObservation")
    root_fields = {
        "schema", "version", "observation_id", "timeline", "coordinates",
        "sources", "artifacts", "cameras", "static_scene", "provenance", "extensions",
    }
    _require_fields(root, "SceneObservation", root_fields, exact=True)
    if root["schema"] != SCENE_OBSERVATION_SCHEMA or root["version"] != CONTRACT_VERSION:
        raise ContractError("SceneObservation: unsupported schema or version")

    coordinates = _mapping(root["coordinates"], "SceneObservation.coordinates")
    expected_coordinates = {
        "world_basis": "first_camera_graphics",
        "handedness": "right",
        "camera_convention": "opencv",
        "transform_notation": "T_parent_child",
        "vector_convention": "column",
        "matrix_storage": "row_major",
    }
    for field, expected in expected_coordinates.items():
        if coordinates.get(field) != expected:
            raise ContractError(f"SceneObservation.coordinates.{field}: must be {expected}")
    scale = _mapping(coordinates.get("scale"), "SceneObservation.coordinates.scale")
    status = scale.get("status")
    meters = scale.get("meters_per_world_unit")
    if status not in {"relative", "metric_measured", "metric_assumed"}:
        raise ContractError("SceneObservation.coordinates.scale.status: unsupported")
    if status == "relative" and meters is not None:
        raise ContractError("relative scale requires null meters_per_world_unit")
    if status != "relative" and _finite(meters, "meters_per_world_unit") <= 0.0:
        raise ContractError("meters_per_world_unit: must be > 0")

    sources = _items_by_id(root["sources"], "SceneObservation.sources")
    if not sources:
        raise ContractError("SceneObservation.sources: must not be empty")
    for source_id, source in sources.items():
        if source.get("kind") != "video":
            raise ContractError(f"SceneObservation.sources.{source_id}: must be video")
        _sha256(source.get("sha256"), f"SceneObservation.sources.{source_id}.sha256")

    timeline = _mapping(root["timeline"], "SceneObservation.timeline")
    if timeline.get("time_unit") != "second":
        raise ContractError("SceneObservation.timeline.time_unit: must be second")
    if timeline.get("origin") != "observation_start":
        raise ContractError("SceneObservation.timeline.origin: must be observation_start")
    samples = _sequence(timeline.get("samples"), "SceneObservation.timeline.samples")
    if not samples:
        raise ContractError("SceneObservation.timeline.samples: must not be empty")
    previous_time = -math.inf
    for index, raw in enumerate(samples):
        sample = _mapping(raw, f"sample[{index}]")
        if _integer(sample.get("sample_index"), f"sample[{index}].sample_index") != index:
            raise ContractError("sample_index must be contiguous from zero")
        timestamp = _finite(sample.get("timestamp_s"), f"sample[{index}].timestamp_s")
        if timestamp < 0.0 or timestamp <= previous_time:
            raise ContractError("timestamps must be non-negative and strictly increasing")
        previous_time = timestamp
        frames = _mapping(sample.get("source_frames"), f"sample[{index}].source_frames")
        if not frames or not set(frames).issubset(sources):
            raise ContractError("source_frames contain an unknown or missing source")
        for source_id, frame in frames.items():
            if _integer(frame, f"source_frames.{source_id}") < 0:
                raise ContractError("source frame must be >= 0")

    artifacts = _items_by_id(root["artifacts"], "SceneObservation.artifacts")
    for artifact_id, artifact in artifacts.items():
        _sha256(artifact.get("sha256"), f"SceneObservation.artifacts.{artifact_id}.sha256")

    cameras = _items_by_id(root["cameras"], "SceneObservation.cameras")
    if not cameras:
        raise ContractError("SceneObservation.cameras: must not be empty")
    first_pose: tuple[float, ...] | None = None
    for camera_id, camera in cameras.items():
        if camera.get("source_id") not in sources or camera.get("projection") != "pinhole":
            raise ContractError(f"SceneObservation.cameras.{camera_id}: invalid source or projection")
        image_size = _sequence(camera.get("image_size_px"), f"{camera_id}.image_size_px")
        if len(image_size) != 2 or any(_integer(x, "image_size") <= 0 for x in image_size):
            raise ContractError(f"SceneObservation.cameras.{camera_id}: invalid image size")
        intrinsics = _mapping(camera.get("intrinsics"), f"{camera_id}.intrinsics")
        if _finite(intrinsics.get("fx_px"), f"{camera_id}.fx_px") <= 0.0:
            raise ContractError(f"{camera_id}.fx_px: must be > 0")
        if _finite(intrinsics.get("fy_px"), f"{camera_id}.fy_px") <= 0.0:
            raise ContractError(f"{camera_id}.fy_px: must be > 0")
        for field in ("cx_px", "cy_px", "skew_px"):
            _finite(intrinsics.get(field), f"{camera_id}.{field}")
        seen_samples: set[int] = set()
        for raw_pose in _sequence(camera.get("poses"), f"{camera_id}.poses"):
            pose = _mapping(raw_pose, f"{camera_id}.pose")
            sample_index = _integer(pose.get("sample_index"), f"{camera_id}.sample_index")
            if sample_index in seen_samples or not 0 <= sample_index < len(samples):
                raise ContractError(f"{camera_id}: invalid pose sample")
            seen_samples.add(sample_index)
            transform = validate_rigid_transform(pose.get("T_world_camera"), f"{camera_id}.pose")
            first_pose = first_pose or transform
    if first_pose is None or any(
        not math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(first_pose or (), FIRST_CAMERA_WORLD_FROM_OPENCV)
    ):
        raise ContractError("first camera pose must define the world gauge")

    scene = _mapping(root["static_scene"], "SceneObservation.static_scene")
    for index, raw in enumerate(_sequence(scene.get("geometry"), "static_scene.geometry")):
        geometry = _mapping(raw, f"static_scene.geometry[{index}]")
        if geometry.get("kind") not in {"mesh", "point_cloud"}:
            raise ContractError("static scene geometry kind is invalid")
        if geometry.get("artifact_id") not in artifacts:
            raise ContractError("static scene geometry does not reference an artifact")
    _mapping(root["provenance"], "SceneObservation.provenance")
    _mapping(root["extensions"], "SceneObservation.extensions")
    canonical_json_bytes(root)
    return root


def validate_physical_scene(document: Any) -> Mapping[str, Any]:
    """Validate the physical hypothesis envelope and execution gate."""

    root = _mapping(document, "PhysicalScene")
    root_fields = {
        "schema", "version", "scene_id", "coordinates", "units",
        "observation_alignment", "execution", "world", "model",
        "parameters", "provenance", "extensions",
    }
    _require_fields(root, "PhysicalScene", root_fields, exact=True)
    if root["schema"] != PHYSICAL_SCENE_SCHEMA or root["version"] != CONTRACT_VERSION:
        raise ContractError("PhysicalScene: unsupported schema or version")

    coordinates = _mapping(root["coordinates"], "PhysicalScene.coordinates")
    expected_coordinates = {
        "handedness": "right",
        "up_axis": "+Y",
        "transform_notation": "T_parent_child",
        "vector_convention": "column",
        "matrix_storage": "row_major",
    }
    for field, expected in expected_coordinates.items():
        if coordinates.get(field) != expected:
            raise ContractError(f"PhysicalScene.coordinates.{field}: must be {expected}")
    units = _mapping(root["units"], "PhysicalScene.units")
    for field, expected in {
        "length": "meter", "mass": "kilogram", "time": "second", "angle": "radian"
    }.items():
        if units.get(field) != expected:
            raise ContractError(f"PhysicalScene.units.{field}: must be {expected}")

    alignment = _mapping(root["observation_alignment"], "observation_alignment")
    _sha256(alignment.get("observation_sha256"), "observation_sha256")
    meters = alignment.get("meters_per_observation_unit")
    if meters is not None and _finite(meters, "meters_per_observation_unit") <= 0.0:
        raise ContractError("meters_per_observation_unit: must be > 0")
    scale_source = alignment.get("scale_source")
    if scale_source not in {"unknown", "measured", "assumed", "fitted"}:
        raise ContractError("scale_source is unsupported")
    transform = alignment.get("T_scene_observation_m")
    if transform is not None:
        validate_rigid_transform(transform, "T_scene_observation_m")

    execution = _mapping(root["execution"], "PhysicalScene.execution")
    status = execution.get("status")
    if status not in {"draft", "executable"}:
        raise ContractError("PhysicalScene.execution.status is unsupported")
    blockers = _sequence(execution.get("blockers"), "PhysicalScene.execution.blockers")
    fixed_step = execution.get("fixed_step_s")
    if fixed_step is not None and _finite(fixed_step, "fixed_step_s") <= 0.0:
        raise ContractError("fixed_step_s: must be > 0")

    world = _mapping(root["world"], "PhysicalScene.world")
    _vector3(world.get("gravity_m_s2"), "PhysicalScene.world.gravity_m_s2")
    model = _mapping(root["model"], "PhysicalScene.model")
    component_fields = {
        "bodies", "articulations", "joints", "constraints", "contacts",
        "materials", "forces", "actuators", "residual_forces",
    }
    _require_fields(model, "PhysicalScene.model", component_fields, exact=True)
    for field in component_fields:
        for component_id, component in _items_by_id(model[field], f"model.{field}").items():
            if not isinstance(component.get("type"), str):
                raise ContractError(f"model.{field}.{component_id}.type: must be a string")
    _items_by_id(root["parameters"], "PhysicalScene.parameters")

    if status == "executable":
        if blockers:
            raise ContractError("PhysicalScene.execution.blockers: must be empty")
        if not execution.get("backend") or fixed_step is None:
            raise ContractError("executable scene requires a backend and fixed step")
        if scale_source == "unknown" or meters is None:
            raise ContractError("PhysicalScene: metric scale is required when executable")
        if transform is None:
            raise ContractError("PhysicalScene: alignment is required when executable")
    _mapping(root["provenance"], "PhysicalScene.provenance")
    _mapping(root["extensions"], "PhysicalScene.extensions")
    canonical_json_bytes(root)
    return root


def load_contract(path: Path) -> Mapping[str, Any]:
    """Load UTF-8 JSON, reject non-standard numbers, then validate it."""

    def reject_constant(value: str) -> None:
        raise ContractError(f"non-finite JSON number: {value}")

    try:
        document = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to load {path}: {error}") from error
    schema = _mapping(document, str(path)).get("schema")
    if schema == SCENE_OBSERVATION_SCHEMA:
        return validate_scene_observation(document)
    if schema == PHYSICAL_SCENE_SCHEMA:
        return validate_physical_scene(document)
    raise ContractError(f"{path}: unknown schema {schema!r}")
