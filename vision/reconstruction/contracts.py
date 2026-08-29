"""Core validation and coordinates for PhysTwin's 3D JSON boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCENE_OBSERVATION_SCHEMA = "phystwin.scene_observation"
PHYSICAL_SCENE_SCHEMA = "phystwin.physical_scene"
SIMULATED_WORLD_STATE_SCHEMA = "phystwin.simulated_world_state"
PHYSICAL_MOTION_OBSERVATION_SCHEMA = "phystwin.physical_motion_observation"
INVERSE_PHYSICS_FIT_SCHEMA = "phystwin.inverse_physics_fit"
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
    observation_uri = alignment.get("observation_uri")
    observation_hash = alignment.get("observation_sha256")
    has_observation = observation_uri is not None or observation_hash is not None
    if has_observation:
        if not isinstance(observation_uri, str) or not observation_uri:
            raise ContractError("observation_uri: must identify the source observation")
        _sha256(observation_hash, "observation_sha256")
    elif any(
        alignment.get(field) is not None
        for field in (
            "meters_per_observation_unit",
            "T_scene_observation_m",
        )
    ):
        raise ContractError("standalone PhysicalScene cannot declare an observation alignment")
    meters = alignment.get("meters_per_observation_unit")
    if meters is not None and _finite(meters, "meters_per_observation_unit") <= 0.0:
        raise ContractError("meters_per_observation_unit: must be > 0")
    scale_source = alignment.get("scale_source")
    if scale_source not in {"not_applicable", "unknown", "measured", "assumed", "fitted"}:
        raise ContractError("scale_source is unsupported")
    if has_observation and scale_source == "not_applicable":
        raise ContractError("observation alignment requires a scale source")
    if not has_observation and scale_source != "not_applicable":
        raise ContractError("standalone PhysicalScene scale_source must be not_applicable")
    transform = alignment.get("T_scene_observation_m")
    if transform is not None:
        validate_rigid_transform(transform, "T_scene_observation_m")
    alignment_source = alignment.get("alignment_source")
    if alignment_source not in {None, "unknown", "measured", "assumed", "fitted"}:
        raise ContractError("alignment_source is unsupported")

    execution = _mapping(root["execution"], "PhysicalScene.execution")
    status = execution.get("status")
    if status not in {"draft", "executable"}:
        raise ContractError("PhysicalScene.execution.status is unsupported")
    blockers = _sequence(execution.get("blockers"), "PhysicalScene.execution.blockers")
    fixed_step = execution.get("fixed_step_s")
    if fixed_step is not None and _finite(fixed_step, "fixed_step_s") <= 0.0:
        raise ContractError("fixed_step_s: must be > 0")
    start_time = execution.get("start_time_s")
    if start_time is not None:
        start_time = _finite(start_time, "start_time_s")
    duration = execution.get("duration_s")
    if duration is not None and _finite(duration, "duration_s") <= 0.0:
        raise ContractError("duration_s: must be > 0")

    world = _mapping(root["world"], "PhysicalScene.world")
    gravity = _vector3(world.get("gravity_m_s2"), "PhysicalScene.world.gravity_m_s2")
    model = _mapping(root["model"], "PhysicalScene.model")
    component_fields = {
        "bodies", "articulations", "joints", "constraints", "contacts",
        "materials", "forces", "actuators", "residual_forces",
    }
    _require_fields(model, "PhysicalScene.model", component_fields, exact=True)
    components: dict[str, dict[str, Mapping[str, Any]]] = {}
    for field in component_fields:
        components[field] = _items_by_id(model[field], f"model.{field}")
        for component_id, component in components[field].items():
            if not isinstance(component.get("type"), str):
                raise ContractError(f"model.{field}.{component_id}.type: must be a string")
    parameters = _items_by_id(root["parameters"], "PhysicalScene.parameters")

    if status == "executable":
        if blockers:
            raise ContractError("PhysicalScene.execution.blockers: must be empty")
        if has_observation:
            if scale_source == "unknown" or meters is None:
                raise ContractError("PhysicalScene: metric scale is required when executable")
            if transform is None:
                raise ContractError("PhysicalScene: alignment is required when executable")
        if execution.get("backend") != "newton" or fixed_step is None or duration is None:
            raise ContractError("executable P4 scene requires Newton, duration, and fixed step")
        if start_time is None:
            raise ContractError("executable P4 scene requires start_time_s")
        device = execution.get("device")
        if not isinstance(device, str) or re.fullmatch(r"cuda:\d+", device) is None:
            raise ContractError("executable P4 scene requires a CUDA device")
        steps = round(duration / fixed_step)
        if steps <= 0 or not math.isclose(steps * fixed_step, duration, abs_tol=1e-12):
            raise ContractError("duration_s must be an integer multiple of fixed_step_s")
        if gravity[1] >= 0.0 or not math.isclose(gravity[0], 0.0, abs_tol=1e-12) or not math.isclose(
            gravity[2], 0.0, abs_tol=1e-12
        ):
            raise ContractError("P4 gravity must point down the -Y axis")
        solver = _mapping(execution.get("solver"), "PhysicalScene.execution.solver")
        if solver.get("type") != "xpbd":
            raise ContractError("executable Newton scenes require the XPBD solver")
        if _integer(solver.get("iterations"), "solver.iterations") <= 0:
            raise ContractError("solver.iterations: must be > 0")
        if solver.get("deterministic_mode") != "run_to_run":
            raise ContractError("P4 solver deterministic_mode must be run_to_run")
        n_constraints = len(components["constraints"])
        if len(components["bodies"]) != 1 or n_constraints not in {0, 1}:
            raise ContractError(
                "executable scene requires one body and either one distance constraint "
                "or zero constraints"
            )
        unsupported = component_fields - {"bodies", "constraints"}
        if parameters or any(components[field] for field in unsupported):
            raise ContractError("P4 executable scene contains unsupported components")
        body_id, body = next(iter(components["bodies"].items()))
        if body.get("type") != "rigid_body":
            raise ContractError(f"model.bodies.{body_id}.type: must be rigid_body")
        shape = _mapping(body.get("shape"), f"model.bodies.{body_id}.shape")
        if shape.get("type") != "sphere":
            raise ContractError(f"model.bodies.{body_id}.shape.type: must be sphere")
        if _finite(shape.get("radius_m"), f"model.bodies.{body_id}.shape.radius_m") <= 0.0:
            raise ContractError("sphere radius_m: must be > 0")
        if _finite(body.get("mass_kg"), f"model.bodies.{body_id}.mass_kg") <= 0.0:
            raise ContractError("body mass_kg: must be > 0")
        initial_transform = validate_rigid_transform(
            body.get("T_world_body_initial"),
            f"model.bodies.{body_id}.T_world_body_initial",
        )
        _vector3(body.get("linear_velocity_m_s"), f"model.bodies.{body_id}.linear_velocity_m_s")
        _vector3(body.get("angular_velocity_rad_s"), f"model.bodies.{body_id}.angular_velocity_rad_s")

        if n_constraints == 1:
            constraint_id, constraint = next(iter(components["constraints"].items()))
            if constraint.get("type") != "distance":
                raise ContractError(f"model.constraints.{constraint_id}.type: must be distance")
            if constraint.get("body_id") != body_id:
                raise ContractError(f"model.constraints.{constraint_id}.body_id: unknown body")
            world_anchor = _vector3(
                constraint.get("world_anchor_m"),
                f"model.constraints.{constraint_id}.world_anchor_m",
            )
            body_attachment = _vector3(
                constraint.get("body_attachment_m"),
                f"model.constraints.{constraint_id}.body_attachment_m",
            )
            rest_length = _finite(
                constraint.get("rest_length_m"),
                f"model.constraints.{constraint_id}.rest_length_m",
            )
            if rest_length <= 0.0:
                raise ContractError("constraint rest_length_m: must be > 0")
            attachment_world = tuple(
                sum(initial_transform[axis * 4 + offset] * body_attachment[offset] for offset in range(3))
                + initial_transform[axis * 4 + 3]
                for axis in range(3)
            )
            initial_distance = math.sqrt(
                sum((attachment_world[axis] - world_anchor[axis]) ** 2 for axis in range(3))
            )
            if not math.isclose(initial_distance, rest_length, abs_tol=1e-6):
                raise ContractError("distance constraint attachment must start at rest_length_m")
    _mapping(root["provenance"], "PhysicalScene.provenance")
    _mapping(root["extensions"], "PhysicalScene.extensions")
    canonical_json_bytes(root)
    return root


def validate_simulated_world_state(document: Any) -> Mapping[str, Any]:
    """Validate the project-owned P4 rollout without backend-native objects."""

    root = _mapping(document, "SimulatedWorldState")
    root_fields = {
        "schema", "version", "rollout_id", "source", "simulator",
        "coordinates", "units", "world", "timeline", "bodies", "constraints",
        "execution", "validation", "reproducibility", "warnings", "failures",
    }
    _require_fields(root, "SimulatedWorldState", root_fields, exact=True)
    if root["schema"] != SIMULATED_WORLD_STATE_SCHEMA or root["version"] != CONTRACT_VERSION:
        raise ContractError("SimulatedWorldState: unsupported schema or version")

    source = _mapping(root["source"], "SimulatedWorldState.source")
    if not isinstance(source.get("physical_scene_id"), str) or not source["physical_scene_id"]:
        raise ContractError("SimulatedWorldState.source.physical_scene_id: must be a string")
    _sha256(source.get("physical_scene_sha256"), "physical_scene_sha256")
    if source.get("hash_encoding") != "canonical JSON, sorted keys, UTF-8, no non-finite numbers":
        raise ContractError("SimulatedWorldState.source.hash_encoding is unsupported")
    simulator = _mapping(root["simulator"], "SimulatedWorldState.simulator")
    if simulator.get("backend") != "newton" or simulator.get("solver") != "xpbd":
        raise ContractError("SimulatedWorldState.simulator: expected Newton XPBD")
    for field in (
        "backend_version", "backend_revision", "warp_version", "warp_revision",
        "device", "device_name", "cuda_toolkit", "cuda_driver_api",
    ):
        if not isinstance(simulator.get(field), str) or not simulator[field]:
            raise ContractError(f"SimulatedWorldState.simulator.{field}: must be a string")
    if simulator["backend_version"] != "1.5.1" or simulator["backend_revision"] != "17c82b57c0cf369ee23baa776636fc633b82ccfa":
        raise ContractError("SimulatedWorldState.simulator: unsupported Newton build")
    if simulator["warp_version"] != "1.16.0" or simulator["warp_revision"] != "86ec8b78cbef8bb570a9877e351ac0f365718e30":
        raise ContractError("SimulatedWorldState.simulator: unsupported Warp build")
    if re.fullmatch(r"cuda:\d+", simulator["device"]) is None:
        raise ContractError("SimulatedWorldState.simulator.device: must be CUDA")
    if simulator.get("up_axis") != "+Y":
        raise ContractError("SimulatedWorldState.simulator.up_axis: must be +Y")

    coordinates = _mapping(root["coordinates"], "SimulatedWorldState.coordinates")
    for field, expected in {
        "handedness": "right",
        "up_axis": "+Y",
        "transform_notation": "T_parent_child",
        "vector_convention": "column",
        "matrix_storage": "row_major",
    }.items():
        if coordinates.get(field) != expected:
            raise ContractError(f"SimulatedWorldState.coordinates.{field}: must be {expected}")
    units = _mapping(root["units"], "SimulatedWorldState.units")
    for field, expected in {
        "length": "meter", "mass": "kilogram", "time": "second", "angle": "radian"
    }.items():
        if units.get(field) != expected:
            raise ContractError(f"SimulatedWorldState.units.{field}: must be {expected}")
    _vector3(_mapping(root["world"], "SimulatedWorldState.world").get("gravity_m_s2"), "gravity_m_s2")

    timeline = _mapping(root["timeline"], "SimulatedWorldState.timeline")
    samples = _sequence(timeline.get("samples"), "SimulatedWorldState.timeline.samples")
    if not samples:
        raise ContractError("SimulatedWorldState.timeline.samples: must not be empty")
    fixed_step = _finite(timeline.get("fixed_step_s"), "timeline.fixed_step_s")
    if fixed_step <= 0.0:
        raise ContractError("timeline.fixed_step_s: must be > 0")
    start_time = _finite(timeline.get("start_time_s"), "timeline.start_time_s")
    duration = _finite(timeline.get("duration_s"), "timeline.duration_s")
    if duration <= 0.0:
        raise ContractError("timeline.duration_s: must be > 0")
    previous_time = -math.inf
    for index, raw in enumerate(samples):
        sample = _mapping(raw, f"timeline.samples[{index}]")
        if _integer(sample.get("sample_index"), "sample_index") != index:
            raise ContractError("rollout sample_index must be contiguous from zero")
        timestamp = _finite(sample.get("timestamp_s"), "timestamp_s")
        expected = start_time + index * fixed_step
        if timestamp <= previous_time or not math.isclose(timestamp, expected, abs_tol=1e-9):
            raise ContractError("rollout timestamps must be strictly increasing at fixed_step_s")
        previous_time = timestamp
    expected_steps = round(duration / fixed_step)
    if (
        expected_steps <= 0
        or len(samples) != expected_steps + 1
        or not math.isclose(samples[-1]["timestamp_s"], start_time + duration, abs_tol=1e-9)
    ):
        raise ContractError("rollout timeline must cover duration_s at fixed_step_s")

    bodies = _items_by_id(root["bodies"], "SimulatedWorldState.bodies")
    if len(bodies) != 1:
        raise ContractError("P4 SimulatedWorldState requires one body")
    body_transforms: dict[str, list[tuple[float, ...]]] = {}
    for body_id, body in bodies.items():
        if body.get("type") != "rigid_body":
            raise ContractError(f"bodies.{body_id}.type: must be rigid_body")
        shape = _mapping(body.get("shape"), f"bodies.{body_id}.shape")
        if shape.get("type") != "sphere" or _finite(shape.get("radius_m"), "radius_m") <= 0.0:
            raise ContractError(f"bodies.{body_id}.shape: expected a positive-radius sphere")
        if _finite(body.get("mass_kg"), f"bodies.{body_id}.mass_kg") <= 0.0:
            raise ContractError(f"bodies.{body_id}.mass_kg: must be > 0")
        body_samples = _sequence(body.get("samples"), f"bodies.{body_id}.samples")
        if len(body_samples) != len(samples):
            raise ContractError(f"bodies.{body_id}.samples: must match timeline")
        body_transforms[body_id] = []
        for index, raw in enumerate(body_samples):
            sample = _mapping(raw, f"bodies.{body_id}.samples[{index}]")
            if _integer(sample.get("sample_index"), "sample_index") != index:
                raise ContractError("body sample_index must match timeline")
            transform = validate_rigid_transform(
                sample.get("T_world_body"),
                f"bodies.{body_id}.T_world_body",
            )
            body_transforms[body_id].append(transform)
            _vector3(sample.get("linear_velocity_m_s"), f"bodies.{body_id}.linear_velocity_m_s")
            _vector3(sample.get("angular_velocity_rad_s"), f"bodies.{body_id}.angular_velocity_rad_s")
    constraints = _items_by_id(
        root["constraints"], "SimulatedWorldState.constraints"
    )
    if len(constraints) not in {0, 1}:
        raise ContractError("SimulatedWorldState allows zero or one constraint")
    computed_tether_errors: list[float] = []
    for constraint_id, constraint in constraints.items():
        if constraint.get("type") != "distance":
            raise ContractError(f"constraints.{constraint_id}.type: must be distance")
        if constraint.get("body_id") not in bodies:
            raise ContractError(f"constraints.{constraint_id}.body_id: unknown body")
        anchor = _vector3(constraint.get("world_anchor_m"), f"constraints.{constraint_id}.world_anchor_m")
        attachment = _vector3(
            constraint.get("body_attachment_m"),
            f"constraints.{constraint_id}.body_attachment_m",
        )
        rest_length = _finite(constraint.get("rest_length_m"), "rest_length_m")
        if rest_length <= 0.0:
            raise ContractError("constraint rest_length_m: must be > 0")
        for transform in body_transforms[constraint["body_id"]]:
            attachment_world = tuple(
                sum(transform[axis * 4 + offset] * attachment[offset] for offset in range(3))
                + transform[axis * 4 + 3]
                for axis in range(3)
            )
            distance = math.sqrt(
                sum((attachment_world[axis] - anchor[axis]) ** 2 for axis in range(3))
            )
            computed_tether_errors.append(abs(distance - rest_length))

    execution = _mapping(root["execution"], "SimulatedWorldState.execution")
    if execution.get("status") != "complete":
        raise ContractError("SimulatedWorldState.execution.status: must be complete")
    if _integer(execution.get("steps"), "execution.steps") != expected_steps:
        raise ContractError("execution.steps must match the timeline")
    if _integer(execution.get("output_samples"), "execution.output_samples") != len(samples):
        raise ContractError("execution.output_samples must match the timeline")
    if _finite(execution.get("wall_seconds"), "execution.wall_seconds") < 0.0:
        raise ContractError("execution.wall_seconds: must be >= 0")
    repeat_wall = execution.get("repeat_wall_seconds")
    if repeat_wall is not None and _finite(repeat_wall, "execution.repeat_wall_seconds") < 0.0:
        raise ContractError("execution.repeat_wall_seconds: must be >= 0")
    if _integer(execution.get("peak_gpu_memory_bytes"), "execution.peak_gpu_memory_bytes") < 0:
        raise ContractError("execution.peak_gpu_memory_bytes: must be >= 0")

    validation = _mapping(root["validation"], "SimulatedWorldState.validation")
    for field in ("passed", "finite_state", "time_monotonic", "gravity_matches_contract"):
        if validation.get(field) is not True:
            raise ContractError(f"SimulatedWorldState.validation.{field}: must be true")
    backend_gravity = _vector3(validation.get("backend_gravity_m_s2"), "backend_gravity_m_s2")
    contract_gravity = _vector3(root["world"]["gravity_m_s2"], "gravity_m_s2")
    if any(
        not math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(backend_gravity, contract_gravity)
    ):
        raise ContractError("backend gravity does not match rollout gravity")
    tether_error = validation.get("tether_error_m")
    if computed_tether_errors:
        tether_error = _mapping(tether_error, "validation.tether_error_m")
        maximum = _finite(tether_error.get("maximum"), "tether_error_m.maximum")
        rms = _finite(tether_error.get("rms"), "tether_error_m.rms")
        computed_maximum = max(computed_tether_errors)
        computed_rms = math.sqrt(
            sum(error * error for error in computed_tether_errors) / len(computed_tether_errors)
        )
        if (
            not math.isclose(maximum, computed_maximum, abs_tol=1e-9)
            or not math.isclose(rms, computed_rms, abs_tol=1e-9)
        ):
            raise ContractError("recorded tether error does not match body transforms")
    else:
        if tether_error is not None:
            raise ContractError("unconstrained rollout must not record tether_error_m")
        maximum = 0.0
    invariant_profile = validation.get("invariant_profile", "p4_fixture")
    if invariant_profile not in {"p4_fixture", "observation_aligned", "free_fall"}:
        raise ContractError("validation.invariant_profile: unsupported")
    if invariant_profile == "p4_fixture" and maximum > 1e-5:
        raise ContractError("recorded tether error does not match body transforms")
    position_ranges = _mapping(
        validation.get("body_position_range_m"),
        "validation.body_position_range_m",
    )
    transforms = next(iter(body_transforms.values()))
    computed_ranges = [
        max(transform[axis * 4 + 3] for transform in transforms)
        - min(transform[axis * 4 + 3] for transform in transforms)
        for axis in range(3)
    ]
    for axis, computed in zip(("x", "y", "z"), computed_ranges):
        if not math.isclose(_finite(position_ranges.get(axis), f"body_position_range_m.{axis}"), computed, abs_tol=1e-9):
            raise ContractError("recorded body position range does not match body transforms")
    recorded_varying = _integer(
        position_ranges.get("varying_axis_count_at_0_05_m"),
        "varying_axis_count_at_0_05_m",
    )
    expected_varying = sum(value >= 0.05 for value in computed_ranges)
    if recorded_varying != expected_varying:
        raise ContractError("recorded varying-axis count does not match body transforms")
    spatial_extent = math.sqrt(sum(value * value for value in computed_ranges))
    if invariant_profile == "p4_fixture":
        if recorded_varying != 3 or any(value < 0.05 for value in computed_ranges):
            raise ContractError("P4 rollout must vary by at least 0.05 m on X, Y, and Z")
    elif invariant_profile == "free_fall":
        if computed_ranges[1] < 0.05:
            raise ContractError("free-fall rollout must travel at least 0.05 m along Y")
    elif spatial_extent < 0.02:
        raise ContractError(
            "observation-aligned rollout needs at least 0.02 m of spatial travel. "
            "Planar motion is allowed."
        )

    reproducibility = _mapping(root["reproducibility"], "SimulatedWorldState.reproducibility")
    if reproducibility.get("stochastic_components") is not False:
        raise ContractError("P4 reproducibility.stochastic_components must be false")
    if reproducibility.get("requested_deterministic_mode") != "Warp RUN_TO_RUN":
        raise ContractError("P4 reproducibility deterministic mode is unsupported")
    repeat = _mapping(reproducibility.get("repeat_run"), "reproducibility.repeat_run")
    if repeat.get("performed"):
        delta = _finite(repeat.get("max_abs_transform_delta"), "repeat_run.max_abs_transform_delta")
        tolerance = _finite(repeat.get("tolerance"), "repeat_run.tolerance")
        if repeat.get("within_tolerance") is not True or delta > tolerance:
            raise ContractError("repeated rollout exceeded its transform tolerance")
    _sequence(root["warnings"], "SimulatedWorldState.warnings")
    failures = _sequence(root["failures"], "SimulatedWorldState.failures")
    if failures:
        raise ContractError("complete SimulatedWorldState cannot contain failures")
    canonical_json_bytes(root)
    return root


def validate_physical_motion_observation(document: Any) -> Mapping[str, Any]:
    """Validate metric 3D point evidence used by the P5 objective."""

    root = _mapping(document, "PhysicalMotionObservation")
    root_fields = {
        "schema", "version", "observation_id", "source", "coordinates",
        "units", "track", "provenance", "warnings",
    }
    _require_fields(root, "PhysicalMotionObservation", root_fields, exact=True)
    if (
        root["schema"] != PHYSICAL_MOTION_OBSERVATION_SCHEMA
        or root["version"] != CONTRACT_VERSION
    ):
        raise ContractError("PhysicalMotionObservation: unsupported schema or version")
    if not isinstance(root["observation_id"], str) or not root["observation_id"]:
        raise ContractError("PhysicalMotionObservation.observation_id: must be a string")

    source = _mapping(root["source"], "PhysicalMotionObservation.source")
    _require_fields(
        source,
        "PhysicalMotionObservation.source",
        {"kind", "id", "sha256"},
        exact=True,
    )
    if source.get("kind") not in {
        "synthetic_rollout",
        "scene_observation_human_root",
        "scene_observation_entity_root",
        "metric_sphere_track",
    }:
        raise ContractError("PhysicalMotionObservation.source.kind: unsupported")
    if not isinstance(source.get("id"), str) or not source["id"]:
        raise ContractError("PhysicalMotionObservation.source.id: must be a string")
    _sha256(source.get("sha256"), "PhysicalMotionObservation.source.sha256")

    coordinates = _mapping(root["coordinates"], "PhysicalMotionObservation.coordinates")
    _require_fields(
        coordinates,
        "PhysicalMotionObservation.coordinates",
        {"handedness", "up_axis", "transform_notation", "vector_convention"},
        exact=True,
    )
    for field, expected in {
        "handedness": "right",
        "up_axis": "+Y",
        "transform_notation": "T_parent_child",
        "vector_convention": "column",
    }.items():
        if coordinates.get(field) != expected:
            raise ContractError(
                f"PhysicalMotionObservation.coordinates.{field}: must be {expected}"
            )
    units = _mapping(root["units"], "PhysicalMotionObservation.units")
    _require_fields(
        units,
        "PhysicalMotionObservation.units",
        {"length", "time"},
        exact=True,
    )
    if units.get("length") != "meter" or units.get("time") != "second":
        raise ContractError("PhysicalMotionObservation units must be meters and seconds")

    track = _mapping(root["track"], "PhysicalMotionObservation.track")
    _require_fields(track, "PhysicalMotionObservation.track", {"body_id", "point", "samples"}, exact=True)
    if not isinstance(track["body_id"], str) or not track["body_id"]:
        raise ContractError("PhysicalMotionObservation.track.body_id: must be a string")
    if track["point"] != "body_origin":
        raise ContractError("PhysicalMotionObservation.track.point: must be body_origin")
    samples = _sequence(track["samples"], "PhysicalMotionObservation.track.samples")
    if len(samples) < 2:
        raise ContractError("PhysicalMotionObservation requires at least two samples")
    previous_time = -math.inf
    for index, raw in enumerate(samples):
        sample = _mapping(raw, f"PhysicalMotionObservation.track.samples[{index}]")
        _require_fields(
            sample,
            f"PhysicalMotionObservation.track.samples[{index}]",
            {"sample_index", "timestamp_s", "position_m", "weight"},
            exact=True,
        )
        if _integer(sample["sample_index"], "sample_index") != index:
            raise ContractError("motion sample_index must be contiguous from zero")
        timestamp = _finite(sample["timestamp_s"], "timestamp_s")
        if timestamp < 0.0 or timestamp <= previous_time:
            raise ContractError("motion timestamps must be non-negative and strictly increasing")
        previous_time = timestamp
        _vector3(sample["position_m"], "position_m")
        weight = _finite(sample["weight"], "weight")
        if not 0.0 < weight <= 1.0:
            raise ContractError("motion sample weight must be in (0, 1]")
    provenance = _mapping(root["provenance"], "PhysicalMotionObservation.provenance")
    if not isinstance(provenance.get("synthetic"), bool):
        raise ContractError("PhysicalMotionObservation.provenance.synthetic must be a boolean")
    if (source["kind"] == "synthetic_rollout") != provenance["synthetic"]:
        raise ContractError("motion source kind and synthetic provenance disagree")
    if source["kind"] != "synthetic_rollout" and provenance.get("truth_parameters") is not None:
        raise ContractError("real motion evidence cannot declare truth_parameters")
    _sequence(root["warnings"], "PhysicalMotionObservation.warnings")
    canonical_json_bytes(root)
    return root


def validate_inverse_physics_fit(document: Any) -> Mapping[str, Any]:
    """Validate a completed or explicitly blocked P5 fit report."""

    root = _mapping(document, "InversePhysicsFit")
    root_fields = {
        "schema", "version", "fit_id", "status", "source", "profile",
        "objective", "parameters", "optimizer", "outputs", "execution",
        "validation", "blockers", "warnings", "failures",
    }
    _require_fields(root, "InversePhysicsFit", root_fields, exact=True)
    if root["schema"] != INVERSE_PHYSICS_FIT_SCHEMA or root["version"] != CONTRACT_VERSION:
        raise ContractError("InversePhysicsFit: unsupported schema or version")
    if not isinstance(root["fit_id"], str) or not root["fit_id"]:
        raise ContractError("InversePhysicsFit.fit_id: must be a string")
    status = root["status"]
    if status not in {"COMPLETE", "BLOCKED_INPUT", "FAILED"}:
        raise ContractError("InversePhysicsFit.status: unsupported")
    if root["profile"] not in {
        "tether_length_initial_tangent_velocity_v1",
        "tether_initial_tangent_velocity_fixed_length_v1",
        "free_fall_gravity_v1",
    }:
        raise ContractError("InversePhysicsFit.profile: unsupported")

    source = _mapping(root["source"], "InversePhysicsFit.source")
    _require_fields(
        source,
        "InversePhysicsFit.source",
        {"template_physical_scene", "motion_observation", "scene_observation"},
        exact=True,
    )
    template = _mapping(source.get("template_physical_scene"), "source.template_physical_scene")
    _require_fields(
        template,
        "source.template_physical_scene",
        {"id", "sha256"},
        exact=True,
    )
    if not isinstance(template.get("id"), str) or not template["id"]:
        raise ContractError("source.template_physical_scene.id: must be a string")
    _sha256(template.get("sha256"), "source.template_physical_scene.sha256")
    motion = source.get("motion_observation")
    if motion is not None:
        motion = _mapping(motion, "source.motion_observation")
        _require_fields(motion, "source.motion_observation", {"id", "sha256"}, exact=True)
        if not isinstance(motion.get("id"), str) or not motion["id"]:
            raise ContractError("source.motion_observation.id: must be a string")
        _sha256(motion.get("sha256"), "source.motion_observation.sha256")
    elif status == "COMPLETE":
        raise ContractError("complete inverse fit requires a motion observation")
    scene_observation = source.get("scene_observation")
    if scene_observation is not None:
        scene_observation = _mapping(scene_observation, "source.scene_observation")
        _require_fields(
            scene_observation,
            "source.scene_observation",
            {"id", "sha256"},
            exact=True,
        )
        if not isinstance(scene_observation.get("id"), str) or not scene_observation["id"]:
            raise ContractError("source.scene_observation.id: must be a string")
        _sha256(scene_observation.get("sha256"), "source.scene_observation.sha256")

    objective = _mapping(root["objective"], "InversePhysicsFit.objective")
    _require_fields(
        objective,
        "InversePhysicsFit.objective",
        {
            "type", "sample_count", "mse_m2", "rmse_m", "trajectory_extent_m",
            "normalized_rmse", "initial_mse_m2", "improvement_ratio",
        },
        exact=True,
    )
    if objective.get("type") != "weighted_position_mse_3d":
        raise ContractError("InversePhysicsFit.objective.type: unsupported")
    sample_count = _integer(objective.get("sample_count"), "objective.sample_count")
    metrics = (
        "mse_m2", "rmse_m", "trajectory_extent_m", "normalized_rmse",
        "initial_mse_m2", "improvement_ratio",
    )
    if status == "COMPLETE":
        if sample_count < 2:
            raise ContractError("complete fit requires at least two objective samples")
        for field in metrics:
            value = _finite(objective.get(field), f"objective.{field}")
            if value < 0.0:
                raise ContractError(f"objective.{field}: must be >= 0")
    elif any(objective.get(field) is not None for field in metrics):
        raise ContractError("blocked or failed fit objective metrics must be null")

    parameters = _items_by_id(root["parameters"], "InversePhysicsFit.parameters")
    if root["profile"] == "free_fall_gravity_v1":
        expected_parameter_ids = {
            "gravity_magnitude_m_s2",
            "initial_velocity_y_m_s",
        }
    else:
        expected_parameter_ids = {
            "rest_length_m", "initial_tangent_velocity_u_m_s",
            "initial_tangent_velocity_v_m_s",
        }
    if set(parameters) != expected_parameter_ids:
        raise ContractError("InversePhysicsFit.parameters: unsupported parameter set")
    for parameter_id, parameter in parameters.items():
        _require_fields(
            parameter,
            f"parameters.{parameter_id}",
            {
                "id", "unit", "lower_bound", "upper_bound",
                "initial", "fitted", "truth", "held_fixed",
            },
            exact=True,
        )
        if not isinstance(parameter.get("held_fixed"), bool):
            raise ContractError(f"parameters.{parameter_id}.held_fixed: must be a boolean")
        if (
            root["profile"] == "tether_initial_tangent_velocity_fixed_length_v1"
            and parameter_id == "rest_length_m"
            and parameter["held_fixed"] is not True
        ):
            raise ContractError("fixed-length profile must hold rest_length_m fixed")
        if (
            root["profile"] == "tether_length_initial_tangent_velocity_v1"
            and parameter["held_fixed"]
        ):
            raise ContractError("length-fitting profile cannot hold a parameter fixed")
        if root["profile"] == "free_fall_gravity_v1" and parameter["held_fixed"]:
            raise ContractError("free-fall profile cannot hold a parameter fixed")
        lower = _finite(parameter.get("lower_bound"), f"parameters.{parameter_id}.lower_bound")
        upper = _finite(parameter.get("upper_bound"), f"parameters.{parameter_id}.upper_bound")
        initial = _finite(parameter.get("initial"), f"parameters.{parameter_id}.initial")
        if not lower < upper or not lower <= initial <= upper:
            raise ContractError(f"parameters.{parameter_id}: invalid bounds or initial value")
        fitted = parameter.get("fitted")
        if status == "COMPLETE":
            fitted_value = _finite(fitted, f"parameters.{parameter_id}.fitted")
            if not lower <= fitted_value <= upper:
                raise ContractError(f"parameters.{parameter_id}.fitted: outside bounds")
        elif fitted is not None:
            raise ContractError("blocked or failed fit parameters must not report fitted values")
        truth = parameter.get("truth")
        if truth is not None:
            _finite(truth, f"parameters.{parameter_id}.truth")
        if not isinstance(parameter.get("unit"), str) or not parameter["unit"]:
            raise ContractError(f"parameters.{parameter_id}.unit: must be a string")

    optimizer = _mapping(root["optimizer"], "InversePhysicsFit.optimizer")
    _require_fields(
        optimizer,
        "InversePhysicsFit.optimizer",
        {
            "method", "seed", "population_size", "generations",
            "coordinate_iterations", "objective_evaluations",
        },
        exact=True,
    )
    if optimizer.get("method") != "bounded_differential_evolution_with_coordinate_refinement":
        raise ContractError("InversePhysicsFit.optimizer.method: unsupported")
    for field in (
        "seed", "population_size", "generations",
        "coordinate_iterations", "objective_evaluations",
    ):
        value = _integer(optimizer.get(field), f"optimizer.{field}")
        if value < 0:
            raise ContractError(f"optimizer.{field}: must be >= 0")

    outputs = _mapping(root["outputs"], "InversePhysicsFit.outputs")
    _require_fields(
        outputs,
        "InversePhysicsFit.outputs",
        {"fitted_physical_scene", "simulated_world_state"},
        exact=True,
    )
    for name in ("fitted_physical_scene", "simulated_world_state"):
        artifact = outputs.get(name)
        if status == "COMPLETE":
            artifact = _mapping(artifact, f"outputs.{name}")
            _require_fields(artifact, f"outputs.{name}", {"uri", "sha256"}, exact=True)
            if not isinstance(artifact.get("uri"), str) or not artifact["uri"]:
                raise ContractError(f"outputs.{name}.uri: must be a string")
            _sha256(artifact.get("sha256"), f"outputs.{name}.sha256")
        elif artifact is not None:
            raise ContractError("blocked or failed fit must not report output artifacts")

    execution = _mapping(root["execution"], "InversePhysicsFit.execution")
    _require_fields(
        execution,
        "InversePhysicsFit.execution",
        {"wall_seconds", "peak_gpu_memory_bytes"},
        exact=True,
    )
    if _finite(execution.get("wall_seconds"), "execution.wall_seconds") < 0.0:
        raise ContractError("execution.wall_seconds: must be >= 0")
    if _integer(execution.get("peak_gpu_memory_bytes"), "execution.peak_gpu_memory_bytes") < 0:
        raise ContractError("execution.peak_gpu_memory_bytes: must be >= 0")

    validation = _mapping(root["validation"], "InversePhysicsFit.validation")
    _require_fields(
        validation,
        "InversePhysicsFit.validation",
        {"passed", "rollout_valid", "execution_valid", "quality", "synthetic_recovery"},
        exact=True,
    )
    if not isinstance(validation.get("passed"), bool):
        raise ContractError("InversePhysicsFit.validation.passed: must be a boolean")
    if not isinstance(validation.get("rollout_valid"), bool):
        raise ContractError("InversePhysicsFit.validation.rollout_valid: must be a boolean")
    if not isinstance(validation.get("execution_valid"), bool):
        raise ContractError("InversePhysicsFit.validation.execution_valid: must be a boolean")
    quality = _mapping(validation.get("quality"), "validation.quality")
    _require_fields(
        quality,
        "validation.quality",
        {"status", "rmse_m", "normalized_rmse"},
        exact=True,
    )
    if quality.get("status") not in {"unassessed", "synthetic_checked"}:
        raise ContractError("validation.quality.status: unsupported")
    if quality["status"] == "unassessed":
        if quality.get("rmse_m") is not None:
            _finite(quality.get("rmse_m"), "validation.quality.rmse_m")
        if quality.get("normalized_rmse") is not None:
            _finite(quality.get("normalized_rmse"), "validation.quality.normalized_rmse")
        if validation["passed"] != validation["execution_valid"]:
            raise ContractError("unassessed quality: passed must equal execution_valid")
    else:
        for field in ("rmse_m", "normalized_rmse"):
            value = _finite(quality.get(field), f"validation.quality.{field}")
            if value < 0.0:
                raise ContractError(f"validation.quality.{field}: must be >= 0")
    recovery = _mapping(validation.get("synthetic_recovery"), "validation.synthetic_recovery")
    _require_fields(
        recovery,
        "validation.synthetic_recovery",
        {"performed", "within_tolerance", "max_normalized_parameter_error"},
        exact=True,
    )
    if not isinstance(recovery.get("performed"), bool):
        raise ContractError("synthetic_recovery.performed: must be a boolean")
    if recovery["performed"]:
        if not isinstance(recovery.get("within_tolerance"), bool):
            raise ContractError("synthetic_recovery.within_tolerance: must be a boolean")
        if _finite(
            recovery.get("max_normalized_parameter_error"),
            "synthetic_recovery.max_normalized_parameter_error",
        ) < 0.0:
            raise ContractError("synthetic recovery error must be >= 0")
        if status == "COMPLETE" and recovery["within_tolerance"] is not True:
            raise ContractError("complete synthetic recovery must be within tolerance")
    elif (
        recovery.get("within_tolerance") is not None
        or recovery.get("max_normalized_parameter_error") is not None
    ):
        raise ContractError("unperformed synthetic recovery metrics must be null")
    if quality["status"] == "synthetic_checked" and recovery["performed"] is not True:
        raise ContractError("synthetic_checked quality requires synthetic_recovery.performed")
    if quality["status"] == "unassessed" and recovery["performed"] is True:
        raise ContractError("unassessed quality cannot claim synthetic recovery")

    blockers = _sequence(root["blockers"], "InversePhysicsFit.blockers")
    warnings = _sequence(root["warnings"], "InversePhysicsFit.warnings")
    failures = _sequence(root["failures"], "InversePhysicsFit.failures")
    if status == "COMPLETE":
        if (
            blockers
            or failures
            or validation["execution_valid"] is not True
            or validation["rollout_valid"] is not True
        ):
            raise ContractError("complete inverse fit must execute without blockers or failures")
        if quality["status"] == "synthetic_checked" and validation["passed"] is not True:
            raise ContractError("complete synthetic recovery must set validation.passed")
    elif status == "BLOCKED_INPUT" and not blockers:
        raise ContractError("blocked inverse fit requires at least one blocker")
    elif status == "FAILED" and not failures:
        raise ContractError("failed inverse fit requires at least one failure")
    canonical_json_bytes(root)
    return root


def require_motion_matches_scene_alignment(
    physical_scene: Mapping[str, Any],
    motion: Mapping[str, Any],
) -> None:
    """Real motion must come from the SceneObservation the PhysicalScene names."""

    if motion.get("source", {}).get("kind") not in {
        "scene_observation_human_root",
        "scene_observation_entity_root",
    }:
        return
    aligned = physical_scene.get("observation_alignment", {}).get("observation_sha256")
    source_hash = motion.get("source", {}).get("sha256")
    if aligned != source_hash:
        raise ContractError(
            "PhysicalScene observation_alignment.observation_sha256 must match "
            "PhysicalMotionObservation.source.sha256"
        )


def validate_inverse_fit_artifacts(
    report_document: Mapping[str, Any],
    template_scene_document: Mapping[str, Any],
    motion_document: Mapping[str, Any],
    fitted_scene_document: Mapping[str, Any],
    rollout_document: Mapping[str, Any],
    *,
    fitted_scene_path: Path,
    rollout_path: Path,
) -> None:
    """Verify IDs and hashes across one complete P5 artifact set."""

    report = validate_inverse_physics_fit(report_document)
    if report["status"] != "COMPLETE":
        raise ContractError("inverse fit artifact linkage requires COMPLETE status")
    template = validate_physical_scene(template_scene_document)
    motion = validate_physical_motion_observation(motion_document)
    fitted_scene = validate_physical_scene(fitted_scene_document)
    validate_rollout_source(rollout_document, fitted_scene)

    template_source = report["source"]["template_physical_scene"]
    if template_source["id"] != template["scene_id"]:
        raise ContractError("inverse fit template scene ID does not match")
    template_hash = hashlib.sha256(canonical_json_bytes(template)).hexdigest()
    if template_source["sha256"] != template_hash:
        raise ContractError("inverse fit template scene hash does not match")

    motion_source = report["source"]["motion_observation"]
    if motion_source["id"] != motion["observation_id"]:
        raise ContractError("inverse fit motion observation ID does not match")
    motion_hash = hashlib.sha256(canonical_json_bytes(motion)).hexdigest()
    if motion_source["sha256"] != motion_hash:
        raise ContractError("inverse fit motion observation hash does not match")
    if report["objective"]["sample_count"] != len(motion["track"]["samples"]):
        raise ContractError("inverse fit objective sample count does not match motion")
    if motion["track"]["body_id"] != fitted_scene["model"]["bodies"][0]["id"]:
        raise ContractError("inverse fit motion body does not match fitted scene")

    scene_observation = report["source"]["scene_observation"]
    if motion["source"]["kind"] in {
        "scene_observation_human_root",
        "scene_observation_entity_root",
    }:
        if scene_observation != {
            "id": motion["source"]["id"],
            "sha256": motion["source"]["sha256"],
        }:
            raise ContractError("inverse fit SceneObservation source does not match motion")
        require_motion_matches_scene_alignment(template, motion)
        require_motion_matches_scene_alignment(fitted_scene, motion)
    elif scene_observation is not None:
        raise ContractError("synthetic inverse fit must not claim a SceneObservation source")

    for name, path in (
        ("fitted_physical_scene", fitted_scene_path),
        ("simulated_world_state", rollout_path),
    ):
        artifact = report["outputs"][name]
        if artifact["uri"] != path.name:
            raise ContractError(f"inverse fit {name} URI does not match output file")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if artifact["sha256"] != actual_hash:
            raise ContractError(f"inverse fit {name} hash does not match output file")


def validate_rollout_source(
    rollout: Mapping[str, Any],
    physical_scene: Mapping[str, Any],
) -> None:
    """Confirm a rollout identifies the exact PhysicalScene being returned."""

    validated_rollout = validate_simulated_world_state(rollout)
    validated_scene = validate_physical_scene(physical_scene)
    source = validated_rollout["source"]
    expected_hash = hashlib.sha256(canonical_json_bytes(validated_scene)).hexdigest()
    if source["physical_scene_id"] != validated_scene["scene_id"]:
        raise ContractError("rollout physical_scene_id does not match PhysicalScene")
    if source["physical_scene_sha256"] != expected_hash:
        raise ContractError("rollout PhysicalScene SHA-256 does not match")

    scene_execution = validated_scene["execution"]
    timeline = validated_rollout["timeline"]
    if (
        validated_rollout["coordinates"] != validated_scene["coordinates"]
        or validated_rollout["units"] != validated_scene["units"]
        or validated_rollout["world"] != validated_scene["world"]
        or validated_rollout["simulator"]["device"] != scene_execution["device"]
        or validated_rollout["simulator"]["solver"] != scene_execution["solver"]["type"]
        or not math.isclose(timeline["start_time_s"], scene_execution["start_time_s"], abs_tol=1e-12)
        or not math.isclose(timeline["duration_s"], scene_execution["duration_s"], abs_tol=1e-12)
        or not math.isclose(timeline["fixed_step_s"], scene_execution["fixed_step_s"], abs_tol=1e-12)
    ):
        raise ContractError("rollout world or execution settings do not match PhysicalScene")

    scene_body = validated_scene["model"]["bodies"][0]
    rollout_body = validated_rollout["bodies"][0]
    initial_sample = rollout_body["samples"][0]
    if (
        rollout_body["id"] != scene_body["id"]
        or rollout_body["type"] != scene_body["type"]
        or rollout_body["shape"] != scene_body["shape"]
        or not math.isclose(rollout_body["mass_kg"], scene_body["mass_kg"], abs_tol=1e-12)
        or any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(
                initial_sample["T_world_body"],
                scene_body["T_world_body_initial"],
            )
        )
        or any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(
                initial_sample["linear_velocity_m_s"],
                scene_body["linear_velocity_m_s"],
            )
        )
        or any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(
                initial_sample["angular_velocity_rad_s"],
                scene_body["angular_velocity_rad_s"],
            )
        )
    ):
        raise ContractError("rollout body metadata or initial state does not match PhysicalScene")
    if validated_rollout["constraints"] != validated_scene["model"]["constraints"]:
        raise ContractError("rollout constraints do not match PhysicalScene")


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
    if schema == SIMULATED_WORLD_STATE_SCHEMA:
        return validate_simulated_world_state(document)
    if schema == PHYSICAL_MOTION_OBSERVATION_SCHEMA:
        return validate_physical_motion_observation(document)
    if schema == INVERSE_PHYSICS_FIT_SCHEMA:
        return validate_inverse_physics_fit(document)
    raise ContractError(f"{path}: unknown schema {schema!r}")
