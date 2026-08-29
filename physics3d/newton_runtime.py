"""Newton/Warp adapter for the first executable PhysicalScene payload."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from vision.reconstruction.contracts import (
    canonical_json_bytes,
    validate_physical_scene,
    validate_simulated_world_state,
)


NEWTON_REVISION = "17c82b57c0cf369ee23baa776636fc633b82ccfa"
WARP_REVISION = "86ec8b78cbef8bb570a9877e351ac0f365718e30"


@dataclass(frozen=True)
class RunData:
    transforms: tuple[tuple[float, ...], ...]
    linear_velocities: tuple[tuple[float, float, float], ...]
    angular_velocities: tuple[tuple[float, float, float], ...]
    backend_gravity: tuple[float, float, float]
    wall_seconds: float
    steps: int
    peak_gpu_memory_bytes: int
    memory_measurement: str
    simulator: dict[str, Any]


def _matrix_to_quaternion(values: list[float]) -> tuple[float, float, float, float]:
    """Convert the contract's row-major rotation to an XYZW quaternion."""

    m00, m01, m02 = values[0], values[1], values[2]
    m10, m11, m12 = values[4], values[5], values[6]
    m20, m21, m22 = values[8], values[9], values[10]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    norm = math.sqrt(sum(item * item for item in quaternion))
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def _transform_to_row_major(transform: np.ndarray) -> tuple[float, ...]:
    """Convert Warp [x,y,z,qx,qy,qz,qw] to contract T_world_body."""

    x, y, z, qx, qy, qz, qw = (float(item) for item in transform)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        x,
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx),
        y,
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (xx + yy),
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def transform_point(T_world_body: tuple[float, ...] | list[float], point_body: list[float]) -> tuple[float, float, float]:
    """Apply a row-major contract transform to one body-local point."""

    return (
        T_world_body[0] * point_body[0] + T_world_body[1] * point_body[1] + T_world_body[2] * point_body[2] + T_world_body[3],
        T_world_body[4] * point_body[0] + T_world_body[5] * point_body[1] + T_world_body[6] * point_body[2] + T_world_body[7],
        T_world_body[8] * point_body[0] + T_world_body[9] * point_body[1] + T_world_body[10] * point_body[2] + T_world_body[11],
    )


def _read_state(state: newton.State, body_index: int) -> tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float]]:
    transform = _transform_to_row_major(state.body_q.numpy()[body_index])
    twist = state.body_qd.numpy()[body_index]
    linear = tuple(float(item) for item in twist[:3])
    angular = tuple(float(item) for item in twist[3:6])
    return transform, linear, angular  # type: ignore[return-value]


def _run_once(scene: dict[str, Any]) -> RunData:
    execution = scene["execution"]
    body = scene["model"]["bodies"][0]
    constraint = scene["model"]["constraints"][0]
    device_name = execution["device"]
    wp.init()
    if newton.__version__ != "1.5.1":
        raise RuntimeError(f"P4 is pinned to Newton 1.5.1, got {newton.__version__}")
    if wp.__version__ != "1.16.0":
        raise RuntimeError(f"P4 is pinned to Warp 1.16.0, got {wp.__version__}")
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise RuntimeError(f"P4 requires a CUDA device, got {device}")
    newton.use_coord_layout_targets = True

    started = time.perf_counter()

    transform = body["T_world_body_initial"]
    quaternion = _matrix_to_quaternion(transform)
    builder = newton.ModelBuilder(
        up_axis=newton.Axis.Y,
        gravity=tuple(scene["world"]["gravity_m_s2"]),
    )
    body_index = builder.add_link(
        xform=wp.transform(
            p=wp.vec3(transform[3], transform[7], transform[11]),
            q=wp.quat(*quaternion),
        ),
        label=body["id"],
    )
    radius = float(body["shape"]["radius_m"])
    volume = 4.0 * math.pi * radius**3 / 3.0
    shape_config = newton.ModelBuilder.ShapeConfig(
        density=float(body["mass_kg"]) / volume,
        has_shape_collision=False,
    )
    builder.add_shape_sphere(body_index, radius=radius, cfg=shape_config, label=f"{body['id']}_shape")
    builder.body_qd[body_index] = wp.spatial_vector(
        *body["linear_velocity_m_s"],
        *body["angular_velocity_rad_s"],
    )
    joint_index = builder.add_joint_distance(
        parent=-1,
        child=body_index,
        parent_xform=wp.transform(
            p=wp.vec3(*constraint["world_anchor_m"]),
            q=wp.quat_identity(),
        ),
        child_xform=wp.transform(
            p=wp.vec3(*constraint["body_attachment_m"]),
            q=wp.quat_identity(),
        ),
        min_distance=float(constraint["rest_length_m"]),
        max_distance=float(constraint["rest_length_m"]),
        label=constraint["id"],
    )
    builder.add_articulation([joint_index], label="p4_tether")

    model = builder.finalize(device=device)
    model.set_gravity(tuple(scene["world"]["gravity_m_s2"]))
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=execution["solver"]["iterations"],
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    duration = float(execution["duration_s"])
    step = float(execution["fixed_step_s"])
    steps = int(round(duration / step))
    if not math.isclose(steps * step, duration, abs_tol=1e-12):
        raise ValueError("duration_s must be an integer multiple of fixed_step_s")

    transforms: list[tuple[float, ...]] = []
    linear_velocities: list[tuple[float, float, float]] = []
    angular_velocities: list[tuple[float, float, float]] = []

    def record() -> None:
        current_transform, linear, angular = _read_state(state_0, body_index)
        transforms.append(current_transform)
        linear_velocities.append(linear)
        angular_velocities.append(angular)

    record()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, step)
        state_0, state_1 = state_1, state_0
        record()

    wp.synchronize_device(device)
    peak_gpu_memory = int(wp.get_mempool_used_mem_high(device))
    gravity_array = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1)
    backend_gravity = tuple(float(item) for item in gravity_array[:3])
    wall_seconds = time.perf_counter() - started
    toolkit = wp.get_cuda_toolkit_version()
    driver = wp.get_cuda_driver_version()
    simulator = {
        "backend": "newton",
        "backend_version": newton.__version__,
        "backend_revision": NEWTON_REVISION,
        "solver": "xpbd",
        "warp_version": wp.__version__,
        "warp_revision": WARP_REVISION,
        "device": str(device),
        "device_name": device.name,
        "up_axis": "+Y",
        "cuda_toolkit": ".".join(str(item) for item in toolkit),
        "cuda_driver_api": ".".join(str(item) for item in driver),
    }
    return RunData(
        transforms=tuple(transforms),
        linear_velocities=tuple(linear_velocities),
        angular_velocities=tuple(angular_velocities),
        backend_gravity=backend_gravity,  # type: ignore[arg-type]
        wall_seconds=wall_seconds,
        steps=steps,
        peak_gpu_memory_bytes=peak_gpu_memory,
        memory_measurement="Warp CUDA mempool used-memory high-water mark",
        simulator=simulator,
    )


def _max_transform_delta(first: RunData, second: RunData) -> float:
    return max(
        abs(left - right)
        for first_transform, second_transform in zip(first.transforms, second.transforms)
        for left, right in zip(first_transform, second_transform)
    )


def simulate_physical_scene(document: dict[str, Any], *, repeat_check: bool = False) -> dict[str, Any]:
    """Execute the P4 Newton scene and return a validated rollout document."""

    scene = dict(validate_physical_scene(document))
    first = _run_once(scene)
    second = _run_once(scene) if repeat_check else None
    source_hash = hashlib.sha256(canonical_json_bytes(scene)).hexdigest()
    execution = scene["execution"]
    body = scene["model"]["bodies"][0]
    constraint = scene["model"]["constraints"][0]
    start_time = float(execution["start_time_s"])
    step = float(execution["fixed_step_s"])
    timeline_samples = [
        {"sample_index": index, "timestamp_s": start_time + index * step}
        for index in range(len(first.transforms))
    ]
    body_samples = [
        {
            "sample_index": index,
            "T_world_body": list(transform),
            "linear_velocity_m_s": list(first.linear_velocities[index]),
            "angular_velocity_rad_s": list(first.angular_velocities[index]),
        }
        for index, transform in enumerate(first.transforms)
    ]

    anchor = constraint["world_anchor_m"]
    attachment = constraint["body_attachment_m"]
    rest_length = float(constraint["rest_length_m"])
    tether_errors = []
    positions = [[], [], []]
    finite_state = True
    for transform, linear, angular in zip(
        first.transforms,
        first.linear_velocities,
        first.angular_velocities,
    ):
        point = transform_point(transform, attachment)
        distance = math.sqrt(sum((point[axis] - anchor[axis]) ** 2 for axis in range(3)))
        tether_errors.append(abs(distance - rest_length))
        for axis in range(3):
            positions[axis].append(transform[axis * 4 + 3])
        finite_state = finite_state and all(
            math.isfinite(value) for value in (*transform, *linear, *angular)
        )
    max_error = max(tether_errors)
    rms_error = math.sqrt(sum(error * error for error in tether_errors) / len(tether_errors))
    axis_ranges = [max(values) - min(values) for values in positions]
    varying_axis_count = sum(value >= 0.05 for value in axis_ranges)
    configured_gravity = tuple(float(item) for item in scene["world"]["gravity_m_s2"])
    gravity_matches = all(
        math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(first.backend_gravity, configured_gravity)
    )
    time_monotonic = all(
        timeline_samples[index]["timestamp_s"] < timeline_samples[index + 1]["timestamp_s"]
        for index in range(len(timeline_samples) - 1)
    )
    passed = finite_state and gravity_matches and time_monotonic and max_error <= 1e-5 and varying_axis_count == 3
    if not passed:
        raise RuntimeError(
            "P4 rollout validation failed: "
            f"finite={finite_state}, gravity={gravity_matches}, time={time_monotonic}, "
            f"max_tether_error={max_error:.6g}, varying_axes={varying_axis_count}"
        )

    repeat_tolerance = 1e-7
    repeat_delta = _max_transform_delta(first, second) if second else None
    repeat_ok = repeat_delta is None or repeat_delta <= repeat_tolerance
    if not repeat_ok:
        raise RuntimeError(
            f"Newton RUN_TO_RUN determinism check failed: {repeat_delta} > {repeat_tolerance}"
        )
    warnings = [
        "Newton XPBD enforces the distance joint numerically. See validation.tether_error_m."
    ]

    rollout = {
        "schema": "phystwin.simulated_world_state",
        "version": 1,
        "rollout_id": f"{scene['scene_id']}-{source_hash[:12]}",
        "source": {
            "physical_scene_id": scene["scene_id"],
            "physical_scene_sha256": source_hash,
            "hash_encoding": "canonical JSON, sorted keys, UTF-8, no non-finite numbers",
        },
        "simulator": first.simulator,
        "coordinates": dict(scene["coordinates"]),
        "units": dict(scene["units"]),
        "world": {"gravity_m_s2": list(configured_gravity)},
        "timeline": {
            "start_time_s": start_time,
            "duration_s": float(execution["duration_s"]),
            "fixed_step_s": step,
            "samples": timeline_samples,
        },
        "bodies": [
            {
                "id": body["id"],
                "type": body["type"],
                "shape": dict(body["shape"]),
                "mass_kg": float(body["mass_kg"]),
                "samples": body_samples,
            }
        ],
        "constraints": [dict(constraint)],
        "execution": {
            "status": "complete",
            "steps": first.steps,
            "output_samples": len(first.transforms),
            "wall_seconds": first.wall_seconds,
            "repeat_wall_seconds": second.wall_seconds if second else None,
            "peak_gpu_memory_bytes": first.peak_gpu_memory_bytes,
            "gpu_memory_measurement": first.memory_measurement,
        },
        "validation": {
            "passed": True,
            "finite_state": finite_state,
            "time_monotonic": time_monotonic,
            "gravity_matches_contract": gravity_matches,
            "backend_gravity_m_s2": list(first.backend_gravity),
            "tether_error_m": {
                "maximum": max_error,
                "rms": rms_error,
            },
            "body_position_range_m": {
                "x": axis_ranges[0],
                "y": axis_ranges[1],
                "z": axis_ranges[2],
                "varying_axis_count_at_0_05_m": varying_axis_count,
            },
        },
        "reproducibility": {
            "stochastic_components": False,
            "random_seed": None,
            "requested_deterministic_mode": "Warp RUN_TO_RUN",
            "repeat_run": {
                "performed": second is not None,
                "max_abs_transform_delta": repeat_delta,
                "tolerance": repeat_tolerance,
                "within_tolerance": repeat_ok if second else None,
            },
        },
        "warnings": warnings,
        "failures": [],
    }
    validate_simulated_world_state(rollout)
    return rollout
