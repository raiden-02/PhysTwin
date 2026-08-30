"""Build a metric free-fall motion observation from known-radius sphere lifts."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import canonical_json_bytes, validate_physical_motion_observation, validate_physical_scene
from .sphere import reconstruct_metric_ball


IDENTITY_TRANSFORM = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]

MIN_PLAUSIBLE_DEPTH_M = 0.30
MAX_PLAUSIBLE_DEPTH_M = 8.00
MAX_FRAME_DEPTH_JUMP_M = 0.40
MAX_STATIC_CAMERA_TRANSLATION_WORLD = 0.05


def camera_translation_world(transform: Sequence[float]) -> list[float]:
    return [float(transform[3]), float(transform[7]), float(transform[11])]


def static_camera_report(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Measure first-to-last DA3 camera translation in reconstruction units.

    SceneObservation scale from DA3 is relative. These numbers are not meters
    unless a later metric conversion has been applied.
    """

    poses = list(camera.get("poses") or [])
    if len(poses) < 2:
        return {
            "assumed_static": True,
            "pose_count": len(poses),
            "max_translation_world": 0.0,
            "units": "da3_reconstruction",
            "scale": "relative",
            "accepted": True,
            "reason": "fewer than two poses; static camera is assumed",
        }
    first = camera_translation_world(poses[0]["T_world_camera"])
    max_delta = 0.0
    for pose in poses[1:]:
        point = camera_translation_world(pose["T_world_camera"])
        delta = math.sqrt(sum((point[axis] - first[axis]) ** 2 for axis in range(3)))
        max_delta = max(max_delta, delta)
    accepted = max_delta <= MAX_STATIC_CAMERA_TRANSLATION_WORLD
    return {
        "assumed_static": True,
        "pose_count": len(poses),
        "max_translation_world": max_delta,
        "units": "da3_reconstruction",
        "scale": "relative",
        "accepted": accepted,
        "reason": None if accepted else (
            f"DA3 camera translation {max_delta:.4f} reconstruction units exceeds "
            f"{MAX_STATIC_CAMERA_TRANSLATION_WORLD:.2f} relative-scale static-camera check"
        ),
    }


def summarize_intrinsics(samples: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Describe DA3 K variation. Used for provenance, not for gravity fitting."""

    if not samples:
        return {"count": 0}
    keys = ("fx_px", "fy_px", "cx_px", "cy_px", "skew_px")
    report: dict[str, Any] = {"count": len(samples), "policy": "per_frame"}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in samples]
        lo = min(values)
        hi = max(values)
        mid = 0.5 * (lo + hi)
        report[key] = {
            "min": lo,
            "max": hi,
            "span": hi - lo,
            "relative_span": 0.0 if abs(mid) < 1e-12 else (hi - lo) / abs(mid),
        }
    return report


def _intrinsics_for_frame(
    frame: Mapping[str, Any],
    shared: Mapping[str, float] | None,
) -> Mapping[str, float]:
    own = frame.get("intrinsics")
    if isinstance(own, Mapping) and "fx_px" in own:
        return own
    if shared is not None:
        return shared
    raise ValueError("each reconstruction sample needs its own or a shared camera K")


def lift_and_filter_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    radius_m: float,
    intrinsics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Reconstruct each mask, then reject implausible depths and scale jumps.

    Prefer a per-frame `intrinsics` field on each sample. A shared K is only
    for tests or cameras that truly publish one matrix.
    """

    records = []
    accepted_depths: list[float] = []
    for frame in frames:
        k = _intrinsics_for_frame(frame, intrinsics)
        lifted = reconstruct_metric_ball(
            frame["mask"],
            radius_m=radius_m,
            intrinsics=k,
        )
        record = {
            "sample_index": int(frame["sample_index"]),
            "source_frame": int(frame["source_frame"]),
            "timestamp_s": float(frame["timestamp_s"]),
            "mask_area_px": lifted.get("area_px"),
            "center_u": lifted.get("center_u"),
            "center_v": lifted.get("center_v"),
            "radius_area_px": lifted.get("radius_area_px"),
            "radius_horizontal_px": lifted.get("radius_horizontal_px"),
            "radius_px": lifted.get("radius_px"),
            "depth_m": lifted.get("depth_m"),
            "position_m": lifted.get("position_m"),
            "intrinsics": {
                "fx_px": float(k["fx_px"]),
                "fy_px": float(k["fy_px"]),
                "cx_px": float(k["cx_px"]),
                "cy_px": float(k["cy_px"]),
                "skew_px": float(k.get("skew_px", 0.0)),
            },
            "accepted": bool(lifted.get("accepted")),
            "reason": lifted.get("reason"),
        }
        if record["accepted"]:
            depth = float(record["depth_m"])
            if not math.isfinite(depth) or depth < MIN_PLAUSIBLE_DEPTH_M or depth > MAX_PLAUSIBLE_DEPTH_M:
                record["accepted"] = False
                record["reason"] = f"depth {depth} m is outside [{MIN_PLAUSIBLE_DEPTH_M}, {MAX_PLAUSIBLE_DEPTH_M}]"
            elif accepted_depths and abs(depth - accepted_depths[-1]) > MAX_FRAME_DEPTH_JUMP_M:
                record["accepted"] = False
                record["reason"] = (
                    f"depth jumped {abs(depth - accepted_depths[-1]):.3f} m from the previous accepted frame"
                )
        if record["accepted"]:
            accepted_depths.append(float(record["depth_m"]))
        records.append(record)
    return {
        "frames": records,
        "accepted": [item for item in records if item["accepted"]],
        "rejected": [item for item in records if not item["accepted"]],
    }


def motion_observation_from_sphere_track(
    accepted: Sequence[Mapping[str, Any]],
    *,
    observation_id: str,
    source_id: str,
    source_sha256: str,
    body_id: str = "ball",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(accepted) < 2:
        raise ValueError("metric sphere track needs at least two accepted frames")
    t0 = float(accepted[0]["timestamp_s"])
    document = {
        "schema": "phystwin.physical_motion_observation",
        "version": 1,
        "observation_id": observation_id,
        "source": {
            "kind": "metric_sphere_track",
            "id": source_id,
            "sha256": source_sha256,
        },
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Y",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
        },
        "units": {"length": "meter", "time": "second"},
        "track": {
            "body_id": body_id,
            "point": "body_origin",
            "samples": [
                {
                    "sample_index": index,
                    "timestamp_s": float(item["timestamp_s"]) - t0,
                    "position_m": [float(value) for value in item["position_m"]],
                    "weight": 1.0,
                }
                for index, item in enumerate(accepted)
            ],
        },
        "provenance": {
            "synthetic": False,
            "reconstruction": "known_radius_sphere",
            "video_time_offset_s": t0,
            **dict(provenance or {}),
        },
        "warnings": [
            "Metric depth uses the measured ball radius and pinhole silhouette geometry. DA3 depth is not used.",
            "Static camera is assumed. Gravity direction is -Y from an assumed level camera.",
        ],
    }
    return dict(validate_physical_motion_observation(document))


def stamp_free_fall_template(
    template: Mapping[str, Any],
    motion: Mapping[str, Any],
    *,
    radius_m: float,
) -> dict[str, Any]:
    """Place the body at the first observed point. Keep template gravity as the search initial."""

    scene = copy.deepcopy(dict(template))
    first = motion["track"]["samples"][0]["position_m"]
    last_time = float(motion["track"]["samples"][-1]["timestamp_s"])
    step = float(scene["execution"]["fixed_step_s"])
    needed = max(float(scene["execution"]["duration_s"]), last_time)
    steps = max(1, math.ceil(needed / step - 1e-12))
    scene["execution"]["start_time_s"] = 0.0
    scene["execution"]["duration_s"] = steps * step
    scene["model"]["bodies"][0]["shape"]["radius_m"] = float(radius_m)
    scene["model"]["bodies"][0]["linear_velocity_m_s"] = [0.0, 0.0, 0.0]
    transform = list(IDENTITY_TRANSFORM)
    transform[3] = float(first[0])
    transform[7] = float(first[1])
    transform[11] = float(first[2])
    scene["model"]["bodies"][0]["T_world_body_initial"] = transform
    motion_hash = hashlib.sha256(canonical_json_bytes(motion)).hexdigest()
    scene["observation_alignment"] = {
        "observation_uri": f"{motion['observation_id']}.json",
        "observation_sha256": motion_hash,
        "meters_per_observation_unit": 1.0,
        "scale_source": "measured",
        "alignment_source": "assumed",
        "up_mode": "level_camera",
        "up_source": "assumed",
        "T_scene_observation_m": list(IDENTITY_TRANSFORM),
    }
    scene["scene_id"] = f"{scene['scene_id']}-aligned"
    return dict(validate_physical_scene(scene))
