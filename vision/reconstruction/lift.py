"""Lift SAM2 masks through per-frame DA3 depth and camera into observation XYZ."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import ContractError
from .transforms import transform_point
from .trajectory_bridge import MIN_MASK_PIXELS, as_bool_mask


def resize_mask_to_depth(
    mask: np.ndarray,
    depth_hw: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor resize a source-resolution mask onto a depth map."""

    binary = as_bool_mask(mask)
    height, width = depth_hw
    if binary.shape == (height, width):
        return binary
    source_h, source_w = binary.shape
    if source_h <= 0 or source_w <= 0 or height <= 0 or width <= 0:
        raise ContractError("mask and depth sizes must be positive")
    ys = np.linspace(0, source_h, height, endpoint=False).astype(np.int64)
    xs = np.linspace(0, source_w, width, endpoint=False).astype(np.int64)
    return binary[np.ix_(ys, xs)]


def erode_mask(mask: np.ndarray) -> np.ndarray:
    """One-pixel 8-connected erosion so the lift prefers interior pixels."""

    binary = as_bool_mask(mask)
    padded = np.pad(binary.astype(np.uint8), 1, mode="constant")
    eroded = np.ones(binary.shape, dtype=bool)
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            eroded &= padded[
                1 + delta_y : 1 + delta_y + binary.shape[0],
                1 + delta_x : 1 + delta_x + binary.shape[1],
            ].astype(bool)
    return eroded


def _valid_mask_pixels(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    confidence: np.ndarray | None,
    confidence_floor: float | None,
) -> tuple[np.ndarray, int]:
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2:
        raise ContractError("depth must be a 2D map")
    binary = resize_mask_to_depth(mask, z.shape)
    interior = erode_mask(binary)
    valid = interior & np.isfinite(z) & (z > 0.0)
    if int(np.count_nonzero(valid)) < MIN_MASK_PIXELS:
        valid = binary & np.isfinite(z) & (z > 0.0)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape != z.shape:
            raise ContractError("confidence map must match depth shape")
        valid &= np.isfinite(conf)
        if confidence_floor is not None:
            valid &= conf >= float(confidence_floor)
        elif int(np.count_nonzero(valid)) > MIN_MASK_PIXELS:
            finite = conf[valid]
            keep = conf >= float(np.median(finite))
            candidate = valid & keep
            if int(np.count_nonzero(candidate)) >= MIN_MASK_PIXELS:
                valid = candidate
    return valid, int(np.count_nonzero(binary))


def unproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    """Unproject depth-map pixels with this frame's K. Returns camera XYZ."""

    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    cx = float(intrinsics["cx_px"])
    cy = float(intrinsics["cy_px"])
    skew = float(intrinsics.get("skew_px", 0.0))
    if fx <= 0.0 or fy <= 0.0 or np.any(depths <= 0.0):
        raise ContractError("unproject requires positive depth and focal length")
    v = ys.astype(np.float64)
    u = xs.astype(np.float64)
    y = (v - cy) / fy
    x = (u - cx - skew * y) / fx
    z = depths.astype(np.float64)
    return np.stack((x * z, y * z, z), axis=1)


def robust_3d_center(points: np.ndarray) -> np.ndarray:
    """Coordinate-wise median, then drop far 3D outliers and re-median."""

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ContractError("robust 3D center needs an Nx3 point list")
    center = np.median(points, axis=0)
    if points.shape[0] < 4:
        return center
    delta = points - center
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    mad = float(np.median(distance))
    if mad <= 1e-9:
        return center
    keep = distance <= (3.0 * 1.4826 * mad)
    if int(np.count_nonzero(keep)) < max(MIN_MASK_PIXELS, points.shape[0] // 4):
        return center
    return np.median(points[keep], axis=0)


def robust_mask_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    confidence_floor: float | None = None,
) -> dict[str, float | int | None]:
    """Legacy depth/pixel stats. Prefer lift_mask_to_world for the 3D center."""

    valid, mask_pixels = _valid_mask_pixels(
        depth,
        mask,
        confidence=confidence,
        confidence_floor=confidence_floor,
    )
    count = int(np.count_nonzero(valid))
    if count < MIN_MASK_PIXELS:
        return {
            "depth": None,
            "u": None,
            "v": None,
            "used_pixels": count,
            "mask_pixels": mask_pixels,
        }
    ys, xs = np.nonzero(valid)
    z = np.asarray(depth, dtype=np.float64)
    return {
        "depth": float(np.median(z[valid])),
        "u": float(np.median(xs)),
        "v": float(np.median(ys)),
        "used_pixels": count,
        "mask_pixels": mask_pixels,
    }


def sample_intrinsics_for_frame(
    camera: Mapping[str, Any],
    sample_index: int,
    da3_extension: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Use the matching sample K when DA3 reports that intrinsics vary."""

    per_sample = None
    if isinstance(da3_extension, Mapping):
        raw = da3_extension.get("sample_intrinsics")
        if isinstance(raw, Sequence) and sample_index < len(raw):
            per_sample = raw[sample_index]
    if per_sample is not None:
        return {
            "fx_px": float(per_sample["fx_px"]),
            "fy_px": float(per_sample["fy_px"]),
            "cx_px": float(per_sample["cx_px"]),
            "cy_px": float(per_sample["cy_px"]),
            "skew_px": float(per_sample.get("skew_px", 0.0)),
        }
    if isinstance(da3_extension, Mapping) and da3_extension.get("intrinsics_vary"):
        raise ContractError(
            "DA3 intrinsics vary by sample, but sample_intrinsics is missing. "
            "Do not fall back to sample-0 K."
        )
    k = camera["intrinsics"]
    return {
        "fx_px": float(k["fx_px"]),
        "fy_px": float(k["fy_px"]),
        "cx_px": float(k["cx_px"]),
        "cy_px": float(k["cy_px"]),
        "skew_px": float(k.get("skew_px", 0.0)),
    }


def camera_pose_for_sample(
    camera: Mapping[str, Any],
    sample_index: int,
) -> list[float]:
    for pose in camera["poses"]:
        if int(pose["sample_index"]) == sample_index:
            return list(pose["T_world_camera"])
    raise ContractError(f"camera has no pose for sample {sample_index}")


def _depth_intrinsics(
    intrinsics: Mapping[str, float],
    *,
    depth_hw: tuple[int, int],
    source_size_px: tuple[int, int] | None,
) -> dict[str, float]:
    if source_size_px is None:
        return {
            "fx_px": float(intrinsics["fx_px"]),
            "fy_px": float(intrinsics["fy_px"]),
            "cx_px": float(intrinsics["cx_px"]),
            "cy_px": float(intrinsics["cy_px"]),
            "skew_px": float(intrinsics.get("skew_px", 0.0)),
        }
    src_w, src_h = source_size_px
    depth_h, depth_w = depth_hw
    return {
        "fx_px": float(intrinsics["fx_px"]) * (depth_w / src_w),
        "fy_px": float(intrinsics["fy_px"]) * (depth_h / src_h),
        "cx_px": float(intrinsics["cx_px"]) * (depth_w / src_w),
        "cy_px": float(intrinsics["cy_px"]) * (depth_h / src_h),
        "skew_px": float(intrinsics.get("skew_px", 0.0)) * (depth_w / src_w),
    }


def lift_mask_to_world(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    T_world_camera: Sequence[float],
    intrinsics: Mapping[str, float],
    confidence: np.ndarray | None = None,
    confidence_floor: float | None = None,
    source_size_px: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Unproject interior mask pixels and take a robust 3D center."""

    valid, mask_pixels = _valid_mask_pixels(
        depth,
        mask,
        confidence=confidence,
        confidence_floor=confidence_floor,
    )
    count = int(np.count_nonzero(valid))
    if count < MIN_MASK_PIXELS:
        return None
    ys, xs = np.nonzero(valid)
    z = np.asarray(depth, dtype=np.float64)
    k = _depth_intrinsics(intrinsics, depth_hw=z.shape, source_size_px=source_size_px)
    camera_points = unproject_pixels(xs, ys, z[ys, xs], k)
    center = robust_3d_center(camera_points)
    world = transform_point(T_world_camera, center)
    if center[2] <= 0.0:
        return None
    depth_u = k["fx_px"] * (center[0] / center[2]) + k["cx_px"] + k["skew_px"] * (
        center[1] / center[2]
    )
    depth_v = k["fy_px"] * (center[1] / center[2]) + k["cy_px"]
    if source_size_px is not None:
        src_w, src_h = source_size_px
        depth_h, depth_w = z.shape
        source_u = depth_u * (src_w / depth_w)
        source_v = depth_v * (src_h / depth_h)
    else:
        source_u = depth_u
        source_v = depth_v
    return {
        "root": [float(world[0]), float(world[1]), float(world[2])],
        "pixel": [float(source_u), float(source_v)],
        "depth": float(center[2]),
        "used_pixels": count,
        "mask_pixels": mask_pixels,
        "visible": True,
        "estimator": "robust_3d_center",
    }


def load_da3_depth_artifact(path) -> dict[str, np.ndarray]:
    """Load the per-frame depth bundle written by the DA3 adapter."""

    data = np.load(path)
    required = ("depth", "processed_intrinsics", "processed_size", "source_size")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ContractError(f"DA3 depth artifact is missing {missing}")
    return {name: data[name] for name in data.files}
