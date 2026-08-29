"""Lift SAM2 masks through per-frame DA3 depth and camera into observation XYZ."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import ContractError
from .transforms import transform_point, unproject_depth_point
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


def robust_mask_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    confidence_floor: float | None = None,
) -> dict[str, float | int | None]:
    """Median depth over valid masked pixels. Rejects a single centroid sample."""

    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2:
        raise ContractError("depth must be a 2D map")
    binary = resize_mask_to_depth(mask, z.shape)
    valid = binary & np.isfinite(z) & (z > 0.0)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape != z.shape:
            raise ContractError("confidence map must match depth shape")
        if confidence_floor is not None:
            valid &= np.isfinite(conf) & (conf >= float(confidence_floor))
    count = int(np.count_nonzero(valid))
    if count < MIN_MASK_PIXELS:
        return {
            "depth": None,
            "u": None,
            "v": None,
            "used_pixels": count,
            "mask_pixels": int(np.count_nonzero(binary)),
        }
    ys, xs = np.nonzero(valid)
    return {
        "depth": float(np.median(z[valid])),
        "u": float(xs.mean()),
        "v": float(ys.mean()),
        "used_pixels": count,
        "mask_pixels": int(np.count_nonzero(binary)),
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
    """Unproject robust mask depth with this frame's K and camera pose."""

    stats = robust_mask_depth(
        depth,
        mask,
        confidence=confidence,
        confidence_floor=confidence_floor,
    )
    if stats["depth"] is None:
        return None
    depth_u = float(stats["u"])
    depth_v = float(stats["v"])
    if source_size_px is not None:
        src_w, src_h = source_size_px
        depth_h, depth_w = np.asarray(depth).shape
        k = {
            "fx_px": float(intrinsics["fx_px"]) * (depth_w / src_w),
            "fy_px": float(intrinsics["fy_px"]) * (depth_h / src_h),
            "cx_px": float(intrinsics["cx_px"]) * (depth_w / src_w),
            "cy_px": float(intrinsics["cy_px"]) * (depth_h / src_h),
            "skew_px": float(intrinsics.get("skew_px", 0.0)) * (depth_w / src_w),
        }
        source_u = depth_u * (src_w / depth_w)
        source_v = depth_v * (src_h / depth_h)
    else:
        k = {
            "fx_px": float(intrinsics["fx_px"]),
            "fy_px": float(intrinsics["fy_px"]),
            "cx_px": float(intrinsics["cx_px"]),
            "cy_px": float(intrinsics["cy_px"]),
            "skew_px": float(intrinsics.get("skew_px", 0.0)),
        }
        source_u = depth_u
        source_v = depth_v
    camera_point = unproject_depth_point(
        depth_u,
        depth_v,
        float(stats["depth"]),
        k["fx_px"],
        k["fy_px"],
        k["cx_px"],
        k["cy_px"],
        k["skew_px"],
    )
    world = transform_point(T_world_camera, camera_point)
    return {
        "root": [float(world[0]), float(world[1]), float(world[2])],
        "pixel": [source_u, source_v],
        "depth": float(stats["depth"]),
        "used_pixels": int(stats["used_pixels"]),
        "mask_pixels": int(stats["mask_pixels"]),
        "visible": True,
    }


def load_da3_depth_artifact(path) -> dict[str, np.ndarray]:
    """Load the per-frame depth bundle written by the DA3 adapter."""

    data = np.load(path)
    required = ("depth", "processed_intrinsics", "processed_size", "source_size")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ContractError(f"DA3 depth artifact is missing {missing}")
    return {name: data[name] for name in data.files}
