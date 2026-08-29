"""Known-radius sphere reconstruction. Depth comes from silhouette geometry, not DA3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .trajectory_bridge import MIN_MASK_PIXELS, as_bool_mask


def apparent_radius_from_mask(mask: np.ndarray) -> dict[str, float | int | str | None]:
    """Measure the image-space ball from a SAM2 mask.

    Vertical motion blur inflates area-equivalent radius. Horizontal half-width
    is the silhouette radius used for depth.
    """

    binary = as_bool_mask(mask)
    area = int(np.count_nonzero(binary))
    if area < MIN_MASK_PIXELS:
        return {
            "accepted": False,
            "reason": f"mask has {area} pixels, need at least {MIN_MASK_PIXELS}",
            "area_px": area,
            "center_u": None,
            "center_v": None,
            "radius_area_px": None,
            "radius_horizontal_px": None,
            "radius_px": None,
        }
    ys, xs = np.nonzero(binary)
    center_u = float(np.mean(xs))
    center_v = float(np.mean(ys))
    radius_area = math.sqrt(area / math.pi)
    half_widths: list[float] = []
    equator_widths: list[float] = []
    band = max(1.5, 0.25 * radius_area)
    for y in np.unique(ys):
        row = xs[ys == y]
        if row.size < 2:
            continue
        half = 0.5 * float(row.max() - row.min())
        half_widths.append(half)
        if abs(float(y) - center_v) <= band:
            equator_widths.append(half)
    if len(half_widths) < 3:
        return {
            "accepted": False,
            "reason": "mask has too few rows to measure a horizontal radius",
            "area_px": area,
            "center_u": center_u,
            "center_v": center_v,
            "radius_area_px": radius_area,
            "radius_horizontal_px": None,
            "radius_px": None,
        }
    measured = equator_widths if len(equator_widths) >= 3 else half_widths
    radius_horizontal = float(np.percentile(np.asarray(measured, dtype=np.float64), 90))
    if radius_horizontal <= 1.0:
        return {
            "accepted": False,
            "reason": "horizontal radius is too small",
            "area_px": area,
            "center_u": center_u,
            "center_v": center_v,
            "radius_area_px": radius_area,
            "radius_horizontal_px": radius_horizontal,
            "radius_px": None,
        }
    return {
        "accepted": True,
        "reason": None,
        "area_px": area,
        "center_u": center_u,
        "center_v": center_v,
        "radius_area_px": radius_area,
        "radius_horizontal_px": radius_horizontal,
        "radius_px": radius_horizontal,
    }


def sphere_center_depth_m(radius_px: float, focal_px: float, radius_m: float) -> float:
    """On-axis pinhole silhouette depth of a sphere center.

    The silhouette cone is tangent to the sphere, so
    Z = sqrt((f R / r)^2 + R^2), not the crude f R / r.
    """

    if radius_px <= 0.0 or focal_px <= 0.0 or radius_m <= 0.0:
        raise ValueError("radius_px, focal_px, and radius_m must be > 0")
    crude = focal_px * radius_m / radius_px
    return math.sqrt(crude * crude + radius_m * radius_m)


def unproject_opencv(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: Mapping[str, float],
) -> tuple[float, float, float]:
    """Camera-frame XYZ in OpenCV convention: +X right, +Y down, +Z forward."""

    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    cx = float(intrinsics["cx_px"])
    cy = float(intrinsics["cy_px"])
    x = (u - cx) / fx * depth_m
    y = (v - cy) / fy * depth_m
    return (x, y, depth_m)


def opencv_to_first_camera_graphics(point: Sequence[float]) -> list[float]:
    """Match DA3's first-camera graphics basis: +Y up, +Z toward the camera."""

    return [float(point[0]), -float(point[1]), -float(point[2])]


def reconstruct_metric_ball(
    mask: np.ndarray,
    *,
    radius_m: float,
    intrinsics: Mapping[str, float],
) -> dict[str, Any]:
    """Lift one mask to metric first-camera-graphics XYZ using known radius."""

    measured = apparent_radius_from_mask(mask)
    if not measured["accepted"]:
        return {
            **measured,
            "depth_m": None,
            "position_camera_opencv_m": None,
            "position_m": None,
        }
    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    focal = 0.5 * (fx + fy)
    depth = sphere_center_depth_m(float(measured["radius_px"]), focal, radius_m)
    camera = unproject_opencv(
        float(measured["center_u"]),
        float(measured["center_v"]),
        depth,
        intrinsics,
    )
    world = opencv_to_first_camera_graphics(camera)
    return {
        **measured,
        "depth_m": depth,
        "position_camera_opencv_m": list(camera),
        "position_m": world,
        "focal_px": focal,
    }
