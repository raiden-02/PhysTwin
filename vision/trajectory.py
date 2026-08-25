"""Mask to trajectory helpers.

Centroids come from SAM 2 masks, not from guessed coordinates.
Empty or tiny masks are rejected so the C++ core never sees invented points.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MIN_MASK_PIXELS = 16


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape {arr.shape}")
    return arr.astype(bool)


def geometry_from_mask(mask: np.ndarray) -> dict[str, float] | None:
    """Return centroid, bbox, and equivalent radius, or None if the mask is empty."""
    binary = _as_bool_mask(mask)
    ys, xs = np.nonzero(binary)
    if xs.size < MIN_MASK_PIXELS:
        return None
    area = float(xs.size)
    return {
        "x": float(xs.mean()),
        "y": float(ys.mean()),
        "bbox_x": float(xs.min()),
        "bbox_y": float(ys.min()),
        "bbox_w": float(xs.max() - xs.min() + 1),
        "bbox_h": float(ys.max() - ys.min() + 1),
        "radius": float(np.sqrt(area / np.pi)),
        "area": area,
    }


def centroid_from_mask(mask: np.ndarray) -> tuple[float, float]:
    geometry = geometry_from_mask(mask)
    if geometry is None:
        raise ValueError("empty or tiny mask")
    return geometry["x"], geometry["y"]


def observation_from_geometry(
    frame: int,
    fps: float,
    geometry: dict[str, float],
    confidence: float | None,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "frame": int(frame),
        "t": float(frame) / float(fps),
        "x": geometry["x"],
        "y": geometry["y"],
        "bbox_x": geometry["bbox_x"],
        "bbox_y": geometry["bbox_y"],
        "bbox_w": geometry["bbox_w"],
        "bbox_h": geometry["bbox_h"],
        "radius": geometry["radius"],
    }
    if confidence is not None:
        obs["confidence"] = float(confidence)
    return obs
