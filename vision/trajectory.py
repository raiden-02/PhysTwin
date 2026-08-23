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


def pair_target_and_anchor(
    target_observations: list[dict[str, Any]],
    anchor_observations: list[dict[str, Any]],
    anchor_click: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Pair rows by frame and keep the clicked point's offset from the anchor mask."""
    if not target_observations:
        raise ValueError("target track is empty")
    if not anchor_observations:
        raise ValueError("anchor track is empty")

    anchors_by_frame = {int(row["frame"]): row for row in anchor_observations}
    first_anchor = anchors_by_frame.get(0)
    if first_anchor is None:
        raise ValueError("anchor mask is empty on frame 0")
    offset_x = float(anchor_click[0]) - float(first_anchor["x"])
    offset_y = float(anchor_click[1]) - float(first_anchor["y"])

    paired_targets: list[dict[str, Any]] = []
    paired_anchors: list[dict[str, Any]] = []
    for target in target_observations:
        anchor = anchors_by_frame.get(int(target["frame"]))
        if anchor is None:
            continue
        adjusted = dict(anchor)
        adjusted["x"] = float(anchor["x"]) + offset_x
        adjusted["y"] = float(anchor["y"]) + offset_y
        paired_targets.append(target)
        paired_anchors.append(adjusted)

    coverage = len(paired_targets) / len(target_observations)
    return paired_targets, paired_anchors, coverage
