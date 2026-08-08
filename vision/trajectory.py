"""Mask to trajectory helpers.

Checkpoint 2 fills these in from SAM 2 masks. Keep raw observations
reproducible. Do not invent coordinates from text.
"""

from __future__ import annotations


def centroid_from_mask(mask) -> tuple[float, float]:
    raise NotImplementedError("mask centroid extraction is Checkpoint 2")
