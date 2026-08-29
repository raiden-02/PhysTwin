"""Small mask helpers shared by reconstruction lift code.

This avoids importing the V1 `vision.trajectory` module from reconstruction
packages that must stay importable without the tracker path on sys.path.
"""

from __future__ import annotations

import numpy as np

MIN_MASK_PIXELS = 16


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape {arr.shape}")
    return arr.astype(bool)
