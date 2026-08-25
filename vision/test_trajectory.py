"""CPU-only checks for mask centroid extraction."""

from __future__ import annotations

import numpy as np

from trajectory import centroid_from_mask, geometry_from_mask


def main() -> int:
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:8, 10:16] = True
    geometry = geometry_from_mask(mask)
    assert geometry is not None
    x, y = centroid_from_mask(mask)
    assert abs(x - 12.5) < 1e-9
    assert abs(y - 5.5) < 1e-9
    assert geometry["bbox_w"] == 6.0
    assert geometry["bbox_h"] == 4.0
    assert geometry_from_mask(np.zeros((8, 8), dtype=bool)) is None
    print("trajectory geometry: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
