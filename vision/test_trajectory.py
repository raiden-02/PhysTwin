"""CPU-only checks for mask centroid extraction."""

from __future__ import annotations

import numpy as np

from trajectory import (
    centroid_from_mask,
    geometry_from_mask,
    pair_target_and_anchor,
)


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

    targets = [
        {"frame": 0, "t": 0.0, "x": 10.0, "y": 20.0},
        {"frame": 1, "t": 0.1, "x": 11.0, "y": 21.0},
        {"frame": 2, "t": 0.2, "x": 12.0, "y": 22.0},
    ]
    anchors = [
        {"frame": 0, "t": 0.0, "x": 100.0, "y": 200.0},
        {"frame": 2, "t": 0.2, "x": 104.0, "y": 206.0},
    ]
    paired_targets, paired_anchors, coverage = pair_target_and_anchor(
        targets, anchors, (103.0, 198.0)
    )
    assert [row["frame"] for row in paired_targets] == [0, 2]
    assert [row["frame"] for row in paired_anchors] == [0, 2]
    assert paired_anchors[0]["x"] == 103.0
    assert paired_anchors[0]["y"] == 198.0
    assert paired_anchors[1]["x"] == 107.0
    assert paired_anchors[1]["y"] == 204.0
    assert abs(coverage - 2.0 / 3.0) < 1e-12

    try:
        pair_target_and_anchor(targets, anchors[1:], (103.0, 198.0))
    except ValueError as error:
        assert "frame 0" in str(error)
    else:
        raise AssertionError("missing frame-0 anchor must be rejected")
    print("trajectory geometry: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
