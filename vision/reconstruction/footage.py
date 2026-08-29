"""Inspect local clips for P5R eligibility. Do not invent metric scale."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .iris import iris_benchmark_review


REQUESTED_CLIP: dict[str, Any] = {
    "duration_s": {"min": 3, "max": 8},
    "subject": "ball or weight on a string or rope",
    "must_show": ["anchor", "object"],
    "motion": "nontrivial spatial motion. Planar pendulums are allowed.",
    "measurement": (
        "tape-measured tether length in meters, or another clearly measured "
        "scene distance in meters"
    ),
    "quality": "limited blur and occlusion",
    "do_not": [
        "guess an object diameter",
        "use cinematic footage as the correctness baseline",
        "mark scale metric_measured without an external measurement",
    ],
}


def inspect_local_footage(root: Path) -> dict[str, Any]:
    """Classify on-disk clips. None are eligible without a known length in meters."""

    candidates = [
        {
            "id": "bounce",
            "path": root / "samples" / "bounce.mp4",
            "kind": "recorded",
            "tethered": False,
            "known_length_m": None,
            "reason": "Mixkit tennis bounce. Not a tethered object and no measured dimension.",
        },
        {
            "id": "pendulum",
            "path": root / "samples" / "recorded" / "pendulum.mp4",
            "kind": "recorded",
            "tethered": True,
            "known_length_m": None,
            "reason": (
                "Physical pendulum exists, but there is no tape-measured tether "
                "length in meters. Image-space radius is not a metric measurement."
            ),
        },
        {
            "id": "cinematic_swing",
            "path": root / "samples" / "cinematic" / "spiderman_swing.mp4",
            "kind": "cinematic",
            "tethered": False,
            "known_length_m": None,
            "reason": "Cinematic stress footage. Not a physical validation clip.",
        },
        {
            "id": "generated_diagonal",
            "path": root / "samples" / "generated_diagonal.mp4",
            "kind": "rendered",
            "tethered": False,
            "known_length_m": None,
            "reason": "Synthetic 2D bounce. Not real footage.",
        },
        {
            "id": "generated_drop",
            "path": root / "samples" / "generated_drop.mp4",
            "kind": "rendered",
            "tethered": False,
            "known_length_m": None,
            "reason": "Synthetic 2D drop. Not real footage.",
        },
    ]
    reviewed = []
    for item in candidates:
        present = item["path"].is_file()
        eligible = bool(
            present
            and item["tethered"]
            and item["known_length_m"] is not None
            and item["kind"] in {"recorded", "external_dataset"}
        )
        reviewed.append(
            {
                "id": item["id"],
                "path": str(item["path"]),
                "present": present,
                "kind": item["kind"],
                "tethered": item["tethered"],
                "known_length_m": item["known_length_m"],
                "eligible": eligible,
                "reason": item["reason"]
                if present
                else f"{item['path'].name} is not on disk",
            }
        )
    iris = iris_benchmark_review(root)
    reviewed.append(iris)
    eligible = [item for item in reviewed if item["eligible"]]
    return {
        "status": "READY" if eligible else "AWAITING_FOOTAGE",
        "eligible": eligible,
        "rejected": [item for item in reviewed if not item["eligible"]],
        "requested_clip": REQUESTED_CLIP,
    }
