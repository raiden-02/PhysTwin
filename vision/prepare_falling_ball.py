#!/usr/bin/env python3
"""Lift one IRIS falling-ball clip with known-radius sphere geometry.

DA3 supplies camera intrinsics and a static-camera check. Moving-object depth
comes from the measured ball radius, not DA3 depth. IRIS gravity is not read
for fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TEMPLATE = (
    ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_free_fall.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "physics3d" / "p5r-falling-ball",
    )
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from vision.reconstruction.cache import sha256_file
    from vision.reconstruction.contracts import canonical_json_bytes, load_contract
    from vision.reconstruction.falling_ball import (
        lift_and_filter_frames,
        motion_observation_from_sphere_track,
        stamp_free_fall_template,
        static_camera_report,
        summarize_intrinsics,
    )
    from vision.reconstruction.iris_falling import (
        assert_no_iris_gravity_truth,
        calibration_provenance,
        load_iris_falling_ball_input,
    )
    from vision.reconstruction.lift import sample_intrinsics_for_frame
    from vision.prepare_real_motion import _load_or_reconstruct
    from vision.reconstruction.track_entities import track_selected_frames

    clip = load_iris_falling_ball_input(ROOT)
    video = clip["video"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = dict(load_contract(args.template.resolve()))
    observation, _cache_dir = _load_or_reconstruct(
        video,
        source_id="iris-falling-ball-big-01",
        start_s=clip["start_s"],
        duration_s=clip["duration_s"],
        max_frames=clip["max_frames"],
        force=args.force,
    )
    camera = observation["cameras"][0]
    da3 = observation.get("extensions", {}).get("phystwin.da3.v1")
    camera_check = static_camera_report(camera)
    source_frames = []
    timestamps = []
    for sample in observation["timeline"]["samples"]:
        source_frames.append(int(next(iter(sample["source_frames"].values()))))
        timestamps.append(float(sample["timestamp_s"]))
    tracked = track_selected_frames(
        video,
        source_frames,
        target=(clip["target_xy"][0], clip["target_xy"][1]),
        anchor=None,
    )
    frame_inputs = []
    for local_index, sample in enumerate(observation["timeline"]["samples"]):
        mask_pack = tracked["masks"].get(local_index)
        mask = None if mask_pack is None else mask_pack.get("target")
        if mask is None:
            continue
        frame_inputs.append(
            {
                "sample_index": int(sample["sample_index"]),
                "source_frame": source_frames[local_index],
                "timestamp_s": timestamps[local_index],
                "mask": mask,
                "intrinsics": sample_intrinsics_for_frame(camera, local_index, da3),
            }
        )
    per_frame_k = [item["intrinsics"] for item in frame_inputs]
    lifted = lift_and_filter_frames(
        frame_inputs,
        radius_m=clip["ball_radius_m"],
    )
    reconstruction = {
        "method": "sam2_mask_known_radius_sphere",
        "ball_radius_m": clip["ball_radius_m"],
        "intrinsics_policy": "per_frame_da3",
        "intrinsics_variation": summarize_intrinsics(per_frame_k),
        "static_camera": camera_check,
        "accepted_frames": len(lifted["accepted"]),
        "rejected_frames": len(lifted["rejected"]),
        "frames": [
            {key: value for key, value in item.items() if key != "mask"}
            for item in lifted["frames"]
        ],
        "warnings": [],
    }
    if not camera_check["accepted"]:
        reconstruction["warnings"].append(camera_check["reason"])
    reconstruction_bytes = (
        json.dumps(reconstruction, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    reconstruction_hash = hashlib.sha256(reconstruction_bytes).hexdigest()
    _write_bytes(output / "reconstruction_frames.json", reconstruction_bytes)

    if len(lifted["accepted"]) < 2:
        _write_json(
            output / "iris_p5r_falling_ball_evidence.json",
            {
                "status": "BLOCKED_INPUT",
                "blockers": ["fewer than two accepted metric sphere frames"],
                "reconstruction_sha256": reconstruction_hash,
                "accepted_frames": len(lifted["accepted"]),
                "rejected_frames": len(lifted["rejected"]),
            },
        )
        print(json.dumps({"status": "BLOCKED_INPUT", "accepted": len(lifted["accepted"])}, indent=2))
        return 2

    motion = motion_observation_from_sphere_track(
        lifted["accepted"],
        observation_id=f"iris-falling-ball-{observation['observation_id']}",
        source_id="iris-falling-ball-big-01",
        source_sha256=reconstruction_hash,
        provenance={
            **calibration_provenance(clip),
            "physical_up": {
                "mode": clip["up_mode"],
                "source": clip["up_source"],
            },
        },
    )
    aligned = stamp_free_fall_template(
        template,
        motion,
        radius_m=clip["ball_radius_m"],
    )
    _write_json(output / "target_motion_observation.json", motion)
    _write_json(output / "aligned_physical_scene_template.json", aligned)
    evidence = {
        "evidence_kind": clip["evidence_kind"],
        "dataset": clip["dataset"],
        "repo_id": clip["repo_id"],
        "source_url": clip["source_url"],
        "relative_video": clip["relative_video"],
        "video_path": str(video),
        "video_sha256": sha256_file(video),
        "ball_radius_m": clip["ball_radius_m"],
        "drop_height_m": clip["drop_height_m"],
        "reconstruction_method": "sam2_mask_known_radius_sphere",
        "gravity_used_during_fit": False,
        "static_camera": camera_check,
        "accepted_frames": len(lifted["accepted"]),
        "rejected_frames": len(lifted["rejected"]),
        "rejected_reasons": [
            {"sample_index": item["sample_index"], "reason": item["reason"]}
            for item in lifted["rejected"]
        ],
        "reconstruction_sha256": reconstruction_hash,
        "physical_motion_observation_sha256": hashlib.sha256(
            canonical_json_bytes(motion)
        ).hexdigest(),
        "aligned_template_sha256": hashlib.sha256(
            canonical_json_bytes(aligned)
        ).hexdigest(),
        "intrinsics_policy": "per_frame_da3",
        "intrinsics_variation": reconstruction["intrinsics_variation"],
    }
    assert_no_iris_gravity_truth(motion)
    assert_no_iris_gravity_truth(aligned)
    assert_no_iris_gravity_truth(evidence)
    _write_json(output / "iris_p5r_falling_ball_evidence.json", evidence)
    print(
        json.dumps(
            {
                "status": "MOTION_READY",
                "accepted_frames": len(lifted["accepted"]),
                "rejected_frames": len(lifted["rejected"]),
                "output": str(output),
                "next": (
                    "Run physics3d.fit_physical_scene on aligned_physical_scene_template.json "
                    "with --motion-observation target_motion_observation.json "
                    "--profile free_fall_gravity_v1"
                ),
            },
            indent=2,
        )
    )
    return 0


def _write_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _write_bytes(path, payload.encode("utf-8"))


def _write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
