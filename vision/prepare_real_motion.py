#!/usr/bin/env python3
"""Lift a recorded clip into entity tracks and a P5R motion observation.

Without a tape-measured scene distance this writes relative entity tracks and
BLOCKED_INPUT. It does not invent metric_measured scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.calibration import reject_direct_metric_scale  # noqa: E402
from vision.reconstruction.footage import inspect_local_footage  # noqa: E402


TEMPLATE = (
    ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_tether_fit_template.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="classify local clips and print the requested measurement. No GPU.",
    )
    parser.add_argument(
        "--iris",
        action="store_true",
        help="use the IRIS pendulum_45/01 external-dataset benchmark",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--target-xy", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--anchor-xy", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--known-distance-m", type=float)
    parser.add_argument(
        "--measurement-source",
        help="name of the external measurement, for example 'tape measure 2026-08-29'",
    )
    parser.add_argument("--from-id", default="target")
    parser.add_argument("--to-id", default="anchor")
    parser.add_argument(
        "--from-physical-point",
        choices=("body_center", "attachment", "anchor"),
        default="body_center",
        help="physical point named by --from-id. Not an arbitrary SAM-mask centroid.",
    )
    parser.add_argument(
        "--to-physical-point",
        choices=("body_center", "attachment", "anchor"),
        default="anchor",
        help="physical point named by --to-id",
    )
    parser.add_argument(
        "--up-mode",
        choices=("level_camera", "supplied_vector"),
        default="level_camera",
        help="level_camera is assumed, not measured gravity",
    )
    parser.add_argument(
        "--up-source",
        choices=("assumed", "measured"),
        default="assumed",
    )
    parser.add_argument(
        "--physical-up",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="physical-up vector in observation coordinates. Implies supplied_vector.",
    )
    parser.add_argument(
        "--circular-with",
        choices=("rest_length_m", "none"),
        default="rest_length_m",
        help="rest_length_m when the measured distance is the tether itself",
    )
    parser.add_argument("--source-id", default="video0")
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "physics3d" / "p5r-real-fit")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inspect:
        payload = inspect_local_footage(ROOT)
        print(json.dumps(payload, indent=2))
        return 0 if payload["status"] == "READY" else 2

    iris_benchmark = None
    iris_calibration_extra: dict[str, Any] = {}
    if args.iris:
        from vision.reconstruction.iris import (
            calibration_provenance,
            load_iris_pendulum_benchmark,
        )

        iris_benchmark = load_iris_pendulum_benchmark(ROOT)
        iris_calibration_extra = calibration_provenance(iris_benchmark)
        if args.video is None:
            args.video = iris_benchmark["video"]
        if args.target_xy is None:
            args.target_xy = iris_benchmark["target_xy"]
        if args.anchor_xy is None:
            args.anchor_xy = iris_benchmark["anchor_xy"]
        if args.known_distance_m is None:
            args.known_distance_m = iris_benchmark["rope_length_m"]
        if not args.measurement_source:
            args.measurement_source = iris_benchmark["measurement_source"]
        args.from_physical_point = iris_benchmark["from_physical_point"]
        args.to_physical_point = iris_benchmark["to_physical_point"]
        args.circular_with = iris_benchmark["circular_with"]
        args.up_mode = iris_benchmark["up_mode"]
        args.up_source = iris_benchmark["up_source"]
        if args.start_s is None:
            args.start_s = iris_benchmark["start_s"]
        if args.duration_s is None:
            args.duration_s = iris_benchmark["duration_s"]
        if args.max_frames is None:
            args.max_frames = iris_benchmark["max_frames"]

    if args.start_s is None:
        args.start_s = 0.0
    if args.duration_s is None:
        args.duration_s = 4.0
    if args.max_frames is None:
        args.max_frames = 16

    if args.video is None:
        payload = inspect_local_footage(ROOT)
        print(json.dumps(payload, indent=2))
        return 0 if payload["status"] == "READY" else 2

    from physics3d.motion_observation import (
        FitInputBlocked,
        entity_observation_blockers,
        motion_observation_from_entities,
    )
    from vision.reconstruction.adapter import (
        ReconstructionRequest,
        VideoInput,
        reconstruction_cache_key,
    )
    from vision.reconstruction.cache import (
        cache_entry,
        is_complete,
        load_cached_observation,
        publish_observation,
        sha256_file,
    )
    from vision.reconstruction.calibration import (
        apply_measured_scale,
        build_known_distance_calibration,
        select_real_fit_profile,
    )
    from vision.reconstruction.contracts import load_contract
    from vision.reconstruction.entities import ENTITIES_EXTENSION
    from vision.reconstruction.real_motion import (
        lift_entities_from_masks,
        stamp_observation_alignment,
    )

    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    if args.target_xy is None or args.anchor_xy is None:
        raise SystemExit("P5R prepare requires --target-xy and --anchor-xy")
    if args.known_distance_m is not None:
        if args.known_distance_m <= 0.0:
            raise SystemExit("known-distance-m must be > 0")
        if not args.measurement_source:
            raise SystemExit("known-distance-m requires --measurement-source")
        reject_direct_metric_scale("metric_measured", args.measurement_source)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = dict(load_contract(args.template.resolve()))
    observation, cache_dir = _load_or_reconstruct(
        video,
        source_id=args.source_id,
        start_s=args.start_s,
        duration_s=args.duration_s,
        max_frames=args.max_frames,
        force=args.force,
    )
    depth_path = _depth_artifact_path(observation, cache_dir)
    if depth_path is None:
        blockers = [
            "SceneObservation has no da3_depth artifact. Re-run P1 reconstruction "
            "so per-frame depth is persisted."
        ]
        return _write_blocked(output, template, observation, blockers)

    from vision.reconstruction.track_entities import track_selected_frames

    source_frames = [
        int(next(iter(sample["source_frames"].values())))
        for sample in observation["timeline"]["samples"]
    ]
    tracked = track_selected_frames(
        video,
        source_frames,
        target=(args.target_xy[0], args.target_xy[1]),
        anchor=(args.anchor_xy[0], args.anchor_xy[1]),
    )
    target_masks = {}
    anchor_masks = {}
    for local_index, sample in tracked["masks"].items():
        if local_index >= len(source_frames):
            continue
        if sample["target"] is not None:
            target_masks[local_index] = sample["target"]
        if sample["anchor"] is not None:
            anchor_masks[local_index] = sample["anchor"]
    lifted = lift_entities_from_masks(
        observation,
        target_masks=target_masks,
        anchor_masks=anchor_masks,
        depth_artifact=depth_path,
    )
    observation = lifted["observation"]
    _write_json(output / "scene_observation_entities.json", observation)
    _write_json(
        output / "lift_counts.json",
        {
            "accepted_lifts": lifted["accepted_lifts"],
            "rejected_lifts": lifted["rejected_lifts"],
        },
    )

    if args.known_distance_m is None:
        blockers = [
            "No external measured length was provided. Entity tracks stay in "
            "relative reconstruction units. Do not mark scale metric_measured."
        ]
        return _write_blocked(output, template, observation, blockers)

    circular = None if args.circular_with == "none" else args.circular_with
    calibration = build_known_distance_calibration(
        calibration_id=f"p5r-{observation['observation_id']}",
        entities=observation["extensions"][ENTITIES_EXTENSION],
        from_id=args.from_id,
        to_id=args.to_id,
        measured_length_m=args.known_distance_m,
        measurement_source=args.measurement_source,
        circular_with_fit_parameter=circular,
        from_physical_point=args.from_physical_point,
        to_physical_point=args.to_physical_point,
        provenance={
            "video": video.name,
            **(iris_calibration_extra if iris_benchmark is not None else {}),
        },
    )
    observation = apply_measured_scale(observation, calibration)
    if iris_benchmark is not None:
        provenance = dict(observation.get("provenance") or {})
        provenance["evidence_kind"] = iris_benchmark["evidence_kind"]
        provenance["dataset"] = iris_benchmark["dataset"]
        provenance["iris"] = {
            "repo_id": iris_benchmark["repo_id"],
            "source_url": iris_benchmark["source_url"],
            "relative_video": iris_benchmark["relative_video"],
            "class_key": iris_benchmark["class_key"],
            "setting_key": iris_benchmark["setting_key"],
            "parameters": iris_benchmark["parameters"],
            "held_fixed_parameter": iris_benchmark["held_fixed_parameter"],
        }
        observation["provenance"] = provenance
    if circular == "rest_length_m":
        template["model"]["constraints"][0]["rest_length_m"] = float(args.known_distance_m)
    physical_up = {
        "mode": "supplied_vector" if args.physical_up is not None else args.up_mode,
        "source": args.up_source,
    }
    if args.physical_up is not None:
        physical_up["vector_observation"] = list(args.physical_up)
    template = stamp_observation_alignment(
        template,
        observation,
        entity_id=args.from_id,
        anchor_id=args.to_id,
        physical_up=physical_up,
    )
    _write_json(output / "metric_calibration.json", calibration)
    _write_json(output / "scene_observation_metric.json", observation)
    _write_json(output / "aligned_physical_scene_template.json", template)

    blockers = entity_observation_blockers(
        observation,
        template,
        entity_id=args.from_id,
    )
    if blockers:
        return _write_blocked(
            output,
            template,
            observation,
            blockers,
            profile=select_real_fit_profile(calibration),
        )
    try:
        motion = motion_observation_from_entities(
            observation,
            template,
            entity_id=args.from_id,
        )
    except FitInputBlocked as error:
        return _write_blocked(
            output,
            template,
            observation,
            list(error.blockers),
            profile=select_real_fit_profile(calibration),
        )
    if iris_benchmark is not None:
        motion_prov = dict(motion.get("provenance") or {})
        motion_prov["evidence_kind"] = iris_benchmark["evidence_kind"]
        motion_prov["dataset"] = iris_benchmark["dataset"]
        motion["provenance"] = motion_prov
    _write_json(output / "target_motion_observation.json", motion)
    if iris_benchmark is not None:
        from vision.reconstruction.cache import sha256_file
        from vision.reconstruction.contracts import canonical_json_bytes
        import hashlib

        _write_json(
            output / "iris_p5r_evidence.json",
            {
                "evidence_kind": iris_benchmark["evidence_kind"],
                "dataset": iris_benchmark["dataset"],
                "repo_id": iris_benchmark["repo_id"],
                "source_url": iris_benchmark["source_url"],
                "relative_video": iris_benchmark["relative_video"],
                "video_path": str(video),
                "video_sha256": sha256_file(video),
                "iris_parameters": iris_benchmark["parameters"],
                "metric_value_m": iris_benchmark["rope_length_m"],
                "metric_name": "rope_length",
                "held_fixed_parameter": iris_benchmark["held_fixed_parameter"],
                "from_physical_point": iris_benchmark["from_physical_point"],
                "to_physical_point": iris_benchmark["to_physical_point"],
                "physical_up": {
                    "mode": iris_benchmark["up_mode"],
                    "source": iris_benchmark["up_source"],
                },
                "scene_observation_sha256": hashlib.sha256(
                    canonical_json_bytes(observation)
                ).hexdigest(),
                "physical_motion_observation_sha256": hashlib.sha256(
                    canonical_json_bytes(motion)
                ).hexdigest(),
                "accepted_lifts": lifted["accepted_lifts"],
                "rejected_lifts": lifted["rejected_lifts"],
            },
        )
    print(
        json.dumps(
            {
                "status": "MOTION_READY",
                "accepted_lifts": lifted["accepted_lifts"],
                "rejected_lifts": lifted["rejected_lifts"],
                "profile": select_real_fit_profile(calibration),
                "output": str(output),
                "next": (
                    "Run physics3d.fit_physical_scene on aligned_physical_scene_template.json "
                    "with --motion-observation target_motion_observation.json"
                ),
            },
            indent=2,
        )
    )
    return 0


def _load_or_reconstruct(
    video: Path,
    *,
    source_id: str,
    start_s: float,
    duration_s: float,
    max_frames: int,
    force: bool,
) -> tuple[dict[str, Any], Path]:
    from vision.reconstruction.adapter import (
        ReconstructionRequest,
        VideoInput,
        reconstruction_cache_key,
    )
    from vision.reconstruction.cache import (
        cache_entry,
        is_complete,
        load_cached_observation,
        publish_observation,
        sha256_file,
    )
    from vision.reconstruction.da3 import (
        Da3ReconstructionAdapter,
        make_descriptor,
        normalize_options,
        resolve_weights_sha256,
    )

    options = normalize_options(
        {
            "start_s": start_s,
            "duration_s": duration_s,
            "max_frames": max_frames,
        }
    )
    weights = resolve_weights_sha256()
    descriptor = make_descriptor(weights)
    request = ReconstructionRequest(
        inputs=(VideoInput(source_id, video, sha256_file(video)),),
        options=options,
    )
    key = reconstruction_cache_key(descriptor, request)
    entry = cache_entry(ROOT, key)
    if is_complete(entry) and not force:
        return dict(load_cached_observation(entry)), entry
    adapter = Da3ReconstructionAdapter(weights)
    observation = publish_observation(
        entry,
        lambda work_dir: adapter.reconstruct(request, work_dir).observation,
    )
    return dict(observation), entry


def _depth_artifact_path(observation: dict[str, Any], cache_dir: Path) -> Path | None:
    for artifact in observation.get("artifacts", []):
        if artifact.get("id") != "da3_depth":
            continue
        path = cache_dir / artifact["uri"]
        if path.is_file():
            return path
    fallback = cache_dir / "artifacts" / "da3_depth.npz"
    return fallback if fallback.is_file() else None


def _write_blocked(
    output: Path,
    template: dict[str, Any],
    observation: dict[str, Any],
    blockers: list[str],
    *,
    profile: str | None = None,
) -> int:
    import hashlib

    from vision.reconstruction.calibration import FIXED_LENGTH_PROFILE, LENGTH_FIT_PROFILE
    from vision.reconstruction.contracts import (
        canonical_json_bytes,
        validate_inverse_physics_fit,
    )

    chosen = profile or LENGTH_FIT_PROFILE
    hold_length = chosen == FIXED_LENGTH_PROFILE
    template_hash = hashlib.sha256(canonical_json_bytes(template)).hexdigest()
    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "template": template_hash,
                "scene_observation": observation_hash,
                "profile": chosen,
            }
        )
    ).hexdigest()
    parameters = [
        {
            "id": "rest_length_m",
            "unit": "meter",
            "lower_bound": 1.6,
            "upper_bound": 2.4,
            "initial": 1.78,
            "fitted": None,
            "truth": None,
            "held_fixed": hold_length,
        },
        {
            "id": "initial_tangent_velocity_u_m_s",
            "unit": "meter_per_second",
            "lower_bound": -0.6,
            "upper_bound": 0.6,
            "initial": -0.12,
            "fitted": None,
            "truth": None,
            "held_fixed": False,
        },
        {
            "id": "initial_tangent_velocity_v_m_s",
            "unit": "meter_per_second",
            "lower_bound": -0.6,
            "upper_bound": 0.6,
            "initial": 0.14,
            "fitted": None,
            "truth": None,
            "held_fixed": False,
        },
    ]
    report = {
        "schema": "phystwin.inverse_physics_fit",
        "version": 1,
        "fit_id": f"blocked-{identity[:12]}",
        "status": "BLOCKED_INPUT",
        "source": {
            "template_physical_scene": {
                "id": template["scene_id"],
                "sha256": template_hash,
            },
            "motion_observation": None,
            "scene_observation": {
                "id": observation["observation_id"],
                "sha256": observation_hash,
            },
        },
        "profile": chosen,
        "objective": {
            "type": "weighted_position_mse_3d",
            "sample_count": 0,
            "mse_m2": None,
            "rmse_m": None,
            "trajectory_extent_m": None,
            "normalized_rmse": None,
            "initial_mse_m2": None,
            "improvement_ratio": None,
        },
        "parameters": parameters,
        "optimizer": {
            "method": "bounded_differential_evolution_with_coordinate_refinement",
            "seed": 0x50545935,
            "population_size": 0,
            "generations": 0,
            "coordinate_iterations": 0,
            "objective_evaluations": 0,
        },
        "outputs": {
            "fitted_physical_scene": None,
            "simulated_world_state": None,
        },
        "execution": {
            "wall_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
        },
        "validation": {
            "passed": False,
            "rollout_valid": False,
            "execution_valid": False,
            "quality": {
                "status": "unassessed",
                "rmse_m": None,
                "normalized_rmse": None,
            },
            "synthetic_recovery": {
                "performed": False,
                "within_tolerance": None,
                "max_normalized_parameter_error": None,
            },
        },
        "blockers": list(blockers),
        "warnings": [],
        "failures": [],
    }
    validate_inverse_physics_fit(report)
    _write_json(output / "inverse_physics_fit.json", report)
    print(json.dumps({"status": "BLOCKED_INPUT", "blockers": blockers, "profile": chosen}, indent=2))
    return 2


def _write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
