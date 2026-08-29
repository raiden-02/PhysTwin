"""Fit the supported P5 tether profile to metric 3D motion evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics3d.inverse_fit import (  # noqa: E402
    DEFAULT_SEED,
    FIXED_LENGTH_PROFILE,
    FREE_FALL_PROFILE,
    PROFILE,
    apply_free_fall_parameters,
    apply_tether_parameters,
    blocked_fit_report,
    fit_free_fall_scene,
    fit_tether_scene,
    select_real_fit_profile,
)
from physics3d.motion_observation import (  # noqa: E402
    FitInputBlocked,
    entity_observation_blockers,
    motion_observation_from_entities,
    motion_observation_from_rollout,
    motion_observation_from_scene_observation,
    scene_observation_blockers,
)
from vision.reconstruction.entities import ENTITIES_EXTENSION  # noqa: E402
from vision.reconstruction.humans import HUMANS_EXTENSION  # noqa: E402
from physics3d.newton_runtime import simulate_physical_scene  # noqa: E402
from vision.reconstruction.contracts import load_contract  # noqa: E402


TRUTH_PARAMETERS = {
    "rest_length_m": 2.08,
    "initial_tangent_velocity_u_m_s": 0.31,
    "initial_tangent_velocity_v_m_s": -0.23,
}
FREE_FALL_TRUTH_PARAMETERS = {
    "gravity_magnitude_m_s2": 9.80665,
    "initial_velocity_y_m_s": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_scene", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture",
        action="store_true",
        help="generate deterministic synthetic evidence from known parameters",
    )
    source.add_argument(
        "--motion-observation",
        type=Path,
        help="project-owned phystwin.physical_motion_observation JSON",
    )
    source.add_argument(
        "--scene-observation",
        type=Path,
        help="metric-measured SceneObservation with humans.v1 evidence",
    )
    parser.add_argument("--person-id")
    parser.add_argument("--entity-id")
    parser.add_argument(
        "--profile",
        choices=(PROFILE, FIXED_LENGTH_PROFILE, FREE_FALL_PROFILE),
        default=None,
        help="omit to choose from calibration: fixed length when tether length set scale",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--coordinate-iterations", type=int, default=12)
    parser.add_argument(
        "--no-repeat-check",
        action="store_true",
        help="skip the final P4 repeated-rollout check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = dict(load_contract(args.physical_scene.resolve()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.fixture:
        if args.profile == FREE_FALL_PROFILE:
            truth_scene = apply_free_fall_parameters(template, FREE_FALL_TRUTH_PARAMETERS)
            truth_params = FREE_FALL_TRUTH_PARAMETERS
        else:
            truth_scene = apply_tether_parameters(template, TRUTH_PARAMETERS)
            truth_params = TRUTH_PARAMETERS
        truth_rollout = simulate_physical_scene(truth_scene, repeat_check=False)
        motion = motion_observation_from_rollout(
            truth_rollout,
            stride=2,
            truth_parameters=truth_params,
        )
        _write_json(output / "truth_physical_scene.json", truth_scene)
        _write_json(output / "target_motion_observation.json", motion)
    elif args.motion_observation:
        motion = dict(load_contract(args.motion_observation.resolve()))
    else:
        scene_observation = dict(load_contract(args.scene_observation.resolve()))
        use_entities = _prefer_entities(scene_observation, args.entity_id, args.person_id)
        if use_entities:
            blockers = entity_observation_blockers(
                scene_observation,
                template,
                entity_id=args.entity_id,
            )
        else:
            blockers = scene_observation_blockers(
                scene_observation,
                template,
                person_id=args.person_id,
            )
        calibration = scene_observation.get("provenance", {}).get("metric_calibration")
        profile = args.profile or select_real_fit_profile(
            calibration if isinstance(calibration, dict) else None
        )
        if blockers:
            report = blocked_fit_report(
                template,
                scene_observation,
                blockers,
                profile=profile,
            )
            _write_json(output / "inverse_physics_fit.json", report)
            print(json.dumps({"status": "BLOCKED_INPUT", "blockers": blockers}, indent=2))
            return 2
        try:
            if use_entities:
                motion = motion_observation_from_entities(
                    scene_observation,
                    template,
                    entity_id=args.entity_id,
                )
            else:
                motion = motion_observation_from_scene_observation(
                    scene_observation,
                    template,
                    person_id=args.person_id,
                )
        except FitInputBlocked as error:
            report = blocked_fit_report(
                template,
                scene_observation,
                error.blockers,
                profile=profile,
            )
            _write_json(output / "inverse_physics_fit.json", report)
            print(
                json.dumps(
                    {"status": "BLOCKED_INPUT", "blockers": list(error.blockers)},
                    indent=2,
                )
            )
            return 2
        _write_json(output / "target_motion_observation.json", motion)

    profile = args.profile
    if profile is None and not args.fixture:
        calibration = motion.get("provenance", {}).get("calibration")
        profile = select_real_fit_profile(
            calibration if isinstance(calibration, dict) else None
        )
    if profile is None:
        profile = PROFILE

    fit_fn = fit_free_fall_scene if profile == FREE_FALL_PROFILE else fit_tether_scene
    result = fit_fn(
        template,
        motion,
        output_dir=output,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        coordinate_iterations=args.coordinate_iterations,
        repeat_check=not args.no_repeat_check,
        profile=profile,
    )
    fit = result["fit"]
    print(
        json.dumps(
            {
                "status": fit["status"],
                "fit_id": fit["fit_id"],
                "output": str(output),
                "objective": fit["objective"],
                "parameters": fit["parameters"],
                "optimizer": fit["optimizer"],
                "execution": fit["execution"],
                "validation": fit["validation"],
            },
            indent=2,
        )
    )
    return 0


def _prefer_entities(
    observation: dict,
    entity_id: str | None,
    person_id: str | None,
) -> bool:
    extensions = observation.get("extensions") or {}
    has_entities = ENTITIES_EXTENSION in extensions
    if entity_id:
        return True
    if person_id:
        return False
    return has_entities


def _write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
