"""P5R CPU tests: lift, calibration, circular claims, and hash continuity."""

from __future__ import annotations

import copy
import hashlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics3d.motion_observation import (
    FitInputBlocked,
    entity_observation_blockers,
    motion_observation_from_entities,
)
from vision.reconstruction.calibration import (
    FIXED_LENGTH_PROFILE,
    LENGTH_FIT_PROFILE,
    apply_measured_scale,
    build_known_distance_calibration,
    refuse_circular_length_fit,
    reject_direct_metric_scale,
    select_real_fit_profile,
    validate_metric_calibration,
)
from vision.reconstruction.contracts import (
    FIRST_CAMERA_WORLD_FROM_OPENCV,
    ContractError,
    canonical_json_bytes,
    load_contract,
    require_motion_matches_scene_alignment,
    validate_inverse_physics_fit,
    validate_physical_motion_observation,
    validate_physical_scene,
)
from vision.reconstruction.entities import attach_entities, entities_payload, entity_payload
from vision.reconstruction.footage import inspect_local_footage
from vision.reconstruction.lift import (
    lift_mask_to_world,
    robust_3d_center,
    sample_intrinsics_for_frame,
    unproject_pixels,
)
from vision.reconstruction.real_motion import (
    stamp_observation_alignment,
    validate_physical_up,
)
from vision.reconstruction.transforms import transform_point, unproject_depth_point

LEVEL_CAMERA = {"mode": "level_camera", "source": "assumed"}
TETHER_POINTS = {"from_physical_point": "body_center", "to_physical_point": "anchor"}


EXAMPLES = ROOT / "contracts" / "3d" / "v1" / "examples"
IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]
K0 = {"fx_px": 100.0, "fy_px": 100.0, "cx_px": 4.5, "cy_px": 4.5, "skew_px": 0.0}


class LiftGeometryTest(unittest.TestCase):
    def test_pixel_depth_camera_to_world_xyz(self) -> None:
        depth = np.full((10, 10), 2.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        lifted = lift_mask_to_world(
            mask,
            depth,
            T_world_camera=IDENTITY,
            intrinsics=K0,
        )
        self.assertIsNotNone(lifted)
        camera = unproject_depth_point(4.5, 4.5, 2.0, 100.0, 100.0, 4.5, 4.5)
        world = transform_point(IDENTITY, camera)
        self.assertAlmostEqual(lifted["root"][0], world[0], places=6)
        self.assertAlmostEqual(lifted["root"][1], world[1], places=6)
        self.assertAlmostEqual(lifted["root"][2], 2.0, places=6)

    def test_varying_intrinsics_use_this_frame_k(self) -> None:
        camera = {"intrinsics": K0, "poses": []}
        da3 = {
            "intrinsics_vary": True,
            "sample_intrinsics": [
                K0,
                {**K0, "fx_px": 200.0, "fy_px": 200.0},
            ],
        }
        first = sample_intrinsics_for_frame(camera, 0, da3)
        second = sample_intrinsics_for_frame(camera, 1, da3)
        self.assertEqual(first["fx_px"], 100.0)
        self.assertEqual(second["fx_px"], 200.0)
        with self.assertRaisesRegex(ContractError, "Do not fall back"):
            sample_intrinsics_for_frame(camera, 0, {"intrinsics_vary": True})

        depth = np.full((10, 10), 2.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[4, 8] = True
        mask[4, 7] = True
        mask[5, 8] = True
        mask[5, 7] = True
        # pad to MIN_MASK_PIXELS
        mask[3:7, 6:10] = True
        wide = lift_mask_to_world(mask, depth, T_world_camera=IDENTITY, intrinsics=first)
        narrow = lift_mask_to_world(mask, depth, T_world_camera=IDENTITY, intrinsics=second)
        self.assertGreater(abs(wide["root"][0] - narrow["root"][0]), 0.01)

    def test_robust_3d_center_ignores_outlier_depth_pixel(self) -> None:
        depth = np.full((12, 12), 2.0)
        depth[6, 6] = 80.0
        mask = np.zeros((12, 12), dtype=bool)
        mask[4:9, 4:9] = True
        lifted = lift_mask_to_world(mask, depth, T_world_camera=IDENTITY, intrinsics=K0)
        self.assertIsNotNone(lifted)
        self.assertAlmostEqual(lifted["root"][2], 2.0, places=1)
        self.assertGreater(abs(lifted["root"][2] - 80.0), 70.0)
        self.assertEqual(lifted["estimator"], "robust_3d_center")

    def test_robust_3d_center_drops_far_outliers(self) -> None:
        near = np.tile(np.array([0.0, 0.0, 2.0]), (24, 1))
        far = np.array([[8.0, 0.0, 40.0]])
        center = robust_3d_center(np.vstack((near, far)))
        np.testing.assert_allclose(center, [0.0, 0.0, 2.0], atol=1e-12)

    def test_per_pixel_unproject_differs_from_median_depth_mean_pixel(self) -> None:
        depth = np.full((12, 12), 2.0)
        depth[2:10, 8:12] = 30.0
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 0:12] = True
        lifted = lift_mask_to_world(mask, depth, T_world_camera=IDENTITY, intrinsics=K0)
        self.assertIsNotNone(lifted)
        ys, xs = np.nonzero(mask)
        naive = unproject_depth_point(
            float(xs.mean()),
            float(ys.mean()),
            float(np.median(depth[mask])),
            100.0,
            100.0,
            4.5,
            4.5,
        )
        camera = unproject_pixels(
            xs.astype(np.float64),
            ys.astype(np.float64),
            depth[ys, xs],
            K0,
        )
        self.assertGreater(
            float(np.linalg.norm(np.asarray(lifted["root"]) - np.asarray(naive))),
            0.02,
        )
        self.assertGreater(
            float(np.linalg.norm(robust_3d_center(camera) - np.asarray(naive))),
            0.02,
        )

    def test_confidence_drops_low_weight_pixels(self) -> None:
        depth = np.full((12, 12), 2.0)
        depth[2:10, 8:12] = 20.0
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 1:11] = True
        confidence = np.ones((12, 12))
        confidence[2:10, 8:12] = 0.01
        lifted = lift_mask_to_world(
            mask,
            depth,
            T_world_camera=IDENTITY,
            intrinsics=K0,
            confidence=confidence,
            confidence_floor=0.5,
        )
        self.assertIsNotNone(lifted)
        self.assertAlmostEqual(lifted["root"][2], 2.0, places=1)


class CalibrationAndGateTest(unittest.TestCase):
    def test_known_distance_calibration_and_fake_metric_rejected(self) -> None:
        entities = _entity_tracks()
        calibration = build_known_distance_calibration(
            calibration_id="tape-tether",
            entities=entities,
            from_id="target",
            to_id="anchor",
            measured_length_m=2.0,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            **TETHER_POINTS,
        )
        validate_metric_calibration(calibration)
        self.assertAlmostEqual(
            calibration["meters_per_world_unit"],
            2.0 / calibration["observed_length_world_units"],
            places=12,
        )
        with self.assertRaisesRegex(ContractError, "known-distance"):
            reject_direct_metric_scale("metric_measured", "guessed")
        with self.assertRaisesRegex(ContractError, "cannot produce metric_measured"):
            build_known_distance_calibration(
                calibration_id="guess",
                entities=entities,
                from_id="target",
                to_id="anchor",
                measured_length_m=2.0,
                measurement_source="diameter_guess",
                circular_with_fit_parameter="rest_length_m",
                **TETHER_POINTS,
            )

    def test_hand_stamped_metric_measured_is_blocked(self) -> None:
        observation, template = _eligible_pair(scale_source="guessed")
        observation["coordinates"]["scale"] = {
            "status": "metric_measured",
            "meters_per_world_unit": 1.0,
            "source": "guessed",
        }
        template = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
            physical_up=LEVEL_CAMERA,
        )
        blockers = entity_observation_blockers(observation, template, entity_id="target")
        self.assertTrue(any("known-distance" in item for item in blockers))

    def test_circular_tether_length_cannot_use_length_fit_profile(self) -> None:
        calibration = {
            "circular_with_fit_parameter": "rest_length_m",
            "measured_length_m": 2.0,
        }
        self.assertEqual(select_real_fit_profile(calibration), FIXED_LENGTH_PROFILE)
        with self.assertRaisesRegex(ValueError, "cannot be fitted"):
            refuse_circular_length_fit(calibration, LENGTH_FIT_PROFILE)
        refuse_circular_length_fit(calibration, FIXED_LENGTH_PROFILE)

    def test_apply_measured_scale_is_the_only_metric_path(self) -> None:
        observation, _template = _eligible_pair(scale_source="relative")
        calibration = build_known_distance_calibration(
            calibration_id="tape-tether",
            entities=observation["extensions"]["phystwin.entities.v1"],
            from_id="target",
            to_id="anchor",
            measured_length_m=2.0,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            **TETHER_POINTS,
        )
        scaled = apply_measured_scale(observation, calibration)
        self.assertEqual(scaled["coordinates"]["scale"]["status"], "metric_measured")
        self.assertEqual(scaled["coordinates"]["scale"]["source"], "known_scene_distance")
        self.assertEqual(scaled["provenance"]["metric_calibration"]["from_physical_point"], "body_center")
        self.assertEqual(scaled["provenance"]["metric_calibration"]["to_physical_point"], "anchor")

    def test_missing_physical_points_are_ambiguous(self) -> None:
        calibration = build_known_distance_calibration(
            calibration_id="tape-tether",
            entities=_entity_tracks(),
            from_id="target",
            to_id="anchor",
            measured_length_m=2.0,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            **TETHER_POINTS,
        )
        del calibration["pair"]["from_physical_point"]
        del calibration["pair"]["to_physical_point"]
        with self.assertRaisesRegex(ContractError, "ambiguous|from_physical_point"):
            validate_metric_calibration(calibration)

    def test_object_to_object_endpoints_are_rejected(self) -> None:
        left = _entity_tracks()["entities"][0]["samples"]
        right = [
            {
                "sample_index": sample["sample_index"],
                "root": [value + 0.4 for value in sample["root"]],
                "visible": True,
            }
            for sample in left
        ]
        tracks = entities_payload(
            [
                entity_payload("left", "object", left),
                entity_payload("right", "object", right),
            ]
        )
        with self.assertRaisesRegex(ContractError, "two different physical points"):
            build_known_distance_calibration(
                calibration_id="two-objects",
                entities=tracks,
                from_id="left",
                to_id="right",
                measured_length_m=2.0,
                measurement_source="tape measure 2026-08-29",
                circular_with_fit_parameter="rest_length_m",
                from_physical_point="body_center",
                to_physical_point="body_center",
            )

    def test_attachment_without_attachment_kind_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "does not match physical point attachment"):
            build_known_distance_calibration(
                calibration_id="fake-attachment",
                entities=_entity_tracks(),
                from_id="target",
                to_id="anchor",
                measured_length_m=2.0,
                measurement_source="tape measure 2026-08-29",
                circular_with_fit_parameter="rest_length_m",
                from_physical_point="attachment",
                to_physical_point="anchor",
            )

    def test_anchor_to_attachment_is_accepted(self) -> None:
        tracks = _entity_tracks()
        tracks["entities"].append(
            entity_payload(
                "knot",
                "attachment",
                [
                    {
                        "sample_index": index,
                        "root": [0.05, 0.9, -2.0],
                        "visible": True,
                    }
                    for index in range(12)
                ],
            )
        )
        calibration = build_known_distance_calibration(
            calibration_id="tape-attachment",
            entities=tracks,
            from_id="knot",
            to_id="anchor",
            measured_length_m=1.8,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            from_physical_point="attachment",
            to_physical_point="anchor",
        )
        validate_metric_calibration(calibration)
        self.assertEqual(calibration["pair"]["from_physical_point"], "attachment")


class PhysicalUpTest(unittest.TestCase):
    def test_stamp_requires_explicit_physical_up(self) -> None:
        observation, template = _eligible_pair(scale_source="known_scene_distance")
        with self.assertRaisesRegex(ContractError, "explicit physical_up"):
            stamp_observation_alignment(
                template,
                observation,
                entity_id="target",
                anchor_id="anchor",
                physical_up=None,
            )

    def test_level_camera_cannot_be_measured(self) -> None:
        with self.assertRaisesRegex(ContractError, "not a measured gravity"):
            validate_physical_up({"mode": "level_camera", "source": "measured"})

    def test_level_camera_assumed_is_honest(self) -> None:
        observation, template = _eligible_pair(scale_source="known_scene_distance")
        stamped = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
            physical_up=LEVEL_CAMERA,
        )
        alignment = stamped["observation_alignment"]
        self.assertEqual(alignment["up_mode"], "level_camera")
        self.assertEqual(alignment["up_source"], "assumed")
        self.assertEqual(alignment["alignment_source"], "assumed")
        self.assertEqual(alignment["scale_source"], "measured")

    def test_supplied_measured_up_rotates_and_is_measured(self) -> None:
        observation, template = _eligible_pair(scale_source="known_scene_distance")
        stamped = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
            physical_up={
                "mode": "supplied_vector",
                "source": "measured",
                "vector_observation": [0.0, 0.0, 1.0],
            },
        )
        alignment = stamped["observation_alignment"]
        self.assertEqual(alignment["up_mode"], "supplied_vector")
        self.assertEqual(alignment["up_source"], "measured")
        self.assertEqual(alignment["alignment_source"], "measured")
        transform = alignment["T_scene_observation_m"]
        self.assertAlmostEqual(transform[0], 1.0, places=6)
        self.assertNotAlmostEqual(transform[5], 1.0, places=6)


class HashContinuityTest(unittest.TestCase):
    def test_entity_motion_hashes_match_observation_and_fit_report(self) -> None:
        observation, template = _eligible_pair(scale_source="relative")
        calibration = build_known_distance_calibration(
            calibration_id="tape-tether",
            entities=observation["extensions"]["phystwin.entities.v1"],
            from_id="target",
            to_id="anchor",
            measured_length_m=2.0,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            **TETHER_POINTS,
        )
        observation = apply_measured_scale(observation, calibration)
        template["model"]["constraints"][0]["rest_length_m"] = 2.0
        template = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
            physical_up=LEVEL_CAMERA,
        )
        validate_physical_scene(template)
        motion = motion_observation_from_entities(observation, template, entity_id="target")
        validate_physical_motion_observation(motion)
        require_motion_matches_scene_alignment(template, motion)
        observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
        self.assertEqual(motion["source"]["kind"], "scene_observation_entity_root")
        self.assertEqual(motion["source"]["sha256"], observation_hash)
        self.assertEqual(
            template["observation_alignment"]["observation_sha256"],
            observation_hash,
        )
        report = {
            "schema": "phystwin.inverse_physics_fit",
            "version": 1,
            "fit_id": "p5r-hash-continuity",
            "status": "BLOCKED_INPUT",
            "source": {
                "template_physical_scene": {
                    "id": template["scene_id"],
                    "sha256": hashlib.sha256(canonical_json_bytes(template)).hexdigest(),
                },
                "motion_observation": None,
                "scene_observation": {
                    "id": observation["observation_id"],
                    "sha256": observation_hash,
                },
            },
            "profile": FIXED_LENGTH_PROFILE,
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
            "parameters": [
                {
                    "id": "rest_length_m",
                    "unit": "meter",
                    "lower_bound": 1.6,
                    "upper_bound": 2.4,
                    "initial": 2.0,
                    "fitted": None,
                    "truth": None,
                    "held_fixed": True,
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
            ],
            "optimizer": {
                "method": "bounded_differential_evolution_with_coordinate_refinement",
                "seed": 1,
                "population_size": 0,
                "generations": 0,
                "coordinate_iterations": 0,
                "objective_evaluations": 0,
            },
            "outputs": {
                "fitted_physical_scene": None,
                "simulated_world_state": None,
            },
            "execution": {"wall_seconds": 0.0, "peak_gpu_memory_bytes": 0},
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
            "blockers": ["hash-continuity fixture"],
            "warnings": [],
            "failures": [],
        }
        validate_inverse_physics_fit(report)
        self.assertEqual(report["source"]["scene_observation"]["sha256"], observation_hash)
        tampered = copy.deepcopy(motion)
        tampered["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "cannot declare|hash|source"):
            if tampered["source"]["sha256"] != observation_hash:
                raise ContractError("inverse fit SceneObservation source does not match motion")

    def test_cross_wired_scene_observation_is_rejected(self) -> None:
        observation, template = _eligible_pair(scale_source="relative")
        calibration = build_known_distance_calibration(
            calibration_id="tape-tether",
            entities=observation["extensions"]["phystwin.entities.v1"],
            from_id="target",
            to_id="anchor",
            measured_length_m=2.0,
            measurement_source="tape measure 2026-08-29",
            circular_with_fit_parameter="rest_length_m",
            **TETHER_POINTS,
        )
        observation = apply_measured_scale(observation, calibration)
        template["model"]["constraints"][0]["rest_length_m"] = 2.0
        template = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
            physical_up=LEVEL_CAMERA,
        )
        motion = motion_observation_from_entities(observation, template, entity_id="target")
        other = copy.deepcopy(observation)
        other["observation_id"] = "scene-observation-other"
        other_hash = hashlib.sha256(canonical_json_bytes(other)).hexdigest()
        self.assertNotEqual(
            other_hash,
            template["observation_alignment"]["observation_sha256"],
        )
        crossed = copy.deepcopy(motion)
        crossed["source"]["id"] = other["observation_id"]
        crossed["source"]["sha256"] = other_hash
        with self.assertRaisesRegex(ContractError, "observation_sha256 must match"):
            require_motion_matches_scene_alignment(template, crossed)
        with self.assertRaises(FitInputBlocked) as raised:
            motion_observation_from_entities(other, template, entity_id="target")
        self.assertTrue(
            any("observation_sha256 does not match" in item for item in raised.exception.blockers)
        )

    def test_inspect_local_footage_does_not_invent_eligibility(self) -> None:
        review = inspect_local_footage(ROOT)
        self.assertEqual(review["status"], "AWAITING_FOOTAGE")
        self.assertEqual(review["eligible"], [])
        self.assertTrue(any(item["id"] == "pendulum" for item in review["rejected"]))


class FitQualitySemanticsTest(unittest.TestCase):
    def test_real_complete_report_with_large_rmse_is_execution_not_quality(self) -> None:
        report = _real_complete_fit_report(rmse_m=0.85, normalized_rmse=0.42)
        validate_inverse_physics_fit(report)
        self.assertTrue(report["validation"]["execution_valid"])
        self.assertTrue(report["validation"]["passed"])
        self.assertEqual(report["validation"]["quality"]["status"], "unassessed")
        self.assertAlmostEqual(report["validation"]["quality"]["rmse_m"], 0.85)
        self.assertAlmostEqual(report["objective"]["rmse_m"], 0.85)
        self.assertFalse(report["validation"]["synthetic_recovery"]["performed"])

    def test_quality_status_cannot_claim_a_pass(self) -> None:
        report = _real_complete_fit_report(rmse_m=0.01, normalized_rmse=0.001)
        report["validation"]["quality"]["status"] = "passed"
        with self.assertRaisesRegex(ContractError, "quality.status: unsupported"):
            validate_inverse_physics_fit(report)

    def test_unassessed_passed_must_match_execution_valid(self) -> None:
        report = _real_complete_fit_report(rmse_m=0.85, normalized_rmse=0.42)
        report["validation"]["passed"] = False
        with self.assertRaisesRegex(ContractError, "passed must equal execution_valid"):
            validate_inverse_physics_fit(report)


def _real_complete_fit_report(*, rmse_m: float, normalized_rmse: float) -> dict:
    report = copy.deepcopy(load_contract(EXAMPLES / "inverse_physics_fit_blocked_input.json"))
    report["status"] = "COMPLETE"
    report["blockers"] = []
    report["source"]["motion_observation"] = {
        "id": "p5r-motion",
        "sha256": "4" * 64,
    }
    report["source"]["scene_observation"] = None
    report["objective"].update(
        {
            "sample_count": 12,
            "mse_m2": rmse_m * rmse_m,
            "rmse_m": rmse_m,
            "trajectory_extent_m": 2.0,
            "normalized_rmse": normalized_rmse,
            "initial_mse_m2": 1.0,
            "improvement_ratio": 1.0 / max(rmse_m * rmse_m, 1e-30),
        }
    )
    for parameter in report["parameters"]:
        parameter["fitted"] = parameter["initial"]
        parameter["truth"] = None
        parameter["held_fixed"] = False
    report["outputs"] = {
        "fitted_physical_scene": {"uri": "fitted_physical_scene.json", "sha256": "2" * 64},
        "simulated_world_state": {"uri": "simulated_world_state.json", "sha256": "3" * 64},
    }
    report["validation"] = {
        "passed": True,
        "rollout_valid": True,
        "execution_valid": True,
        "quality": {
            "status": "unassessed",
            "rmse_m": rmse_m,
            "normalized_rmse": normalized_rmse,
        },
        "synthetic_recovery": {
            "performed": False,
            "within_tolerance": None,
            "max_normalized_parameter_error": None,
        },
    }
    return report


def _entity_tracks() -> dict:
    target = []
    anchor = []
    for index in range(12):
        t = index / 11.0
        target.append(
            {
                "sample_index": index,
                "root": [0.4 + 0.2 * t, -1.0 + 0.3 * math.sin(t * math.pi), -2.0 - 0.25 * t],
                "visible": True,
            }
        )
        anchor.append(
            {
                "sample_index": index,
                "root": [0.0, 1.0, -2.0],
                "visible": True,
            }
        )
    return entities_payload(
        [
            entity_payload("target", "object", target),
            entity_payload("anchor", "anchor", anchor),
        ]
    )


def _eligible_pair(*, scale_source: str = "known_scene_distance"):
    observation = copy.deepcopy(load_contract(EXAMPLES / "scene_observation.json"))
    samples = []
    poses = []
    for index in range(12):
        samples.append(
            {
                "sample_index": index,
                "timestamp_s": index * 0.05,
                "source_frames": {"video0": index},
            }
        )
        poses.append(
            {
                "sample_index": index,
                "T_world_camera": list(FIRST_CAMERA_WORLD_FROM_OPENCV),
            }
        )
    observation["timeline"]["samples"] = samples
    observation["cameras"][0]["poses"] = poses
    if scale_source == "relative":
        observation["coordinates"]["scale"] = {
            "status": "relative",
            "meters_per_world_unit": None,
            "source": "estimator",
        }
    else:
        observation["coordinates"]["scale"] = {
            "status": "metric_measured",
            "meters_per_world_unit": 1.0,
            "source": scale_source,
        }
    observation = attach_entities(observation, _entity_tracks())
    template = copy.deepcopy(load_contract(EXAMPLES / "physical_scene_tether_fit_template.json"))
    template["model"]["constraints"][0]["body_attachment_m"] = [0.0, 0.0, 0.0]
    return observation, template


if __name__ == "__main__":
    unittest.main()
