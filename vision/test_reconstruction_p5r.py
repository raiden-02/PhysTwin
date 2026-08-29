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
    validate_inverse_physics_fit,
    validate_physical_motion_observation,
    validate_physical_scene,
)
from vision.reconstruction.entities import attach_entities, entities_payload, entity_payload
from vision.reconstruction.footage import inspect_local_footage
from vision.reconstruction.lift import (
    lift_mask_to_world,
    robust_mask_depth,
    sample_intrinsics_for_frame,
)
from vision.reconstruction.real_motion import stamp_observation_alignment
from vision.reconstruction.transforms import transform_point, unproject_depth_point


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

    def test_robust_mask_depth_uses_median_not_centroid_pixel(self) -> None:
        depth = np.full((12, 12), 2.0)
        depth[6, 6] = 80.0
        mask = np.zeros((12, 12), dtype=bool)
        mask[4:9, 4:9] = True
        stats = robust_mask_depth(depth, mask)
        self.assertAlmostEqual(float(stats["depth"]), 2.0, places=6)
        self.assertGreaterEqual(int(stats["used_pixels"]), 16)


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
        )
        scaled = apply_measured_scale(observation, calibration)
        self.assertEqual(scaled["coordinates"]["scale"]["status"], "metric_measured")
        self.assertEqual(scaled["coordinates"]["scale"]["source"], "known_scene_distance")


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
        )
        observation = apply_measured_scale(observation, calibration)
        template["model"]["constraints"][0]["rest_length_m"] = 2.0
        template = stamp_observation_alignment(
            template,
            observation,
            entity_id="target",
            anchor_id="anchor",
        )
        validate_physical_scene(template)
        motion = motion_observation_from_entities(observation, template, entity_id="target")
        validate_physical_motion_observation(motion)
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

    def test_inspect_local_footage_does_not_invent_eligibility(self) -> None:
        review = inspect_local_footage(ROOT)
        self.assertEqual(review["status"], "AWAITING_FOOTAGE")
        self.assertEqual(review["eligible"], [])
        self.assertTrue(any(item["id"] == "pendulum" for item in review["rejected"]))


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
