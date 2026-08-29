from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physics3d.bounded_search import ParameterSpec, bounded_differential_search
from physics3d.fit_physical_scene import TRUTH_PARAMETERS
from physics3d.inverse_fit import (
    DEFAULT_PARAMETERS,
    FIXED_LENGTH_PROFILE,
    PROFILE,
    apply_tether_parameters,
    blocked_fit_report,
    fit_tether_scene,
    refuse_circular_length_fit,
    select_real_fit_profile,
)
from physics3d.motion_observation import (
    motion_observation_from_rollout,
    scene_observation_blockers,
)
from physics3d.newton_runtime import simulate_physical_scene, transform_point
from vision.reconstruction.contracts import (
    ContractError,
    canonical_json_bytes,
    load_contract,
    validate_inverse_fit_artifacts,
    validate_inverse_physics_fit,
    validate_physical_motion_observation,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts" / "3d" / "v1" / "examples"


class P5ContractAndSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = dict(
            load_contract(EXAMPLES / "physical_scene_tether_fit_template.json")
        )

    def test_parameter_patch_preserves_exact_tether_geometry(self) -> None:
        scene = apply_tether_parameters(self.template, TRUTH_PARAMETERS)
        body = scene["model"]["bodies"][0]
        tether = scene["model"]["constraints"][0]
        attachment = transform_point(
            body["T_world_body_initial"],
            tether["body_attachment_m"],
        )
        distance = float(
            np.linalg.norm(
                np.asarray(attachment) - np.asarray(tether["world_anchor_m"])
            )
        )
        self.assertAlmostEqual(distance, TRUTH_PARAMETERS["rest_length_m"], places=12)
        velocity = np.asarray(body["linear_velocity_m_s"])
        radial = (
            np.asarray(attachment) - np.asarray(tether["world_anchor_m"])
        ) / distance
        self.assertAlmostEqual(float(np.dot(velocity, radial)), 0.0, places=12)

    def test_fixed_seed_search_is_deterministic(self) -> None:
        parameters = (
            ParameterSpec("x", -2.0, 2.0, -1.5, "unit"),
            ParameterSpec("y", -2.0, 2.0, 1.5, "unit"),
        )

        def objective(values: tuple[float, ...]) -> float:
            return (values[0] - 0.25) ** 2 + (values[1] + 0.5) ** 2

        first = bounded_differential_search(
            objective,
            parameters,
            seed=7,
            population_size=8,
            generations=8,
            coordinate_iterations=16,
        )
        second = bounded_differential_search(
            objective,
            parameters,
            seed=7,
            population_size=8,
            generations=8,
            coordinate_iterations=16,
        )
        self.assertEqual(first, second)
        self.assertLess(first.objective, 1e-5)
        self.assertGreater(first.initial_objective, first.objective)

    def test_held_fixed_parameter_stays_at_initial(self) -> None:
        parameters = (
            ParameterSpec("x", -2.0, 2.0, -1.5, "unit", held_fixed=True),
            ParameterSpec("y", -2.0, 2.0, 1.5, "unit"),
        )

        def objective(values: tuple[float, ...]) -> float:
            return (values[0] - 0.25) ** 2 + (values[1] + 0.5) ** 2

        result = bounded_differential_search(
            objective,
            parameters,
            seed=7,
            population_size=8,
            generations=8,
            coordinate_iterations=16,
        )
        self.assertAlmostEqual(result.values[0], -1.5, places=12)
        self.assertLess(abs(result.values[1] + 0.5), 1e-4)

    def test_circular_tether_calibration_refuses_length_fit(self) -> None:
        calibration = {"circular_with_fit_parameter": "rest_length_m"}
        self.assertEqual(select_real_fit_profile(calibration), FIXED_LENGTH_PROFILE)
        with self.assertRaisesRegex(ValueError, "cannot be fitted"):
            refuse_circular_length_fit(calibration, PROFILE)
        refuse_circular_length_fit(calibration, FIXED_LENGTH_PROFILE)

    def test_invalid_motion_sample_order_is_rejected(self) -> None:
        motion = _minimal_motion_observation()
        motion["track"]["samples"][1]["timestamp_s"] = 0.0
        with self.assertRaisesRegex(ContractError, "strictly increasing"):
            validate_physical_motion_observation(motion)

    def test_p5_contract_examples_validate(self) -> None:
        motion = load_contract(
            EXAMPLES / "physical_motion_observation_tether_synthetic.json"
        )
        fit = load_contract(EXAMPLES / "inverse_physics_fit_blocked_input.json")
        self.assertEqual(motion["track"]["body_id"], "bob")
        self.assertEqual(fit["status"], "BLOCKED_INPUT")

    def test_complete_fit_rejects_contradictory_recovery_fields(self) -> None:
        fit = _minimal_complete_fit_report()
        without_motion = copy.deepcopy(fit)
        without_motion["source"]["motion_observation"] = None
        with self.assertRaisesRegex(ContractError, "requires a motion observation"):
            validate_inverse_physics_fit(without_motion)

        failed_recovery = copy.deepcopy(fit)
        failed_recovery["validation"]["synthetic_recovery"]["within_tolerance"] = False
        with self.assertRaisesRegex(ContractError, "within tolerance"):
            validate_inverse_physics_fit(failed_recovery)

        unperformed = copy.deepcopy(fit)
        unperformed["validation"]["synthetic_recovery"]["performed"] = False
        with self.assertRaisesRegex(ContractError, "must be null"):
            validate_inverse_physics_fit(unperformed)

    def test_real_evidence_outside_scene_timeline_is_blocked(self) -> None:
        observation = copy.deepcopy(
            load_contract(EXAMPLES / "scene_observation.json")
        )
        observation["coordinates"]["scale"] = {
            "status": "metric_measured",
            "meters_per_world_unit": 1.0,
            "source": "measured_test_fixture",
        }
        joints = [[0.0, 0.0, -1.0] for _ in range(24)]
        observation["extensions"]["phystwin.humans.v1"] = {
            "joint_layout": "smpl24",
            "coordinate_frame": "observation_world",
            "people": [
                {
                    "id": "human0",
                    "samples": [
                        {
                            "sample_index": index,
                            "root": [0.0, 0.0, -1.0],
                            "joints": joints,
                        }
                        for index in range(2)
                    ],
                }
            ],
        }
        template = copy.deepcopy(self.template)
        template["execution"]["start_time_s"] = 1.0
        template["observation_alignment"] = {
            "observation_uri": "scene_observation.json",
            "observation_sha256": hashlib.sha256(
                canonical_json_bytes(observation)
            ).hexdigest(),
            "meters_per_observation_unit": 1.0,
            "scale_source": "measured",
            "alignment_source": "measured",
            "T_scene_observation_m": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
        blockers = scene_observation_blockers(observation, template)
        self.assertTrue(any("inside the PhysicalScene timeline" in item for item in blockers))

    def test_real_motion_cannot_declare_synthetic_truth(self) -> None:
        motion = _minimal_motion_observation()
        motion["source"]["kind"] = "scene_observation_human_root"
        motion["provenance"] = {
            "synthetic": False,
            "truth_parameters": TRUTH_PARAMETERS,
        }
        with self.assertRaisesRegex(ContractError, "cannot declare truth_parameters"):
            validate_physical_motion_observation(motion)

    def test_current_relative_observation_is_blocked_not_fitted(self) -> None:
        observation = dict(load_contract(EXAMPLES / "scene_observation.json"))
        blockers = scene_observation_blockers(observation, self.template)
        self.assertTrue(any("metric_measured" in item for item in blockers))
        report = blocked_fit_report(self.template, observation, blockers)
        validate_inverse_physics_fit(report)
        self.assertEqual(report["status"], "BLOCKED_INPUT")
        self.assertIsNone(report["source"]["motion_observation"])


class P5NewtonRecoveryTest(unittest.TestCase):
    def test_recovers_three_synthetic_parameters_and_beats_initial_control(self) -> None:
        template = dict(
            load_contract(EXAMPLES / "physical_scene_tether_fit_template.json")
        )
        truth_scene = apply_tether_parameters(template, TRUTH_PARAMETERS)
        truth_rollout = simulate_physical_scene(truth_scene, repeat_check=False)
        motion = motion_observation_from_rollout(
            truth_rollout,
            stride=2,
            truth_parameters=TRUTH_PARAMETERS,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = fit_tether_scene(
                template,
                motion,
                output_dir=Path(temporary),
                repeat_check=False,
            )
            fit = result["fit"]
            self.assertEqual(fit["status"], "COMPLETE")
            self.assertLess(fit["objective"]["normalized_rmse"], 0.002)
            self.assertGreater(fit["objective"]["improvement_ratio"], 100.0)
            self.assertTrue(
                fit["validation"]["synthetic_recovery"]["within_tolerance"]
            )
            fitted_by_id = {
                item["id"]: item["fitted"] for item in fit["parameters"]
            }
            for spec in DEFAULT_PARAMETERS:
                normalized_error = abs(
                    fitted_by_id[spec.id] - TRUTH_PARAMETERS[spec.id]
                ) / (spec.upper_bound - spec.lower_bound)
                self.assertLessEqual(normalized_error, 0.03)
            for output_name, filename in (
                ("fitted_physical_scene", "fitted_physical_scene.json"),
                ("simulated_world_state", "simulated_world_state.json"),
            ):
                content = (Path(temporary) / filename).read_bytes()
                self.assertEqual(
                    hashlib.sha256(content).hexdigest(),
                    fit["outputs"][output_name]["sha256"],
                )
            scene_path = Path(temporary) / "fitted_physical_scene.json"
            rollout_path = Path(temporary) / "simulated_world_state.json"
            scene_path.write_bytes(scene_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ContractError, "hash does not match"):
                validate_inverse_fit_artifacts(
                    fit,
                    template,
                    motion,
                    result["physical_scene"],
                    result["rollout"],
                    fitted_scene_path=scene_path,
                    rollout_path=rollout_path,
                )


def _minimal_motion_observation() -> dict:
    return {
        "schema": "phystwin.physical_motion_observation",
        "version": 1,
        "observation_id": "test-motion",
        "source": {
            "kind": "synthetic_rollout",
            "id": "test-rollout",
            "sha256": "0" * 64,
        },
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Y",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
        },
        "units": {"length": "meter", "time": "second"},
        "track": {
            "body_id": "bob",
            "point": "body_origin",
            "samples": [
                {
                    "sample_index": 0,
                    "timestamp_s": 0.0,
                    "position_m": [0.0, 0.0, 0.0],
                    "weight": 1.0,
                },
                {
                    "sample_index": 1,
                    "timestamp_s": 0.1,
                    "position_m": [0.1, 0.0, 0.0],
                    "weight": 1.0,
                },
            ],
        },
        "provenance": {"synthetic": True},
        "warnings": [],
    }


def _minimal_complete_fit_report() -> dict:
    report = copy.deepcopy(
        load_contract(EXAMPLES / "inverse_physics_fit_blocked_input.json")
    )
    report["fit_id"] = "test-complete-fit"
    report["status"] = "COMPLETE"
    report["source"]["motion_observation"] = {
        "id": "test-motion",
        "sha256": "1" * 64,
    }
    report["source"]["scene_observation"] = None
    report["objective"].update(
        {
            "sample_count": 2,
            "mse_m2": 0.0,
            "rmse_m": 0.0,
            "trajectory_extent_m": 1.0,
            "normalized_rmse": 0.0,
            "initial_mse_m2": 1.0,
            "improvement_ratio": 1e30,
        }
    )
    for parameter in report["parameters"]:
        parameter["fitted"] = parameter["initial"]
        parameter["truth"] = parameter["initial"]
        parameter["held_fixed"] = False
    report["outputs"] = {
        "fitted_physical_scene": {
            "uri": "fitted_physical_scene.json",
            "sha256": "2" * 64,
        },
        "simulated_world_state": {
            "uri": "simulated_world_state.json",
            "sha256": "3" * 64,
        },
    }
    report["validation"] = {
        "passed": True,
        "rollout_valid": True,
        "synthetic_recovery": {
            "performed": True,
            "within_tolerance": True,
            "max_normalized_parameter_error": 0.0,
        },
    }
    report["blockers"] = []
    return report


if __name__ == "__main__":
    unittest.main()
