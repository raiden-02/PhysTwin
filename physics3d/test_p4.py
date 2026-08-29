"""Contract and runtime checks for the P4 Newton/Warp vertical slice."""

from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path

from vision.reconstruction.contracts import (
    ContractError,
    load_contract,
    validate_physical_scene,
    validate_rollout_source,
    validate_simulated_world_state,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_tether.json"
NEWTON_AVAILABLE = importlib.util.find_spec("newton") is not None


class P4ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = dict(load_contract(FIXTURE))

    def test_fixture_is_standalone_metric_physical_scene(self) -> None:
        self.assertEqual(self.scene["execution"]["status"], "executable")
        self.assertEqual(self.scene["execution"]["backend"], "newton")
        self.assertIsNone(self.scene["observation_alignment"]["observation_uri"])
        self.assertEqual(self.scene["observation_alignment"]["scale_source"], "not_applicable")
        gravity = self.scene["world"]["gravity_m_s2"]
        self.assertEqual(gravity[0], 0.0)
        self.assertLess(gravity[1], 0.0)
        self.assertAlmostEqual(abs(gravity[1]), 9.80665)
        self.assertEqual(gravity[2], 0.0)

    def test_coordinate_and_unit_mistakes_are_rejected(self) -> None:
        for section, field, invalid, message in (
            ("coordinates", "up_axis", "+Z", "up_axis"),
            ("coordinates", "matrix_storage", "column_major", "matrix_storage"),
            ("units", "length", "centimeter", "length"),
        ):
            scene = copy.deepcopy(self.scene)
            scene[section][field] = invalid
            with self.assertRaisesRegex(ContractError, message):
                validate_physical_scene(scene)

    def test_transposed_transform_is_rejected(self) -> None:
        scene = copy.deepcopy(self.scene)
        matrix = scene["model"]["bodies"][0]["T_world_body_initial"]
        scene["model"]["bodies"][0]["T_world_body_initial"] = [
            matrix[column * 4 + row] for row in range(4) for column in range(4)
        ]
        with self.assertRaisesRegex(ContractError, "last row"):
            validate_physical_scene(scene)

    def test_body_local_attachment_controls_initial_tether_length(self) -> None:
        scene = copy.deepcopy(self.scene)
        scene["model"]["constraints"][0]["body_attachment_m"][1] = -0.15
        with self.assertRaisesRegex(ContractError, "attachment"):
            validate_physical_scene(scene)

    def test_unsupported_execution_inputs_are_rejected(self) -> None:
        invalid_scenes = []
        upward_gravity = copy.deepcopy(self.scene)
        upward_gravity["world"]["gravity_m_s2"][1] = 9.80665
        invalid_scenes.append((upward_gravity, "gravity"))
        cpu = copy.deepcopy(self.scene)
        cpu["execution"]["device"] = "cpu"
        invalid_scenes.append((cpu, "CUDA"))
        fractional_steps = copy.deepcopy(self.scene)
        fractional_steps["execution"]["duration_s"] = 4.001
        invalid_scenes.append((fractional_steps, "integer multiple"))
        unsupported_force = copy.deepcopy(self.scene)
        unsupported_force["model"]["forces"] = [{"id": "wind", "type": "constant"}]
        invalid_scenes.append((unsupported_force, "unsupported components"))
        for scene, message in invalid_scenes:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    validate_physical_scene(scene)


@unittest.skipUnless(NEWTON_AVAILABLE, "run with .venv-physics for Newton/Warp validation")
class P4NewtonRuntimeTest(unittest.TestCase):
    def test_gpu_rollout_is_finite_constrained_and_three_dimensional(self) -> None:
        from physics3d.newton_runtime import simulate_physical_scene, transform_point

        scene = dict(load_contract(FIXTURE))
        rollout = simulate_physical_scene(scene, repeat_check=True)
        self.assertEqual(rollout["simulator"]["backend"], "newton")
        self.assertEqual(rollout["simulator"]["device"], "cuda:0")
        self.assertEqual(rollout["simulator"]["up_axis"], "+Y")
        self.assertTrue(rollout["validation"]["passed"])
        self.assertTrue(rollout["validation"]["finite_state"])
        self.assertTrue(rollout["validation"]["gravity_matches_contract"])
        self.assertLess(rollout["validation"]["backend_gravity_m_s2"][1], 0.0)
        self.assertAlmostEqual(
            abs(rollout["validation"]["backend_gravity_m_s2"][1]),
            9.80665,
            places=5,
        )
        self.assertLess(rollout["validation"]["tether_error_m"]["maximum"], 1e-5)
        self.assertEqual(
            rollout["validation"]["body_position_range_m"]["varying_axis_count_at_0_05_m"],
            3,
        )
        repeat = rollout["reproducibility"]["repeat_run"]
        self.assertTrue(repeat["performed"])
        self.assertTrue(repeat["within_tolerance"])
        self.assertEqual(repeat["max_abs_transform_delta"], 0.0)

        body = rollout["bodies"][0]
        constraint = rollout["constraints"][0]
        expected_initial = scene["model"]["bodies"][0]["T_world_body_initial"]
        for actual, expected in zip(body["samples"][0]["T_world_body"], expected_initial):
            self.assertAlmostEqual(actual, expected, places=6)
        for sample in (body["samples"][0], body["samples"][len(body["samples"]) // 2], body["samples"][-1]):
            attachment_world = transform_point(
                sample["T_world_body"],
                constraint["body_attachment_m"],
            )
            distance = math.sqrt(
                sum(
                    (attachment_world[axis] - constraint["world_anchor_m"][axis]) ** 2
                    for axis in range(3)
                )
            )
            self.assertAlmostEqual(distance, constraint["rest_length_m"], delta=1e-5)

        malformed = copy.deepcopy(rollout)
        malformed["bodies"][0]["shape"]["radius_m"] = -1.0
        with self.assertRaisesRegex(ContractError, "positive-radius sphere"):
            validate_simulated_world_state(malformed)
        wrong_source = copy.deepcopy(rollout)
        wrong_source["source"]["physical_scene_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "SHA-256"):
            validate_rollout_source(wrong_source, scene)
        wrong_mass = copy.deepcopy(rollout)
        wrong_mass["bodies"][0]["mass_kg"] = 2.0
        with self.assertRaisesRegex(ContractError, "body metadata"):
            validate_rollout_source(wrong_mass, scene)


if __name__ == "__main__":
    unittest.main()
