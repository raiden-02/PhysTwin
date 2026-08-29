"""Contract and Newton checks for unconstrained free fall."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from physics3d.fit_physical_scene import FREE_FALL_TRUTH_PARAMETERS
from physics3d.inverse_fit import (
    FREE_FALL_PROFILE,
    apply_free_fall_parameters,
    fit_free_fall_scene,
    parameter_specs_for_profile,
)
from physics3d.motion_observation import motion_observation_from_rollout
from vision.reconstruction.contracts import (
    load_contract,
    validate_physical_scene,
    validate_rollout_source,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_free_fall.json"
TETHER = ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_tether.json"
NEWTON_AVAILABLE = importlib.util.find_spec("newton") is not None


class FreeFallContractTest(unittest.TestCase):
    def test_fixture_is_one_body_zero_constraints(self) -> None:
        scene = dict(load_contract(FIXTURE))
        self.assertEqual(len(scene["model"]["bodies"]), 1)
        self.assertEqual(scene["model"]["constraints"], [])
        gravity = scene["world"]["gravity_m_s2"]
        self.assertEqual(gravity[0], 0.0)
        self.assertEqual(gravity[2], 0.0)
        self.assertLess(gravity[1], 0.0)
        self.assertNotAlmostEqual(abs(gravity[1]), 9.81, places=2)
        self.assertNotAlmostEqual(abs(gravity[1]), 9.80665, places=4)

    def test_tether_fixture_still_requires_one_constraint(self) -> None:
        scene = dict(load_contract(TETHER))
        self.assertEqual(len(scene["model"]["constraints"]), 1)
        validate_physical_scene(scene)

    def test_free_fall_profile_does_not_use_earth_gravity_as_initial(self) -> None:
        specs = parameter_specs_for_profile(FREE_FALL_PROFILE)
        ids = {spec.id for spec in specs}
        self.assertEqual(ids, {"gravity_magnitude_m_s2", "initial_velocity_y_m_s"})
        gravity = next(spec for spec in specs if spec.id == "gravity_magnitude_m_s2")
        self.assertEqual(gravity.initial, 6.0)
        self.assertEqual(gravity.lower_bound, 2.0)
        self.assertEqual(gravity.upper_bound, 20.0)
        self.assertFalse(gravity.held_fixed)
        self.assertGreater(9.81 - gravity.lower_bound, 5.0)
        self.assertGreater(gravity.upper_bound - 9.81, 5.0)

    def test_apply_free_fall_sets_minus_y_gravity(self) -> None:
        scene = apply_free_fall_parameters(
            dict(load_contract(FIXTURE)),
            FREE_FALL_TRUTH_PARAMETERS,
        )
        self.assertEqual(scene["world"]["gravity_m_s2"][0], 0.0)
        self.assertAlmostEqual(scene["world"]["gravity_m_s2"][1], -9.80665)
        self.assertEqual(scene["model"]["bodies"][0]["linear_velocity_m_s"][1], 0.0)


@unittest.skipUnless(NEWTON_AVAILABLE, "Newton is required for free-fall runtime tests")
class FreeFallRuntimeTest(unittest.TestCase):
    def test_unconstrained_body_follows_analytic_gravity(self) -> None:
        from physics3d.newton_runtime import simulate_physical_scene

        scene = apply_free_fall_parameters(
            dict(load_contract(FIXTURE)),
            {"gravity_magnitude_m_s2": 8.0, "initial_velocity_y_m_s": -0.2},
        )
        rollout = simulate_physical_scene(scene, repeat_check=True)
        validate_rollout_source(rollout, scene)
        self.assertIsNone(rollout["validation"]["tether_error_m"])
        self.assertEqual(rollout["validation"]["invariant_profile"], "free_fall")
        self.assertEqual(rollout["constraints"], [])
        y0 = scene["model"]["bodies"][0]["T_world_body_initial"][7]
        vy0 = -0.2
        g = 8.0
        worst = 0.0
        last_y = y0
        for sample in rollout["bodies"][0]["samples"]:
            t = float(rollout["timeline"]["samples"][sample["sample_index"]]["timestamp_s"])
            expected = y0 + vy0 * t - 0.5 * g * t * t
            last_y = float(sample["T_world_body"][7])
            worst = max(worst, abs(last_y - expected))
            self.assertAlmostEqual(float(sample["T_world_body"][3]), 0.0, places=4)
            self.assertAlmostEqual(float(sample["T_world_body"][11]), 0.0, places=4)
        # XPBD uses semi-implicit Euler. At 60 Hz the extra 0.5 g t dt term is ~3 cm.
        self.assertLess(worst, 0.05)
        self.assertLess(last_y, y0 - 0.8)

    def test_inverse_fit_recovers_synthetic_gravity(self) -> None:
        from physics3d.newton_runtime import simulate_physical_scene

        template = dict(load_contract(FIXTURE))
        truth_scene = apply_free_fall_parameters(template, FREE_FALL_TRUTH_PARAMETERS)
        truth_rollout = simulate_physical_scene(truth_scene, repeat_check=False)
        motion = motion_observation_from_rollout(
            truth_rollout,
            stride=2,
            truth_parameters=FREE_FALL_TRUTH_PARAMETERS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = fit_free_fall_scene(
                template,
                motion,
                output_dir=Path(tmp),
                population_size=8,
                generations=4,
                coordinate_iterations=12,
                repeat_check=False,
            )
        fit = result["fit"]
        self.assertEqual(fit["status"], "COMPLETE")
        self.assertEqual(fit["profile"], FREE_FALL_PROFILE)
        recovered = {
            item["id"]: item["fitted"]
            for item in fit["parameters"]
        }
        self.assertAlmostEqual(
            recovered["gravity_magnitude_m_s2"],
            FREE_FALL_TRUTH_PARAMETERS["gravity_magnitude_m_s2"],
            delta=0.50,
        )
        self.assertLess(fit["objective"]["normalized_rmse"], 0.02)
        self.assertTrue(fit["validation"]["synthetic_recovery"]["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
