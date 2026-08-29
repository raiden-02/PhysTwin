"""Counterfactual gravity edits must slow the fall when gravity is weaker."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from physics3d.counterfactual import MOON_GRAVITY_M_S2, clone_scene_with_gravity
from physics3d.fit_physical_scene import FREE_FALL_TRUTH_PARAMETERS
from physics3d.inverse_fit import apply_free_fall_parameters
from vision.reconstruction.contracts import load_contract


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_free_fall.json"
NEWTON_AVAILABLE = importlib.util.find_spec("newton") is not None


class CounterfactualContractTest(unittest.TestCase):
    def test_clone_changes_only_gravity_and_records_provenance(self) -> None:
        source = apply_free_fall_parameters(
            dict(load_contract(FIXTURE)),
            FREE_FALL_TRUTH_PARAMETERS,
        )
        moon = clone_scene_with_gravity(
            source,
            gravity_magnitude_m_s2=MOON_GRAVITY_M_S2,
            label="moon",
        )
        self.assertAlmostEqual(abs(moon["world"]["gravity_m_s2"][1]), MOON_GRAVITY_M_S2)
        self.assertEqual(
            source["model"]["bodies"][0]["T_world_body_initial"],
            moon["model"]["bodies"][0]["T_world_body_initial"],
        )
        self.assertEqual(
            source["model"]["bodies"][0]["linear_velocity_m_s"],
            moon["model"]["bodies"][0]["linear_velocity_m_s"],
        )
        meta = moon["provenance"]["counterfactual"]
        self.assertFalse(meta["observed"])
        self.assertEqual(meta["parameter"], "gravity_magnitude_m_s2")
        self.assertAlmostEqual(meta["original_value"], 9.80665)
        self.assertAlmostEqual(meta["counterfactual_value"], MOON_GRAVITY_M_S2)


@unittest.skipUnless(NEWTON_AVAILABLE, "Newton is required for counterfactual runtime tests")
class CounterfactualRuntimeTest(unittest.TestCase):
    def test_moon_gravity_falls_slower(self) -> None:
        from physics3d.newton_runtime import simulate_physical_scene

        earth = apply_free_fall_parameters(
            dict(load_contract(FIXTURE)),
            FREE_FALL_TRUTH_PARAMETERS,
        )
        moon = clone_scene_with_gravity(
            earth,
            gravity_magnitude_m_s2=MOON_GRAVITY_M_S2,
            label="moon",
        )
        earth_rollout = simulate_physical_scene(earth, repeat_check=False)
        moon_rollout = simulate_physical_scene(moon, repeat_check=False)
        earth_y = [sample["T_world_body"][7] for sample in earth_rollout["bodies"][0]["samples"]]
        moon_y = [sample["T_world_body"][7] for sample in moon_rollout["bodies"][0]["samples"]]
        self.assertLess(earth_y[-1], moon_y[-1])
        self.assertLess(earth_y[-1] - earth_y[0], moon_y[-1] - moon_y[0])


if __name__ == "__main__":
    unittest.main()
