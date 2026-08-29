"""CPU tests for IRIS metadata and the first real pendulum clip."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.iris import (
    EVIDENCE_KIND,
    IRIS_REPO_ID,
    iris_benchmark_review,
    load_clip_config,
    load_iris_parameters,
    measured_mean_m,
    setting_parameters,
)


class IrisMetadataTest(unittest.TestCase):
    def test_checked_in_clip_config_is_external_dataset(self) -> None:
        config = load_clip_config(ROOT)
        self.assertEqual(config["evidence_kind"], EVIDENCE_KIND)
        self.assertEqual(config["repo_id"], IRIS_REPO_ID)
        self.assertEqual(config["relative_video"], "Pendulum/pendulum_45/01.mp4")
        self.assertEqual(config["from_physical_point"], "body_center")
        self.assertEqual(config["to_physical_point"], "anchor")
        self.assertEqual(config["circular_with"], "rest_length_m")
        self.assertEqual(len(config["target_xy"]), 2)
        self.assertEqual(len(config["anchor_xy"]), 2)

    def test_parameters_fixture_exposes_rope_length(self) -> None:
        payload = {
            "pendulum": {
                "pendulum_45": {
                    "angle": {"mean": 45.0, "std": 0.0, "min": 45.0, "max": 45.0},
                    "rope_length": {"mean": 0.50, "std": 0.0, "min": 0.50, "max": 0.50},
                    "camera_to_cable": {"mean": 0.96, "std": 0.0, "min": 0.96, "max": 0.96},
                }
            }
        }
        setting = setting_parameters(payload, "pendulum", "pendulum_45")
        self.assertAlmostEqual(measured_mean_m(setting, "rope_length"), 0.50)
        with self.assertRaises(KeyError):
            setting_parameters(payload, "pendulum", "missing")

    def test_review_without_files_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            examples = fake_root / "contracts" / "3d" / "v1" / "examples"
            examples.mkdir(parents=True)
            examples.joinpath("p5r_iris_pendulum_45_01.json").write_text(
                json.dumps(load_clip_config(ROOT)),
                encoding="utf-8",
            )
            review = iris_benchmark_review(fake_root)
            self.assertFalse(review["present"])
            self.assertFalse(review["eligible"])
            self.assertEqual(review["kind"], EVIDENCE_KIND)

    def test_committed_run_record_is_not_length_recovery(self) -> None:
        record = json.loads(
            (ROOT / "docs" / "evaluation" / "iris-p5r-pendulum-45-01.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(record["provenance"]["evidence_kind"], EVIDENCE_KIND)
        self.assertEqual(record["provenance"]["iris_rope_length_m"], 0.5)
        self.assertEqual(record["provenance"]["held_fixed_parameter"], "rest_length_m")
        self.assertFalse(record["independent_rope_length_recovery"])
        self.assertEqual(record["quality_status"], "unassessed")
        self.assertNotIn("proceed_to_p7", record)
        self.assertFalse(record["overlay"]["same_qualitative_motion"])

    def test_on_disk_iris_parameters_match_the_clip_config(self) -> None:
        dataset = ROOT / "datasets" / "IRIS"
        params_path = dataset / "parameters.json"
        if not params_path.is_file():
            self.skipTest("IRIS parameters.json is not downloaded")
        parameters = load_iris_parameters(dataset)
        config = load_clip_config(ROOT)
        setting = setting_parameters(parameters, config["class_key"], config["setting_key"])
        self.assertAlmostEqual(measured_mean_m(setting, "rope_length"), 0.50)
        review = iris_benchmark_review(ROOT)
        if review["present"]:
            self.assertTrue(review["eligible"])
            self.assertEqual(review["kind"], "external_dataset")


if __name__ == "__main__":
    unittest.main()
