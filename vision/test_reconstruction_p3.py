"""P3 checks for first-camera-aligned camera and body metrics."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.evaluate_reconstruction import _fixture_case
from vision.reconstruction.evaluation import (
    evaluate_observation,
    load_emdb_reference,
    save_evaluation,
    validate_evaluation_report,
)


class ReconstructionEvaluationTest(unittest.TestCase):
    def test_known_body_offset_separates_world_and_local_error(self) -> None:
        prediction, reference = _fixture_case(0.05)
        output = evaluate_observation(prediction, reference)
        metrics = output.report["metrics"]
        self.assertAlmostEqual(metrics["root_position_rmse_m"]["value"], 0.05)
        self.assertAlmostEqual(metrics["world_mpjpe_mm"]["value"], 50.0)
        self.assertAlmostEqual(metrics["pelvis_aligned_mpjpe_mm"]["value"], 0.0)
        self.assertAlmostEqual(metrics["pa_mpjpe_mm"]["value"], 0.0, places=5)
        self.assertAlmostEqual(metrics["camera_position_rmse_m"]["value"], 0.0)
        self.assertAlmostEqual(metrics["camera_rotation_mean_deg"]["value"], 0.0)

    def test_relative_scale_blocks_metric_world_errors(self) -> None:
        prediction, reference = _fixture_case(0.0)
        relative = copy.deepcopy(prediction)
        relative["coordinates"]["scale"] = {
            "status": "relative",
            "meters_per_world_unit": None,
            "source": "test",
        }
        output = evaluate_observation(relative, reference)
        metrics = output.report["metrics"]
        self.assertEqual(metrics["world_mpjpe_mm"]["status"], "blocked")
        self.assertEqual(metrics["camera_position_rmse_m"]["status"], "blocked")
        self.assertEqual(metrics["pa_mpjpe_mm"]["status"], "available")
        self.assertEqual(metrics["camera_rotation_mean_deg"]["status"], "available")

    def test_report_and_svg_are_written_with_artifact_hash(self) -> None:
        prediction, reference = _fixture_case(0.05)
        output = evaluate_observation(prediction, reference)
        with tempfile.TemporaryDirectory() as temp:
            report = save_evaluation(Path(temp), output)
            validate_evaluation_report(report)
            svg = Path(temp) / "trajectory_comparison.svg"
            document = Path(temp) / "reconstruction_evaluation.json"
            self.assertTrue(svg.is_file())
            self.assertTrue(document.is_file())
            self.assertEqual(len(report["artifacts"][0]["sha256"]), 64)

    def test_emdb_loader_requires_approved_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "approved EMDB"):
                load_emdb_reference(Path(temp), Path(temp))

    def test_camera_rotation_metric_uses_geodesic_degrees(self) -> None:
        prediction, reference = _fixture_case(0.0)
        rotated = copy.deepcopy(prediction)
        angle = np.deg2rad(10.0)
        pose = rotated["cameras"][0]["poses"][1]["T_world_camera"]
        current = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        delta = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        current[:3, :3] = delta @ current[:3, :3]
        pose[:] = current.reshape(-1).tolist()
        output = evaluate_observation(rotated, reference)
        per_frame = output.report["per_frame"]["camera_rotation_error_deg"]
        self.assertAlmostEqual(per_frame[1], 10.0, places=5)

    def test_varying_intrinsics_block_single_k_reprojection(self) -> None:
        prediction, reference = _fixture_case(0.0)
        prediction["cameras"][0]["lens_distortion"] = "removed"
        prediction["extensions"]["phystwin.da3.v1"] = {"intrinsics_vary": True}
        reference = replace(
            reference,
            keypoints_2d_px=np.zeros((12, 24, 2), dtype=np.float64),
        )
        output = evaluate_observation(prediction, reference)
        metric = output.report["metrics"]["reprojection_mpjpe_px"]
        self.assertEqual(metric["status"], "blocked")
        self.assertIn("intrinsics vary", metric["reason"])


if __name__ == "__main__":
    unittest.main()
