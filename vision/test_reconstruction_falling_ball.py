"""CPU tests for IRIS falling-ball metadata and known-radius motion assembly."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.falling_ball import (
    lift_and_filter_frames,
    motion_observation_from_sphere_track,
    stamp_free_fall_template,
    static_camera_report,
)
from vision.reconstruction.iris_falling import (
    CLIP_CONFIG_NAME,
    assert_no_iris_gravity_truth,
    calibration_provenance,
    evaluation_gravity_m_s2,
    load_falling_ball_clip_config,
)
from vision.reconstruction.contracts import load_contract


class FallingBallMetadataTest(unittest.TestCase):
    def test_checked_in_clip_uses_big_ball_and_hides_gravity_from_fit(self) -> None:
        config = load_falling_ball_clip_config(ROOT)
        self.assertEqual(config["class_key"], "falling_ball")
        self.assertEqual(config["setting_key"], "big")
        self.assertEqual(config["relative_video"], "Falling_ball/big/01.mp4")
        self.assertAlmostEqual(config["ball_radius_m"], 0.11)
        self.assertTrue(config["static_camera"])
        self.assertNotIn("gravity", config)
        self.assertIn("no gravity field", config["seed_notes"])
        assert_no_iris_gravity_truth(config)

    def test_committed_falling_ball_result_is_metric_gravity_recovery(self) -> None:
        record = json.loads(
            (ROOT / "docs" / "evaluation" / "iris-p5r-falling-ball.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(record["gravity_used_during_fit"])
        self.assertTrue(record["gate"]["accepted"])
        self.assertLess(record["gravity_percent_error"], 20.0)
        self.assertLess(record["normalized_rmse"], 0.20)
        self.assertEqual(record["accepted_frames"], 16)
        self.assertEqual(record["rejected_frames"], 0)

    def test_evaluation_gravity_requires_an_existing_fit_artifact(self) -> None:
        missing = ROOT / "docs" / "evaluation" / "does-not-exist-fit.json"
        with self.assertRaises(FileNotFoundError):
            evaluation_gravity_m_s2(ROOT, fit_artifact=missing)

    def test_fit_preparation_provenance_contains_no_iris_gravity(self) -> None:
        config = load_falling_ball_clip_config(ROOT)
        clip_input = {
            "dataset": "IRIS",
            "evidence_kind": "external_dataset",
            "repo_id": "rasulkhanbayov/IRIS",
            "source_url": "https://huggingface.co/datasets/rasulkhanbayov/IRIS",
            "class_key": config["class_key"],
            "setting_key": config["setting_key"],
            "relative_video": config["relative_video"],
            "ball_radius": {"mean": 0.11, "unit": "m"},
            "drop_height": {"mean": 1.0, "unit": "m"},
            "ball_radius_m": 0.11,
            "drop_height_m": 1.0,
        }
        assert_no_iris_gravity_truth(clip_input)
        provenance = calibration_provenance(clip_input)
        assert_no_iris_gravity_truth(provenance)
        leaked = dict(clip_input)
        leaked["gravity_truth_m_s2"] = 9.81
        with self.assertRaisesRegex(AssertionError, "gravity"):
            assert_no_iris_gravity_truth(leaked)


class FallingBallReconstructionTest(unittest.TestCase):
    def test_static_camera_accepts_tiny_da3_drift(self) -> None:
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        shifted = list(identity)
        shifted[3] = 0.01
        report = static_camera_report(
            {
                "poses": [
                    {"T_world_camera": identity},
                    {"T_world_camera": shifted},
                ]
            }
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(report["assumed_static"])
        self.assertEqual(report["units"], "da3_reconstruction")
        self.assertEqual(report["scale"], "relative")
        self.assertIn("max_translation_world", report)
        self.assertNotIn("max_translation_m", report)

    def test_depth_jump_is_rejected(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        yy, xx = np.ogrid[:80, :80]
        mask[(xx - 40) ** 2 + (yy - 40) ** 2 <= 14 ** 2] = True
        far = np.zeros((80, 80), dtype=bool)
        far[(xx - 40) ** 2 + (yy - 40) ** 2 <= 6 ** 2] = True
        lifted = lift_and_filter_frames(
            [
                {
                    "sample_index": 0,
                    "source_frame": 0,
                    "timestamp_s": 0.0,
                    "mask": mask,
                },
                {
                    "sample_index": 1,
                    "source_frame": 1,
                    "timestamp_s": 0.03,
                    "mask": far,
                },
            ],
            radius_m=0.11,
            intrinsics={"fx_px": 220.0, "fy_px": 220.0, "cx_px": 40.0, "cy_px": 40.0},
        )
        self.assertEqual(len(lifted["accepted"]), 1)
        self.assertIn("jumped", lifted["rejected"][0]["reason"])

    def test_per_frame_intrinsics_change_reconstructed_depth(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        yy, xx = np.ogrid[:80, :80]
        mask[(xx - 40) ** 2 + (yy - 40) ** 2 <= 14 ** 2] = True
        shared_k = {"fx_px": 220.0, "fy_px": 220.0, "cx_px": 40.0, "cy_px": 40.0}
        wider_k = {"fx_px": 260.0, "fy_px": 260.0, "cx_px": 40.0, "cy_px": 40.0}
        frames = [
            {
                "sample_index": 0,
                "source_frame": 0,
                "timestamp_s": 0.0,
                "mask": mask,
                "intrinsics": shared_k,
            },
            {
                "sample_index": 1,
                "source_frame": 1,
                "timestamp_s": 0.03,
                "mask": mask,
                "intrinsics": wider_k,
            },
        ]
        per_frame = lift_and_filter_frames(frames, radius_m=0.11)
        silent_frame0 = lift_and_filter_frames(
            [
                {key: value for key, value in frames[0].items() if key != "intrinsics"},
                {key: value for key, value in frames[1].items() if key != "intrinsics"},
            ],
            radius_m=0.11,
            intrinsics=shared_k,
        )
        self.assertEqual(len(per_frame["accepted"]), 2)
        self.assertGreater(
            abs(per_frame["accepted"][0]["depth_m"] - per_frame["accepted"][1]["depth_m"]),
            0.05,
        )
        self.assertAlmostEqual(
            silent_frame0["accepted"][0]["depth_m"],
            silent_frame0["accepted"][1]["depth_m"],
            places=6,
        )
        self.assertNotAlmostEqual(
            per_frame["accepted"][1]["depth_m"],
            silent_frame0["accepted"][1]["depth_m"],
            places=3,
        )

    def test_motion_and_template_use_first_accepted_point(self) -> None:
        accepted = [
            {
                "timestamp_s": 1.40,
                "position_m": [0.1, 0.8, -1.5],
            },
            {
                "timestamp_s": 1.50,
                "position_m": [0.1, 0.4, -1.5],
            },
        ]
        motion = motion_observation_from_sphere_track(
            accepted,
            observation_id="test-fall",
            source_id="test",
            source_sha256="0" * 64,
        )
        self.assertEqual(motion["source"]["kind"], "metric_sphere_track")
        self.assertAlmostEqual(motion["track"]["samples"][0]["timestamp_s"], 0.0)
        self.assertAlmostEqual(motion["track"]["samples"][1]["timestamp_s"], 0.1)
        template = dict(
            load_contract(ROOT / "contracts" / "3d" / "v1" / "examples" / "physical_scene_free_fall.json")
        )
        scene = stamp_free_fall_template(template, motion, radius_m=0.11)
        body = scene["model"]["bodies"][0]
        self.assertAlmostEqual(body["T_world_body_initial"][3], 0.1)
        self.assertAlmostEqual(body["T_world_body_initial"][7], 0.8)
        self.assertAlmostEqual(body["T_world_body_initial"][11], -1.5)
        self.assertEqual(scene["observation_alignment"]["scale_source"], "measured")
        self.assertAlmostEqual(abs(scene["world"]["gravity_m_s2"][1]), 6.0)


if __name__ == "__main__":
    unittest.main()
