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
        self.assertIn("evaluation ground truth", config["seed_notes"])

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

    def test_evaluation_gravity_helper_is_separate_from_clip_config(self) -> None:
        config = load_falling_ball_clip_config(ROOT)
        self.assertNotIn("gravity_truth_m_s2", config)
        benchmark = {
            "gravity_truth_m_s2": 9.81,
        }
        self.assertAlmostEqual(evaluation_gravity_m_s2(benchmark), 9.81)


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
