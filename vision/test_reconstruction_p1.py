"""P1 checks for DA3 pose conversion, cache publish, and GLB identity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.cache import is_complete, publish_observation, sha256_file
from vision.reconstruction.contracts import (
    FIRST_CAMERA_WORLD_FROM_OPENCV,
    project_world_point,
    validate_scene_observation,
)
from vision.reconstruction.geometry import read_glb_point_count, write_point_cloud_glb
from vision.reconstruction.transforms import (
    canonical_poses_from_da3_w2c,
    scale_intrinsics,
    transform_point,
    unproject_depth_point,
)
from vision.reconstruction.video import choose_source_frames


FIXTURE = ROOT / "contracts" / "3d" / "v1" / "examples" / "da3_w2c_fixture.json"


class Da3ConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_identity_w2c_defines_first_camera_graphics_world(self) -> None:
        _t_obs, poses = canonical_poses_from_da3_w2c(self.fixture["w2c"])
        for actual, expected in zip(poses[0], FIRST_CAMERA_WORLD_FROM_OPENCV):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(poses[1], self.fixture["expected_T_world_camera"][1]):
            self.assertAlmostEqual(actual, expected)

    def test_intrinsics_scale_to_source_pixels(self) -> None:
        scaled = scale_intrinsics(
            self.fixture["processed_intrinsics"],
            source_size=tuple(self.fixture["source_size_px"]),
            processed_size=tuple(self.fixture["processed_size_px"]),
        )
        self.assertEqual(scaled, self.fixture["expected_source_intrinsics"])

    def test_optical_axis_projects_to_principal_point(self) -> None:
        _t_obs, poses = canonical_poses_from_da3_w2c(self.fixture["w2c"])
        k = self.fixture["expected_source_intrinsics"]
        camera_point = unproject_depth_point(k["cx_px"], k["cy_px"], 2.0, k["fx_px"], k["fy_px"], k["cx_px"], k["cy_px"])
        world = transform_point(poses[0], camera_point)
        self.assertAlmostEqual(world[0], 0.0)
        self.assertAlmostEqual(world[1], 0.0)
        self.assertAlmostEqual(world[2], -2.0)
        u, v = project_world_point(world, poses[0], k)
        self.assertAlmostEqual(u, k["cx_px"])
        self.assertAlmostEqual(v, k["cy_px"])

    def test_frame_window_is_evenly_spaced(self) -> None:
        frames = choose_source_frames(48, 24.0, start_s=0.0, duration_s=2.0, max_frames=5)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 47)
        self.assertEqual(len(frames), 5)
        self.assertEqual(len(set(frames)), 5)


class ArtifactCacheTest(unittest.TestCase):
    def test_glb_round_trip_and_atomic_publish(self) -> None:
        points = np.array([[0.0, 0.0, -1.0], [0.2, 0.1, -1.5]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / "results" / "cache" / "reconstruction" / "abc"

            def build(work_dir: Path):
                path = work_dir / "artifacts" / "scene.glb"
                write_point_cloud_glb(path, points, colors)
                self.assertEqual(read_glb_point_count(path), 2)
                return {
                    "schema": "phystwin.scene_observation",
                    "version": 1,
                    "observation_id": "cache-test",
                    "timeline": {
                        "time_unit": "second",
                        "origin": "observation_start",
                        "samples": [
                            {
                                "sample_index": 0,
                                "timestamp_s": 0.0,
                                "source_frames": {"video0": 0},
                            }
                        ],
                    },
                    "coordinates": {
                        "world_basis": "first_camera_graphics",
                        "handedness": "right",
                        "camera_convention": "opencv",
                        "transform_notation": "T_parent_child",
                        "vector_convention": "column",
                        "matrix_storage": "row_major",
                        "scale": {
                            "status": "relative",
                            "meters_per_world_unit": None,
                            "source": "estimator",
                        },
                    },
                    "sources": [
                        {
                            "id": "video0",
                            "kind": "video",
                            "uri": "clip.mp4",
                            "media_type": "video/mp4",
                            "sha256": "1" * 64,
                        }
                    ],
                    "artifacts": [
                        {
                            "id": "scene_geometry",
                            "uri": "artifacts/scene.glb",
                            "media_type": "model/gltf-binary",
                            "sha256": sha256_file(path),
                        }
                    ],
                    "cameras": [
                        {
                            "id": "camera0",
                            "source_id": "video0",
                            "projection": "pinhole",
                            "image_size_px": [320, 240],
                            "intrinsics": {
                                "fx_px": 200.0,
                                "fy_px": 200.0,
                                "cx_px": 160.0,
                                "cy_px": 120.0,
                                "skew_px": 0.0,
                            },
                            "poses": [
                                {
                                    "sample_index": 0,
                                    "T_world_camera": list(FIRST_CAMERA_WORLD_FROM_OPENCV),
                                }
                            ],
                        }
                    ],
                    "static_scene": {
                        "geometry": [{"kind": "point_cloud", "artifact_id": "scene_geometry"}]
                    },
                    "provenance": {"adapter": "test"},
                    "extensions": {},
                }

            observation = publish_observation(entry, build)
            validate_scene_observation(observation)
            self.assertTrue(is_complete(entry))
            self.assertTrue((entry / "COMPLETE").is_file())
            self.assertFalse(any(entry.parent.glob(".tmp-*")))


if __name__ == "__main__":
    unittest.main()
