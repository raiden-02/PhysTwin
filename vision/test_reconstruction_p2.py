"""P2 checks for TRAM camera/body conversion and humans.v1."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.cache import humans_cache_entry, is_complete, publish_observation
from vision.reconstruction.contracts import (
    FIRST_CAMERA_WORLD_FROM_OPENCV,
    ContractError,
    project_world_point,
    validate_scene_observation,
)
from vision.reconstruction.humans import (
    HUMANS_EXTENSION,
    HumanReconstructionRequest,
    attach_humans,
    human_cache_key,
    humans_payload,
    lift_joints_to_world,
    person_payload,
    sample_payload,
    validate_humans_v1,
)
from vision.reconstruction.tram import (
    TRAM_UNAVAILABLE,
    TramHumanAdapter,
    TramUnavailableError,
    convert_tram_cameras,
    load_tram_c2w_fixture,
    make_descriptor,
    native_from_fixture,
    normalize_human_options,
    observation_from_tram_native,
    rest_smpl24_camera,
    synthetic_walk_native,
    write_projected_skeleton_video,
)
from vision.reconstruction.transforms import transform_point


FIXTURE = ROOT / "contracts" / "3d" / "v1" / "examples" / "tram_c2w_fixture.json"


def _apply_f(point: list[float]) -> list[float]:
    return [point[0], -point[1], -point[2]]


class TramConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_tram_c2w_fixture(FIXTURE)

    def test_identity_c2w_defines_first_camera_graphics_world(self) -> None:
        _t_obs, poses = convert_tram_cameras(self.fixture["pred_cam_R"], self.fixture["pred_cam_T"])
        for actual, expected in zip(poses[0], FIRST_CAMERA_WORLD_FROM_OPENCV):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(poses[1], self.fixture["expected_T_world_camera"][1]):
            self.assertAlmostEqual(actual, expected)

    def test_optical_axis_joint_lands_on_negative_z(self) -> None:
        _t_obs, poses = convert_tram_cameras(self.fixture["pred_cam_R"], self.fixture["pred_cam_T"])
        world = transform_point(poses[0], self.fixture["optical_axis_camera"])
        for actual, expected in zip(world, self.fixture["expected_optical_axis_world"]):
            self.assertAlmostEqual(actual, expected)
        k = {
            "fx_px": self.fixture["img_focal"],
            "fy_px": self.fixture["img_focal"],
            "cx_px": self.fixture["img_center"][0],
            "cy_px": self.fixture["img_center"][1],
            "skew_px": 0.0,
        }
        u, v = project_world_point(world, poses[0], k)
        self.assertAlmostEqual(u, k["cx_px"])
        self.assertAlmostEqual(v, k["cy_px"])

    def test_camera_space_joints_match_independent_gauge(self) -> None:
        _t_obs, poses = convert_tram_cameras(self.fixture["pred_cam_R"], self.fixture["pred_cam_T"])
        lifted = lift_joints_to_world(self.fixture["joints_camera"][0], poses[0])
        for joint, camera in zip(lifted, self.fixture["joints_camera"][0]):
            expected = _apply_f(camera)
            for actual, value in zip(joint, expected):
                self.assertAlmostEqual(actual, value)
        lifted1 = lift_joints_to_world(self.fixture["joints_camera"][1], poses[1])
        shifted = [
            self.fixture["joints_camera"][1][0][0] + 0.25,
            self.fixture["joints_camera"][1][0][1],
            self.fixture["joints_camera"][1][0][2],
        ]
        for actual, expected in zip(lifted1[0], _apply_f(shifted)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(lifted[0], self.fixture["expected_pelvis_world"][0]):
            self.assertAlmostEqual(actual, expected)

    def test_native_world_then_gauge_equals_pose_lift(self) -> None:
        t_obs, poses = convert_tram_cameras(self.fixture["pred_cam_R"], self.fixture["pred_cam_T"])
        camera_joint = self.fixture["joints_camera"][1][0]
        native = [
            camera_joint[0] + self.fixture["pred_cam_T"][1][0],
            camera_joint[1] + self.fixture["pred_cam_T"][1][1],
            camera_joint[2] + self.fixture["pred_cam_T"][1][2],
        ]
        gauged = transform_point(t_obs, native)
        lifted = transform_point(poses[1], camera_joint)
        for actual, expected in zip(gauged, lifted):
            self.assertAlmostEqual(actual, expected)

    def test_rest_pose_has_smpl24_and_head_above_pelvis_in_world(self) -> None:
        self.assertEqual(len(rest_smpl24_camera()), 24)
        _t_obs, poses = convert_tram_cameras(self.fixture["pred_cam_R"], self.fixture["pred_cam_T"])
        world = lift_joints_to_world(self.fixture["joints_camera"][0], poses[0])
        self.assertGreater(world[15][1], world[0][1])
        self.assertLess(world[7][1], world[0][1])


class HumansContractTest(unittest.TestCase):
    def test_humans_v1_rejects_wrong_joint_count(self) -> None:
        with self.assertRaisesRegex(ContractError, "24"):
            validate_humans_v1(
                {
                    "joint_layout": "smpl24",
                    "coordinate_frame": "observation_world",
                    "people": [
                        {
                            "id": "human0",
                            "samples": [
                                {
                                    "sample_index": 0,
                                    "root": [0.0, 0.0, -1.0],
                                    "joints": [[0.0, 0.0, -1.0]] * 23,
                                }
                            ],
                        }
                    ],
                }
            )

    def test_root_must_match_pelvis(self) -> None:
        joints = [[float(index), 0.0, -1.0] for index in range(24)]
        payload = {
            "joint_layout": "smpl24",
            "coordinate_frame": "observation_world",
            "people": [
                {
                    "id": "human0",
                    "samples": [
                        {
                            "sample_index": 0,
                            "root": [9.0, 0.0, -1.0],
                            "joints": joints,
                        }
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(ContractError, "pelvis"):
            validate_humans_v1(payload)

    def test_attach_keeps_core_observation_fields(self) -> None:
        fixture = load_tram_c2w_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            observation = observation_from_tram_native(
                native_from_fixture(fixture),
                Path(tmp),
                video_sha256="a" * 64,
                scale={
                    "status": "metric_assumed",
                    "meters_per_world_unit": 1.0,
                    "source": "tram_c2w_fixture",
                },
                method="tram_c2w_fixture",
            )
        core_before = {key: observation[key] for key in observation if key != "extensions"}
        humans = observation["extensions"][HUMANS_EXTENSION]
        attached = attach_humans(observation, humans)
        validate_scene_observation(attached)
        core_after = {key: attached[key] for key in attached if key != "extensions"}
        self.assertEqual(core_before["cameras"], core_after["cameras"])
        self.assertEqual(core_before["timeline"], core_after["timeline"])
        self.assertIn(HUMANS_EXTENSION, attached["extensions"])


class TramAdapterTest(unittest.TestCase):
    def test_fixture_adapter_writes_valid_observation(self) -> None:
        adapter = TramHumanAdapter()
        request = HumanReconstructionRequest(options=normalize_human_options({"source": "fixture"}))
        with tempfile.TemporaryDirectory() as tmp:
            output = adapter.reconstruct_humans(request, Path(tmp))
        validate_scene_observation(output.observation)
        humans = output.observation["extensions"][HUMANS_EXTENSION]
        validate_humans_v1(humans, sample_count=2)
        first = output.observation["cameras"][0]["poses"][0]["T_world_camera"]
        for actual, expected in zip(first, FIRST_CAMERA_WORLD_FROM_OPENCV):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(output.observation["coordinates"]["scale"]["status"], "metric_assumed")

    def test_walk_fixture_has_strictly_increasing_time(self) -> None:
        native = synthetic_walk_native(n_frames=12, image_size=(640, 480), fps=24.0)
        self.assertEqual(len(native["tracks"][0]["joints_camera"]), 12)
        adapter = TramHumanAdapter()
        request = HumanReconstructionRequest(
            options=normalize_human_options({"source": "walk_fixture", "walk_frames": 12})
        )
        with tempfile.TemporaryDirectory() as tmp:
            observation = adapter.reconstruct_humans(request, Path(tmp)).observation
        times = [sample["timestamp_s"] for sample in observation["timeline"]["samples"]]
        self.assertEqual(times, [index / 24.0 for index in range(12)])

    def test_attach_to_parent_uses_parent_poses(self) -> None:
        fixture = load_tram_c2w_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp) / "parent"
            parent_dir.mkdir()
            parent = observation_from_tram_native(
                native_from_fixture(fixture),
                parent_dir,
                video_sha256="b" * 64,
                scale={"status": "relative", "meters_per_world_unit": None, "source": "da3"},
                method="parent",
            )
            adapter = TramHumanAdapter()
            attached = adapter.reconstruct_humans(
                HumanReconstructionRequest(
                    options=normalize_human_options({"source": "fixture"}),
                    parent_observation=parent,
                ),
                Path(tmp) / "child",
            ).observation
        self.assertEqual(attached["cameras"], parent["cameras"])
        self.assertEqual(attached["coordinates"]["scale"]["status"], "relative")
        self.assertIn("phystwin.humans.v1", attached["extensions"])

    def test_missing_tram_dir_is_explicit(self) -> None:
        adapter = TramHumanAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TramUnavailableError) as raised:
                adapter.reconstruct_humans(
                    HumanReconstructionRequest(
                        options=normalize_human_options(
                            {"source": "tram_dir", "tram_dir": str(Path(tmp) / "missing")}
                        )
                    ),
                    Path(tmp),
                )
        self.assertIn("SMPL", str(raised.exception))
        self.assertIn("MIT", TRAM_UNAVAILABLE)

    def test_human_cache_key_ignores_paths(self) -> None:
        descriptor = make_descriptor()
        first = HumanReconstructionRequest(options=normalize_human_options({"source": "fixture"}))
        same = HumanReconstructionRequest(options=normalize_human_options({"source": "fixture"}))
        self.assertEqual(human_cache_key(descriptor, first), human_cache_key(descriptor, same))
        other = HumanReconstructionRequest(
            options=normalize_human_options({"source": "walk_fixture"})
        )
        self.assertNotEqual(human_cache_key(descriptor, first), human_cache_key(descriptor, other))

    def test_publish_and_skeleton_video(self) -> None:
        adapter = TramHumanAdapter()
        request = HumanReconstructionRequest(options=normalize_human_options({"source": "fixture"}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = humans_cache_entry(root, "p2test")
            observation = publish_observation(
                entry,
                lambda work_dir: adapter.reconstruct_humans(request, work_dir).observation,
            )
            self.assertTrue(is_complete(entry))
            video = write_projected_skeleton_video(root / "body.mp4", observation, fps=24.0)
            self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
