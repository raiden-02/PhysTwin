"""Focused standard-library checks for the P0 3D contract boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.adapter import (  # noqa: E402
    EstimatorDescriptor,
    ReconstructionRequest,
    VideoInput,
    reconstruction_cache_key,
)
from vision.reconstruction.contracts import (  # noqa: E402
    ContractError,
    canonical_json_bytes,
    load_contract,
    opencv_camera_to_three_world,
    project_world_point,
    validate_physical_scene,
    validate_scene_observation,
)


EXAMPLES = ROOT / "contracts" / "3d" / "v1" / "examples"


class ReconstructionContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = dict(load_contract(EXAMPLES / "scene_observation.json"))
        self.physical_scene = dict(load_contract(EXAMPLES / "physical_scene.json"))

    def test_examples_round_trip_without_semantic_change(self) -> None:
        encoded = canonical_json_bytes(self.observation)
        decoded = json.loads(encoded)
        validate_scene_observation(decoded)
        self.assertEqual(decoded, self.observation)
        validate_physical_scene(self.physical_scene)
        observation_hash = hashlib.sha256(
            (EXAMPLES / "scene_observation.json").read_bytes()
        ).hexdigest()
        self.assertEqual(
            observation_hash,
            self.physical_scene["observation_alignment"]["observation_sha256"],
        )

    def test_timeline_and_intrinsics_invariants(self) -> None:
        duplicate_time = copy.deepcopy(self.observation)
        duplicate_time["timeline"]["samples"][1]["timestamp_s"] = 0.0
        with self.assertRaisesRegex(ContractError, "strictly increasing"):
            validate_scene_observation(duplicate_time)

        invalid_intrinsics = copy.deepcopy(self.observation)
        invalid_intrinsics["cameras"][0]["intrinsics"]["fx_px"] = 0.0
        with self.assertRaisesRegex(ContractError, "must be > 0"):
            validate_scene_observation(invalid_intrinsics)

    def test_transform_and_artifact_invariants(self) -> None:
        scaled_pose = copy.deepcopy(self.observation)
        scaled_pose["cameras"][0]["poses"][1]["T_world_camera"][0] = 2.0
        with self.assertRaisesRegex(ContractError, "not unit length"):
            validate_scene_observation(scaled_pose)

        dangling_geometry = copy.deepcopy(self.observation)
        dangling_geometry["static_scene"]["geometry"][0]["artifact_id"] = "missing"
        with self.assertRaisesRegex(ContractError, "does not reference an artifact"):
            validate_scene_observation(dangling_geometry)

    def test_first_camera_projection_and_three_conversion(self) -> None:
        camera = self.observation["cameras"][0]
        first_pose = camera["poses"][0]["T_world_camera"]
        projected = project_world_point(
            [0.0, 0.0, -2.0],
            first_pose,
            camera["intrinsics"],
        )
        self.assertAlmostEqual(projected[0], camera["intrinsics"]["cx_px"])
        self.assertAlmostEqual(projected[1], camera["intrinsics"]["cy_px"])

        three_transform = opencv_camera_to_three_world(first_pose)
        identity = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        for actual, expected in zip(three_transform, identity):
            self.assertAlmostEqual(actual, expected)

    def test_unresolved_scale_cannot_be_executable(self) -> None:
        scene = copy.deepcopy(self.physical_scene)
        scene["execution"] = {
            "status": "executable",
            "blockers": [],
            "backend": "example",
            "fixed_step_s": 1.0 / 120.0,
        }
        with self.assertRaisesRegex(ContractError, "metric scale is required"):
            validate_physical_scene(scene)

    def test_cache_key_is_deterministic_and_path_independent(self) -> None:
        descriptor = EstimatorDescriptor(
            adapter="example",
            adapter_version="1.0.0",
            model="example-model",
            model_revision="abc123",
            weights_sha256="a" * 64,
        )
        first = ReconstructionRequest(
            inputs=(VideoInput("video0", Path("one.mp4"), "b" * 64),),
            options={"precision": "fp16", "nested": {"b": 2, "a": 1}},
        )
        same_identity = ReconstructionRequest(
            inputs=(VideoInput("video0", Path("different-location.mp4"), "b" * 64),),
            options={"nested": {"a": 1, "b": 2}, "precision": "fp16"},
        )
        key = reconstruction_cache_key(descriptor, first)
        self.assertEqual(key, reconstruction_cache_key(descriptor, same_identity))
        self.assertEqual(len(key), 64)

        changed_revision = replace(descriptor, model_revision="def456")
        self.assertNotEqual(key, reconstruction_cache_key(changed_revision, first))

        changed_source = ReconstructionRequest(
            inputs=(VideoInput("video0", Path("one.mp4"), "c" * 64),),
            options=first.options,
        )
        self.assertNotEqual(key, reconstruction_cache_key(descriptor, changed_source))


if __name__ == "__main__":
    unittest.main()
