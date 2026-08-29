"""CPU tests for known-radius sphere geometry."""

from __future__ import annotations

import math
import unittest

import numpy as np

from vision.reconstruction.sphere import (
    apparent_radius_from_mask,
    opencv_to_first_camera_graphics,
    reconstruct_metric_ball,
    sphere_center_depth_m,
    unproject_opencv,
)


class SphereGeometryTest(unittest.TestCase):
    def test_silhouette_depth_is_not_crude_fr_over_r(self) -> None:
        radius_m = 0.11
        focal = 1800.0
        depth = 1.40
        radius_px = focal * radius_m / math.sqrt(depth * depth - radius_m * radius_m)
        recovered = sphere_center_depth_m(radius_px, focal, radius_m)
        crude = focal * radius_m / radius_px
        self.assertAlmostEqual(recovered, depth, places=9)
        self.assertGreater(abs(crude - depth), 0.004)

    def test_mask_horizontal_radius_survives_vertical_blur(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        center = (40, 40)
        radius = 12
        for y in range(80):
            for x in range(80):
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius:
                    mask[y, x] = True
        sharp = apparent_radius_from_mask(mask)
        self.assertTrue(sharp["accepted"])
        self.assertAlmostEqual(float(sharp["radius_px"]), radius, delta=1.0)
        blurred = mask.copy()
        blurred[20:28, 28:52] = True
        blurred[52:60, 28:52] = True
        smeared = apparent_radius_from_mask(blurred)
        self.assertTrue(smeared["accepted"])
        self.assertLess(abs(float(smeared["radius_horizontal_px"]) - radius), 2.5)
        self.assertGreater(float(smeared["radius_area_px"]), float(smeared["radius_horizontal_px"]))

    def test_tiny_mask_is_rejected(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        measured = apparent_radius_from_mask(mask)
        self.assertFalse(measured["accepted"])

    def test_unproject_and_graphics_basis(self) -> None:
        K = {"fx_px": 1000.0, "fy_px": 1000.0, "cx_px": 320.0, "cy_px": 240.0}
        camera = unproject_opencv(320.0, 240.0, 2.0, K)
        self.assertAlmostEqual(camera[0], 0.0)
        self.assertAlmostEqual(camera[1], 0.0)
        self.assertAlmostEqual(camera[2], 2.0)
        graphics = opencv_to_first_camera_graphics(camera)
        self.assertEqual(graphics, [0.0, 0.0, -2.0])

    def test_reconstruct_recovers_synthetic_on_axis_sphere(self) -> None:
        radius_m = 0.11
        depth = 1.25
        fx = 1600.0
        fy = 1600.0
        cx = 200.0
        cy = 180.0
        radius_px = fx * radius_m / math.sqrt(depth * depth - radius_m * radius_m)
        mask = np.zeros((360, 400), dtype=bool)
        yy, xx = np.ogrid[:360, :400]
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2] = True
        lifted = reconstruct_metric_ball(
            mask,
            radius_m=radius_m,
            intrinsics={"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy},
        )
        self.assertTrue(lifted["accepted"])
        self.assertAlmostEqual(float(lifted["depth_m"]), depth, delta=0.02)
        self.assertAlmostEqual(lifted["position_m"][0], 0.0, delta=0.01)
        self.assertAlmostEqual(lifted["position_m"][1], 0.0, delta=0.01)
        self.assertAlmostEqual(lifted["position_m"][2], -depth, delta=0.02)


if __name__ == "__main__":
    unittest.main()
