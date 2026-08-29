"""Helpers for the local UI server. Does not start SAM 2."""

from __future__ import annotations

import unittest
from pathlib import Path

import serve


class ServeHelpersTest(unittest.TestCase):
    def test_find_phystwin(self) -> None:
        path = serve.find_phystwin()
        self.assertTrue(path.is_file(), path)

    def test_list_samples_records_kind(self) -> None:
        samples = {item["id"]: item for item in serve.list_samples()}
        bounce = Path("samples/bounce.mp4")
        if bounce.is_file():
            self.assertIn("bounce", samples)
            self.assertEqual(samples["bounce"]["kind"], "recorded")
            self.assertEqual(samples["bounce"]["suggested_point"], [375.0, 722.0])

    def test_physics_routes_are_registered(self) -> None:
        paths = {route.path for route in serve.app.routes}
        self.assertIn("/api/physics-fixture", paths)
        self.assertIn("/api/physics-fit-fixture", paths)


if __name__ == "__main__":
    unittest.main()
