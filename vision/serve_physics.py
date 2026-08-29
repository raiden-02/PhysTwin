"""FastAPI route that runs P4 through the isolated physics subprocess."""

from __future__ import annotations

import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from reconstruction.contracts import load_contract, validate_rollout_source


def register_physics_routes(
    app: FastAPI,
    *,
    root: Path,
    jobs_root: Path,
    run_lock: threading.Lock,
) -> None:
    @app.post("/api/physics-fixture")
    def run_physics_fixture() -> dict:
        physics_python = root / ".venv-physics" / "Scripts" / "python.exe"
        scene_path = root / "contracts" / "3d" / "v1" / "examples" / "physical_scene_tether.json"
        if not physics_python.is_file():
            raise HTTPException(400, "missing .venv-physics. Run scripts\\setup-physics.ps1.")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "another GPU stage is already running")
        try:
            output_dir = jobs_root / f"physics-{uuid.uuid4().hex[:12]}"
            completed = subprocess.run(
                [
                    str(physics_python),
                    "-m",
                    "physics3d.simulate_physical_scene",
                    str(scene_path),
                    "--output",
                    str(output_dir),
                    "--repeat-check",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "physics subprocess failed").strip()
                raise HTTPException(400, detail)
            rollout_path = output_dir / "simulated_world_state.json"
            rollout = dict(load_contract(rollout_path))
            scene = dict(load_contract(scene_path))
            validate_rollout_source(rollout, scene)
            return {
                "physical_scene": scene,
                "rollout": rollout,
                "stdout": completed.stdout,
            }
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504, "P4 physics subprocess exceeded 180 seconds") from error
        finally:
            run_lock.release()
