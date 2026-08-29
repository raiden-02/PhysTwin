"""FastAPI route that runs P4 through the isolated physics subprocess."""

from __future__ import annotations

import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from reconstruction.contracts import (
    load_contract,
    validate_inverse_fit_artifacts,
    validate_inverse_physics_fit,
    validate_rollout_source,
)
from reconstruction.footage import REQUESTED_CLIP, inspect_local_footage


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

    @app.post("/api/physics-fit-fixture")
    def run_physics_fit_fixture() -> dict:
        physics_python = root / ".venv-physics" / "Scripts" / "python.exe"
        scene_path = (
            root
            / "contracts"
            / "3d"
            / "v1"
            / "examples"
            / "physical_scene_tether_fit_template.json"
        )
        if not physics_python.is_file():
            raise HTTPException(400, "missing .venv-physics. Run scripts\\setup-physics.ps1.")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "another GPU stage is already running")
        try:
            output_dir = jobs_root / f"physics-fit-{uuid.uuid4().hex[:12]}"
            completed = subprocess.run(
                [
                    str(physics_python),
                    "-m",
                    "physics3d.fit_physical_scene",
                    str(scene_path),
                    "--fixture",
                    "--output",
                    str(output_dir),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=360,
                check=False,
            )
            if completed.returncode != 0:
                detail = (
                    completed.stderr
                    or completed.stdout
                    or "physics fit subprocess failed"
                ).strip()
                raise HTTPException(400, detail)
            report = dict(load_contract(output_dir / "inverse_physics_fit.json"))
            motion = dict(load_contract(output_dir / "target_motion_observation.json"))
            rollout = dict(load_contract(output_dir / "simulated_world_state.json"))
            scene = dict(load_contract(output_dir / "fitted_physical_scene.json"))
            template = dict(load_contract(scene_path))
            validate_inverse_physics_fit(report)
            validate_rollout_source(rollout, scene)
            validate_inverse_fit_artifacts(
                report,
                template,
                motion,
                scene,
                rollout,
                fitted_scene_path=output_dir / "fitted_physical_scene.json",
                rollout_path=output_dir / "simulated_world_state.json",
            )
            return {
                "fit": report,
                "motion_observation": motion,
                "physical_scene": scene,
                "rollout": rollout,
                "stdout": completed.stdout,
            }
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504, "P5 physics fit exceeded 360 seconds") from error
        finally:
            run_lock.release()

    @app.get("/api/physics-real-fit")
    def get_physics_real_fit() -> dict:
        footage = inspect_local_footage(root)
        saved = _saved_real_fit(root)
        status = saved["fit"]["status"] if saved and saved.get("fit") else footage["status"]
        if footage["status"] == "AWAITING_FOOTAGE" and not saved:
            status = "AWAITING_FOOTAGE"
        return {
            "status": status,
            "footage": footage,
            "requested_clip": REQUESTED_CLIP,
            **(saved or {}),
        }

    @app.post("/api/physics-real-fit")
    def run_physics_real_fit() -> dict:
        footage = inspect_local_footage(root)
        saved = _saved_real_fit(root)
        status = saved["fit"]["status"] if saved and saved.get("fit") else footage["status"]
        if footage["status"] == "AWAITING_FOOTAGE" and not saved:
            status = "AWAITING_FOOTAGE"
        blockers = []
        if footage["status"] == "AWAITING_FOOTAGE" and not saved:
            blockers = [
                "No eligible clip with a measured length is available. "
                "The inspect path does not invent metric scale or run Newton on ineligible footage."
            ]
        return {
            "status": status,
            "footage": footage,
            "requested_clip": REQUESTED_CLIP,
            **(saved or {"fit": None}),
            "blockers": blockers,
        }


def _saved_real_fit(root: Path) -> dict | None:
    output = root / "results" / "physics3d" / "p5r-real-fit"
    report_path = output / "inverse_physics_fit.json"
    if not report_path.is_file():
        return None
    try:
        report = dict(load_contract(report_path))
    except Exception:
        return None
    payload: dict = {"fit": report}
    motion_path = output / "target_motion_observation.json"
    scene_path = output / "fitted_physical_scene.json"
    rollout_path = output / "simulated_world_state.json"
    if motion_path.is_file() and scene_path.is_file() and rollout_path.is_file():
        try:
            payload["motion_observation"] = dict(load_contract(motion_path))
            payload["physical_scene"] = dict(load_contract(scene_path))
            payload["rollout"] = dict(load_contract(rollout_path))
        except Exception:
            return {"fit": report}
    return payload
