#!/usr/bin/env python3
"""Local FastAPI process that connects the browser to SAM 2 and phystwin.exe.

Bind is 127.0.0.1 only. No queue, Redis, or extra services.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
VISION = Path(__file__).resolve().parent
sys.path.insert(0, str(VISION))
os.chdir(ROOT)

import track as track_mod  # noqa: E402

JOBS_ROOT = ROOT / "results" / "jobs"
DIST = ROOT / "frontend" / "dist"
HOST = "127.0.0.1"
PORT = 8765

SAMPLE_SPECS = [
    {
        "id": "bounce",
        "path": ROOT / "samples" / "bounce.mp4",
        "label": "Mixkit tennis bounce (recorded)",
        "kind": "recorded",
        "hint": "Click the ball. For this clip the known working click is 375, 722.",
        "suggested_point": [375.0, 722.0],
        "suggested_pivot": None,
    },
    {
        "id": "generated_diagonal",
        "path": ROOT / "samples" / "generated_diagonal.mp4",
        "label": "Generated diagonal bounce",
        "kind": "rendered",
        "hint": "Rendered pipeline check, not real-footage accuracy. Suggested click 80, 40.",
        "suggested_point": [80.0, 40.0],
        "suggested_pivot": None,
    },
    {
        "id": "generated_drop",
        "path": ROOT / "samples" / "generated_drop.mp4",
        "label": "Generated drop",
        "kind": "rendered",
        "hint": "Rendered pipeline check, not real-footage accuracy. Suggested click 320, 36.",
        "suggested_point": [320.0, 36.0],
        "suggested_pivot": None,
    },
    {
        "id": "pendulum",
        "path": ROOT / "samples" / "recorded" / "pendulum.mp4",
        "label": "Physical pendulum (recorded)",
        "kind": "recorded",
        "hint": (
            "Click the brass bob, then the fixed string pivot. "
            "Known working selections are target 111,858 and pivot 385,92."
        ),
        "suggested_point": [111.0, 858.0],
        "suggested_pivot": [385.0, 92.0],
    },
    {
        "id": "cinematic_swing",
        "path": ROOT / "samples" / "cinematic" / "spiderman_swing.mp4",
        "label": "Spider-Man crane swing (cinematic stress case)",
        "kind": "cinematic",
        "hint": (
            "Approximate stress footage, not physical validation. Select target "
            "875,490 and the lower red crane beacon at 1115,663. Tracked anchor "
            "coverage is 86.5%, but the simple pendulum fit is poor."
        ),
        "suggested_point": [875.0, 490.0],
        "suggested_pivot": [1115.0, 663.0],
    },
]

_jobs_lock = threading.Lock()
_run_lock = threading.Lock()
_jobs: dict[str, dict] = {}


class RunBody(BaseModel):
    x: float
    y: float
    pivot_x: float | None = None
    pivot_y: float | None = None
    anchor_mode: Literal["fixed", "tracked"] = "fixed"
    ground_y: float | None = Field(default=None)


def find_phystwin() -> Path:
    candidates = [
        ROOT / "build" / "Release" / "phystwin.exe",
        ROOT / "build" / "RelWithDebInfo" / "phystwin.exe",
        ROOT / "build" / "phystwin.exe",
        ROOT / "build" / "phystwin",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "phystwin.exe not found. Build first: powershell -ExecutionPolicy Bypass "
        "-File .\\scripts\\build.ps1"
    )


def list_samples() -> list[dict]:
    samples = []
    for spec in SAMPLE_SPECS:
        path: Path = spec["path"]
        if not path.is_file():
            continue
        samples.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "kind": spec["kind"],
                "hint": spec["hint"],
                "suggested_point": spec["suggested_point"],
                "suggested_pivot": spec["suggested_pivot"],
                "filename": path.name,
                "bytes": path.stat().st_size,
            }
        )
    return samples


def _job_or_404(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job


def _snapshot(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "source_name": job["source_name"],
        "kind": job["kind"],
        "hint": job["hint"],
        "suggested_point": job["suggested_point"],
        "suggested_pivot": job["suggested_pivot"],
        "model": job["model"],
        "fps": job["fps"],
        "width": job["width"],
        "height": job["height"],
        "n_frames": job["n_frames"],
        "point": job["point"],
        "pivot": job["pivot"],
        "anchor_mode": job["anchor_mode"],
        "quality": job["quality"],
        "fit_exit": job["fit_exit"],
        "error": job["error"],
        "progress": job["progress"],
    }


def _emit(job: dict, stage: str, **info: object) -> None:
    event = {"stage": stage, "t": time.time(), **info}
    with _jobs_lock:
        job["stage"] = stage
        job["progress"] = {
            "current": info.get("current"),
            "total": info.get("total"),
            "detail": info.get("detail"),
        }
        if stage == "failed":
            job["status"] = "failed"
            job["error"] = str(info.get("error") or "processing failed")
        elif stage == "complete":
            job["status"] = "complete"
        job["events"].append(event)
        waiters = list(job["queues"])
    payload = json.dumps(event)
    for waiter in waiters:
        waiter.put(payload)


def _prepare_job(
    video_path: Path,
    source_name: str,
    kind: str,
    hint: str,
    suggested_point: list[float] | None,
    suggested_pivot: list[float] | None,
    model: Literal["projectile_bounce", "pendulum"],
    job_dir: Path | None = None,
) -> dict:
    fps, width, height, n_frames = track_mod.read_video_meta(video_path)
    if job_dir is None:
        job_dir = JOBS_ROOT / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)
    job_id = job_dir.name
    track_mod.dump_frame0(video_path, job_dir / "frame0.png")
    job = {
        "id": job_id,
        "dir": job_dir,
        "video_path": video_path,
        "source_name": source_name,
        "kind": kind,
        "hint": hint,
        "suggested_point": suggested_point,
        "suggested_pivot": suggested_pivot,
        "model": model,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "status": "ready",
        "stage": "ready",
        "point": None,
        "pivot": None,
        "anchor_mode": "fixed",
        "quality": None,
        "fit_exit": None,
        "error": None,
        "progress": {},
        "events": [],
        "queues": [],
        "lock": threading.Lock(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _run_pipeline(
    job: dict,
    x: float,
    y: float,
    pivot: tuple[float, float] | None,
    anchor_mode: Literal["fixed", "tracked"],
    ground_y: float | None,
) -> None:
    tracking_path = job["dir"] / "tracking.json"
    reconstruction_path = job["dir"] / "reconstruction.json"
    try:
        if not _run_lock.acquire(blocking=False):
            raise RuntimeError(
                "another reconstruction is already running. The GPU tracker handles one clip at a time."
            )
        try:
            exe = find_phystwin()

            def on_progress(stage: str, info: dict) -> None:
                _emit(job, stage, **info)

            _emit(job, "reading_video", detail=job["source_name"])
            track_mod.track(
                video=job["video_path"],
                point=(x, y),
                output=tracking_path,
                checkpoint=ROOT / track_mod.DEFAULT_CHECKPOINT,
                config=track_mod.DEFAULT_CONFIG,
                max_frames=None,
                viz=None,
                keep_frames=None,
                on_progress=on_progress,
                model=job["model"],
                pivot=pivot,
                anchor_mode=anchor_mode,
            )
            _emit(job, "fitting", detail="phystwin fit")
            cmd = [str(exe), "fit", str(tracking_path), "--output", str(reconstruction_path)]
            if ground_y is not None:
                cmd.extend(["--ground-y", str(ground_y)])
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            log_path = job["dir"] / "fit.log"
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            job["fit_exit"] = completed.returncode
            if completed.returncode not in (0, 2):
                raise RuntimeError(
                    (completed.stderr or completed.stdout or "phystwin fit failed").strip()
                )
            reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
            quality = reconstruction["metrics"]["quality"]
            job["quality"] = quality
            _emit(
                job,
                "complete",
                quality=quality,
                fit_exit=completed.returncode,
                rmse=reconstruction["metrics"]["rmse"],
            )
        finally:
            _run_lock.release()
    except Exception as exc:
        _emit(job, "failed", error=str(exc))


app = FastAPI(title="PhysTwin local UI", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        f"http://{HOST}:{PORT}",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    exe_ok = True
    exe_path = None
    try:
        exe_path = str(find_phystwin())
    except FileNotFoundError:
        exe_ok = False
    return {
        "ok": True,
        "phystwin": exe_ok,
        "phystwin_path": exe_path,
        "samples": [s["id"] for s in list_samples()],
    }


@app.get("/api/samples")
def samples() -> dict:
    return {"samples": list_samples()}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    model: Literal["projectile_bounce", "pendulum"] = Form(
        default="projectile_bounce"
    ),
) -> dict:
    if file is None and not sample_id:
        raise HTTPException(400, "upload a video or pass sample_id")
    if file is not None and sample_id:
        raise HTTPException(400, "pass either a file or sample_id, not both")

    if sample_id:
        spec = next((s for s in SAMPLE_SPECS if s["id"] == sample_id), None)
        if spec is None or not spec["path"].is_file():
            raise HTTPException(404, f"sample {sample_id} is not on disk")
        job = _prepare_job(
            spec["path"],
            spec["path"].name,
            spec["kind"],
            spec["hint"],
            spec["suggested_point"],
            spec["suggested_pivot"],
            model,
        )
        return _snapshot(job)

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        raise HTTPException(400, "supported video types: mp4, mov, webm, avi, mkv")
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = JOBS_ROOT / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"video{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        job = _prepare_job(
            dest,
            file.filename or dest.name,
            "upload",
            (
                "Click the moving target, then the fixed pivot."
                if model == "pendulum"
                else "Click the moving object on frame 0."
            ),
            None,
            None,
            model,
            job_dir=job_dir,
        )
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return _snapshot(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _snapshot(_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/frame0")
def get_frame0(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    path = job["dir"] / "frame0.png"
    if not path.is_file():
        raise HTTPException(404, "frame 0 is missing")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    path: Path = job["video_path"]
    if not path.is_file():
        raise HTTPException(404, "video is missing")
    return FileResponse(path)


@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str, body: RunBody) -> dict:
    job = _job_or_404(job_id)
    with job["lock"]:
        if job["status"] == "running":
            raise HTTPException(409, "this job is already running")
        if not (0.0 <= body.x <= job["width"] and 0.0 <= body.y <= job["height"]):
            raise HTTPException(
                400,
                f"click ({body.x:.1f}, {body.y:.1f}) is outside "
                f"{job['width']}x{job['height']}",
            )
        pivot = None
        if job["model"] == "pendulum":
            if body.pivot_x is None or body.pivot_y is None:
                raise HTTPException(400, "pendulum mode requires a pivot click")
            if not (
                0.0 <= body.pivot_x <= job["width"]
                and 0.0 <= body.pivot_y <= job["height"]
            ):
                raise HTTPException(
                    400,
                    f"pivot ({body.pivot_x:.1f}, {body.pivot_y:.1f}) is outside "
                    f"{job['width']}x{job['height']}",
                )
            if body.ground_y is not None:
                raise HTTPException(
                    400, "ground y is only valid for Projectile / Bounce"
                )
            pivot = (body.pivot_x, body.pivot_y)
        elif body.anchor_mode != "fixed":
            raise HTTPException(
                400, "tracked anchor mode is only valid for Swing / Pendulum"
            )
        job["status"] = "running"
        job["point"] = [body.x, body.y]
        job["pivot"] = None if pivot is None else [pivot[0], pivot[1]]
        job["anchor_mode"] = body.anchor_mode
        job["error"] = None
        job["quality"] = None
        job["events"] = []
    _emit(job, "queued", detail="starting local SAM 2 + phystwin fit")
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job, body.x, body.y, pivot, body.anchor_mode, body.ground_y),
        daemon=True,
    )
    thread.start()
    return _snapshot(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = _job_or_404(job_id)
    waiter: queue.Queue[str] = queue.Queue()
    with _jobs_lock:
        history = list(job["events"])
        job["queues"].append(waiter)

    async def gen():
        try:
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in {"complete", "failed"}:
                    return
            while True:
                payload = await asyncio.to_thread(waiter.get)
                yield f"data: {payload}\n\n"
                parsed = json.loads(payload)
                if parsed.get("stage") in {"complete", "failed"}:
                    return
        finally:
            with _jobs_lock:
                if waiter in job["queues"]:
                    job["queues"].remove(waiter)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job["status"] == "failed":
        raise HTTPException(400, job["error"] or "processing failed")
    tracking_path = job["dir"] / "tracking.json"
    reconstruction_path = job["dir"] / "reconstruction.json"
    if not tracking_path.is_file() or not reconstruction_path.is_file():
        raise HTTPException(404, "result is not ready")
    raw_path = tracking_path.with_name("tracking_raw.json")
    timing = None
    if raw_path.is_file():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        timing = {
            "device": raw.get("device"),
            "n_frames": raw.get("n_frames"),
            "skipped_empty_masks": raw.get("skipped_empty_masks"),
            "skipped_anchor_masks": raw.get("skipped_anchor_masks"),
            "anchor_coverage": raw.get("anchor_coverage"),
            "end_to_end_seconds": raw.get("end_to_end_seconds"),
            "end_to_end_fps": raw.get("end_to_end_fps"),
            "timing_includes": raw.get("timing_includes"),
        }
    return {
        "job": _snapshot(job),
        "tracking": json.loads(tracking_path.read_text(encoding="utf-8")),
        "reconstruction": json.loads(reconstruction_path.read_text(encoding="utf-8")),
        "timing": timing,
    }


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="ui")
else:

    @app.get("/")
    def ui_not_built() -> dict:
        return {
            "error": "frontend/dist is missing",
            "fix": r"cd frontend; npm install; npm run build  OR  powershell -File .\scripts\serve-ui.ps1",
        }


def main() -> None:
    import uvicorn

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"PhysTwin UI  http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
