"""Additive FastAPI routes for P1 SceneObservation jobs. V1 /api/jobs is unchanged."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from reconstruction.adapter import ReconstructionRequest, VideoInput, reconstruction_cache_key
from reconstruction.cache import (
    cache_entry,
    humans_cache_entry,
    is_complete,
    load_cached_observation,
    publish_observation,
    sha256_file,
)
from reconstruction.da3 import Da3ReconstructionAdapter, make_descriptor, normalize_options, resolve_weights_sha256
from reconstruction.humans import HumanReconstructionRequest, human_cache_key
from reconstruction.tram import (
    TRAM_UNAVAILABLE,
    TramHumanAdapter,
    TramUnavailableError,
    make_descriptor as make_human_descriptor,
    normalize_human_options,
    write_projected_skeleton_video,
)
from reconstruction.video import read_video_meta

_obs_lock = threading.Lock()
_observations: dict[str, dict[str, Any]] = {}


class ObservationRunBody(BaseModel):
    start_s: float = 0.0
    duration_s: float = 2.0
    max_frames: int = 12


class HumansRunBody(BaseModel):
    tram_dir: str | None = None


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "source_name": job["source_name"],
        "kind": job["kind"],
        "fps": job["fps"],
        "width": job["width"],
        "height": job["height"],
        "n_frames": job["n_frames"],
        "cache_key": job.get("cache_key"),
        "error": job["error"],
        "progress": job["progress"],
    }


def _emit(job: dict[str, Any], stage: str, **info: object) -> None:
    event = {"stage": stage, "t": time.time(), **info}
    with _obs_lock:
        job["stage"] = stage
        job["progress"] = {
            "current": info.get("current"),
            "total": info.get("total"),
            "detail": info.get("detail"),
        }
        if stage == "failed":
            job["status"] = "failed"
            job["error"] = str(info.get("error") or "reconstruction failed")
        elif stage == "complete":
            job["status"] = "complete"
        job["events"].append(event)
        waiters = list(job["queues"])
    payload = json.dumps(event)
    for waiter in waiters:
        waiter.put(payload)


def register_observation_routes(
    app: FastAPI,
    *,
    root: Path,
    jobs_root: Path,
    run_lock: threading.Lock,
    sample_specs: list[dict[str, Any]],
) -> None:
    def _job_or_404(job_id: str) -> dict[str, Any]:
        with _obs_lock:
            job = _observations.get(job_id)
            if job is None:
                raise HTTPException(404, f"unknown observation {job_id}")
            return job

    def _prepare(
        video_path: Path,
        source_name: str,
        kind: str,
        job_dir: Path | None = None,
    ) -> dict[str, Any]:
        fps, width, height, n_frames = read_video_meta(video_path)
        if job_dir is None:
            job_dir = jobs_root / f"obs-{uuid.uuid4().hex[:12]}"
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_dir.name,
            "dir": job_dir,
            "video_path": video_path,
            "source_name": source_name,
            "kind": kind,
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "status": "ready",
            "stage": "ready",
            "cache_key": None,
            "error": None,
            "progress": {},
            "events": [],
            "queues": [],
            "lock": threading.Lock(),
        }
        with _obs_lock:
            _observations[job["id"]] = job
        return job

    def _run(job: dict[str, Any], body: ObservationRunBody) -> None:
        try:
            if not run_lock.acquire(blocking=False):
                raise RuntimeError(
                    "another reconstruction is already running. The GPU handles one clip at a time."
                )
            try:
                _emit(job, "resolving_weights", detail="hashing pinned DA3-BASE weights")
                options = normalize_options(
                    {
                        "start_s": body.start_s,
                        "duration_s": body.duration_s,
                        "max_frames": body.max_frames,
                    }
                )
                weights = resolve_weights_sha256()
                video_hash = sha256_file(job["video_path"])
                request = ReconstructionRequest(
                    inputs=(VideoInput("video0", job["video_path"], video_hash),),
                    options=options,
                )
                key = reconstruction_cache_key(make_descriptor(weights), request)
                job["cache_key"] = key
                entry = cache_entry(root, key)
                if is_complete(entry):
                    _emit(job, "cache_hit", detail=key)
                    observation = load_cached_observation(entry)
                else:
                    _emit(job, "reconstructing", detail="DA3-BASE camera and geometry")
                    adapter = Da3ReconstructionAdapter(weights)
                    observation = publish_observation(
                        entry,
                        lambda work_dir: adapter.reconstruct(request, work_dir).observation,
                    )
                job["entry"] = entry
                (job["dir"] / "scene_observation.json").write_text(
                    json.dumps(observation, indent=2),
                    encoding="utf-8",
                )
                _emit(job, "complete", cache_key=key)
            finally:
                run_lock.release()
        except Exception as exc:
            _emit(job, "failed", error=str(exc))

    @app.post("/api/observations")
    async def create_observation(
        file: UploadFile | None = File(default=None),
        sample_id: str | None = Form(default=None),
    ) -> dict[str, Any]:
        if file is None and not sample_id:
            raise HTTPException(400, "upload a video or pass sample_id")
        if file is not None and sample_id:
            raise HTTPException(400, "pass either a file or sample_id, not both")
        if sample_id:
            spec = next((item for item in sample_specs if item["id"] == sample_id), None)
            if spec is None or not spec["path"].is_file():
                raise HTTPException(404, f"sample {sample_id} is not on disk")
            return _snapshot(_prepare(spec["path"], spec["path"].name, spec["kind"]))

        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
            raise HTTPException(400, "supported video types: mp4, mov, webm, avi, mkv")
        jobs_root.mkdir(parents=True, exist_ok=True)
        job_dir = jobs_root / f"obs-{uuid.uuid4().hex[:12]}"
        job_dir.mkdir(parents=True, exist_ok=True)
        dest = job_dir / f"video{suffix}"
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            return _snapshot(_prepare(dest, file.filename or dest.name, "upload", job_dir=job_dir))
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/observations/{job_id}")
    def get_observation(job_id: str) -> dict[str, Any]:
        return _snapshot(_job_or_404(job_id))

    @app.get("/api/observations/{job_id}/video")
    def observation_video(job_id: str) -> FileResponse:
        job = _job_or_404(job_id)
        path: Path = job["video_path"]
        if not path.is_file():
            raise HTTPException(404, "video is missing")
        return FileResponse(path)

    @app.post("/api/observations/{job_id}/run")
    def run_observation(job_id: str, body: ObservationRunBody | None = None) -> dict[str, Any]:
        job = _job_or_404(job_id)
        request = body or ObservationRunBody()
        with job["lock"]:
            if job["status"] == "running":
                raise HTTPException(409, "this observation is already running")
            job["status"] = "running"
            job["error"] = None
            job["events"] = []
        _emit(job, "queued", detail="starting DA3 reconstruction")
        threading.Thread(target=_run, args=(job, request), daemon=True).start()
        return _snapshot(job)

    @app.get("/api/observations/{job_id}/events")
    async def observation_events(job_id: str) -> StreamingResponse:
        import asyncio

        job = _job_or_404(job_id)
        waiter: queue.Queue[str] = queue.Queue()
        with _obs_lock:
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
                with _obs_lock:
                    if waiter in job["queues"]:
                        job["queues"].remove(waiter)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/observations/{job_id}/result")
    def observation_result(job_id: str) -> dict[str, Any]:
        job = _job_or_404(job_id)
        if job["status"] == "failed":
            raise HTTPException(400, job["error"] or "reconstruction failed")
        entry = job.get("entry")
        if entry is None or not is_complete(entry):
            raise HTTPException(404, "observation is not ready")
        observation = load_cached_observation(entry)
        return {
            "job": _snapshot(job),
            "observation": observation,
            "runtime": observation.get("provenance", {}).get("runtime"),
            "cache_key": job.get("cache_key"),
        }

    @app.get("/api/observations/{job_id}/artifacts/{artifact_id}")
    def observation_artifact(job_id: str, artifact_id: str) -> FileResponse:
        job = _job_or_404(job_id)
        entry = job.get("entry")
        if entry is None:
            raise HTTPException(404, "observation is not ready")
        observation = load_cached_observation(entry)
        artifact = next(
            (item for item in observation["artifacts"] if item["id"] == artifact_id),
            None,
        )
        if artifact is None:
            raise HTTPException(404, f"unknown artifact {artifact_id}")
        path = Path(entry) / artifact["uri"]
        if not path.is_file():
            raise HTTPException(404, "artifact file is missing")
        return FileResponse(path, media_type=artifact.get("media_type") or "application/octet-stream")

    @app.post("/api/human-fixtures")
    def create_human_fixture() -> dict[str, Any]:
        options = normalize_human_options({"source": "walk_fixture", "walk_frames": 12})
        request = HumanReconstructionRequest(options=options)
        key = human_cache_key(make_human_descriptor(), request)
        entry = humans_cache_entry(root, key)
        adapter = TramHumanAdapter()

        def build(work_dir: Path) -> dict[str, Any]:
            observation = adapter.reconstruct_humans(request, work_dir).observation
            video_path = work_dir / "input.mp4"
            write_projected_skeleton_video(video_path, observation, fps=24.0)
            document = dict(observation)
            source = dict(document["sources"][0])
            source["sha256"] = sha256_file(video_path)
            source["uri"] = "input.mp4"
            document["sources"] = [source]
            return document

        try:
            observation = (
                load_cached_observation(entry)
                if is_complete(entry) and (entry / "input.mp4").is_file()
                else publish_observation(entry, build)
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        video = entry / "input.mp4"
        if not video.is_file():
            write_projected_skeleton_video(video, observation, fps=24.0)
        job = _prepare(video, "human_fixture.mp4", "synthetic")
        job["entry"] = entry
        job["cache_key"] = key
        job["status"] = "complete"
        job["stage"] = "complete"
        (job["dir"] / "scene_observation.json").write_text(
            json.dumps(observation, indent=2),
            encoding="utf-8",
        )
        return _snapshot(job)

    @app.post("/api/observations/{job_id}/humans")
    def attach_observation_humans(job_id: str, body: HumansRunBody | None = None) -> dict[str, Any]:
        job = _job_or_404(job_id)
        if job["status"] != "complete" or job.get("entry") is None:
            raise HTTPException(400, "reconstruct the scene before attaching humans")
        request_body = body or HumansRunBody()
        tram_dir = request_body.tram_dir or os.environ.get("PHYSTWIN_TRAM_DIR")
        if not tram_dir:
            raise HTTPException(400, TRAM_UNAVAILABLE)
        parent = load_cached_observation(job["entry"])
        options = normalize_human_options({"source": "tram_dir", "tram_dir": tram_dir})
        request = HumanReconstructionRequest(
            options=options,
            parent_observation=parent,
            video_sha256=sha256_file(job["video_path"]),
        )
        key = human_cache_key(make_human_descriptor(), request)
        entry = humans_cache_entry(root, key)
        adapter = TramHumanAdapter()

        def build(work_dir: Path) -> dict[str, Any]:
            artifacts = Path(job["entry"]) / "artifacts"
            if artifacts.is_dir():
                shutil.copytree(artifacts, work_dir / "artifacts", dirs_exist_ok=True)
            return adapter.reconstruct_humans(request, work_dir).observation

        try:
            if not run_lock.acquire(blocking=False):
                raise RuntimeError("another reconstruction is already running. The GPU handles one clip at a time.")
            try:
                observation = (
                    load_cached_observation(entry)
                    if is_complete(entry)
                    else publish_observation(entry, build)
                )
            finally:
                run_lock.release()
        except TramUnavailableError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        job["entry"] = entry
        job["cache_key"] = key
        (job["dir"] / "scene_observation.json").write_text(
            json.dumps(observation, indent=2),
            encoding="utf-8",
        )
        return _snapshot(job)
