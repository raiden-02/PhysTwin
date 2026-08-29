"""Depth Anything 3 adapter. Estimator-specific code stays in this module."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import (
    EstimatorDescriptor,
    ReconstructionOutput,
    ReconstructionRequest,
)
from .cache import sha256_file
from .contracts import FIRST_CAMERA_WORLD_FROM_OPENCV, ContractError
from .geometry import write_point_cloud_glb
from .transforms import canonical_poses_from_da3_w2c, da3_w2c_to_c2w, scale_intrinsics
from .video import choose_source_frames, read_video_meta, sample_video_frames, sha256_file as video_sha256

ADAPTER_NAME = "da3"
ADAPTER_VERSION = "1.0.0"
PACKAGE_NAME = "depth-anything-3"
PACKAGE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PACKAGE_LICENSE = "Apache-2.0"
MODEL_ID = "depth-anything/DA3-BASE"
MODEL_REVISION = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
WEIGHTS_LICENSE = "Apache-2.0"
MODEL_HUB_FILE = "model.safetensors"

DEFAULT_OPTIONS: dict[str, Any] = {
    "start_s": 0.0,
    "duration_s": 2.0,
    "max_frames": 12,
    "process_res": 504,
    "process_res_method": "upper_bound_resize",
    "ref_view_strategy": "middle",
    "use_ray_pose": False,
    "conf_percentile": 40.0,
    "max_points": 250000,
}


def normalize_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_OPTIONS)
    if options:
        merged.update(dict(options))
    merged["start_s"] = float(merged["start_s"])
    duration = merged["duration_s"]
    merged["duration_s"] = None if duration is None else float(duration)
    merged["max_frames"] = int(merged["max_frames"])
    merged["process_res"] = int(merged["process_res"])
    merged["conf_percentile"] = float(merged["conf_percentile"])
    merged["max_points"] = int(merged["max_points"])
    merged["use_ray_pose"] = bool(merged["use_ray_pose"])
    if merged["max_frames"] <= 0:
        raise ValueError("max_frames must be > 0")
    if merged["max_points"] <= 0:
        raise ValueError("max_points must be > 0")
    return merged


def resolve_weights_sha256() -> str:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_HUB_FILE,
            revision=MODEL_REVISION,
        )
    )
    return sha256_file(path)


def make_descriptor(weights_sha256: str | None = None) -> EstimatorDescriptor:
    return EstimatorDescriptor(
        adapter=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        weights_sha256=weights_sha256,
    )


class Da3ReconstructionAdapter:
    """Run pinned DA3-BASE and emit a canonical SceneObservation."""

    def __init__(self, weights_sha256: str | None = None) -> None:
        self._weights_sha256 = weights_sha256

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return make_descriptor(self._weights_sha256)

    def reconstruct(
        self,
        request: ReconstructionRequest,
        work_dir: Path,
    ) -> ReconstructionOutput:
        if len(request.inputs) != 1:
            raise ValueError("P1 DA3 adapter accepts exactly one video")
        options = normalize_options(request.options)
        source = request.inputs[0]
        fps, width, height, n_frames = read_video_meta(source.path)
        source_frames = choose_source_frames(
            n_frames,
            fps,
            start_s=options["start_s"],
            duration_s=options["duration_s"],
            max_frames=options["max_frames"],
        )
        frames = sample_video_frames(source.path, source_frames)
        images = [cv2_to_rgb(item.image_bgr) for item in frames]
        if source.sha256 != video_sha256(source.path):
            raise ValueError("video content hash does not match the request")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DA3 reconstruction requires CUDA")
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction, weights_sha256 = _run_da3(images, options)
        wall_seconds = time.perf_counter() - started
        peak_gpu = int(torch.cuda.max_memory_allocated())
        self._weights_sha256 = weights_sha256

        if prediction.extrinsics is None or prediction.intrinsics is None:
            raise ContractError("DA3 returned no camera poses or intrinsics")
        if prediction.depth is None or prediction.processed_images is None:
            raise ContractError("DA3 returned no depth or processed images")

        t_obs_from_native, poses = canonical_poses_from_da3_w2c(prediction.extrinsics)
        processed_h, processed_w = prediction.depth.shape[1], prediction.depth.shape[2]
        first_k = scale_intrinsics(
            prediction.intrinsics[0],
            source_size=(width, height),
            processed_size=(processed_w, processed_h),
        )
        vary = _intrinsics_vary(prediction.intrinsics)
        artifact_rel = Path("artifacts") / "scene.glb"
        artifact_path = work_dir / artifact_rel
        points, colors = _canonical_point_cloud(
            prediction,
            t_obs_from_native,
            conf_percentile=options["conf_percentile"],
            max_points=options["max_points"],
        )
        write_point_cloud_glb(artifact_path, points, colors)

        samples = []
        camera_poses = []
        for index, (frame, pose) in enumerate(zip(frames, poses)):
            samples.append(
                {
                    "sample_index": index,
                    "timestamp_s": frame.timestamp_s,
                    "source_frames": {source.id: frame.source_frame},
                }
            )
            camera_poses.append(
                {
                    "sample_index": index,
                    "T_world_camera": [float(value) for value in pose],
                }
            )

        observation = {
            "schema": "phystwin.scene_observation",
            "version": 1,
            "observation_id": f"da3-{source.id}-{source.sha256[:12]}",
            "timeline": {
                "time_unit": "second",
                "origin": "observation_start",
                "samples": samples,
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
                    "id": source.id,
                    "kind": "video",
                    "uri": source.path.name,
                    "media_type": "video/mp4",
                    "sha256": source.sha256,
                }
            ],
            "artifacts": [
                {
                    "id": "scene_geometry",
                    "uri": artifact_rel.as_posix(),
                    "media_type": "model/gltf-binary",
                    "sha256": sha256_file(artifact_path),
                }
            ],
            "cameras": [
                {
                    "id": "camera0",
                    "source_id": source.id,
                    "projection": "pinhole",
                    "lens_distortion": "unknown",
                    "image_size_px": [width, height],
                    "intrinsics": first_k,
                    "poses": camera_poses,
                }
            ],
            "static_scene": {
                "geometry": [
                    {
                        "kind": "point_cloud",
                        "artifact_id": "scene_geometry",
                    }
                ]
            },
            "provenance": {
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "producer": {
                    "adapter": ADAPTER_NAME,
                    "adapter_version": ADAPTER_VERSION,
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "weights_sha256": weights_sha256,
                    "package": PACKAGE_NAME,
                    "package_revision": PACKAGE_REVISION,
                    "license": {
                        "code": PACKAGE_LICENSE,
                        "weights": WEIGHTS_LICENSE,
                    },
                },
                "options": options,
                "runtime": {
                    "wall_seconds": wall_seconds,
                    "device": device_name,
                    "peak_gpu_memory_bytes": peak_gpu,
                    "n_frames": len(frames),
                    "source_frame_count": n_frames,
                    "fps_metadata": fps,
                },
            },
            "extensions": {
                "phystwin.da3.v1": {
                    "native_extrinsics": "opencv_w2c",
                    "processed_image_size_px": [processed_w, processed_h],
                    "intrinsics_vary": vary,
                    "timestamp_source": frames[0].timestamp_source,
                    "lens_distortion": "unknown",
                    "selected_source_frames": [item.source_frame for item in frames],
                    "first_camera_world_from_opencv": list(FIRST_CAMERA_WORLD_FROM_OPENCV),
                    "point_count": int(points.shape[0]),
                }
            },
        }
        runtime = {
            "wall_seconds": wall_seconds,
            "device": device_name,
            "peak_gpu_memory_bytes": peak_gpu,
            "weights_sha256": weights_sha256,
            "n_frames": len(frames),
        }
        return ReconstructionOutput(observation=observation, runtime=runtime)


def cv2_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr[:, :, ::-1].copy()


def _run_da3(images: list[np.ndarray], options: Mapping[str, Any]):
    import torch
    from depth_anything_3.api import DepthAnything3
    from huggingface_hub import hf_hub_download

    weights_path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_HUB_FILE,
            revision=MODEL_REVISION,
        )
    )
    weights_sha256 = sha256_file(weights_path)
    model = DepthAnything3.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to("cuda")
    model.eval()
    try:
        prediction = model.inference(
            images,
            process_res=options["process_res"],
            process_res_method=options["process_res_method"],
            ref_view_strategy=options["ref_view_strategy"],
            use_ray_pose=options["use_ray_pose"],
        )
    finally:
        del model
        torch.cuda.empty_cache()
    return prediction, weights_sha256


def _intrinsics_vary(intrinsics: np.ndarray, rel_tol: float = 1e-3) -> bool:
    first = intrinsics[0]
    for matrix in intrinsics[1:]:
        if not np.allclose(matrix, first, rtol=rel_tol, atol=1e-3):
            return True
    return False


def _canonical_point_cloud(
    prediction: Any,
    t_obs_from_native: tuple[float, ...],
    *,
    conf_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = prediction.depth
    images = prediction.processed_images
    extrinsics = prediction.extrinsics
    intrinsics = prediction.intrinsics
    conf = prediction.conf
    n, height, width = depth.shape
    if conf is not None:
        threshold = float(np.percentile(conf[np.isfinite(conf)], conf_percentile))
    else:
        threshold = 0.0

    us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for index in range(n):
        z = depth[index]
        valid = np.isfinite(z) & (z > 0)
        if conf is not None:
            valid &= conf[index] >= threshold
        if not np.any(valid):
            continue
        k = intrinsics[index]
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        skew = float(k[0, 1])
        c2w = da3_w2c_to_c2w(extrinsics[index])
        yy = (vs[valid] - cy) / fy
        xx = (us[valid] - cx - skew * yy) / fx
        zz = z[valid]
        cam = np.stack([xx * zz, yy * zz, zz], axis=1)
        ones = np.ones((cam.shape[0], 1), dtype=np.float64)
        native = (
            np.asarray(c2w, dtype=np.float64).reshape(4, 4)
            @ np.concatenate([cam.astype(np.float64), ones], axis=1).T
        ).T[:, :3]
        obs = (
            np.asarray(t_obs_from_native, dtype=np.float64).reshape(4, 4)
            @ np.concatenate([native, np.ones((native.shape[0], 1))], axis=1).T
        ).T[:, :3]
        pts.append(obs.astype(np.float32))
        cols.append(images[index][valid])

    if not pts:
        raise ContractError("DA3 produced no confident 3D points")
    points = np.concatenate(pts, axis=0)
    colors = np.concatenate(cols, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors
