"""TRAM human adapter. Estimator-specific code stays in this module.

TRAM code is MIT. Official inference still needs DROID-SLAM, Detectron2, VIMO,
and SMPL weights the user must obtain from the SMPL/SMPLify site. This adapter
does not download SMPL and does not invent joints when those files are missing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import EstimatorDescriptor
from .cache import sha256_file
from .contracts import FIRST_CAMERA_WORLD_FROM_OPENCV, ContractError, validate_scene_observation
from .geometry import write_point_cloud_glb
from .humans import (
    HUMANS_EXTENSION,
    HumanReconstructionError,
    HumanReconstructionOutput,
    HumanReconstructionRequest,
    SMPL24_BONES,
    SMPL24_COUNT,
    attach_humans,
    humans_payload,
    lift_joints_to_world,
    person_payload,
    sample_payload,
    validate_humans_v1,
)
from .transforms import (
    canonical_poses_from_tram_c2w,
    transform_point,
)

ADAPTER_NAME = "tram"
ADAPTER_VERSION = "1.0.0"
PACKAGE_NAME = "tram"
PACKAGE_REVISION = "4861c112f3c148201326680a50c9199650da6088"
PACKAGE_LICENSE = "MIT"
MODEL_ID = "yufu-wang/tram"
MODEL_REVISION = PACKAGE_REVISION
WEIGHTS_LICENSE = "SMPL-registration required for live inference"
FIXTURE_NAME = "tram_c2w_fixture.json"

TRAM_UNAVAILABLE = (
    "Live TRAM inference is not installed here. TRAM (MIT, "
    f"{PACKAGE_REVISION}) needs its own Linux/conda environment with "
    "DROID-SLAM, Detectron2, VIMO, and SMPL weights from the SMPL/SMPLify site. "
    "Clone https://github.com/yufu-wang/tram, run its install, then "
    "`python scripts/estimate_camera.py --video <clip>` and "
    "`python scripts/estimate_humans.py --video <clip>`. Import the folder with "
    "`python vision/reconstruct_humans.py --tram-dir <tram>/results/<seq>`. "
    "Or inspect the conversion fixture with "
    "`python vision/reconstruct_humans.py --from-fixture`."
)


class TramUnavailableError(HumanReconstructionError):
    """Official TRAM weights or outputs are not available."""


def make_descriptor(weights_sha256: str | None = None) -> EstimatorDescriptor:
    return EstimatorDescriptor(
        adapter=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        weights_sha256=weights_sha256,
    )


def fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "3d"
        / "v1"
        / "examples"
        / FIXTURE_NAME
    )


def load_tram_c2w_fixture(path: Path | None = None) -> dict[str, Any]:
    target = path or fixture_path()
    return json.loads(target.read_text(encoding="utf-8"))


def normalize_human_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {
        "source": "fixture",
        "tram_dir": None,
        "walk_frames": 12,
        "fps": 24.0,
        "image_size_px": [640, 480],
        "video_codec": "h264_yuv420p",
    }
    if options:
        merged.update(dict(options))
    merged["source"] = str(merged["source"])
    merged["walk_frames"] = int(merged["walk_frames"])
    merged["fps"] = float(merged["fps"])
    size = list(merged["image_size_px"])
    if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
        raise ValueError("image_size_px must be two positive integers")
    merged["image_size_px"] = [int(size[0]), int(size[1])]
    tram_dir = merged["tram_dir"]
    merged["tram_dir"] = None if tram_dir in (None, "") else str(tram_dir)
    if merged["walk_frames"] <= 0:
        raise ValueError("walk_frames must be > 0")
    if merged["fps"] <= 0.0:
        raise ValueError("fps must be > 0")
    return merged


def convert_tram_cameras(
    rotations: Sequence[Sequence[Sequence[float]]],
    translations: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], list[tuple[float, ...]]]:
    """Return (T_obs_from_native, T_world_camera list) from TRAM c2w R,t."""

    return canonical_poses_from_tram_c2w(rotations, translations)


def convert_tram_joints(
    joints_camera: Sequence[Sequence[Sequence[float]]],
    T_world_camera: Sequence[Sequence[float]],
) -> list[list[list[float]]]:
    """Lift per-sample camera-space SMPL24 joints into observation world."""

    if len(joints_camera) != len(T_world_camera):
        raise ContractError("joint samples and camera poses must align")
    return [
        lift_joints_to_world(joints, pose)
        for joints, pose in zip(joints_camera, T_world_camera)
    ]


class TramHumanAdapter:
    """Map TRAM-native cameras and camera-space joints into SceneObservation."""

    def __init__(self, weights_sha256: str | None = None) -> None:
        self._weights_sha256 = weights_sha256

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return make_descriptor(self._weights_sha256)

    def reconstruct_humans(
        self,
        request: HumanReconstructionRequest,
        work_dir: Path,
    ) -> HumanReconstructionOutput:
        options = normalize_human_options(request.options)
        source = options["source"]
        if source == "tram_dir":
            native = load_tram_seq_folder(_require_tram_dir(options["tram_dir"]))
            scale = {
                "status": "metric_assumed",
                "meters_per_world_unit": 1.0,
                "source": "tram_zoedepth",
            }
            method = "tram_seq_folder"
        elif source == "fixture":
            native = native_from_fixture(load_tram_c2w_fixture())
            scale = {
                "status": "metric_assumed",
                "meters_per_world_unit": 1.0,
                "source": "tram_c2w_fixture",
            }
            method = "tram_c2w_fixture"
        elif source == "walk_fixture":
            native = synthetic_walk_native(
                n_frames=options["walk_frames"],
                image_size=tuple(options["image_size_px"]),
                fps=options["fps"],
            )
            scale = {
                "status": "metric_assumed",
                "meters_per_world_unit": 1.0,
                "source": "synthetic_walk_fixture",
            }
            method = "synthetic_walk_fixture"
        else:
            raise HumanReconstructionError(
                f"unsupported human source {source!r}. Use tram_dir, fixture, or walk_fixture."
            )

        parent = request.parent_observation
        if parent is None:
            observation = observation_from_tram_native(
                native,
                work_dir,
                video_sha256=request.video_sha256,
                scale=scale,
                method=method,
            )
        else:
            observation = attach_tram_to_observation(parent, native, method=method)
        validate_scene_observation(observation)
        validate_humans_v1(
            observation["extensions"][HUMANS_EXTENSION],
            sample_count=len(observation["timeline"]["samples"]),
        )
        return HumanReconstructionOutput(
            observation=observation,
            runtime={"method": method, "n_people": len(native["tracks"])},
        )


def _require_tram_dir(raw: str | None) -> Path:
    if not raw:
        raise TramUnavailableError(TRAM_UNAVAILABLE)
    path = Path(raw)
    if not path.is_dir():
        raise TramUnavailableError(f"{TRAM_UNAVAILABLE} Missing folder: {path}")
    return path


def load_tram_seq_folder(seq_folder: Path) -> dict[str, Any]:
    """Read official TRAM `camera.npy` plus per-track `hps/*.npy` joints."""

    camera_path = seq_folder / "camera.npy"
    if not camera_path.is_file():
        raise TramUnavailableError(f"{TRAM_UNAVAILABLE} Missing {camera_path}")
    try:
        camera = np.load(camera_path, allow_pickle=True).item()
    except Exception as error:
        raise TramUnavailableError(f"failed to read TRAM camera.npy: {error}") from error

    # Use pre-align c2w. `world_cam_*` is gravity/floor aligned for their renderer.
    if "pred_cam_R" not in camera or "pred_cam_T" not in camera:
        raise ContractError("TRAM camera.npy must contain pred_cam_R and pred_cam_T")
    rotations = _as_rotations(camera["pred_cam_R"])
    translations = _as_translations(camera["pred_cam_T"])
    img_focal = float(np.asarray(camera["img_focal"]).reshape(-1)[0])
    center = camera.get("img_center")
    if center is None:
        raise ContractError("TRAM camera.npy must contain img_center")
    img_center = [float(center[0]), float(center[1])]
    width = int(round(img_center[0] * 2.0))
    height = int(round(img_center[1] * 2.0))

    tracks = []
    hps_dir = seq_folder / "hps"
    files = sorted(hps_dir.glob("*.npy")) if hps_dir.is_dir() else []
    if not files:
        raise TramUnavailableError(f"{TRAM_UNAVAILABLE} Missing {hps_dir}/*.npy")
    for index, path in enumerate(files):
        pred = np.load(path, allow_pickle=True).item()
        if "pred_j3d" not in pred:
            raise TramUnavailableError(
                f"{path.name} has no pred_j3d. Official TRAM stores SMPL params "
                "and needs the SMPL layer to recover joints. Export pred_j3d "
                "(T, 24, 3) camera-space joints, or use --from-fixture."
            )
        joints = np.asarray(pred["pred_j3d"], dtype=np.float64)
        frames = np.asarray(pred["frame"]).reshape(-1).astype(int)
        if joints.ndim != 3 or joints.shape[1:] != (SMPL24_COUNT, 3):
            raise ContractError(f"{path.name}: pred_j3d must have shape (T, 24, 3)")
        if len(frames) != len(joints):
            raise ContractError(f"{path.name}: frame and pred_j3d length differ")
        tracks.append(
            {
                "id": f"human{index}",
                "track_id": index,
                "frames": frames.tolist(),
                "joints_camera": joints.tolist(),
            }
        )
    return {
        "rotations": rotations,
        "translations": translations,
        "img_focal": img_focal,
        "img_center": img_center,
        "image_size_px": [max(width, 1), max(height, 1)],
        "fps": 24.0,
        "tracks": tracks,
        "source_frames": list(range(len(rotations))),
    }


def native_from_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    joints = fixture["joints_camera"]
    frames = list(range(len(joints)))
    return {
        "rotations": fixture["pred_cam_R"],
        "translations": fixture["pred_cam_T"],
        "img_focal": float(fixture["img_focal"]),
        "img_center": list(fixture["img_center"]),
        "image_size_px": list(fixture["image_size_px"]),
        "fps": float(fixture.get("fps", 24.0)),
        "tracks": [
            {
                "id": "human0",
                "track_id": 0,
                "frames": frames,
                "joints_camera": joints,
            }
        ],
        "source_frames": list(fixture.get("source_frames", frames)),
    }


def rest_smpl24_camera() -> list[list[float]]:
    """Standing SMPL24 offsets in OpenCV camera space. +Y is down."""

    offsets = (
        (0.00, 0.00, 0.00),
        (-0.09, 0.05, 0.00),
        (0.09, 0.05, 0.00),
        (0.00, -0.12, 0.00),
        (-0.09, 0.45, 0.00),
        (0.09, 0.45, 0.00),
        (0.00, -0.24, 0.00),
        (-0.09, 0.85, 0.00),
        (0.09, 0.85, 0.00),
        (0.00, -0.36, 0.00),
        (-0.09, 0.90, 0.08),
        (0.09, 0.90, 0.08),
        (0.00, -0.48, 0.00),
        (-0.08, -0.38, 0.00),
        (0.08, -0.38, 0.00),
        (0.00, -0.64, 0.00),
        (-0.20, -0.38, 0.00),
        (0.20, -0.38, 0.00),
        (-0.45, -0.38, 0.00),
        (0.45, -0.38, 0.00),
        (-0.68, -0.38, 0.00),
        (0.68, -0.38, 0.00),
        (-0.76, -0.38, 0.00),
        (0.76, -0.38, 0.00),
    )
    return [list(item) for item in offsets]


def synthetic_walk_native(
    *,
    n_frames: int,
    image_size: tuple[int, int],
    fps: float,
) -> dict[str, Any]:
    """Deterministic walk in TRAM camera space. Not a live TRAM run."""

    width, height = image_size
    rest = np.asarray(rest_smpl24_camera(), dtype=np.float64)
    joints = []
    for index in range(n_frames):
        phase = 2.0 * np.pi * index / max(n_frames, 1)
        pelvis = np.array(
            [0.18 * np.sin(phase * 0.5), 0.12, 3.4 + 0.05 * np.sin(phase)],
            dtype=np.float64,
        )
        pose = rest.copy()
        swing = 0.16 * np.sin(phase)
        pose[4, 2] += swing
        pose[5, 2] -= swing
        pose[7, 2] += swing * 1.4
        pose[8, 2] -= swing * 1.4
        pose[10, 2] += swing * 1.4
        pose[11, 2] -= swing * 1.4
        joints.append((pose + pelvis).tolist())
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "rotations": [identity for _ in range(n_frames)],
        "translations": [[0.0, 0.0, 0.0] for _ in range(n_frames)],
        "img_focal": float(max(width, height)),
        "img_center": [width / 2.0 - 0.5, height / 2.0 - 0.5],
        "image_size_px": [width, height],
        "fps": fps,
        "tracks": [
            {
                "id": "human0",
                "track_id": 0,
                "frames": list(range(n_frames)),
                "joints_camera": joints,
            }
        ],
        "source_frames": list(range(n_frames)),
    }


def observation_from_tram_native(
    native: Mapping[str, Any],
    work_dir: Path,
    *,
    video_sha256: str | None,
    scale: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    t_obs, poses = convert_tram_cameras(native["rotations"], native["translations"])
    width, height = native["image_size_px"]
    fps = float(native["fps"])
    source_frames = list(native["source_frames"])
    if len(source_frames) != len(poses):
        raise ContractError("TRAM source frames must match camera count")
    timestamps = [index / fps for index in range(len(poses))]
    humans = _humans_from_tracks(native["tracks"], poses, source_frames)
    points, colors = _body_preview_points(humans, poses)
    artifact_dir = work_dir / "artifacts"
    glb_path = artifact_dir / "scene.glb"
    write_point_cloud_glb(glb_path, points, colors)
    source_hash = video_sha256 or ("0" * 64)
    observation = {
        "schema": "phystwin.scene_observation",
        "version": 1,
        "observation_id": f"tram-{method}",
        "timeline": {
            "time_unit": "second",
            "origin": "observation_start",
            "samples": [
                {
                    "sample_index": index,
                    "timestamp_s": timestamps[index],
                    "source_frames": {"video0": int(source_frames[index])},
                }
                for index in range(len(poses))
            ],
        },
        "coordinates": {
            "world_basis": "first_camera_graphics",
            "handedness": "right",
            "camera_convention": "opencv",
            "transform_notation": "T_parent_child",
            "vector_convention": "column",
            "matrix_storage": "row_major",
            "scale": dict(scale),
        },
        "sources": [
            {
                "id": "video0",
                "kind": "video",
                "uri": "input.mp4",
                "media_type": "video/mp4",
                "sha256": source_hash,
            }
        ],
        "artifacts": [
            {
                "id": "scene_geometry",
                "uri": "artifacts/scene.glb",
                "media_type": "model/gltf-binary",
                "sha256": sha256_file(glb_path),
            }
        ],
        "cameras": [
            {
                "id": "camera0",
                "source_id": "video0",
                "projection": "pinhole",
                "lens_distortion": "unknown",
                "image_size_px": [int(width), int(height)],
                "intrinsics": {
                    "fx_px": float(native["img_focal"]),
                    "fy_px": float(native["img_focal"]),
                    "cx_px": float(native["img_center"][0]),
                    "cy_px": float(native["img_center"][1]),
                    "skew_px": 0.0,
                },
                "poses": [
                    {
                        "sample_index": index,
                        "T_world_camera": list(pose),
                    }
                    for index, pose in enumerate(poses)
                ],
            }
        ],
        "static_scene": {
            "geometry": [{"kind": "point_cloud", "artifact_id": "scene_geometry"}]
        },
        "provenance": {
            "producer": {
                "adapter": ADAPTER_NAME,
                "adapter_version": ADAPTER_VERSION,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "weights_sha256": None,
                "license": {"code": PACKAGE_LICENSE, "weights": WEIGHTS_LICENSE},
            },
            "humans": {
                "method": method,
                "t_obs_from_native": list(t_obs),
                "native_camera": "tram_pred_cam_c2w",
            },
        },
        "extensions": {},
    }
    return attach_humans(observation, humans)


def attach_tram_to_observation(
    observation: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Lift TRAM camera-space joints through an existing P1 observation world."""

    camera = observation["cameras"][0]
    poses_by_sample = {
        int(pose["sample_index"]): pose["T_world_camera"] for pose in camera["poses"]
    }
    source_by_sample = {
        int(sample["sample_index"]): int(next(iter(sample["source_frames"].values())))
        for sample in observation["timeline"]["samples"]
    }
    people = []
    for track in native["tracks"]:
        samples = []
        frame_to_joints = {
            int(frame): joints
            for frame, joints in zip(track["frames"], track["joints_camera"])
        }
        for sample_index, source_frame in source_by_sample.items():
            if source_frame not in frame_to_joints or sample_index not in poses_by_sample:
                continue
            samples.append(
                sample_payload(
                    sample_index,
                    lift_joints_to_world(frame_to_joints[source_frame], poses_by_sample[sample_index]),
                )
            )
        if samples:
            people.append(person_payload(track["id"], samples, track_id=int(track["track_id"])))
    if not people:
        raise HumanReconstructionError(
            "TRAM frames do not overlap the observation samples. Run TRAM on the same clip window."
        )
    return attach_humans(
        observation,
        humans_payload(people),
        provenance_extra={
            "method": method,
            "attached_to": observation.get("observation_id"),
            "native_camera": "unused_parent_poses",
            "scale_note": (
                "Joints were lifted through the parent T_world_camera. "
                "DA3 relative scale and TRAM metric camera-space units can disagree."
            ),
        },
    )


def write_projected_skeleton_video(
    path: Path,
    observation: Mapping[str, Any],
    *,
    fps: float | None = None,
) -> Path:
    """Draw the observation-world skeleton back onto OpenCV pixels."""

    import cv2

    humans = validate_humans_v1(
        observation["extensions"][HUMANS_EXTENSION],
        sample_count=len(observation["timeline"]["samples"]),
    )
    camera = observation["cameras"][0]
    width, height = camera["image_size_px"]
    k = camera["intrinsics"]
    poses = {int(pose["sample_index"]): pose["T_world_camera"] for pose in camera["poses"]}
    samples_by_index = {
        int(sample["sample_index"]): sample
        for sample in humans["people"][0]["samples"]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = observation["timeline"]["samples"]
    if fps is not None:
        rate = float(fps)
    elif len(samples) >= 2:
        rate = 1.0 / max(float(samples[1]["timestamp_s"]) - float(samples[0]["timestamp_s"]), 1e-6)
    else:
        rate = 24.0
    frames = []
    for sample in observation["timeline"]["samples"]:
        index = int(sample["sample_index"])
        frame = np.full((int(height), int(width), 3), 28, dtype=np.uint8)
        pose = poses.get(index)
        body = samples_by_index.get(index)
        if pose is not None and body is not None:
            pixels = [_project_world(joint, pose, k) for joint in body["joints"]]
            for start, end in SMPL24_BONES:
                if pixels[start] is None or pixels[end] is None:
                    continue
                cv2.line(frame, pixels[start], pixels[end], (48, 180, 255), 2, cv2.LINE_AA)
            for pixel in pixels:
                if pixel is not None:
                    cv2.circle(frame, pixel, 3, (80, 220, 255), -1, cv2.LINE_AA)
        frames.append(frame)
    _write_h264_yuv420p(path, frames, rate)
    return path


def _write_h264_yuv420p(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write browser-playable H.264. OpenCV mp4v is not a Chrome source."""

    import subprocess

    import imageio_ffmpeg

    if not frames:
        raise HumanReconstructionError("cannot write an empty skeleton video")
    height, width = frames[0].shape[:2]
    if width % 2 or height % 2:
        raise HumanReconstructionError("H.264 yuv420p needs even width and height")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.communicate()[1]
    except Exception:
        process.kill()
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-400:]
        raise HumanReconstructionError(f"ffmpeg failed to write {path}: {detail}")


def _project_world(
    point: Sequence[float],
    T_world_camera: Sequence[float],
    intrinsics: Mapping[str, Any],
) -> tuple[int, int] | None:
    camera = transform_point(_invert(T_world_camera), point)
    if camera[2] <= 1e-6:
        return None
    u = (
        float(intrinsics["fx_px"]) * camera[0] / camera[2]
        + float(intrinsics.get("skew_px", 0.0)) * camera[1] / camera[2]
        + float(intrinsics["cx_px"])
    )
    v = float(intrinsics["fy_px"]) * camera[1] / camera[2] + float(intrinsics["cy_px"])
    return int(round(u)), int(round(v))


def _invert(matrix: Sequence[float]) -> tuple[float, ...]:
    from .transforms import invert_rigid

    return invert_rigid(matrix)


def _humans_from_tracks(
    tracks: Sequence[Mapping[str, Any]],
    poses: Sequence[Sequence[float]],
    source_frames: Sequence[int],
) -> dict[str, Any]:
    frame_to_sample = {int(frame): index for index, frame in enumerate(source_frames)}
    people = []
    for track in tracks:
        samples = []
        for frame, joints in zip(track["frames"], track["joints_camera"]):
            sample_index = frame_to_sample.get(int(frame))
            if sample_index is None:
                continue
            samples.append(
                sample_payload(sample_index, lift_joints_to_world(joints, poses[sample_index]))
            )
        if samples:
            people.append(person_payload(track["id"], samples, track_id=int(track["track_id"])))
    if not people:
        raise HumanReconstructionError("TRAM produced no joints on the kept samples")
    return humans_payload(people)


def _body_preview_points(
    humans: Mapping[str, Any],
    poses: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for person in humans["people"]:
        for sample in person["samples"]:
            points.extend(sample["joints"])
    first = transform_point(poses[0], (0.0, 0.0, 1.0))
    points.extend(
        [
            [first[0] + dx, first[1] - 1.1, first[2] + dz]
            for dx in (-0.6, 0.0, 0.6)
            for dz in (-0.6, 0.0, 0.6)
        ]
    )
    xyz = np.asarray(points, dtype=np.float32)
    colors = np.full((len(xyz), 3), 180, dtype=np.uint8)
    colors[: max(len(points) - 9, 0)] = (255, 140, 48)
    return xyz, colors


def _as_rotations(value: Any) -> list[list[list[float]]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ContractError("pred_cam_R must have shape (N, 3, 3)")
    return array.tolist()


def _as_translations(value: Any) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ContractError("pred_cam_T must have shape (N, 3)")
    return array.tolist()


# Keep the public first-camera matrix imported so tests can pin the gauge.
assert FIRST_CAMERA_WORLD_FROM_OPENCV[5] == -1.0
