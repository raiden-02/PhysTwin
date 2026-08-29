"""Measured camera and SMPL24 evaluation in the observation-world frame."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import sha256_file
from .contracts import (
    ContractError,
    canonical_json_bytes,
    project_world_point,
    validate_scene_observation,
)
from .humans import HUMANS_EXTENSION, validate_humans_v1
from .transforms import (
    da3_w2c_to_c2w,
    multiply_4x4,
    observation_from_native,
    transform_point,
)

EVALUATION_SCHEMA = "phystwin.reconstruction_evaluation"
EVALUATION_VERSION = 1
EMDB_CODE_REVISION = "9a4eab677181a3789bda7ba5c36ab8cff797380c"
EMDB_CODE_LICENSE = "MIT"
EMDB_DATASET_LICENSE = "non-commercial academic use; approved institutional account required"


@dataclass(frozen=True)
class ReferenceSequence:
    """Ground-truth camera and SMPL24 joints in the benchmark native world."""

    benchmark: str
    sequence_id: str
    world_to_camera: np.ndarray
    joints_world_m: np.ndarray
    good_frames: np.ndarray
    keypoints_2d_px: np.ndarray | None
    fps: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationOutput:
    """Serializable report plus arrays used by the trajectory plot."""

    report: Mapping[str, Any]
    camera_pred_m: np.ndarray | None
    camera_gt_m: np.ndarray
    root_pred_m: np.ndarray | None
    root_gt_m: np.ndarray


def load_emdb_reference(sequence_root: Path, smpl_model_root: Path) -> ReferenceSequence:
    """Load one approved EMDB sequence and evaluate its SMPL parameters on CPU.

    EMDB annotations are pickle files. Only load files obtained from ETH's
    official download because unpickling untrusted files can execute code.
    """

    data_files = sorted(sequence_root.glob("*_data.pkl"))
    if len(data_files) != 1:
        raise FileNotFoundError(
            f"{sequence_root}: expected exactly one *_data.pkl from the approved EMDB download"
        )
    if not smpl_model_root.is_dir():
        raise FileNotFoundError(
            f"SMPL model root not found: {smpl_model_root}. "
            "Download SMPL separately under its registration terms."
        )
    try:
        import smplx
        import torch
    except ImportError as error:
        raise RuntimeError(
            "EMDB joint evaluation needs `smplx`. Run scripts/setup-evaluation.ps1."
        ) from error

    with data_files[0].open("rb") as handle:
        data = pickle.load(handle)  # noqa: S301 - official licensed EMDB file only

    n_frames = _positive_int(data.get("n_frames"), "EMDB.n_frames")
    gender = data.get("gender")
    if gender not in {"female", "male", "neutral"}:
        raise ContractError("EMDB.gender must be female, male, or neutral")
    camera = _mapping(data.get("camera"), "EMDB.camera")
    smpl = _mapping(data.get("smpl"), "EMDB.smpl")
    world_to_camera = _array(camera.get("extrinsics"), (n_frames, 4, 4), "camera.extrinsics")
    poses_root = _array(smpl.get("poses_root"), (n_frames, 3), "smpl.poses_root")
    poses_body = _array(smpl.get("poses_body"), (n_frames, 69), "smpl.poses_body")
    translation = _array(smpl.get("trans"), (n_frames, 3), "smpl.trans")
    betas = _array(smpl.get("betas"), (10,), "smpl.betas")
    good_frames = np.asarray(data.get("good_frames_mask"), dtype=bool)
    if good_frames.shape != (n_frames,):
        raise ContractError("EMDB.good_frames_mask must have shape (n_frames,)")
    kp2d = np.asarray(data.get("kp2d"), dtype=np.float64)
    if kp2d.shape != (n_frames, 24, 2):
        raise ContractError("EMDB.kp2d must have shape (n_frames, 24, 2)")

    model = smplx.create(
        str(smpl_model_root),
        model_type="smpl",
        gender=gender,
        ext="pkl",
        use_pca=False,
        batch_size=n_frames,
    )
    with torch.no_grad():
        output = model(
            global_orient=torch.as_tensor(poses_root, dtype=torch.float32),
            body_pose=torch.as_tensor(poses_body, dtype=torch.float32),
            betas=torch.as_tensor(
                np.repeat(betas[None, :], n_frames, axis=0),
                dtype=torch.float32,
            ),
            transl=torch.as_tensor(translation, dtype=torch.float32),
        )
    joints = output.joints[:, :24].detach().cpu().numpy().astype(np.float64)
    if joints.shape != (n_frames, 24, 3):
        raise ContractError("SMPL output must contain 24 joints")

    return ReferenceSequence(
        benchmark="EMDB",
        sequence_id=str(data.get("name") or sequence_root.name),
        world_to_camera=world_to_camera,
        joints_world_m=joints,
        good_frames=good_frames,
        keypoints_2d_px=kp2d,
        fps=_video_fps(sequence_root) or 30.0,
        provenance={
            "annotation_file": data_files[0].name,
            "annotation_sha256": sha256_file(data_files[0]),
            "emdb1": bool(data.get("emdb1")),
            "emdb2": bool(data.get("emdb2")),
            "code_revision": EMDB_CODE_REVISION,
            "code_license": EMDB_CODE_LICENSE,
            "dataset_license": EMDB_DATASET_LICENSE,
        },
    )


def reference_from_observation(
    observation: Mapping[str, Any],
    *,
    sequence_id: str = "synthetic-alignment-fixture",
) -> ReferenceSequence:
    """Create a synthetic reference from an observation for regression tests."""

    validate_scene_observation(observation)
    humans = validate_humans_v1(
        observation["extensions"][HUMANS_EXTENSION],
        sample_count=len(observation["timeline"]["samples"]),
    )
    camera = observation["cameras"][0]
    poses = {int(item["sample_index"]): item["T_world_camera"] for item in camera["poses"]}
    person_samples = {
        int(item["sample_index"]): item for item in humans["people"][0]["samples"]
    }
    timeline = observation["timeline"]["samples"]
    frame_count = max(
        int(next(iter(sample["source_frames"].values()))) for sample in timeline
    ) + 1
    w2c = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    joints = np.zeros((frame_count, 24, 3), dtype=np.float64)
    good = np.zeros(frame_count, dtype=bool)
    for sample in timeline:
        sample_index = int(sample["sample_index"])
        source_frame = int(next(iter(sample["source_frames"].values())))
        if sample_index not in poses or sample_index not in person_samples:
            continue
        w2c[source_frame] = np.asarray(_invert_pose(poses[sample_index]), dtype=np.float64).reshape(4, 4)
        joints[source_frame] = np.asarray(person_samples[sample_index]["joints"], dtype=np.float64)
        good[source_frame] = True
    return ReferenceSequence(
        benchmark="synthetic_alignment_fixture",
        sequence_id=sequence_id,
        world_to_camera=w2c,
        joints_world_m=joints,
        good_frames=good,
        keypoints_2d_px=None,
        fps=_timeline_fps(timeline),
        provenance={
            "source": "SceneObservation self-reference",
            "dataset_license": "project fixture",
        },
    )


def evaluate_observation(
    observation: Mapping[str, Any],
    reference: ReferenceSequence,
    *,
    person_id: str | None = None,
) -> EvaluationOutput:
    """Compare one SceneObservation to reference frames with exact source IDs."""

    validate_scene_observation(observation)
    timeline = observation["timeline"]["samples"]
    humans = validate_humans_v1(
        observation["extensions"].get(HUMANS_EXTENSION),
        sample_count=len(timeline),
    )
    people = list(humans["people"])
    person = next(
        (item for item in people if person_id is None or item["id"] == person_id),
        None,
    )
    if person is None:
        raise ContractError(f"human {person_id!r} is not in the observation")

    camera = observation["cameras"][0]
    source_id = camera["source_id"]
    timeline_by_sample = {int(item["sample_index"]): item for item in timeline}
    poses_by_sample = {
        int(item["sample_index"]): item["T_world_camera"] for item in camera["poses"]
    }
    body_by_sample = {
        int(item["sample_index"]): item
        for item in person["samples"]
        if item.get("visible", True)
    }
    if not poses_by_sample:
        raise ContractError("prediction has no camera poses")

    first_sample_index = min(poses_by_sample)
    first_source_frame = _source_frame(
        timeline_by_sample[first_sample_index], source_id
    )
    _check_reference_frame(reference, first_source_frame)
    gt_c2w0 = da3_w2c_to_c2w(reference.world_to_camera[first_source_frame])
    t_obs_from_gt = observation_from_native(gt_c2w0)

    pred_poses: list[np.ndarray] = []
    gt_poses: list[np.ndarray] = []
    pred_joints: list[np.ndarray] = []
    gt_joints: list[np.ndarray] = []
    source_frames: list[int] = []
    sample_indices: list[int] = []
    timestamps: list[float] = []
    excluded_invalid = 0
    excluded_missing_body = 0
    excluded_missing_pose = 0

    for sample_index in sorted(timeline_by_sample):
        sample = timeline_by_sample[sample_index]
        frame = _source_frame(sample, source_id)
        _check_reference_frame(reference, frame)
        if not bool(reference.good_frames[frame]):
            excluded_invalid += 1
            continue
        if sample_index not in poses_by_sample:
            excluded_missing_pose += 1
            continue
        if sample_index not in body_by_sample:
            excluded_missing_body += 1
            continue
        gt_c2w = da3_w2c_to_c2w(reference.world_to_camera[frame])
        gt_pose = multiply_4x4(t_obs_from_gt, gt_c2w)
        gt_body = [
            transform_point(t_obs_from_gt, point)
            for point in reference.joints_world_m[frame]
        ]
        pred_poses.append(np.asarray(poses_by_sample[sample_index], dtype=np.float64).reshape(4, 4))
        gt_poses.append(np.asarray(gt_pose, dtype=np.float64).reshape(4, 4))
        pred_joints.append(np.asarray(body_by_sample[sample_index]["joints"], dtype=np.float64))
        gt_joints.append(np.asarray(gt_body, dtype=np.float64))
        source_frames.append(frame)
        sample_indices.append(sample_index)
        timestamps.append(float(sample["timestamp_s"]))

    if not sample_indices:
        raise ContractError("no valid prediction samples overlap the reference")

    pred_pose_array = np.stack(pred_poses)
    gt_pose_array = np.stack(gt_poses)
    pred_joint_array = np.stack(pred_joints)
    gt_joint_array = np.stack(gt_joints)
    pred_camera_units = pred_pose_array[:, :3, 3]
    gt_camera_m = gt_pose_array[:, :3, 3]
    pred_root_units = pred_joint_array[:, 0]
    gt_root_m = gt_joint_array[:, 0]

    scale_status = observation["coordinates"]["scale"]["status"]
    meters_per_unit = observation["coordinates"]["scale"].get("meters_per_world_unit")
    pred_camera_m = None
    pred_root_m = None
    pred_joints_m = None
    if meters_per_unit is not None:
        scale = float(meters_per_unit)
        pred_camera_m = pred_camera_units * scale
        pred_root_m = pred_root_units * scale
        pred_joints_m = pred_joint_array * scale

    rotation_errors = _rotation_errors_deg(pred_pose_array[:, :3, :3], gt_pose_array[:, :3, :3])
    pa_errors_mm = np.asarray(
        [
            _procrustes_mpjpe(pred_joint_array[index], gt_joint_array[index]) * 1000.0
            for index in range(len(sample_indices))
        ]
    )
    metrics: dict[str, Any] = {
        "camera_rotation_mean_deg": _available(
            float(rotation_errors.mean()), "degree", scale_status="scale_independent"
        ),
        "camera_rotation_p95_deg": _available(
            float(np.percentile(rotation_errors, 95)), "degree", scale_status="scale_independent"
        ),
        "pa_mpjpe_mm": _available(
            float(pa_errors_mm.mean()),
            "millimeter",
            scale_status="per_frame_similarity_aligned",
        ),
    }
    per_frame: dict[str, Any] = {
        "sample_index": sample_indices,
        "source_frame": source_frames,
        "timestamp_s": timestamps,
        "camera_rotation_error_deg": rotation_errors.tolist(),
        "pa_mpjpe_mm": pa_errors_mm.tolist(),
    }

    if pred_joints_m is None or pred_camera_m is None or pred_root_m is None:
        reason = "prediction scale is relative; no meters_per_world_unit is available"
        for name, unit in (
            ("camera_position_rmse_m", "meter"),
            ("root_position_rmse_m", "meter"),
            ("world_mpjpe_mm", "millimeter"),
            ("pelvis_aligned_mpjpe_mm", "millimeter"),
        ):
            metrics[name] = _blocked(unit, reason)
    else:
        camera_errors_m = np.linalg.norm(pred_camera_m - gt_camera_m, axis=1)
        root_errors_m = np.linalg.norm(pred_root_m - gt_root_m, axis=1)
        joint_errors_m = np.linalg.norm(pred_joints_m - gt_joint_array, axis=2)
        pred_local = pred_joints_m - pred_joints_m[:, [0]]
        gt_local = gt_joint_array - gt_joint_array[:, [0]]
        pelvis_errors_mm = np.linalg.norm(pred_local - gt_local, axis=2) * 1000.0
        metrics.update(
            {
                "camera_position_rmse_m": _available(
                    _rmse(camera_errors_m), "meter", scale_status=scale_status
                ),
                "root_position_rmse_m": _available(
                    _rmse(root_errors_m), "meter", scale_status=scale_status
                ),
                "world_mpjpe_mm": _available(
                    float(joint_errors_m.mean() * 1000.0),
                    "millimeter",
                    scale_status=scale_status,
                ),
                "pelvis_aligned_mpjpe_mm": _available(
                    float(pelvis_errors_mm.mean()),
                    "millimeter",
                    scale_status=scale_status,
                ),
            }
        )
        per_frame.update(
            {
                "camera_position_error_m": camera_errors_m.tolist(),
                "root_position_error_m": root_errors_m.tolist(),
                "world_mpjpe_mm": (joint_errors_m.mean(axis=1) * 1000.0).tolist(),
                "pelvis_aligned_mpjpe_mm": pelvis_errors_mm.mean(axis=1).tolist(),
            }
        )

    scale_fit = _origin_scale_fit(pred_camera_units, gt_camera_m)
    if scale_fit is None:
        metrics["camera_scale_aligned_rmse_m"] = _blocked(
            "meter", "camera trajectory is static or too short to estimate scale"
        )
    else:
        fitted_scale, aligned_camera = scale_fit
        metrics["camera_scale_aligned_rmse_m"] = _available(
            _rmse(np.linalg.norm(aligned_camera - gt_camera_m, axis=1)),
            "meter",
            scale_status="single_scale_about_first_camera_origin",
        )
        metrics["camera_fitted_meters_per_unit"] = _available(
            fitted_scale,
            "meter_per_world_unit",
            scale_status="diagnostic_fit",
        )

    lens = camera.get("lens_distortion", "unknown")
    da3_extension = observation["extensions"].get("phystwin.da3.v1", {})
    if reference.keypoints_2d_px is None:
        metrics["reprojection_mpjpe_px"] = _blocked(
            "pixel", "reference has no 2D keypoints"
        )
    elif isinstance(da3_extension, Mapping) and da3_extension.get("intrinsics_vary"):
        metrics["reprojection_mpjpe_px"] = _blocked(
            "pixel",
            "prediction intrinsics vary by sample but the core observation stores only sample 0",
        )
    elif lens not in {"none", "removed"}:
        metrics["reprojection_mpjpe_px"] = _blocked(
            "pixel",
            f"prediction lens distortion is {lens!r}; pinhole reprojection would be invalid",
        )
    else:
        reprojection_errors = _reprojection_errors(
            pred_joint_array,
            pred_pose_array,
            camera["intrinsics"],
            reference.keypoints_2d_px[source_frames],
        )
        metrics["reprojection_mpjpe_px"] = _available(
            float(reprojection_errors.mean()), "pixel", scale_status="scale_independent"
        )
        per_frame["reprojection_mpjpe_px"] = reprojection_errors.mean(axis=1).tolist()

    observation_hash = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    identity = {
        "prediction": observation_hash,
        "benchmark": reference.benchmark,
        "sequence": reference.sequence_id,
        "frames": source_frames,
        "person": person["id"],
    }
    evaluation_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:20]
    status = "measured" if reference.benchmark == "EMDB" else "synthetic_check"
    report = {
        "schema": EVALUATION_SCHEMA,
        "version": EVALUATION_VERSION,
        "evaluation_id": evaluation_id,
        "status": status,
        "benchmark": {
            "name": reference.benchmark,
            "sequence_id": reference.sequence_id,
            "fps": reference.fps,
            "provenance": dict(reference.provenance),
        },
        "prediction": {
            "observation_id": observation["observation_id"],
            "observation_sha256": observation_hash,
            "person_id": person["id"],
            "producer": copy.deepcopy(observation.get("provenance", {}).get("producer", {})),
            "scale": copy.deepcopy(observation["coordinates"]["scale"]),
        },
        "alignment": {
            "time": "exact source frame index",
            "world": "prediction and reference independently gauged to the prediction's first camera frame",
            "world_basis": "first_camera_graphics",
            "metric_alignment": "none",
            "camera_scale_diagnostic": "one scalar about the shared first-camera origin",
            "local_pose": "per-frame pelvis and similarity alignment reported separately",
        },
        "coverage": {
            "prediction_samples": len(timeline),
            "matched_good_samples": len(sample_indices),
            "excluded_reference_invalid": excluded_invalid,
            "excluded_missing_body": excluded_missing_body,
            "excluded_missing_camera_pose": excluded_missing_pose,
        },
        "metrics": metrics,
        "per_frame": per_frame,
        "artifacts": [],
        "limitations": [
            "Metric-assumed prediction scale is an estimator assumption, not measured calibration.",
            "PA-MPJPE removes per-frame scale, rotation, and translation and does not test world trajectory.",
            "Exact source-frame matching does not interpolate missing prediction frames.",
            "Camera gauge, scale, body frame, and sample synchronization errors invalidate later physics even when local-pose error is low.",
            "Low reprojection error alone cannot establish accurate 3D camera or body motion.",
        ],
    }
    validate_evaluation_report(report)
    return EvaluationOutput(
        report=report,
        camera_pred_m=pred_camera_m,
        camera_gt_m=gt_camera_m,
        root_pred_m=pred_root_m,
        root_gt_m=gt_root_m,
    )


def save_evaluation(output_dir: Path, evaluation: EvaluationOutput) -> Mapping[str, Any]:
    """Write a deterministic report and one inspectable trajectory SVG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "trajectory_comparison.svg"
    write_trajectory_svg(plot_path, evaluation)
    report = copy.deepcopy(dict(evaluation.report))
    report["artifacts"] = [
        {
            "id": "trajectory_comparison",
            "uri": plot_path.name,
            "media_type": "image/svg+xml",
            "sha256": sha256_file(plot_path),
        }
    ]
    validate_evaluation_report(report)
    report_path = output_dir / "reconstruction_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_evaluation_report(report: Any) -> Mapping[str, Any]:
    root = _mapping(report, "ReconstructionEvaluation")
    required = {
        "schema",
        "version",
        "evaluation_id",
        "status",
        "benchmark",
        "prediction",
        "alignment",
        "coverage",
        "metrics",
        "per_frame",
        "artifacts",
        "limitations",
    }
    missing = required - set(root)
    if missing:
        raise ContractError(f"ReconstructionEvaluation: missing {', '.join(sorted(missing))}")
    if root["schema"] != EVALUATION_SCHEMA or root["version"] != EVALUATION_VERSION:
        raise ContractError("ReconstructionEvaluation: unsupported schema or version")
    if root["status"] not in {"measured", "synthetic_check"}:
        raise ContractError("ReconstructionEvaluation.status is invalid")
    metrics = _mapping(root["metrics"], "ReconstructionEvaluation.metrics")
    if not metrics:
        raise ContractError("ReconstructionEvaluation.metrics must not be empty")
    for name, raw in metrics.items():
        metric = _mapping(raw, f"metrics.{name}")
        if metric.get("status") not in {"available", "blocked"}:
            raise ContractError(f"metrics.{name}.status is invalid")
        value = metric.get("value")
        if metric["status"] == "available":
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(f"metrics.{name}.value must be finite")
        elif value is not None or not metric.get("reason"):
            raise ContractError(f"metrics.{name}: blocked metric needs null value and reason")
    canonical_json_bytes(root)
    return root


def write_trajectory_svg(path: Path, evaluation: EvaluationOutput) -> Path:
    """Write top and side trajectory views without a plotting dependency."""

    gt_camera = evaluation.camera_gt_m
    gt_root = evaluation.root_gt_m
    pred_camera = evaluation.camera_pred_m
    pred_root = evaluation.root_pred_m
    if pred_camera is None or pred_root is None:
        pred_camera = np.empty((0, 3), dtype=np.float64)
        pred_root = np.empty((0, 3), dtype=np.float64)
    series = [
        ("GT camera", gt_camera, "#6aa9ff"),
        ("Pred camera", pred_camera, "#30b4dc"),
        ("GT pelvis", gt_root, "#ffb05a"),
        ("Pred pelvis", pred_root, "#ff6030"),
    ]
    width, height = 960, 440
    panels = [
        (40, 70, 420, 310, 0, 2, "Top view: X / Z"),
        (500, 70, 420, 310, 0, 1, "Front view: X / Y"),
    ]
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" viewBox="0 0 960 440">',
        '<rect width="960" height="440" fill="#0f1113"/>',
        '<text x="40" y="34" fill="#e7eaee" font-family="Segoe UI, sans-serif" font-size="20">'
        + html.escape(str(evaluation.report["benchmark"]["sequence_id"]))
        + "</text>",
    ]
    for x, y, panel_w, panel_h, axis_x, axis_y, title in panels:
        chunks.append(
            f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" '
            'fill="#171a1e" stroke="#2c323a"/>'
        )
        chunks.append(
            f'<text x="{x + 12}" y="{y + 24}" fill="#9aa3ad" '
            f'font-family="Segoe UI, sans-serif" font-size="14">{title}</text>'
        )
        arrays = [values[:, [axis_x, axis_y]] for _, values, _ in series if len(values)]
        all_points = np.concatenate(arrays, axis=0)
        low = all_points.min(axis=0)
        high = all_points.max(axis=0)
        span = np.maximum(high - low, 1e-6)
        margin = span * 0.08
        low -= margin
        high += margin
        span = high - low
        for label, values, color in series:
            if not len(values):
                continue
            points = []
            for point in values[:, [axis_x, axis_y]]:
                px = x + 16 + (point[0] - low[0]) / span[0] * (panel_w - 32)
                py = y + panel_h - 16 - (point[1] - low[1]) / span[1] * (panel_h - 56)
                points.append(f"{px:.2f},{py:.2f}")
            chunks.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            if points:
                px, py = points[-1].split(",")
                chunks.append(
                    f'<circle cx="{px}" cy="{py}" r="4" fill="{color}">'
                    f"<title>{html.escape(label)}</title></circle>"
                )
    legend_x = 40
    for label, _values, color in series:
        chunks.append(f'<rect x="{legend_x}" y="408" width="12" height="12" fill="{color}"/>')
        chunks.append(
            f'<text x="{legend_x + 18}" y="419" fill="#e7eaee" '
            f'font-family="Segoe UI, sans-serif" font-size="13">{html.escape(label)}</text>'
        )
        legend_x += 150
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return path


def _procrustes_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    source_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    source = predicted - source_mean
    goal = target - target_mean
    variance = float(np.sum(source * source))
    if variance <= 1e-12:
        raise ContractError("cannot similarity-align degenerate joints")
    covariance = source.T @ goal
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vh.T @ u.T))
    rotation = vh.T @ correction @ u.T
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (predicted @ rotation.T) + translation
    return float(np.linalg.norm(aligned - target, axis=1).mean())


def _rotation_errors_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = predicted @ np.transpose(target, (0, 2, 1))
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _origin_scale_fit(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray] | None:
    denominator = float(np.sum(predicted * predicted))
    if len(predicted) < 2 or denominator <= 1e-12:
        return None
    scale = float(np.sum(predicted * target) / denominator)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale, predicted * scale


def _reprojection_errors(
    joints: np.ndarray,
    poses: np.ndarray,
    intrinsics: Mapping[str, Any],
    target_2d: np.ndarray,
) -> np.ndarray:
    errors = np.empty((len(joints), 24), dtype=np.float64)
    for frame in range(len(joints)):
        for joint in range(24):
            pixel = project_world_point(
                joints[frame, joint],
                poses[frame].reshape(-1),
                intrinsics,
            )
            errors[frame, joint] = np.linalg.norm(
                np.asarray(pixel) - target_2d[frame, joint]
            )
    return errors


def _available(
    value: float, unit: str, *, scale_status: str
) -> dict[str, Any]:
    return {
        "status": "available",
        "value": float(value),
        "unit": unit,
        "scale_basis": scale_status,
    }


def _blocked(unit: str, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "value": None,
        "unit": unit,
        "reason": reason,
    }


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{path}: must be a positive integer")
    return value


def _array(value: Any, shape: tuple[int, ...], path: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{path}: must be finite with shape {shape}")
    return array


def _source_frame(sample: Mapping[str, Any], source_id: str) -> int:
    frames = sample["source_frames"]
    if source_id not in frames:
        raise ContractError(f"timeline sample has no frame for source {source_id}")
    return int(frames[source_id])


def _check_reference_frame(reference: ReferenceSequence, frame: int) -> None:
    if not 0 <= frame < len(reference.good_frames):
        raise ContractError(
            f"source frame {frame} is outside reference sequence {reference.sequence_id}"
        )


def _invert_pose(pose: Sequence[float]) -> tuple[float, ...]:
    from .transforms import invert_rigid

    return invert_rigid(pose)


def _timeline_fps(samples: Sequence[Mapping[str, Any]]) -> float:
    if len(samples) < 2:
        return 30.0
    delta = float(samples[1]["timestamp_s"]) - float(samples[0]["timestamp_s"])
    return 1.0 / delta if delta > 0.0 else 30.0


def _video_fps(sequence_root: Path) -> float | None:
    videos = sorted(sequence_root.glob("*_video.mp4"))
    if not videos:
        return None
    try:
        import cv2

        capture = cv2.VideoCapture(str(videos[0]))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return fps if math.isfinite(fps) and fps > 0.0 else None
    except ImportError:
        return None
