"""Convert DA3 OpenCV world-to-camera poses into the canonical observation frame."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import FIRST_CAMERA_WORLD_FROM_OPENCV, ContractError, validate_rigid_transform

Matrix16 = tuple[float, ...]


def as_homogeneous44(matrix: Sequence[Sequence[float]] | Sequence[float]) -> Matrix16:
    """Accept a 3x4 or 4x4 matrix and return a row-major 4x4 rigid transform."""

    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()  # type: ignore[assignment]
    if len(matrix) == 16 and not isinstance(matrix[0], (list, tuple)):
        values = [float(item) for item in matrix]  # type: ignore[arg-type]
        return validate_rigid_transform(values, "transform")

    rows = list(matrix)
    if len(rows) == 3 and len(rows[0]) == 4:
        values = [float(item) for row in rows for item in row] + [0.0, 0.0, 0.0, 1.0]
        return validate_rigid_transform(values, "transform")
    if len(rows) == 4 and len(rows[0]) == 4:
        values = [float(item) for row in rows for item in row]
        return validate_rigid_transform(values, "transform")
    raise ContractError("extrinsic must be 3x4 or 4x4")


def invert_rigid(matrix: Sequence[float]) -> Matrix16:
    """Invert a validated row-major rigid 4x4 transform."""

    pose = validate_rigid_transform(matrix, "transform")
    rotation = (
        (pose[0], pose[4], pose[8]),
        (pose[1], pose[5], pose[9]),
        (pose[2], pose[6], pose[10]),
    )
    translation = (pose[3], pose[7], pose[11])
    inv_t = tuple(-sum(rotation[row][k] * translation[k] for k in range(3)) for row in range(3))
    return (
        rotation[0][0], rotation[0][1], rotation[0][2], inv_t[0],
        rotation[1][0], rotation[1][1], rotation[1][2], inv_t[1],
        rotation[2][0], rotation[2][1], rotation[2][2], inv_t[2],
        0.0, 0.0, 0.0, 1.0,
    )


def multiply_4x4(a: Sequence[float], b: Sequence[float]) -> Matrix16:
    return tuple(
        sum(a[row * 4 + k] * b[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    )


def transform_point(matrix: Sequence[float], point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def da3_w2c_to_c2w(w2c: Sequence[Sequence[float]] | Sequence[float]) -> Matrix16:
    """Invert one DA3 OpenCV/COLMAP world-to-camera extrinsic."""

    return invert_rigid(as_homogeneous44(w2c))


def observation_from_native(native_c2w0: Sequence[float]) -> Matrix16:
    """Map DA3's native world onto the first-camera graphics gauge."""

    return multiply_4x4(FIRST_CAMERA_WORLD_FROM_OPENCV, invert_rigid(native_c2w0))


def canonical_poses_from_da3_w2c(
    w2c_list: Sequence[Sequence[Sequence[float]] | Sequence[float]],
) -> tuple[Matrix16, list[Matrix16]]:
    """Return (T_obs_from_native, T_world_camera for each sample)."""

    if w2c_list is None or len(w2c_list) == 0:
        raise ContractError("DA3 returned no camera extrinsics")
    native_c2w = [da3_w2c_to_c2w(item) for item in w2c_list]
    t_obs_from_native = observation_from_native(native_c2w[0])
    poses = [multiply_4x4(t_obs_from_native, pose) for pose in native_c2w]
    first = poses[0]
    for actual, expected in zip(first, FIRST_CAMERA_WORLD_FROM_OPENCV):
        if not math.isclose(actual, expected, abs_tol=1e-5):
            raise ContractError("gauged first camera pose is not the observation world")
    return t_obs_from_native, poses


def scale_intrinsics(
    matrix: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> dict[str, float]:
    """Scale DA3 processed-resolution K to source pixels."""

    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()  # type: ignore[assignment]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ContractError("intrinsics must be 3x3")
    src_w, src_h = source_size
    proc_w, proc_h = processed_size
    if min(src_w, src_h, proc_w, proc_h) <= 0:
        raise ContractError("image sizes must be positive")
    sx = src_w / proc_w
    sy = src_h / proc_h
    return {
        "fx_px": float(matrix[0][0]) * sx,
        "fy_px": float(matrix[1][1]) * sy,
        "cx_px": float(matrix[0][2]) * sx,
        "cy_px": float(matrix[1][2]) * sy,
        "skew_px": float(matrix[0][1]) * sx,
    }


def unproject_depth_point(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    skew: float = 0.0,
) -> tuple[float, float, float]:
    """Unproject one OpenCV pixel with pinhole depth Z."""

    if depth <= 0.0 or fx <= 0.0 or fy <= 0.0:
        raise ContractError("unproject requires positive depth and focal length")
    y = (v - cy) / fy
    x = (u - cx - skew * y) / fx
    return (x * depth, y * depth, depth)
