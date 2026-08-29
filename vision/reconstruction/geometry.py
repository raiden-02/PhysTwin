"""Write a small inspectable point-cloud GLB without extra 3D libraries."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
POINTS_MODE = 0


def write_point_cloud_glb(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> Path:
    """Write XYZ RGB points as a glTF POINTS primitive."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must match points")
    if points.shape[0] == 0:
        raise ValueError("point cloud is empty")

    xyz = np.ascontiguousarray(points.astype(np.float32))
    rgb = np.ascontiguousarray(np.clip(colors, 0, 255).astype(np.uint8))
    position_bytes = xyz.tobytes()
    color_bytes = rgb.tobytes()
    bin_padding = (4 - (len(color_bytes) % 4)) % 4
    bin_blob = position_bytes + color_bytes + (b"\x00" * bin_padding)

    mins = xyz.min(axis=0).tolist()
    maxs = xyz.max(axis=0).tolist()
    count = int(xyz.shape[0])
    document = {
        "asset": {"version": "2.0", "generator": "phystwin-reconstruction-p1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": POINTS_MODE,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": count,
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": count,
                "type": "VEC3",
                "normalized": True,
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(position_bytes),
                "byteLength": len(color_bytes),
            },
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_padding

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total))
        handle.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(bin_blob), BIN_CHUNK))
        handle.write(bin_blob)
    return path


def read_glb_point_count(path: Path) -> int:
    """Read the POSITION accessor count from a GLB written by this module."""

    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path} is not a GLB")
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise ValueError(f"{path} is not a glTF 2 GLB")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != JSON_CHUNK:
        raise ValueError(f"{path} is missing a JSON chunk")
    document = json.loads(data[20 : 20 + json_length])
    return int(document["accessors"][0]["count"])
