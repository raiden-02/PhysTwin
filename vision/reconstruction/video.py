"""Sample source frames and timestamps without assuming a constant frame rate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    source_frame: int
    timestamp_s: float
    timestamp_source: str
    image_bgr: np.ndarray


def sha256_file(path: Path) -> str:
    from .cache import sha256_file as _sha256_file

    return _sha256_file(path)


def read_video_meta(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not np.isfinite(fps) or fps <= 1e-3:
        raise RuntimeError(f"video fps metadata is missing or invalid ({fps})")
    if width <= 0 or height <= 0 or count <= 0:
        raise RuntimeError(f"video metadata is invalid: {width}x{height} / {count} frames")
    return fps, width, height, count


def choose_source_frames(
    n_frames: int,
    fps: float,
    *,
    start_s: float,
    duration_s: float | None,
    max_frames: int,
) -> list[int]:
    """Pick an evenly spaced subset of source frames inside a time window."""

    if n_frames <= 0:
        raise ValueError("video has no frames")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    start = max(0, int(round(start_s * fps)))
    if duration_s is None:
        end = n_frames
    else:
        end = min(n_frames, start + max(1, int(round(duration_s * fps))))
    if start >= n_frames:
        raise ValueError("start_s is past the end of the video")
    window = list(range(start, end))
    if len(window) <= max_frames:
        return window
    if max_frames == 1:
        return [window[0]]
    indexes = np.linspace(0, len(window) - 1, max_frames)
    return [window[int(round(index))] for index in indexes]


def sample_video_frames(
    path: Path,
    source_frames: list[int],
) -> list[SampledFrame]:
    """Decode selected frames and record container timestamps when they increase."""

    wanted = sorted(set(source_frames))
    if not wanted:
        raise ValueError("no source frames requested")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        cap.release()
        raise RuntimeError(f"video fps metadata is missing or invalid ({fps})")

    grabbed: dict[int, tuple[np.ndarray, float]] = {}
    index = 0
    try:
        while index <= wanted[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if index in wanted:
                grabbed[index] = (frame, msec)
            index += 1
    finally:
        cap.release()

    missing = [frame for frame in wanted if frame not in grabbed]
    if missing:
        raise RuntimeError(f"failed to decode source frames {missing} from {path}")

    raw_times = []
    for frame_index in wanted:
        msec = grabbed[frame_index][1]
        if np.isfinite(msec) and msec >= 0.0:
            raw_times.append(msec / 1000.0)
        else:
            raw_times.append(float("nan"))
    use_container = (
        all(np.isfinite(raw_times))
        and all(raw_times[i] < raw_times[i + 1] for i in range(len(raw_times) - 1))
    )
    origin = raw_times[0] if use_container else 0.0
    samples: list[SampledFrame] = []
    for order, frame_index in enumerate(wanted):
        if use_container:
            timestamp = max(0.0, raw_times[order] - origin)
            source = "container"
        else:
            timestamp = frame_index / fps
            source = "fps"
        samples.append(
            SampledFrame(
                source_frame=frame_index,
                timestamp_s=timestamp,
                timestamp_source=source,
                image_bgr=grabbed[frame_index][0],
            )
        )
    if samples[0].timestamp_s != 0.0:
        first = samples[0].timestamp_s
        samples = [
            SampledFrame(
                source_frame=item.source_frame,
                timestamp_s=item.timestamp_s - first,
                timestamp_source=item.timestamp_source,
                image_bgr=item.image_bgr,
            )
            for item in samples
        ]
    return samples
