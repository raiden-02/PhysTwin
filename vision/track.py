#!/usr/bin/env python3
"""SAM 2 video tracking worker.

Reads one click on frame 0, propagates the mask on CUDA, writes tracking.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory import geometry_from_mask, observation_from_geometry


DEFAULT_CHECKPOINT = Path("checkpoints") / "sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)


def parse_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected x,y as numbers") from exc


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch does not see CUDA. Checkpoint 2 requires the RTX 4080.")
    name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__}", file=sys.stderr)
    print(f"cuda {torch.version.cuda}", file=sys.stderr)
    print(f"device 0: {name}", file=sys.stderr)
    if "4080" not in name:
        print(f"warning: expected RTX 4080, got {name}", file=sys.stderr)
    return name


def download_checkpoint(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"downloading SAM 2.1 tiny checkpoint to {path}", file=sys.stderr)
    urllib.request.urlretrieve(CHECKPOINT_URL, path)
    return path


def read_video_meta(video: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        raise RuntimeError(
            f"video fps metadata is missing or invalid ({fps}). "
            "Refusing to assume 30 fps because gravity scale depends on the time base."
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count


ProgressFn = Callable[[str, dict], None]


def extract_jpeg_frames(
    video: Path,
    frame_dir: Path,
    max_frames: int | None,
    on_progress: ProgressFn | None = None,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    while True:
        if max_frames is not None and written >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        out = frame_dir / f"{written:05d}.jpg"
        if not cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to write {out}")
        written += 1
        if on_progress and written % 30 == 0:
            on_progress("extracting_frames", {"current": written})
    cap.release()
    if written == 0:
        raise RuntimeError(f"no frames decoded from {video}")
    if on_progress:
        on_progress("extracting_frames", {"current": written, "total": written})
    return written


def dump_frame0(video: Path, output: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame 0 from {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"failed to write {output}")
    print(f"wrote {output} size={frame.shape[1]}x{frame.shape[0]}", file=sys.stderr)


def mask_confidence(logits: np.ndarray, binary: np.ndarray) -> float:
    values = np.asarray(logits).reshape(binary.shape)
    if binary.any():
        return float(1.0 / (1.0 + np.exp(-values[binary].mean())))
    return 0.0


def write_preview(
    frame_dir: Path,
    observations: list[dict],
    output: Path,
    prompt: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not observations:
        raise RuntimeError("no observations to plot")

    first = cv2.imread(str(frame_dir / "00000.jpg"))
    last_idx = observations[-1]["frame"]
    last = cv2.imread(str(frame_dir / f"{last_idx:05d}.jpg"))
    if first is None or last is None:
        raise RuntimeError("failed to load preview frames")

    xs = [obs["x"] for obs in observations]
    ys = [obs["y"] for obs in observations]
    ts = [obs["t"] for obs in observations]

    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    last_rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(first_rgb)
    axes[0].scatter([prompt[0]], [prompt[1]], c="lime", s=40, marker="+")
    axes[0].set_title("frame 0 + click")
    axes[0].axis("off")

    axes[1].imshow(last_rgb)
    axes[1].plot(xs, ys, color="cyan", linewidth=1.5)
    axes[1].scatter(xs[-1], ys[-1], c="red", s=20)
    axes[1].set_title(f"last tracked frame {last_idx}")
    axes[1].axis("off")

    axes[2].plot(ts, xs, label="x px")
    axes[2].plot(ts, ys, label="y px (+y down)")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("pixels")
    axes[2].set_title("trajectory")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def track(
    video: Path,
    point: tuple[float, float],
    output: Path,
    checkpoint: Path,
    config: str,
    max_frames: int | None,
    viz: Path | None,
    keep_frames: Path | None,
    on_progress: ProgressFn | None = None,
) -> dict:
    def emit(stage: str, **info: object) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    require_cuda()
    fps, width, height, reported = read_video_meta(video)
    print(
        f"video {video} fps={fps:.4f} size={width}x{height} reported_frames={reported}",
        file=sys.stderr,
    )
    emit("reading_video", fps=fps, width=width, height=height, reported_frames=reported)

    checkpoint = download_checkpoint(checkpoint)
    from sam2.build_sam import build_sam2_video_predictor

    tmp_owned = None
    frame_dir = keep_frames
    if frame_dir is None:
        tmp_owned = tempfile.mkdtemp(prefix="phystwin_frames_")
        frame_dir = Path(tmp_owned)
    try:
        emit("extracting_frames", current=0, total=reported if reported > 0 else None)
        n_frames = extract_jpeg_frames(video, frame_dir, max_frames, on_progress)
        print(f"extracted {n_frames} jpeg frames to {frame_dir}", file=sys.stderr)

        emit("loading_sam2")
        start = time.perf_counter()
        predictor = build_sam2_video_predictor(config, str(checkpoint), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            emit("initializing_tracker", total=n_frames)
            state = predictor.init_state(str(frame_dir))
            points = np.array([[point[0], point[1]]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            raw_rows: list[dict] = []
            observations: list[dict] = []
            skipped = 0
            emit("tracking", current=0, total=n_frames)
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                logits = mask_logits[0].detach().float().cpu().numpy()
                binary = np.squeeze(logits) > 0.0
                geometry = geometry_from_mask(binary)
                conf = mask_confidence(np.squeeze(logits), binary) if geometry else 0.0
                raw_rows.append(
                    {
                        "frame": int(frame_idx),
                        "empty": geometry is None,
                        "confidence": conf,
                        "area": 0.0 if geometry is None else geometry["area"],
                    }
                )
                if geometry is None:
                    skipped += 1
                else:
                    observations.append(
                        observation_from_geometry(int(frame_idx), fps, geometry, conf)
                    )
                if on_progress and (
                    frame_idx == 0
                    or (frame_idx + 1) % 5 == 0
                    or frame_idx + 1 == n_frames
                ):
                    emit(
                        "tracking",
                        current=int(frame_idx) + 1,
                        total=n_frames,
                        skipped=skipped,
                    )
        elapsed = time.perf_counter() - start
        infer_fps = n_frames / elapsed if elapsed > 0 else 0.0
        print(
            f"tracked {len(observations)}/{n_frames} frames in {elapsed:.2f}s "
            f"end-to-end ({infer_fps:.1f} FPS including model load, JPEG decode, "
            f"and propagation), skipped {skipped} empty masks",
            file=sys.stderr,
        )
        if not observations:
            raise RuntimeError("SAM 2 produced no valid masks. Try a different --point.")

        emit("writing_tracking", n=len(observations), skipped=skipped)
        payload = {
            "version": 1,
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "observations": observations,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        raw_path = output.with_name(output.stem + "_raw.json")
        raw_path.write_text(
            json.dumps(
                {
                    "video": str(video),
                    "point": [point[0], point[1]],
                    "n_frames": n_frames,
                    "skipped_empty_masks": skipped,
                    "end_to_end_seconds": elapsed,
                    "end_to_end_fps": infer_fps,
                    "timing_includes": "model_load,init_state,jpeg_decode,propagate",
                    "inference_seconds": elapsed,
                    "inference_fps": infer_fps,
                    "device": torch.cuda.get_device_name(0),
                    "frames": raw_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}", file=sys.stderr)
        print(f"wrote raw log {raw_path}", file=sys.stderr)

        if viz is not None:
            write_preview(frame_dir, observations, viz, point)
            print(f"wrote preview {viz}", file=sys.stderr)
        return payload
    finally:
        if tmp_owned is not None:
            shutil.rmtree(tmp_owned, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track one object in a video and write tracking.json"
    )
    parser.add_argument("video", help="input video path")
    parser.add_argument(
        "--point",
        type=parse_point,
        help="initial click in pixel coordinates, x,y",
    )
    parser.add_argument("--output", default="results/tracking.json", help="tracking.json path")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="SAM 2.1 checkpoint path",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SAM 2 config name")
    parser.add_argument("--max-frames", type=int, default=None, help="optional frame cap")
    parser.add_argument(
        "--viz",
        default="results/tracking_preview.png",
        help="trajectory/mask preview PNG. empty string disables it",
    )
    parser.add_argument(
        "--keep-frames",
        default="",
        help="optional directory to keep extracted JPEG frames",
    )
    parser.add_argument(
        "--dump-frame",
        default="",
        help="write frame 0 to this PNG and exit, for choosing --point",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1

    try:
        if args.dump_frame:
            dump_frame0(video, Path(args.dump_frame))
            return 0
        if args.point is None:
            print("error: --point x,y is required unless --dump-frame is set", file=sys.stderr)
            return 1
        viz = Path(args.viz) if args.viz else None
        keep = Path(args.keep_frames) if args.keep_frames else None
        track(
            video=video,
            point=args.point,
            output=Path(args.output),
            checkpoint=Path(args.checkpoint),
            config=args.config,
            max_frames=args.max_frames,
            viz=viz,
            keep_frames=keep,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
