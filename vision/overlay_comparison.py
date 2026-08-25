"""Draw observed vs simulated motion on the source video.

Writes a side-by-side MP4, an optional GIF, and a still PNG. No web UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import contact_frames, contact_timing


OBS_COLOR = (220, 180, 48)  # BGR cyan-ish
SIM_COLOR = (48, 96, 255)  # BGR orange-ish
TRAIL = 28


def _by_frame(points: list[dict]) -> dict[int, dict]:
    return {int(point["frame"]): point for point in points}


def _draw_panel(
    frame: np.ndarray,
    point: dict | None,
    history: list[tuple[int, int]],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    panel = frame.copy()
    if len(history) >= 2:
        pts = np.array(history, dtype=np.int32)
        cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)
    if point is not None:
        cx = int(round(point["x"]))
        cy = int(round(point["y"]))
        cv2.circle(panel, (cx, cy), 8, color, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 36), (20, 20, 20), -1)
    cv2.putText(
        panel,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return panel


def _letterbox(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _header(width: int, lines: list[str]) -> np.ndarray:
    scale = 0.62 if width >= 800 else 0.48
    bar_h = 18 + 24 * len(lines)
    bar = np.full((bar_h, width, 3), 18, dtype=np.uint8)
    y = 22
    for line in lines:
        cv2.putText(
            bar,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return bar


def _compose(
    frame: np.ndarray,
    obs: dict | None,
    sim: dict | None,
    obs_hist: list[tuple[int, int]],
    sim_hist: list[tuple[int, int]],
    header: list[str],
    panel_height: int,
) -> np.ndarray:
    left = _draw_panel(frame, obs, obs_hist, OBS_COLOR, "observed")
    right = _draw_panel(frame, sim, sim_hist, SIM_COLOR, "simulated")
    left = _letterbox(left, panel_height)
    right = _letterbox(right, panel_height)
    body = np.hstack([left, right])
    return np.vstack([_header(body.shape[1], header), body])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("tracking")
    parser.add_argument("reconstruction")
    parser.add_argument("--output", default="results/overlay.mp4")
    parser.add_argument("--gif", default="")
    parser.add_argument("--still", default="results/overlay_still.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--gif-stride", type=int, default=3)
    parser.add_argument("--gif-max-width", type=int, default=960)
    parser.add_argument("--still-frame", type=int, default=-1)
    args = parser.parse_args()

    tracking = json.loads(Path(args.tracking).read_text(encoding="utf-8"))
    reconstruction = json.loads(Path(args.reconstruction).read_text(encoding="utf-8"))
    obs_map = _by_frame(tracking["observations"])
    sim_map = _by_frame(reconstruction["simulated"])
    metrics = reconstruction["metrics"]
    parameters = reconstruction["parameters"]
    timing = contact_timing(tracking, reconstruction)
    title = args.title or Path(args.video).stem
    metrics_line = (
        f"RMSE {metrics['rmse']:.2f} px ({metrics['quality']})  "
        f"g={parameters['g']:.1f} px/s^2  e={parameters['e']:.3f}"
    )
    if timing.get("paired"):
        metrics_line += f"  bounce {timing['mean_error_frames']:.2f} fr"
    header = [title, metrics_line]

    contacts = contact_frames(tracking["observations"])
    still_index = args.still_frame
    if still_index < 0:
        if contacts and contacts[0] >= 12:
            still_index = contacts[0] - 10
        else:
            still_index = int(tracking["observations"][len(tracking["observations"]) // 3]["frame"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(tracking["fps"])
    sample = _compose(
        np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8),
        None,
        None,
        [],
        [],
        header,
        args.panel_height,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (sample.shape[1], sample.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    gif_frames: list[Image.Image] = []
    still = None
    obs_hist: list[tuple[int, int]] = []
    sim_hist: list[tuple[int, int]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        obs = obs_map.get(frame_idx)
        sim = sim_map.get(frame_idx)
        if obs is not None:
            obs_hist.append((int(round(obs["x"])), int(round(obs["y"]))))
            obs_hist = obs_hist[-TRAIL:]
        if sim is not None:
            sim_hist.append((int(round(sim["x"])), int(round(sim["y"]))))
            sim_hist = sim_hist[-TRAIL:]
        composed = _compose(frame, obs, sim, obs_hist, sim_hist, header, args.panel_height)
        writer.write(composed)
        if frame_idx == still_index:
            still = composed.copy()
        if args.gif and frame_idx % args.gif_stride == 0:
            rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if image.width > args.gif_max_width:
                ratio = args.gif_max_width / image.width
                image = image.resize(
                    (args.gif_max_width, max(1, int(round(image.height * ratio)))),
                    Image.Resampling.LANCZOS,
                )
            gif_frames.append(image)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"wrote {out_path} frames={frame_idx}")

    if still is None:
        still = composed
    still_path = Path(args.still)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(still_path), still)
    print(f"wrote {still_path}")

    if args.gif and gif_frames:
        gif_path = Path(args.gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = int(round(1000.0 * args.gif_stride / fps))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"wrote {gif_path} n={len(gif_frames)} duration_ms={duration_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
