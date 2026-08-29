"""SAM2 masks on the same source frames DA3 reconstructed."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video import sample_video_frames


DEFAULT_CHECKPOINT = Path("checkpoints") / "sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def track_selected_frames(
    video: Path,
    source_frames: Sequence[int],
    *,
    target: tuple[float, float],
    anchor: tuple[float, float] | None = None,
    checkpoint: Path | None = None,
    config: str = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Propagate SAM2 on DA3-selected frames only. Returns source-resolution masks."""

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("P5R SAM2 tracking requires CUDA")
    frames = sample_video_frames(video, list(source_frames))
    checkpoint_path = checkpoint or DEFAULT_CHECKPOINT
    with tempfile.TemporaryDirectory(prefix="phystwin_p5r_frames_") as tmp:
        frame_dir = Path(tmp)
        for index, frame in enumerate(frames):
            path = frame_dir / f"{index:05d}.jpg"
            if not cv2.imwrite(str(path), frame.image_bgr):
                raise RuntimeError(f"failed to write {path}")
        predictor = build_sam2_video_predictor(config, str(checkpoint_path), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(frame_dir))
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[target[0], target[1]]], dtype=np.float32),
                labels=labels,
            )
            if anchor is not None:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=2,
                    points=np.array([[anchor[0], anchor[1]]], dtype=np.float32),
                    labels=labels,
                )
            masks: dict[int, dict[str, np.ndarray | None]] = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                sample = {"target": None, "anchor": None}
                for index, raw_obj_id in enumerate(obj_ids):
                    obj_id = int(raw_obj_id.item()) if hasattr(raw_obj_id, "item") else int(raw_obj_id)
                    binary = np.squeeze(mask_logits[index].detach().float().cpu().numpy()) > 0.0
                    if obj_id == 1:
                        sample["target"] = binary.astype(bool)
                    elif obj_id == 2:
                        sample["anchor"] = binary.astype(bool)
                masks[int(frame_idx)] = sample
    return {
        "source_frames": [frame.source_frame for frame in frames],
        "masks": masks,
        "image_size_px": [int(frames[0].image_bgr.shape[1]), int(frames[0].image_bgr.shape[0])],
    }
