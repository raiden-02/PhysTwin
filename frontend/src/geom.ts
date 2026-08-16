import type { Point } from "./types";

export const OBS_COLOR = "#30b4dc";
export const SIM_COLOR = "#ff6030";

export function interpolate(points: Point[], t: number): Point {
  if (points.length === 0) {
    return { frame: 0, t: 0, x: 0, y: 0 };
  }
  if (t <= points[0].t) return points[0];
  const last = points[points.length - 1];
  if (t >= last.t) return last;
  let lo = 0;
  let hi = points.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = points[lo];
  const b = points[hi];
  const span = b.t - a.t;
  const u = span > 1e-9 ? (t - a.t) / span : 0;
  return {
    frame: u < 0.5 ? a.frame : b.frame,
    t,
    x: a.x + (b.x - a.x) * u,
    y: a.y + (b.y - a.y) * u,
    radius: a.radius,
  };
}

export function imageClick(
  event: { clientX: number; clientY: number },
  rect: DOMRect,
  videoWidth: number,
  videoHeight: number,
): { x: number; y: number } | null {
  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const contentW = videoWidth * scale;
  const contentH = videoHeight * scale;
  const ox = rect.left + (rect.width - contentW) / 2;
  const oy = rect.top + (rect.height - contentH) / 2;
  const x = (event.clientX - ox) / scale;
  const y = (event.clientY - oy) / scale;
  if (x < 0 || y < 0 || x > videoWidth || y > videoHeight) return null;
  return { x, y };
}

export const STAGE_LABELS: Record<string, string> = {
  ready: "Waiting for a click",
  queued: "Starting local pipeline",
  reading_video: "Reading video metadata",
  extracting_frames: "Decoding frames",
  loading_sam2: "Loading SAM 2 on the GPU",
  initializing_tracker: "Initializing the tracker",
  tracking: "Tracking the object",
  writing_tracking: "Writing tracking.json",
  fitting: "Fitting physics with phystwin",
  complete: "Done",
  failed: "Failed",
};

// Ordered pipeline stages. Used to show which step the run has reached.
export const STAGE_ORDER = [
  "queued",
  "reading_video",
  "extracting_frames",
  "loading_sam2",
  "initializing_tracker",
  "tracking",
  "writing_tracking",
  "fitting",
  "complete",
];
