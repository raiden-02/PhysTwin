export type DemoPath = {
  times_s: number[];
  positions_m: number[][];
};

export type FallingBallDemo = {
  title: string;
  dataset: string;
  relative_video: string;
  video_start_s: number;
  video_duration_s: number;
  gpu: string;
  metrics: {
    recovered_gravity_m_s2: number;
    iris_gravity_m_s2: number;
    gravity_percent_error: number;
    rmse_m: number;
    normalized_rmse: number;
  };
  observed: DemoPath;
  fitted: DemoPath;
  moon: DemoPath & {
    gravity_m_s2: number;
    source_fitted_scene_sha256: string;
    rollout_sha256: string;
  };
  hashes: {
    video_sha256: string;
    fitted_scene_sha256: string;
    fitted_rollout_sha256: string;
    moon_scene_sha256: string;
    moon_rollout_sha256: string;
  };
};

export async function fetchFallingBallDemo(): Promise<FallingBallDemo> {
  const response = await fetch("/api/demo/falling-ball");
  if (!response.ok) {
    throw new Error(`Demo payload is missing (${response.status})`);
  }
  return (await response.json()) as FallingBallDemo;
}

export function positionAt(path: DemoPath, time: number): number[] {
  if (path.positions_m.length === 0) return [0, 0, 0];
  if (time <= path.times_s[0]) return path.positions_m[0];
  const last = path.times_s.length - 1;
  if (time >= path.times_s[last]) return path.positions_m[last];
  let index = 1;
  while (index < path.times_s.length && path.times_s[index] < time) {
    index += 1;
  }
  const previous = index - 1;
  const span = path.times_s[index] - path.times_s[previous];
  const alpha = span <= 1e-9 ? 0 : (time - path.times_s[previous]) / span;
  return path.positions_m[previous].map(
    (value, axis) => value + alpha * (path.positions_m[index][axis] - value),
  );
}
