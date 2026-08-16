export type Sample = {
  id: string;
  label: string;
  kind: string;
  hint: string;
  suggested_point: [number, number] | null;
  filename: string;
  bytes: number;
};

export type Job = {
  id: string;
  status: "ready" | "running" | "complete" | "failed";
  stage: string;
  source_name: string;
  kind: string;
  hint: string;
  suggested_point: [number, number] | null;
  fps: number;
  width: number;
  height: number;
  n_frames: number;
  point: [number, number] | null;
  quality: string | null;
  fit_exit: number | null;
  error: string | null;
  progress: {
    current?: number | null;
    total?: number | null;
    detail?: string | null;
  };
};

export type Point = {
  frame: number;
  t: number;
  x: number;
  y: number;
  radius?: number;
};

export type Reconstruction = {
  version: number;
  parameters: { vx0: number; vy0: number; g: number; e: number };
  environment: { x0: number; y0: number; y_ground: number; dt: number };
  units: {
    position: string;
    time: string;
    velocity: string;
    gravity: string;
    restitution: string;
  };
  metrics: {
    rmse: number;
    mae: number;
    rmse_x: number;
    rmse_y: number;
    normalized_rmse: number;
    worst_axis_normalized_rmse: number;
    ground_violation: number;
    quality: string;
    ground_source: string;
    n: number;
    search_generations: number;
    refinement_iterations: number;
    iterations: number;
    fit_seconds: number;
  };
  simulated: Point[];
};

export type Tracking = {
  version: number;
  fps: number;
  frame_width: number;
  frame_height: number;
  observations: Point[];
};

export type Timing = {
  device?: string;
  n_frames?: number;
  skipped_empty_masks?: number;
  end_to_end_seconds?: number;
  end_to_end_fps?: number;
  timing_includes?: string;
};

export type Result = {
  job: Job;
  tracking: Tracking;
  reconstruction: Reconstruction;
  timing: Timing | null;
};

export type ProgressEvent = {
  stage: string;
  t?: number;
  current?: number;
  total?: number;
  detail?: string;
  error?: string;
  quality?: string;
  skipped?: number;
};
