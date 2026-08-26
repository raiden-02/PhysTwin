export type DynamicsModel = "projectile_bounce" | "pendulum";

export type Sample = {
  id: string;
  label: string;
  kind: string;
  hint: string;
  suggested_point: [number, number] | null;
  suggested_pivot: [number, number] | null;
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
  suggested_pivot: [number, number] | null;
  model: DynamicsModel;
  fps: number;
  width: number;
  height: number;
  n_frames: number;
  point: [number, number] | null;
  pivot: [number, number] | null;
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

type CommonMetrics = {
  rmse: number;
  mae: number;
  rmse_x: number;
  rmse_y: number;
  normalized_rmse: number;
  quality: string;
  n: number;
  search_generations: number;
  refinement_iterations: number;
  fit_seconds: number;
};

export type ProjectileReconstruction = {
  version: number;
  model: "projectile_bounce";
  parameters: { vx0: number; vy0: number; g: number; e: number };
  environment: { x0: number; y0: number; y_ground: number; dt: number };
  units: {
    position: string;
    time: string;
    velocity: string;
    gravity: string;
    restitution: string;
  };
  metrics: CommonMetrics & {
    worst_axis_normalized_rmse: number;
    ground_violation: number;
    ground_source: string;
    iterations: number;
  };
  simulated: Point[];
};

export type PendulumReconstruction = {
  version: number;
  model: "pendulum";
  parameters: { omega0: number; lambda: number; damping: number };
  environment: {
    pivot_x: number;
    pivot_y: number;
    radius: number;
    theta0: number;
    integration_step: number;
  };
  units: {
    position: string;
    time: string;
    angle: string;
    angular_velocity: string;
    lambda: string;
    damping: string;
  };
  metrics: CommonMetrics & {
    robust_cost: number;
    radial_mad: number;
    angular_span: number;
    pivot_adjustment: number;
  };
  simulated: Point[];
};

export type Reconstruction = ProjectileReconstruction | PendulumReconstruction;

export type Tracking = {
  version: number;
  model: DynamicsModel;
  fps: number;
  frame_width: number;
  frame_height: number;
  reference?: { pivot_x: number; pivot_y: number };
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
