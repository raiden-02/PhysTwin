export type PhysicalScene = {
  scene_id: string;
  world: { gravity_m_s2: [number, number, number] };
  execution: {
    duration_s: number;
    fixed_step_s: number;
    solver: { type: string; iterations: number; deterministic_mode: string };
  };
};

export type RolloutBodySample = {
  sample_index: number;
  T_world_body: number[];
  linear_velocity_m_s: [number, number, number];
  angular_velocity_rad_s: [number, number, number];
};

export type SimulatedWorldState = {
  schema: string;
  version: number;
  rollout_id: string;
  source: {
    physical_scene_id: string;
    physical_scene_sha256: string;
  };
  simulator: {
    backend: string;
    backend_version: string;
    backend_revision: string;
    solver: string;
    warp_version: string;
    warp_revision: string;
    device: string;
    device_name: string;
    up_axis: string;
    cuda_toolkit: string;
    cuda_driver_api: string;
  };
  world: { gravity_m_s2: [number, number, number] };
  timeline: {
    start_time_s: number;
    duration_s: number;
    fixed_step_s: number;
    samples: Array<{ sample_index: number; timestamp_s: number }>;
  };
  bodies: Array<{
    id: string;
    type: string;
    shape: { type: string; radius_m: number };
    mass_kg: number;
    samples: RolloutBodySample[];
  }>;
  constraints: Array<{
    id: string;
    type: string;
    body_id: string;
    world_anchor_m: [number, number, number];
    body_attachment_m: [number, number, number];
    rest_length_m: number;
  }>;
  execution: {
    status: string;
    steps: number;
    output_samples: number;
    wall_seconds: number;
    peak_gpu_memory_bytes: number;
  };
  validation: {
    passed: boolean;
    finite_state: boolean;
    time_monotonic: boolean;
    gravity_matches_contract: boolean;
    tether_error_m: { maximum: number; rms: number };
    body_position_range_m: {
      x: number;
      y: number;
      z: number;
      varying_axis_count_at_0_05_m: number;
    };
  };
  reproducibility: {
    requested_deterministic_mode: string;
    repeat_run: {
      performed: boolean;
      max_abs_transform_delta: number | null;
      tolerance: number;
      within_tolerance: boolean | null;
    };
  };
  warnings: string[];
  failures: string[];
};

export type PhysicsFixtureResult = {
  physical_scene: PhysicalScene;
  rollout: SimulatedWorldState;
  stdout: string;
};
