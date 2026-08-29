export type ObservationJob = {
  id: string;
  status: "ready" | "running" | "complete" | "failed";
  stage: string;
  source_name: string;
  kind: string;
  fps: number;
  width: number;
  height: number;
  n_frames: number;
  cache_key: string | null;
  error: string | null;
  progress: {
    current?: number | null;
    total?: number | null;
    detail?: string | null;
  };
};

export type ObservationCoordinates = {
  world_basis: string;
  handedness: string;
  camera_convention: string;
  scale: {
    status: string;
    meters_per_world_unit: number | null;
    source?: string;
  };
};

export type ObservationSample = {
  sample_index: number;
  timestamp_s: number;
  source_frames: Record<string, number>;
};

export type CameraPose = {
  sample_index: number;
  T_world_camera: number[];
  confidence?: number;
};

export type ObservationCamera = {
  id: string;
  source_id: string;
  projection: string;
  lens_distortion?: string;
  image_size_px: [number, number];
  intrinsics: {
    fx_px: number;
    fy_px: number;
    cx_px: number;
    cy_px: number;
    skew_px: number;
  };
  poses: CameraPose[];
};

export type ObservationArtifact = {
  id: string;
  uri: string;
  media_type: string;
  sha256: string;
};

export type SceneObservation = {
  schema: string;
  version: number;
  observation_id: string;
  timeline: {
    time_unit: string;
    origin: string;
    samples: ObservationSample[];
  };
  coordinates: ObservationCoordinates;
  sources: Array<{ id: string; kind: string; uri: string; sha256: string }>;
  artifacts: ObservationArtifact[];
  cameras: ObservationCamera[];
  static_scene: { geometry: Array<{ kind: string; artifact_id: string }> };
  provenance: {
    created_utc?: string;
    producer?: {
      adapter?: string;
      adapter_version?: string;
      model?: string;
      model_revision?: string;
      weights_sha256?: string | null;
      license?: { code?: string; weights?: string };
    };
    options?: Record<string, unknown>;
    runtime?: {
      wall_seconds?: number | null;
      device?: string | null;
      peak_gpu_memory_bytes?: number | null;
      n_frames?: number;
    };
  };
  extensions: Record<string, unknown>;
};

export type ObservationResult = {
  job: ObservationJob;
  observation: SceneObservation;
  runtime: SceneObservation["provenance"]["runtime"] | null;
  cache_key: string | null;
};

export type ObservationProgress = {
  stage: string;
  detail?: string;
  error?: string;
  cache_key?: string;
};
