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

export const SMPL24_BONES: Array<[number, number]> = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 4],
  [4, 7],
  [7, 10],
  [2, 5],
  [5, 8],
  [8, 11],
  [3, 6],
  [6, 9],
  [9, 12],
  [12, 15],
  [9, 13],
  [13, 16],
  [16, 18],
  [18, 20],
  [20, 22],
  [9, 14],
  [14, 17],
  [17, 19],
  [19, 21],
  [21, 23],
];

export type HumanSample = {
  sample_index: number;
  root: [number, number, number];
  joints: Array<[number, number, number]>;
  visible?: boolean;
};

export type HumanPerson = {
  id: string;
  track_id?: number;
  samples: HumanSample[];
};

export type HumansV1 = {
  joint_layout: string;
  coordinate_frame: string;
  people: HumanPerson[];
};

export function readHumans(observation: SceneObservation): HumansV1 | null {
  const raw = observation.extensions["phystwin.humans.v1"];
  if (!raw || typeof raw !== "object") return null;
  const humans = raw as HumansV1;
  if (!Array.isArray(humans.people) || humans.people.length === 0) return null;
  return humans;
}

export function humanSampleAt(person: HumanPerson, sampleIndex: number): HumanSample | null {
  return person.samples.find((sample) => sample.sample_index === sampleIndex) ?? person.samples[0] ?? null;
}
