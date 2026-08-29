import type {
  ObservationJob,
  ObservationProgress,
  ObservationResult,
} from "./observation";
import type {
  AnchorMode,
  DynamicsModel,
  Job,
  ProgressEvent,
  Result,
  Sample,
} from "./types";
import type {
  PhysicsFitFixtureResult,
  PhysicsFixtureResult,
  PhysicsRealFitResult,
} from "./physics";

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item: { msg?: string }) => item.msg ?? JSON.stringify(item)).join("; ");
    }
    if (typeof body.error === "string") return body.error;
  } catch {
    /* use status text */
  }
  return `${response.status} ${response.statusText}`;
}

export async function fetchSamples(): Promise<Sample[]> {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error(await parseError(response));
  const body = await response.json();
  return body.samples as Sample[];
}

export async function createJobFromSample(
  sampleId: string,
  model: DynamicsModel,
): Promise<Job> {
  const body = new FormData();
  body.append("sample_id", sampleId);
  body.append("model", model);
  const response = await fetch("/api/jobs", { method: "POST", body });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function createJobFromFile(
  file: File,
  model: DynamicsModel,
): Promise<Job> {
  const body = new FormData();
  body.append("file", file);
  body.append("model", model);
  const response = await fetch("/api/jobs", { method: "POST", body });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function runJob(
  jobId: string,
  x: number,
  y: number,
  pivot: { x: number; y: number } | null,
  anchorMode: AnchorMode,
  groundY: number | null,
): Promise<Job> {
  const response = await fetch(`/api/jobs/${jobId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      x,
      y,
      pivot_x: pivot?.x ?? null,
      pivot_y: pivot?.y ?? null,
      anchor_mode: anchorMode,
      ground_y: groundY,
    }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchResult(jobId: string): Promise<Result> {
  const response = await fetch(`/api/jobs/${jobId}/result`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export function subscribeProgress(
  jobId: string,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (message) => {
    const event = JSON.parse(message.data) as ProgressEvent;
    onEvent(event);
  };
  source.onerror = () => {
    /* EventSource retries. The run also polls job status. */
  };
  return () => source.close();
}

export function frame0Url(jobId: string): string {
  return `/api/jobs/${jobId}/frame0`;
}

export function videoUrl(jobId: string): string {
  return `/api/jobs/${jobId}/video`;
}

export async function createObservationFromSample(sampleId: string): Promise<ObservationJob> {
  const body = new FormData();
  body.append("sample_id", sampleId);
  const response = await fetch("/api/observations", { method: "POST", body });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function createObservationFromFile(file: File): Promise<ObservationJob> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/observations", { method: "POST", body });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function runObservation(jobId: string): Promise<ObservationJob> {
  const response = await fetch(`/api/observations/${jobId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_s: 0, duration_s: 2, max_frames: 12 }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchObservationResult(jobId: string): Promise<ObservationResult> {
  const response = await fetch(`/api/observations/${jobId}/result`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export function subscribeObservationProgress(
  jobId: string,
  onEvent: (event: ObservationProgress) => void,
): () => void {
  const source = new EventSource(`/api/observations/${jobId}/events`);
  source.onmessage = (message) => {
    onEvent(JSON.parse(message.data) as ObservationProgress);
  };
  return () => source.close();
}

export function observationVideoUrl(jobId: string): string {
  return `/api/observations/${jobId}/video`;
}

export function observationArtifactUrl(jobId: string, artifactId: string): string {
  return `/api/observations/${jobId}/artifacts/${artifactId}`;
}

export async function createHumanFixture(): Promise<ObservationJob> {
  const response = await fetch("/api/human-fixtures", { method: "POST" });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function attachObservationHumans(
  jobId: string,
  tramDir?: string,
): Promise<ObservationJob> {
  const response = await fetch(`/api/observations/${jobId}/humans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tram_dir: tramDir ?? null }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function runPhysicsFixture(): Promise<PhysicsFixtureResult> {
  const response = await fetch("/api/physics-fixture", { method: "POST" });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function runPhysicsFitFixture(): Promise<PhysicsFitFixtureResult> {
  const response = await fetch("/api/physics-fit-fixture", { method: "POST" });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchPhysicsRealFit(): Promise<PhysicsRealFitResult> {
  const response = await fetch("/api/physics-real-fit");
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function inspectPhysicsRealFit(): Promise<PhysicsRealFitResult> {
  const response = await fetch("/api/physics-real-fit", { method: "POST" });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}
