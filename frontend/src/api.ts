import type { DynamicsModel, Job, ProgressEvent, Result, Sample } from "./types";

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
