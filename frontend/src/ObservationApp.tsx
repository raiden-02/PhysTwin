import { useEffect, useRef, useState } from "react";
import {
  createObservationFromFile,
  createObservationFromSample,
  fetchObservationResult,
  observationArtifactUrl,
  observationVideoUrl,
  runObservation,
  subscribeObservationProgress,
} from "./api";
import { ObservationScene } from "./ObservationScene";
import type { Sample } from "./types";
import type { ObservationJob, ObservationProgress, ObservationResult } from "./observation";

function fmt(n: number, digits = 2): string {
  return n.toFixed(digits);
}

function megabytes(bytes: number | null | undefined): string {
  if (bytes == null) return "n/a";
  return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
}

type Props = {
  samples: Sample[];
};

export function ObservationApp({ samples }: Props) {
  const [job, setJob] = useState<ObservationJob | null>(null);
  const [result, setResult] = useState<ObservationResult | null>(null);
  const [progress, setProgress] = useState<ObservationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const stop = subscribeObservationProgress(job.id, (event) => {
      setProgress(event);
      if (event.stage === "failed") {
        setError(event.error ?? "reconstruction failed");
        setJob((prev) =>
          prev ? { ...prev, status: "failed", error: event.error ?? null } : prev,
        );
      }
      if (event.stage === "complete") {
        fetchObservationResult(job.id)
          .then((next) => {
            setResult(next);
            setJob(next.job);
            setTime(0);
          })
          .catch((err: Error) => setError(err.message));
      }
    });
    return stop;
  }, [job?.id, job?.status]);

  async function onSample(sampleId: string) {
    setError(null);
    setResult(null);
    setProgress(null);
    setBusy(true);
    try {
      setJob(await createObservationFromSample(sampleId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(file: File) {
    setError(null);
    setResult(null);
    setProgress(null);
    setBusy(true);
    try {
      setJob(await createObservationFromFile(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onRun() {
    if (!job) return;
    setError(null);
    setBusy(true);
    try {
      setJob(await runObservation(job.id));
      setProgress({ stage: "queued", detail: "starting DA3 reconstruction" });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const running = job?.status === "running";
  const observation = result?.observation;
  const lastSample = observation?.timeline.samples[observation.timeline.samples.length - 1];

  return (
    <>
      {error ? <div className="banner bad">{error}</div> : null}

      {!job ? (
        <section className="panel">
          <h2>3D reconstruction</h2>
          <p className="hint">
            P1 recovers a camera path and one inspectable point cloud. Scale stays relative.
            This does not fit physics.
          </p>
          <div className="row">
            {samples.map((sample) => (
              <button key={sample.id} type="button" disabled={busy} onClick={() => void onSample(sample.id)}>
                {sample.label}
                <span className="kind">{sample.kind}</span>
              </button>
            ))}
            <label className="upload">
              Upload video
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                disabled={busy}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void onUpload(file);
                  event.target.value = "";
                }}
              />
            </label>
          </div>
        </section>
      ) : null}

      {job ? (
        <div className="sourcebar">
          <span className="name">{job.source_name}</span>
          <span className="facts">
            3D observation · {job.width}×{job.height} · {fmt(job.fps, 2)} fps · {job.n_frames}{" "}
            frames
            {job.kind === "recorded" ? " · recorded footage" : ""}
          </span>
          <button
            type="button"
            disabled={busy || running}
            onClick={() => {
              setJob(null);
              setResult(null);
              setProgress(null);
              setError(null);
            }}
          >
            Change video
          </button>
        </div>
      ) : null}

      {job && !running && !result ? (
        <section className="panel">
          <h2>Ready to reconstruct</h2>
          <p className="hint">
            Uses the first 2 seconds, up to 12 frames, and pinned DA3-BASE. Output is cached by
            content hash.
          </p>
          <button className="primary" type="button" disabled={busy} onClick={() => void onRun()}>
            Reconstruct camera and scene
          </button>
        </section>
      ) : null}

      {running ? (
        <section className="panel">
          <h2>Processing</h2>
          <p className="stage">
            {progress?.detail ? `${progress.stage}: ${progress.detail}` : job?.stage}
          </p>
          <p className="hint small">
            Reconstruction runs locally on the GPU. A cache hit skips the model.
          </p>
        </section>
      ) : null}

      {result && observation ? (
        <section className="result">
          <div className="viewer">
            <div className="pane media">
              <h2>Recording</h2>
              <video
                ref={videoRef}
                src={observationVideoUrl(result.job.id)}
                playsInline
                title="source recording"
                onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
                onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              />
            </div>
            <div className="pane media">
              <h2>Recovered camera and geometry</h2>
              <ObservationScene
                observation={observation}
                artifactUrl={observationArtifactUrl(result.job.id, "scene_geometry")}
                time={time}
              />
            </div>
          </div>
          <div className="transport">
            <button
              type="button"
              onClick={() => {
                const video = videoRef.current;
                if (!video) return;
                if (video.paused) void video.play();
                else video.pause();
              }}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={duration || lastSample?.timestamp_s || 0}
              step={0.01}
              value={Math.min(time, duration || time)}
              onChange={(event) => {
                const next = Number(event.target.value);
                setTime(next);
                if (videoRef.current) videoRef.current.currentTime = next;
              }}
            />
            <span className="coords">t={fmt(time, 2)} s</span>
          </div>
          <div className="verdict">
            <strong>relative scale</strong>
            <span>
              Camera poses are in the first-camera graphics world. No metric calibration was
              applied.
            </span>
          </div>
          <div className="cards">
            <div className="card">
              <span className="label">samples</span>
              <span className="value">{observation.timeline.samples.length}</span>
              <span className="sub">kept source frames</span>
            </div>
            <div className="card">
              <span className="label">runtime</span>
              <span className="value">
                {result.runtime?.wall_seconds != null
                  ? `${fmt(result.runtime.wall_seconds, 1)} s`
                  : "n/a"}
              </span>
              <span className="sub">{result.runtime?.device ?? "device unknown"}</span>
            </div>
            <div className="card">
              <span className="label">peak GPU</span>
              <span className="value">{megabytes(result.runtime?.peak_gpu_memory_bytes)}</span>
              <span className="sub">max allocated</span>
            </div>
            <div className="card">
              <span className="label">model</span>
              <span className="value">DA3-BASE</span>
              <span className="sub">
                {observation.provenance.producer?.license?.weights ?? "Apache-2.0"}
              </span>
            </div>
          </div>
          <div className="pane">
            <h2>Run details</h2>
            <dl className="kv">
              <dt>adapter</dt>
              <dd>
                {observation.provenance.producer?.adapter}{" "}
                {observation.provenance.producer?.adapter_version}
              </dd>
              <dt>weights</dt>
              <dd>{observation.provenance.producer?.weights_sha256 ?? "n/a"}</dd>
              <dt>cache key</dt>
              <dd>{result.cache_key ?? "n/a"}</dd>
              <dt>lens</dt>
              <dd>{observation.cameras[0]?.lens_distortion ?? "unknown"}</dd>
            </dl>
          </div>
        </section>
      ) : null}
    </>
  );
}
