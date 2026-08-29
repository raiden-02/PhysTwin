import { FormEvent, MouseEvent, useEffect, useRef, useState } from "react";
import {
  createJobFromFile,
  createJobFromSample,
  fetchResult,
  fetchSamples,
  frame0Url,
  runJob,
  subscribeProgress,
  videoUrl,
} from "./api";
import { imageClick, STAGE_LABELS, STAGE_ORDER } from "./geom";
import { ObservationApp } from "./ObservationApp";
import { PhysicsApp } from "./PhysicsApp";
import { ReconstructionScene } from "./Scene";
import { TrajectoryPlot } from "./Trajectory";
import type {
  AnchorMode,
  DynamicsModel,
  Job,
  ProgressEvent,
  Result,
  Sample,
} from "./types";

function fmt(n: number, digits = 2): string {
  return n.toFixed(digits);
}

function stageText(event: ProgressEvent | null, job: Job | null): string {
  const stage = event?.stage ?? job?.stage ?? "ready";
  const label = STAGE_LABELS[stage] ?? stage;
  const current = event?.current ?? job?.progress.current;
  const total = event?.total ?? job?.progress.total;
  if (current != null && total != null) return `${label} ${current} / ${total}`;
  if (event?.detail) return `${label}: ${event.detail}`;
  return label;
}

const QUALITY_TEXT: Record<string, string> = {
  good: "The simulation matches the recording closely.",
  fair: "Usable, but check the two views before trusting the parameters.",
  poor: "Do not trust these parameters. The selected model does not explain this track well.",
};

export function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [model, setModel] = useState<DynamicsModel>("projectile_bounce");
  const [anchorMode, setAnchorMode] = useState<AnchorMode>("fixed");
  const [click, setClick] = useState<{ x: number; y: number } | null>(null);
  const [pivot, setPivot] = useState<{ x: number; y: number } | null>(null);
  const [groundText, setGroundText] = useState("");
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [productMode, setProductMode] = useState<"physics" | "observation" | "physics3d">("physics");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    fetchSamples()
      .then(setSamples)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const stop = subscribeProgress(job.id, (event) => {
      setProgress(event);
      if (event.stage === "failed") {
        setError(event.error ?? "processing failed");
        setJob((prev) =>
          prev ? { ...prev, status: "failed", error: event.error ?? null } : prev,
        );
      }
      if (event.stage === "complete") {
        fetchResult(job.id)
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

  const setVideoNode = (node: HTMLVideoElement | null) => {
    videoRef.current = node;
    setVideoEl(node);
  };

  function reset() {
    setError(null);
    setResult(null);
    setProgress(null);
    setClick(null);
    setPivot(null);
    setAnchorMode("fixed");
  }

  async function onSample(sampleId: string) {
    reset();
    setBusy(true);
    try {
      setJob(await createJobFromSample(sampleId, model));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(file: File) {
    reset();
    setBusy(true);
    try {
      setJob(await createJobFromFile(file, model));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function onFrameClick(event: MouseEvent<HTMLImageElement>) {
    if (!job || !frameRef.current) return;
    const point = imageClick(
      event,
      frameRef.current.getBoundingClientRect(),
      job.width,
      job.height,
    );
    if (!point) return;
    if (model === "pendulum" && click && !pivot) {
      setPivot(point);
    } else {
      setClick(point);
      if (model === "pendulum") setPivot(null);
    }
  }

  async function onProcess(event: FormEvent) {
    event.preventDefault();
    if (!job || !click || (model === "pendulum" && !pivot)) return;
    setError(null);
    setResult(null);
    const groundY =
      model === "pendulum" || groundText.trim() === "" ? null : Number(groundText);
    if (groundY != null && !Number.isFinite(groundY)) {
      setError("ground y must be a number, or leave it empty");
      return;
    }
    setBusy(true);
    try {
      setJob(await runJob(job.id, click.x, click.y, pivot, anchorMode, groundY));
      setProgress({ stage: "queued", detail: "starting local SAM 2 + phystwin fit" });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function togglePlay() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }

  function onScrub(value: number) {
    setTime(value);
    if (videoRef.current) videoRef.current.currentTime = value;
  }

  const quality = result?.reconstruction.metrics.quality ?? job?.quality;
  const running = job?.status === "running";
  const step = result ? 3 : job ? 2 : 1;
  const lastFrame =
    result?.tracking.observations[result.tracking.observations.length - 1]?.frame ?? 0;
  const stageNow = progress?.stage ?? job?.stage ?? "queued";
  const stageIndex = STAGE_ORDER.indexOf(stageNow);

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>PhysTwin lab</h1>
          <p className="lede"><a href="#/">Back to portfolio demo</a></p>
        </div>
        <div className="mode-choice" role="tablist" aria-label="Product mode">
          <button
            type="button"
            className={productMode === "physics" ? "selected" : ""}
            onClick={() => setProductMode("physics")}
          >
            2D physics twin
          </button>
          <button
            type="button"
            className={productMode === "observation" ? "selected" : ""}
            onClick={() => setProductMode("observation")}
          >
            3D scene + camera
          </button>
          <button
            type="button"
            className={productMode === "physics3d" ? "selected" : ""}
            onClick={() => setProductMode("physics3d")}
          >
            3D physics
          </button>
        </div>
        {productMode === "physics" ? (
          <ol className="stepper">
            <li className={step === 1 ? "on" : "done"}>Pick a video</li>
            <li className={step === 2 ? "on" : step > 2 ? "done" : ""}>
              Select target and reference
            </li>
            <li className={step === 3 ? "on" : ""}>Compare and read the fit</li>
          </ol>
        ) : productMode === "observation" ? (
          <ol className="stepper">
            <li>Video to reconstructed camera, geometry, and body</li>
          </ol>
        ) : (
          <ol className="stepper">
            <li>Lab inspectors for P4/P5 fixtures. The portfolio demo is the default page.</li>
          </ol>
        )}
      </header>

      {productMode === "observation" ? <ObservationApp samples={samples} /> : null}
      {productMode === "physics3d" ? <PhysicsApp /> : null}
      {productMode === "physics" && error ? <div className="banner bad">{error}</div> : null}

      {productMode === "physics" && job ? (
        <div className="sourcebar">
          <span className="name">{job.source_name}</span>
          <span className="facts">
            {job.model === "pendulum" ? "Swing / Pendulum" : "Projectile / Bounce"} ·{" "}
            {job.width}×{job.height} · {fmt(job.fps, 2)} fps · {job.n_frames} frames
            {job.kind === "rendered" ? " · rendered check, not real-footage accuracy" : ""}
            {job.kind === "recorded" ? " · recorded footage" : ""}
            {job.kind === "cinematic"
              ? " · cinematic stress footage, not physical validation"
              : ""}
          </span>
          <button
            type="button"
            disabled={busy || running}
            onClick={() => {
              reset();
              setJob(null);
            }}
          >
            Change video
          </button>
        </div>
      ) : null}

      {productMode === "physics" && !job ? (
        <section className="panel">
          <h2>Physics model</h2>
          <div className="model-choice" role="radiogroup" aria-label="Physics model">
            <label className={model === "projectile_bounce" ? "selected" : ""}>
              <input
                type="radio"
                name="model"
                checked={model === "projectile_bounce"}
                onChange={() => setModel("projectile_bounce")}
              />
              <strong>Projectile / Bounce</strong>
              <span>Free flight, gravity, ground collision, restitution</span>
            </label>
            <label className={model === "pendulum" ? "selected" : ""}>
              <input
                type="radio"
                name="model"
                checked={model === "pendulum"}
                onChange={() => setModel("pendulum")}
              />
              <strong>Swing / Pendulum</strong>
              <span>Fixed or tracked anchor, nonlinear swing, effective g/L, damping</span>
            </label>
          </div>
          <h2>Video</h2>
          <p className="hint">
            {model === "pendulum"
              ? "Upload a fixed-camera clip with one visible bob and a fixed pivot."
              : "Use a local sample, or upload a fixed-camera clip of one bouncing object."}
          </p>
          <div className="row">
            {samples.map((sample) => (
              <button
                key={sample.id}
                type="button"
                disabled={busy}
                onClick={() => void onSample(sample.id)}
              >
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

      {productMode === "physics" && job && !running && !result ? (
        <section className="panel clickstep">
          <div className="clickmedia">
            <div className="frame-box">
              <img ref={frameRef} src={frame0Url(job.id)} alt="first frame" onClick={onFrameClick} />
              {click ? (
                <span
                  className="marker target"
                  style={{
                    left: `${(click.x / job.width) * 100}%`,
                    top: `${(click.y / job.height) * 100}%`,
                  }}
                />
              ) : null}
              {pivot ? (
                <span
                  className="marker pivot"
                  style={{
                    left: `${(pivot.x / job.width) * 100}%`,
                    top: `${(pivot.y / job.height) * 100}%`,
                  }}
                />
              ) : null}
            </div>
          </div>
          <form className="clickform" onSubmit={(event) => void onProcess(event)}>
            <h2>
              {model === "pendulum"
                ? click
                  ? pivot
                    ? "Ready to track and fit"
                    : anchorMode === "tracked"
                      ? "Now click the moving anchor"
                      : "Now click the fixed pivot"
                  : "First click the moving bob"
                : "Click the object on frame 0"}
            </h2>
            <p className="hint">{job.hint}</p>
            {model === "pendulum" ? (
              <fieldset className="anchor-choice">
                <legend>Anchor reference</legend>
                <label>
                  <input
                    type="radio"
                    name="anchor-mode"
                    checked={anchorMode === "fixed"}
                    onChange={() => setAnchorMode("fixed")}
                  />
                  <span>
                    <strong>Fixed pivot</strong>
                    <small>Default for a locked camera and stationary pivot.</small>
                  </span>
                </label>
                <label>
                  <input
                    type="radio"
                    name="anchor-mode"
                    checked={anchorMode === "tracked"}
                    onChange={() => setAnchorMode("tracked")}
                  />
                  <span>
                    <strong>Track anchor through clip</strong>
                    <small>
                      Use only when the physical attachment point stays visible and moves in
                      frame.
                    </small>
                  </span>
                </label>
              </fieldset>
            ) : null}
            <p className="coords">
              {click
                ? `target ${fmt(click.x, 1)}, ${fmt(click.y, 1)} px`
                : "target not selected"}
              {model === "pendulum" &&
                (pivot
                  ? ` · ${anchorMode === "tracked" ? "anchor" : "pivot"} ${fmt(
                      pivot.x,
                      1,
                    )}, ${fmt(pivot.y, 1)} px`
                  : ` · ${anchorMode === "tracked" ? "anchor" : "pivot"} not selected`)}
            </p>
            {job.suggested_point ? (
              <button
                type="button"
                onClick={() => {
                  setClick({
                    x: job.suggested_point![0],
                    y: job.suggested_point![1],
                  });
                  if (job.suggested_pivot) {
                    setPivot({
                      x: job.suggested_pivot[0],
                      y: job.suggested_pivot[1],
                    });
                  }
                }}
              >
                {model === "pendulum"
                  ? "Use suggested selections"
                  : "Use suggested click"}
              </button>
            ) : null}
            {model === "projectile_bounce" ? (
              <label className="ground">
                ground y (optional)
                <input
                  value={groundText}
                  onChange={(event) => setGroundText(event.target.value)}
                  placeholder="max centroid"
                />
              </label>
            ) : null}
            {model === "pendulum" && click ? (
              <button
                type="button"
                onClick={() => {
                  setClick(null);
                  setPivot(null);
                }}
              >
                Reset selections
              </button>
            ) : null}
            <button
              className="primary"
              type="submit"
              disabled={!click || (model === "pendulum" && !pivot) || busy}
            >
              Track and fit
            </button>
            <p className="hint small">
              Tracking runs locally on the GPU. Runtime scales with clip length.
            </p>
          </form>
        </section>
      ) : null}

      {productMode === "physics" && running ? (
        <section className="panel">
          <h2>Processing</h2>
          <p className="stage">{stageText(progress, job)}</p>
          <ol className="stages">
            {STAGE_ORDER.slice(0, -1).map((name, index) => (
              <li
                key={name}
                className={
                  index < stageIndex ? "done" : index === stageIndex ? "on" : ""
                }
              >
                {STAGE_LABELS[name]}
              </li>
            ))}
          </ol>
          <p className="hint small">
            Real pipeline stages, not a fake percent bar. Tracking is the slow step. Fitting
            takes tens of milliseconds.
          </p>
        </section>
      ) : null}

      {productMode === "physics" && result ? (
        <section className="result">
          <div className="viewer">
            <div className="pane media">
              <h2>Recording</h2>
              <video
                ref={setVideoNode}
                src={videoUrl(result.job.id)}
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
              <h2>
                {result.reconstruction.model === "pendulum"
                  ? "Pendulum physics twin"
                  : "Projectile physics twin"}
              </h2>
              <ReconstructionScene
                video={videoEl}
                tracking={result.tracking}
                reconstruction={result.reconstruction}
                time={time}
              />
            </div>
          </div>

          <div className="transport">
            <button type="button" onClick={togglePlay}>
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={result.tracking.fps > 0 ? 1 / result.tracking.fps : 0.01}
              value={Math.min(time, duration || time)}
              onChange={(event) => onScrub(Number(event.target.value))}
            />
            <span className="coords">
              t={fmt(time, 2)} s · frame {Math.min(Math.round(time * result.tracking.fps), lastFrame)}{" "}
              / {lastFrame}
            </span>
            <span className="legend">
              <span className="obs">observed</span>
              <span className="sim">simulated</span>
            </span>
            <button
              type="button"
              onClick={() => {
                setResult(null);
                setProgress(null);
                setClick(null);
                setPivot(null);
                setJob((prev) => (prev ? { ...prev, status: "ready", stage: "ready" } : prev));
              }}
            >
              Try another click
            </button>
          </div>

          <div className={`verdict ${quality ?? ""}`}>
            <strong>{quality}</strong>
            <span>{quality ? QUALITY_TEXT[quality] : ""}</span>
          </div>

          <div className="cards">
            <div className="card">
              <span className="label">error (RMSE)</span>
              <span className="value">{fmt(result.reconstruction.metrics.rmse, 2)} px</span>
              <span className="sub">
                x {fmt(result.reconstruction.metrics.rmse_x, 2)} · y{" "}
                {fmt(result.reconstruction.metrics.rmse_y, 2)}
              </span>
            </div>
            {result.reconstruction.model === "projectile_bounce" ? (
              <>
                <div className="card">
                  <span className="label">gravity scale</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.g, 1)}
                  </span>
                  <span className="sub">px/s², not 9.81 m/s²</span>
                </div>
                <div className="card">
                  <span className="label">bounciness</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.e, 3)}
                  </span>
                  <span className="sub">restitution e, 0 to 1</span>
                </div>
                <div className="card">
                  <span className="label">start velocity</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.vy0, 0)} px/s
                  </span>
                  <span className="sub">
                    down · across {fmt(result.reconstruction.parameters.vx0, 1)}
                  </span>
                </div>
              </>
            ) : (
              <>
                <div className="card">
                  <span className="label">effective lambda</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.lambda, 3)} s⁻²
                  </span>
                  <span className="sub">effective g/L, not metric gravity</span>
                </div>
                <div className="card">
                  <span className="label">damping</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.damping, 3)} s⁻¹
                  </span>
                  <span className="sub">non-negative angular damping</span>
                </div>
                <div className="card">
                  <span className="label">initial angular speed</span>
                  <span className="value">
                    {fmt(result.reconstruction.parameters.omega0, 3)} rad/s
                  </span>
                  <span className="sub">
                    radius {fmt(result.reconstruction.environment.radius, 1)} px
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="split metrics">
            <div className="pane">
              <h2>Observed vs simulated</h2>
              <TrajectoryPlot
                observed={result.tracking.observations}
                simulated={result.reconstruction.simulated}
                time={time}
              />
            </div>
            <div className="pane">
              <h2>Run details</h2>
              <dl className="kv">
                {result.reconstruction.model === "projectile_bounce" ? (
                  <>
                    <dt>ground violation</dt>
                    <dd>
                      {fmt(result.reconstruction.metrics.ground_violation, 2)} px
                    </dd>
                    <dt>y_ground</dt>
                    <dd>
                      {fmt(result.reconstruction.environment.y_ground, 1)} px (
                      {result.reconstruction.metrics.ground_source})
                    </dd>
                  </>
                ) : (
                  <>
                    <dt>fitted pivot</dt>
                    <dd>
                      {fmt(result.reconstruction.environment.pivot_x, 1)},{" "}
                      {fmt(result.reconstruction.environment.pivot_y, 1)} px
                    </dd>
                    <dt>pivot adjustment</dt>
                    <dd>
                      {fmt(result.reconstruction.metrics.pivot_adjustment, 2)} px
                    </dd>
                    <dt>anchor reference</dt>
                    <dd>{result.reconstruction.environment.reference_mode}</dd>
                    {result.reconstruction.environment.reference_mode === "tracked" ? (
                      <>
                        <dt>anchor coverage</dt>
                        <dd>
                          {fmt(
                            result.reconstruction.metrics.anchor_track_coverage * 100,
                            1,
                          )}
                          %
                        </dd>
                      </>
                    ) : null}
                    <dt>radial MAD</dt>
                    <dd>{fmt(result.reconstruction.metrics.radial_mad, 2)} px</dd>
                    <dt>angular span</dt>
                    <dd>{fmt(result.reconstruction.metrics.angular_span, 3)} rad</dd>
                  </>
                )}
                <dt>click</dt>
                <dd>
                  {result.job.point
                    ? `${fmt(result.job.point[0], 1)}, ${fmt(result.job.point[1], 1)} px`
                    : "n/a"}
                </dd>
                <dt>fit time</dt>
                <dd>
                  {fmt(result.reconstruction.metrics.fit_seconds, 3)} s · exit{" "}
                  {result.job.fit_exit ?? 0}
                  {result.job.fit_exit === 2 ? " (poor fit still writes JSON)" : ""}
                </dd>
                {result.timing?.end_to_end_seconds != null ? (
                  <>
                    <dt>track time</dt>
                    <dd>
                      {fmt(result.timing.end_to_end_seconds, 2)} s end-to-end
                      {result.timing.end_to_end_fps != null
                        ? ` (${fmt(result.timing.end_to_end_fps, 1)} FPS)`
                        : ""}
                    </dd>
                  </>
                ) : null}
                {result.timing?.device ? (
                  <>
                    <dt>device</dt>
                    <dd>{result.timing.device}</dd>
                  </>
                ) : null}
              </dl>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
