import { useEffect, useMemo, useRef, useState } from "react";
import { DemoScene } from "./DemoScene";
import { fetchFallingBallDemo, type FallingBallDemo } from "./demo";

function fmt(value: number, digits: number): string {
  return value.toFixed(digits);
}

export function PortfolioApp() {
  const [demo, setDemo] = useState<FallingBallDemo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [showMoon, setShowMoon] = useState(true);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    fetchFallingBallDemo()
      .then(setDemo)
      .catch((err: Error) => setError(err.message));
  }, []);

  const duration = useMemo(() => {
    if (!demo) return 0.5;
    const last = demo.observed.times_s[demo.observed.times_s.length - 1] ?? 0.5;
    return Math.max(last, demo.video_duration_s);
  }, [demo]);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setTime((prev) => {
        const next = prev + dt;
        return next >= duration ? 0 : next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, duration]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !demo) return;
    const target = demo.video_start_s + time;
    if (Math.abs(video.currentTime - target) > 0.08) {
      video.currentTime = target;
    }
  }, [time, demo]);

  if (error) {
    return (
      <div className="portfolio">
        <h1>PhysTwin</h1>
        <p className="lede">Could not load the falling-ball demo. {error}</p>
      </div>
    );
  }
  if (!demo) {
    return (
      <div className="portfolio">
        <h1>PhysTwin</h1>
        <p className="lede">Loading the falling-ball demo…</p>
      </div>
    );
  }

  return (
    <div className="portfolio">
      <header className="hero">
        <h1>PhysTwin</h1>
        <p className="lede">
          Reconstruct 3D motion from a video, fit a rigid-body scene in Newton,
          then resimulate the same motion under different gravity.
        </p>
      </header>

      <section className="demo-grid" aria-label="IRIS Falling Ball">
        <figure className="panel">
          <figcaption>Source video</figcaption>
          {videoError ? (
            <p className="hint">
              The IRIS clip is not on this machine. The 3D paths still come from
              the saved reconstruction.
            </p>
          ) : (
            <video
              ref={videoRef}
              src="/api/demo/falling-ball/video"
              muted
              playsInline
              onError={() => setVideoError("missing")}
              aria-label="IRIS falling-ball source video"
            />
          )}
        </figure>
        <figure className="panel">
          <figcaption>3D paths</figcaption>
          <DemoScene
            observed={demo.observed}
            fitted={demo.fitted}
            moon={demo.moon}
            time={time}
            showMoon={showMoon}
          />
          <ul className="legend">
            <li><span className="swatch observed" /> Observed</li>
            <li><span className="swatch fitted" /> Fitted</li>
            <li><span className="swatch moon" /> Moon</li>
          </ul>
        </figure>
      </section>

      <section className="controls" aria-label="Playback">
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setTime(0);
          }}
        >
          Reset
        </button>
        <label className="scrubber">
          Timeline
          <input
            type="range"
            min={0}
            max={duration}
            step={0.01}
            value={time}
            onChange={(event) => {
              setPlaying(false);
              setTime(Number(event.target.value));
            }}
          />
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showMoon}
            onChange={(event) => setShowMoon(event.target.checked)}
          />
          Show Moon gravity
        </label>
      </section>

      <section className="metrics" aria-label="Measured result">
        <div>
          <span>Recovered gravity</span>
          <strong>{fmt(demo.metrics.recovered_gravity_m_s2, 2)} m/s²</strong>
        </div>
        <div>
          <span>IRIS ground truth</span>
          <strong>{fmt(demo.metrics.iris_gravity_m_s2, 2)} m/s²</strong>
        </div>
        <div>
          <span>Error</span>
          <strong>{fmt(demo.metrics.gravity_percent_error, 1)}%</strong>
        </div>
        <div>
          <span>Trajectory RMSE</span>
          <strong>{fmt(demo.metrics.rmse_m, 3)} m</strong>
        </div>
        <div>
          <span>GPU</span>
          <strong>{demo.gpu}</strong>
        </div>
      </section>

      <details className="fold">
        <summary>How it works</summary>
        <pre className="pipeline">{`Video
↓
SAM2 tracking
↓
DA3 camera
↓
known-radius sphere reconstruction
↓
PhysicalMotionObservation
↓
Newton + Warp inverse fit
↓
PhysicalScene
↓
counterfactual rollout`}</pre>
      </details>

      <details className="fold">
        <summary>Limitations</summary>
        <p>
          Metric scale uses the measured IRIS ball radius. Gravity direction is
          assumed down the camera +Y axis. The model is one rigid sphere with
          no drag or contact. A pendulum experiment with DA3 depth and a short
          XPBD rod did not recover a usable trajectory.
        </p>
      </details>

      <details className="fold">
        <summary>Method detail</summary>
        <ul>
          <li>Dataset: {demo.dataset} {demo.relative_video}</li>
          <li>Normalized RMSE: {fmt(demo.metrics.normalized_rmse, 3)}</li>
          <li>Video SHA-256: {demo.hashes.video_sha256}</li>
          <li>Fitted scene: {demo.hashes.fitted_scene_sha256}</li>
          <li>Moon rollout: {demo.hashes.moon_rollout_sha256}</li>
          <li>
            Moon gravity is a simulated hypothesis. It was not observed in the video.
          </li>
          <li>
            <a href="#/lab">Diagnostics</a>
          </li>
        </ul>
      </details>
    </div>
  );
}
