import { useEffect, useState } from "react";
import {
  fetchPhysicsRealFit,
  inspectPhysicsRealFit,
  runPhysicsFitFixture,
  runPhysicsFixture,
} from "./api";
import { PhysicsScene } from "./PhysicsScene";
import type {
  PhysicsFitFixtureResult,
  PhysicsFixtureResult,
  PhysicsRealFitResult,
} from "./physics";

type View = PhysicsFixtureResult | PhysicsFitFixtureResult | PhysicsRealFitResult;

function fmt(value: number, digits = 3): string {
  return value.toFixed(digits);
}

function memorySize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
}

export function PhysicsApp() {
  const [result, setResult] = useState<View | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing || !result || !("rollout" in result) || !result.rollout) return;
    let frame = 0;
    let previous = performance.now();
    const startTime = result.rollout.timeline.start_time_s;
    const endTime = startTime + result.rollout.timeline.duration_s;
    const tick = (now: number) => {
      const elapsed = (now - previous) / 1000;
      previous = now;
      setTime((current) => {
        const next = current + elapsed;
        return next > endTime ? startTime : next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, result]);

  async function runFixture() {
    setBusy(true);
    setError(null);
    try {
      const next = await runPhysicsFixture();
      setResult(next);
      setTime(next.rollout.timeline.start_time_s);
      setPlaying(false);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    } finally {
      setBusy(false);
    }
  }

  async function inspectRealFit() {
    setBusy(true);
    setError(null);
    try {
      const next = await inspectPhysicsRealFit();
      setResult(next);
      if (next.rollout) {
        setTime(next.rollout.timeline.start_time_s);
      }
      setPlaying(false);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void fetchPhysicsRealFit()
      .then((next) => {
        if (next.fit?.status === "COMPLETE" && next.rollout) {
          setResult(next);
          setTime(next.rollout.timeline.start_time_s);
        }
      })
      .catch(() => {
        /* readiness is loaded on demand */
      });
  }, []);

  async function runFitFixture() {
    setBusy(true);
    setError(null);
    try {
      const next = await runPhysicsFitFixture();
      setResult(next);
      setTime(next.rollout.timeline.start_time_s);
      setPlaying(false);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    } finally {
      setBusy(false);
    }
  }

  if (!result) {
    return (
      <>
        {error ? <div className="banner bad">{error}</div> : null}
        <section className="panel">
          <h2>Executable 3D physics</h2>
          <p className="hint">
            Run the project-owned tether scene in Newton XPBD on CUDA. The browser displays the
            saved rollout and does not integrate physics.
          </p>
          <button className="primary" type="button" disabled={busy} onClick={() => void runFixture()}>
            {busy ? "Running Newton/Warp..." : "Inspect P4 physics fixture"}
          </button>
          <button type="button" disabled={busy} onClick={() => void runFitFixture()}>
            {busy ? "GPU stage running..." : "Inspect P5 synthetic fit"}
          </button>
          <button type="button" disabled={busy} onClick={() => void inspectRealFit()}>
            {busy ? "Checking footage..." : "Inspect P5R real fit"}
          </button>
        </section>
      </>
    );
  }

  if ("footage" in result && !result.rollout) {
    const requested = result.requested_clip;
    return (
      <section className="result">
        {error ? <div className="banner bad">{error}</div> : null}
        <div className="sourcebar">
          <span className="name">P5R real-video inverse physics</span>
          <span className="facts">{result.status}</span>
          <button type="button" disabled={busy} onClick={() => void runFixture()}>
            Run P4
          </button>
          <button type="button" disabled={busy} onClick={() => void runFitFixture()}>
            Run P5 fit
          </button>
          <button type="button" disabled={busy} onClick={() => void inspectRealFit()}>
            Recheck footage
          </button>
        </div>
        <div className="verdict">
          <strong>Waiting for a measured tether clip</strong>
          <span>
            Local videos were inspected. None have a tape-measured length in meters, so
            scale stays unmarked and Newton is not run.
          </span>
        </div>
        <div className="pane">
          <h2>Requested clip</h2>
          <dl className="kv">
            <dt>duration</dt>
            <dd>
              {requested.duration_s.min}–{requested.duration_s.max} s
            </dd>
            <dt>subject</dt>
            <dd>{requested.subject}</dd>
            <dt>must show</dt>
            <dd>{requested.must_show.join(", ")}</dd>
            <dt>motion</dt>
            <dd>{requested.motion}</dd>
            <dt>measurement</dt>
            <dd>{requested.measurement}</dd>
            <dt>quality</dt>
            <dd>{requested.quality}</dd>
          </dl>
          <p className="hint">
            Do not guess an object diameter. Do not use cinematic footage as the
            correctness baseline.
          </p>
        </div>
        <div className="pane">
          <h2>Local clips</h2>
          <dl className="kv">
            {result.footage.rejected.map((clip) => (
              <div key={clip.id}>
                <dt>{clip.id}</dt>
                <dd>
                  {clip.present ? "present" : "missing"} · {clip.reason}
                </dd>
              </div>
            ))}
          </dl>
        </div>
        {result.fit ? (
          <div className="pane">
            <h2>Last blocked report</h2>
            <p className="hint">{result.fit.blockers.join(" ")}</p>
          </div>
        ) : null}
      </section>
    );
  }

  if (!("rollout" in result) || !result.rollout) {
    return null;
  }

  const rollout = result.rollout;
  const fitReport =
    "fit" in result && result.fit && result.rollout ? result.fit : null;
  const motionObservation =
    "motion_observation" in result ? result.motion_observation ?? undefined : undefined;
  const realFit = "footage" in result ? result : null;
  const constraint = rollout.constraints[0];
  const ranges = rollout.validation.body_position_range_m;
  const repeat = rollout.reproducibility.repeat_run;
  return (
    <section className="result">
      {error ? <div className="banner bad">{error}</div> : null}
      <div className="sourcebar">
        <span className="name">{rollout.source.physical_scene_id}</span>
        <span className="facts">
          Newton {rollout.simulator.backend_version} · Warp {rollout.simulator.warp_version} ·{" "}
          {rollout.simulator.device_name}
        </span>
        <button type="button" disabled={busy} onClick={() => void runFixture()}>
          Run P4
        </button>
        <button type="button" disabled={busy} onClick={() => void runFitFixture()}>
          Run P5 fit
        </button>
        <button type="button" disabled={busy} onClick={() => void inspectRealFit()}>
          Inspect P5R
        </button>
      </div>

      <div className="pane media physics-pane">
        <h2>{fitReport ? "Observed and fitted 3D motion" : "Simulated 3D rollout"}</h2>
        <PhysicsScene
          rollout={rollout}
          time={time}
          motionObservation={motionObservation}
        />
        {fitReport ? (
          <p className="hint">Blue: target samples. Orange: fitted Newton rollout.</p>
        ) : null}
      </div>
      <div className="transport">
        <button type="button" onClick={() => setPlaying((current) => !current)}>
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={rollout.timeline.start_time_s}
          max={rollout.timeline.start_time_s + rollout.timeline.duration_s}
          step={rollout.timeline.fixed_step_s}
          value={time}
          onChange={(event) => {
            setPlaying(false);
            setTime(Number(event.target.value));
          }}
        />
        <span className="coords">t={fmt(time, 2)} s</span>
      </div>

      <div className="verdict">
        <strong>
          {realFit
            ? realFit.fit?.status === "COMPLETE"
              ? `Newton execution ${
                  realFit.fit.validation.execution_valid ? "valid" : "invalid"
                } · quality ${realFit.fit.validation.quality.status}`
              : "P5R is blocked or awaiting footage"
            : fitReport
              ? fitReport.validation.quality.status === "synthetic_checked" &&
                fitReport.validation.passed
                ? "P5 synthetic recovery passes"
                : "P5 fit validation failed"
              : rollout.validation.passed
                ? "P4 invariants pass"
                : "P4 validation failed"}
        </strong>
        <span>
          {realFit
            ? "RMSE is reported. It is not a physics-quality pass. validation.passed means the Newton run executed, not that the residual is good."
            : fitReport
              ? "The target is PhysicalMotionObservation. The orange path is the fitted SimulatedWorldState."
              : "The line endpoint, body transform, and trajectory come from SimulatedWorldState."}
        </span>
      </div>
      {fitReport ? (
        <div className="cards">
          <div className="card">
            <span className="label">fit RMSE</span>
            <span className="value">
              {fitReport.objective.rmse_m == null
                ? "n/a"
                : `${fmt(fitReport.objective.rmse_m * 1000, 2)} mm`}
            </span>
            <span className="sub">
              normalized{" "}
              {fitReport.objective.normalized_rmse == null
                ? "n/a"
                : fmt(fitReport.objective.normalized_rmse, 6)}
            </span>
          </div>
          {fitReport.parameters.map((parameter) => (
            <div className="card" key={parameter.id}>
              <span className="label">{parameter.id}</span>
              <span className="value">
                {parameter.fitted == null ? "held / not fitted" : fmt(parameter.fitted, 4)}
              </span>
              <span className="sub">
                {parameter.held_fixed ? "fixed from calibration · " : "fitted · "}
                truth {parameter.truth === null ? "not recorded" : fmt(parameter.truth, 4)}{" "}
                {parameter.unit}
              </span>
            </div>
          ))}
        </div>
      ) : null}
      <div className="cards">
        <div className="card">
          <span className="label">tether max / RMS</span>
          <span className="value">
            {fmt(rollout.validation.tether_error_m.maximum * 1e6, 2)} /{" "}
            {fmt(rollout.validation.tether_error_m.rms * 1e6, 2)} µm
          </span>
          <span className="sub">{constraint.rest_length_m} m fixed distance</span>
        </div>
        <div className="card">
          <span className="label">XYZ travel</span>
          <span className="value">
            {fmt(ranges.x)} / {fmt(ranges.y)} / {fmt(ranges.z)} m
          </span>
          <span className="sub">{ranges.varying_axis_count_at_0_05_m} axes exceed 0.05 m</span>
        </div>
        <div className="card">
          <span className="label">runtime</span>
          <span className="value">{fmt(rollout.execution.wall_seconds, 2)} s</span>
          <span className="sub">
            {rollout.execution.steps} steps at {fmt(1 / rollout.timeline.fixed_step_s, 0)} Hz
          </span>
        </div>
        <div className="card">
          <span className="label">GPU allocation</span>
          <span className="value">{memorySize(rollout.execution.peak_gpu_memory_bytes)}</span>
          <span className="sub">Warp mempool high-water</span>
        </div>
      </div>
      <div className="pane">
        <h2>Execution details</h2>
        <dl className="kv">
          <dt>gravity</dt>
          <dd>[{rollout.world.gravity_m_s2.join(", ")}] m/s², +Y up</dd>
          <dt>constraint</dt>
          <dd>
            world anchor [{constraint.world_anchor_m.join(", ")}] to body attachment [
            {constraint.body_attachment_m.join(", ")}]
          </dd>
          <dt>finite / time</dt>
          <dd>
            {String(rollout.validation.finite_state)} / {String(rollout.validation.time_monotonic)}
          </dd>
          <dt>determinism</dt>
          <dd>
            {repeat.performed
              ? `repeat delta ${repeat.max_abs_transform_delta}`
              : `${rollout.reproducibility.requested_deterministic_mode}, repeat not recorded`}
          </dd>
          <dt>scene hash</dt>
          <dd>{rollout.source.physical_scene_sha256}</dd>
        </dl>
      </div>
    </section>
  );
}
