import { useEffect, useState } from "react";
import { runPhysicsFixture } from "./api";
import { PhysicsScene } from "./PhysicsScene";
import type { PhysicsFixtureResult } from "./physics";

function fmt(value: number, digits = 3): string {
  return value.toFixed(digits);
}

function memorySize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
}

export function PhysicsApp() {
  const [result, setResult] = useState<PhysicsFixtureResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing || !result) return;
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

  if (!result) {
    return (
      <>
        {error ? <div className="banner bad">{error}</div> : null}
        <section className="panel">
          <h2>P4 executable physics</h2>
          <p className="hint">
            Run the project-owned tether scene in Newton XPBD on CUDA. The browser displays the
            saved rollout and does not integrate physics.
          </p>
          <button className="primary" type="button" disabled={busy} onClick={() => void runFixture()}>
            {busy ? "Running Newton/Warp..." : "Inspect P4 physics fixture"}
          </button>
        </section>
      </>
    );
  }

  const rollout = result.rollout;
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
          Run again
        </button>
      </div>

      <div className="pane media physics-pane">
        <h2>Simulated 3D rollout</h2>
        <PhysicsScene rollout={rollout} time={time} />
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
        <strong>{rollout.validation.passed ? "P4 invariants pass" : "P4 validation failed"}</strong>
        <span>
          The line endpoint, body transform, and trajectory come from{" "}
          <code>SimulatedWorldState</code>.
        </span>
      </div>
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
