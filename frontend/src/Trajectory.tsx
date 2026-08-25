import { useEffect, useRef } from "react";
import { interpolate, OBS_COLOR, SIM_COLOR } from "./geom";
import type { Point } from "./types";

type Props = {
  observed: Point[];
  simulated: Point[];
  time: number;
};

function drawSeries(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  pick: (p: Point) => number,
  x0: number,
  y0: number,
  w: number,
  h: number,
  t0: number,
  t1: number,
  yMin: number,
  yMax: number,
  color: string,
) {
  if (points.length === 0) return;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = x0 + ((p.t - t0) / Math.max(t1 - t0, 1e-6)) * w;
    const y = y0 + h - ((pick(p) - yMin) / Math.max(yMax - yMin, 1e-6)) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

export function TrajectoryPlot({ observed, simulated, time }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const render = () => {
      const dpr = Math.min(window.devicePixelRatio, 2);
      const width = parent.clientWidth;
      const height = Math.max(220, Math.floor(parent.clientWidth * 0.28));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#14171b";
      ctx.fillRect(0, 0, width, height);

      const all = [...observed, ...simulated];
      if (all.length === 0) return;
      const t0 = all[0].t;
      const t1 = all[all.length - 1].t;
      const ys = all.map((p) => p.y);
      const xs = all.map((p) => p.x);
      const pad = 42;
      const gap = 16;
      const plotW = width - pad - 16;
      const plotH = (height - 28 - gap) / 2;

      ctx.fillStyle = "#9aa3ad";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText("y (px, +down)", pad, 14);
      ctx.fillText("x (px)", pad, 14 + plotH + gap);

      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);

      ctx.strokeStyle = "#2a2f36";
      ctx.strokeRect(pad, 18, plotW, plotH);
      ctx.strokeRect(pad, 18 + plotH + gap, plotW, plotH);

      drawSeries(ctx, observed, (p) => p.y, pad, 18, plotW, plotH, t0, t1, yMin, yMax, OBS_COLOR);
      drawSeries(ctx, simulated, (p) => p.y, pad, 18, plotW, plotH, t0, t1, yMin, yMax, SIM_COLOR);
      drawSeries(
        ctx,
        observed,
        (p) => p.x,
        pad,
        18 + plotH + gap,
        plotW,
        plotH,
        t0,
        t1,
        xMin,
        xMax,
        OBS_COLOR,
      );
      drawSeries(
        ctx,
        simulated,
        (p) => p.x,
        pad,
        18 + plotH + gap,
        plotW,
        plotH,
        t0,
        t1,
        xMin,
        xMax,
        SIM_COLOR,
      );

      const markerX = pad + ((time - t0) / Math.max(t1 - t0, 1e-6)) * plotW;
      ctx.strokeStyle = "#e7eaee";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(markerX, 18);
      ctx.lineTo(markerX, 18 + plotH);
      ctx.moveTo(markerX, 18 + plotH + gap);
      ctx.lineTo(markerX, 18 + plotH + gap + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      const obs = interpolate(observed, time);
      const sim = interpolate(simulated, time);
      const simLabel = `sim y=${sim.y.toFixed(0)} x=${sim.x.toFixed(0)}`;
      const obsLabel = `obs y=${obs.y.toFixed(0)} x=${obs.x.toFixed(0)}`;
      ctx.fillStyle = SIM_COLOR;
      ctx.textAlign = "right";
      ctx.fillText(simLabel, pad + plotW, 14);
      ctx.fillStyle = OBS_COLOR;
      ctx.fillText(obsLabel, pad + plotW - ctx.measureText(simLabel).width - 14, 14);
      ctx.textAlign = "left";
    };

    render();
    const observer = new ResizeObserver(render);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [observed, simulated, time]);

  return <canvas ref={ref} className="traj-canvas" />;
}
