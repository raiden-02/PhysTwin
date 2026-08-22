import { useEffect, useRef } from "react";
import * as THREE from "three";
import { interpolate, OBS_COLOR, SIM_COLOR } from "./geom";
import type { Reconstruction, Tracking } from "./types";

type Props = {
  video: HTMLVideoElement | null;
  tracking: Tracking;
  reconstruction: Reconstruction;
  time: number;
};

function lineFromPoints(
  points: { x: number; y: number }[],
  color: string,
  z: number,
): THREE.Line {
  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i += 1) {
    positions[i * 3] = points[i].x;
    positions[i * 3 + 1] = -points[i].y;
    positions[i * 3 + 2] = z;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(geometry, material);
}

export function ReconstructionScene({ video, tracking, reconstruction, time }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ballRef = useRef<THREE.Mesh | null>(null);
  const obsBallRef = useRef<THREE.Mesh | null>(null);
  const rodRef = useRef<THREE.Line | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const width = tracking.frame_width;
    const height = tracking.frame_height;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);

    // Image space: x right, y down, so world y is -pixel y. The frustum below
    // is in absolute image coordinates, so the camera must stay at x=y=0.
    // Translating it too would offset the whole scene by half a frame.
    const camera = new THREE.OrthographicCamera(0, width, 0, -height, 0.1, 100);
    camera.position.set(0, 0, 20);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;

    let texture: THREE.VideoTexture | null = null;
    if (video) {
      texture = new THREE.VideoTexture(video);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.colorSpace = THREE.SRGBColorSpace;
      // Dimmed so the fitted trajectory and the two markers read as the
      // simulation, not as a second copy of the recording.
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(width, height),
        new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.35 }),
      );
      plane.position.set(width / 2, -height / 2, 0);
      scene.add(plane);
      const onSeeked = () => {
        if (texture) texture.needsUpdate = true;
      };
      video.addEventListener("seeked", onSeeked);
      const cleanupSeek = () => video.removeEventListener("seeked", onSeeked);
      (plane.userData as { cleanupSeek?: () => void }).cleanupSeek = cleanupSeek;
    }

    scene.add(lineFromPoints(tracking.observations, OBS_COLOR, 1.2));
    scene.add(lineFromPoints(reconstruction.simulated, SIM_COLOR, 1.4));

    if (reconstruction.model === "projectile_bounce") {
      const yGround = reconstruction.environment.y_ground;
      scene.add(
        lineFromPoints(
          [
            { x: 0, y: yGround },
            { x: width, y: yGround },
          ],
          "#8b9098",
          1.1,
        ),
      );
    } else {
      const { pivot_x: pivotX, pivot_y: pivotY, radius: swingRadius } =
        reconstruction.environment;
      const pivot = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(6, Math.min(16, swingRadius * 0.04)), 20, 12),
        new THREE.MeshBasicMaterial({ color: "#f2d96b" }),
      );
      pivot.position.set(pivotX, -pivotY, 3.2);
      scene.add(pivot);

      const first = reconstruction.simulated[0];
      const rod = lineFromPoints(
        [
          { x: pivotX, y: pivotY },
          { x: first.x, y: first.y },
        ],
        "#d5d9de",
        2.2,
      );
      scene.add(rod);
      rodRef.current = rod;
    }

    const radius =
      tracking.observations.find((p) => p.radius && p.radius > 1)?.radius ?? 18;
    const proxyRadius = Math.max(7, Math.min(28, radius * 0.2));
    const simBall = new THREE.Mesh(
      new THREE.SphereGeometry(proxyRadius, 24, 16),
      new THREE.MeshBasicMaterial({ color: SIM_COLOR }),
    );
    const obsBall = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(5, 0.7 * proxyRadius), 16, 12),
      new THREE.MeshBasicMaterial({ color: OBS_COLOR }),
    );
    scene.add(simBall);
    scene.add(obsBall);
    ballRef.current = simBall;
    obsBallRef.current = obsBall;

    const fit = () => {
      const boxW = wrap.clientWidth;
      const boxH = wrap.clientHeight;
      if (boxW === 0 || boxH === 0) return;
      // updateStyle=false because CSS pins the canvas to the wrapper. Sizing
      // the buffer from that same box keeps both in the same aspect ratio.
      renderer.setSize(boxW, boxH, false);
      const scale = Math.min(boxW / width, boxH / height);
      const viewW = boxW / scale;
      const viewH = boxH / scale;
      const extraX = Math.max(0, (viewW - width) / 2);
      const extraY = Math.max(0, (viewH - height) / 2);
      camera.left = -extraX;
      camera.right = width + extraX;
      camera.top = extraY;
      camera.bottom = -height - extraY;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(wrap);

    const state = { scene, camera, renderer, texture };
    (canvas as HTMLCanvasElement & { __pt?: typeof state }).__pt = state;

    return () => {
      observer.disconnect();
      const plane = scene.children.find((c) => c.userData?.cleanupSeek);
      if (plane?.userData?.cleanupSeek) plane.userData.cleanupSeek();
      texture?.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry.dispose();
          const material = obj.material;
          if (Array.isArray(material)) material.forEach((m) => m.dispose());
          else material.dispose();
        }
      });
      rendererRef.current = null;
      ballRef.current = null;
      obsBallRef.current = null;
      rodRef.current = null;
    };
  }, [video, tracking, reconstruction]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const canvas = canvasRef.current as
      | (HTMLCanvasElement & { __pt?: { scene: THREE.Scene; camera: THREE.Camera } })
      | null;
    if (!renderer || !canvas?.__pt) return;
    const sim = interpolate(reconstruction.simulated, time);
    const obs = interpolate(tracking.observations, time);
    if (ballRef.current) {
      ballRef.current.position.set(sim.x, -sim.y, 3);
    }
    if (obsBallRef.current) {
      obsBallRef.current.position.set(obs.x, -obs.y, 2.6);
    }
    if (rodRef.current && reconstruction.model === "pendulum") {
      const positions = rodRef.current.geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;
      positions.setXYZ(
        0,
        reconstruction.environment.pivot_x,
        -reconstruction.environment.pivot_y,
        2.2,
      );
      positions.setXYZ(1, sim.x, -sim.y, 2.2);
      positions.needsUpdate = true;
      rodRef.current.geometry.computeBoundingSphere();
    }
    renderer.render(canvas.__pt.scene, canvas.__pt.camera);
  }, [time, reconstruction, tracking]);

  return (
    <div className="scene-wrap" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}
