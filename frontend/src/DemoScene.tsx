import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { positionAt, type DemoPath } from "./demo";

type Props = {
  observed: DemoPath;
  fitted: DemoPath;
  moon: DemoPath;
  time: number;
  showMoon: boolean;
};

function addPath(scene: THREE.Scene, path: DemoPath, color: number) {
  const positions = new Float32Array(path.positions_m.length * 3);
  path.positions_m.forEach((point, index) => {
    positions[index * 3] = point[0];
    positions[index * 3 + 1] = point[1];
    positions[index * 3 + 2] = point[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color }),
  );
  scene.add(line);
  return geometry;
}

export function DemoScene({ observed, fitted, moon, time, showMoon }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const observedBall = useRef<THREE.Mesh | null>(null);
  const fittedBall = useRef<THREE.Mesh | null>(null);
  const moonBall = useRef<THREE.Mesh | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101318);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 40);
    camera.position.set(1.6, 0.4, 1.1);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0.0, -0.1, -1.5);
    scene.add(new THREE.AmbientLight(0xffffff, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 0.7);
    key.position.set(2, 3, 1);
    scene.add(key);
    const grid = new THREE.GridHelper(3, 12, 0x3a4450, 0x232a31);
    grid.position.set(0, -0.55, -1.5);
    scene.add(grid);
    addPath(scene, observed, 0x30b4dc);
    addPath(scene, fitted, 0xff6030);
    addPath(scene, moon, 0xb9a0ff);
    const makeBall = (color: number) => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 24, 18),
        new THREE.MeshStandardMaterial({ color, roughness: 0.4 }),
      );
      scene.add(mesh);
      return mesh;
    };
    observedBall.current = makeBall(0x30b4dc);
    fittedBall.current = makeBall(0xff6030);
    moonBall.current = makeBall(0xb9a0ff);
    const resize = () => {
      const width = wrap.clientWidth;
      const height = wrap.clientHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(wrap);
    let frame = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(tick);
    };
    tick();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.dispose();
    };
  }, [observed, fitted, moon]);

  useEffect(() => {
    const place = (mesh: THREE.Mesh | null, path: DemoPath) => {
      if (!mesh) return;
      const point = positionAt(path, time);
      mesh.position.set(point[0], point[1], point[2]);
    };
    place(observedBall.current, observed);
    place(fittedBall.current, fitted);
    if (moonBall.current) {
      moonBall.current.visible = showMoon;
      place(moonBall.current, moon);
    }
  }, [time, observed, fitted, moon, showMoon]);

  return (
    <div className="demo-stage" ref={wrapRef}>
      <canvas ref={canvasRef} aria-label="Reconstructed and simulated 3D trajectories" />
    </div>
  );
}
