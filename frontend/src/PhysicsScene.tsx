import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { nearestSampleIndex } from "./obsMath";
import type { SimulatedWorldState } from "./physics";

type Props = {
  rollout: SimulatedWorldState;
  time: number;
};

function setRowMajor(matrix: THREE.Matrix4, values: number[]) {
  matrix.set(
    values[0], values[1], values[2], values[3],
    values[4], values[5], values[6], values[7],
    values[8], values[9], values[10], values[11],
    values[12], values[13], values[14], values[15],
  );
}

export function PhysicsScene({ rollout, time }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const bodyRef = useRef<THREE.Mesh | null>(null);
  const tetherRef = useRef<THREE.BufferGeometry | null>(null);
  const viewRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const body = rollout.bodies[0];
    const constraint = rollout.constraints[0];
    if (!canvas || !wrap || !body || !constraint) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    camera.position.set(4.5, 3.4, 5.0);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 1.5, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    scene.add(new THREE.AxesHelper(1));
    const grid = new THREE.GridHelper(8, 16, 0x44505c, 0x252b31);
    scene.add(grid);

    const anchor = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x30b4dc }),
    );
    anchor.position.fromArray(constraint.world_anchor_m);
    scene.add(anchor);

    const bodyMesh = new THREE.Mesh(
      new THREE.SphereGeometry(body.shape.radius_m, 24, 18),
      new THREE.MeshStandardMaterial({ color: 0xff6030, roughness: 0.45 }),
    );
    bodyMesh.matrixAutoUpdate = false;
    scene.add(bodyMesh);
    bodyRef.current = bodyMesh;

    const trajectoryPositions = new Float32Array(body.samples.length * 3);
    body.samples.forEach((sample, index) => {
      trajectoryPositions[index * 3] = sample.T_world_body[3];
      trajectoryPositions[index * 3 + 1] = sample.T_world_body[7];
      trajectoryPositions[index * 3 + 2] = sample.T_world_body[11];
    });
    const trajectoryGeometry = new THREE.BufferGeometry();
    trajectoryGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(trajectoryPositions, 3),
    );
    scene.add(
      new THREE.Line(
        trajectoryGeometry,
        new THREE.LineBasicMaterial({ color: 0xffb36b, transparent: true, opacity: 0.7 }),
      ),
    );

    const tetherGeometry = new THREE.BufferGeometry();
    tetherGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(6), 3),
    );
    scene.add(
      new THREE.Line(
        tetherGeometry,
        new THREE.LineBasicMaterial({ color: 0xe7eaee }),
      ),
    );
    tetherRef.current = tetherGeometry;

    const fit = () => {
      const width = wrap.clientWidth;
      const height = wrap.clientHeight;
      if (width === 0 || height === 0) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(wrap);
    const onControl = () => renderer.render(scene, camera);
    controls.addEventListener("change", onControl);
    viewRef.current = { scene, camera, controls };

    return () => {
      observer.disconnect();
      controls.removeEventListener("change", onControl);
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (
          object instanceof THREE.Mesh ||
          object instanceof THREE.Line ||
          object instanceof THREE.GridHelper
        ) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material.dispose();
        }
      });
      rendererRef.current = null;
      bodyRef.current = null;
      tetherRef.current = null;
      viewRef.current = null;
    };
  }, [rollout]);

  useEffect(() => {
    const view = viewRef.current;
    const renderer = rendererRef.current;
    const bodyMesh = bodyRef.current;
    const tetherGeometry = tetherRef.current;
    const body = rollout.bodies[0];
    const constraint = rollout.constraints[0];
    if (!view || !renderer || !bodyMesh || !tetherGeometry || !body || !constraint) return;

    const timestamps = rollout.timeline.samples.map((sample) => sample.timestamp_s);
    const index = nearestSampleIndex(timestamps, time);
    const sample = body.samples[index] ?? body.samples[0];
    setRowMajor(bodyMesh.matrix, sample.T_world_body);
    bodyMesh.matrixWorldNeedsUpdate = true;

    const attachment = new THREE.Vector3(...constraint.body_attachment_m);
    attachment.applyMatrix4(bodyMesh.matrix);
    const positions = tetherGeometry.getAttribute("position");
    positions.setXYZ(0, ...constraint.world_anchor_m);
    positions.setXYZ(1, attachment.x, attachment.y, attachment.z);
    positions.needsUpdate = true;
    tetherGeometry.computeBoundingSphere();
    renderer.render(view.scene, view.camera);
  }, [rollout, time]);

  return (
    <div className="scene-wrap physics-scene" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}
