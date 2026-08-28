import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { nearestSampleIndex, opencvCameraToThreeWorld, verticalFovDeg } from "./obsMath";
import type { SceneObservation } from "./observation";

type Props = {
  observation: SceneObservation;
  artifactUrl: string;
  time: number;
};

function setRowMajor(matrix: THREE.Matrix4, values: number[]) {
  matrix.set(
    values[0],
    values[1],
    values[2],
    values[3],
    values[4],
    values[5],
    values[6],
    values[7],
    values[8],
    values[9],
    values[10],
    values[11],
    values[12],
    values[13],
    values[14],
    values[15],
  );
}

export function ObservationScene({ observation, artifactUrl, time }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const helperRef = useRef<THREE.CameraHelper | null>(null);
  const inspectCamRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const viewRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
    camera.position.set(1.6, 1.1, 2.4);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false;

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const axes = new THREE.AxesHelper(0.4);
    scene.add(axes);

    const cameraObj = observation.cameras[0];
    const poses = cameraObj.poses;
    const path = new THREE.BufferGeometry();
    const pathPositions = new Float32Array(poses.length * 3);
    for (let index = 0; index < poses.length; index += 1) {
      const threeMat = opencvCameraToThreeWorld(poses[index].T_world_camera);
      pathPositions[index * 3] = threeMat[3];
      pathPositions[index * 3 + 1] = threeMat[7];
      pathPositions[index * 3 + 2] = threeMat[11];
    }
    path.setAttribute("position", new THREE.BufferAttribute(pathPositions, 3));
    scene.add(new THREE.Line(path, new THREE.LineBasicMaterial({ color: 0x30b4dc })));

    const [width, height] = cameraObj.image_size_px;
    const inspectCam = new THREE.PerspectiveCamera(
      verticalFovDeg(cameraObj.intrinsics.fy_px, height),
      width / height,
      0.05,
      1.6,
    );
    inspectCam.matrixAutoUpdate = false;
    const helper = new THREE.CameraHelper(inspectCam);
    scene.add(inspectCam);
    scene.add(helper);
    inspectCamRef.current = inspectCam;
    helperRef.current = helper;

    const loader = new GLTFLoader();
    loader.load(artifactUrl, (gltf) => {
      scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      if (!box.isEmpty()) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        controls.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.4, size * 0.8));
        camera.near = Math.max(size / 400, 0.01);
        camera.far = Math.max(size * 20, 20);
        camera.updateProjectionMatrix();
        inspectCam.far = Math.max(size * 0.35, 0.4);
        inspectCam.updateProjectionMatrix();
        helper.update();
      }
      renderer.render(scene, camera);
    });

    const fit = () => {
      const boxW = wrap.clientWidth;
      const boxH = wrap.clientHeight;
      if (boxW === 0 || boxH === 0) return;
      renderer.setSize(boxW, boxH, false);
      camera.aspect = boxW / boxH;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(wrap);
    viewRef.current = { scene, camera, controls };

    const onControl = () => renderer.render(scene, camera);
    controls.addEventListener("change", onControl);

    return () => {
      observer.disconnect();
      controls.removeEventListener("change", onControl);
      controls.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Points) {
          obj.geometry.dispose();
          const material = obj.material;
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material.dispose();
        }
      });
      rendererRef.current = null;
      helperRef.current = null;
      inspectCamRef.current = null;
      viewRef.current = null;
    };
  }, [artifactUrl, observation]);

  useEffect(() => {
    const view = viewRef.current;
    const inspectCam = inspectCamRef.current;
    const helper = helperRef.current;
    const renderer = rendererRef.current;
    if (!view || !inspectCam || !helper || !renderer) return;
    const cameraObj = observation.cameras[0];
    const timestamps = observation.timeline.samples.map((sample) => sample.timestamp_s);
    const sampleIndex = nearestSampleIndex(timestamps, time);
    const pose = cameraObj.poses.find((item) => item.sample_index === sampleIndex) ?? cameraObj.poses[0];
    setRowMajor(inspectCam.matrix, opencvCameraToThreeWorld(pose.T_world_camera));
    inspectCam.matrixWorldNeedsUpdate = true;
    helper.update();
    renderer.render(view.scene, view.camera);
  }, [time, observation]);

  return (
    <div className="scene-wrap" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}
