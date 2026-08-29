import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { nearestSampleIndex, opencvCameraToThreeWorld, verticalFovDeg } from "./obsMath";
import {
  SMPL24_BONES,
  humanSampleAt,
  readHumans,
  type SceneObservation,
} from "./observation";

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
  const humansRef = useRef<THREE.Group | null>(null);
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

    const humans = readHumans(observation);
    const humanGroup = new THREE.Group();
    humanGroup.name = "humans";
    scene.add(humanGroup);
    humansRef.current = humanGroup;
    if (humans) {
      for (const person of humans.people) {
        const sample = person.samples[0];
        if (!sample) continue;
        const positions = new Float32Array(SMPL24_BONES.length * 6);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const lines = new THREE.LineSegments(
          geometry,
          new THREE.LineBasicMaterial({ color: 0xff8c30 }),
        );
        lines.userData.personId = person.id;
        lines.userData.kind = "bones";
        humanGroup.add(lines);
        const joints = new THREE.InstancedMesh(
          new THREE.SphereGeometry(0.045, 10, 8),
          new THREE.MeshBasicMaterial({ color: 0xffc078 }),
          24,
        );
        joints.userData.personId = person.id;
        joints.userData.kind = "joints";
        humanGroup.add(joints);
        writeSkeleton(positions, sample.joints);
        writeJointInstances(joints, sample.joints);
        geometry.computeBoundingSphere();
        geometry.attributes.position.needsUpdate = true;
        lines.frustumCulled = false;
        joints.frustumCulled = false;
      }
    }

    const frameScene = (root: THREE.Object3D) => {
      const box = new THREE.Box3().setFromObject(root);
      if (humans) {
        box.expandByObject(humanGroup);
      }
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.4, size * 0.8));
      camera.near = Math.max(size / 400, 0.01);
      camera.far = Math.max(size * 20, 20);
      camera.updateProjectionMatrix();
      inspectCam.far = Math.max(size * 0.8, 2.4);
      inspectCam.updateProjectionMatrix();
      helper.update();
    };

    if (humans) {
      const sample = humans.people[0]?.samples[0];
      const pelvis = sample?.joints[0];
      const head = sample?.joints[15] ?? pelvis;
      if (pelvis && head) {
        const target = new THREE.Vector3(pelvis[0], (pelvis[1] + head[1]) / 2, pelvis[2]);
        controls.target.copy(target);
        camera.position.set(target.x + 2.0, target.y + 0.85, target.z + 1.6);
        camera.near = 0.05;
        camera.far = 50;
        camera.updateProjectionMatrix();
        inspectCam.far = 8;
        inspectCam.updateProjectionMatrix();
        helper.update();
      } else {
        frameScene(humanGroup);
      }
    }
    const loader = new GLTFLoader();
    loader.load(
      artifactUrl,
      (gltf) => {
        scene.add(gltf.scene);
        if (!humans) frameScene(gltf.scene);
        renderer.render(scene, camera);
      },
      undefined,
      () => {
        renderer.render(scene, camera);
      },
    );

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
      humansRef.current = null;
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
    const humans = readHumans(observation);
    const humanGroup = humansRef.current;
    if (humans && humanGroup) {
      for (const child of humanGroup.children) {
        const person = humans.people.find((item) => item.id === child.userData.personId);
        const sample = person ? humanSampleAt(person, sampleIndex) : null;
        if (!sample) continue;
        if (child instanceof THREE.LineSegments) {
          const attribute = child.geometry.getAttribute("position");
          if (!(attribute instanceof THREE.BufferAttribute)) continue;
          writeSkeleton(attribute.array as Float32Array, sample.joints);
          attribute.needsUpdate = true;
          child.geometry.computeBoundingSphere();
        }
        if (child instanceof THREE.InstancedMesh) {
          writeJointInstances(child, sample.joints);
        }
      }
    }
    renderer.render(view.scene, view.camera);
  }, [time, observation]);

  return (
    <div className="scene-wrap" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function writeJointInstances(
  mesh: THREE.InstancedMesh,
  joints: Array<[number, number, number]>,
) {
  const matrix = new THREE.Matrix4();
  for (let index = 0; index < 24; index += 1) {
    const joint = joints[index];
    if (!joint) continue;
    matrix.makeTranslation(joint[0], joint[1], joint[2]);
    mesh.setMatrixAt(index, matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

function writeSkeleton(target: Float32Array, joints: Array<[number, number, number]>) {
  for (let index = 0; index < SMPL24_BONES.length; index += 1) {
    const [start, end] = SMPL24_BONES[index];
    const a = joints[start];
    const b = joints[end];
    const offset = index * 6;
    if (!a || !b) continue;
    target[offset] = a[0];
    target[offset + 1] = a[1];
    target[offset + 2] = a[2];
    target[offset + 3] = b[0];
    target[offset + 4] = b[1];
    target[offset + 5] = b[2];
  }
}
