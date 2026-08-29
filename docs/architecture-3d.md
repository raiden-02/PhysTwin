# PhysTwin 3D migration architecture

## P0 status

P0 freezes the working 2D product and defines the boundary for the 3D path.
It does not reconstruct a 3D scene or run a 3D physics engine.

- V1 baseline branch: `main`
- V1 baseline commit: `b14b9a95f676e571e4b096f643663ef76cf34e03`
- Pivot branch: `feat/3d-video-to-sim-pivot`
- Legacy V1 architecture: [architecture.md](architecture.md)
- New contract examples: [`contracts/3d/v1`](../contracts/3d/v1/)

The pivot stays in this repository. The branch is a direct descendant of the
V1 baseline so every later checkpoint can compare against known working code.

## Why the canonical state changes

V1 proves the full product loop:

```text
video
  -> SAM 2 pixel track
  -> selected 2D dynamics family
  -> deterministic C++ fit and rollout
  -> measured error
  -> synchronized Three.js view
```

That loop is useful and remains supported. Its physical state is still tied to
pixels and to one hand-written model family. Camera rotation, perspective,
depth, articulation, contacts, active control, and changing constraints cannot
be represented by adding more terms to a 2D pendulum or projectile equation.

The new north-star is:

```text
video(s)
  -> reconstruction adapter
  -> SceneObservation
  -> physical inference
  -> PhysicalScene
  -> GPU simulator
  -> SimulatedWorldState / rollout
  -> 3D and reprojection evaluation
  -> Three.js inspection and counterfactual controls
```

The architectural rule is:

> 2D is evidence, not the physical state.

Masks, keypoints, pixels, and reprojection errors remain valuable. They do not
define the state integrated by the future simulator.

## Two separate canonical concepts

### `SceneObservation`

`SceneObservation` records reconstructed visual evidence. It does not select a
physics model.

The P0 contract contains only the stable outer boundary:

- source videos and their content identities;
- one shared timeline with original source frame numbers;
- camera intrinsics and world-from-camera poses;
- references to large geometry artifacts;
- the coordinate, scale, and transform conventions;
- estimator and adapter provenance;
- namespaced extension data.

Large point clouds, meshes, masks, and estimator-native arrays stay in
referenced artifacts. The JSON document remains small enough to inspect,
validate, cache, and pass between languages.

Later checkpoints may add namespaced extensions for:

- object geometry and `T_world_entity(t)`;
- articulated-human root, joints, and body pose;
- 2D masks, keypoints, and reprojection evidence;
- confidence, covariance, and estimator-specific uncertainty.

Those payloads are not frozen in P0 because no integrated estimator has
justified their exact shape yet. Unknown extension keys cannot change the
meaning of the core fields.

### `PhysicalScene`

`PhysicalScene` records one physical explanation to test. It is not perception
output and it does not overwrite `SceneObservation`.

The P0 contract freezes:

- the link to the source observation;
- SI units for executable simulation;
- observation-to-physical scale and rigid alignment;
- execution status and blockers;
- containers for bodies, articulations, constraints, contacts, materials,
  forces, actuators, and residual forces;
- fitted or assumed parameters and bounds;
- provenance.

P0 does not define Newton-specific body or joint payloads. P4 will define the
first executable component types against the actual simulator API.

An unresolved scene can still be serialized as a draft. It cannot be marked
`executable`.

## Coordinate ownership

### Time

Canonical time is in seconds. `timestamp_s = 0` is the first kept sample in
the observation, not Unix time and not necessarily source frame zero. Samples
are strictly increasing. Each sample keeps the original frame number for every
source that contributed evidence.

Consumers use the stored timestamps. They do not rebuild time as `frame / fps`.
A future multi-view adapter must synchronize every source onto this shared
timeline and record how synchronization was established.

### Observation world coordinates

`SceneObservation` uses a deterministic reconstruction gauge:

- right-handed;
- the first valid camera pose defines the world frame;
- `+X` points to the first camera's right;
- `+Y` points to the first camera's image up;
- `+Z` points backward from the first camera.

This is a graphics-style first-camera frame. It does not claim that `+Y` is
gravity-up. A reconstruction adapter must convert estimator-native output to
this frame before writing the canonical contract.

The first camera uses OpenCV camera coordinates:

- `+X` right;
- `+Y` down;
- `+Z` forward.

Its initial world-from-camera rotation is therefore:

```text
diag(1, -1, -1)
```

### Image coordinates

Image evidence keeps the OpenCV convention:

- pixel origin at the center of the upper-left pixel;
- `u` increases right;
- `v` increases down;
- intrinsics are in pixels.

For a pinhole camera:

```text
p_camera = inverse(T_world_camera) * p_world

u = fx * X / Z + skew * Y / Z + cx
v = fy * Y / Z + cy
```

Only camera-space points with `Z > 0` are in front of the camera. Lens
distortion must be declared or removed before using the pinhole projection.
P1 must not silently treat distorted pixels as undistorted.

### Transform notation and storage

`T_A_B` maps a point from frame `B` into frame `A`:

```text
p_A = T_A_B * [p_B.x, p_B.y, p_B.z, 1]^T
```

Transforms are rigid 4x4 matrices. JSON stores the 16 values in row-major
order. Math uses column vectors. Translation is at flat indices `3`, `7`, and
`11`. Quaternions, if added later, use `[x, y, z, w]`.

Adapters own conversion from estimator-native coordinates into this contract.
Simulator adapters own conversion from `PhysicalScene` into backend-native
coordinates. Neither estimator nor simulator conventions leak across the
canonical boundaries.

### Three.js conversion

The canonical observation world already matches the Three.js world basis:
`+X` right, `+Y` up, and camera forward along `-Z`.

Three.js camera-local coordinates differ from OpenCV. Convert with:

```text
F = diag(1, -1, -1, 1)
T_world_three_camera = T_world_camera * F
```

Contract matrices are row-major. `THREE.Matrix4.fromArray()` reads a flat
array as column-major, so the frontend must use `Matrix4.set(...values)` or an
equivalent explicit conversion.

Three.js consumes canonical observation and rollout data. It never becomes the
source of camera poses, physical state, constraints, or simulation results.

### Physical coordinates and units

An executable `PhysicalScene` uses:

- right-handed coordinates;
- `+Y` as physical up;
- meters;
- kilograms;
- seconds;
- radians.

`SceneObservation` scale is one of:

- `relative`: no metric conversion is known;
- `metric_measured`: a measured calibration provides meters per world unit;
- `metric_assumed`: an explicit assumption provides meters per world unit.

Relative scale must keep `meters_per_world_unit` as `null`.

The physical alignment is:

```text
p_scene_m =
  T_scene_observation_m
  * [meters_per_observation_unit * p_observation, 1]^T
```

Scale is applied before the rigid transform. The transform translation is in
meters. If scale or alignment is unresolved, both remain `null` and
`PhysicalScene.execution.status` remains `draft`.

An assumed scale is valid for a clearly labelled counterfactual. It is not a
recovered metric measurement.

## Module boundaries

P0 adds only this path:

```text
vision/reconstruction/
  adapter.py       estimator-independent request/result protocol
  contracts.py     canonical validation, hashing, transforms, projection

contracts/3d/v1/
  README.md
  examples/
```

The reconstruction adapter accepts source identities and explicit options. It
writes heavy artifacts under a work directory and returns a canonical
`SceneObservation` payload plus runtime metadata. The caller validates and
serializes that payload.

There is no plugin registry. P1 can instantiate one DA3 adapter directly.
Another estimator can implement the same narrow protocol later.

## Offline execution and cache

The target machine has one RTX 4080-class GPU. Large stages run sequentially:

```text
load one estimator
  -> run it
  -> validate and cache artifacts
  -> release GPU memory
  -> start the next stage
```

The reconstruction cache key is SHA-256 over canonical JSON containing:

- contract schema and version;
- adapter name and version;
- model revision and weights hash;
- ordered source IDs and content hashes;
- reconstruction options.

Paths and wall-clock timestamps are not part of the key. Mapping keys are
sorted, UTF-8 is used, and non-finite JSON numbers are rejected.

P1 should store entries under:

```text
results/cache/reconstruction/<cache-key>/
```

It should write into a temporary sibling directory, validate and hash every
artifact, write the canonical document, and publish a completion marker last.
A partial or hash-mismatched entry is a cache miss. Cached outputs are
immutable.

P0 defines the identity function but does not implement model execution or a
cache manager.

## V1 reuse decisions

### Reuse unchanged

- Build scripts, CMake layout, deterministic CTest suite, and V1 CLI.
- Existing projectile and pendulum implementations and all saved evaluations.
- Local-only FastAPI process and single-GPU execution lock.
- Video metadata, frame extraction, progress events, upload, and artifact
  serving patterns.
- React timeline, video playback, scrubbing, and stage presentation.
- Three.js as the inspection renderer.
- Evaluation principles: saved artifacts, explicit negative controls, measured
  error, and no success claim from a poor fit.

### Reuse behind an adapter

- SAM 2 masks and tracks become optional image evidence. Their centroids are
  not 3D physical state.
- FastAPI process orchestration should call explicit reconstruction and physics
  stages instead of one model-selected V1 function.
- Frontend data loading should gain a 3D result adapter while preserving the
  existing V1 result view.
- Plotting and evaluation helpers can keep timing, provenance, and artifact
  reporting patterns, but 3D and reprojection metrics need new inputs.
- `nlohmann/json`, explicit file boundaries, deterministic serialization, and
  validation-first loading are reusable. The current `load_tracking()` and
  model-specific reconstruction writers remain on the V1 path.
- Fixed-seed search, bounded parameter fitting, robust loss, and explicit
  negative controls are reusable numerical approaches. `Fitter` and
  `PendulumFitter` are not generic optimizers and stay with their V1 models.
- Timestamp-based sampling is reusable. The 2D `Observation {x, y}` and
  index-paired `metrics.cpp` functions are not 3D contracts.

### Legacy V1 only

- `tracking.json` as a model-selected physics input.
- Pixel centroids as canonical physical state.
- `DynamicsModel { projectile_bounce, pendulum }` as the top-level product
  hypothesis space.
- `phystwin fit` dispatch through that enum. Future 3D dispatch selects a
  `PhysicalScene` backend and component support, not an image-space equation.
- Pixel gravity, one horizontal image ground, image pendulum radius, and
  tracked-anchor translation compensation.
- The orthographic image-plane Three.js reconstruction.

These remain supported regression evidence. They are not extended with more
2D model families.

### Replace later

- The direct `track.py -> phystwin fit` FastAPI pipeline for new 3D jobs.
- Model-specific TypeScript result unions as the only frontend result contract.
- 2D-only fit grading when 3D evidence becomes available.
- The current hand-authored simulator once `PhysicalScene` execution is
  available through Newton/Warp.

Replacement is additive until a later checkpoint has equivalent evidence.

### Uncertain until measured

- DA3 output quality, memory use, camera convention, and license fit.
- TRAM live-inference quality on real clips (P2 only locks conversion and UI).
- GLB mesh versus point cloud as the first inspection artifact.
- OpenUSD as interchange after simulator integration.
- Newton versus MuJoCo Warp for later articulated cases.
- Which confidence values are calibrated enough to affect fitting.

## V1 audit findings left unchanged

P0 preserves V1 behavior, including a few known contract and reporting gaps:

- `tracking.json` allows a missing `model` and defaults to
  `projectile_bounce`. Current Python output always writes the field.
- Python writes tracked-anchor coverage, but the C++ loader does not require
  it. A hand-authored tracked input can omit coverage and later report the
  default `1.0`.
- Tracker runtime is labelled as including JPEG decode, but its timer starts
  after frame extraction and checkpoint handling. Existing values are useful
  for regression comparison, not full upload-to-result latency.
- `scripts/test.ps1` runs CTest only. Python checks and the frontend build are
  separate commands.
- Gitignored root `results/` files include older contract variants. The
  committed `docs/evaluation.json` and its referenced case artifacts are the
  V1 evidence index.

These are not 3D contract semantics. Fix them in a focused V1 maintenance
change if they become blockers. Do not silently reinterpret old evidence.

## P1 status

P1 adds one reconstruction path on this branch:

```text
short real video
  -> vision/reconstruction/da3.py
  -> cached SceneObservation
  -> artifacts/scene.glb point cloud
  -> recovered camera path
  -> Three.js inspection beside the recording
```

The V1 `/api/jobs` loop is unchanged. The UI default remains the 2D physics
twin. `3D scene + camera` is a separate mode that calls `/api/observations`.

### Pinned DA3

- Code package: `depth-anything-3` from
  `https://github.com/ByteDance-Seed/Depth-Anything-3`
- Code revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Code license: Apache-2.0
- Weights: `depth-anything/DA3-BASE`
- Weights revision: `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`
- Weights license: Apache-2.0
- Adapter: `vision/reconstruction/da3.py`, version `1.0.0`

DA3-LARGE and DA3NESTED-GIANT-LARGE are CC BY-NC 4.0. P1 does not use them.
Install extras with `scripts/setup-reconstruction.ps1`. Do not replace the V1
venv.

### Estimator convention and conversion

DA3 extrinsics are OpenCV / COLMAP world-to-camera matrices, shape `(N, 3, 4)`.
Intrinsics and depth are at the processed resolution, not the source pixels.

The adapter:

1. inverts each `w2c` to `T_native_camera`;
2. gauges the first camera with `T_obs_from_native = F * inverse(T_native_camera0)`
   where `F = diag(1, -1, -1, 1)`;
3. writes `T_world_camera_i = T_obs_from_native * T_native_camera_i`;
4. unprojects confident depth in OpenCV camera space, then applies the same
   gauge to the points;
5. scales the first-sample `K` to source pixels for the contract.

The saved fixture `contracts/3d/v1/examples/da3_w2c_fixture.json` locks that
conversion. P1 does not use DA3's GLB exporter. That exporter also centers the
point cloud and would move the first-camera origin.

### P1 decisions

- Intrinsics: one camera object stores the first sample, scaled to source
  pixels. If later samples differ, `extensions.phystwin.da3.v1.intrinsics_vary`
  is true. The core `intrinsics` field still comes from sample 0.
- Lens distortion: declared `unknown`. P1 does not treat source pixels as
  undistorted.
- Geometry artifact: colored point-cloud GLB in observation world coordinates.
  Kind is `point_cloud`.
- Time: container `CAP_PROP_POS_MSEC` when those values are finite and strictly
  increasing. Otherwise `frame / fps`. The source is recorded on the samples.
- Missing poses: a missing first pose fails the adapter. Later missing poses
  are not filled in.
- Scale: `relative`. `meters_per_world_unit` stays `null`.
- Default clip window: first 2 seconds, at most 12 frames, `ref_view_strategy`
  `middle`.
- Cache: `results/cache/reconstruction/<sha256>/` with a sibling temp directory
  and a `COMPLETE` marker written last.

P1 does not add physical inference, Newton/Warp, metric scale claims, inverse
physics, or cinematic tuning. Human reconstruction is P2.

## P2 status

P2 adds one human path on this branch:

```text
TRAM-native camera + camera-space SMPL24 joints
  -> vision/reconstruction/tram.py
  -> extensions.phystwin.humans.v1
  -> same first-camera graphics world as P1
  -> Three.js stick figure synced to the source video
```

The V1 `/api/jobs` loop and the P1 DA3 path stay unchanged.

### License and output comparison

| Estimator | Code license | Output used here | Why accepted or rejected |
| --- | --- | --- | --- |
| TRAM (`yufu-wang/tram`) | MIT | OpenCV `pred_cam_R` / `pred_cam_T` as camera-to-world, VIMO joints in camera space | Chosen. Commercial-friendly code license. Same OpenCV camera basis as P1. |
| GVHMR (`zju3dv/GVHMR`) | Non-commercial research only | Gravity-view then world via camera rotation | Rejected. Needs written permission for commercial use. |
| PromptHMR | Non-commercial research only | Wraps TRAM | Rejected as the P2 estimator. Do not install it for this path. |

Pinned TRAM code revision: `4861c112f3c148201326680a50c9199650da6088`.
Adapter: `vision/reconstruction/tram.py`, version `1.0.0`.

SMPL / SMPLify body weights are third-party and are not downloaded by PhysTwin.
Live TRAM also needs DROID-SLAM, Detectron2, and VIMO in a separate Linux/conda
environment. This venv only converts TRAM-native files or the committed fixture.

### Estimator convention and conversion

TRAM camera.npy stores OpenCV camera-to-world poses:

```text
P_native = pred_cam_R @ P_camera + pred_cam_T
```

The adapter uses `pred_cam_*`, not later `world_cam_*` gravity/floor alignment.

The gauge is the same as P1:

1. build `T_native_camera` from each `R`, `t`;
2. `T_obs_from_native = F * inverse(T_native_camera0)` where `F = diag(1, -1, -1, 1)`;
3. `T_world_camera_i = T_obs_from_native * T_native_camera_i`;
4. `p_obs = T_world_camera * p_camera`.

`contracts/3d/v1/examples/tram_c2w_fixture.json` locks that conversion.

If a P1 `SceneObservation` is supplied, cameras stay as they are. Camera-space
joints are lifted through that observation's `T_world_camera`. The world basis
matches P1. DA3 relative scale and TRAM metric camera-space units can still
disagree. That mismatch is recorded on provenance. It is not silently "fixed."

Standalone TRAM cameras are labelled `metric_assumed` with
`meters_per_world_unit = 1`. That is ZoeDepth / fixture assumption, not a
measured calibration.

### humans.v1

Body evidence lives under `extensions.phystwin.humans.v1`. Core observation
fields do not change meaning.

- `joint_layout`: `smpl24`
- `coordinate_frame`: `observation_world`
- each person sample stores `root` (pelvis), 24 joints, and `sample_index`

CLI: `vision/reconstruct_humans.py --from-fixture` or `--tram-dir <seq>`.
Cache: `results/cache/humans/<sha256>/`.
UI: **Inspect P2 human fixture** draws a projected skeleton video next to the
3D body. **Attach TRAM humans** imports an official results folder onto a P1
observation.

P2 does not add Newton/Warp, `PhysicalScene` execution, inverse physics, or
EMDB evaluation.

## P3 status

```text
P3 evaluator implementation      COMPLETE
P3 synthetic validation          COMPLETE
EMDB adapter/tooling             RETAINED
EMDB measured benchmark          OPTIONAL / UNAVAILABLE
physics development              NOT BLOCKED
```

The evaluator is `vision/reconstruction/evaluation.py`. The CLI is
`vision/evaluate_reconstruction.py`. Full usage and metric definitions are in
[`reconstruction-evaluation-p3.md`](reconstruction-evaluation-p3.md).

The synthetic fixture is a known `0.05 m` body translation. That check is
complete. It is not an EMDB benchmark.

EMDB code is MIT and pinned at
`9a4eab677181a3789bda7ba5c36ab8cff797380c`. The dataset is restricted to
approved non-commercial academic use and requires an institutional email.
SMPL model files have separate registration terms. The adapter stays in the
repo. There is no approved EMDB sequence here, and that is no longer a
project blocker.

P3 does not add Newton/Warp, physics fitting, `PhysicalScene` execution, or
counterfactual controls.

## P4 status

P4 adds one executable physical scene without changing reconstruction:

```text
physical_scene_tether.json
  -> .venv-physics Python subprocess
  -> Newton 1.5.1 SolverXPBD on Warp 1.16.0 CUDA
  -> phystwin.simulated_world_state version 1
  -> Three.js anchor, tether, body, trajectory, and playback
```

The input is a standalone SI-unit scene. It has one 1 kg sphere, gravity,
one world anchor, one body-local attachment, and one fixed 2 m distance.
Newton's native `ModelBuilder.add_joint_distance` represents the tether.
`SolverXPBD` is used because Newton 1.5.1's support table lists it as the only
solver that enforces `DISTANCE`.

The simulator adapter keeps the project world as right-handed and `+Y` up.
It constructs Newton's `ModelBuilder` with `newton.Axis.Y` and passes gravity
directly, so no axis swap is needed. The adapter converts row-major
`T_world_body` matrices to XYZW quaternions at the input boundary and converts
Newton body transforms back to row-major matrices for the rollout.

P4 runs in `.venv-physics`. FastAPI stays in `.venv` and invokes physics
through JSON files and a subprocess. See
[`physics-runtime-p4.md`](physics-runtime-p4.md) for setup, versions,
licenses, exact payload, validation, and measured runtime.

P4 does not fit the scene to DA3 or TRAM, add articulated-human physics,
introduce counterfactual controls, or start P5.

## P5 status

P5 adds one bounded inverse-physics profile:

```text
PhysicalMotionObservation
  -> bounded differential evolution and coordinate refinement
  -> repeated Newton/Warp objective rollouts
  -> fitted PhysicalScene
  -> validated SimulatedWorldState
  -> InversePhysicsFit
```

The profile fits tether rest length and two initial tangent-velocity
components. The anchor, attachment, gravity, mass, body shape, solver, and
timestep remain fixed. Candidate initial geometry is repaired so the body
attachment starts exactly at the candidate rest length.

The objective is weighted 3D body-origin position MSE in square meters. The
optimizer uses a fixed seed and finite bounds. The final candidate goes through
the full P4 rollout and validation path.

Synthetic evidence comes from a known-parameter Newton rollout. The P1/P2
adapter has a strict gate for `metric_measured` scale, matching source hashes,
measured rigid alignment, enough visible pelvis samples, and observable XYZ
motion. Existing DA3 and TRAM results do not pass this gate. Their real fit
status is `BLOCKED_INPUT`; P5 does not reinterpret relative or assumed units as
measured meters.

The new JSON boundaries are `phystwin.physical_motion_observation` and
`phystwin.inverse_physics_fit`. Three.js overlays the target samples and the
fitted `SimulatedWorldState`. It does not run the optimizer or simulator.

See [`physics-fitting-p5.md`](physics-fitting-p5.md) for exact parameters,
bounds, objective, commands, failure modes, and measured synthetic recovery.

P5 does not add articulated-human dynamics, active control, inferred forces,
topology search, VLM hypotheses, or counterfactual controls.

## P5R status

P5R is implemented and awaiting a measured real clip.

```text
recorded video
  → DA3 + SAM2
  → entities.v1 world XYZ
  → known-distance metric_measured scale
  → PhysicalMotionObservation
  → existing P5 Newton fitter
  → observed vs simulated Three.js view
```

See [`physics-fitting-p5r.md`](physics-fitting-p5r.md). Do not start
articulated control or hypothesis generation from this checkpoint.

P5R does not invent metric scale. Local footage inspection currently returns
`AWAITING_FOOTAGE`.

## P0 exclusions

P0 does not:

- install DA3, Newton, Warp, GVHMR, TRAM, OpenUSD, or another large dependency;
- run depth, camera, human, or scene reconstruction;
- add simulator-specific rigid-body or joint payloads;
- convert V1 pixel tracks into fake 3D observations;
- change V1 fitting, evaluation values, API behavior, or UI behavior;
- infer scale, gravity direction, mass, contacts, or forces;
- add distributed or cloud infrastructure.

## P1 run record

Measured on this machine with `vision/reconstruct.py samples/bounce.mp4 --max-frames 12 --duration-s 2`:

- clip: `samples/bounce.mp4`, 281 frames at 24 fps, first 2 seconds, 12 kept frames
- cache key: `bacb1959e742f3f3bad59c7f1b79392286233b53ca08be7ce66daf5e7e35aaec`
- wall seconds: `4.68`
- device: NVIDIA GeForce RTX 4080 SUPER
- peak GPU memory bytes: `2058493440` (1963 MiB)
- weights SHA-256: `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`
- point count: `250000` after confidence filter and downsample
- notes: timestamps came from the container. DA3 intrinsics varied across samples, so the contract stores sample 0 and sets `intrinsics_vary` true. Scale is relative. Forward pass was 0.64 s after weights were already local.
