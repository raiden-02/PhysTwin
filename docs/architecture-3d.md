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
- GVHMR versus TRAM for world-space humans.
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

## P1 scope

P1 implements only:

```text
short real video
  -> one DA3 adapter
  -> cached SceneObservation
  -> inspectable geometry artifact
  -> recovered camera path
  -> synchronized Three.js inspection
```

P1 records the exact package, license, revision, model/weights identity,
runtime, and peak GPU memory when practical. It validates estimator coordinate
conversion with a saved fixture or benchmark.

P1 does not add physical inference, Newton/Warp, human reconstruction, metric
scale claims, inverse physics, or cinematic tuning.

## P0 exclusions

P0 does not:

- install DA3, Newton, Warp, GVHMR, TRAM, OpenUSD, or another large dependency;
- run depth, camera, human, or scene reconstruction;
- add simulator-specific rigid-body or joint payloads;
- convert V1 pixel tracks into fake 3D observations;
- change V1 fitting, evaluation values, API behavior, or UI behavior;
- infer scale, gravity direction, mass, contacts, or forces;
- add distributed or cloud infrastructure.

## Decisions P1 must resolve

Before P1 is complete, record:

- the exact DA3 package, license, immutable revision, and weight identity;
- the estimator's native camera/world convention and tested canonical
  conversion;
- whether intrinsics are fixed or vary by sample;
- whether source pixels are distorted, undistorted, or unknown;
- the geometry artifact chosen for the first UI path;
- how variable-frame-rate timestamps are obtained;
- how missing camera poses and low-confidence regions are represented;
- measured runtime and peak GPU memory on the target machine;
- whether any real scale cue exists. The default is relative scale.
