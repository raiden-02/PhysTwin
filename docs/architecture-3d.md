# PhysTwin 3D architecture

This is the current architecture. The older image-space product is documented in [architecture.md](architecture.md).

## Overview

PhysTwin turns a video into an executable rigid-body scene, then resimulates that scene under edited physics.

```text
video
  -> reconstruction adapter
  -> SceneObservation
  -> metric motion (known length or known radius)
  -> PhysicalMotionObservation
  -> inverse fit (Newton on every objective step)
  -> PhysicalScene
  -> SimulatedWorldState
  -> Three.js inspection and counterfactual rollout
```

2D tracks, masks, and reprojection errors stay useful as evidence. They are not the state the simulator integrates.

The working real example is one IRIS falling ball. A pendulum attempt with DA3 depth and a short XPBD rod did not recover a usable trajectory. That failure is kept as engineering evidence.

## System flow

1. A reconstruction adapter reads source video identities and explicit options. It writes heavy artifacts under a work directory and returns a `SceneObservation`.
2. Object motion is lifted into meters. For the falling ball, SAM2 supplies the mask and the measured ball radius supplies metric scale. DA3 supplies camera K and a relative-scale pose check. DA3 depth is not used for the moving ball.
3. The lift becomes a `PhysicalMotionObservation`: timed body-origin samples in meters.
4. An inverse-physics profile edits a small `PhysicalScene` template. Every objective evaluation runs Newton 1.5.1 / Warp 1.16 on CUDA.
5. The fitted scene writes a `SimulatedWorldState` rollout. A counterfactual clone can change gravity and roll the same initial state again.
6. The browser reads project JSON. It does not integrate physics.

Vision and FastAPI live in `.venv`. Physics lives in `.venv-physics`. The two talk through files and a subprocess.

## Data contracts

Examples live in [`contracts/3d/v1`](../contracts/3d/v1/). Validation is in `vision/reconstruction/contracts.py`.

### SceneObservation

Reconstructed visual evidence. It does not select a physics model.

It stores:

- source videos and content hashes
- one shared timeline with original source frame numbers
- camera intrinsics and `T_world_camera` poses
- references to large geometry artifacts
- coordinate, scale, and transform conventions
- estimator provenance
- namespaced `extensions`

Large point clouds, meshes, masks, and estimator arrays stay in artifacts. Unknown extension keys cannot change core field meaning.

DA3 writes `extensions.phystwin.da3.v1` with `intrinsics_vary` and `sample_intrinsics`. The core `cameras[0].intrinsics` field is still sample 0, scaled to source pixels. Consumers that need K must call `sample_intrinsics_for_frame`. Falling-ball reconstruction does that per sample.

### PhysicalMotionObservation

Metric 3D body-origin samples for one inverse-physics objective. It is not a second scene observation.

It stores source identity, right-handed `+Y`-up coordinates, meters and seconds, strictly increasing timestamps, positions, weights, and provenance.

Synthetic targets come from a Newton rollout. Real targets come from a known-radius sphere track or a known-distance entity lift. Relative or assumed scale stays `BLOCKED_INPUT`.

### PhysicalScene

One physical explanation to test. It does not overwrite `SceneObservation`.

An executable scene currently supports:

- one rigid sphere
- gravity along `-Y`
- zero or one fixed-distance world tether
- Newton XPBD execution settings

Free-fall uses a sphere and no tether. Draft scenes can serialize without being marked executable.

### SimulatedWorldState

Project-owned simulator output. Three.js reads this JSON, not Newton objects.

It stores the source scene ID and canonical SHA-256, solver and device identity, the fixed-step timeline, body transforms and velocities, constraint geometry, runtime metadata, and validation checks.

## Coordinate conventions

Canonical time is seconds. `timestamp_s = 0` is the first kept sample. Consumers use stored timestamps. They do not rebuild time as `frame / fps`.

`SceneObservation` uses a first-camera graphics world:

- right-handed
- first valid camera pose defines the world
- `+X` first-camera right
- `+Y` first-camera image up
- `+Z` backward from the first camera

This does not claim that `+Y` is gravity-up.

The first camera itself is OpenCV: `+X` right, `+Y` down, `+Z` forward. Its initial world-from-camera rotation is `diag(1, -1, -1)`.

Image evidence is OpenCV: origin at the center of the upper-left pixel, `u` right, `v` down, intrinsics in pixels. Lens distortion is declared `unknown` unless removed.

`T_A_B` maps a point from frame `B` into frame `A`. JSON stores 16 values row-major. Math uses column vectors. Translation sits at indices 3, 7, and 11.

The observation world already matches the Three.js world basis. Three.js camera-local coordinates differ from OpenCV. Convert with `T_world_three_camera = T_world_camera * diag(1, -1, -1, 1)`. `THREE.Matrix4.fromArray()` reads column-major, so the frontend sets matrices explicitly.

An executable `PhysicalScene` uses right-handed `+Y` up, meters, kilograms, seconds, and radians.

`SceneObservation` scale is one of:

- `relative`: no metric conversion is known. `meters_per_world_unit` is `null`.
- `metric_measured`: a measured calibration provides meters per world unit.
- `metric_assumed`: an explicit assumption provides meters per world unit.

Physical alignment is:

```text
p_scene_m = T_scene_observation_m * [meters_per_observation_unit * p_observation, 1]
```

Scale is applied before the rigid transform. An assumed scale is a labelled hypothesis, not a recovered measurement.

## Reconstruction

The adapter protocol lives in `vision/reconstruction/adapter.py`. There is no plugin registry. One adapter is instantiated directly.

### DA3

- Package: `depth-anything-3` from `https://github.com/ByteDance-Seed/Depth-Anything-3`
- Code revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, Apache-2.0
- Weights: `depth-anything/DA3-BASE`, revision `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`, Apache-2.0
- Adapter: `vision/reconstruction/da3.py`

DA3-LARGE and DA3NESTED-GIANT-LARGE are CC BY-NC 4.0 and are not used.

DA3 extrinsics are OpenCV / COLMAP world-to-camera `(N, 3, 4)`. Intrinsics and depth are at processed resolution. The adapter inverts each `w2c`, gauges the first camera with `T_obs_from_native = F * inverse(T_native_camera0)` where `F = diag(1, -1, -1, 1)`, writes `T_world_camera_i` in that gauge, and scales K to source pixels. `contracts/3d/v1/examples/da3_w2c_fixture.json` locks the conversion. DA3's GLB exporter is not used.

SceneObservation from DA3 is relative-scale. Camera translation from DA3 is reported in reconstruction units, not meters.

### TRAM humans

Human evidence is optional and lives under `extensions.phystwin.humans.v1`.

- Estimator: TRAM (`yufu-wang/tram`), MIT, revision `4861c112f3c148201326680a50c9199650da6088`
- Adapter: `vision/reconstruction/tram.py`
- Joint layout: SMPL24
- Gauge: same first-camera graphics world as DA3

GVHMR and PromptHMR were not used. Their licenses are non-commercial research only.

SMPL weights are third-party and are not downloaded by PhysTwin. Live TRAM needs a separate Linux/conda stack. This repo converts TRAM-native files or the committed fixture.

Standalone TRAM cameras are labelled `metric_assumed` with `meters_per_world_unit = 1`. That is an estimator assumption, not a measured calibration.

## Metric reconstruction

Two measured-scale paths exist.

**Known-radius sphere** (falling ball): SAM2 mask, image center, horizontal silhouette radius, per-frame DA3 K, pinhole projected-sphere depth `Z = sqrt((f R / r)^2 + R^2)`. The first-camera graphics frame becomes XYZ. DA3 depth is unused for the ball. IRIS gravity is not read on this path.

**Known-distance pair** (tether attempt): DA3 depth plus SAM2 masks, robust 3D centers, then `known_scene_distance` between named physical points. Guessed diameters and estimator scales cannot produce `metric_measured`. If the measured distance is the tether itself, rest length is held fixed. That run is not independent length recovery.

Physical up must be declared. `level_camera` + `assumed` treats first-camera `+Y` as up. That is not a measured gravity vector.

## Physics execution

See [newton-runtime.md](newton-runtime.md).

Newton 1.5.1 XPBD on Warp 1.16 CUDA runs in `.venv-physics`. The adapter keeps project `+Y` up, so no axis swap is needed. Row-major `T_world_body` converts to XYZW quaternions at the input boundary and back for the rollout.

The tether fixture uses Newton's native distance joint. `SolverXPBD` is the only Newton 1.5.1 solver that enforces `DISTANCE`. Free-fall uses an implicit free joint and writes linear velocity into `joint_qd`.

## Inverse fitting

See [inverse-physics.md](inverse-physics.md) and [real-video-reconstruction.md](real-video-reconstruction.md).

Every candidate runs the full Newton rollout. The objective is weighted 3D body-origin MSE. Status values are `COMPLETE`, `BLOCKED_INPUT`, and `FAILED`.

On a real fit, `validation.passed` means Newton executed. It does not grade residual quality. `quality.status` stays `unassessed` unless a synthetic recovery check ran.

IRIS gravity is loaded only by evaluation, and only after `inverse_physics_fit.json` exists.

## Counterfactual simulation

A fitted `PhysicalScene` is cloned. One parameter changes. Newton rolls the same initial state. The result is labelled `observed: false`. The Moon path uses 1.62 m/s².

## Caching and provenance

Large stages run one at a time on one GPU: load, run, validate, cache, release, next stage.

The reconstruction cache key is SHA-256 over canonical JSON of schema, adapter, model revision, weights hash, source IDs, content hashes, and options. Paths and wall-clock times are not in the key.

Entries live under `results/cache/reconstruction/<cache-key>/`. A sibling temp directory is used, artifacts are hashed, and a `COMPLETE` marker is written last. A partial or hash-mismatched entry is a cache miss.

Fit reports store exact-byte SHA-256 for output files. Source scene and observation hashes use canonical JSON bytes.

## Evaluation philosophy

Synthetic recovery, real-dataset recovery, and image-space V1 numbers are separate claims.

A complete Newton run is not a physics-quality pass. RMSE and hashes are recorded either way. A poor residual is kept, not rewritten as success.

EMDB evaluation tooling exists and is optional. There is no approved EMDB sequence in this workspace. That does not block physics work.

## Known limitations

- Monocular reconstruction is ambiguous without a measured length or radius.
- DA3 camera translation is relative-scale. Do not read it as meters.
- Physics is one rigid sphere plus gravity, optionally one distance constraint.
- There is no contact, drag, damping fit, or articulated-human dynamics.
- Short XPBD rods near 0.50 m are not rigid in the validated tether setup.
- Counterfactuals are simulated hypotheses.
- Lens distortion is unknown. Distorted pixels are not treated as undistorted.
