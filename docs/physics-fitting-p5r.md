# P5R real-video inverse physics

P5R connects a recorded clip to the existing P5 Newton fitter. It does not add
a second physics implementation.

```text
recorded video
  → DA3 scene/camera and per-frame depth
  → SAM2 masks on the same selected frames
  → interior-mask unproject + robust 3D center
  → entities.v1 object, anchor, or attachment tracks
  → known-distance metric calibration
  → PhysicalMotionObservation
  → P5 Newton/Warp fitter
  → fitted PhysicalScene + SimulatedWorldState
```

P5R is implemented. The first completed real Newton fit used IRIS
`Pendulum/pendulum_45/01.mp4`. Local recorded clips still have no
tape-measured length. The pipeline will not invent meters.

IRIS `rope_length` `0.50 m` set metric scale and held `rest_length_m` fixed.
That run is real-video physics fitting, not independent rope-length recovery.
See [evaluation/iris-p5r-pendulum-45-01.md](evaluation/iris-p5r-pendulum-45-01.md).

## What is new

`phystwin.entities.v1` stores generic object and anchor tracks on
`SceneObservation`. Real inverse physics does not require `humans.v1`.

Lift uses:

- DA3 depth from `artifacts/da3_depth.npz`
- the camera pose for that sample
- that sample's intrinsics when DA3 says K varies. Sample-0 K is not used as
  a silent fallback
- valid interior SAM-mask pixels, each unprojected with that pixel's depth
  and this frame's K
- a coordinate-wise median 3D center, then a cheap 3σ-MAD outlier drop
- DA3 confidence when the depth artifact includes `conf`

The object root is that robust 3D center. It is not median-depth at the mean
mask pixel.

`phystwin.metric_calibration` records one known scene distance. The only
accepted method is `known_scene_distance`. Sources such as `guessed`,
`estimator`, `diameter_guess`, or `assumed` cannot produce `metric_measured`.

The pair must name the physical points the tape connects:

- `from_physical_point` / `to_physical_point`: `anchor` and `body_center`,
  or `anchor` and `attachment`
- entity `kind` must match those points (`object` for `body_center`)
- an object-mask centroid is not accepted as a tether knot

If the measured distance is the tether itself, `circular_with_fit_parameter`
is `rest_length_m`. The fitter then uses
`tether_initial_tangent_velocity_fixed_length_v1` and holds `rest_length_m`
fixed. It does not claim an independent length recovery.

Physical up is required and must be declared:

- `level_camera` + `assumed`: the camera was leveled, so first-camera `+Y`
  is treated as physical up. This is not a measured gravity vector.
- `supplied_vector` + `assumed` or `measured`: rotate that observation
  vector onto physical `+Y`, then translate the first scaled anchor.

`scale_source` stays `measured` after known-distance calibration.
`alignment_source` equals the physical-up source.

For a real SceneObservation-derived motion, 
`PhysicalScene.observation_alignment.observation_sha256` must equal
`PhysicalMotionObservation.source.sha256`. Cross-wiring another observation
is rejected.

A complete Newton run is `execution_valid`. Real-fit `quality.status` is
`unassessed`. RMSE and normalized RMSE are stored as numbers. They are not a
pass/fail. `validation.passed` equals `execution_valid` on real fits. Do not
read that as "physics fit passed". Synthetic fits still use
`quality.status = synthetic_checked` and the existing 0.02 / 0.03 checks.

## Local footage

`vision/prepare_real_motion.py --inspect` classifies on-disk clips.

Local recorded clips still fail the metric gate. The Mixkit bounce, generated
clips, and cinematic swing are not tether validation. The local pendulum has
no tape-measured length in meters. Image-space radius is not a metric
measurement. Without IRIS those clips alone stay `AWAITING_FOOTAGE`.

IRIS `pendulum_45/01` is eligible when `datasets/IRIS/` is present. Then
inspect is `READY`. That is an `external_dataset` source, not `recorded_real`.

## Commands

Inspect clips, no GPU:

```powershell
.\.venv\Scripts\python.exe vision\prepare_real_motion.py --inspect
```

After a measured clip exists:

```powershell
.\.venv\Scripts\python.exe vision\prepare_real_motion.py `
  --video path\to\tether.mp4 `
  --target-xy U V `
  --anchor-xy U V `
  --known-distance-m 2.00 `
  --measurement-source "tape measure YYYY-MM-DD" `
  --circular-with rest_length_m `
  --from-physical-point body_center `
  --to-physical-point anchor `
  --up-mode level_camera `
  --up-source assumed `
  --output results\physics3d\p5r-real-fit
```

`--from-physical-point` / `--to-physical-point` name the tape ends. Use
`attachment` only when that entity `kind` is `attachment`.

`--up-mode level_camera` is the controlled-level-camera path. It stays
`assumed`. To record a measured up, pass `--physical-up X Y Z` and
`--up-source measured`.

Then run the existing P5 fitter in the physics environment:

```powershell
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  results\physics3d\p5r-real-fit\aligned_physical_scene_template.json `
  --motion-observation results\physics3d\p5r-real-fit\target_motion_observation.json `
  --output results\physics3d\p5r-real-fit
```

`--profile` can be omitted. The fitter chooses the fixed-length profile when
the motion provenance says tether length set the scale.

## UI

**3D physics → Inspect P5R real fit** shows the requested clip, local
inspection, and any saved complete overlay. It does not fabricate a rollout
when footage is missing.

## Requested clip

- 3 to 8 seconds
- ball or weight on a string or rope
- visible anchor and object
- nontrivial spatial motion. Planar pendulums are allowed
- known tether length in meters
- limited blur and occlusion

Do not guess an object diameter. Do not treat cinematic footage as the
correctness baseline.

## IRIS first real source

The first real P5R clip is IRIS `Pendulum/pendulum_45/01.mp4` from
`rasulkhanbayov/IRIS`. Evidence kind is `external_dataset`, not
`recorded_real`.

Download stays under `datasets/IRIS/` and is gitignored.

```powershell
.\.venv\Scripts\python.exe vision\prepare_real_motion.py --iris --output results\physics3d\p5r-real-fit
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  results\physics3d\p5r-real-fit\aligned_physical_scene_template.json `
  --motion-observation results\physics3d\p5r-real-fit\target_motion_observation.json `
  --output results\physics3d\p5r-real-fit
```

`parameters.json` `pendulum.pendulum_45.rope_length` is `0.50 m`. That value
sets metric scale and holds `rest_length_m` fixed. PhysTwin does not claim an
independent rope-length recovery.

The first run on this machine is recorded in
[evaluation/iris-p5r-pendulum-45-01.md](evaluation/iris-p5r-pendulum-45-01.md).
Status `COMPLETE` and `execution_valid` mean Newton ran. Final RMSE is
`0.616 m`. Quality stays `unassessed`. The overlay does not follow the
observed path. Do not treat that residual as a pass.

Seed pixels for that clip are in
`contracts/3d/v1/examples/p5r_iris_pendulum_45_01.json`. Prepare starts at
`t=2.0 s` so the hand is no longer on the ball.

Physical up for this run is `level_camera` / `assumed`.

Observation-aligned rollouts keep the P4 fixture 1e-5 / 3-axis checks
off. They still record the real XPBD tether residual and AABB travel.
A planar swing is allowed. A P4 standalone fixture still has to meet
the original 1e-5 and 3-axis invariants.

## P3 note

The reconstruction evaluator is complete. Synthetic validation is complete.
EMDB remains optional and unavailable here. It is not a P5R blocker. The EMDB
adapter stays in the repo.
