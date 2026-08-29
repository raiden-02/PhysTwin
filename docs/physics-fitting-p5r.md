# P5R real-video inverse physics

P5R connects a recorded clip to the existing P5 Newton fitter. It does not add
a second physics implementation.

```text
recorded video
  → DA3 scene/camera and per-frame depth
  → SAM2 masks on the same selected frames
  → robust mask depth + per-frame K and pose
  → entities.v1 object and anchor tracks
  → known-distance metric calibration
  → PhysicalMotionObservation
  → P5 Newton/Warp fitter
  → fitted PhysicalScene + SimulatedWorldState
```

P5R is implemented. A completed real Newton fit still needs a short tethered
clip with a tape-measured length. No local clip currently qualifies. The
pipeline will not invent meters.

## What is new

`phystwin.entities.v1` stores generic object and anchor tracks on
`SceneObservation`. Real inverse physics does not require `humans.v1`.

Lift uses:

- DA3 depth from `artifacts/da3_depth.npz`
- the camera pose for that sample
- that sample's intrinsics when DA3 says K varies. Sample-0 K is not used as
  a silent fallback
- the median valid depth inside the mask, not one centroid pixel

`phystwin.metric_calibration` records one known scene distance. The only
accepted method is `known_scene_distance`. Sources such as `guessed`,
`estimator`, `diameter_guess`, or `assumed` cannot produce `metric_measured`.

If the measured distance is the tether itself, `circular_with_fit_parameter`
is `rest_length_m`. The fitter then uses
`tether_initial_tangent_velocity_fixed_length_v1` and holds `rest_length_m`
fixed. It does not claim an independent length recovery.

Rigid alignment is assumed: first-camera `+Y` as physical up, then translate
the first scaled anchor onto the template world anchor. That is not a
measured gravity direction.

## Local footage

`vision/prepare_real_motion.py --inspect` classifies on-disk clips. Current
result: `AWAITING_FOOTAGE`.

The recorded pendulum exists, but there is no tape-measured length in meters.
Image-space radius is not a metric measurement.

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
  --output results\physics3d\p5r-real-fit
```

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
- noticeable out-of-plane motion
- known tether length in meters
- limited blur and occlusion

Do not guess an object diameter. Do not treat cinematic footage as the
correctness baseline.

## P3 note

The reconstruction evaluator is complete. Synthetic validation is complete.
EMDB remains optional and unavailable here. It is not a P5R blocker. The EMDB
adapter stays in the repo.
