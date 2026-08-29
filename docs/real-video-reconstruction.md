# Real-video reconstruction

This path lifts a recorded clip into a metric `PhysicalMotionObservation` and
fits it with the existing Newton inverse-physics profiles. It does not add a
second physics implementation.

Two lift methods exist.

**Known-radius sphere** (falling ball):

```text
recorded video
  → SAM2 mask
  → image center and horizontal silhouette radius
  → per-frame DA3 camera K
  → pinhole projected-sphere depth
  → PhysicalMotionObservation
  → free_fall_gravity_v1
```

See [evaluation/iris-falling-ball.md](evaluation/iris-falling-ball.md).

**Known-distance pair** (tether / pendulum):

```text
recorded video
  → DA3 scene/camera and per-frame depth
  → SAM2 masks on the same selected frames
  → interior-mask unproject + robust 3D center
  → entities.v1 object, anchor, or attachment tracks
  → known-distance metric calibration
  → PhysicalMotionObservation
  → tether fit with rest length held if it set the scale
```

See [evaluation/iris-pendulum-45-01.md](evaluation/iris-pendulum-45-01.md).

The pipeline does not invent meters. Local recorded clips without a tape
measurement stay ineligible.

## Entity tracks

`phystwin.entities.v1` stores generic object and anchor tracks on
`SceneObservation`. Real inverse physics does not require `humans.v1`.

The DA3-depth lift uses:

- depth from `artifacts/da3_depth.npz`
- the camera pose for that sample
- that sample's intrinsics when DA3 says K varies. Sample-0 K is not used as
  a silent fallback
- valid interior SAM-mask pixels, each unprojected with that pixel's depth
  and this frame's K
- a coordinate-wise median 3D center, then a cheap 3σ-MAD outlier drop
- DA3 confidence when the depth artifact includes `conf`

The object root is that robust 3D center. It is not median-depth at the mean
mask pixel.

The falling-ball path does not use DA3 depth. It uses the measured radius and
per-frame K.

## Metric calibration

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
pass or fail. `validation.passed` equals `execution_valid` on real fits.

## Commands

Inspect clips, no GPU:

```powershell
.\.venv\Scripts\python.exe vision\prepare_real_motion.py --inspect
```

After a measured tether clip exists:

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

Then run the fitter in the physics environment:

```powershell
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  results\physics3d\p5r-real-fit\aligned_physical_scene_template.json `
  --motion-observation results\physics3d\p5r-real-fit\target_motion_observation.json `
  --output results\physics3d\p5r-real-fit
```

`--profile` can be omitted. The fitter chooses the fixed-length profile when
the motion provenance says tether length set the scale.

Falling-ball prepare and fit:

```powershell
.\.venv\Scripts\python.exe vision\prepare_falling_ball.py
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  results\physics3d\p5r-falling-ball\aligned_physical_scene_template.json `
  --motion-observation results\physics3d\p5r-falling-ball\target_motion_observation.json `
  --profile free_fall_gravity_v1 `
  --output results\physics3d\p5r-falling-ball
```

IRIS media stays under `datasets/IRIS/` and is gitignored.

IRIS `pendulum.pendulum_45.rope_length` is `0.50 m`. That value sets metric
scale and holds `rest_length_m` fixed. PhysTwin does not claim an independent
rope-length recovery from that clip.

Observation-aligned rollouts keep the standalone fixture 1e-5 / 3-axis checks
off. They still record the real XPBD tether residual and AABB travel.
A planar swing is allowed. A standalone tether fixture still has to meet
the original 1e-5 and 3-axis invariants.
