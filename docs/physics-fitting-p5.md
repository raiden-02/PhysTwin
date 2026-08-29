# P5 inverse-physics fitting

P5 fits one small 3D tether model. It does not infer a general physical scene.
It keeps the P4 `PhysicalScene` input and `SimulatedWorldState` output.
Every objective evaluation runs Newton XPBD through Warp on CUDA.

## Fit profile

The only supported profile is
`tether_length_initial_tangent_velocity_v1`.

It fits three values:

- `rest_length_m`: bounds `[1.6, 2.4]`, initial search value `1.78`.
- `initial_tangent_velocity_u_m_s`: bounds `[-0.6, 0.6]`, initial search value `-0.12`.
- `initial_tangent_velocity_v_m_s`: bounds `[-0.6, 0.6]`, initial search value `0.14`.

The tangent basis comes from the template scene. `r` points from the world
anchor to the initial body attachment. `u` is the normalized cross product of
`r` and physical `+Y`. A fixed `+X` fallback handles a vertical tether. `v` is
the normalized cross product of `r` and `u`.

Each candidate moves the initial body transform along `r` so the body
attachment starts exactly at the candidate rest length. It then sets:

```text
initial_velocity = velocity_u * u + velocity_v * v
```

This removes radial initial velocity from the fit. Mass stays fixed. In this
gravity-only ideal tether, mass does not change the trajectory. P5 does not fit
damping because the validated P4 model has no damping parameter.

## Input evidence

`phystwin.physical_motion_observation`, version 1, contains the metric body
origin samples used by the objective. It stores:

- source identity and SHA-256
- right-handed, `+Y`-up coordinates
- meters and seconds
- strictly increasing timestamps
- 3D body-origin positions
- a weight in `(0, 1]` for each sample
- provenance and warnings

Synthetic evidence comes from a validated P4 rollout. The generator decimates
the saved rollout. It does not integrate a second model.

The P1/P2 adapter accepts a `SceneObservation` only when all of these checks
pass:

- scale status is `metric_measured`
- `meters_per_world_unit` is finite
- the `PhysicalScene` stores the matching canonical observation SHA-256
- scale and rigid alignment sources are both `measured`
- `T_scene_observation_m` is present
- the `humans.v1` pelvis track is valid
- at least 12 visible samples cover at least 0.5 seconds
- the physical body attachment is the body origin
- transformed motion varies by at least `0.02 m` on X, Y, and Z

Scale is applied before `T_scene_observation_m`, as defined by the 3D
architecture.

Current DA3 output has relative scale. Current TRAM output either inherits that
relative scale or uses the labelled `metric_assumed` fixture scale. Neither is
eligible. The CLI writes an `InversePhysicsFit` with status `BLOCKED_INPUT` and
the exact failed checks. It does not run an optimizer or report fitted values.

## Objective and search

For observed positions `p_i`, Newton positions `q_i`, and weights `w_i`, P5
minimizes:

```text
weighted_position_mse_3d =
  sum_i(w_i * ||q_i - p_i||^2) / sum_i(w_i)
```

Newton runs at the `PhysicalScene.fixed_step_s`. Requested observation times
are sampled from the adjacent integration steps with linear interpolation.

The report includes:

- MSE in square meters
- RMSE in meters
- observed 3D bounding-box diagonal
- RMSE divided by that diagonal
- MSE at the declared initial search point
- improvement ratio from the initial point

The optimizer is fixed-seed bounded differential evolution followed by
coordinate refinement. The default seed is `1347705141`. The default budget is
8 population members, 4 generations, and 12 coordinate iterations. The
synthetic run uses 112 objective evaluations.

The objective path records only body positions at requested times. The final
candidate still runs through the full P4 rollout writer and validator.

## Synthetic recovery

The template is
`contracts/3d/v1/examples/physical_scene_tether_fit_template.json`.
It runs for 1 second at 60 Hz with 24 XPBD iterations.

The known values are:

- rest length: `2.08 m`
- tangent velocity u: `0.31 m/s`
- tangent velocity v: `-0.23 m/s`

The generated target has 31 samples at 30 Hz and varies in X, Y, and Z.
Recovery passes when the largest parameter error, divided by that parameter's
bound width, is at most `0.03`. The trajectory normalized RMSE must be at most
`0.02`.

Run it from the repository root:

```powershell
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  contracts\3d\v1\examples\physical_scene_tether_fit_template.json `
  --fixture `
  --output results\physics3d\p5-tether-fit
```

The output directory contains:

- `truth_physical_scene.json`
- `target_motion_observation.json`
- `fitted_physical_scene.json`
- `simulated_world_state.json`
- `inverse_physics_fit.json`

The fit report stores exact-byte SHA-256 values for its two output files.
Source scene and observation hashes use canonical JSON bytes.
The fit writer and FastAPI verify those hashes, source IDs, sample count, body
ID, and rollout source before returning a complete artifact set.

Measured on this machine on 2026-08-28:

- GPU: NVIDIA GeForce RTX 4080 SUPER
- objective evaluations: `112`
- fit wall time: `152.09 s`
- peak Warp mempool allocation: `70,706 bytes`
- initial MSE: `0.1655932621 m²`
- fitted MSE: `0.0000025103 m²`
- fitted RMSE: `0.0015844 m`
- normalized RMSE: `0.0007936562`
- improvement ratio: `65,966.09`
- fitted rest length: `2.081498 m` for truth `2.08 m`
- fitted tangent velocity u: `0.309394 m/s` for truth `0.31 m/s`
- fitted tangent velocity v: `-0.227516 m/s` for truth `-0.23 m/s`
- largest normalized parameter error: `0.00207011`
- final tether maximum and RMS error: `0.963 µm` and `0.537 µm`
- final XYZ ranges: `1.781 m`, `0.506 m`, and `0.750 m`
- repeated final rollout maximum transform delta: `0`

These are local engineering measurements, not a benchmark claim.

## Real-evidence gate

Use a user-supplied observation with:

```powershell
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  path\to\physical_scene.json `
  --scene-observation path\to\scene_observation.json `
  --person-id human0 `
  --output results\physics3d\p5-real-fit
```

Exit code `2` means `BLOCKED_INPUT`. The JSON report lists all known blockers.
This is different from an optimizer failure.

## Result contract and failure modes

`phystwin.inverse_physics_fit`, version 1, has three statuses:

- `COMPLETE`: output scene and rollout exist, hashes are present, and validation
  passed
- `BLOCKED_INPUT`: evidence gates failed, fitted values and objective metrics
  are null, and no output scene or rollout is claimed
- `FAILED`: an attempted fit failed and the report contains failures

Contract validation rejects non-finite values, invalid bounds, fitted values
outside bounds, missing artifact hashes, and status/result contradictions.
Runtime fitting also rejects mismatched body IDs and observation times outside
the scene timeline.

The current CLI lets contract and runtime exceptions fail with a non-zero
process status. `FAILED` is reserved in the contract for an orchestrator that
persists such a failure report.

## Runtime and UI

P5 uses the same isolated environment and pins as P4:

- Python `3.11`
- Newton `1.5.1`, Apache-2.0
- Warp `1.16.0`, Apache-2.0 with NVIDIA product-specific terms
- CUDA device `cuda:0`

FastAPI runs the fit in `.venv-physics` and reads only project JSON afterward.
The endpoint is `POST /api/physics-fit-fixture`.

In the **3D physics** view, select **Inspect P5 synthetic fit**. Blue points and
the blue line are `PhysicalMotionObservation`. The orange line and moving
sphere are the fitted `SimulatedWorldState`. Three.js does not integrate
physics.

## Limits

P5 validates deterministic synthetic recovery for one passive rigid sphere and
one fixed-distance world tether. It does not fit anchor position, gravity,
mass, body orientation, body attachment, damping, forces, contacts, articulated
motion, active human control, or topology.

No eligible real metric tether observation is in this repository. P5 therefore
makes no real-video parameter-recovery claim.
