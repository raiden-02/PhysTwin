# IRIS pendulum_45/01

Experiment: first real-video Newton tether fit.

It is not a physics-quality pass. It is not independent rope-length recovery.

Machine-readable copy: [`iris-pendulum-45-01.json`](iris-pendulum-45-01.json).

Large local files stay in `results/physics3d/p5r-real-fit/` and
`datasets/IRIS/`. Those directories are gitignored.

## Setup

Source: Hugging Face dataset `rasulkhanbayov/IRIS`, clip
`Pendulum/pendulum_45/01.mp4`.

Evidence kind is `external_dataset`.

Clip config:
`contracts/3d/v1/examples/p5r_iris_pendulum_45_01.json`.

Prepare window: start `2.0 s`, duration `4.0 s`, 16 kept frames. Frame 0 of the
source file still has a hand on the ball.

IRIS `parameters.json` `pendulum.pendulum_45.rope_length` is `0.50 m`. That
value set metric scale through `known_scene_distance` from `body_center` to
`anchor`. The fitter then held `rest_length_m` at `0.50`. PhysTwin did not
recover rope length from motion.

Physical up for this run is `level_camera` / `assumed`.

Profile: `tether_initial_tangent_velocity_fixed_length_v1`.

Fit id: `p5-359580f85437`. Date: 2026-08-29.

## Result

Copied from the local artifacts on this machine. Re-running the GPU fit can
change the hashes.

Lifts: accepted `32`, rejected `0`.

Observed body-origin travel (16 samples):

- X: `0.286 m`
- Y: `0.033 m`
- Z: `0.140 m`
- AABB diagonal / trajectory extent: `0.320 m`

Newton rollout body travel:

- X: `0.074 m`
- Y: `0.532 m`
- Z: `0.835 m`

Objective (`weighted_position_mse_3d`):

- initial MSE: `0.37973 m²`
- final MSE: `0.37969 m²`
- final RMSE: `0.616 m`
- normalized RMSE: `1.926` (RMSE / observed extent)
- improvement ratio: `1.00011`

Fitted parameters:

- `rest_length_m`: `0.50`, held fixed
- `initial_tangent_velocity_u_m_s`: `-3.914`, hit the lower bound
- `initial_tangent_velocity_v_m_s`: `-1.237`

XPBD tether residual on the saved rollout:

- max: `0.300 m`
- RMS: `0.173 m`

Optimizer: bounded differential evolution with coordinate refinement, `76`
evaluations, `226 s`, NVIDIA GeForce RTX 4080 SUPER.

`execution_valid` is true. `quality.status` is `unassessed`.
`validation.passed` is true because Newton executed, not because the residual
is good.

## Overlay

Blue is the observed 3D samples. Orange is the fitted Newton path.

They do not follow the same motion.

The observed samples sit in a tight, nearly horizontal cluster about `0.32 m`
across, with only `3 cm` of vertical travel. The Newton path swings about
`0.53 m` in Y and `0.83 m` in Z. Per-sample 3D error starts near `0.08 m` at
`t=0` and reaches about `1.02 m`. Mean error is `0.49 m`.

A residual larger than the whole observed trajectory is a poor fit. `COMPLETE`
and `execution_valid` only mean the solver ran.

## Diagnosis

A later audit of these same artifacts is in
[iris-pendulum-45-01-diagnosis.md](iris-pendulum-45-01-diagnosis.md).
The poor residual is not an optimizer-budget problem.
