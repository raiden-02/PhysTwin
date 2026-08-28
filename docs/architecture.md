# PhysTwin architecture

PhysTwin reconstructs one tracked object's 2D image-space motion with a selected projectile/bounce or nonlinear pendulum model.

## Pipeline

```text
video + model choice + target/reference clicks
        |
        v
Python worker (SAM 2 / PyTorch / CUDA)
  track.py writes model-aware tracking.json
        |
        v
C++20 core
  phystwin fit dispatches to projectile or pendulum
  and writes model-aware reconstruction.json
        |
        +--> plot_reconstruction.py / overlay_comparison.py
        |
        +--> vision/serve.py + frontend/
             local React UI: stages, playback, Three.js reconstruction
```

No gRPC, queues, or service mesh. The languages talk through JSON files. FastAPI is only a localhost adapter.

## Repository layout

```text
phystwin/
  CMakeLists.txt          C++20 library, CLI, tests
  cpp/include/phystwin/   public types and interfaces
  cpp/src/                implementations
  cpp/tests/              CTest targets
  vision/                 Python SAM 2 worker + FastAPI UI adapter
  frontend/               React + TypeScript + Three.js local UI
  samples/                tiny JSON fixtures. Large videos stay local
  results/                measured outputs. Not committed
  docs/architecture.md    this file
  docs/evaluation.json    measured evaluation numbers
  docs/demo/              committed overlays, GIFs, and comparison plots
```

`scripts/serve-ui.ps1` builds `frontend/` and serves it from `vision/serve.py` on `http://127.0.0.1:8765`.

## Language split

C++20 owns:

- trajectory types and JSON I/O
- coordinate and time conventions
- physics model and fixed-step integration
- parameter fitting
- metrics and synthetic tests
- the `phystwin` CLI

Python owns the ML worker and the localhost adapter:

- load SAM 2 / PyTorch
- accept a target click on frame 0
- accept a fixed pivot or tracked-anchor click for pendulum mode
- propagate the target and optional anchor masks through the clip on the local RTX 4080
- emit `tracking.json`
- `vision/serve.py` runs those steps and `phystwin.exe` for the browser

TypeScript, React, and Three.js own the product loop: model choice, upload, selections, stage display, synchronized playback, reconstructed motion, and metrics. Three.js renders C++ reconstruction samples. It does not integrate either physics model.

## JSON contracts

### `tracking.json` (Python → C++)

```json
{
  "version": 1,
  "model": "pendulum",
  "fps": 60.0,
  "frame_width": 1920,
  "frame_height": 1080,
  "reference": {
    "mode": "tracked",
    "pivot_x": 960.0,
    "pivot_y": 120.0,
    "coverage": 0.95
  },
  "observations": [
    {"frame": 0, "t": 0.0, "x": 531.2, "y": 312.7}
  ],
  "anchor_observations": [
    {"frame": 0, "t": 0.0, "x": 960.0, "y": 120.0}
  ]
}
```

Required:

- `version` must be `1`
- `fps` > 0
- `observations` non-empty
- each observation has `frame`, `t`, `x`, `y`
- `model` is `projectile_bounce` or `pendulum`. Missing means `projectile_bounce` for backward compatibility
- pendulum input requires `reference.pivot_x` and `reference.pivot_y`
- `reference.mode` is `fixed` by default for backward compatibility
- tracked mode requires at least 12 `anchor_observations`, one per target row with matching `frame` and `t`
- tracked coverage must be at least 60%

Optional on each observation:

- `confidence`
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
- `radius`

Positions are image pixels. Origin is the top-left of the frame. `x` increases right. `y` increases down. That matches OpenCV and SAM 2, so overlay does not need a flip.

Time `t` is seconds from the start of the clip. The usual value is `frame / fps`. The fitter uses the `t` values in the file, not a reconstructed index, so dropped frames stay honest.

### `reconstruction.json` (C++ output)

```json
{
  "version": 1,
  "model": "projectile_bounce",
  "parameters": { "vx0": 0.0, "vy0": 0.0, "g": 0.0, "e": 0.0 },
  "environment": { "x0": 0.0, "y0": 0.0, "y_ground": 0.0, "dt": 0.016667 },
  "units": {
    "position": "pixels",
    "time": "seconds",
    "velocity": "pixels_per_second",
    "gravity": "pixels_per_second_squared",
    "restitution": "dimensionless"
  },
  "metrics": {
    "rmse": 0.0,
    "rmse_x": 0.0,
    "rmse_y": 0.0,
    "normalized_rmse": 0.0,
    "worst_axis_normalized_rmse": 0.0,
    "quality": "good",
    "ground_source": "max_observed_centroid_y",
    "ground_violation": 0.0,
    "n": 0,
    "search_generations": 160,
    "refinement_iterations": 0,
    "iterations": 160,
    "fit_seconds": 0.0
  },
  "simulated": [
    {"frame": 0, "t": 0.0, "x": 0.0, "y": 0.0}
  ]
}
```

Pendulum output uses the same `simulated` point array with model-specific fields:

```json
{
  "version": 1,
  "model": "pendulum",
  "parameters": {
    "omega0": -0.35,
    "lambda": 7.2,
    "damping": 0.22
  },
  "environment": {
    "pivot_x": 400.0,
    "pivot_y": 170.0,
    "radius": 220.0,
    "theta0": 0.9,
    "integration_step": 0.004167,
    "reference_mode": "tracked",
    "anchor_path": [
      {"frame": 0, "t": 0.0, "x": 400.0, "y": 170.0}
    ]
  },
  "metrics": {
    "rmse": 0.0,
    "normalized_rmse": 0.0,
    "robust_cost": 0.0,
    "radial_mad": 0.0,
    "angular_span": 1.8,
    "pivot_adjustment": 0.0,
    "anchor_track_coverage": 0.95,
    "quality": "good"
  },
  "simulated": [
    {"frame": 0, "t": 0.0, "x": 572.3, "y": 306.8}
  ]
}
```

`phystwin fit` writes one simulated Cartesian bob position for each input observation. Three.js consumes these positions directly.

## Coordinates, units, scale

Monocular video does not give metric scale.

PhysTwin reports:

- position in pixels
- velocity in pixels per second
- gravity as a **gravity scale** `g` in pixels per second squared, not `9.81 m/s²`
- restitution `e` dimensionless in `[0, 1]`
- pendulum angles in radians
- angular velocity in radians per second
- effective pendulum `lambda = g/L` in inverse seconds squared
- damping in inverse seconds

Do not interpret image-space radius as physical pendulum length or `lambda` as metric gravity.

Normalized `[0, 1]` coordinates are not the default. Pixel RMSE is the number a reviewer can check against a plot. If we normalize later, we store the scale used.

## Physics models

### Projectile / Bounce

One point mass in 2D image space. No rotation, drag, friction, or object-object collision.

Parameters θ:

- `vx0` initial horizontal velocity (px/s)
- `vy0` initial vertical velocity (px/s, +y down)
- `g` gravity scale (px/s², +y down, typically positive)
- `e` coefficient of restitution

Not fitted:

- `x0`, `y0` taken from the first observation
- `y_ground` is the object's center y at contact. It defaults to the maximum observed centroid y or is passed as `--ground-y`
- `dt` tied to video timing

Integrator: semi-implicit Euler, **fixed timestep** `dt = 1 / fps` from the tracking file.

```text
vy += g * dt
x  += vx * dt
y  += vy * dt
if y >= y_ground:
    y = y_ground
    vy = -e * vy
```

Collision is a hard clamp against one horizontal ground line. Contact timing is therefore grid-snapped to `dt`. That is a documented limitation, not a hidden bug.

Why this integrator: it is deterministic, matches frame-rate sampling, and is easy to explain. It is not energy-conserving. Bounce timing error of about one frame is expected.

### Swing / Pendulum

The image-space angle is measured from the downward vertical. Fixed mode uses one pivot. Tracked mode uses the paired anchor point at each timestamp:

```text
theta = atan2(target_x - pivot_x, target_y - pivot_y)
theta(t) = atan2(target_x(t) - anchor_x(t), target_y(t) - anchor_y(t))
```

The simulator integrates:

```text
theta'' = -lambda * sin(theta) - damping * theta'
```

It uses RK4 with a bounded internal step and samples the result at the actual observation timestamps. Simulated bob positions are:

```text
x = pivot_x + radius * sin(theta)
y = pivot_y + radius * cos(theta)

x(t) = anchor_x(t) + radius * sin(theta(t))
y(t) = anchor_y(t) + radius * cos(theta(t))
```

The radius is the median target-to-reference distance. Fixed mode may adjust the clicked pivot by a small deterministic bound if that improves radial consistency. Tracked mode uses the measured anchor path without geometry refinement.

## System identification

Observed points `P_obs(t_i)`. Simulator points `P_sim(t_i; θ)` are sampled at the same times.

```text
θ* = arg min_θ  Σ_i ||P_obs(t_i) - P_sim(t_i; θ)||²
```

The projectile fitter uses unweighted residuals. `vx0` is separable from the collision dynamics, so it is solved by scalar linear least squares. The remaining `(vy0, g, e)` values minimize the vertical squared residuals.

Restitution and the hard ground clamp make the objective non-smooth near bounce times. The current fitter uses a fixed-seed bounded differential search followed by deterministic coordinate refinement. It adds no numerical dependency and produces repeatable results. This is still an unweighted least-squares fit: the search minimizes the mean squared position residual.

Bounds come from the observed vertical speed and duration. `g` is constrained to be non-negative. `e` is constrained to `[0, 1]`. The initial vertical velocity and gravity estimates use pre-contact finite differences when frame-aligned samples are available.

Fit quality is based on per-axis RMSE divided by observed travel on that axis. Axes with less than 10 px or 5% of the largest-axis travel are ignored for this grade so stationary-axis tracking noise does not dominate. `good` is at most 5%, `fair` is at most 15%, and larger error is `poor`. Overall pixel RMSE is always reported separately.

An explicit ground is `poor` if observed centroids cross it by more than 5 px or 2% of trajectory extent. The CLI still writes the reconstruction for diagnosis, prints a warning, and exits with code 2.

`vision/plot_reconstruction.py` performs a simple contact-timing check by pairing high-y local maxima in observed and simulated trajectories. It is an evaluation heuristic, not part of the optimizer.

The differential-evolution generation count is a **fixed budget of 160**, not an adaptive stopping time. Coordinate refinement then halves a 0.05 step until it is below `1e-8`. On the measured cases that loop accepts no improvement after DE, so the printed `refinement_iterations` value is a safety-net cost, not evidence of extra convergence.

The pendulum fitter derives and unwraps the observed angle series. It rejects fewer than 12 observations, duration below 0.25 seconds, radius below 5 px, insufficient angular span, non-increasing timestamps, and inconsistent target-to-reference radius. Tracked mode also rejects missing or misaligned anchor rows and coverage below 60%.

It fits `(omega0, lambda, damping)` with a fixed-seed 220-generation bounded differential search followed by deterministic coordinate refinement. Same-direction zero crossings provide a measured period that seeds and bounds the `lambda` search. This prevents a fast real pendulum from being clipped by an arbitrary low upper bound. The objective applies a Huber loss to tangential position error, so a few noisy SAM centroids do not dominate. Final RMSE still uses the raw observed and simulated Cartesian positions.

Pendulum quality uses `RMSE / radius`. `good` is at most 5%, `fair` is at most 15%, and larger error is `poor`.

## Dependencies

The C++ graph contains the standard library and `nlohmann/json` 3.11.3 through CMake FetchContent. It does not depend on Ceres, Eigen, OpenCV, or PyTorch.

The Python 3.11 environment contains PyTorch, SAM 2.1 tiny, OpenCV, FastAPI, matplotlib, and Pillow. `SAM2_BUILD_CUDA=0` because `nvcc` is unavailable. Model inference still runs through CUDA-enabled PyTorch.

The frontend uses React, TypeScript, and Three.js.

## CLI

```text
phystwin inspect tracking.json
phystwin fit tracking.json --output reconstruction.json
python vision/track.py input.mp4 --point 531,312 --output tracking.json
python vision/track.py pendulum.mp4 --model pendulum \
  --point 111,858 --pivot 385,92 --output tracking.json
python vision/track.py cinematic.mp4 --model pendulum --anchor-mode tracked \
  --point 875,490 --pivot 1115,663 --output tracking.json
```

- `inspect` loads the contract and prints a summary
- `fit` dispatches from `tracking.json.model`, writes reconstruction JSON, and prints quality
- `fit --ground-y PIXELS` overrides the default maximum-centroid ground estimate
- poor fits write diagnostic output and exit with code 2
- `vision/track.py` runs SAM 2 on CUDA and writes target plus optional frame-paired anchor observations
- `vision/track.py` fails if the video has no valid fps metadata. It does not assume 30 fps
- tracking timing is **end-to-end** (model load, JPEG decode, init, propagation)
- empty masks are omitted from `observations` and recorded in `tracking_raw.json`
- `vision/plot_reconstruction.py` plots observed vs simulated trajectories
- `vision/overlay_comparison.py` writes a side-by-side MP4, still PNG, and optional GIF
- `vision/plot_evaluation.py` builds the saved comparison figure from `results/cases/manifest.json`
- `scripts/run-eval.ps1` refreshes projectile and pendulum synthetic, video, and failure evidence
- `vision/serve.py` is a localhost FastAPI adapter for model choice, target/pivot selections, SAM 2, `phystwin fit`, SSE stages, and JSON results
- `frontend/` is the React + TypeScript + Three.js UI

Create the venv once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-vision.ps1
```

SAM 2.1 tiny weights download to `checkpoints/sam2.1_hiera_tiny.pt` on first run. That file is gitignored.

## Tests

| Target | Coverage |
|---|---|
| `io_roundtrip` | write/load fixed and tracked-anchor `tracking.json`, assert alignment and identical-trajectory RMSE |
| `synthetic_fit` | generate 241 frames with two ground contacts, recover known `vx0, vy0, g, e`, enforce explicit tolerances, and reject a perturbed negative control |
| `dropped_frame` | drop a contiguous interior gap from a synthetic trajectory, assert the simulator samples the remaining timestamps, and recover the same parameters |
| `pendulum_fit` | recover fixed, fast, and moving-camera tracked-anchor motion, test deterministic noise/outliers, and reject eight degenerate inputs |
| `vision/test_trajectory.py` | CPU mask geometry, frame pairing, click-offset preservation, and missing frame-0 anchor rejection |
| `vision/test_serve.py` | UI server can find `phystwin.exe` and list Mixkit when present |

## Out of scope

- Ceres
- physical-scale calibration
- camera rotation, zoom, perspective, or general camera-motion compensation
- depth or 3D camera-space reconstruction
- automatic model discovery
- bungee, spring, or general rigid-body dynamics
- Eigen or OpenCV C++ packages
- SAM 2 CUDA post-process extension (`nvcc` not installed)
