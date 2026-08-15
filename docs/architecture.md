# PhysTwin V1 architecture

PhysTwin reconstructs a simple 2D physics motion from a short fixed-camera video of one moving object.

This document is the Day-1 design. It matches `career-os/portfolio/phystwin.md`. Checkpoint 6 applies the accepted V1 audit: honest recorded vs rendered vs synthetic labels, a saved poor-fit case, and a dropped-frame test. React/Three.js and Ceres are still out of this checkpoint.

## Pipeline

```text
real video + one click/box
        |
        v
Python worker (SAM 2 / PyTorch / CUDA)
  track.py writes tracking.json
        |
        v
C++20 core (this repo)
  phystwin fit tracking.json --output reconstruction.json
        |
        v
plot_reconstruction.py and overlay_comparison.py
  observed vs simulated + RMSE, optional GIF
```

No gRPC, queues, or service mesh in V1. The languages talk through JSON files.

## Repository layout

```text
phystwin/
  CMakeLists.txt          C++20 library, CLI, tests
  cpp/include/phystwin/   public types and interfaces
  cpp/src/                implementations
  cpp/tests/              CTest targets
  vision/                 Python SAM 2 worker (Checkpoint 2)
  samples/                tiny JSON fixtures. Large videos stay local
  results/                measured outputs. Not committed
  docs/architecture.md    this file
  docs/evaluation.json    measured evaluation numbers
  docs/demo/              committed overlays, GIF, three-case plot
```

`frontend/` was not created. The overlay GIF and trajectory plot are the V1 demo.

## Language split

C++20 owns:

- trajectory types and JSON I/O
- coordinate and time conventions
- physics model and fixed-step integration
- parameter fitting
- metrics and synthetic tests
- the `phystwin` CLI

Python owns only the ML worker:

- load SAM 2 / PyTorch
- accept a click or box on frame 0
- propagate the mask through the clip on the local RTX 4080
- emit `tracking.json`

Do not port SAM 2 to C++ in V1.

TypeScript / React / Three.js was skipped. Checkpoint 4 ships matplotlib plots plus an OpenCV/Pillow side-by-side overlay.

## JSON contracts

### `tracking.json` (Python → C++)

```json
{
  "version": 1,
  "fps": 60.0,
  "frame_width": 1920,
  "frame_height": 1080,
  "observations": [
    {"frame": 0, "t": 0.0, "x": 531.2, "y": 312.7}
  ]
}
```

Required:

- `version` must be `1`
- `fps` > 0
- `observations` non-empty
- each observation has `frame`, `t`, `x`, `y`

Optional on each observation, ignored by the V1 fitter if present:

- `confidence`
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
- `radius`

Positions are image pixels. Origin is the top-left of the frame. `x` increases right. `y` increases down. That matches OpenCV and SAM 2, so overlay does not need a flip.

Time `t` is seconds from the start of the clip. The usual value is `frame / fps`. The fitter uses the `t` values in the file, not a reconstructed index, so dropped frames stay honest.

### `reconstruction.json` (C++ output, Checkpoint 1+)

```json
{
  "version": 1,
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

`phystwin fit` writes this file. The simulated array has one point for each input observation.

## Coordinates, units, scale

Monocular video does not give metric scale.

V1 reports:

- position in pixels
- velocity in pixels per second
- gravity as a **gravity scale** `g` in pixels per second squared, not `9.81 m/s²`
- restitution `e` dimensionless in `[0, 1]`

Do not claim SI units unless a later checkpoint adds a known scene length (ball diameter or similar).

Normalized `[0, 1]` coordinates are not the default. Pixel RMSE is the number a reviewer can check against a plot. If we normalize later, we store the scale used.

## Physics model

One rigid point mass in 2D image space. No rotation, drag, friction, or object-object collision in V1.

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

## System identification

Observed points `P_obs(t_i)`. Simulator points `P_sim(t_i; θ)` sampled at the same times.

```text
θ* = arg min_θ  Σ_i ||P_obs(t_i) - P_sim(t_i; θ)||²
```

The Checkpoint 1 fitter minimizes unweighted residuals. `vx0` is separable from the collision dynamics, so it is solved exactly by scalar linear least squares. The remaining `(vy0, g, e)` values minimize the vertical squared residuals.

Restitution and the hard ground clamp make the objective non-smooth near bounce times. The current fitter uses a fixed-seed bounded differential search followed by deterministic coordinate refinement. It adds no numerical dependency and produces repeatable results. This is still an unweighted least-squares fit: the search minimizes the mean squared position residual.

Bounds come from the observed vertical speed and duration. `g` is constrained to be non-negative. `e` is constrained to `[0, 1]`. The initial vertical velocity and gravity estimates use pre-contact finite differences when frame-aligned samples are available.

Fit quality is based on per-axis RMSE divided by observed travel on that axis. Axes with less than 10 px or 5% of the largest-axis travel are ignored for this grade so stationary-axis tracking noise does not dominate. `good` is at most 5%, `fair` is at most 15%, and larger error is `poor`. Overall pixel RMSE is always reported separately.

An explicit ground is `poor` if observed centroids cross it by more than 5 px or 2% of trajectory extent. The CLI still writes the reconstruction for diagnosis, prints a warning, and exits with code 2.

`vision/plot_reconstruction.py` performs a simple contact-timing check by pairing high-y local maxima in observed and simulated trajectories. It is an evaluation heuristic, not part of the optimizer.

The differential-evolution generation count is a **fixed budget of 160**, not an adaptive stopping time. Coordinate refinement then halves a 0.05 step until it is below `1e-8`. On the measured cases that loop accepts no improvement after DE, so the printed `refinement_iterations` value is a safety-net cost, not evidence of extra convergence.

Fallback order if real tracking is unstable:

1. detect bounce times from the observed `y` series and fit flight segments
2. robust loss
3. derivative-free initialization, then local refinement
4. keep the collision model simple

The choice must be justified from observed behavior, not from solver fashion.

## Dependency plan

| Need | Library | Checkpoint | Install plan |
|---|---|---|---|
| C++20 compile | MSVC (VS 18 Community) | 0 | already present |
| `tracking.json` I/O | nlohmann/json 3.11.3 | 0 | CMake FetchContent |
| Physics + synthetic tests | this repo | 1 | implemented, no extra dep |
| Nonlinear least squares | deterministic bounded search | 1 | implemented in-repo because the collision residual is non-smooth |
| Dense linear algebra | Eigen | later only if needed | skipped in Checkpoint 1 |
| Video / masks | OpenCV via `opencv-python` | 2 | implemented in the 3.11 venv |
| SAM 2 inference | PyTorch 2.13.0+cu126 + SAM 2.1 tiny | 2 | implemented on RTX 4080 SUPER. `SAM2_BUILD_CUDA=0` because `nvcc` is missing |
| Plots | matplotlib `plot_reconstruction.py` / `plot_evaluation.py` | 4 | implemented |
| Overlay GIF | OpenCV + Pillow `overlay_comparison.py` | 4 | implemented, no frontend |
| Interactive UI | React + TypeScript + Three.js | skipped | overlay plot/GIF is the demo |

Ceres, Eigen, OpenCV, and PyTorch are not in the C++ CMake graph. A missing optional package must not break `cmake --build`.

## CLI

```text
phystwin inspect tracking.json
phystwin fit tracking.json --output reconstruction.json
python vision/track.py input.mp4 --point 531,312 --output tracking.json
```

Checkpoint 6:

- `inspect` loads the contract and prints a summary
- `fit` loads observations, fits parameters, writes reconstruction JSON, and prints quality
- `fit --ground-y PIXELS` overrides the default maximum-centroid ground estimate
- poor fits write diagnostic output and exit with code 2
- `vision/track.py` runs SAM 2 on CUDA and writes `tracking.json`
- `vision/track.py` fails if the video has no valid fps metadata. It does not assume 30 fps
- tracking timing is **end-to-end** (model load, JPEG decode, init, propagation)
- empty masks are omitted from `observations` and recorded in `tracking_raw.json`
- `vision/plot_reconstruction.py` plots observed vs simulated trajectories
- `vision/overlay_comparison.py` writes a side-by-side MP4, still PNG, and optional GIF
- `vision/plot_evaluation.py` builds the three-case README figure from `results/cases/manifest.json`
- `scripts/run-eval.ps1` requires `samples/bounce.mp4`, tracks Mixkit into a dedicated path, writes a poor-fit case, and refreshes `docs/evaluation.json`

Create the venv once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-vision.ps1
```

SAM 2.1 tiny weights download to `checkpoints/sam2.1_hiera_tiny.pt` on first run. That file is gitignored.

## Tests

| Target | Coverage |
|---|---|
| `io_roundtrip` | write/load `tracking.json`, assert fields and identical-trajectory RMSE |
| `synthetic_fit` | generate 241 frames with two ground contacts, recover known `vx0, vy0, g, e`, enforce explicit tolerances, and reject a perturbed negative control |
| `dropped_frame` | drop a contiguous interior gap from a synthetic trajectory, assert the simulator samples the remaining timestamps, and recover the same parameters |
| `vision/test_trajectory.py` | CPU mask centroid/bbox extraction |

## What is intentionally not in Checkpoint 6

- React / TypeScript / Three.js
- Ceres
- physical-scale calibration
- additional recorded camera clips
- robust loss, confidence weights, smoothing, or outlier rejection
- Eigen or OpenCV C++ packages
- SAM 2 CUDA post-process extension (`nvcc` not installed)
- a phone-camera clip in git. Recorded evidence is the Mixkit stock clip. The other two video cases are rendered, then tracked with SAM 2
