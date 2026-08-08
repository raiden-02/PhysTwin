# PhysTwin V1 architecture

PhysTwin reconstructs a simple 2D physics motion from a short fixed-camera video of one moving object.

This document is the Day-1 design. It matches `career-os/portfolio/phystwin.md`. Implementation follows the checkpoint order in that contract. Checkpoint 0 ships structure, types, and the JSON boundary. Simulation, fitting, GPU tracking, and demo work come later.

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
plot or UI: observed vs simulated + RMSE
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
```

`frontend/` is not created until the core loop and metrics exist.

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

TypeScript / React / Three.js is optional after the CLI loop is measured. A plotted overlay is enough for acceptance.

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
  "metrics": { "rmse": 0.0, "mae": 0.0, "n": 0, "iterations": 0, "fit_seconds": 0.0 },
  "simulated": [
    {"frame": 0, "t": 0.0, "x": 0.0, "y": 0.0}
  ]
}
```

`phystwin fit` does not write this file yet. Checkpoint 0 only loads and inspects `tracking.json`.

## Coordinates, units, scale

Monocular video does not give metric scale.

V1 reports:

- position in pixels
- velocity in pixels per second
- gravity as a **gravity scale** `g` in pixels per second squared, not `9.81 m/s²`
- restitution `e` dimensionless in `[0, 1]`

Do not claim SI units unless a later checkpoint adds a known scene length (ball diameter or similar).

Normalized `[0, 1]` coordinates are not the default. Pixel RMSE is the number a reviewer can check against a plot. If we normalize later, we store the scale used.

## Physics model (Checkpoint 1)

One rigid point mass in 2D image space. No rotation, drag, friction, or object-object collision in V1.

Parameters θ:

- `vx0` initial horizontal velocity (px/s)
- `vy0` initial vertical velocity (px/s, +y down)
- `g` gravity scale (px/s², +y down, typically positive)
- `e` coefficient of restitution

Not fitted:

- `x0`, `y0` taken from the first observation
- `y_ground` estimated from the trajectory (high observed `y`) or passed as `--ground`
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

## System identification (Checkpoint 1)

Observed points `P_obs(t_i)`. Simulator points `P_sim(t_i; θ)` sampled at the same times.

```text
θ* = arg min_θ  Σ_i ||P_obs(t_i) - P_sim(t_i; θ)||²
```

Unweighted residuals first. Confidence weights and outlier rejection only if real tracking noise requires them after the unweighted fit works.

Restitution makes the objective non-smooth near bounce times. Planned fallback order if Ceres / least squares is unstable:

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
| Physics + synthetic tests | this repo | 1 | no extra dep |
| Nonlinear least squares | Ceres Solver preferred | 1 | vcpkg or source. Fallback: small in-repo optimizer (Nelder-Mead or finite-difference Gauss-Newton) if Ceres blocks the day |
| Dense linear algebra | Eigen | 1 only if used | FetchContent. Skip if unused |
| Video / masks | OpenCV via `opencv-python` | 2 | Python venv |
| SAM 2 inference | PyTorch + SAM 2 + CUDA | 2 | Python 3.11 or 3.12 venv on the RTX 4080. System Python is currently 3.14 beta and is not trusted for PyTorch |
| Plots | matplotlib or a tiny C++ dump + Python plot | 4 | after RMSE exists |
| Interactive UI | React + TypeScript + Three.js | 4 optional | only if the CLI loop is already measured |

Do not add Ceres, Eigen, OpenCV, or PyTorch to the C++ CMake graph in Checkpoint 0. A missing optional package must not break `cmake --build`.

## CLI

```text
phystwin inspect tracking.json
phystwin fit tracking.json --output reconstruction.json
python vision/track.py input.mp4 --point 531,312 --output tracking.json
```

Checkpoint 0:

- `inspect` loads the contract and prints a summary
- `fit` is a stub (non-zero exit)
- `track.py` is a stub (non-zero exit)

## Tests

| Target | Role now | Role later |
|---|---|---|
| `io_roundtrip` | write/load `tracking.json`, assert fields | keep as contract test |
| `synthetic_fit` | placeholder that currently passes without recovery | Checkpoint 1: recover known `vx0, vy0, g, e` within a stated tolerance |

## What is intentionally not in Checkpoint 0

- numerical integration
- parameter recovery
- SAM 2 / GPU tracking
- reconstruction JSON writer used by a real fit
- plots, GIFs, frontend
- Ceres / Eigen / OpenCV C++ packages
