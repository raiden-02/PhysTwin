# PhysTwin

**Video → inferred dynamics → interactive physics twin**

PhysTwin tracks one object in a short video, fits a selected dynamics model in C++20, and replays the reconstructed motion beside the recording in Three.js.

Implemented model families:

- **Projectile / Bounce:** initial velocity, image-space gravity, ground collision, and restitution.
- **Swing / Pendulum:** full nonlinear damped motion, fixed pivot, image-space radius, initial angular velocity, effective `g/L`, and damping.

The Three.js scene uses positions from C++ reconstruction output. It does not run a second physics model in the browser.

![Recorded Mixkit tennis: observed vs simulated overlay](docs/demo/mixkit_overlay.gif)

This is the recorded projectile case. Mixkit [Tennis Ball Bouncing in Slow Motion](https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/), 281 frames at 24 fps.

```text
RMSE            13.79 px   fair
RMSE_x           8.40 px   9.30% of 90.3 px horizontal travel
RMSE_y          10.94 px   1.57% of 695.6 px vertical travel
bounce timing    1.00 frame (41.67 ms)
g              173.15 px/s^2
e                0.665
track           19.96 s end-to-end (14.1 FPS including model load and decode)
fit              0.019 s
```

The horizontal signal is weak because its 90 px travel is smaller than the tracked mask radius. The vertical bounce is the useful signal and fits to 1.57% of its travel.

## Evidence

Evidence classes are kept separate:

- **Recorded:** real camera footage.
- **Rendered:** generated or CGI video passed through the full SAM 2 and C++ pipeline.
- **Synthetic:** direct known-parameter C++ validation without video.

Current measured results are saved in [docs/evaluation.json](docs/evaluation.json).

### Recorded projectile

- Mixkit tennis bounce: `13.79 px RMSE`, `fair`.
- Vertical RMSE: `10.94 px`, or `1.57%` of vertical travel.
- Bounce timing error: `1.00 frame`, or `41.67 ms`.
- Fitted `g`: `173.15 px/s²`.
- Fitted restitution: `0.665`.

### Pendulum synthetic validation

- Noise-free nonlinear recovery: `1.29e-10 px RMSE`.
- Recovered `omega0`, `lambda`, and damping agree with their known values to about `1e-12`.
- Deterministic noise and three outliers: `2.63 px RMSE`.
- Noisy `lambda`: `7.19915` for known `7.2`.
- Noisy damping: `0.217734` for known `0.22`.
- Five failure checks cover too few observations, stationary motion, near-zero radius, inconsistent pivot geometry, and invalid timestamps.
- A fast-motion regression case recovers `lambda=32.0` at `5.84e-12 px RMSE` and protects the period-derived search bound.

### Recorded physical pendulum

![Recorded physical pendulum: observed vs simulated overlay](docs/demo/pendulum_recorded_overlay.gif)

The fixed-camera physical pendulum is 663 frames at 30 fps after trimming the first 70 source frames to remove hand contact:

- 663 of 663 frames tracked.
- `43.27 s` end-to-end tracking on an RTX 4080 SUPER.
- `4.00 px RMSE`, or `0.49%` of the fitted `824.75 px` radius, classified `good`.
- `3.36 px` horizontal RMSE and `2.18 px` vertical RMSE.
- Fitted `lambda=28.711 s⁻²`, damping `0.0246 s⁻¹`, and initial angular velocity `-0.314 rad/s`.
- Target-to-pivot radial MAD: `1.17 px`.

## Local UI

One command after the venv and `phystwin.exe` exist:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\serve-ui.ps1
```

Open http://127.0.0.1:8765.

For projectile motion:

1. Choose **Projectile / Bounce**.
2. Pick a sample or upload a video.
3. Click the tracked object.
4. Run, then play or scrub the synchronized recording and twin.

For pendulum motion:

1. Choose **Swing / Pendulum**.
2. Upload a fixed-camera clip.
3. Click the bob.
4. Click the fixed pivot.
5. Run, then inspect the pivot, rod, reconstructed bob, trajectories, parameters, and fit warning.

`scripts\serve-ui.ps1 -Dev` runs Vite at http://127.0.0.1:5173 and proxies `/api` to the same FastAPI process.

The browser never talks to C++ or PyTorch directly. `vision/serve.py` binds 127.0.0.1 only.

## How it works

See [docs/architecture.md](docs/architecture.md) for coordinates, units, the physics model, and the JSON boundary.

The ownership split is explicit:

- **C++20:** simulation, system identification, robust objectives, geometry checks, reconstruction samples, and metrics.
- **Python, PyTorch, and SAM 2:** GPU video tracking and pixel-space observations.
- **React, TypeScript, and Three.js:** interaction, synchronized playback, and visualization.
- **FastAPI:** localhost-only orchestration between the browser, tracker, and executable.

## Architecture

```text
video + model choice + selections
    → vision/serve.py              FastAPI on 127.0.0.1:8765
    → vision/track.py              Python / PyTorch / SAM 2 / RTX 4080
    → model-aware tracking.json
    → phystwin fit                 C++20 model dispatch
    → model-aware reconstruction.json
    → frontend/                    React + TypeScript + Three.js
```

CLI overlay plots still work. They are not required for the product loop.

No gRPC, queues, or cloud services.

## System identification

Projectile fitting uses scalar least squares for `vx0`. A fixed-seed bounded differential search followed by deterministic coordinate refinement fits `vy0`, image-space gravity `g`, and restitution `e`.

Pendulum fitting uses:

```text
theta'' = -lambda * sin(theta) - damping * theta'
```

The fitter uses actual observation timestamps. It derives angle from target-to-pivot geometry, unwraps the angle series, estimates radius with medians, allows a bounded pivot refinement, and fits initial angular velocity, effective `lambda = g/L`, and non-negative damping. The observed zero-crossing period seeds and bounds the `lambda` search. A Huber objective limits the effect of a few bad observations.

Pendulum quality is RMSE divided by fitted image-space radius. `good` is at most 5%, `fair` is at most 15%, otherwise `poor`.

Both solvers are deterministic. `poor` still writes reconstruction JSON for diagnosis and exits with code 2. Ceres is not used.

## GPU video tracking

`vision/track.py` loads official SAM 2.1 tiny on CUDA, takes one click on frame 0, propagates the mask, and writes mask-derived centroids. If the video has no valid fps metadata, tracking fails instead of assuming 30 fps.

Device on these runs: **NVIDIA GeForce RTX 4080 SUPER**, torch `2.13.0+cu126`. Model: SAM 2.1 Hiera tiny. The recorded pendulum tracked 663 of 663 frames in `43.27 s` end-to-end.

`--point` is numeric `x,y`. Clicking a PNG in the editor only zooms the image.

Tracking FPS in the table is **end-to-end**: model load, JPEG decode, SAM 2 init, and propagation.

## Evaluation

Seven cases live in [docs/evaluation.json](docs/evaluation.json):

- recorded projectile;
- explicit projectile poor fit;
- two rendered projectile controls;
- projectile synthetic recovery;
- pendulum synthetic and noisy/outlier recovery;
- recorded physical pendulum.

Re-run with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-eval.ps1
```

The script requires `samples\bounce.mp4`. If `samples\recorded\pendulum.mp4` exists, it also evaluates that clip with the supplied pendulum point and pivot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-eval.ps1 `
  -PendulumPoint "111,858" -PendulumPivot "385,92"
```

Bounce timing is a high-y local-max heuristic paired within 8 frames. It is an evaluation check, not part of the optimizer.

## Build and run

Windows, Visual Studio 18 Community, Node.js (for the UI). CMake is bundled with Visual Studio. It is **not** on PATH in a regular PowerShell.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup-vision.ps1
.\.venv\Scripts\python.exe vision\check_cuda.py
```

`scripts\build.ps1` already runs CTest. To re-run tests later from a regular PowerShell (CMake is not on PATH):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

Local product UI (needs Node.js, the venv, and `build\Release\phystwin.exe`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\serve-ui.ps1
```

One Mixkit reconstruction (needs `samples\bounce.mp4` locally):

```powershell
.\.venv\Scripts\python.exe vision\track.py samples\bounce.mp4 --dump-frame results\frame0.png
.\.venv\Scripts\python.exe vision\track.py samples\bounce.mp4 --point 375,722 --output results\cases\mixkit_tennis\tracking.json
.\build\Release\phystwin.exe fit results\cases\mixkit_tennis\tracking.json --output results\cases\mixkit_tennis\reconstruction.json
.\.venv\Scripts\python.exe vision\plot_reconstruction.py results\cases\mixkit_tennis\tracking.json results\cases\mixkit_tennis\reconstruction.json
.\.venv\Scripts\python.exe vision\overlay_comparison.py samples\bounce.mp4 results\cases\mixkit_tennis\tracking.json results\cases\mixkit_tennis\reconstruction.json --gif docs\demo\mixkit_overlay.gif --still docs\demo\mixkit_overlay.png --panel-height 320 --gif-stride 4 --gif-max-width 540 --gif-colors 48 --title "Mixkit tennis bounce (recorded)"
```

Generated diagonal clip through the same loop (pipeline check, not real-footage accuracy):

```powershell
.\.venv\Scripts\python.exe vision\make_bounce_clip.py --output samples\generated_diagonal.mp4 --x0 80 --y0 40 --vx 140 --vy 20 --g 1800 --e 0.72
.\.venv\Scripts\python.exe vision\track.py samples\generated_diagonal.mp4 --point 80,40 --output results\cases\generated_diagonal\tracking.json
.\build\Release\phystwin.exe fit results\cases\generated_diagonal\tracking.json --output results\cases\generated_diagonal\reconstruction.json
.\.venv\Scripts\python.exe vision\overlay_comparison.py samples\generated_diagonal.mp4 results\cases\generated_diagonal\tracking.json results\cases\generated_diagonal\reconstruction.json --gif docs\demo\diagonal_overlay.gif --still docs\demo\diagonal_overlay.png --title "Generated diagonal bounce"
```

Generated drop uses `--x0 320 --y0 36 --vx 4 --vy 50 --g 1600 --e 0.40` and `--point 320,36`.

Pendulum CLI path:

```powershell
.\.venv\Scripts\python.exe vision\track.py samples\recorded\pendulum.mp4 `
  --model pendulum --point 111,858 --pivot 385,92 `
  --output results\cases\pendulum_recorded\tracking.json

.\build\Release\phystwin.exe fit `
  results\cases\pendulum_recorded\tracking.json `
  --output results\cases\pendulum_recorded\reconstruction.json
```

The current recorded clip exits 0 with quality `good` and `4.00 px RMSE`.

CMake without the helper, after putting VS CMake on the current session PATH:

```powershell
. .\scripts\vs-cmake.ps1
Use-VsCMake
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Limitations

- All motion and fitted scales are image-space. There is no metric calibration, physical length, or metric gravity recovery.
- Pendulum assumes a fixed camera, fixed pivot, one bob, planar motion, and one uninterrupted interval.
- Projectile assumes one point mass and one horizontal ground line. It does not model spin, drag, friction, or wall collisions.
- SAM 2's optional CUDA hole-filling extension is unavailable. PyTorch inference still runs on the GPU.
- The UI binds localhost only and runs one GPU job at a time.
