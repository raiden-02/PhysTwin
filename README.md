# PhysTwin

PhysTwin turns a short real-world video into a small, quantitatively fitted physics reconstruction: select one moving object, track it on the GPU, infer the physical parameters that best reproduce its motion, and compare the recorded and simulated trajectories side by side.

**Status:** Checkpoint 3 measured loop. SAM 2 tracks a clicked object on the RTX 4080 SUPER. The C++20 CLI consumes `tracking.json`, fits image-space physics, writes `reconstruction.json`, and reports RMSE. The demo UI is not implemented yet.

[DEMO GIF / VIDEO: not recorded yet]

## What it does

Planned V1 loop:

```text
real video
  → select object
  → GPU video segmentation/tracking (SAM 2)
  → observed trajectory (tracking.json)
  → C++ physics + parameter fitting
  → reconstructed trajectory (reconstruction.json)
  → real-vs-sim visualization + RMSE
```

Shipped: C++ synthetic core, Python SAM 2 tracking, the JSON boundary, C++ fitting, fit-quality checks, and an observed-vs-simulated trajectory plot.

## Results

The deterministic 241-frame synthetic test uses two ground contacts:

```text
parameter     actual       recovered      absolute error
vx0           180          180            2.84e-14 px/s
vy0          -420         -420            1.54e-07 px/s
g             980          980            2.51e-06 px/s^2
e             0.72         0.72           7.92e-10

trajectory RMSE: 9.01e-07 px
perturbed negative-control RMSE: 65.75 px
```

These numbers are from the noise-free synthetic test on 2026-08-25. They validate the simulator and fitter together. They are not real-video results and are not résumé evidence for GPU tracking.

The first recorded-video pass uses Mixkit's free [Tennis Ball Bouncing in Slow Motion](https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/) clip:

```text
frames:            281 at 24 fps, 720x1280
tracking:          281/281 masks, 19.96 s, 14.1 FPS
ground center y:   908.99 px (maximum observed centroid y)
vx0:               -9.51 px/s
vy0:                706.54 px/s
gravity scale:      173.15 px/s^2
restitution:          0.665
trajectory RMSE:     13.79 px
x RMSE:               8.40 px
y RMSE:              10.94 px
normalized RMSE:      1.98% of largest-axis travel
worst-axis error:     9.30%
contact timing error: 1.00 frame mean (41.67 ms)
quality:             fair
```

The vertical bounce timing and shape are close. Horizontal velocity is not constant across the close-up clip, so the fit is `fair`, not `good`. These are image-space values from slow-motion footage. They are not SI measurements.

## How it works

See [docs/architecture.md](docs/architecture.md) for coordinates, units, the physics model, the JSON boundary, and the dependency plan.

Short version: Python writes pixel-space observations. C++ fits `vx0`, `vy0`, gravity scale `g`, and restitution `e` in image coordinates. Gravity is not claimed as `9.81 m/s²` without a metric calibration.

## Architecture

```text
video + click
    → vision/track.py   (Python / PyTorch / SAM 2 / RTX 4080)
    → tracking.json
    → phystwin fit      (C++20)
    → reconstruction.json
    → plot or UI
```

## System identification

Implemented for synthetic and tracked trajectories. `vx0` uses scalar linear least squares. A fixed-seed bounded search followed by coordinate refinement minimizes squared vertical residuals for `vy0`, `g`, and `e`. This derivative-free choice handles the hard collision clamp without adding Ceres.

## GPU video tracking

Implemented. `vision/track.py` loads official SAM 2.1 tiny on CUDA, takes one click on frame 0, propagates the mask, and writes mask-derived centroids.

Measured on 2026-08-25 against the 281-frame Mixkit tennis clip:

```text
device: NVIDIA GeForce RTX 4080 SUPER
torch:  2.13.0+cu126
model:  SAM 2.1 Hiera tiny
tracked frames: 281/281 (0 empty masks)
wall time: 19.96 s  (~14.1 FPS including JPEG load + propagation)
```

The stock clip is recorded footage but uses slow motion and a close-up composition. Drop a fixed-camera phone clip in `samples/` and rerun with `--dump-frame` then `--point` for a stronger physics case.

C++ inspect of the emitted file:

```powershell
.\build\Release\phystwin.exe inspect results\tracking.json
```

## Evaluation

Synthetic recovery and one stock-footage reconstruction are recorded. Two more real clips remain before V1 acceptance.

## Build and run

Windows, Visual Studio 18 Community. CMake is bundled with Visual Studio. It is **not** on PATH in a regular PowerShell.

One command from a normal PowerShell in the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Or add VS CMake to the current session, then use the usual commands:

```powershell
. .\scripts\vs-cmake.ps1
Use-VsCMake

cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Inspect the sample contract file (this does not need `cmake` on PATH if `build\Release\phystwin.exe` already exists):

```powershell
.\build\Release\phystwin.exe inspect samples\example_tracking.json
```

Python worker (needs the 3.11 venv, not system 3.14):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-vision.ps1
.\.venv\Scripts\python.exe vision\check_cuda.py
.\.venv\Scripts\python.exe vision\test_trajectory.py
.\.venv\Scripts\python.exe vision\track.py samples\bounce.mp4 --dump-frame results\frame0.png
.\.venv\Scripts\python.exe vision\track.py samples\bounce.mp4 --point 375,722 --output results\tracking.json
.\build\Release\phystwin.exe fit results\tracking.json --output results\reconstruction.json
.\.venv\Scripts\python.exe vision\plot_reconstruction.py results\tracking.json results\reconstruction.json
```

Open `results\tracking_preview.png` to inspect segmentation tracking. Open `results\reconstruction_preview.png` to inspect observed vs simulated motion.

## Limitations

- One slow-motion stock clip has been reconstructed. Two more validation clips are still required.
- The close-up clip violates constant horizontal velocity enough to receive a `fair` grade.
- SAM 2's optional CUDA hole-filling kernel is not built (`nvcc` is missing). Tracking still runs on GPU through PyTorch.
- Monocular video has no metric scale. Later fits will report a gravity scale in px/s².
- One object, fixed camera, one ground line, no spin, no drag in V1.
- System Python on this machine is 3.14 beta. SAM 2 / PyTorch uses the checked-in setup script to create a 3.11 venv.

## Roadmap

V1 first: complete measured loop, then plots, then optional Three.js UI.

After V1: scale calibration, 3D, extra collision planes, rotation, friction. Not Day 1.
