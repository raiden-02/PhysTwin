# PhysTwin

PhysTwin turns a short real-world video into a small, quantitatively fitted physics reconstruction: select one moving object, track it on the GPU, infer the physical parameters that best reproduce its motion, and compare the recorded and simulated trajectories side by side.

**Status:** Checkpoint 2 GPU tracking. The C++20 synthetic core still passes. SAM 2 now tracks one clicked object on the RTX 4080 SUPER and writes `tracking.json`. Real-video C++ fitting and the demo UI are not implemented yet.

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

Shipped: C++ synthetic core plus a Python SAM 2 worker that emits the `tracking.json` contract from a point prompt.

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

Implemented for deterministic synthetic trajectories. `vx0` uses scalar linear least squares. A fixed-seed bounded search followed by coordinate refinement minimizes squared vertical residuals for `vy0`, `g`, and `e`. This derivative-free choice handles the hard collision clamp without adding Ceres.

## GPU video tracking

Implemented. `vision/track.py` loads official SAM 2.1 tiny on CUDA, takes one click on frame 0, propagates the mask, and writes mask-derived centroids.

Measured on 2026-08-25 against a 180-frame generated bounce clip (`samples/generated_bounce.mp4`, 640x360, 60 fps):

```text
device: NVIDIA GeForce RTX 4080 SUPER
torch:  2.13.0+cu126
model:  SAM 2.1 Hiera tiny
tracked frames: 180/180 (0 empty masks)
wall time: 11.35 s  (~15.9 FPS including JPEG load + propagation)
first centroid: 81.98, 41.20 px
last centroid:  500.06, 301.89 px
```

This clip is generated local footage used to prove the GPU loop. It is not a phone recording. Drop a real fixed-camera clip in `samples/` and rerun with `--dump-frame` then `--point`. Do not treat these numbers as real-world tracking accuracy.

C++ inspect of the emitted file:

```powershell
.\build\Release\phystwin.exe inspect results\tracking.json
```

## Evaluation

Synthetic recovery is implemented and enforced by CTest. Three real clips remain for later checkpoints.

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
.\.venv\Scripts\python.exe vision\make_bounce_clip.py
.\.venv\Scripts\python.exe vision\track.py samples\generated_bounce.mp4 --dump-frame results\frame0.png
.\.venv\Scripts\python.exe vision\track.py samples\generated_bounce.mp4 --point 80,40 --output results\tracking.json
.\build\Release\phystwin.exe inspect results\tracking.json
```

Open `results\tracking_preview.png` and confirm the overlay follows the ball.

## Limitations

- GPU tracking is proven on a generated bounce clip. A phone/real-camera clip has not been run yet.
- SAM 2's optional CUDA hole-filling kernel is not built (`nvcc` is missing). Tracking still runs on GPU through PyTorch.
- Monocular video has no metric scale. Later fits will report a gravity scale in px/s².
- One object, fixed camera, one ground line, no spin, no drag in V1.
- System Python on this machine is 3.14 beta. SAM 2 / PyTorch will need a 3.11 or 3.12 venv.

## Roadmap

V1 first: complete measured loop, then plots, then optional Three.js UI.

After V1: scale calibration, 3D, extra collision planes, rotation, friction. Not Day 1.
