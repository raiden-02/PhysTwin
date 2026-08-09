# PhysTwin

PhysTwin turns a short real-world video into a small, quantitatively fitted physics reconstruction: select one moving object, track it on the GPU, infer the physical parameters that best reproduce its motion, and compare the recorded and simulated trajectories side by side.

**Status:** Checkpoint 1 synthetic core. The C++20 project builds, simulates deterministic 2D bounce motion, recovers known synthetic parameters, and computes RMSE/MAE. SAM 2 tracking, real-video fitting through the CLI, and the demo are not implemented yet.

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

Shipped: project layout, C++ types, JSON inspect path, deterministic simulator, synthetic generator, unweighted least-squares fitter, RMSE/MAE, and Python worker stubs.

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

Not implemented. Checkpoint 2.

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

Python worker (stub, exits 2):

```powershell
python vision\track.py input.mp4 --point 531,312 --output tracking.json
```

## Limitations

- Synthetic core only. No real-video tracker or CLI reconstruction yet.
- Monocular video has no metric scale. Later fits will report a gravity scale in px/s².
- One object, fixed camera, one ground line, no spin, no drag in V1.
- System Python on this machine is 3.14 beta. SAM 2 / PyTorch will need a 3.11 or 3.12 venv.

## Roadmap

V1 first: complete measured loop, then plots, then optional Three.js UI.

After V1: scale calibration, 3D, extra collision planes, rotation, friction. Not Day 1.
