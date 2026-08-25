# PhysTwin

PhysTwin turns a short real-world video into a small, quantitatively fitted physics reconstruction: select one moving object, track it on the GPU, infer the physical parameters that best reproduce its motion, and compare the recorded and simulated trajectories side by side.

**Status:** Checkpoint 0 scaffold. The C++20 project builds, the `tracking.json` contract loads, and the CLI/test targets exist. Simulation, fitting, SAM 2 tracking, and demo metrics are not implemented yet.

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

Shipped today: project layout, C++ types, JSON inspect path, Python worker stubs.

## Results

No measured results yet. Do not copy placeholder RMSE or recovered parameters onto a résumé.

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

Not implemented. Checkpoint 1 will recover known synthetic parameters before any real video is trusted.

## GPU video tracking

Not implemented. Checkpoint 2.

## Evaluation

Not implemented. Target: synthetic recovery plus three short clips, with RMSE taken from saved files.

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

- Scaffold only. No physics, no fitter, no tracker.
- Monocular video has no metric scale. Later fits will report a gravity scale in px/s².
- One object, fixed camera, one ground line, no spin, no drag in V1.
- System Python on this machine is 3.14 beta. SAM 2 / PyTorch will need a 3.11 or 3.12 venv.

## Roadmap

V1 first: complete measured loop, then plots, then optional Three.js UI.

After V1: scale calibration, 3D, extra collision planes, rotation, friction. Not Day 1.
