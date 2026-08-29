# Newton runtime

One executable 3D physics path:

```text
PhysicalScene JSON
  -> isolated Python 3.11 subprocess
  -> Newton XPBD on Warp CUDA
  -> SimulatedWorldState JSON
  -> Three.js playback
```

This is forward simulation. It does not fit parameters to DA3 or TRAM.

## Tested runtime and package pins

- Measured Python: `3.11.15`
- Newton package: `newton==1.5.1`
- Newton release commit: `17c82b57c0cf369ee23baa776636fc633b82ccfa`
- Newton license: Apache-2.0
- Warp package: `warp-lang==1.16.0`
- Warp release commit: `86ec8b78cbef8bb570a9877e351ac0f365718e30`
- Warp source license: Apache-2.0
- Installed Warp runtime: CUDA `12.9`
- Measured NVIDIA driver: `610.88`
- Measured device: `NVIDIA GeForce RTX 4080 SUPER`, 16,376 MiB

The PyPI Warp wheel includes its CUDA runtime. A local CUDA Toolkit is not
required. Warp's CUDA 12 wheel requires NVIDIA driver 525 or newer. Newton
1.5.1 sets a stricter CUDA 12 minimum of driver 545, so this runtime requires
545 or newer.

Newton 1.5.1 requires Python 3.10 or newer and depends on Warp 1.16.0 or newer.
Both packages are pinned so the contract and measurements refer to one tested
combination.

The Warp wheel also ships third-party components under their own notices.
The installed `warp_lang-1.16.0.dist-info\licenses\licenses` directory includes
the CUDA, NVRTC, libmathdx, LLVM, NanoVDB, USD, CUBQL, and other bundled
component license files. Apache-2.0 describes Warp source, not every bundled
binary.

## Why Newton XPBD

Newton 1.5.1 has a native distance joint:

```python
ModelBuilder.add_joint_distance(
    parent,
    child,
    parent_xform=...,
    child_xform=...,
    min_distance=...,
    max_distance=...,
)
```

Newton's solver support table states that `SolverXPBD` is the only current
solver that enforces `DISTANCE`. `SolverSemiImplicit` and
`SolverFeatherstone` treat it as a free joint. The tether fixture therefore
uses:

- one body created with `ModelBuilder.add_link`
- world parent index `-1`
- `parent_xform` at the world anchor
- `child_xform` at the body-local attachment
- equal `min_distance` and `max_distance`
- `SolverXPBD` with 16 iterations on the 240 Hz fixture

Equal minimum and maximum distances make this a bilateral fixed-length
constraint. It is not a slack rope.

Official sources:

- Newton 1.5.1 installation: <https://github.com/newton-physics/newton/blob/v1.5.1/docs/guide/installation.rst>
- Newton 1.5.1 solver support: <https://github.com/newton-physics/newton/blob/v1.5.1/docs/solvers/index.rst>
- Newton 1.5.1 `ModelBuilder`: <https://github.com/newton-physics/newton/blob/v1.5.1/newton/_src/sim/builder.py>
- Newton tag: <https://github.com/newton-physics/newton/tree/v1.5.1>
- Warp 1.16.0 installation: <https://github.com/NVIDIA/warp/blob/v1.16.0/docs/user_guide/installation.rst>
- Warp tag: <https://github.com/NVIDIA/warp/tree/v1.16.0>

Free-fall uses `builder.add_body` (implicit free joint) and writes linear
velocity into `joint_qd`. XPBD is semi-implicit Euler. Analytic Y can be about
`0.5 g t dt` off at 60 Hz.

## Environment boundary

Run physics in `.venv-physics`. Keep DA3, TRAM, SAM 2, FastAPI, and the image-space
tools in `.venv`.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-physics.ps1
```

```powershell
.\.venv-physics\Scripts\python.exe -m physics3d.simulate_physical_scene `
  contracts\3d\v1\examples\physical_scene_tether.json `
  --output results\physics3d\p4-tether `
  --repeat-check
```

## Input payload

`contracts/3d/v1/examples/physical_scene_tether.json` is a standalone
metric scene. It does not claim a source observation.

- gravity: `[0, -9.80665, 0]` m/s²
- body: one 1 kg sphere with 0.18 m radius
- body initial position: `[1.2, 1.35, 0.5567764363]` m
- body initial orientation: 20 degrees about `+Y`
- body initial velocity: `[0.25, 0.2, 0]` m/s
- anchor: `[0, 3, 0]` m
- body-local attachment: `[0, 0.15, 0]` m
- fixed distance: `2` m
- duration: `4` s
- fixed step: `1/240` s
- solver steps: `960`

The initial attachment lies exactly 2 m from the anchor. Its nonzero X and Z
offset and tangential initial velocity produce motion in X, Y, and Z.

The scene declares mass. The adapter derives sphere density as
`mass / (4*pi*r^3/3)`. Newton uses that density and geometry to compute the
body inertia.

## Coordinates and units

The simulator boundary preserves the project convention:

- right-handed world
- `+Y` up
- meters, kilograms, seconds, and radians
- `T_parent_child`
- column-vector transform math
- row-major JSON matrices
- translation at flat indices 3, 7, and 11
- quaternions in `[x, y, z, w]` order

No axis swap is needed. The adapter creates `ModelBuilder` with
`up_axis=newton.Axis.Y` and passes the project gravity vector directly. It
also calls `Model.set_gravity` with the same vector after finalization.

Newton 1.5.1 exposes `State.body_qd` as linear velocity followed by angular
velocity. Both are in world coordinates. This differs from Warp's native
spatial-vector convention, so the adapter follows Newton's public convention.

Tests reject `+Z` up, column-major declarations, non-meter lengths,
transposed transforms, and a body attachment that does not start at the
declared distance.

## Rollout contract

`phystwin.simulated_world_state` version 1 contains:

- source `PhysicalScene` ID and canonical SHA-256
- Newton, Warp, solver, CUDA, and device identity
- the physical coordinate and unit declarations
- gravity
- fixed-step timeline
- body shape, mass, transforms, and velocities
- world anchor, body-local attachment, and rest length
- execution time, step count, output count, and Warp mempool high-water mark
- finite-state, time, gravity, tether, and XYZ-motion validation
- repeated-run metadata
- warnings and failures

Three.js reads this JSON. It does not step or correct the simulation.

## Measured fixture result

One repeated CLI run on the device above produced:

- 960 steps and 961 output samples
- 4.0 s simulated duration
- fixed step `0.004166666666666667` s
- primary wall time `8.317180` s
- Warp mempool peak GPU allocation `70,706` bytes, about 69 KiB
- maximum tether error `0.000001051` m
- RMS tether error `0.000000552` m
- body X range `2.397290` m
- body Y range `0.498717` m
- body Z range `1.082716` m
- repeated-run maximum transform delta `0`

The GPU-memory value is Warp's CUDA mempool used-memory high-water mark.
It is process-local allocator data, not a full driver-level profiler trace.
The first run after changing kernels can also include Warp compilation time.

Warp `RUN_TO_RUN` determinism was requested. The repeated run on this GPU was
bit-identical for the serialized body transforms. This does not claim
cross-GPU or cross-version determinism.

## Current limits

- The runtime supports one sphere and either free fall or one world-to-body
  bilateral fixed-distance constraint. It does not model a slack rope.
- It does not model contacts, damping, or ropes with mass.
- XPBD enforces distance numerically. The rollout records max and RMS error.
- A 0.50 m rod is not rigid in this XPBD setup. The 2 m fixture is.
- The GPU-memory value does not include allocations outside Warp's mempool.
