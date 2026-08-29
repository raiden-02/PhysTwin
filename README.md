# PhysTwin

Turn ordinary video into an executable 3D physics model.

Real IRIS video → metric 3D trajectory → recovered gravity → Newton GPU rollout → Moon-gravity counterfactual.

On `Falling_ball/big/01.mp4`, PhysTwin recovered **9.04 m/s²** versus IRIS **9.81 m/s²** (7.8% error). Trajectory RMSE is **0.063 m** (normalized 0.106) on an RTX 4080 SUPER.

Technologies actually used: C++20, Python, SAM2, Depth Anything 3, NVIDIA Newton 1.5.1, Warp 1.16, CUDA, React, TypeScript, Three.js.

## What it does

PhysTwin reconstructs metric 3D motion from a real clip, fits an executable rigid-body scene by running Newton/Warp on every optimizer step, then resimulates the same drop under different gravity.

The default UI shows one finished example: IRIS Falling Ball.

```text
1  Real Video
2  Reconstructed 3D Motion
3  Fitted Physics
4  Counterfactual (Moon gravity)
```

## Demo

Clip: IRIS `Falling_ball/big/01.mp4`, window 1.40–1.90 s, 16 frames.

- Ball radius used for metric scale: 0.11 m
- Reconstruction: SAM2 mask + projected-sphere depth. DA3 depth is not used for the ball.
- Recovered gravity: 9.044 m/s²
- IRIS ground truth (unread during the fit): 9.81 m/s²
- Gravity error: 7.8%
- RMSE: 0.063 m
- Normalized RMSE: 0.106
- GPU: NVIDIA GeForce RTX 4080 SUPER, 134 Newton evaluations, 26.2 s

The Moon path is a simulated hypothesis. It was not observed.

Details: [docs/evaluation/iris-p5r-falling-ball.md](docs/evaluation/iris-p5r-falling-ball.md)

## Pipeline

```text
Video
  → SAM2 tracking
  → DA3 camera / scene reconstruction
  → metric sphere reconstruction
  → PhysicalMotionObservation
  → Newton + Warp inverse fit
  → PhysicalScene
  → counterfactual rollout
```

## Technical architecture

- **C++20**: V1 image-space physics and the native test suite
- **Python**: reconstruction, fitting, FastAPI
- **SAM2**: object masks
- **DA3**: camera intrinsics and a static-camera check
- **Newton 1.5.1 / Warp 1.16 / CUDA**: executable 3D rigid-body simulation
- **React / TypeScript / Three.js**: portfolio demo and lab inspectors

## Evaluation

Keep these three results separate.

**Synthetic** (Newton generates the motion, then the fitter recovers parameters):

- P5 tether: rest length and two tangent speeds, RMSE 1.58 mm
- Free-fall: gravity recovered near 9.81 m/s² without using that value as the search start

**External real dataset** (IRIS):

- Falling ball: gravity 9.04 vs 9.81 m/s², RMSE 0.063 m. This is the V2 real-physics proof.
- Pendulum `pendulum_45/01`: rejected. RMSE 0.616 m on a 0.320 m extent. Small-object DA3 depth and a 0.50 m XPBD rod both failed. The system did not present that fit as valid physics. See [docs/evaluation/iris-p5r-pendulum-45-01-diagnosis.md](docs/evaluation/iris-p5r-pendulum-45-01-diagnosis.md).

**Counterfactual**:

- Same fitted drop, gravity changed to 1.62 m/s². Simulated only.

V1 also has recorded 2D Mixkit and pendulum numbers in [docs/evaluation.json](docs/evaluation.json). Those are image-space fits, not metric 3D gravity recovery.

## Limitations

- Monocular reconstruction is ambiguous. The successful case uses the measured ball radius for metric scale.
- Physics is one rigid sphere plus gravity. No contact, drag, or articulated humans.
- Counterfactuals are simulated hypotheses, not observations.
- The pendulum failure is kept as honest evidence. It is not the default demo.

## Run locally

Windows. Visual Studio 18 Community, Node.js, CUDA GPU.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup-vision.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup-physics.ps1
```

Portfolio UI (binds 127.0.0.1). If port 8765 is taken, set `PHYSTWIN_PORT=8766`:

```powershell
cd frontend
npm install
npm run build
cd ..
$env:PHYSTWIN_PORT = "8766"
.\.venv\Scripts\python.exe vision\serve.py
```

IRIS media stays out of git. Download `rasulkhanbayov/IRIS` under `datasets/IRIS/` to see the source video. The 3D paths still load from the committed demo payload.

Falling-ball prepare and fit (needs the IRIS clip and both venvs):

```powershell
.\.venv\Scripts\python.exe vision\prepare_falling_ball.py
.\.venv-physics\Scripts\python.exe -m physics3d.fit_physical_scene `
  results\physics3d\p5r-falling-ball\aligned_physical_scene_template.json `
  --motion-observation results\physics3d\p5r-falling-ball\target_motion_observation.json `
  --profile free_fall_gravity_v1 `
  --output results\physics3d\p5r-falling-ball
```

CTest:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

## Future work

Not in V2:

- articulated humans
- active control
- candidate physics / model selection
- automatic physical hypothesis generation
- cinematic Spider-Man stress test
- richer contact mechanics

The 3D design notes live in [docs/architecture-3d.md](docs/architecture-3d.md).
