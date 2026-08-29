# Reconstruction evaluation

This evaluator compares a human `SceneObservation` to camera and SMPL24 ground
truth. It does not run physics or fit a `PhysicalScene`.

## EMDB access

The EMDB code is MIT. The evaluator pins the public repository at
`9a4eab677181a3789bda7ba5c36ab8cff797380c`.

The EMDB dataset has different terms. It is limited to approved
non-commercial academic use. Access requires an institutional email at
<https://emdb.ait.ethz.ch/>. Do not commit or redistribute its videos,
annotations, or derived body data.

EMDB stores annotations in `*_data.pkl`. Only load pickle files downloaded
from the official ETH site. Pickle can execute code while loading.

SMPL model files have their own registration terms. Download them separately.
Keep both EMDB and SMPL outside this repository.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-evaluation.ps1
```

Run a synthetic coordinate and metric check without EMDB:

```powershell
.\.venv\Scripts\python.exe vision\evaluate_reconstruction.py `
  --fixture `
  --output results\evaluation3d\p3-fixture
```

This check adds a known `0.05 m` body translation. It must report:

- `50 mm` world MPJPE
- `0.05 m` root RMSE
- approximately zero pelvis-aligned MPJPE
- approximately zero PA-MPJPE
- zero camera error

It is not benchmark evidence.

Run one approved EMDB sequence:

```powershell
.\.venv\Scripts\python.exe vision\evaluate_reconstruction.py `
  --observation results\cache\humans\<key>\scene_observation.json `
  --emdb-sequence D:\datasets\EMDB\P0\<sequence> `
  --smpl-model-root D:\models\smplx `
  --accept-emdb-license `
  --output results\evaluation3d\<sequence>
```

The `P0` folder name is EMDB's own subject layout, not a PhysTwin label.

The output folder contains `reconstruction_evaluation.json` and
`trajectory_comparison.svg`.

## Alignment

Samples match by the original source frame index. Missing predictions are not
interpolated.

EMDB camera extrinsics are OpenCV world-to-camera matrices. The evaluator
inverts them and applies the same gauge as DA3 and TRAM:

```text
T_obs_from_emdb = F * inverse(T_emdb_camera_first_prediction_frame)
F = diag(1, -1, -1, 1)
```

The prediction is already in its first-camera graphics world. The EMDB
reference is gauged at the same source frame. Direct world metrics apply no
extra rigid or similarity alignment.

## Metrics

- `camera_position_rmse_m`: camera-center trajectory error
- `camera_rotation_mean_deg` and `p95`: rotation geodesic error
- `root_position_rmse_m`: pelvis trajectory error
- `world_mpjpe_mm`: direct world-space joint error
- `pelvis_aligned_mpjpe_mm`: local articulation after removing each pelvis
- `pa_mpjpe_mm`: per-frame Procrustes error. This removes scale, rotation, and
  translation, so it does not validate world motion
- `camera_scale_aligned_rmse_m`: a diagnostic scalar fit around the shared
  first-camera origin. It is not the headline metric
- `reprojection_mpjpe_px`: emitted only when lens distortion is declared
  removed or absent. `unknown` blocks this metric. DA3 observations with
  varying intrinsics also block it because the core contract stores only the
  first sample's intrinsics

Direct metric errors are blocked when prediction scale is `relative`.
`metric_assumed` values are reported but remain estimator assumptions.

## Status

The evaluator and the synthetic fixture are in the repo. There is no approved
EMDB sequence or registered SMPL model in this workspace. No EMDB metric is
claimed. The adapter stays for later use.
