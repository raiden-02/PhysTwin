# IRIS falling_ball/big/01

This is the successful real-video gravity recovery for PhysTwin V2.

The pendulum case is closed. Do not treat that failure as this result.

## Clip

- Dataset: IRIS (`rasulkhanbayov/IRIS`), evidence kind `external_dataset`
- File: `Falling_ball/big/01.mp4`
- Video SHA-256: `e7aa7ec3d88ee9dafd7d5aeba806de0a713585547e1ee243ae7ac61421cb01ea`
- Window: 1.40 s to 1.90 s, 16 frames
- Ball radius (metric scale): 0.11 m
- Drop height (context only): 1.00 m
- IRIS gravity is evaluation ground truth only. It was not used to initialize or bound the fit.

## Reconstruction

Method: SAM2 mask, image-space center and horizontal silhouette radius, DA3 camera intrinsics, pinhole projected-sphere depth `Z = sqrt((f R / r)^2 + R^2)`.

DA3 depth is not used for the moving ball. The camera is treated as static. Measured DA3 pose drift was 3 mm.

Accepted frames: 16. Rejected frames: 0.

Depth stayed near 1.45 m to 1.59 m. The ball first rises, then falls. That matches a toss or late release, not a drop from rest.

## Fit

Profile: `free_fall_gravity_v1`. Every objective evaluation ran Newton 1.5.1 / Warp 1.16 on CUDA.

- Recovered gravity: 9.044 m/s²
- IRIS ground truth: 9.81 m/s²
- Absolute error: 0.766 m/s²
- Percent error: 7.8%
- Trajectory RMSE: 0.063 m
- Normalized RMSE: 0.106
- Observed extent: 0.596 m
- Fitted initial vertical velocity: +1.17 m/s
- Optimizer evaluations: 134
- Runtime: 26.2 s
- GPU: NVIDIA GeForce RTX 4080 SUPER
- Bound hits: none

The gravity error is inside the 10% ideal target. Normalized RMSE is just above 0.10 and inside the 0.20 acceptance target. The optimizer did not sit on a bound. The visual overlay follows the same rise-then-fall path.

## Counterfactual

The fitted scene was cloned. Only gravity magnitude changed to 1.62 m/s² (Moon).

- Source fitted scene SHA-256: `edb9f8298f8e19456448b00fd094b04ef1bf443cb98d2368f14d9c77cc27f450`
- Moon scene SHA-256: `24d060413dfb6e190fe8ed2569f375545ceef8cdef8708e8a1b227db081f584d`
- Moon rollout SHA-256: `b1fc4a9473dffae40c8449d1f473418c8d766f0096b05db999ce6085180641a4`

The Moon rollout is a simulated hypothesis. It was not observed.

## Limitations

- Metric scale uses the measured IRIS ball radius.
- Gravity direction is `-Y` from an assumed level camera.
- The body is one rigid sphere. There is no drag, bounce, or contact.
- The last frames still include some depth drift from apparent-radius change.

## Gate

Accepted as portfolio evidence: gravity error 7.8%, normalized RMSE 0.106, no bound hit, same qualitative motion.
