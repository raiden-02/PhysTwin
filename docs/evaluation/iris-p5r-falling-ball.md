# IRIS falling_ball/big/01

## Setup

- Dataset: IRIS (`rasulkhanbayov/IRIS`), evidence kind `external_dataset`
- File: `Falling_ball/big/01.mp4`
- Video SHA-256: `e7aa7ec3d88ee9dafd7d5aeba806de0a713585547e1ee243ae7ac61421cb01ea`
- Window: 1.40 s to 1.90 s, 16 frames
- Ball radius (metric scale): 0.11 m
- Drop height (context only): 1.00 m
- IRIS gravity is evaluation ground truth. The prepare and fit path never load it.

## Reconstruction method

SAM2 mask, image-space center and horizontal silhouette radius, per-frame DA3
camera intrinsics, pinhole projected-sphere depth
`Z = sqrt((f R / r)^2 + R^2)`.

DA3 depth is not used for the moving ball. The camera is treated as static.
DA3 first-to-last camera translation is `0.00298` reconstruction units
(relative scale, not meters).

DA3 reports varying K. This run used each sample's K. Focal-length relative
span was about 0.5% (`fx` 2468.7 to 2481.4 px). That is small, but frame 0
is not used as a silent fallback.

Accepted frames: 16. Rejected frames: 0.

Depth stayed near 1.45 m to 1.59 m. The ball first rises, then falls. That
matches a toss or late release, not a drop from rest.

## Fit

Profile: `free_fall_gravity_v1`. Every objective evaluation ran Newton 1.5.1 /
Warp 1.16 on CUDA.

- Recovered gravity: 9.912 m/s²
- IRIS ground truth: 9.81 m/s²
- Absolute error: 0.102 m/s²
- Percent error: 1.0%
- Trajectory RMSE: 0.064 m
- Normalized RMSE: 0.107
- Observed extent: 0.597 m
- Fitted initial vertical velocity: +1.34 m/s
- Optimizer evaluations: 88
- Runtime: 18.6 s
- GPU: NVIDIA GeForce RTX 4080 SUPER
- Bound hits: none

Gravity error is well inside 10%. Normalized RMSE is just above 0.10 and
inside 0.20. The optimizer did not sit on a bound. The overlay follows the
same rise-then-fall path.

## Result

This is a successful real-video gravity recovery on one IRIS clip. It is not
a general video-to-physics benchmark.

## Counterfactual

The fitted scene was cloned. Only gravity magnitude changed to 1.62 m/s²
(Moon).

- Fitted scene SHA-256: `ecc26452a0baff48e4ccd4b87c06a92f8cf3da6d2ae89021d7417819fe67f019`
- Fitted rollout SHA-256: `3283d5ee9a04a394509556c1464baa785eaa8d8895fa3110ad6b2059f87537b9`
- Moon scene SHA-256: `7365ba5cbec9f736fe91c2f3b84cd6079307c586fb3226a6329c6c2f1f119028`
- Moon rollout SHA-256: `9501df37bb65eed75ad34e762b2f6c7ceb137a2c42eaf25795d8c817b419cf6a`

The Moon rollout is a simulated hypothesis. It was not observed.

## Assumptions

- Metric scale uses the measured IRIS ball radius.
- Gravity direction is `-Y` from an assumed level camera.
- The body is one rigid sphere.

## Limitations

- There is no drag, bounce, or contact.
- The last frames still include some depth drift from apparent-radius change.
- Normalized RMSE is 0.107, so the trajectory match is usable but not tight.
- One clip does not establish a dataset-wide error rate.
