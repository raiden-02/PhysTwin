# IRIS pendulum_45/01 failure diagnosis

This audit uses the existing `results/physics3d/p5r-real-fit` artifacts. It
does not rewrite those files. It does not rerun the optimizer.

The saved fit remains a poor residual. The cause is the observation and the
0.50 m XPBD rod, not a small search-budget miss.

## Reconstruction before scale

Unscaled DA3+SAM2 `anchor → target` distances, 16 frames:

| sample | source frame | t (s) | distance (world units) |
|---|---|---|---|
| 0 | 120 | 0.000 | 0.763 |
| 1 | 136 | 0.267 | 0.620 |
| 2 | 152 | 0.533 | 0.595 |
| 3 | 168 | 0.800 | 0.800 |
| 4 | 184 | 1.067 | 0.643 |
| 5 | 200 | 1.333 | 0.669 |
| 6 | 216 | 1.600 | 0.736 |
| 7 | 232 | 1.867 | 0.615 |
| 8 | 247 | 2.117 | 0.593 |
| 9 | 263 | 2.383 | 0.751 |
| 10 | 279 | 2.650 | 0.600 |
| 11 | 295 | 2.917 | 0.695 |
| 12 | 311 | 3.183 | 0.626 |
| 13 | 327 | 3.450 | 0.606 |
| 14 | 343 | 3.717 | 0.679 |
| 15 | 359 | 3.983 | 0.672 |

Stats in world units:

- min 0.593
- max 0.800
- mean 0.666
- median 0.656 (this is the calibration length)
- std 0.064
- coefficient of variation 0.096
- first-frame minus median +0.107

After the 0.50 / 0.656 scale, those distances are 0.452 m to 0.610 m.
A rigid 0.50 m rope should not move by 16 cm.

First-frame scaled length is 0.582 m. Projecting that first body point onto
the 0.50 m constraint sphere moves it 0.082 m.

## Anchor and target ranges

Unscaled anchor XYZ travel (supposedly fixed):

- X 0.022
- Y 0.050
- Z 0.128
- AABB diagonal 0.139

After scale that is about 1.7 cm, 3.8 cm, 9.8 cm. The pivot jitters 10 cm
in depth.

Unscaled target XYZ travel:

- X 0.375
- Y 0.043
- Z 0.184

Image-space target travel is 2239 px horizontal and 380 px vertical. The 3D
lift flattened the vertical swing to about 3 cm.

DA3 depths, then scaled by 0.762 m/unit:

- target median depth 0.530 m
- anchor median depth 0.933 m
- IRIS `camera_to_cable` 0.96 m

The clamp depth matches the IRIS camera distance. The tennis ball is about
43 cm too close. That depth split is why the reconstructed 3D "rope" leans
along the camera axis.

## First-frame angle vs IRIS 45°

Reconstructed first-frame angle from assumed physical down (`-Y`) is 57.1°.
Most of that angle is the camera-axis depth lean, not the image-plane swing.

The prepare window starts at t=2.0 s. Image-plane angle from vertical at that
frame is 10°. Later peaks are 36° to 39°. So t=2 s is near the bottom of the
swing, not the IRIS 45° release pose. Do not treat 57° vs 45° as a failed
release-angle check.

`level_camera` / first-camera `+Y` is plausible for a leveled lab camera.
The image has a real vertical swing. The 3D lift does not keep that vertical,
so the gravity axis is not usable on these XYZ samples.

## Tracking endpoints

Green is the lifted target pixel. Cyan is the seed click. Red is the lifted
anchor pixel. Magenta is the anchor seed.

The target SAM2 mask follows the tennis ball. The robust 3D center sits on
the ball. The seed is a few tens of pixels below that center. That is a
normal click-vs-centroid offset.

The anchor seed sits on the clamp body. The robust 3D center jumps to the
top corner of that clamp (first frame pixel 2073, 28 vs seed 2039, 92).
Anchor masks are 105 to 130 pixels. That is the clamp, not a huge support
cloud, and it is also not the point where the string leaves the hardware.

The 0.50 m IRIS `rope_length` was applied from `body_center` to that clamp
centroid. IRIS does not define the tape ends in `parameters.json`. The
physical rope starts at the string exit, below the clamp body. The
calibration pair is not clearly what IRIS measured.

## Newton constraint

No optimizer. Rest length held at 0.50 m. Speed along tangent `u` only.

Current 60 Hz, 24 XPBD iterations:

| u (m/s) | max tether error (m) | RMS (m) |
|---|---|---|
| 0.0 | 0.300 | 0.173 |
| 0.5 | 0.300 | 0.173 |
| 1.0 | 0.300 | 0.173 |
| 1.5 | 0.300 | 0.173 |
| 2.0 | 0.300 | 0.173 |

Same 0.50 m IRIS scene at 240 Hz, 16 or 32 iterations: max error still 0.300 m.

Synthetic P5 template, same solver:

| rest (m) | step | iterations | max error |
|---|---|---|---|
| 2.00 | 60 Hz | 24 | 0.77 µm |
| 0.50 | 60 Hz | 24 | 0.368 m |
| 0.50 | 240 Hz | 16 | 0.446 m |
| 0.50 | 240 Hz | 32 | 0.420 m |
| 0.50 | 480 Hz | 32 | 0.446 m |

The current joint holds a 2 m rod. It does not hold a 0.50 m rod. Finer
substeps do not fix it. A candidate with ~0.30 m tether error is not a valid
rigid-tether hypothesis. The 60 Hz / 24-iteration setup is not the isolated
cause. Short rest length is outside this XPBD distance-joint regime.

## Classification

Main failures, in order:

1. Bad 3D observation / DA3 depth. Bob at 0.53 m vs cable at 0.96 m. 3D
   rope length CV 9.6%. Vertical swing flattened.
2. Bad anchor endpoint. Clamp-mask centroid, not the string exit.
3. Bad scale calibration. Consequence of (1) and (2). Median 0.50 m is
   forced onto a varying, wrong 3D pair.
4. Newton constraint stability. 0.50 m XPBD rod is not rigid here, even
   on the synthetic template.
5. Wrong physical-up on the lifted points. The camera-level assumption is
   fine. The XYZ track is not.
6. Wrong initial state, secondary. t=2 s is near bottom, not 45° release.
   First body is 8 cm off the 0.50 m sphere.
7. Physical-model mismatch, secondary. Bifilar tennis ball vs an ideal rod.
8. Optimizer issue, not primary. The search had nothing physical to match.

No code fix in this checkpoint. A local solver tweak or more generations
cannot repair the lift or the pivot.

## What to do next

Do not spend another optimizer run on `pendulum_45/01`.

Pendulum is worth another attempt only after all three of these exist:

- a 3D lift that keeps bob and cable near 0.96 m
- a pivot at the actual string exit
- a constraint that can hold 0.50 m, or a scene scaled so rest length stays
  in the validated ~2 m regime

The faster correct P5R proof is IRIS `falling_ball`. That class has
`drop_height`, `ball_radius`, and `gravity`. Scale from `ball_radius` or
another non-gravity length, then recover gravity. No tether endpoint and no
XPBD distance joint.

Do not start P7 from the current pendulum fit.
