Keep large videos out of git.

Put Mixkit's tennis clip at `samples/bounce.mp4` for the recorded case
(https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/).
Click `--point 375,722`.

Put a physical pendulum clip at `samples/recorded/pendulum.mp4`. Use a fixed
camera, visible bob, visible pivot, several swings, and minimal occlusion.

The current local file is a fixed-camera physical pendulum. Its first 70 source
frames were removed to exclude hand contact and start at a clean turning point.
It is H.264 with `yuv420p` pixel format so browser playback works. Use target
`111,858` and pivot `385,92`.

`vision/make_bounce_clip.py` writes the generated diagonal and drop clips used
by `scripts/run-eval.ps1`. Those mp4 files stay local.

`example_tracking.json` is the Python-to-C++ contract fixture.
