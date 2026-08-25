Keep large videos out of git.

Put Mixkit's tennis clip at `samples/bounce.mp4` for the recorded case
(https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/).
Click `--point 375,722`.

`vision/make_bounce_clip.py` writes the generated diagonal and drop clips used
by `scripts/run-eval.ps1`. Those mp4 files stay local.

`example_tracking.json` is the V1 Python→C++ contract fixture.
