# Recorded, rendered, synthetic, and explicit poor-fit evaluation.
# Writes measured JSON and demo artifacts.
# Does not invent metrics.
param(
    [switch]$SkipTracking,
    [string]$PendulumPoint = "111,858",
    [string]$PendulumPivot = "385,92"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$py = ".\.venv\Scripts\python.exe"
$exe = ".\build\Release\phystwin.exe"
$synth = ".\build\Release\phystwin_synthetic_fit_test.exe"
$pendulumSynth = ".\build\Release\phystwin_pendulum_fit_test.exe"

if (-not (Test-Path $py)) { throw "missing $py. Run scripts\setup-vision.ps1 first." }
if (-not (Test-Path $exe)) { throw "missing $exe. Run scripts\build.ps1 first." }
if (-not (Test-Path $synth)) { throw "missing $synth. Run scripts\build.ps1 first." }
if (-not (Test-Path $pendulumSynth)) { throw "missing $pendulumSynth. Run scripts\build.ps1 first." }
if (-not (Test-Path samples\bounce.mp4)) {
    throw "missing samples\bounce.mp4. Download the Mixkit tennis clip before running evaluation."
}

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "command failed with exit $LASTEXITCODE : $Command" }
}

New-Item -ItemType Directory -Force -Path `
    results\cases\synthetic, `
    results\cases\pendulum_synthetic, `
    results\cases\mixkit_tennis, `
    results\cases\mixkit_bad_ground, `
    results\cases\generated_diagonal, `
    results\cases\generated_drop, `
    results\cases\pendulum_recorded, `
    docs\demo | Out-Null

Write-Host "=== synthetic recovery ==="
$synthLog = "results\cases\synthetic\stdout.txt"
cmd /c ".\build\Release\phystwin_synthetic_fit_test.exe > $synthLog"
if ($LASTEXITCODE -ne 0) { throw "synthetic_fit test failed" }
Get-Content $synthLog

Write-Host "=== pendulum synthetic recovery and robustness ==="
$pendulumSynthLog = "results\cases\pendulum_synthetic\stdout.txt"
cmd /c ".\build\Release\phystwin_pendulum_fit_test.exe > $pendulumSynthLog"
if ($LASTEXITCODE -ne 0) { throw "pendulum_fit test failed" }
Get-Content $pendulumSynthLog

Write-Host "=== generate clips ==="
Invoke-Checked { & $py vision\make_bounce_clip.py `
    --output samples\generated_diagonal.mp4 `
    --x0 80 --y0 40 --vx 140 --vy 20 --g 1800 --e 0.72 }
Invoke-Checked { & $py vision\make_bounce_clip.py `
    --output samples\generated_drop.mp4 `
    --x0 320 --y0 36 --vx 4 --vy 50 --g 1600 --e 0.40 }

if (-not (Test-Path results\cases\mixkit_tennis\tracking.json)) {
    if ($SkipTracking) {
        throw "Mixkit tracking.json is missing and -SkipTracking was set"
    }
    Write-Host "=== track Mixkit tennis ==="
    Invoke-Checked { & $py vision\track.py samples\bounce.mp4 --point 375,722 `
        --output results\cases\mixkit_tennis\tracking.json `
        --viz results\cases\mixkit_tennis\tracking_preview.png }
}

if (-not $SkipTracking) {
    if (-not (Test-Path results\cases\generated_diagonal\tracking.json)) {
        Write-Host "=== track generated diagonal ==="
        Invoke-Checked { & $py vision\track.py samples\generated_diagonal.mp4 --point 80,40 `
            --output results\cases\generated_diagonal\tracking.json `
            --viz results\cases\generated_diagonal\tracking_preview.png }
    }
    if (-not (Test-Path results\cases\generated_drop\tracking.json)) {
        Write-Host "=== track generated drop ==="
        Invoke-Checked { & $py vision\track.py samples\generated_drop.mp4 --point 320,36 `
            --output results\cases\generated_drop\tracking.json `
            --viz results\cases\generated_drop\tracking_preview.png }
    }
}

$hasPendulumClip = Test-Path samples\recorded\pendulum.mp4
$pendulumTracking = "results\cases\pendulum_recorded\tracking.json"
$pendulumReconstruction = "results\cases\pendulum_recorded\reconstruction.json"
if ($hasPendulumClip -and -not (Test-Path $pendulumTracking)) {
    if ($SkipTracking) {
        throw "pendulum tracking is missing and -SkipTracking was set"
    }
    Write-Host "=== track recorded physical pendulum ==="
    Invoke-Checked { & $py vision\track.py samples\recorded\pendulum.mp4 `
        --model pendulum --point $PendulumPoint --pivot $PendulumPivot `
        --output $pendulumTracking `
        --viz results\cases\pendulum_recorded\tracking_preview.png }
}

Write-Host "=== fit ==="
Invoke-Checked { & $exe fit results\cases\mixkit_tennis\tracking.json `
    --output results\cases\mixkit_tennis\reconstruction.json }
Invoke-Checked { & $exe fit results\cases\generated_diagonal\tracking.json `
    --output results\cases\generated_diagonal\reconstruction.json }
Invoke-Checked { & $exe fit results\cases\generated_drop\tracking.json `
    --output results\cases\generated_drop\reconstruction.json }
if ($hasPendulumClip) {
    & $exe fit $pendulumTracking --output $pendulumReconstruction
    if ($LASTEXITCODE -notin 0, 2) {
        throw "pendulum fit failed with exit $LASTEXITCODE"
    }
}

Write-Host "=== explicit poor-fit case (ground below observed centroids) ==="
& $exe fit results\cases\mixkit_tennis\tracking.json `
    --ground-y 800 `
    --output results\cases\mixkit_bad_ground\reconstruction.json
if ($LASTEXITCODE -ne 2) {
    throw "expected exit code 2 for --ground-y 800, got $LASTEXITCODE"
}
Copy-Item results\cases\mixkit_tennis\tracking.json `
    results\cases\mixkit_bad_ground\tracking.json -Force
if (Test-Path results\cases\mixkit_tennis\tracking_raw.json) {
    Copy-Item results\cases\mixkit_tennis\tracking_raw.json `
        results\cases\mixkit_bad_ground\tracking_raw.json -Force
}

Write-Host "=== plots and overlays ==="
Invoke-Checked { & $py vision\plot_reconstruction.py `
    results\cases\mixkit_tennis\tracking.json `
    results\cases\mixkit_tennis\reconstruction.json `
    --output results\cases\mixkit_tennis\reconstruction_preview.png `
    --title "Mixkit tennis bounce (recorded)" }
Invoke-Checked { & $py vision\overlay_comparison.py `
    samples\bounce.mp4 `
    results\cases\mixkit_tennis\tracking.json `
    results\cases\mixkit_tennis\reconstruction.json `
    --output results\cases\mixkit_tennis\overlay.mp4 `
    --gif docs\demo\mixkit_overlay.gif `
    --still docs\demo\mixkit_overlay.png `
    --panel-height 320 `
    --gif-stride 4 `
    --gif-max-width 540 `
    --gif-colors 48 `
    --title "Mixkit tennis bounce (recorded)" }
Invoke-Checked { & $py vision\plot_reconstruction.py `
    results\cases\generated_diagonal\tracking.json `
    results\cases\generated_diagonal\reconstruction.json `
    --output results\cases\generated_diagonal\reconstruction_preview.png `
    --title "Generated diagonal bounce" }
Invoke-Checked { & $py vision\overlay_comparison.py `
    samples\generated_diagonal.mp4 `
    results\cases\generated_diagonal\tracking.json `
    results\cases\generated_diagonal\reconstruction.json `
    --output results\cases\generated_diagonal\overlay.mp4 `
    --gif docs\demo\diagonal_overlay.gif `
    --still docs\demo\diagonal_overlay.png `
    --title "Generated diagonal bounce" }
Invoke-Checked { & $py vision\plot_reconstruction.py `
    results\cases\generated_drop\tracking.json `
    results\cases\generated_drop\reconstruction.json `
    --output results\cases\generated_drop\reconstruction_preview.png `
    --title "Generated near-vertical drop" }
Invoke-Checked { & $py vision\overlay_comparison.py `
    samples\generated_drop.mp4 `
    results\cases\generated_drop\tracking.json `
    results\cases\generated_drop\reconstruction.json `
    --output results\cases\generated_drop\overlay.mp4 `
    --gif docs\demo\drop_overlay.gif `
    --still docs\demo\drop_overlay.png `
    --title "Generated near-vertical drop" }
if ($hasPendulumClip) {
    Invoke-Checked { & $py vision\plot_reconstruction.py `
        $pendulumTracking `
        $pendulumReconstruction `
        --output results\cases\pendulum_recorded\reconstruction_preview.png `
        --title "Recorded physical pendulum" }
    Invoke-Checked { & $py vision\overlay_comparison.py `
        samples\recorded\pendulum.mp4 `
        $pendulumTracking `
        $pendulumReconstruction `
        --output results\cases\pendulum_recorded\overlay.mp4 `
        --gif docs\demo\pendulum_recorded_overlay.gif `
        --still docs\demo\pendulum_recorded_overlay.png `
        --panel-height 320 `
        --gif-stride 4 `
        --gif-max-width 540 `
        --gif-colors 48 `
        --title "Recorded physical pendulum" }
}

$manifest = @{
    cases = @(
        @{
            id = "synthetic_noise_free"
            title = "C++ synthetic recovery"
            kind = "cpp_synthetic"
            stdout = "results/cases/synthetic/stdout.txt"
            notes = "Noise-free 241-frame two-bounce case from phystwin_synthetic_fit_test. Not video evidence."
        }
        @{
            id = "pendulum_synthetic"
            title = "C++ pendulum synthetic recovery"
            kind = "cpp_pendulum_synthetic"
            stdout = "results/cases/pendulum_synthetic/stdout.txt"
            notes = "Full nonlinear damped-pendulum recovery, deterministic noise/outliers, and five degenerate-input checks. Not video evidence."
        }
        @{
            id = "mixkit_tennis"
            title = "Mixkit tennis bounce"
            kind = "recorded_video"
            video = "samples/bounce.mp4"
            point = "375,722"
            source = "https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/"
            notes = "Slow-motion close-up recorded clip. Primary external-validity case."
            tracking = "results/cases/mixkit_tennis/tracking.json"
            tracking_raw = "results/cases/mixkit_tennis/tracking_raw.json"
            reconstruction = "results/cases/mixkit_tennis/reconstruction.json"
            plot = "results/cases/mixkit_tennis/reconstruction_preview.png"
            overlay = "results/cases/mixkit_tennis/overlay.mp4"
            gif = "docs/demo/mixkit_overlay.gif"
            still = "docs/demo/mixkit_overlay.png"
        }
        @{
            id = "mixkit_bad_ground"
            title = "Mixkit explicit bad ground"
            kind = "recorded_video_failure"
            video = "samples/bounce.mp4"
            point = "375,722"
            notes = "Same Mixkit tracking with --ground-y 800. Expected quality poor and CLI exit 2."
            tracking = "results/cases/mixkit_bad_ground/tracking.json"
            tracking_raw = "results/cases/mixkit_bad_ground/tracking_raw.json"
            reconstruction = "results/cases/mixkit_bad_ground/reconstruction.json"
        }
        @{
            id = "generated_diagonal"
            title = "Generated diagonal bounce"
            kind = "generated_video"
            video = "samples/generated_diagonal.mp4"
            point = "80,40"
            notes = "Rendered diagonal bounce (vx=140, vy=20, g=1800, e=0.72). Same integrator family as the fitter. Pipeline check, not real-footage accuracy."
            tracking = "results/cases/generated_diagonal/tracking.json"
            tracking_raw = "results/cases/generated_diagonal/tracking_raw.json"
            reconstruction = "results/cases/generated_diagonal/reconstruction.json"
            plot = "results/cases/generated_diagonal/reconstruction_preview.png"
            overlay = "results/cases/generated_diagonal/overlay.mp4"
            gif = "docs/demo/diagonal_overlay.gif"
            still = "docs/demo/diagonal_overlay.png"
        }
        @{
            id = "generated_drop"
            title = "Generated near-vertical drop"
            kind = "generated_video"
            video = "samples/generated_drop.mp4"
            point = "320,36"
            notes = "Rendered near-vertical drop (vx=4, vy=50, g=1600, e=0.40). Same integrator family as the fitter. Pipeline check, not real-footage accuracy."
            tracking = "results/cases/generated_drop/tracking.json"
            tracking_raw = "results/cases/generated_drop/tracking_raw.json"
            reconstruction = "results/cases/generated_drop/reconstruction.json"
            plot = "results/cases/generated_drop/reconstruction_preview.png"
            overlay = "results/cases/generated_drop/overlay.mp4"
            gif = "docs/demo/drop_overlay.gif"
            still = "docs/demo/drop_overlay.png"
        }
    )
}
if ($hasPendulumClip) {
    $manifest.cases += @{
        id = "pendulum_recorded"
        title = "Recorded physical pendulum"
        kind = "recorded_video"
        video = "samples/recorded/pendulum.mp4"
        point = $PendulumPoint
        pivot = $PendulumPivot
        source = "https://www.youtube.com/shorts/ZveeQePGkNg"
        notes = "Fixed-camera physical pendulum. The source is trimmed at frame 70 to remove hand contact and start at a clean turning point."
        tracking = $pendulumTracking.Replace("\", "/")
        tracking_raw = "results/cases/pendulum_recorded/tracking_raw.json"
        reconstruction = $pendulumReconstruction.Replace("\", "/")
        plot = "results/cases/pendulum_recorded/reconstruction_preview.png"
        overlay = "results/cases/pendulum_recorded/overlay.mp4"
        gif = "docs/demo/pendulum_recorded_overlay.gif"
        still = "docs/demo/pendulum_recorded_overlay.png"
    }
}

$manifestPath = "results\cases\manifest.json"
($manifest | ConvertTo-Json -Depth 6) | Set-Content -Encoding utf8 $manifestPath
Write-Host "wrote $manifestPath"

Invoke-Checked { & $py vision\collect_evaluation.py --manifest $manifestPath --output docs\evaluation.json }
Invoke-Checked { & $py vision\plot_evaluation.py --manifest $manifestPath --output docs\demo\observed_vs_simulated.png }

Write-Host "Evaluation artifacts are in docs\demo and docs\evaluation.json"
