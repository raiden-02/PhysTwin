# Checkpoint 4 evaluation: two generated clips + Mixkit + synthetic recovery.
# Writes measured JSON and demo artifacts. Does not invent metrics.
param(
    [switch]$SkipTracking
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$py = ".\.venv\Scripts\python.exe"
$exe = ".\build\Release\phystwin.exe"
$synth = ".\build\Release\phystwin_synthetic_fit_test.exe"

if (-not (Test-Path $py)) { throw "missing $py. Run scripts\setup-vision.ps1 first." }
if (-not (Test-Path $exe)) { throw "missing $exe. Run scripts\build.ps1 first." }
if (-not (Test-Path $synth)) { throw "missing $synth. Run scripts\build.ps1 first." }

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "command failed with exit $LASTEXITCODE : $Command" }
}

New-Item -ItemType Directory -Force -Path `
    results\cases\synthetic, `
    results\cases\mixkit_tennis, `
    results\cases\generated_diagonal, `
    results\cases\generated_drop, `
    docs\demo | Out-Null

Write-Host "=== synthetic recovery ==="
$synthLog = "results\cases\synthetic\stdout.txt"
cmd /c ".\build\Release\phystwin_synthetic_fit_test.exe > $synthLog"
if ($LASTEXITCODE -ne 0) { throw "synthetic_fit test failed" }
Get-Content $synthLog

Write-Host "=== generate clips ==="
Invoke-Checked { & $py vision\make_bounce_clip.py `
    --output samples\generated_diagonal.mp4 `
    --x0 80 --y0 40 --vx 140 --vy 20 --g 1800 --e 0.72 }
Invoke-Checked { & $py vision\make_bounce_clip.py `
    --output samples\generated_drop.mp4 `
    --x0 320 --y0 36 --vx 4 --vy 50 --g 1600 --e 0.40 }

if (Test-Path samples\bounce.mp4) {
    if (-not (Test-Path results\cases\mixkit_tennis\tracking.json)) {
        if (Test-Path results\tracking.json) {
            Copy-Item results\tracking.json results\cases\mixkit_tennis\tracking.json -Force
            if (Test-Path results\tracking_raw.json) {
                Copy-Item results\tracking_raw.json results\cases\mixkit_tennis\tracking_raw.json -Force
            }
        } elseif (-not $SkipTracking) {
            Write-Host "=== track Mixkit tennis ==="
            Invoke-Checked { & $py vision\track.py samples\bounce.mp4 --point 375,722 `
                --output results\cases\mixkit_tennis\tracking.json `
                --viz results\cases\mixkit_tennis\tracking_preview.png }
        } else {
            throw "Mixkit tracking.json is missing and -SkipTracking was set"
        }
    }
} else {
    Write-Host "warning: samples\bounce.mp4 missing. Mixkit case will be skipped."
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

Write-Host "=== fit ==="
if (Test-Path results\cases\mixkit_tennis\tracking.json) {
    Invoke-Checked { & $exe fit results\cases\mixkit_tennis\tracking.json `
        --output results\cases\mixkit_tennis\reconstruction.json }
}
Invoke-Checked { & $exe fit results\cases\generated_diagonal\tracking.json `
    --output results\cases\generated_diagonal\reconstruction.json }
Invoke-Checked { & $exe fit results\cases\generated_drop\tracking.json `
    --output results\cases\generated_drop\reconstruction.json }

Write-Host "=== plots and overlays ==="
if (Test-Path results\cases\mixkit_tennis\tracking.json) {
    Invoke-Checked { & $py vision\plot_reconstruction.py `
        results\cases\mixkit_tennis\tracking.json `
        results\cases\mixkit_tennis\reconstruction.json `
        --output results\cases\mixkit_tennis\reconstruction_preview.png `
        --title "Mixkit tennis bounce" }
    Invoke-Checked { & $py vision\overlay_comparison.py `
        samples\bounce.mp4 `
        results\cases\mixkit_tennis\tracking.json `
        results\cases\mixkit_tennis\reconstruction.json `
        --output results\cases\mixkit_tennis\overlay.mp4 `
        --still docs\demo\mixkit_overlay.png `
        --panel-height 420 `
        --title "Mixkit tennis bounce" }
}
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

$manifest = @{
    date = "2026-08-25"
    notes = "Numbers copied from measured reconstruction.json files and synthetic_fit stdout. Do not edit by hand unless you re-run the pipeline."
    cases = @(
        @{
            id = "synthetic_noise_free"
            title = "C++ synthetic recovery"
            kind = "cpp_synthetic"
            stdout = "results/cases/synthetic/stdout.txt"
            notes = "Noise-free 241-frame two-bounce case from phystwin_synthetic_fit_test. Not video evidence."
        }
    )
}
if (Test-Path results\cases\mixkit_tennis\reconstruction.json) {
    $manifest.cases += @{
        id = "mixkit_tennis"
        title = "Mixkit tennis bounce"
        kind = "recorded_video"
        video = "samples/bounce.mp4"
        point = "375,722"
        source = "https://mixkit.co/free-stock-video/tennis-ball-bouncing-in-slow-motion-101289/"
        notes = "Slow-motion close-up recorded clip. Horizontal velocity is not constant, so the V1 model grades fair."
        tracking = "results/cases/mixkit_tennis/tracking.json"
        tracking_raw = "results/cases/mixkit_tennis/tracking_raw.json"
        reconstruction = "results/cases/mixkit_tennis/reconstruction.json"
        plot = "results/cases/mixkit_tennis/reconstruction_preview.png"
        overlay = "results/cases/mixkit_tennis/overlay.mp4"
        still = "docs/demo/mixkit_overlay.png"
    }
}
$manifest.cases += @{
    id = "generated_diagonal"
    title = "Generated diagonal bounce"
    kind = "generated_video"
    video = "samples/generated_diagonal.mp4"
    point = "80,40"
    notes = "Rendered diagonal bounce (vx=140, vy=20, g=1800, e=0.72). Tracked with SAM 2, then fitted in C++."
    tracking = "results/cases/generated_diagonal/tracking.json"
    tracking_raw = "results/cases/generated_diagonal/tracking_raw.json"
    reconstruction = "results/cases/generated_diagonal/reconstruction.json"
    plot = "results/cases/generated_diagonal/reconstruction_preview.png"
    overlay = "results/cases/generated_diagonal/overlay.mp4"
    gif = "docs/demo/diagonal_overlay.gif"
    still = "docs/demo/diagonal_overlay.png"
}
$manifest.cases += @{
    id = "generated_drop"
    title = "Generated near-vertical drop"
    kind = "generated_video"
    video = "samples/generated_drop.mp4"
    point = "320,36"
    notes = "Rendered near-vertical drop (vx=4, vy=50, g=1600, e=0.40). Different restitution from the diagonal case."
    tracking = "results/cases/generated_drop/tracking.json"
    tracking_raw = "results/cases/generated_drop/tracking_raw.json"
    reconstruction = "results/cases/generated_drop/reconstruction.json"
    plot = "results/cases/generated_drop/reconstruction_preview.png"
    overlay = "results/cases/generated_drop/overlay.mp4"
    gif = "docs/demo/drop_overlay.gif"
    still = "docs/demo/drop_overlay.png"
}

$manifestPath = "results\cases\manifest.json"
($manifest | ConvertTo-Json -Depth 6) | Set-Content -Encoding utf8 $manifestPath
Write-Host "wrote $manifestPath"

Invoke-Checked { & $py vision\collect_evaluation.py --manifest $manifestPath --output docs\evaluation.json }
Invoke-Checked { & $py vision\plot_evaluation.py --manifest $manifestPath --output docs\demo\observed_vs_simulated.png }

Write-Host "Checkpoint 4 eval artifacts are in docs\demo and docs\evaluation.json"
