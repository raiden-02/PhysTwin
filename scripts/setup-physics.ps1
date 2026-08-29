# Create the isolated Python 3.11 environment for P4 Newton/Warp simulation.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python311 = "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
if (-not (Test-Path $python311)) {
    $pyList = & py -0p 2>$null
    $match = $pyList | Select-String -Pattern '3\.11'
    if ($match) {
        $python311 = ($match.Line -split '\s+', 2)[-1].Trim()
    }
}
if (-not (Test-Path $python311)) {
    throw "Python 3.11 is required. Install it with: uv python install 3.11"
}

$venv = Join-Path (Get-Location) ".venv-physics"
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating $venv with $python311"
    & $python311 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "physics venv creation failed" }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $python -m pip install "newton==1.5.1" "warp-lang==1.16.0"
if ($LASTEXITCODE -ne 0) { throw "Newton/Warp install failed" }

& $python -c "import newton, warp as wp; wp.init(); d=wp.get_device('cuda:0'); assert d.is_cuda; print('Newton', newton.__version__); print('Warp', wp.__version__); print('device', d.name)"
if ($LASTEXITCODE -ne 0) { throw "Newton/Warp CUDA smoke test failed" }

Write-Host ""
Write-Host "Run the P4 fixture:"
Write-Host "  .\.venv-physics\Scripts\python.exe -m physics3d.simulate_physical_scene ``"
Write-Host "    contracts\3d\v1\examples\physical_scene_tether.json ``"
Write-Host "    --output results\physics3d\p4-tether --repeat-check"
