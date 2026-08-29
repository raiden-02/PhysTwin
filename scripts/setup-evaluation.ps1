# Install the small P3 loader dependency. EMDB and SMPL stay user-managed.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$pip = Join-Path (Get-Location) ".venv\Scripts\pip.exe"
if (-not (Test-Path $python)) {
    throw "missing .venv. Run scripts\setup-vision.ps1 first."
}

Write-Host "EMDB code is MIT at 9a4eab677181a3789bda7ba5c36ab8cff797380c."
Write-Host "EMDB data is restricted to approved non-commercial academic use."
Write-Host "Apply with an institutional email at https://emdb.ait.ethz.ch/."
Write-Host "Download SMPL separately under its registration terms."

& $pip install smplx
if ($LASTEXITCODE -ne 0) {
    throw "smplx install failed"
}

Write-Host ""
Write-Host "Run one approved sequence:"
Write-Host "  .\.venv\Scripts\python.exe vision\evaluate_reconstruction.py `"
Write-Host "    --observation <scene_observation.json> `"
Write-Host "    --emdb-sequence <EMDB_ROOT>\P0\<sequence> `"
Write-Host "    --smpl-model-root <SMPL_MODEL_ROOT> `"
Write-Host "    --accept-emdb-license `"
Write-Host "    --output results\evaluation3d\<sequence>"
