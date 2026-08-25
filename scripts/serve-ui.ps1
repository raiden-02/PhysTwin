param(
    [switch]$Dev
)

# Start the local PhysTwin UI.
# Default: build the frontend and serve it from FastAPI on http://127.0.0.1:8765
# -Dev: FastAPI on 8765 plus Vite on http://127.0.0.1:5173
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$pip = Join-Path (Get-Location) ".venv\Scripts\pip.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv. Run scripts\setup-vision.ps1 first."
}

& $python -c "import fastapi, uvicorn, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing FastAPI, uvicorn, python-multipart into .venv"
    & $pip install fastapi uvicorn python-multipart
}

$exe = Join-Path (Get-Location) "build\Release\phystwin.exe"
if (-not (Test-Path $exe)) {
    throw "Missing build\Release\phystwin.exe. Run scripts\build.ps1 first."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is required for the UI. Install Node.js."
}

Push-Location frontend
try {
    if (-not (Test-Path "node_modules")) {
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }
    if (-not $Dev) {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
    }
}
finally {
    Pop-Location
}

if ($Dev) {
    Write-Host "API  http://127.0.0.1:8765"
    Write-Host "UI   http://127.0.0.1:5173"
    $api = Start-Process -FilePath $python -ArgumentList @("vision\serve.py") -PassThru -NoNewWindow
    try {
        Push-Location frontend
        npm run dev
    }
    finally {
        Pop-Location
        if (-not $api.HasExited) {
            Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
else {
    Write-Host "PhysTwin UI  http://127.0.0.1:8765"
    & $python vision\serve.py
}
