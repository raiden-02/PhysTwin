# Install the pinned DA3-BASE reconstruction extra on top of the existing venv.
# Does not reinstall PyTorch or SAM 2.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$pip = Join-Path (Get-Location) ".venv\Scripts\pip.exe"
if (-not (Test-Path $python)) {
    throw "missing .venv. Run scripts\setup-vision.ps1 first."
}

& $pip install einops omegaconf safetensors trimesh imageio e3nn addict plyfile pycolmap "moviepy==1.0.3" evo
& $pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

Write-Host "Pinned DA3 code: 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 (Apache-2.0)"
Write-Host "Pinned weights: depth-anything/DA3-BASE @ f4a6c9b3c95e41c82048423d3493a81ec3fa810e (Apache-2.0)"
& $python -c "from depth_anything_3.api import DepthAnything3; print('depth_anything_3 import ok')"
