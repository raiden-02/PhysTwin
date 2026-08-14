# Create the Python 3.11 venv and install CUDA PyTorch + SAM 2.
# SAM 2's optional CUDA post-process kernel needs nvcc. This machine has no nvcc,
# so we set SAM2_BUILD_CUDA=0. Video tracking still runs on GPU through PyTorch.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python311 = $null
$candidates = @(
    "$env:USERPROFILE\.local\bin\python3.11.exe",
    "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
)
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $python311 = $candidate
        break
    }
}
if (-not $python311) {
    $pyList = & py -0p 2>$null
    $astral = $pyList | Select-String -Pattern '3\.11'
    if ($astral) {
        $python311 = ($astral.Line -split '\s+', 2)[-1].Trim()
    }
}
if (-not $python311) {
    throw "Python 3.11 not found. Install 3.11 (uv/astral python3.11.exe is expected on this machine)."
}

$venv = Join-Path (Get-Location) ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating Python 3.11 venv at $venv using $python311"
    if (Test-Path $venv) {
        Remove-Item -Recurse -Force $venv
    }
    & $python311 -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

& $python -m pip install --upgrade pip
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $pip install numpy opencv-python matplotlib tqdm pillow huggingface_hub

$env:SAM2_BUILD_CUDA = "0"
& $pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"

Write-Host "Vision environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
& $python -c "import torch; print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('torch', torch.__version__)"
