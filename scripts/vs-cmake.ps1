# Locate CMake/CTest bundled with Visual Studio and optionally prepend them to PATH.
# Regular PowerShell does not have cmake on PATH. Developer PowerShell does.

function Get-VsCMakeBin {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio with the C++ workload."
    }
    $cmake = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find **\cmake.exe |
        Select-Object -First 1
    if (-not $cmake) {
        throw "CMake not found inside Visual Studio. Install the C++ CMake tools component."
    }
    return Split-Path -Parent $cmake
}

function Use-VsCMake {
    $bin = Get-VsCMakeBin
    if ($env:PATH -notlike "*$bin*") {
        $env:PATH = "$bin;$env:PATH"
    }
    Write-Host "Using CMake from $bin"
}
