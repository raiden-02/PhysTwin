# Run CTest from a regular PowerShell. CMake is not on PATH unless vs-cmake is sourced.
param(
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\vs-cmake.ps1"
Use-VsCMake

if (-not (Test-Path "build")) {
    throw "missing build/. Run scripts\build.ps1 first."
}

ctest --test-dir build -C $Config --output-on-failure
exit $LASTEXITCODE
