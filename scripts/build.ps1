# Configure, build, and test from a regular PowerShell (no Developer Prompt required).
param(
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\vs-cmake.ps1"
Use-VsCMake

cmake -S . -B build -G "Visual Studio 18 2026" -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build build --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

ctest --test-dir build -C $Config --output-on-failure
exit $LASTEXITCODE
