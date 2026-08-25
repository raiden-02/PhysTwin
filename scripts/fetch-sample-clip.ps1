# Optional helper: download a Mixkit basketball clip if you want real footage.
# Videos stay gitignored. Prefer a phone recording of one bouncing object.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = Join-Path (Get-Location) "samples\bounce.mp4"
if (Test-Path $out) {
    Write-Host "already exists: $out"
    exit 0
}
$url = "https://assets.mixkit.co/videos/2272/2272-720.mp4"
Write-Host "downloading $url"
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
Get-Item $out | Select-Object FullName, Length
