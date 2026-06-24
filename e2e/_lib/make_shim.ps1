# Create a noop executable shim at $Dir\$Name.cmd for use in PATH during
# native (non-Docker) e2e tests on Windows. Mirrors e2e/_lib/shims.py
# make_shim(noop).
#
# Usage: make_shim.ps1 -Name <name> -Dir <dir>

param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Dir
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dir)) {
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
}

$path = Join-Path $Dir "$Name.cmd"
$content = @"
@echo off
exit /b 0
"@
Set-Content -Path $path -Value $content -Encoding ASCII -NoNewline
Write-Output $path
