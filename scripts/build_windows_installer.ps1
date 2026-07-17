# Local Windows installer build — mirrors .github/workflows/build.yml's
# Windows job so you can git pull on a Windows machine, produce MSI/NSIS
# without tagging, and test Setup → Chat there.
#
# Usage (from repo root, in Windows PowerShell or PowerShell 7):
#   .\scripts\build_windows_installer.ps1
#   .\scripts\build_windows_installer.ps1 -Version 0.2.11
#
# If scripts are blocked by execution policy:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installer.ps1
#
# Note: `pwsh` is PowerShell 7+ only. Built-in Windows PowerShell is
# `powershell` — you do not need `pwsh` for this script.
# Prerequisites: Python 3.12+, Node/pnpm, Rust stable + VS Build Tools
# (C++ workload). WiX for MSI is fetched by Tauri on first build.
# See desktop/src-tauri/README.md.
param(
    [string]$Version = "0.0.0"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $RepoRoot "frontend"
$DesktopDir = Join-Path $RepoRoot "desktop"
$BundleDir = Join-Path $RepoRoot "desktop\src-tauri\target\release\bundle"

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

Write-Host "==> Checking prerequisites"
Assert-Command python
Assert-Command pnpm
Assert-Command rustc
Assert-Command cargo

$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVer -lt [version]"3.12") {
    throw "Python 3.12+ required (found $pyVer)"
}

if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Version must be numeric major.minor.build (MSI ProductVersion); got: $Version"
}

Write-Host "==> Install frontend dependencies"
Push-Location $FrontendDir
try {
    pnpm install --frozen-lockfile
} finally {
    Pop-Location
}

Write-Host "==> Install desktop (Tauri CLI) dependencies"
Push-Location $DesktopDir
try {
    pnpm install --frozen-lockfile
} finally {
    Pop-Location
}

Write-Host "==> Freeze backend (thin build — same as CI)"
& (Join-Path $ScriptDir "build_backend.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "build_backend.ps1 failed with exit code $LASTEXITCODE"
}

Write-Host "==> Fetch bundled llama-server + setup-python"
python -m pip install --quiet requests
Push-Location $RepoRoot
try {
    python scripts/fetch_llama_cpp.py
    if ($LASTEXITCODE -ne 0) { throw "fetch_llama_cpp.py failed" }
    python scripts/fetch_setup_python.py
    if ($LASTEXITCODE -ne 0) { throw "fetch_setup_python.py failed" }
} finally {
    Pop-Location
}

Write-Host "==> Tauri build (version $Version)"
Push-Location $DesktopDir
try {
    # Match CI: pass version via --config so tauri.conf.json need not be edited.
    # JSON must be a single argument with no spaces around the colon/value.
    $configArg = "{`"version`":`"$Version`"}"
    pnpm exec tauri build --config $configArg
    if ($LASTEXITCODE -ne 0) {
        throw "tauri build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Build complete. Installers:"
$msi = Join-Path $BundleDir "msi"
$nsis = Join-Path $BundleDir "nsis"
if (Test-Path $msi) {
    Get-ChildItem $msi -Filter *.msi | ForEach-Object { Write-Host "  MSI:  $($_.FullName)" }
} else {
    Write-Host "  (no MSI dir yet — check Tauri bundle output above)"
}
if (Test-Path $nsis) {
    Get-ChildItem $nsis -Filter *.exe | ForEach-Object { Write-Host "  NSIS: $($_.FullName)" }
} else {
    Write-Host "  (no NSIS dir yet — check Tauri bundle output above)"
}
Write-Host ""
Write-Host "Install one of the above, then exercise Setup → Chat."
Write-Host "Unsigned builds may trigger SmartScreen; use More info → Run anyway on this machine."
