# Windows equivalent of build_backend.sh — see that file and
# backend/hearth-backend.spec for what this actually does and
# what's verified vs. still-risky. Single source of truth used by both
# local release builds and CI (.github/workflows/build.yml).
#
# THIN BUILD: only installs requirements-common.txt — see build_backend.sh's
# matching comment for why (no tier-specific TTS stack is installed at
# freeze time anymore; that now happens at first run on the user's own
# machine instead).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RepoRoot "backend"
$ResourcesDir = Join-Path $RepoRoot "desktop\src-tauri\resources"

# Stamp app/VERSION from HEARTH_APP_VERSION or the nearest v* tag so the
# frozen backend matches the Tauri installer (v0.3.4 → 0.3.4).
function Resolve-AppVersion {
    $ver = $env:HEARTH_APP_VERSION
    if (-not $ver) {
        $tag = git -C $RepoRoot describe --tags --match "v*" --abbrev=0 2>$null
        if ($LASTEXITCODE -eq 0 -and $tag) { $ver = $tag }
    }
    if ($ver) {
        $ver = $ver.Trim()
        if ($ver.StartsWith("v") -or $ver.StartsWith("V")) {
            $ver = $ver.Substring(1)
        }
    }
    if (-not $ver) { $ver = "0.0.0" }
    return $ver
}

$AppVersion = Resolve-AppVersion
$env:HEARTH_APP_VERSION = $AppVersion
Set-Content -Path (Join-Path $BackendDir "app\VERSION") -Value $AppVersion -NoNewline
Write-Host "Stamping backend version $AppVersion"

python -m pip install --quiet uv

uv pip install --quiet --system --only-binary=:all: -r "$BackendDir\requirements-common.txt"
uv pip install --quiet --system pyinstaller

Push-Location $BackendDir
try {
    python -m PyInstaller --noconfirm --clean `
        --distpath "$ResourcesDir" `
        --workpath "$RepoRoot\.pyinstaller-build" `
        hearth-backend.spec
    # $ErrorActionPreference only turns PowerShell-native terminating errors
    # into stops — it does NOT apply to a non-zero exit code from an external
    # command like PyInstaller, so a real freeze failure here would otherwise
    # print "Frozen backend" and let CI proceed straight into a confusing
    # "resource path doesn't exist" error at the Tauri build step instead.
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Frozen backend at: $ResourcesDir\hearth-backend"
