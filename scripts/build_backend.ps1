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
