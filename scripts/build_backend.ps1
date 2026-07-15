# Windows equivalent of build_backend.sh — see that file and
# backend/hearth-backend.spec for what this actually does and
# what's verified vs. still-risky. Single source of truth used by both
# local release builds and CI (.github/workflows/build.yml).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RepoRoot "backend"
$ResourcesDir = Join-Path $RepoRoot "desktop\src-tauri\resources"

# requirements-common.txt first so detect_tier_requirements.py has psutil
# (and everything else it transitively needs) before it runs.
pip install --quiet --only-binary=:all: -r "$BackendDir\requirements-common.txt"
$TierReq = (python "$ScriptDir\detect_tier_requirements.py").Trim()
Write-Host "Detected tier requirements: $TierReq"
pip install --quiet --only-binary=:all: -r "$BackendDir\$TierReq" pyinstaller

Push-Location $BackendDir
try {
    pyinstaller --noconfirm --clean `
        --distpath "$ResourcesDir" `
        --workpath "$RepoRoot\.pyinstaller-build" `
        hearth-backend.spec
} finally {
    Pop-Location
}

Write-Host "Frozen backend at: $ResourcesDir\hearth-backend"
