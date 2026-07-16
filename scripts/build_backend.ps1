# Windows equivalent of build_backend.sh — see that file and
# backend/hearth-backend.spec for what this actually does and
# what's verified vs. still-risky. Single source of truth used by both
# local release builds and CI (.github/workflows/build.yml).
#
# Takes an optional -Tier argument ("gpu" or "cpu") selecting
# requirements-gpu.txt/-cpu.txt directly — see build_backend.sh's matching
# comment for why CI always passes this explicitly instead of
# autodetecting from the build machine's hardware. Defaults to "cpu" when
# omitted (e.g. a local run), since a GPU-tier Windows build additionally
# needs a CUDA-enabled torch wheel that plain `pip install` won't select.
param(
    [ValidateSet("gpu", "cpu")]
    [string]$Tier = "cpu"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RepoRoot "backend"
$ResourcesDir = Join-Path $RepoRoot "desktop\src-tauri\resources"

# uv's resolver/installer is a drop-in for pip here (same --only-binary/-r
# flags) and is dramatically faster for this repo's heavier tier stacks
# (torch, transformers) — --system installs into the current interpreter
# rather than requiring a venv, matching plain `pip install`'s behavior.
python -m pip install --quiet uv

uv pip install --quiet --system --only-binary=:all: -r "$BackendDir\requirements-common.txt"
$TierReq = "requirements-$Tier.txt"
Write-Host "Using tier requirements: $TierReq"
# No --only-binary=:all: here (unlike the common install above):
# requirements-gpu.txt installs parler-tts from a git ref, which has no
# wheel and must be built from source.
uv pip install --quiet --system -r "$BackendDir\$TierReq"
# Ensure PyInstaller is available for the subsequent freeze step.
uv pip install --quiet --system pyinstaller

Push-Location $BackendDir
try {
    python -m PyInstaller --noconfirm --clean `
        --distpath "$ResourcesDir" `
        --workpath "$RepoRoot\.pyinstaller-build" `
        hearth-backend.spec
} finally {
    Pop-Location
}

Write-Host "Frozen backend at: $ResourcesDir\hearth-backend"
