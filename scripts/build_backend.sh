#!/usr/bin/env bash
# Freezes the Python backend into a standalone onedir bundle via PyInstaller
# (see backend/hearth-backend.spec for exactly what's collected and
# why, plus what's actually been verified vs. still-risky). Single source of
# truth for this invocation — used by both local release builds and CI
# (.github/workflows/build.yml). Windows equivalent: build_backend.ps1.
#
# Takes an optional tier argument ("gpu" or "cpu") selecting
# requirements-gpu.txt/-cpu.txt directly. CI always passes this explicitly
# (build.yml's matrix builds one installer per tier per OS) rather than
# relying on detect_tier_requirements.py's autodetection, since that
# detects the BUILD machine's hardware, not the end user's — fine for a
# local one-off build, wrong for a CI runner producing a shipped installer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$REPO_ROOT/backend"
RESOURCES_DIR="$REPO_ROOT/desktop/src-tauri/resources"

TIER="${1:-}"

# uv's resolver/installer is a drop-in for pip here (same --only-binary/-r
# flags) and is dramatically faster for this repo's heavier tier stacks
# (torch, transformers) — --system installs into the current interpreter
# rather than requiring a venv, matching plain `pip install`'s behavior.
python3 -m pip install --quiet uv

# requirements-common.txt first so detect_tier_requirements.py has psutil
# (and everything else it transitively needs) before it runs.
uv pip install --quiet --system --only-binary=:all: -r "$BACKEND_DIR/requirements-common.txt"
case "$TIER" in
  gpu) TIER_REQ="requirements-gpu.txt" ;;
  cpu) TIER_REQ="requirements-cpu.txt" ;;
  "") TIER_REQ="$(python3 "$SCRIPT_DIR/detect_tier_requirements.py")" ;;
  *) echo "Unknown tier '$TIER' — expected 'gpu' or 'cpu'" >&2; exit 1 ;;
esac
echo "Using tier requirements: $TIER_REQ"
# No --only-binary=:all: here (unlike the common install above):
# requirements-gpu.txt installs parler-tts from a git ref, which has no
# wheel and must be built from source.
uv pip install --quiet --system -r "$BACKEND_DIR/$TIER_REQ" pyinstaller

cd "$BACKEND_DIR"
pyinstaller --noconfirm --clean \
  --distpath "$RESOURCES_DIR" \
  --workpath "$REPO_ROOT/.pyinstaller-build" \
  hearth-backend.spec

echo "Frozen backend at: $RESOURCES_DIR/hearth-backend"
