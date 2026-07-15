#!/usr/bin/env bash
# Freezes the Python backend into a standalone onedir bundle via PyInstaller
# (see backend/hearth-backend.spec for exactly what's collected and
# why, plus what's actually been verified vs. still-risky). Single source of
# truth for this invocation — used by both local release builds and CI
# (.github/workflows/build.yml). Windows equivalent: build_backend.ps1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$REPO_ROOT/backend"
RESOURCES_DIR="$REPO_ROOT/desktop/src-tauri/resources"

# requirements-common.txt first so detect_tier_requirements.py has psutil
# (and everything else it transitively needs) before it runs.
pip install --quiet --only-binary=:all: -r "$BACKEND_DIR/requirements-common.txt"
TIER_REQ="$(python3 "$SCRIPT_DIR/detect_tier_requirements.py")"
echo "Detected tier requirements: $TIER_REQ"
pip install --quiet --only-binary=:all: -r "$BACKEND_DIR/$TIER_REQ" pyinstaller

cd "$BACKEND_DIR"
pyinstaller --noconfirm --clean \
  --distpath "$RESOURCES_DIR" \
  --workpath "$REPO_ROOT/.pyinstaller-build" \
  hearth-backend.spec

echo "Frozen backend at: $RESOURCES_DIR/hearth-backend"
