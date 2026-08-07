#!/usr/bin/env bash
# Freezes the Python backend into a standalone onedir bundle via PyInstaller
# (see backend/hearth-backend.spec for exactly what's collected and
# why, plus what's actually been verified vs. still-risky). Single source of
# truth for this invocation — used by both local release builds and CI
# (.github/workflows/build.yml). Windows equivalent: build_backend.ps1.
#
# THIN BUILD: only installs requirements-common.txt — no torch,
# onnxruntime, parler-tts, or ttstokenizer. Neither hardware tier's TTS
# stack gets installed at freeze time anymore; CI has no GPU to match
# either one against, so that decision (and which CUDA/ROCm/DirectML
# variant to use) now happens on the user's own machine at first run
# instead, via backend/app/setup/'s /api/setup/* endpoints. This used to
# take a tier argument and install requirements-gpu.txt/-cpu.txt
# explicitly — removed entirely along with that whole per-tier install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$REPO_ROOT/backend"
RESOURCES_DIR="$REPO_ROOT/desktop/src-tauri/resources"

# Stamp app/VERSION from HEARTH_APP_VERSION or the nearest v* tag so the
# frozen backend reports the same release as the Tauri installer (tag
# v0.3.4 → 0.3.4). MSI/ProductVersion needs a bare numeric triple.
resolve_version() {
  local ver="${HEARTH_APP_VERSION:-}"
  if [ -z "$ver" ]; then
    ver="$(git -C "$REPO_ROOT" describe --tags --match 'v*' --abbrev=0 2>/dev/null || true)"
  fi
  ver="${ver#v}"
  ver="${ver#V}"
  if [ -z "$ver" ]; then
    ver="0.0.0"
  fi
  printf '%s' "$ver"
}

APP_VERSION="$(resolve_version)"
export HEARTH_APP_VERSION="$APP_VERSION"
printf '%s\n' "$APP_VERSION" > "$BACKEND_DIR/app/VERSION"
echo "Stamping backend version $APP_VERSION"

# uv's resolver/installer is a drop-in for pip here (same --only-binary/-r
# flags) and is dramatically faster than pip — --system installs into the
# current interpreter rather than requiring a venv, matching plain
# `pip install`'s behavior.
python3 -m pip install --quiet uv

uv pip install --quiet --system --only-binary=:all: -r "$BACKEND_DIR/requirements-common.txt" pyinstaller

cd "$BACKEND_DIR"
pyinstaller --noconfirm --clean \
  --distpath "$RESOURCES_DIR" \
  --workpath "$REPO_ROOT/.pyinstaller-build" \
  hearth-backend.spec

echo "Frozen backend at: $RESOURCES_DIR/hearth-backend"
