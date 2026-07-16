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

# torch's default Linux wheel declares nvidia-cudnn-cu13/nvidia-cusparselt-
# cu13/nvidia-nccl-cu13/nvidia-nvshmem-cu13 as hard dependencies whenever
# platform_system == "Linux" (verified via PyPI metadata) — each of those
# ships hundreds of MB to 1GB+ of CUDA runtime binaries this app never
# uses (no GPU acceleration is bundled anywhere in this project; see
# desktop/src-tauri/README.md's "GPU acceleration isn't bundled" note).
# There's no equivalent platform_system == "Windows"/"Darwin" requirement
# in torch's own metadata, so this is Linux-only — confirmed by the actual
# CI asset sizes: this pushed the Linux GPU .deb well past GitHub's 2GB
# release-asset limit while the equivalent Windows/macOS GPU installers
# stayed under 450MB. Installing the dedicated CPU-only build first keeps
# the subsequent `-r requirements-gpu.txt` install (torch is one of
# parler-tts's own unconstrained transitive deps) from resolving the
# CUDA-pulling default from PyPI instead — verified with a real `uv pip
# install --target ... --index-url https://download.pytorch.org/whl/cpu
# torch`, which resolves to `torch==2.13.0+cpu` with zero nvidia-*/triton
# packages pulled in, vs. the plain-PyPI resolution which pulls all four.
if [[ "$TIER_REQ" == "requirements-gpu.txt" && "$(uname)" == "Linux" ]]; then
  uv pip install --quiet --system --index-url https://download.pytorch.org/whl/cpu torch
fi
# No --only-binary=:all: here (unlike the common install above):
# requirements-gpu.txt's parler-tts and its descript-audiotools-unofficial/
# descript-audio-codec-unofficial dependencies only publish sdists on PyPI
# (no wheels), so they must be built from source — verified via PyPI
# metadata for all three.
uv pip install --quiet --system -r "$BACKEND_DIR/$TIER_REQ" pyinstaller

cd "$BACKEND_DIR"
pyinstaller --noconfirm --clean \
  --distpath "$RESOURCES_DIR" \
  --workpath "$REPO_ROOT/.pyinstaller-build" \
  hearth-backend.spec

echo "Frozen backend at: $RESOURCES_DIR/hearth-backend"
