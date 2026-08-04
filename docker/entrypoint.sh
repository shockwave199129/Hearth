#!/usr/bin/env bash
# Container entrypoint: ensure model files exist, then start the backend.
set -euo pipefail

cd /app/backend

mkdir -p /app/backend/data /app/backend/models \
  /root/.local/share/python_keyring

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-48173}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/llama-cpp/llama-server}"
export PYTHON_KEYRING_BACKEND="${PYTHON_KEYRING_BACKEND:-keyrings.alt.file.PlaintextKeyring}"
export LD_LIBRARY_PATH="/opt/llama-cpp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Deps are baked into the image, so /api/setup/start must not try to
# bootstrap the desktop app's bundled standalone Python (absent here).
export HEARTH_SKIP_PACKAGE_INSTALL="${HEARTH_SKIP_PACKAGE_INSTALL:-1}"

if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "ERROR: llama-server not found at ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

# Idempotent: every file already on the models volume is a stat check, so
# this also repairs a partially downloaded tier instead of leaving the UI
# stuck on the Setup screen.
if [[ "${HEARTH_SKIP_MODEL_SETUP:-0}" != "1" ]]; then
  echo "==> Ensuring models for the detected hardware tier…"
  python /app/scripts/setup.py
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU visible in container:"
  nvidia-smi -L || true
else
  echo "==> No nvidia-smi in container — running CPU path"
fi

exec "$@"
