#!/usr/bin/env bash
# Start Hearth via Docker Compose, attaching the GPU override when possible.
#
# Usage:
#   ./scripts/docker-up.sh              # auto: GPU if nvidia + toolkit available
#   ./scripts/docker-up.sh --cpu        # force CPU image
#   ./scripts/docker-up.sh --gpu        # force GPU image (fails if no GPU)
#   ./scripts/docker-up.sh --build      # pass --build through to compose
#   ./scripts/docker-up.sh --detach     # run in background
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE=auto
COMPOSE_ARGS=()

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE=cpu; shift ;;
    --gpu) MODE=gpu; shift ;;
    --auto) MODE=auto; shift ;;
    --build|--detach|-d|--force-recreate|--no-deps)
      COMPOSE_ARGS+=("$1"); shift ;;
    --help|-h) usage 0 ;;
    *)
      COMPOSE_ARGS+=("$1"); shift ;;
  esac
done

have_nvidia_host() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

have_nvidia_docker() {
  # Probe the toolkit without pulling a large image when possible.
  if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    return 0
  fi
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L >/dev/null 2>&1
}

FILES=(-f docker-compose.yml)
LABEL=cpu

case "${MODE}" in
  cpu)
    LABEL=cpu
    ;;
  gpu)
    if ! have_nvidia_host; then
      echo "ERROR: --gpu requested but nvidia-smi failed on the host." >&2
      exit 1
    fi
    if ! have_nvidia_docker; then
      echo "ERROR: NVIDIA Container Toolkit not usable (docker --gpus all failed)." >&2
      echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
      exit 1
    fi
    FILES+=(-f docker-compose.gpu.yml)
    LABEL=gpu
    ;;
  auto)
    if have_nvidia_host && have_nvidia_docker; then
      FILES+=(-f docker-compose.gpu.yml)
      LABEL=gpu
      echo "==> NVIDIA GPU detected — using GPU backend image"
    else
      LABEL=cpu
      if have_nvidia_host; then
        echo "==> NVIDIA GPU on host, but Docker cannot access it — falling back to CPU"
        echo "    Install/enable the NVIDIA Container Toolkit to use the GPU image."
      else
        echo "==> No NVIDIA GPU detected — using CPU backend image"
      fi
    fi
    ;;
esac

echo "==> docker compose ${FILES[*]} up ${COMPOSE_ARGS[*]:-}"
echo "    UI:  http://localhost:48176"
echo "    API: http://localhost:48173  (mode=${LABEL})"
exec docker compose "${FILES[@]}" up "${COMPOSE_ARGS[@]}"
