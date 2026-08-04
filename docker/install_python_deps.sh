#!/usr/bin/env bash
# Installs backend Python deps inside the image build.
# HEARTH_DEVICE=cpu|gpu  TORCH_CUDA_INDEX=cu126 (gpu only)
set -euo pipefail

DEVICE="${HEARTH_DEVICE:-cpu}"
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu126}"

# Isolate from distro Python (Ubuntu 24.04 is PEP 668 externally managed).
python -m venv /opt/venv
# shellcheck disable=SC1091
source /opt/venv/bin/activate
export PATH="/opt/venv/bin:${PATH}"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir "keyrings.alt>=5.0"

if [[ "${DEVICE}" == "gpu" ]]; then
  python -m pip install --no-cache-dir \
    torch torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
  # requirements-gpu.txt includes -r common; install numpy pin + parler next.
  python -m pip install --no-cache-dir -r /app/backend/requirements-gpu.txt
  # onnxruntime-gpu helps Moonshine / NLP when CUDA is present. Also pull
  # ttstokenizer so a GPU image can still fall back to Kokoro if the
  # container is started without --gpus (tier B/C).
  python -m pip install --no-cache-dir onnxruntime-gpu ttstokenizer || \
    python -m pip install --no-cache-dir onnxruntime ttstokenizer
else
  python -m pip install --no-cache-dir -r /app/backend/requirements-cpu.txt
fi

# Kokoro's ttstokenizer reaches for NLTK corpora during synthesis, which would
# otherwise be an unannounced network fetch on the first spoken reply. NLTK
# >=3.9's pos_tag loads the *_eng tagger while ttstokenizer's own presence
# check still looks for the legacy name, so both have to be present.
python -m nltk.downloader -d /usr/local/share/nltk_data \
  averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict
