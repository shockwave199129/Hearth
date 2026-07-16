"""First-run hardware-aware setup: detects the actual machine's GPU vendor
(not just its VRAM, which app/hardware/detect.py already covers) and
installs the matching torch/onnxruntime build, then downloads models. See
project root's setup plan for why this exists — CI has no GPU, so every
hardware-specific package decision has to happen on the real machine, not
at build time."""
