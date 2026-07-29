#!/usr/bin/env python3
"""Preflight the training environment: interpreter, deps, torch/CUDA, GPU, bf16.

Worth running before a multi-hour job. Three things bite here:

* The wrong interpreter. Everything is pip-installed into ``.venv``, so a run
  launched with the system Python finds no torch at all.
* RTX 50-series is Blackwell (sm_120). A pre-CUDA-12.8 wheel has no kernel for
  it and either falls back to CPU or dies mid-run with "no kernel image is
  available for execution on the device".
* A dep that is only needed by a later stage (onnx for export) being absent,
  which otherwise surfaces an hour in.

Usage::

    python scripts/check_gpu.py
    python scripts/check_gpu.py --require-cuda     # non-zero exit if no GPU
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

CU128_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cu128"

# (import name, pip target, stage it gates, fatal when missing)
REQUIREMENTS = [
    ("torch", "torch", "training", True),
    ("tokenizers", "-r backend/requirements-common.txt", "shared tokenizer", True),
    ("datasets", "-r hearth_ai/data/prepare/requirements.txt", "HF prepare", False),
    ("onnx", "-r hearth_ai/requirements-export.txt", "ONNX export", False),
    ("onnxruntime", "-r hearth_ai/requirements-export.txt", "export parity + golden eval", False),
]


def report_interpreter() -> list[str]:
    """Print interpreter details; return warnings about running outside a venv."""
    warnings: list[str] = []

    print(f"python   {sys.version.split()[0]}")
    print(f"exe      {sys.executable}")

    in_venv = sys.prefix != sys.base_prefix
    active = os.environ.get("VIRTUAL_ENV")
    if in_venv:
        print(f"venv     {sys.prefix}")
        # An activated venv that isn't the one running us means PATH and the
        # interpreter disagree, and pip installs would land somewhere unused.
        if active and Path(active).resolve() != Path(sys.prefix).resolve():
            warnings.append(f"VIRTUAL_ENV is {active} but running from {sys.prefix}")
    else:
        print("venv     NONE - using the system interpreter")
        warnings.append(
            "not running inside .venv, where the deps are installed. "
            "Activate it (.venv\\Scripts\\Activate.ps1) or set HEARTH_PYTHON"
        )

    return warnings


def report_packages() -> tuple[list[str], bool]:
    """Print dep availability; return (warnings, any_fatal_missing)."""
    warnings: list[str] = []
    fatal = False

    for module, pip_target, stage, is_fatal in REQUIREMENTS:
        # find_spec avoids importing heavy packages just to test presence.
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False

        if found:
            print(f"dep      {module:<12} ok")
            continue

        print(f"dep      {module:<12} MISSING  ({stage})")
        warnings.append(f"{module} missing, needed for {stage}: pip install {pip_target}")
        fatal = fatal or is_fatal

    return warnings, fatal


def report_torch(require_cuda: bool) -> tuple[list[str], int]:
    """Print torch/CUDA/GPU details; return (warnings, exit_code)."""
    import torch

    warnings: list[str] = []

    cuda_build = torch.version.cuda or "none (CPU-only build)"
    print(f"torch    {torch.__version__}  cuda build {cuda_build}")

    if not torch.cuda.is_available():
        print("cuda     NOT AVAILABLE - training would run on CPU")
        warnings.append(f"no usable CUDA device. {CU128_HINT}")
        return warnings, 1 if require_cuda else 0

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"gpu      {name}  sm_{major}{minor}  {total_gb:.1f} GiB")
    print(f"bf16     {'yes' if torch.cuda.is_bf16_supported() else 'no (will use fp16)'}")

    try:
        build = tuple(int(p) for p in str(torch.version.cuda).split(".")[:2])
    except (ValueError, AttributeError):
        build = (0, 0)
    if major >= 12 and build < (12, 8):
        warnings.append(
            f"sm_{major}{minor} GPU on a CUDA {cuda_build} torch build - "
            f"expect 'no kernel image' errors. Reinstall: {CU128_HINT}"
        )

    if total_gb < 9:
        warnings.append(
            f"{total_gb:.1f} GiB VRAM - if training OOMs, use "
            "--batch-size 16 --grad-accum 2 (same effective batch)"
        )

    # A real kernel launch is the only proof the build matches the device;
    # is_available() returns True on a wheel with no kernels for this card.
    try:
        probe = torch.ones(64, 64, device="cuda")
        _ = (probe @ probe).sum().item()
        print("kernel   launch OK")
    except Exception as exc:  # noqa: BLE001 - report any driver/ABI failure verbatim
        print(f"kernel   LAUNCH FAILED: {exc}")
        warnings.append(f"CUDA kernel launch failed. {CU128_HINT}")
        return warnings, 1

    return warnings, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero when no usable CUDA device is present",
    )
    args = parser.parse_args()

    warnings = report_interpreter()

    package_warnings, missing_fatal = report_packages()
    warnings += package_warnings

    exit_code = 0
    if missing_fatal:
        exit_code = 1
    else:
        torch_warnings, exit_code = report_torch(args.require_cuda)
        warnings += torch_warnings

    for warning in warnings:
        print(f"WARNING  {warning}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
