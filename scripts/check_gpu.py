#!/usr/bin/env python3
"""Preflight the training environment: interpreter, driver, deps, torch/CUDA, bf16.

Worth running before a multi-hour job. Four things bite here:

* The wrong interpreter. Everything is pip-installed into ``.venv``, so a run
  launched with the system Python finds no torch at all.
* A torch wheel too old for the card. RTX 50-series is Blackwell (sm_120), which
  needs CUDA 12.8+; older wheels have no kernel for it and either fall back to
  CPU or die mid-run with "no kernel image is available for execution".
* A torch wheel too new for the driver, which fails to load the CUDA runtime.
* A dep only needed by a later stage (onnx for export) being absent, which
  otherwise surfaces an hour in.

Usage::

    python scripts/check_gpu.py
    python scripts/check_gpu.py --require-cuda     # non-zero exit if no GPU
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/"

# Published CUDA wheel variants, most capable first. Picking one means matching
# the driver: a wheel's CUDA major must not exceed what the driver supports.
WHEEL_VARIANTS = ((13, 2, "cu132"), (13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"))

# Blackwell (sm_120) kernels first shipped in CUDA 12.8.
BLACKWELL_MIN_CUDA = (12, 8)

# (import name, pip target, stage it gates, fatal when missing)
REQUIREMENTS = [
    ("torch", None, "training", True),
    ("tokenizers", "-r backend/requirements-common.txt", "shared tokenizer", True),
    ("datasets", "-r hearth_ai/data/prepare/requirements.txt", "HF prepare", False),
    ("onnx", "-r hearth_ai/requirements-export.txt", "ONNX export", False),
    ("onnxruntime", "-r hearth_ai/requirements-export.txt", "export parity + golden eval", False),
]


def format_cuda(version: tuple[int, int] | None) -> str:
    return f"{version[0]}.{version[1]}" if version else "unknown"


def recommend_wheel(driver_cuda: tuple[int, int] | None) -> str | None:
    """Newest wheel variant the driver can actually load, or None if too old."""
    if driver_cuda is None:
        # Unknown driver: cu128 is the broadest build that still covers sm_120.
        return "cu128"
    for major, minor, name in WHEEL_VARIANTS:
        if driver_cuda >= (major, minor):
            return name
    return None


def install_hint(driver_cuda: tuple[int, int] | None) -> str:
    variant = recommend_wheel(driver_cuda)
    if variant is None:
        return (
            f"driver supports only CUDA {format_cuda(driver_cuda)}; Blackwell needs "
            f"{format_cuda(BLACKWELL_MIN_CUDA)}+ - update the NVIDIA driver first"
        )
    return f"pip install torch --index-url {TORCH_INDEX}{variant}"


def probe_nvidia_smi() -> tuple[str | None, tuple[int, int] | None]:
    """Return (driver version, max CUDA the driver supports) from nvidia-smi.

    Read before importing torch so a missing-torch environment can still be told
    which wheel to install.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None, None

    driver = None
    if match := re.search(r"Driver Version:\s*([\d.]+)", out):
        driver = match.group(1)

    cuda = None
    if match := re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out):
        cuda = (int(match.group(1)), int(match.group(2)))

    return driver, cuda


def report_interpreter() -> list[str]:
    """Print interpreter details; return warnings about running outside a venv."""
    warnings: list[str] = []

    print(f"python   {sys.version.split()[0]}")
    print(f"exe      {sys.executable}")

    active = os.environ.get("VIRTUAL_ENV")
    if sys.prefix != sys.base_prefix:
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


def report_driver(driver: str | None, driver_cuda: tuple[int, int] | None) -> list[str]:
    """Print driver details; return warnings if it can't support this GPU."""
    if driver is None and driver_cuda is None:
        print("driver   nvidia-smi not found - no NVIDIA driver visible")
        return ["nvidia-smi not found; a CUDA run needs the NVIDIA driver installed"]

    print(f"driver   {driver or 'unknown'}  supports CUDA up to {format_cuda(driver_cuda)}")

    if driver_cuda and driver_cuda < BLACKWELL_MIN_CUDA:
        return [
            f"driver caps at CUDA {format_cuda(driver_cuda)}, below the "
            f"{format_cuda(BLACKWELL_MIN_CUDA)} Blackwell needs - update the driver"
        ]
    return []


def report_packages(driver_cuda: tuple[int, int] | None) -> tuple[list[str], bool]:
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
        fix = install_hint(driver_cuda) if pip_target is None else f"pip install {pip_target}"
        warnings.append(f"{module} missing, needed for {stage}: {fix}")
        fatal = fatal or is_fatal

    return warnings, fatal


def report_torch(
    driver_cuda: tuple[int, int] | None, require_cuda: bool
) -> tuple[list[str], int]:
    """Print torch/CUDA/GPU details; return (warnings, exit_code)."""
    import torch

    warnings: list[str] = []
    hint = install_hint(driver_cuda)

    try:
        wheel_cuda: tuple[int, int] | None = tuple(  # type: ignore[assignment]
            int(part) for part in str(torch.version.cuda).split(".")[:2]
        )
    except (ValueError, AttributeError):
        wheel_cuda = None

    build = format_cuda(wheel_cuda) if wheel_cuda else "none (CPU-only build)"
    print(f"torch    {torch.__version__}  cuda build {build}")

    if wheel_cuda is None:
        print("cuda     CPU-ONLY BUILD - training would run on CPU")
        warnings.append(f"torch has no CUDA support. {hint}")
        return warnings, 1 if require_cuda else 0

    # CUDA guarantees minor-version compatibility, so only a newer *major* in the
    # wheel than the driver is a hard mismatch.
    if driver_cuda and wheel_cuda[0] > driver_cuda[0]:
        warnings.append(
            f"wheel needs CUDA {build} but the driver supports only "
            f"{format_cuda(driver_cuda)} - the runtime will fail to load. {hint}"
        )
    elif driver_cuda and wheel_cuda > driver_cuda:
        warnings.append(
            f"wheel CUDA {build} is newer than the driver's "
            f"{format_cuda(driver_cuda)}; usually fine via minor-version compatibility"
        )

    if wheel_cuda < BLACKWELL_MIN_CUDA:
        warnings.append(
            f"CUDA {build} wheel has no Blackwell (sm_120) kernels - expect "
            f"'no kernel image' errors. {hint}"
        )

    if not torch.cuda.is_available():
        print("cuda     NOT AVAILABLE - training would run on CPU")
        warnings.append(f"no usable CUDA device. {hint}")
        return warnings, 1 if require_cuda else 0

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"gpu      {name}  sm_{major}{minor}  {total_gb:.1f} GiB")
    print(f"bf16     {'yes' if torch.cuda.is_bf16_supported() else 'no (will use fp16)'}")

    # Point at a newer variant only when the current one is actually usable;
    # otherwise the mismatch warning below already carries the right command.
    best = recommend_wheel(driver_cuda)
    usable = not driver_cuda or wheel_cuda <= driver_cuda
    if usable and best and best != f"cu{wheel_cuda[0]}{wheel_cuda[1]}":
        print(f"note     driver also supports {best} wheels ({TORCH_INDEX}{best})")

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
        warnings.append(f"CUDA kernel launch failed. {hint}")
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

    driver, driver_cuda = probe_nvidia_smi()

    warnings = report_interpreter()
    warnings += report_driver(driver, driver_cuda)

    package_warnings, missing_fatal = report_packages(driver_cuda)
    warnings += package_warnings

    if missing_fatal:
        exit_code = 1
    else:
        torch_warnings, exit_code = report_torch(driver_cuda, args.require_cuda)
        warnings += torch_warnings

    for warning in warnings:
        print(f"WARNING  {warning}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
