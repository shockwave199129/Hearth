"""Detects GPU vendor + OS, then picks the matching `torch`/`onnxruntime`
install source. CI has no GPU, so this has to run on the actual machine —
see backend/app/setup/installer.py for where these get used.

Confidence note (see the setup plan): the torch/CUDA-index selection below
uses the same mechanism already verified against real installs this
session (CPU-only torch to dodge unwanted CUDA deps, torchaudio ABI
pairing). The onnxruntime-gpu/-rocm/-directml package choices are backed
by confirming those packages exist on PyPI, but have NOT been test-
installed or run — flagged explicitly rather than implied equally solid.
"""
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.hardware.detect import _detect_nvidia_gpu, _detect_rocm_gpu

# Used only if the live index query itself fails (offline, index down) —
# better than crashing setup outright. Update periodically; staleness here
# just means a slightly older (but still valid) CUDA/ROCm build gets used.
_FALLBACK_CUDA_INDEX = "cu126"
_FALLBACK_ROCM_INDEX = "rocm6.2"


@dataclass(frozen=True)
class HardwareVariant:
    vendor: str  # "nvidia" | "amd" | "none"
    gpu_name: str | None
    os_name: str  # "Windows" | "Linux" | "Darwin"
    torch_index_url: str | None  # None means plain PyPI (no --index-url needed)
    onnxruntime_package: str


def _detect_vendor() -> tuple[str, str | None]:
    nvidia = _detect_nvidia_gpu()
    if nvidia:
        return "nvidia", nvidia["name"]
    amd = _detect_rocm_gpu()
    if amd:
        return "amd", amd["name"]
    return "none", None


def _nvidia_driver_cuda_version() -> str | None:
    """Parses the "CUDA Version: X.Y" field nvidia-smi prints in its own
    header table — the max CUDA runtime version this driver supports, not
    necessarily what's installed. Returns None if nvidia-smi is missing or
    its output doesn't match (never block setup on a parsing surprise)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return match.group(1) if match else None


def _best_available_index(kind: str, driver_version: str | None, fallback: str) -> str:
    """Queries download.pytorch.org/whl/torch/'s index page for every
    published cuXXX or rocmX.Y variant, and picks the highest one that
    doesn't exceed what the driver reports supporting. `kind` is "cu" or
    "rocm". Falls back to a hardcoded last-known-good index on any network
    failure, or if the driver version couldn't be determined."""
    try:
        with urllib.request.urlopen("https://download.pytorch.org/whl/torch/", timeout=10) as r:
            html = r.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError):
        return fallback

    def _version_tuple(index: str) -> tuple[int, ...]:
        raw = index[len("cu") :] if kind == "cu" else index[len("rocm") :]
        # cuXXX has no dots (e.g. "cu126" -> 12.6) — reinsert the decimal
        # point one digit from the end to compare against driver_version's
        # own X.Y form; rocmX.Y already has dots.
        if kind == "cu" and "." not in raw:
            raw = f"{raw[:-1]}.{raw[-1]}"
        return tuple(int(p) for p in raw.split("."))

    pattern = r"/whl/(cu\d+)/" if kind == "cu" else r"/whl/(rocm[\d.]+)/"
    # Sorted by parsed version, NOT string order — "cu92" (9.2) would
    # otherwise sort after "cu128" (12.8) lexicographically ('9' > '1'),
    # picking a much older index than intended. Caught by actually running
    # this against the real index rather than just reading the logic.
    candidates = sorted(set(re.findall(pattern, html)), key=_version_tuple)
    if not candidates:
        return fallback

    if driver_version is None:
        return candidates[-1]  # no way to bound it — use the newest available

    driver_tuple = tuple(int(p) for p in driver_version.split("."))
    usable = [c for c in candidates if _version_tuple(c) <= driver_tuple]
    return usable[-1] if usable else fallback


def detect() -> HardwareVariant:
    vendor, gpu_name = _detect_vendor()
    os_name = platform.system()

    if vendor == "nvidia":
        driver_cuda = _nvidia_driver_cuda_version()
        index = _best_available_index("cu", driver_cuda, _FALLBACK_CUDA_INDEX)
        return HardwareVariant(
            vendor=vendor,
            gpu_name=gpu_name,
            os_name=os_name,
            torch_index_url=f"https://download.pytorch.org/whl/{index}",
            onnxruntime_package="onnxruntime-gpu",
        )

    if vendor == "amd" and os_name == "Linux":
        index = _best_available_index("rocm", None, _FALLBACK_ROCM_INDEX)
        return HardwareVariant(
            vendor=vendor,
            gpu_name=gpu_name,
            os_name=os_name,
            torch_index_url=f"https://download.pytorch.org/whl/{index}",
            onnxruntime_package="onnxruntime-rocm",
        )

    if vendor == "amd" and os_name == "Windows":
        # No ROCm PyTorch build exists for Windows (verified against
        # PyTorch's own published wheel index) — CPU torch, but DirectML
        # still gets onnxruntime GPU acceleration via DirectX 12 instead.
        return HardwareVariant(
            vendor=vendor,
            gpu_name=gpu_name,
            os_name=os_name,
            torch_index_url="https://download.pytorch.org/whl/cpu",
            onnxruntime_package="onnxruntime-directml",
        )

    # vendor == "none", or an AMD GPU on an OS with no supported path above.
    return HardwareVariant(
        vendor=vendor,
        gpu_name=gpu_name,
        os_name=os_name,
        torch_index_url="https://download.pytorch.org/whl/cpu" if os_name != "Darwin" else None,
        onnxruntime_package="onnxruntime",
    )
