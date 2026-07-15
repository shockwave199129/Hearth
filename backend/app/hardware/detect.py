"""RAM, VRAM, CPU probe — run once per launch, see project-plan.md §2."""
import re
import shutil
import subprocess

import psutil


def _detect_nvidia_gpu() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, vram_mib = out.splitlines()[0].split(",")
    return {"name": name.strip(), "vram_gb": round(float(vram_mib) / 1024, 1)}


def _detect_rocm_gpu() -> dict | None:
    if shutil.which("rocm-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    name_match = re.search(r"Card series:\s*(.+)", out)
    vram_match = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", out)
    if not (name_match and vram_match):
        return None
    return {
        "name": name_match.group(1).strip(),
        "vram_gb": round(int(vram_match.group(1)) / (1024**3), 1),
    }


def detect_gpu() -> dict | None:
    return _detect_nvidia_gpu() or _detect_rocm_gpu()


def detect_hardware() -> dict:
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu = detect_gpu()
    return {
        "ram_gb": round(ram_gb, 1),
        "cpu_count": psutil.cpu_count(logical=True) or 1,
        "gpu_name": gpu["name"] if gpu else None,
        "vram_gb": gpu["vram_gb"] if gpu else 0,
    }
