#!/usr/bin/env python3
"""Downloads a prebuilt `llama-server` (+ its required shared libraries —
the release is NOT a standalone binary, confirmed by extracting and
inspecting the actual archives) from ggml-org/llama.cpp's GitHub releases,
and extracts it into desktop/src-tauri/resources/llama-cpp/ as a Tauri
bundle resource (see tauri.conf.json's bundle.resources — this is a whole
directory, not a single externalBin sidecar, because of that shared-library
requirement).

CPU-only builds for Linux/Windows; macOS's one and only build variant
always includes Metal (no CPU-only Mac option — normal for Apple Silicon).
GPU acceleration (CUDA/Vulkan/ROCm) on Linux/Windows is explicitly out of
scope here — see desktop/src-tauri/README.md.

Pin a specific release tag rather than "latest" — llama.cpp cuts new
releases almost daily and the exact asset/library set has changed before.
Re-verify this tag and the _ASSET_BY_PLATFORM map periodically.
"""
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_ROOT / "desktop" / "src-tauri" / "resources"
LLAMA_CPP_DIR = RESOURCES_DIR / "llama-cpp"

# Verified against the actual ggml-org/llama.cpp release (July 2026) —
# see the module docstring for what's deliberately not bundled (GPU builds).
# platform.machine() reports "aarch64" on Linux ARM64 and "ARM64" on
# Windows ARM64 — different strings for the same architecture family, a
# known cross-platform quirk (macOS instead reports "arm64", lowercase).
LLAMA_CPP_TAG = "b10016"
_ASSET_BY_PLATFORM = {
    ("Linux", "x86_64"): f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz",
    ("Linux", "aarch64"): f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-arm64.tar.gz",
    ("Windows", "AMD64"): f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-x64.zip",
    ("Windows", "ARM64"): f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-arm64.zip",
    ("Darwin", "arm64"): f"llama-{LLAMA_CPP_TAG}-bin-macos-arm64.tar.gz",
    ("Darwin", "x86_64"): f"llama-{LLAMA_CPP_TAG}-bin-macos-x64.tar.gz",
}


def _current_platform_key() -> tuple[str, str]:
    return (platform.system(), platform.machine())


def _download(url: str, dest: Path) -> None:
    import requests

    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _extract_flattened(archive_path: Path, dest_dir: Path) -> None:
    """Both the .tar.gz and .zip assets wrap everything in one top-level
    directory (confirmed: `llama-b10016/llama-server`, `llama-b10016/
    libggml-base.so`, etc, all flat inside that one folder) — strip it so
    `llama-server`/`llama-server.exe` sits directly in dest_dir alongside
    its libraries, matching how main.rs resolves the bundled resource path."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    staging_dir = dest_dir.parent / f"{dest_dir.name}-staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(staging_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(staging_dir, filter="tar")

    entries = list(staging_dir.iterdir())
    # Exactly one top-level wrapper directory, per the verified archive layout.
    source_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging_dir
    shutil.move(str(source_dir), str(dest_dir))
    shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    key = _current_platform_key()
    asset = _ASSET_BY_PLATFORM.get(key)
    if asset is None:
        print(f"No bundled llama-server asset mapped for platform {key}.", file=sys.stderr)
        print(f"Supported: {list(_ASSET_BY_PLATFORM)}", file=sys.stderr)
        sys.exit(1)

    if LLAMA_CPP_DIR.exists() and any(LLAMA_CPP_DIR.iterdir()):
        print(f"already have {LLAMA_CPP_DIR}")
        return

    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/{asset}"
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RESOURCES_DIR / asset
    _download(url, archive_path)
    _extract_flattened(archive_path, LLAMA_CPP_DIR)
    archive_path.unlink()
    print(f"Extracted to {LLAMA_CPP_DIR}")


if __name__ == "__main__":
    main()
