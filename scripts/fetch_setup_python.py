#!/usr/bin/env python3
"""Downloads a standalone Python distribution (astral-sh/python-build-
standalone's `install_only` variant — pip pre-installed, no bootstrap
needed, confirmed via its own docs) into
desktop/src-tauri/resources/setup-python/, as a Tauri bundle resource (see
tauri.conf.json's bundle.resources).

Unlike fetch_llama_cpp.py, this deliberately does NOT extract the archive
here — backend/app/setup/installer.py extracts it lazily on the user's own
machine at first setup, so the installer itself only carries the
compressed ~24-106MB archive, not the larger extracted tree. This
standalone Python is used only to run `pip install` for the hardware-
matched torch/onnxruntime build during first-run setup — see the project
setup plan for why (PyInstaller-frozen apps don't reliably support
installing new packages into themselves at runtime).

Pin a specific release tag rather than "latest" — re-verify this tag and
the _ASSET_BY_PLATFORM map periodically.
"""
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_ROOT / "desktop" / "src-tauri" / "resources"
SETUP_PYTHON_DIR = RESOURCES_DIR / "setup-python"

# Verified against the actual astral-sh/python-build-standalone release
# (July 2026) — matches this project's own Python 3.12 (see
# backend/hearth-backend.spec's target, requirements files).
PBS_TAG = "20260623"
PBS_PYTHON_VERSION = "3.12.13"
_ASSET_BY_PLATFORM = {
    ("Linux", "x86_64"): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz",
    ("Windows", "AMD64"): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz",
    ("Darwin", "arm64"): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz",
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


def main() -> None:
    key = _current_platform_key()
    asset = _ASSET_BY_PLATFORM.get(key)
    if asset is None:
        print(f"No bundled setup-python asset mapped for platform {key}.", file=sys.stderr)
        print(f"Supported: {list(_ASSET_BY_PLATFORM)}", file=sys.stderr)
        sys.exit(1)

    if SETUP_PYTHON_DIR.exists() and any(SETUP_PYTHON_DIR.glob("*.tar.gz")):
        print(f"already have an archive in {SETUP_PYTHON_DIR}")
        return

    url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_TAG}/{asset}"
    SETUP_PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    _download(url, SETUP_PYTHON_DIR / asset)
    print(f"Downloaded to {SETUP_PYTHON_DIR / asset}")


if __name__ == "__main__":
    main()
