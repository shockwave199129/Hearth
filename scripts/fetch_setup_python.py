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
standalone Python is used only to bootstrap `uv` and run
`python -m uv pip install --target …` for the hardware-matched
torch/onnxruntime build during first-run setup — see the project setup
plan for why (PyInstaller-frozen apps don't reliably support installing
new packages into themselves at runtime).

Pin a specific release tag rather than "latest" — re-verify this tag and
the _ASSET_BY_PLATFORM map periodically.

Every asset is also pinned by SHA-256 and verified on download. This
archive ships inside the installer and is extracted and *executed* on the
user's machine at first setup (backend/app/setup/installer.py), so the
same reasoning as fetch_llama_cpp.py applies: TLS proves who served the
bytes, not which bytes we expected. When bumping PBS_TAG or
PBS_PYTHON_VERSION, run
`python scripts/fetch_setup_python.py --print-checksums` and paste the
result into _SHA256_BY_ASSET.
"""

import hashlib
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
    (
        "Linux",
        "x86_64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz",
    (
        "Linux",
        "aarch64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz",
    (
        "Windows",
        "AMD64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz",
    (
        "Windows",
        "ARM64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-pc-windows-msvc-install_only.tar.gz",
    (
        "Darwin",
        "arm64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz",
    (
        "Darwin",
        "x86_64",
    ): f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-apple-darwin-install_only.tar.gz",
}

# SHA-256 of each asset above, captured from the real astral-sh release for
# tag 20260623 via --print-checksums. Same trust-on-first-use caveat as
# fetch_llama_cpp.py's map: this pins the artifact, it does not attest
# upstream. Any change here must be a deliberate, reviewed commit.
_SHA256_BY_ASSET = {
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz": "9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930",
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz": "b14d074c43fdf03f01822fd07a15b3039eb0558503d1cb791791602cbe32908b",
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz": "c6af85bb83d5158c9ff71f50dfad467853d1cd236f932b144e87e26e2ea2a83e",
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-pc-windows-msvc-install_only.tar.gz": "d459934cc52d28212a438cb6c9cfcb5396a8f23ed74cb47744d7af7fac618ad8",
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz": "3724aa4dafb5f7b6c2cf98e89914e4248dc6bd2fe40407df4a2d73de99615f16",
    f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}-x86_64-apple-darwin-install_only.tar.gz": "7c57fdd1fa675190093700eb0d8e7117e1f9eae7c30a46dea5f8d5266bcfc791",
}

_RELEASE_BASE_URL = (
    f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_TAG}"
)


def _current_platform_key() -> tuple[str, str]:
    return (platform.system(), platform.machine())


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Streams to `dest` while hashing, then verifies. On mismatch the
    partial file is removed — otherwise the next run's "already have an
    archive" short-circuit below would happily bundle the rejected bytes."""
    import requests

    print(f"Downloading {url} ...")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
                f.write(chunk)

    actual = digest.hexdigest()
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 mismatch for {dest.name}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "Refusing to bundle. If this asset was legitimately re-cut "
            "upstream, verify the new artifact by hand and update "
            "_SHA256_BY_ASSET in a reviewed commit."
        )
    print(f"  sha256 ok ({actual})")


def _print_checksums() -> None:
    """Re-pin every platform's asset after a PBS_TAG/PBS_PYTHON_VERSION
    bump. Streams each archive without keeping it."""
    import requests

    prefix = f"cpython-{PBS_PYTHON_VERSION}+{PBS_TAG}"
    print("_SHA256_BY_ASSET = {")
    for asset in dict.fromkeys(_ASSET_BY_PLATFORM.values()):
        digest = hashlib.sha256()
        with requests.get(f"{_RELEASE_BASE_URL}/{asset}", stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
        suffix = asset[len(prefix):]
        print(
            f'    f"cpython-{{PBS_PYTHON_VERSION}}+{{PBS_TAG}}{suffix}": '
            f'"{digest.hexdigest()}",'
        )
    print("}")


def main() -> None:
    if "--print-checksums" in sys.argv[1:]:
        _print_checksums()
        return

    key = _current_platform_key()
    asset = _ASSET_BY_PLATFORM.get(key)
    if asset is None:
        print(
            f"No bundled setup-python asset mapped for platform {key}.", file=sys.stderr
        )
        print(f"Supported: {list(_ASSET_BY_PLATFORM)}", file=sys.stderr)
        sys.exit(1)

    # Fail closed — see fetch_llama_cpp.py's identical guard.
    expected_sha256 = _SHA256_BY_ASSET.get(asset)
    if expected_sha256 is None:
        print(
            f"No pinned SHA-256 for {asset}. Run "
            "`python scripts/fetch_setup_python.py --print-checksums` and "
            "update _SHA256_BY_ASSET before building.",
            file=sys.stderr,
        )
        sys.exit(1)

    if SETUP_PYTHON_DIR.exists() and any(SETUP_PYTHON_DIR.glob("*.tar.gz")):
        print(f"already have an archive in {SETUP_PYTHON_DIR}")
        return

    url = f"{_RELEASE_BASE_URL}/{asset}"
    SETUP_PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    _download(url, SETUP_PYTHON_DIR / asset, expected_sha256)
    print(f"Downloaded to {SETUP_PYTHON_DIR / asset}")


if __name__ == "__main__":
    main()
