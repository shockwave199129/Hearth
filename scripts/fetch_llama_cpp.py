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

Every asset is also pinned by SHA-256 and verified before extraction —
`llama-server` is bundled into the installer and then executed as a
subprocess (desktop/src-tauri/src/main.rs), so an unverified download is
arbitrary code execution on every user's machine wearing our installer's
reputation. TLS is not a substitute: it attests who served the bytes, not
which bytes we expected. When bumping LLAMA_CPP_TAG, run
`python scripts/fetch_llama_cpp.py --print-checksums` and paste the result
into _SHA256_BY_ASSET.
"""
import hashlib
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

# SHA-256 of each asset above, captured from the real ggml-org release for
# tag b10016 via --print-checksums. ggml-org publishes no signatures or
# sidecar digest files, so this is trust-on-first-use pinning: it does not
# prove upstream was honest on the day we recorded it, but it does mean
# every subsequent build gets byte-identical artifacts, and a later
# silent re-upload, hijacked account, or MITM against a CI runner fails
# the build instead of shipping. Any digest change here must be a
# deliberate, reviewed commit.
_SHA256_BY_ASSET = {
    f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz": "9e5c413565e70ddcb2e28a41c9727277135425803a371e070ba7155ea5475893",
    f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-arm64.tar.gz": "c4f8dbdecf6439dfbdae6f003df92a2dfa0a9747e4863e09f8f8bea5b1e55776",
    f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-x64.zip": "5322309f2bde31f8c40f7f041f1e3d8fa08603a5e979c7ff9f4057ac18e37ec6",
    f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-arm64.zip": "c6e410ccf9cde4d8e994ed946422fa0146e8f1c66f5e332dd716cd6cdd579fda",
    f"llama-{LLAMA_CPP_TAG}-bin-macos-arm64.tar.gz": "845211ba3fd3fe5cf365de8ceaa0d73ef85d89830cf5dbddd5d645a2cdb8e09c",
    f"llama-{LLAMA_CPP_TAG}-bin-macos-x64.tar.gz": "b0978851a45a5f786ba03aa058683b38f5da9a110255d68a82be4a913f07e0f5",
}

_RELEASE_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}"


def _current_platform_key() -> tuple[str, str]:
    return (platform.system(), platform.machine())


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Streams to `dest` while hashing, then verifies before returning. On
    mismatch the partial file is removed so a retry can't pick up a
    half-written or hostile archive that was already on disk."""
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
            "Refusing to extract. If this asset was legitimately re-cut "
            "upstream, verify the new artifact by hand and update "
            "_SHA256_BY_ASSET in a reviewed commit."
        )
    print(f"  sha256 ok ({actual})")


def _safe_zip_members(zf: zipfile.ZipFile, staging_dir: Path) -> list[str]:
    """Reject any member that would land outside `staging_dir`.

    CPython's own `extractall` already drops `..` and leading separators,
    but that behaviour is an implementation detail of `_extract_member`,
    not a documented guarantee — and this archive is attacker-controlled
    input if upstream is ever compromised. Check explicitly rather than
    depend on a stdlib internal staying the way it is."""
    base = staging_dir.resolve()
    names = zf.namelist()
    for name in names:
        target = (staging_dir / name).resolve()
        if target != base and base not in target.parents:
            raise SystemExit(
                f"Refusing to extract: archive member {name!r} resolves "
                f"outside {staging_dir} (zip-slip)."
            )
    return names


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
            zf.extractall(staging_dir, members=_safe_zip_members(zf, staging_dir))
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(staging_dir, filter="tar")

    entries = list(staging_dir.iterdir())
    # Exactly one top-level wrapper directory, per the verified archive layout.
    source_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging_dir
    shutil.move(str(source_dir), str(dest_dir))
    shutil.rmtree(staging_dir, ignore_errors=True)


def _print_checksums() -> None:
    """Re-pin every platform's asset after a LLAMA_CPP_TAG bump. Streams
    each archive without keeping it, and prints a paste-ready block."""
    import requests

    print("_SHA256_BY_ASSET = {")
    for asset in dict.fromkeys(_ASSET_BY_PLATFORM.values()):
        digest = hashlib.sha256()
        with requests.get(f"{_RELEASE_BASE_URL}/{asset}", stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
        suffix = asset.split(LLAMA_CPP_TAG, 1)[1]
        print(f'    f"llama-{{LLAMA_CPP_TAG}}{suffix}": "{digest.hexdigest()}",')
    print("}")


def main() -> None:
    if "--print-checksums" in sys.argv[1:]:
        _print_checksums()
        return

    key = _current_platform_key()
    asset = _ASSET_BY_PLATFORM.get(key)
    if asset is None:
        print(f"No bundled llama-server asset mapped for platform {key}.", file=sys.stderr)
        print(f"Supported: {list(_ASSET_BY_PLATFORM)}", file=sys.stderr)
        sys.exit(1)

    # Fail closed: an asset mapped but not pinned means someone bumped the
    # tag without re-pinning, and silently downloading it unverified would
    # defeat the whole point of the map below.
    expected_sha256 = _SHA256_BY_ASSET.get(asset)
    if expected_sha256 is None:
        print(
            f"No pinned SHA-256 for {asset}. Run "
            "`python scripts/fetch_llama_cpp.py --print-checksums` and update "
            "_SHA256_BY_ASSET before building.",
            file=sys.stderr,
        )
        sys.exit(1)

    if LLAMA_CPP_DIR.exists() and any(LLAMA_CPP_DIR.iterdir()):
        print(f"already have {LLAMA_CPP_DIR}")
        return

    url = f"{_RELEASE_BASE_URL}/{asset}"
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RESOURCES_DIR / asset
    _download(url, archive_path, expected_sha256)
    _extract_flattened(archive_path, LLAMA_CPP_DIR)
    archive_path.unlink()
    print(f"Extracted to {LLAMA_CPP_DIR}")


if __name__ == "__main__":
    main()
