"""Fetch the optional voice-input models into models/voice/.

Two ONNX files, both consumed by `backend/app/voice/`:

- **silero_vad.onnx** — Silero VAD v5 (MIT). Decides whether a captured
  buffer contains speech at all, so a television or a fan does not become a
  transcript entry.
- **voxceleb_resnet34_LM.onnx** — WeSpeaker ResNet34, VoxCeleb, large-margin
  (CC-BY-4.0, © the WeSpeaker authors — attribution required, see
  docs/attributions.md). Produces 256-dim speaker embeddings for
  verification.

Same trust model and failure behaviour as `fetch_llama_cpp.py`: the digests
below are pinned trust-on-first-use, and a mismatch aborts rather than
installing. Neither upstream publishes signatures, so re-pinning is a
reviewed-commit operation: run `--print-checksums`, verify the new artifact
by hand, and explain the change in the commit message.

Unlike the llama/Python fetches, this one is **not** required to build or run
Hearth. Absent weights leave the voice path exactly as it behaves without
this feature — VAD fails open, speaker verification reports "unavailable"
and never claims a turn was verified. That is why this is a standalone
script rather than a step in `app/setup/orchestrator.py`: enrolling a
voiceprint is an explicit opt-in (it stores biometric data), so downloading
a 26 MB model for every install would be presumptuous.

Usage:
    python scripts/fetch_voice_models.py [--print-checksums] [--dest DIR]
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "models" / "voice"

# Pinned upstream revisions. Both are content-addressed by the digests below,
# so a branch moving under us is caught rather than silently accepted.
_SILERO_REVISION = "v5.1.2"
_WESPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

_ASSETS = {
    "silero_vad.onnx": (
        f"https://raw.githubusercontent.com/snakers4/silero-vad/{_SILERO_REVISION}"
        "/src/silero_vad/data/silero_vad.onnx"
    ),
    "voxceleb_resnet34_LM.onnx": (
        f"https://huggingface.co/{_WESPEAKER_REPO}/resolve/main/voxceleb_resnet34_LM.onnx"
    ),
}

# Verified 2026-08-17 by downloading each asset and hashing it; the ResNet34
# model was additionally checked end-to-end (0.0% EER over 435 same/different
# speaker pairs from LibriSpeech validation-clean) before being pinned here.
_SHA256_BY_ASSET = {
    "silero_vad.onnx": "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
    "voxceleb_resnet34_LM.onnx": "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
}


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Stream to `dest` while hashing, then verify. A rejected file is
    removed so the next run cannot short-circuit onto the bad bytes."""
    import requests

    print(f"Downloading {url} ...")
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
                handle.write(chunk)

    actual = digest.hexdigest()
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 mismatch for {dest.name}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "Refusing to install. If this asset was legitimately re-cut "
            "upstream, verify the new artifact by hand and update "
            "_SHA256_BY_ASSET in a reviewed commit."
        )
    temporary.replace(dest)
    print(f"  sha256 ok ({actual})")


def _print_checksums(dest: Path) -> None:
    print("_SHA256_BY_ASSET = {")
    for name in _ASSETS:
        path = dest / name
        if not path.is_file():
            print(f'    # {name}: not present at {path}')
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        print(f'    "{name}": "{digest.hexdigest()}",')
    print("}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--print-checksums", action="store_true")
    args = parser.parse_args()

    if args.print_checksums:
        _print_checksums(args.dest)
        return 0

    for name, url in _ASSETS.items():
        target = args.dest / name
        expected = _SHA256_BY_ASSET.get(name)
        if expected is None:
            print(
                f"No pinned digest for {name}. Run with --print-checksums and "
                "update _SHA256_BY_ASSET before installing.",
                file=sys.stderr,
            )
            return 1
        if target.is_file():
            print(f"{name} already present at {target} — skipping.")
            continue
        _download(url, target, expected)

    print(f"\nVoice models ready in {args.dest}")
    print("Enrollment is opt-in: Settings -> Your voice -> Set up voice recognition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
