#!/usr/bin/env python3
"""Regenerate the desktop app's icon set from assets/logo.png.

Tauri does not resize anything at build time — tauri-action bundles whatever
is sitting in desktop/src-tauri/icons/ and listed under bundle.icon in
desktop/src-tauri/tauri.conf.json. So the generated files are committed, and
running this script is how the branding actually reaches an installer.

    python scripts/generate_icons.py            # regenerate from assets/logo.png
    python scripts/generate_icons.py --check    # fail if anything is stale

--check compares bytes, so it only means "same source, same Pillow". It is a
local sanity check after editing assets/logo.png — not a CI gate, since a
Pillow upgrade can re-encode identical pixels differently and fail it.

Requires Pillow (not a runtime dependency — install it ad hoc):

    pip install pillow
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "assets" / "logo.png"
ICON_DIR = REPO_ROOT / "desktop" / "src-tauri" / "icons"

# Square PNGs, keyed by output filename. 128x128@2x is Tauri's retina variant
# of 128x128 and must be exactly 256px. 64x64 and 256x256 are not in
# tauri.conf.json's icon list but ship in the directory, so keep them in sync.
PNG_SIZES = {
    "32x32.png": 32,
    "64x64.png": 64,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "256x256.png": 256,
    "icon.png": 1024,
}

# Windows reads whichever frame matches the current DPI, so the .ico carries
# the whole ladder. 48 is included because Explorer's medium-icon view uses it.
ICO_SIZES = [16, 32, 48, 64, 128, 256]

# macOS .icns — Pillow writes the frames it can from a 1024px master.
ICNS_SIZE = 1024

DPI = (300, 300)


def load_square_master(path: Path) -> Image.Image:
    """Load the source logo and pad it to a transparent square canvas."""
    img = Image.open(path).convert("RGBA")

    max_side = max(img.width, img.height)
    square = Image.new("RGBA", (max_side, max_side), (0, 0, 0, 0))
    square.paste(img, ((max_side - img.width) // 2, (max_side - img.height) // 2))
    return square


def render(master: Image.Image, size: int) -> Image.Image:
    return master.resize((size, size), Image.Resampling.LANCZOS)


def encode_png(master: Image.Image, size: int) -> bytes:
    buf = io.BytesIO()
    render(master, size).save(buf, format="PNG", dpi=DPI)
    return buf.getvalue()


def encode_ico(master: Image.Image) -> bytes:
    buf = io.BytesIO()
    # Pillow downsamples internally for the extra frames; hand it the largest
    # frame so nothing is upscaled.
    render(master, max(ICO_SIZES)).save(
        buf, format="ICO", sizes=[(s, s) for s in ICO_SIZES]
    )
    return buf.getvalue()


def encode_icns(master: Image.Image) -> bytes:
    buf = io.BytesIO()
    render(master, ICNS_SIZE).save(buf, format="ICNS")
    return buf.getvalue()


def build(master: Image.Image) -> dict[str, bytes]:
    outputs = {name: encode_png(master, size) for name, size in PNG_SIZES.items()}
    outputs["icon.ico"] = encode_ico(master)
    outputs["icon.icns"] = encode_icns(master)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed icons match assets/logo.png; write nothing",
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"error: source logo not found: {SOURCE}", file=sys.stderr)
        return 1

    master = load_square_master(SOURCE)
    if master.width < ICNS_SIZE:
        print(
            f"warning: {SOURCE.name} is {master.width}px square; "
            f"the {ICNS_SIZE}px outputs are upscaled.",
            file=sys.stderr,
        )

    outputs = build(master)
    ICON_DIR.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    for name, data in sorted(outputs.items()):
        target = ICON_DIR / name
        current = target.read_bytes() if target.exists() else None

        if args.check:
            if current != data:
                stale.append(name)
            continue

        if current == data:
            print(f"unchanged: {target.relative_to(REPO_ROOT)}")
            continue

        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()[:12]
        print(f"wrote:     {target.relative_to(REPO_ROOT)}  ({len(data):,} B, {digest})")

    if args.check:
        if stale:
            print(
                "error: icons are out of date with assets/logo.png: "
                + ", ".join(stale)
                + "\nrun: python scripts/generate_icons.py",
                file=sys.stderr,
            )
            return 1
        print(f"all {len(outputs)} icons match {SOURCE.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
