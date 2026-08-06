"""Install hearth_ai NLP ONNX package into ``MODELS_DIR/nlp``.

Source order (first complete package wins):
  1. ``backend/bundled/nlp`` (packaged / optional ship)
  2. Repo ``models/nlp`` (dev checkout)
  3. Existing ``NLP_MODELS_DIR`` env if it already points at a complete tree
  4. Public bucket download (``NLP_MODELS_BUCKET_URL``) — the normal path
     for an installed app, since the ~336MB ONNX weights are gitignored
     and never frozen into the installer itself.

Destination is always ``{MODELS_DIR}/nlp`` so runtime resolution prefers the
installed copy. Missing source → log and skip (fail-soft; app still runs).

Downloaded files are pinned by SHA-256 and verified before install. These
are ONNX graphs that get loaded and executed by onnxruntime, fetched over a
base URL that is environment-overridable — the weakest integrity story of
any artifact this app fetches, and the reason ``_SHA256_BY_RELATIVE``
exists. Note the deliberate asymmetry with fail-soft: a *missing* classifier
package is fine (the app degrades gracefully), but a classifier that fails
its digest is refused outright rather than used.

Locally sourced packages (``backend/bundled/nlp``, the dev checkout's
``models/nlp``) are NOT hash-gated: those ship inside the installer or come
from the developer's own working tree, where re-exporting a retrained model
is routine and a pinned digest would only produce false alarms.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from app.config import BACKEND_DIR, MODELS_DIR, NLP_MODELS_BUCKET_URL

ProgressFn = Callable[[str], None]

NLP_INSTALL_DIR = MODELS_DIR / "nlp"

_NLP_REQUIRED_RELATIVE = (
    "manifest.json",
    "tokenizer.json",
    "encoder/model.onnx",
    "emotion/model.onnx",
    "emotion/labels.json",
    "intent/model.onnx",
    "intent/labels.json",
    "memory/model.onnx",
    "memory/labels.json",
    "relationship/model.onnx",
    "relationship/labels.json",
    "strategy/model.onnx",
    "strategy/labels.json",
)

# SHA-256 of every file above as published in the bucket. Regenerate with
# `python -m app.setup.nlp_models --print-checksums` from `backend/` after
# exporting a new package, and land the change as a reviewed commit — an
# unreviewed digest bump is indistinguishable from an attack.
_SHA256_BY_RELATIVE = {
    "manifest.json": "cad3ac8768f4efee0821baf61421bde77a2240b1ddfb7e1cf44cc800e94f62e7",
    "tokenizer.json": "f34f595ffaebaad2bac6b845a69ffd47381d5a73a1c6c0967aac7d17b72ce5da",
    "encoder/model.onnx": "dd402674ccd84994413e01efbeab33d1e69c48c2eeb0e221b35e9b958da8e87d",
    "emotion/model.onnx": "75202f1948899bd8148b34259e4cba9bd077499a417cd3b3556efd21b6619ae8",
    "emotion/labels.json": "ffcc20424b3f7689d68eef1f2bffb6e747b4e0cc47c07cfba1d10df8c3a21a1a",
    "intent/model.onnx": "689e0164063b58a83ce653193220a8fc6ee3b9a120d045adb54f6aa8e05689e4",
    "intent/labels.json": "f2285379df22cdece6419878168672b9336c12f578041ce29e08e308bd1435b3",
    "memory/model.onnx": "c04f177b63f69606f10cc2a2fb99b492c7c58c5c2d29ced39be851edd165feb1",
    "memory/labels.json": "4c41bab7bf8f28d3aa4e405096c3a092039ef35d207daa6d263125b00906ac47",
    "relationship/model.onnx": "293c8f7b4ca310e7ad01bfded89ea1722d561c7e211fba7edf5fd9962da3463a",
    "relationship/labels.json": "56bf02cdde2451d9042a1fdf76980e6cf59a28f91e62a010037ffed1212297f4",
    "strategy/model.onnx": "703070d95f6dfe6ae45ed051dc3a08d6a392948b2f5f19ffc121bdb6a3c5d3cf",
    "strategy/labels.json": "8ace05f35f81d5ec28343bc7b2fd084033cbf4669d02f6e7f137374d06714afc",
}


def nlp_package_complete(root: Path | None) -> bool:
    if root is None or not root.is_dir():
        return False
    return all((root / rel).is_file() for rel in _NLP_REQUIRED_RELATIVE)


def find_nlp_source() -> Path | None:
    """Locate a complete ONNX package to install from."""
    candidates: list[Path] = []
    env = os.environ.get("NLP_MODELS_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(BACKEND_DIR / "bundled" / "nlp")
    if not getattr(__import__("sys"), "frozen", False):
        candidates.append(BACKEND_DIR.parent / "models" / "nlp")
    for path in candidates:
        if nlp_package_complete(path):
            return path
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NlpIntegrityError(RuntimeError):
    """A downloaded NLP artifact did not match its pinned digest."""


def _download_file(url: str, dest: Path, rel: str, log: ProgressFn) -> None:
    """Stream ``url`` into ``dest`` via a temp file + atomic replace, so a
    crash/interrupt mid-download never leaves a half-written file behind
    for ``nlp_package_complete`` to mistake as usable.

    The digest is checked on the temp file, before the replace — a file that
    fails verification never exists at its final path at all, so no later
    "already present" short-circuit can pick it up."""
    expected = _SHA256_BY_RELATIVE.get(rel)
    if expected is None:
        raise NlpIntegrityError(
            f"no pinned SHA-256 for {rel}. Run "
            "`python -m app.setup.nlp_models --print-checksums` and update "
            "_SHA256_BY_RELATIVE."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    log(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)

    actual = _sha256_file(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise NlpIntegrityError(
            f"SHA-256 mismatch for {rel}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    tmp.replace(dest)


def download_nlp_models(log: ProgressFn = print) -> Path | None:
    """Fetch the ONNX package from ``NLP_MODELS_BUCKET_URL`` into
    ``NLP_INSTALL_DIR``, one file per required relative path (the bucket
    denies ``ListBucket`` for these credentials, so discovery isn't an
    option — the required-file list is the source of truth on both ends).
    """
    if not NLP_MODELS_BUCKET_URL:
        log("NLP_MODELS_BUCKET_URL not set — skipping NLP classifier download")
        return None

    # The URL is env-overridable, so the transport is not guaranteed by the
    # default alone. Plain HTTP would let anyone on the path swap the ONNX
    # graphs; the digest check would catch it, but there is no reason to
    # accept the downgrade in the first place.
    if urlparse(NLP_MODELS_BUCKET_URL).scheme != "https":
        log(
            f"refusing non-HTTPS NLP_MODELS_BUCKET_URL ({NLP_MODELS_BUCKET_URL!r}) "
            "— classifiers will fail-soft"
        )
        return None

    log(f"downloading NLP classifiers from {NLP_MODELS_BUCKET_URL} ...")
    for rel in _NLP_REQUIRED_RELATIVE:
        dest = NLP_INSTALL_DIR / rel
        if dest.is_file() and dest.stat().st_size > 0:
            # Present from an earlier run. Verify rather than assume: this is
            # the one path where a file predating the digest map, or one left
            # by an interrupted older build, could otherwise be installed
            # unchecked. Mismatch → delete and re-fetch.
            if _sha256_file(dest) == _SHA256_BY_RELATIVE.get(rel):
                continue
            log(f"{rel} does not match its pinned digest — discarding and re-fetching")
            dest.unlink(missing_ok=True)
        try:
            _download_file(f"{NLP_MODELS_BUCKET_URL}/{rel}", dest, rel, log)
        except OSError as exc:
            log(f"NLP classifier download failed ({rel}: {exc}) — classifiers will fail-soft")
            return None
        except NlpIntegrityError as exc:
            # Not fail-soft in the usual sense: absent classifiers are fine,
            # wrong ones are not. Abort the whole install so a partially
            # verified tree never reaches nlp_package_complete.
            log(f"NLP classifier integrity check failed — refusing to install\n{exc}")
            shutil.rmtree(NLP_INSTALL_DIR, ignore_errors=True)
            return None

    if not nlp_package_complete(NLP_INSTALL_DIR):
        log(f"NLP download incomplete at {NLP_INSTALL_DIR}")
        return None
    log("NLP classifiers downloaded")
    return NLP_INSTALL_DIR


def ensure_nlp_models(log: ProgressFn = print) -> Path | None:
    """Copy or download NLP classifiers into ``MODELS_DIR/nlp`` if not
    already present.

    Returns the install directory when usable, else None.
    """
    if nlp_package_complete(NLP_INSTALL_DIR):
        log(f"already have NLP classifiers at {NLP_INSTALL_DIR}")
        return NLP_INSTALL_DIR

    src = find_nlp_source()
    if src is None:
        return download_nlp_models(log)

    if src.resolve() == NLP_INSTALL_DIR.resolve():
        log(f"NLP classifiers already at install path {NLP_INSTALL_DIR}")
        return NLP_INSTALL_DIR

    log(f"installing NLP classifiers from {src} → {NLP_INSTALL_DIR}")
    if NLP_INSTALL_DIR.exists():
        shutil.rmtree(NLP_INSTALL_DIR)
    shutil.copytree(src, NLP_INSTALL_DIR)

    if not nlp_package_complete(NLP_INSTALL_DIR):
        log(f"NLP install incomplete at {NLP_INSTALL_DIR}")
        return None
    log("NLP classifiers installed")
    return NLP_INSTALL_DIR


def _print_checksums() -> None:
    """Regenerate _SHA256_BY_RELATIVE from whatever the bucket serves now.

    Streams each file without keeping it. Read the diff before committing:
    this prints what upstream *is*, which is only the same as what it
    *should be* if nothing has been tampered with in between."""
    print("_SHA256_BY_RELATIVE = {")
    for rel in _NLP_REQUIRED_RELATIVE:
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{NLP_MODELS_BUCKET_URL}/{rel}", timeout=600) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                digest.update(chunk)
        print(f'    "{rel}": "{digest.hexdigest()}",')
    print("}")


if __name__ == "__main__":
    import sys

    if "--print-checksums" in sys.argv[1:]:
        _print_checksums()
    else:
        print("usage: python -m app.setup.nlp_models --print-checksums", file=sys.stderr)
        sys.exit(1)
