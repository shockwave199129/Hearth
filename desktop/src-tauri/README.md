# Tauri packaging

This wraps the built frontend (`../../frontend/dist`) and manages the
backend as a child process, killing it when the window closes. See
`src/main.rs` for the exact wiring. The Tauri CLI itself lives in
`desktop/package.json`, decoupled from the web app's own dependencies
(`frontend/package.json`).

**Dev builds** spawn `python3 -m app.main` in `../../backend` directly —
fast iteration, assumes a dev Python env, unchanged from earlier passes.

**Release builds** need neither Python nor a separately-installed
`llama-server` on the machine running the installed app:
- The backend is frozen into a standalone executable via PyInstaller
  (`backend/hearth-backend.spec`, built by
  `scripts/build_backend.sh`/`.ps1`) and bundled as a Tauri resource.
- A prebuilt `llama-server` (from `ggml-org/llama.cpp`'s GitHub releases,
  fetched by `scripts/fetch_llama_cpp.py`) is bundled the same way.
- Both land in `desktop/src-tauri/resources/` before `tauri build` runs
  (gitignored — regenerated fresh each build, see `.github/workflows/build.yml`),
  and `main.rs` spawns the frozen backend pointed at the bundled
  llama-server via the `LLAMA_SERVER_BIN` env var
  (`backend/app/config.py` already reads this).

The CI/release freeze is now a **thin build**: `scripts/build_backend.sh`/
`.ps1` install only `backend/requirements-common.txt` before running
PyInstaller — no torch, onnxruntime, parler-tts, or ttstokenizer gets
frozen into the installer at all, for either hardware tier. That used to
be split by tier (`backend/requirements-gpu.txt` for Parler-TTS-Tiny-v1
tier S/A, `backend/requirements-cpu.txt` for Kokoro/onnxruntime tier B/C),
with `.github/workflows/build.yml`'s matrix passing the tier explicitly
(one CI job per OS per tier, six total) and `scripts/
detect_tier_requirements.py` autodetecting the *build* machine's hardware
for local no-argument runs. Both of those are gone: CI has no GPU to match
a build-time tier decision against anyway (that autodetection script has
been deleted), and freezing a heavy tier-specific stack in at build time
is exactly what caused the 2GB GitHub release-asset limit and RPM
bundling-slowness problems this project used to work around.

That tier decision — and which CUDA/ROCm/DirectML variant to use — now
happens on the *end user's own machine*, at first launch, via a new in-app
setup flow (`backend/app/setup/hardware_variant.py`, `installer.py`,
`models.py`, `orchestrator.py`, exposed as `GET /api/setup/status`,
`POST /api/setup/start`, `GET /api/setup/progress`). It detects NVIDIA
GPU + CUDA driver version, AMD GPU (ROCm on Linux), or no GPU (CPU-only,
or `onnxruntime-directml` on Windows), then pip-installs the matching
`torch`/`onnxruntime` build using a standalone Python interpreter bundled
as a Tauri resource (`desktop/src-tauri/resources/setup-python/`, fetched
by `scripts/fetch_setup_python.py` from `astral-sh/python-build-standalone`
— PyInstaller-frozen apps don't reliably support installing new packages
into themselves at runtime, hence the separate interpreter). The same
in-app flow also downloads the LLM/embedding model files, superseding
`scripts/setup.py` as the primary path — that script still exists and
still works as a manual/headless CLI alternative for local dev (see the
root README), but the packaged app no longer needs it run beforehand.
`requirements-gpu.txt`/`requirements-cpu.txt` still exist and still encode
the tier S/A vs B/C package split; they're just installed by
`backend/app/setup/orchestrator.py` at first run now instead of by
`build_backend.sh`/`.ps1` at freeze time.

This in-app setup flow has been verified locally in isolation (hardware
detection, package-index selection, the model-download logic it shares
with `scripts/setup.py`), but **not** end-to-end against a real installed
app on real GPU/CPU hardware — treat it as unverified in that sense until
someone actually runs the packaged installer's first-launch flow on a
target machine.

## What this does NOT do yet (real gaps, not oversights)

- **GPU acceleration isn't bundled.** The fetched `llama-server` is
  CPU-only on Linux/Windows (macOS's one variant always includes Metal).
  Tier S/A users on Linux/Windows wanting GPU accel still need to swap in
  their own GPU-enabled `llama-server` build via `LLAMA_SERVER_BIN` — the
  same escape hatch, just no longer the default path.
- **Icons are placeholders.** All of `icons/` (including `icon.ico`/
  `icon.icns`, generated via `pnpm tauri icon src-tauri/icons/icon.png`
  after Windows CI failed with `icons/icon.ico not found` — tauri-build's
  build.rs needs that exact file to embed a Windows resource icon,
  independent of tauri.conf.json's `icon` list) are still just the
  original solid-color placeholder upscaled, not real artwork. Re-run that
  same command against a real app icon before shipping.
- **Unsigned builds** — see the code-signing section below.

## Linux build prerequisites

`cargo check`/`cargo build` need these system dev packages (not Rust
crates) on Linux — install once per machine:

```
sudo apt-get install libwebkit2gtk-4.1-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev libglib2.0-dev
```

Without them, GTK/GLib `-sys` crate build scripts fail at the `pkg-config`
step before your own Rust code even compiles — that's a missing-system-
library error, not a bug in this scaffold.

## Commands

- `pnpm run tauri:dev` — run the desktop shell against the Vite dev
  server, spawning the dev-mode Python backend (fast iteration, no freeze
  needed).
- Before a real release build, populate the bundled resources once:
  ```
  bash scripts/build_backend.sh      # or scripts/build_backend.ps1 on Windows
  python3 scripts/fetch_llama_cpp.py
  python3 scripts/fetch_setup_python.py
  ```
- `pnpm run tauri:build` — build the frontend and produce installers for
  the *current* platform only (Tauri's bundlers are native, OS-specific
  tooling — you can't produce a `.msi` from Linux or a `.dmg` from Windows;
  `tauri build` only ever builds what the host OS supports).

### Local Windows test builds (no tag / no CI wait)

On a Windows machine, one script mirrors the GitHub Actions Windows job
(pnpm install → freeze backend → fetch llama-server + setup-python →
`tauri build`):

```powershell
git pull
.\scripts\build_windows_installer.ps1
# optional: .\scripts\build_windows_installer.ps1 -Version 0.2.11
#
# If you get an execution-policy error:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installer.ps1
```

(`pwsh` is PowerShell 7+; the built-in Windows shell is fine — run the
`.ps1` directly as above.)
Installers land under `desktop/src-tauri/target/release/bundle/msi/` and
`.../nsis/`. Install one, then exercise Setup → Chat. Default version is
`0.0.0` (same as CI `workflow_dispatch` non-tag runs). Unsigned builds may
hit SmartScreen; on the build machine use More info → Run anyway.

`tauri.conf.json`'s `bundle.targets` is
`["deb", "rpm", "msi", "nsis", "app", "dmg"]` — deliberately no
`appimage`. Each OS only builds the subset of that list that applies to it,
so this one config covers all three platforms without per-OS forks.

## Building all three installers — Linux (deb/rpm), Windows (msi/nsis), macOS (dmg/app)

Since a single machine can only build its own platform's targets, use
either:

**Option A — CI (recommended, no extra machines needed).**
`.github/workflows/build.yml` matrixes across
`ubuntu-24.04`/`windows-latest`/`macos-latest` only — three jobs total,
one per OS, not per hardware tier. Each job installs the Linux system deps
above (Linux only), freezes the backend as the thin build described
above, fetches the bundled `llama-server`, fetches the bundled
setup-time Python (`scripts/fetch_setup_python.py`), and builds via the
official [`tauri-apps/tauri-action`](https://github.com/tauri-apps/tauri-action)
— there's just one `productName` (`Hearth`, from `tauri.conf.json`) now,
since there's no longer a GPU-tier/CPU-tier split to keep as distinct
assets. (This used to matrix across a hardware-tier axis too — six jobs,
with a tier-specific `Hearth-GPU`/`Hearth-CPU` `productName` override to
keep the two tiers' installers from colliding — collapsed back to three
jobs and one plain name now that neither installer needs to differ by
tier at all.) Push a `v*` tag (or run it manually via "Run workflow") — it
opens a draft GitHub Release with every installer from every job attached.
Needs a real git repo pushed to GitHub to actually run (this project
doesn't have one yet).

**Option B — build manually on each OS:**

| Platform | Prerequisites | Command | Output |
|---|---|---|---|
| Linux | The `apt-get install` above, Python 3.12 | `scripts/build_backend.sh` → `scripts/fetch_llama_cpp.py` → `scripts/fetch_setup_python.py` → `pnpm run tauri:build` (from `desktop/`) | `src-tauri/target/release/bundle/deb/*.deb`, `.../rpm/*.rpm` |
| Windows | Rust (MSVC toolchain) + [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (C++ workload), Python 3.12, Node/pnpm. WiX Toolset v3 for `.msi` is auto-downloaded by Tauri's bundler on first build. | **Preferred:** `.\scripts\build_windows_installer.ps1` from repo root (CI parity). Manual: `scripts/build_backend.ps1` → `python scripts/fetch_llama_cpp.py` → `python scripts/fetch_setup_python.py` → `pnpm run tauri:build` (from `desktop/`) | `.../bundle/msi/*.msi`, `.../nsis/*-setup.exe` |
| macOS | Xcode Command Line Tools (`xcode-select --install`), Python 3.12 | `scripts/build_backend.sh` → `scripts/fetch_llama_cpp.py` → `scripts/fetch_setup_python.py` → `pnpm run tauri:build` (from `desktop/`) | `.../bundle/dmg/*.dmg`, `.../bundle/macos/*.app` |

**Code signing — not done here.** Unsigned builds will warn or outright
block users on Windows (SmartScreen) and macOS (Gatekeeper will say the
app "is damaged" or refuse to open it on any machine but the one that
built it). Before real distribution you need:
- **Windows**: a code-signing certificate, wired into `tauri-action` via
  the `TAURI_SIGNING_PRIVATE_KEY`-style secrets (commented in
  `.github/workflows/build.yml`).
- **macOS**: an Apple Developer ID certificate + notarization
  (`APPLE_CERTIFICATE`/`APPLE_ID`/`APPLE_TEAM_ID` etc., also commented in
  the workflow) — Apple Developer Program membership required ($99/yr).
- **Linux**: no signing required for local `.deb`/`.rpm` install, though a
  real distribution channel (a PPA, a signed repo) would want GPG-signed
  packages.

None of the CI matrix / real Windows/macOS builds can be verified in this
sandbox (no Windows/macOS runners here). The thin PyInstaller freeze
(`requirements-common.txt` only, no TTS stack at all) hasn't been
re-verified end-to-end against a real `pyinstaller` run in this sandbox
either — no output-size or import-chain claim should be assumed to hold
until that's actually done. The in-app setup flow that now performs the
former tier-specific install
(`backend/app/setup/orchestrator.py` pip-installing
`requirements-gpu.txt`/`requirements-cpu.txt` against the bundled
setup-Python) has likewise only been exercised locally in isolation, not
against a real frozen/installed build on real GPU or CPU hardware —
treat both as unverified in that sense. Separately,
the `llama-server` download/extraction genuinely works and the extracted
binary runs with `LD_LIBRARY_PATH` pointed at its own directory (exactly
what `main.rs` sets up); `cargo check` on `main.rs` resolves its
dependency graph correctly (blocked from finishing only by the same
pre-existing Linux system-library gap noted above, not by anything in
this change).
