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

Backend dependencies are split by tier — `backend/requirements-gpu.txt`
(tier S/A, Chatterbox TTS) vs `backend/requirements-cpu.txt` (tier B/C,
Kokoro TTS) — because chatterbox-tts and kokoro-onnx pin mutually
incompatible numpy ranges, so no single environment can have both
installed. `scripts/build_backend.sh`/`.ps1` detect which one applies
(`scripts/detect_tier_requirements.py`) and install only that one before
freezing; `hearth-backend.spec` probes which package actually got
installed (real `import` check, not a guess) and only references that
tier's `collect_all()`/hidden-imports.

## What this does NOT do yet (real gaps, not oversights)

- **GPU acceleration isn't bundled.** The fetched `llama-server` is
  CPU-only on Linux/Windows (macOS's one variant always includes Metal).
  Tier S/A users on Linux/Windows wanting GPU accel still need to swap in
  their own GPU-enabled `llama-server` build via `LLAMA_SERVER_BIN` — the
  same escape hatch, just no longer the default path.
- **Icons are placeholders.** `icons/*.png` are solid-color placeholders
  generated to unblock `tauri build`; there's no `.ico`/`.icns` yet, which
  Windows/macOS bundling need for a polished result. Run `npx tauri icon
  <source.png>` against a real app icon before shipping.
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
  ```
- `pnpm run tauri:build` — build the frontend and produce installers for
  the *current* platform only (Tauri's bundlers are native, OS-specific
  tooling — you can't produce a `.msi` from Linux or a `.dmg` from Windows;
  `tauri build` only ever builds what the host OS supports).

`tauri.conf.json`'s `bundle.targets` is
`["deb", "rpm", "msi", "nsis", "app", "dmg"]` — deliberately no
`appimage`. Each OS only builds the subset of that list that applies to it,
so this one config covers all three platforms without per-OS forks.

## Building all three installers — Linux (deb/rpm), Windows (msi/nsis), macOS (dmg/app)

Since a single machine can only build its own platform's targets, use
either:

**Option A — CI (recommended, no extra machines needed).**
`.github/workflows/build.yml` matrixes across
`ubuntu-24.04`/`windows-latest`/`macos-latest` using the official
[`tauri-apps/tauri-action`](https://github.com/tauri-apps/tauri-action).
Each job installs the Linux system deps above (Linux only), freezes the
backend, fetches the bundled `llama-server`, and produces all six bundle
files in one run. Push a `v*` tag (or run it manually via "Run workflow")
— it opens a draft GitHub Release with every installer attached. Needs a
real git repo pushed to GitHub to actually run (this project doesn't have
one yet).

**Option B — build manually on each OS:**

| Platform | Prerequisites | Command | Output |
|---|---|---|---|
| Linux | The `apt-get install` above, Python 3.12 | `scripts/build_backend.sh` → `scripts/fetch_llama_cpp.py` → `pnpm run tauri:build` (from `desktop/`) | `src-tauri/target/release/bundle/deb/*.deb`, `.../rpm/*.rpm` |
| Windows | Rust (MSVC toolchain) + [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (C++ workload), Python 3.12. WiX Toolset v3 for `.msi` is auto-downloaded by Tauri's bundler on first build. | `scripts/build_backend.ps1` → `python scripts/fetch_llama_cpp.py` → `pnpm run tauri:build` (from `desktop/`, in PowerShell) | `.../bundle/msi/*.msi`, `.../nsis/*-setup.exe` |
| macOS | Xcode Command Line Tools (`xcode-select --install`), Python 3.12 | `scripts/build_backend.sh` → `scripts/fetch_llama_cpp.py` → `pnpm run tauri:build` (from `desktop/`) | `.../bundle/dmg/*.dmg`, `.../bundle/macos/*.app` |

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
sandbox (no Windows/macOS runners here). What *has* been verified for
real, directly in this sandbox: the PyInstaller freeze builds and the
frozen executable actually starts up correctly for **both** tiers —
once against `requirements-cpu.txt` (kokoro-onnx, ~1.9GB output) and once
against `requirements-gpu.txt` (chatterbox-tts==0.1.7, ~6.0GB output,
`torch==2.6.0+cu124`) — in each case reaching exactly the expected
failure point for this sandbox (`llama-server` not on `PATH`), meaning
every real import in the chain works frozen; the `llama-server`
download/extraction genuinely works and the extracted binary runs with
`LD_LIBRARY_PATH` pointed at its own directory (exactly what `main.rs`
sets up); `cargo check` on `main.rs` resolves its dependency graph
correctly (blocked from finishing only by the same pre-existing Linux
system-library gap noted above, not by anything in this change).
