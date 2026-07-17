# Hearth Desktop Installer - Build Instructions

This directory contains the cross-platform installer wizard for Hearth.

## Overview

The installer is built with:
- **Rust + Tauri**: Native desktop application for hardware detection and installation
- **React + TypeScript**: Beautiful, responsive UI for the wizard
- **Platform-specific packaging**: NSIS (Windows), DMG (macOS), DEB/RPM (Linux)

## Directory Structure

```
installer/
  src/
    main.rs              # Tauri app entry point
    lib.rs               # Library exports
    hardware.rs          # Hardware detection logic
    installer.rs         # Installation logic
    ui.rs                # UI utilities
  Cargo.toml             # Rust dependencies
  tauri.conf.json        # Tauri configuration

installer-ui/
  src/
    pages/               # React components for each wizard step
    styles/              # CSS styling
    App.tsx              # Main app component
  package.json           # Node dependencies

scripts/
  build-windows.sh/.ps1  # Windows installer build
  build-macos.sh         # macOS DMG build
  build-linux.sh         # Linux DEB/RPM build

nsis/
  installer.nsi          # NSIS installer script
```

## Building the Installer

### Prerequisites

All platforms:
```bash
# Install Node.js 20+
npm install -g pnpm

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Windows (NSIS)

1. Install [NSIS](https://nsis.sourceforge.io/Download)

2. Build:
```bash
# Using PowerShell
.\desktop\installer\scripts\build-windows.ps1

# Or Bash (Git Bash/WSL)
bash ./desktop/installer/scripts/build-windows.sh
```

3. Output: `Hearth-0.2.6-installer.exe`

### macOS (DMG)

1. Prerequisites:
```bash
# Install build tools
xcode-select --install

# Install Rust targets
rustup target add aarch64-apple-darwin x86_64-apple-darwin
```

2. Build:
```bash
bash ./desktop/installer/scripts/build-macos.sh
```

3. Output: `Hearth-0.2.6.dmg`

### Linux (DEB/RPM)

1. Prerequisites (Ubuntu/Debian):
```bash
sudo apt-get install build-essential libssl-dev libfontconfig1-dev

# For RPM support
sudo apt-get install rpm
```

2. Prerequisites (Fedora/RHEL):
```bash
sudo dnf install gcc openssl-devel fontconfig-devel rpm-build
```

3. Build:
```bash
bash ./desktop/installer/scripts/build-linux.sh
```

4. Output: `hearth_0.2.6_amd64.deb`, `hearth-0.2.6-1.x86_64.rpm`

## Installer Wizard Flow

1. **Welcome Page**
   - Explains Hearth
   - Lists features and requirements

2. **System Detection**
   - Auto-detects GPU (NVIDIA, AMD, Apple Silicon)
   - Detects CPU, RAM, disk space
   - Recommends hardware tier (S/A/B/C)
   - Shows model sizes and TTS engine

3. **Install Path**
   - Default paths per platform
   - Browse directory picker
   - Validates write permissions

4. **Review**
   - Shows all installation settings
   - Note about model downloads on first launch

5. **Installation**
   - Progress bar
   - Extracts files
   - Installs dependencies
   - Downloads models (if applicable)

6. **Completion**
   - Success message
   - Option to launch app
   - Next steps

## Hardware Tier Detection

### Tier S (High Performance)
- NVIDIA GPU: RTX 3070+, RTX 4060+
- 8GB+ RAM
- Full LFM2.5 1.2B model
- Parler-TTS-Tiny-v1 (high quality)

### Tier A (Balanced)
- NVIDIA GPU (6GB+), Apple Silicon, or strong integrated GPU
- 6GB+ RAM
- Full LFM2.5 model
- Parler-TTS-Tiny-v1

### Tier B (Standard)
- AMD GPU or integrated graphics
- 4GB+ RAM
- Quantized LFM2.5 model
- Kokoro (lightweight TTS)

### Tier C (CPU Only)
- Any CPU
- 4GB+ RAM (minimum)
- Quantized LFM2.5 model
- Kokoro TTS

## Development

### Run Installer UI in Dev Mode

```bash
cd desktop/installer-ui
pnpm install
pnpm run dev
```

Access at: `http://localhost:5174`

### Build Installer App (Dev)

```bash
cd desktop/installer
cargo build
```

### Run Tauri Dev Mode

```bash
cd desktop/installer
CARGO_MANIFEST_DIR=. cargo tauri dev
```

## Customization

### Change Installer Colors

Edit `desktop/installer-ui/src/App.css`:
```css
.installer-app {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Change Default Install Path

Edit `desktop/installer/src/installer.rs`:
```rust
pub fn get_default_install_path() -> PathBuf {
    // Windows
    PathBuf::from("C:\\Program Files\\MyApp")
    // macOS
    PathBuf::from("/Applications/MyApp.app")
    // Linux
    PathBuf::from("/opt/myapp")
}
```

### Add Custom Steps

1. Add new page component in `desktop/installer-ui/src/pages/`
2. Add step enum in `desktop/installer-ui/src/App.tsx`
3. Add backend command in `desktop/installer/src/main.rs`

## Troubleshooting

### NSIS Not Found (Windows)
```
Error: NSIS not found
```
Install from: https://nsis.sourceforge.io/Download

### Tauri Build Fails (All Platforms)
```
Error: failed to compile Tauri
```
Update Rust:
```bash
rustup update
rustup component add clippy rustfmt
```

### GPU Detection Not Working
```
Warning: GPU detection failed
```
- Windows: Install NVIDIA/AMD drivers
- Linux: Install `nvidia-smi` or `rocm-smi`
- macOS: Automatic (Metal always available)

## License

Same as Hearth (see `LICENSE`)
