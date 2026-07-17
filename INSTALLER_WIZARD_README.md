# Custom Installation Wizard for Hearth

This branch implements a comprehensive cross-platform installation wizard for Hearth that runs when users download and install `.exe`, `.msi`, `.dmg`, or `.deb`/`.rpm` files.

## What's New

### 🎯 Smart Hardware Detection
- **Automatic GPU Detection**: Detects NVIDIA, AMD, and Apple Silicon GPUs
- **Hardware Tier Recommendation**: Automatically recommends Tier S/A/B/C based on:
  - GPU model and VRAM
  - CPU cores and RAM
  - Available disk space
- **Platform-Specific**: Tailored detection for Windows, macOS, and Linux

### 🎨 Beautiful Multi-Step Wizard
1. **Welcome** - Introduction and requirements
2. **System Detection** - Hardware analysis and tier recommendation
3. **Install Path** - Choose installation location with validation
4. **Review** - Confirm settings before installing
5. **Installation** - Progress tracking with status updates
6. **Completion** - Success message with next steps

### 🚀 Technology Stack
- **Backend**: Rust + Tauri (native performance, small footprint)
- **Frontend**: React + TypeScript (responsive, accessible UI)
- **Packaging**: 
  - Windows: NSIS installer
  - macOS: DMG disk image
  - Linux: DEB and RPM packages

## Installation Flow

### First-Time User Downloads `.exe`/`.dmg`/`.deb`

```
Download Installer
       ↓
  Launch Wizard
       ↓
  Hardware Detection (automatic)
       ↓
  Tier Recommendation
       ↓
  Choose Install Path
       ↓
  Review Settings
       ↓
  Install Files + Shortcuts
       ↓
  Completion Screen
       ↓
  User can launch Hearth app
       ↓
  In-app setup continues (model downloads, onboarding)
```

## Hardware Tier Recommendations

### Tier S - High Performance
- **GPU**: NVIDIA RTX 3070+, RTX 4060+
- **RAM**: 8GB+
- **Model**: Full LFM2.5 1.2B
- **TTS**: Parler-TTS-Tiny-v1 (high quality)
- **Speed**: ~200ms per response

### Tier A - Balanced
- **GPU**: NVIDIA (6GB+), Apple Silicon, strong integrated GPU
- **RAM**: 6GB+
- **Model**: Full LFM2.5 1.2B
- **TTS**: Parler-TTS-Tiny-v1
- **Speed**: ~300-400ms per response

### Tier B - Standard
- **GPU**: AMD GPU, Intel integrated GPU
- **RAM**: 4GB+
- **Model**: Quantized LFM2.5
- **TTS**: Kokoro (lightweight)
- **Speed**: ~500-800ms per response

### Tier C - CPU Only
- **GPU**: None
- **RAM**: 4GB+
- **Model**: Quantized LFM2.5
- **TTS**: Kokoro
- **Speed**: ~1-2s per response

## Files Added

```
desktop/
├── installer/              # NEW: Tauri app for installer
│   ├── src/
│   │   ├── main.rs
│   │   ├── hardware.rs     # GPU/CPU detection
│   │   ├── installer.rs    # Installation logic
│   │   └── ui.rs
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── scripts/
│       ├── build-windows.ps1
│       ├── build-macos.sh
│       └── build-linux.sh
│
├── installer-ui/           # NEW: React frontend for wizard
│   ├── src/
│   │   ├── pages/          # Wizard step components
│   │   ├── styles/         # CSS
│   │   └── App.tsx
│   └── package.json
│
└── .github/
    └── workflows/
        └── build-installers.yml  # NEW: CI/CD for multi-platform builds
```

## Building Installers

### Quick Start (macOS)
```bash
cd desktop/installer/scripts
bash ./build-macos.sh
# Output: Hearth-0.2.6.dmg
```

### Quick Start (Windows)
```powershell
cd desktop/installer/scripts
.\build-windows.ps1
# Output: Hearth-0.2.6-installer.exe
```

### Quick Start (Linux)
```bash
cd desktop/installer/scripts
bash ./build-linux.sh
# Output: hearth_0.2.6_amd64.deb, hearth-0.2.6-1.x86_64.rpm
```

## Key Features

✅ **Cross-Platform**: Windows, macOS, Linux (ARM64 + x86_64)
✅ **Smart Detection**: Automatically detects GPU, CPU, RAM, disk
✅ **No External Dependencies**: Everything bundled in installer
✅ **Accessible UI**: Works with screen readers, keyboard navigation
✅ **Fast & Lightweight**: ~50MB installer, instant launch
✅ **Error Handling**: Validates paths, checks disk space, graceful failures
✅ **Localization Ready**: UI structure supports multiple languages

## Usage for End Users

### Windows
1. Download `Hearth-0.2.6-installer.exe`
2. Run the installer
3. Follow the wizard steps
4. Hearth is installed to `C:\Program Files\Hearth`
5. Launch from Start Menu or Desktop shortcut

### macOS
1. Download `Hearth-0.2.6.dmg`
2. Double-click to mount
3. Drag Hearth to Applications (or run installer wizard)
4. Open from Applications folder

### Linux
1. Download `hearth_0.2.6_amd64.deb` or `.rpm`
2. Install: `sudo dpkg -i hearth_0.2.6_amd64.deb`
3. Launch: `hearth-installer` or from application menu

## Integration with Main App

The installer wizard is **separate from the in-app setup flow**:

- **Installer Wizard** (this PR): 
  - First-time installation experience
  - Hardware detection at install-time
  - Platform-specific packaging

- **In-App Setup** (existing, unchanged):
  - Runs on first app launch after install
  - Hardware tier confirmation
  - Model downloads
  - User onboarding (name, preferences, etc.)

## Development

### Run in Dev Mode
```bash
cd desktop/installer-ui
pnpm install
pnpm run dev

# In another terminal
cd desktop/installer
cargo tauri dev
```

### Testing Hardware Detection
```rust
// Test in Rust
cd desktop/installer
cargo test

// Or run directly
echo '{}' | cargo run -- detect_system_info
```

## Future Enhancements

- [ ] Custom installer themes/branding
- [ ] Multilingual support (French, Spanish, German, etc.)
- [ ] Pre-download models during installation
- [ ] System tray integration
- [ ] Automatic updates
- [ ] User feedback/telemetry (optional, privacy-respecting)

## Questions?

See `desktop/installer/INSTALLER_README.md` for detailed documentation.
