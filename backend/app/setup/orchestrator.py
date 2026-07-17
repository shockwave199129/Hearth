"""Ties hardware/tier detection, package installation, and model downloads
into one setup flow — see main.py's /api/setup/* endpoints for how the
frontend drives this, and the project setup plan for the overall design.
"""
from pathlib import Path

from app.config import BACKEND_DIR, EMBEDDING_MODEL_FILE, EMBEDDING_MODELS_DIR, LLM_MODELS_DIR
from app.hardware.detect import detect_hardware
from app.hardware.tier_manager import TierConfig, pick_tier, resolve_paths
from app.setup import hardware_variant, models
from app.setup.installer import InstallProgress, SetupError, install_packages
from app.setup.state import clear_setup_complete, is_setup_complete
from app.setup.state import mark_setup_complete as mark_complete

# Bundled via hearth-backend.spec's datas (dest ".") specifically so this
# resolves correctly in both dev and frozen modes without its own
# frozen/dev branching — see that spec's comment on why setup-python (a
# Tauri resource, not a PyInstaller data file) needs one but this doesn't.
REQUIREMENTS_GPU = BACKEND_DIR / "requirements-gpu.txt"
REQUIREMENTS_CPU = BACKEND_DIR / "requirements-cpu.txt"


def _pip_lines(path: Path) -> list[str]:
    """Every real pip-installable line from a requirements file — skips
    comments, blanks, and -r includes (requirements-common.txt is already
    installed at CI freeze time, see hearth-backend.spec). Keeps version
    pins single-sourced from the requirements files rather than
    duplicating them here."""
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-r"):
            lines.append(line)
    return lines


def _packages_engine_importable(tier: TierConfig) -> bool:
    try:
        if tier.tts_engine in ("parler_gpu", "parler_cpu"):
            import parler_tts  # noqa: F401
        else:
            import onnxruntime  # noqa: F401
            import ttstokenizer  # noqa: F401
    except ImportError:
        return False
    return True


def _models_present(tier: TierConfig) -> bool:
    llm_path = Path(resolve_paths(tier)["llm_path"])
    embedding_path = EMBEDDING_MODELS_DIR / EMBEDDING_MODEL_FILE
    return llm_path.exists() and embedding_path.exists()


def detect_status() -> dict:
    """Used by GET /api/setup/status — reports hardware, the tier it maps
    to, and whether setup (packages + models) is already done.

    Completion is persisted in profile.db (`setup_state`) after the first
    successful run so later launches skip the Setup UI. The flag alone is
    not enough if model files are gone — that clears the flag and shows
    Setup again. Before the flag exists, we still fall back to the live
    packages+models check (dev installs that used scripts/setup.py)."""
    hw = detect_hardware()
    tier = pick_tier(hw)
    variant = hardware_variant.detect()
    models_ok = _models_present(tier)
    flagged = is_setup_complete()

    if flagged:
        if not models_ok:
            clear_setup_complete()
            complete = False
        else:
            # Trust the DB flag — don't re-probe TTS imports on every boot.
            complete = True
    else:
        complete = _packages_engine_importable(tier) and models_ok

    return {
        "hardware": hw,
        "tier": tier.tier,
        "tts_engine": tier.tts_engine,
        "gpu_vendor": variant.vendor,
        "complete": complete,
    }


def run_setup(progress: InstallProgress) -> None:
    """The actual setup flow — run in a background thread by
    POST /api/setup/start. Safe to re-run after a partial failure: package
    installs are --target-based (pip skips what's already satisfied) and
    model downloads skip anything already on disk (see app/setup/models.py)."""
    try:
        progress.set_step("detecting")
        hw = detect_hardware()
        tier = pick_tier(hw)
        variant = hardware_variant.detect()
        progress.append_log(
            f"tier {tier.tier} ({tier.tts_engine}), GPU vendor: {variant.vendor} "
            f"({variant.gpu_name or 'none'})"
        )

        progress.set_step("installing_packages")
        if tier.tts_engine in ("parler_gpu", "parler_cpu"):
            # torch (+torchaudio, which must match torch's own ABI — see
            # installer.py's docstring and this session's earlier
            # reproduction of the ABI-mismatch failure mode) come from the
            # hardware-matched index first; parler-tts's own unconstrained
            # transitive `torch` dependency is then satisfied by what's
            # already installed rather than re-resolved from plain PyPI —
            # verified this exact sequencing this session.
            install_packages(["torch", "torchaudio"], variant.torch_index_url, progress)
            # numpy pin in the SAME pip invocation as parler-tts: _pip_lines
            # skips `-r common`, and a separate numpy-only install is not
            # enough — parler's resolver can still pull an unbounded numpy
            # (and a second torch from plain PyPI) unless the pin is visible
            # in this resolve. See requirements-gpu.txt.
            install_packages(_pip_lines(REQUIREMENTS_GPU), None, progress)
        else:
            onnxruntime_line = variant.onnxruntime_package  # bare name, no version pin —
            # the GPU-variant onnxruntime packages (onnxruntime-gpu/-rocm/
            # -directml) have their own independent release cadence from
            # plain onnxruntime, so requirements-cpu.txt's `onnxruntime>=
            # 1.20.1` pin isn't safe to carry over onto a different
            # package name; let the resolver pick that package's own
            # latest instead.
            other_lines = [line for line in _pip_lines(REQUIREMENTS_CPU) if not line.startswith("onnxruntime")]
            install_packages([onnxruntime_line, *other_lines], None, progress)

        progress.set_step("downloading_models")
        models.download_models(tier, log=progress.append_log)
        # Do NOT set "done" here — main.py's /api/setup/start thread still
        # has to construct Pipeline() (STT/TTS/LLM warm-up) and call
        # mark_complete() before the UI may leave the Setup screen.
    except SetupError as exc:
        progress.set_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — surface any unexpected failure to the UI, don't crash the thread silently
        progress.set_error(f"unexpected error: {exc}")
