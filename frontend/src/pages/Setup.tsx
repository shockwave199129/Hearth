import "./Setup.css";
import type { SetupProgress, SetupStatus } from "../hooks/useSetupStatus";

const STEP_LABELS: Record<string, string> = {
  idle: "Ready to set up",
  detecting: "Detecting your hardware…",
  installing_packages: "Installing the right components for your machine…",
  downloading_models: "Downloading models…",
  done: "All set.",
  error: "Setup hit a problem.",
};

interface SetupProps {
  status: SetupStatus | null;
  statusError: string | null;
  progress: SetupProgress | null;
  starting: boolean;
  onStart: () => void;
}

function gpuSummary(status: SetupStatus): string {
  if (status.gpu_vendor === "none" || !status.hardware.gpu_name) {
    return "No GPU detected — using CPU.";
  }
  return `${status.hardware.gpu_name} (${status.hardware.vram_gb} GB VRAM)`;
}

export function Setup({ status, statusError, progress, starting, onStart }: SetupProps) {
  const step = progress?.step ?? "idle";
  const isRunning = step !== "idle" && step !== "done" && step !== "error";

  return (
    <div className="setup">
      <div className="setup__card">
        <h1>Setting things up</h1>
        <p className="setup__hint">
          This machine's hardware determines which version of a few components to install — done once,
          here, rather than guessed ahead of time.
        </p>

        {statusError && <p className="setup__error">{statusError}</p>}

        {status && (
          <div className="setup__hardware">
            <div className="setup__hardware-row">
              <span>RAM</span>
              <span>{status.hardware.ram_gb} GB</span>
            </div>
            <div className="setup__hardware-row">
              <span>GPU</span>
              <span>{gpuSummary(status)}</span>
            </div>
            <div className="setup__hardware-row">
              <span>Detected tier</span>
              <span>{status.tier}</span>
            </div>
          </div>
        )}

        {step !== "idle" && (
          <div className="setup__progress">
            <p className="setup__progress-label" aria-live="polite">
              {STEP_LABELS[step] ?? step}
            </p>
            {progress?.error && <p className="setup__error">{progress.error}</p>}
          </div>
        )}

        <button
          type="button"
          className="setup__button"
          onClick={onStart}
          disabled={starting || isRunning || !status}
        >
          {step === "error" ? "Retry setup" : isRunning ? "Setting up…" : "Start setup"}
        </button>
      </div>
    </div>
  );
}
