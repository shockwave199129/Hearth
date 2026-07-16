import { useCallback, useEffect, useRef, useState } from "react";
import { friendlyFetchError } from "../lib/errors";
import { fetchWithTimeout, retryWithBackoff } from "../lib/backendFetch";

export interface SetupHardware {
  ram_gb: number;
  cpu_count: number;
  gpu_name: string | null;
  vram_gb: number;
}

export interface SetupStatus {
  hardware: SetupHardware;
  tier: string;
  tts_engine: string;
  gpu_vendor: string; // "nvidia" | "amd" | "none"
  complete: boolean;
}

export type SetupStep = "idle" | "detecting" | "installing_packages" | "downloading_models" | "done" | "error";

export interface SetupProgress {
  step: SetupStep;
  log_tail: string[];
  error: string | null;
}

const PROGRESS_POLL_MS = 1000;

/** Gates app boot the same way useProfile/useTierStatus do (see
 * lib/backendFetch.ts) — a packaged app's backend can take a while to come
 * up even before setup is a factor. Once `startSetup()` is called, this
 * also polls /api/setup/progress on a fixed interval until the backend
 * reports "done" or "error", since that's a long-running background
 * install+download, not a single request/response. */
export function useSetupStatus() {
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<SetupProgress | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const cancelledRef = { current: false };

    const attempt = async (): Promise<SetupStatus> => {
      const res = await fetchWithTimeout("/api/setup/status");
      if (!res.ok) throw new Error(`status ${res.status}`);
      return (await res.json()) as SetupStatus;
    };

    retryWithBackoff(attempt, cancelledRef)
      .then((data) => !cancelledRef.current && setStatus(data))
      .catch((err) => !cancelledRef.current && setError(friendlyFetchError(err, "useSetupStatus")));

    return () => {
      cancelledRef.current = true;
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Stop the interval on unmount — otherwise it'd keep firing against an
  // unmounted component.
  useEffect(() => stopPolling, [stopPolling]);

  const pollProgress = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(() => {
      fetch("/api/setup/progress")
        .then((res) => (res.ok ? (res.json() as Promise<SetupProgress>) : null))
        .then((data) => {
          if (!data) return;
          setProgress(data);
          if (data.step === "done") {
            stopPolling();
            setStatus((prev) => (prev ? { ...prev, complete: true } : prev));
          } else if (data.step === "error") {
            stopPolling();
          }
        })
        .catch(() => {
          // Transient poll failure — the next interval tick retries; no
          // need to surface a one-off network blip as a hard error here.
        });
    }, PROGRESS_POLL_MS);
  }, [stopPolling]);

  const startSetup = useCallback(async () => {
    setStarting(true);
    try {
      const res = await fetch("/api/setup/start", { method: "POST" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as SetupProgress;
      setProgress(data);
      if (data.step !== "done" && data.step !== "error") {
        pollProgress();
      }
    } finally {
      setStarting(false);
    }
  }, [pollProgress]);

  return { status, error, progress, starting, startSetup };
}
