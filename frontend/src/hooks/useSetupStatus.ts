import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch, fetchWithTimeout, retryWithBackoff } from "../lib/backendFetch";
import { friendlyActionError, friendlyFetchError } from "../lib/errors";

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

export type SetupStep =
  | "idle"
  | "detecting"
  | "installing_packages"
  | "downloading_models"
  | "starting_engines"
  | "done"
  | "error";

export interface SetupProgress {
  step: SetupStep;
  log_tail: string[];
  error: string | null;
}

const PROGRESS_POLL_MS = 1000;
const EMPTY_PROGRESS: SetupProgress = { step: "idle", log_tail: [], error: null };

const IN_FLIGHT_STEPS: ReadonlySet<SetupStep> = new Set([
  "detecting",
  "installing_packages",
  "downloading_models",
  "starting_engines",
]);

function isInFlight(step: SetupStep): boolean {
  return IN_FLIGHT_STEPS.has(step);
}

/** Gates app boot the same way useProfile/useTierStatus do (see
 * lib/backendFetch.ts) — a packaged app's backend can take a while to come
 * up even before setup is a factor. Once `startSetup()` is called, this
 * also polls /api/setup/progress on a fixed interval until the backend
 * reports "done" or "error", since that's a long-running background
 * install+download+Pipeline warm-up, not a single request/response. */
export function useSetupStatus() {
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<SetupProgress | null>(null);
  const [starting, setStarting] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Stop the interval on unmount — otherwise it'd keep firing against an
  // unmounted component.
  useEffect(() => stopPolling, [stopPolling]);

  const confirmCompleteFromStatus = useCallback(() => {
    // Progress "done" is only set after Pipeline() + mark_complete() — still
    // re-fetch /api/setup/status so we don't leave Setup on a stale local
    // complete flip before the DB flag exists.
    void backendFetch("/api/setup/status")
      .then((res) => (res.ok ? (res.json() as Promise<SetupStatus>) : null))
      .then((data) => {
        if (data?.complete) {
          setStatus(data);
          setError(null);
          return;
        }
        setProgress((prev) => ({
          step: "error",
          log_tail: prev?.log_tail ?? [],
          error: "Setup finished but the app isn't ready yet. Please retry.",
        }));
      })
      .catch(() => {
        // Status re-fetch failed after progress already said done (backend
        // marked complete before emitting done) — trust done and proceed.
        setStatus((prev) => (prev ? { ...prev, complete: true } : prev));
        setError(null);
      });
  }, []);

  const pollProgress = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(() => {
      backendFetch("/api/setup/progress")
        .then((res) => (res.ok ? (res.json() as Promise<SetupProgress>) : null))
        .then((data) => {
          if (!data) return;
          setProgress(data);
          if (data.step === "done") {
            stopPolling();
            confirmCompleteFromStatus();
          } else if (data.step === "error") {
            stopPolling();
          }
        })
        .catch(() => {
          // Transient poll failure — the next interval tick retries; no
          // need to surface a one-off network blip as a hard error here.
        });
    }, PROGRESS_POLL_MS);
  }, [stopPolling, confirmCompleteFromStatus]);

  // Desktop shell dispatches this when the backend child fails to spawn
  // (see desktop/src-tauri/src/main.rs) — surface it before the status
  // retry loop burns ~34s on a backend that will never appear.
  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<string>).detail;
      setError(
        typeof detail === "string" && detail.trim()
          ? detail
          : "Couldn't start the companion backend.",
      );
    };
    window.addEventListener("backend-spawn-failed", handler);
    return () => window.removeEventListener("backend-spawn-failed", handler);
  }, []);

  useEffect(() => {
    const cancelledRef = { current: false };

    const attempt = async (): Promise<SetupStatus> => {
      const res = await fetchWithTimeout("/api/setup/status");
      if (!res.ok) throw new Error(`status ${res.status}`);
      return (await res.json()) as SetupStatus;
    };

    setError(null);
    setStatus(null);

    retryWithBackoff(attempt, cancelledRef)
      .then(async (data) => {
        if (cancelledRef.current) return;
        setStatus(data);
        setError(null);
        // Resume an in-flight setup (e.g. user refreshed mid-install) —
        // POST /start is idempotent, but we only need to poll progress.
        try {
          const res = await backendFetch("/api/setup/progress");
          if (!res.ok || cancelledRef.current) return;
          const prog = (await res.json()) as SetupProgress;
          if (cancelledRef.current) return;
          setProgress(prog);
          if (isInFlight(prog.step)) {
            pollProgress();
          } else if (prog.step === "done" && !data.complete) {
            confirmCompleteFromStatus();
          }
        } catch {
          // Progress probe is best-effort on boot.
        }
      })
      .catch((err) => !cancelledRef.current && setError(friendlyFetchError(err, "useSetupStatus")));

    return () => {
      cancelledRef.current = true;
    };
  }, [retryToken, pollProgress, confirmCompleteFromStatus]);

  const retryStatus = useCallback(() => {
    stopPolling();
    setProgress(null);
    setRetryToken((t) => t + 1);
  }, [stopPolling]);

  const startSetup = useCallback(async () => {
    // Clear any prior failure so Retry doesn't leave a stale error/log on
    // screen while the new run is starting.
    setProgress(EMPTY_PROGRESS);
    setStarting(true);
    try {
      const res = await backendFetch("/api/setup/start", { method: "POST" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as SetupProgress;
      setProgress(data);
      if (data.step === "done") {
        confirmCompleteFromStatus();
      } else if (data.step !== "error") {
        pollProgress();
      }
    } catch (err) {
      setProgress({
        step: "error",
        log_tail: [],
        error: friendlyActionError(err, "startSetup", "Couldn't start setup — is the backend still running?"),
      });
    } finally {
      setStarting(false);
    }
  }, [pollProgress, confirmCompleteFromStatus]);

  return { status, error, progress, starting, startSetup, retryStatus };
}
