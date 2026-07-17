/** Retry-with-backoff for the initial backend checks that gate app boot
 * (useProfile, useTierStatus, useSetupStatus) — a packaged app's backend can
 * take anywhere from a couple seconds to over a minute to come up (hardware
 * detection, loading a local LLM into memory, starting STT/TTS), and Tauri's
 * main.rs shows the window immediately rather than blocking on backend
 * readiness (see desktop/src-tauri/src/main.rs). Without this, a fetch that
 * fires before the backend is listening would fail once and permanently show
 * an error, instead of the backend just still being mid-boot. */

import { apiUrl } from "./backendUrl";

/** All frontend→backend HTTP goes through here so packaged builds hit
 * `127.0.0.1:48173` instead of Tauri's asset origin (see backendUrl.ts). */
export function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), init);
}

/** Per-attempt timeout — a backend that's mid-boot may have its port bound
 * but be blocked inside FastAPI's startup event (building the whole model
 * pipeline) for a long time; this keeps one retry attempt from hanging for
 * the full boot duration instead of moving on to the next backoff step. */
export async function fetchWithTimeout(path: string, timeoutMs = 3000): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await backendFetch(path, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ~10 attempts, ~34s of retrying total — comfortably past typical local LLM
// load times without retrying forever; after this, a real error is shown.
const RETRY_DELAYS_MS = [500, 1000, 2000, 3000, 5000, 5000, 5000, 5000, 5000, 5000];

/** Calls `attempt()` repeatedly on the backoff schedule above until it
 * resolves without throwing, or the schedule is exhausted (rethrows the
 * last error). `cancelledRef` should be a ref whose `.current` becomes
 * true on unmount, so a pending retry never sets state after unmount. */
export async function retryWithBackoff<T>(attempt: () => Promise<T>, cancelledRef: { current: boolean }): Promise<T> {
  for (let i = 0; i < RETRY_DELAYS_MS.length; i++) {
    try {
      return await attempt();
    } catch (err) {
      if (cancelledRef.current) throw err;
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[i]));
    }
  }
  return attempt();
}
