import { useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";
import { fetchWithTimeout, retryWithBackoff } from "../lib/backendFetch";

export interface TierStatus {
  tier: string;
  llm_gguf: string;
  stt_model: string;
  tts_engine: string;
  hardware: {
    ram_gb: number;
    gpu_name: string | null;
    vram_gb: number;
  };
}

/** Retries with backoff (see lib/backendFetch.ts) rather than failing on
 * the first attempt — a packaged app's backend can take a while to come up
 * (local LLM load, etc.), and this is what backs the TierBadge, which
 * should read "detecting hardware" throughout that window, not "hardware
 * check failed" the moment the very first request loses the race. */
export function useTierStatus() {
  const [status, setStatus] = useState<TierStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const cancelledRef = { current: false };

    const attempt = async (): Promise<TierStatus> => {
      const res = await fetchWithTimeout("/api/status");
      if (!res.ok) throw new Error(`status ${res.status}`);
      return res.json();
    };

    retryWithBackoff(attempt, cancelledRef)
      .then((data) => !cancelledRef.current && setStatus(data))
      .catch((err) => !cancelledRef.current && setError(friendlyFetchError(err, "useTierStatus")));

    return () => {
      cancelledRef.current = true;
    };
  }, []);

  return { status, error };
}
