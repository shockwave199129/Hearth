import { useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";

export interface SafetyStatus {
  recent_crisis_events: number;
  last_escalation_at: string | null;
}

interface UseSafetyStatusResult {
  status: SafetyStatus | null;
  loading: boolean;
  error: string | null;
}

/** Backs Settings → Safety. The crisis/escalation path (safety/crisis_detector.py,
 * safety/escalation.py) runs entirely in the background — this is a
 * read-only transparency view, same "never actually hidden" principle as
 * Memory/Skills/Check-ins. See project-plan.md §9. */
export function useSafetyStatus(): UseSafetyStatusResult {
  const [status, setStatus] = useState<SafetyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/safety/status")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<SafetyStatus>;
      })
      .then((data) => !cancelled && setStatus(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useSafetyStatus")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return { status, loading, error };
}
