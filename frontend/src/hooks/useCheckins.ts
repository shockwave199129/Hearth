import { useEffect, useState } from "react";
import { backendFetch } from "../lib/backendFetch";
import { friendlyFetchError } from "../lib/errors";

export interface CheckinStatus {
  last_checkin_at: string | null;
  days_since_last_checkin: number | null;
}

interface UseCheckinsResult {
  status: CheckinStatus | null;
  loading: boolean;
  error: string | null;
}

/** Backs Settings → Check-ins. The companion decides check-in timing itself
 * (system prompt state + mark_checkin tool, checkin/tools.py) — this is a
 * read-only view of that state, not a control. See project-plan.md §8. */
export function useCheckins(): UseCheckinsResult {
  const [status, setStatus] = useState<CheckinStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    backendFetch("/api/checkin")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<CheckinStatus>;
      })
      .then((data) => !cancelled && setStatus(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useCheckins")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return { status, loading, error };
}
