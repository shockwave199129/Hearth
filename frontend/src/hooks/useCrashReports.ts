import { useCallback, useEffect, useState } from "react";
import { backendFetch } from "../lib/backendFetch";
import { friendlyActionError } from "../lib/errors";

export interface PendingCrashReport {
  id: string;
  created_at: string | null;
  source: string | null;
  message: string;
}

interface UseCrashReportsResult {
  report: PendingCrashReport | null;
  busy: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  send: () => Promise<boolean>;
  dismiss: () => Promise<void>;
}

export function useCrashReports(): UseCrashReportsResult {
  const [report, setReport] = useState<PendingCrashReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await backendFetch("/api/crash-reports/pending");
      if (!res.ok) return;
      const data = (await res.json()) as { reports: PendingCrashReport[] };
      setReport(data.reports[0] ?? null);
    } catch {
      /* backend still starting — try again on next mount / focus */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const send = useCallback(async () => {
    if (!report) return false;
    setBusy(true);
    setError(null);
    try {
      const res = await backendFetch(`/api/crash-reports/${report.id}/send`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : res.status === 503
              ? "No internet connection — connect and try again."
              : "Couldn't upload the crash report.";
        setError(detail);
        return false;
      }
      setReport(null);
      await refresh();
      return true;
    } catch (err) {
      setError(friendlyActionError(err, "CrashReport.send", "Couldn't upload the crash report."));
      return false;
    } finally {
      setBusy(false);
    }
  }, [report, refresh]);

  const dismiss = useCallback(async () => {
    if (!report) return;
    setBusy(true);
    setError(null);
    try {
      await backendFetch(`/api/crash-reports/${report.id}/dismiss`, { method: "POST" });
      setReport(null);
      await refresh();
    } catch (err) {
      setError(friendlyActionError(err, "CrashReport.dismiss", "Couldn't dismiss the report."));
    } finally {
      setBusy(false);
    }
  }, [report, refresh]);

  return { report, busy, error, refresh, send, dismiss };
}
