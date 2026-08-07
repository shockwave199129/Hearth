import { useAlert } from "../lib/alerts";
import { useCrashReports } from "../hooks/useCrashReports";
import "./CrashReportPrompt.css";

/** Shown when a previous launch left a buffered crash report on disk.
 * Send needs internet and uploads only after an explicit Yes. */
export function CrashReportPrompt() {
  const { report, busy, error, send, dismiss } = useCrashReports();
  const { showAlert } = useAlert();

  if (!report) return null;

  const when = report.created_at
    ? new Date(report.created_at).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : null;

  const handleSend = async () => {
    const ok = await send();
    if (ok) {
      showAlert({ type: "success", message: "Thanks — crash report sent." });
    }
  };

  return (
    <div className="crash-prompt" role="dialog" aria-modal="true" aria-labelledby="crash-prompt-title">
      <div className="crash-prompt__card">
        <h2 id="crash-prompt-title">Send a crash report?</h2>
        <p className="crash-prompt__body">
          Hearth closed unexpectedly{when ? ` (${when})` : ""}. A diagnostic log is saved on this
          device. Sending it needs an internet connection and uploads only the stack trace, app
          version, and OS details — never your conversations or memories.
        </p>
        {report.message && (
          <p className="crash-prompt__detail" title={report.message}>
            {report.message}
          </p>
        )}
        {error && <p className="crash-prompt__error">{error}</p>}
        <div className="crash-prompt__actions">
          <button type="button" className="crash-prompt__btn crash-prompt__btn--ghost" onClick={() => void dismiss()} disabled={busy}>
            Don&apos;t send
          </button>
          <button type="button" className="crash-prompt__btn crash-prompt__btn--primary" onClick={() => void handleSend()} disabled={busy}>
            {busy ? "Working…" : "Send report"}
          </button>
        </div>
      </div>
    </div>
  );
}
