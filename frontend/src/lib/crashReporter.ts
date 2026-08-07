/** Capture frontend unhandled errors into the local crash-log queue.
 *
 * The backend buffers them on disk; nothing is uploaded until the user
 * opts in via CrashReportPrompt. Failures here are swallowed — reporting
 * must never break the app further.
 */
import { backendFetch } from "./backendFetch";

let installed = false;

function report(message: string, stack: string, component?: string): void {
  void backendFetch("/api/crash-reports", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: message.slice(0, 2000),
      stack: (stack || "").slice(0, 32000),
      component: component?.slice(0, 200),
    }),
  }).catch(() => {
    /* backend may be down mid-crash — local Python hooks still cover process deaths */
  });
}

export function installFrontendCrashReporter(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;

  window.addEventListener("error", (event) => {
    const message = event.message || String(event.error ?? "unknown error");
    const stack =
      (event.error instanceof Error && event.error.stack) ||
      `${event.filename || ""}:${event.lineno || 0}:${event.colno || 0}`;
    report(message, stack, "window.onerror");
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    const message =
      reason instanceof Error
        ? `${reason.name}: ${reason.message}`
        : `Unhandled rejection: ${String(reason)}`;
    const stack = reason instanceof Error ? reason.stack || "" : "";
    report(message, stack, "unhandledrejection");
  });
}
