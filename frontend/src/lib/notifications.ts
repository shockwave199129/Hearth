/** Thin wrapper around the standard Web Notification API — local OS-level
 * notifications only, no push service, no external network call. Works
 * identically in the Vite dev browser and inside Tauri's native webview
 * (WRY uses each platform's real browser engine, which implements this API
 * natively). See lib/alerts.tsx for how this is actually invoked. */

export function isSupported(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

export function getPermission(): NotificationPermission | "unsupported" {
  if (!isSupported()) return "unsupported";
  return Notification.permission;
}

export async function requestPermission(): Promise<NotificationPermission | "unsupported"> {
  if (!isSupported()) return "unsupported";
  return Notification.requestPermission();
}

/** No-ops unless supported, already granted, and the document is actually
 * hidden — a safety net even though callers (lib/alerts.tsx) already check
 * visibility, so this module is safe to call directly too. */
export function notify(title: string, body: string): void {
  if (!isSupported()) return;
  if (Notification.permission !== "granted") return;
  if (!isEnabledPreference()) return;
  if (document.visibilityState !== "hidden") return;
  new Notification(title, { body });
}

const PREFERENCE_KEY = "companion:notifications";

/** Local UI preference (not backend/profile data) — mirrors lib/theme.ts's
 * localStorage pattern. Distinct from the actual browser permission: a user
 * could grant permission then later turn this off without revoking it. */
export function isEnabledPreference(): boolean {
  return localStorage.getItem(PREFERENCE_KEY) === "true";
}

export function setEnabledPreference(enabled: boolean): void {
  localStorage.setItem(PREFERENCE_KEY, String(enabled));
}
