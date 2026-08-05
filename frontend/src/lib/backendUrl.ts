/** Backend HTTP origin for the packaged Tauri app.
 *
 * Dev (`import.meta.env.DEV`): empty — Vite proxies `/api` and `/ws` to
 * `127.0.0.1:48173` (see vite.config.ts).
 *
 * Production: the UI is served from Tauri's own origin (`https://tauri.localhost`
 * etc.), NOT from the FastAPI process. Relative `/api/...` fetches would hit
 * Tauri and 404, which is why a packaged install showed "Couldn't reach the
 * companion" instead of the setup screen. Absolute URLs go to the backend
 * child process Tauri spawns (desktop/src-tauri/src/main.rs). */
const BACKEND_HTTP = import.meta.env.DEV ? "" : "http://127.0.0.1:48173";

export const API_TOKEN_HEADER = "X-Hearth-Token";

declare global {
  interface Window {
    __HEARTH_API_TOKEN__?: string;
  }
}

/** Per-launch shared secret the Tauri shell generates and gives to both the
 * backend child process and this webview, so other local processes can't
 * read the journal over `127.0.0.1:48173` (see desktop/src-tauri/src/main.rs).
 * Undefined when the UI runs in a plain browser against a manually started
 * backend — that backend has no token either, and skips the check. */
export function apiToken(): string | undefined {
  return window.__HEARTH_API_TOKEN__;
}

/** Resolve an API path (e.g. `/api/setup/status`) for the current runtime. */
export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${BACKEND_HTTP}${path}`;
}

/** WebSocket URL for `/ws` — same packaged-vs-dev split as `apiUrl`.
 * The token rides as a query parameter because `new WebSocket()` cannot set
 * request headers; it never leaves the loopback interface. */
export function wsUrl(path = "/ws"): string {
  const base = import.meta.env.DEV
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${path}`
    : `ws://127.0.0.1:48173${path}`;
  const token = apiToken();
  return token ? `${base}?token=${encodeURIComponent(token)}` : base;
}
