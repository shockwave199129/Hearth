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

/** Resolve an API path (e.g. `/api/setup/status`) for the current runtime. */
export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${BACKEND_HTTP}${path}`;
}

/** WebSocket URL for `/ws` — same packaged-vs-dev split as `apiUrl`. */
export function wsUrl(path = "/ws"): string {
  if (import.meta.env.DEV) {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${window.location.host}${path}`;
  }
  return `ws://127.0.0.1:48173${path}`;
}
