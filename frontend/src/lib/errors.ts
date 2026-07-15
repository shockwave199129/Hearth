/** Every fetch-hook's catch block used to surface the raw error message
 * (often just `` `status ${res.status}` ``) straight to the user. This
 * logs the real error for debugging but always returns a fixed, friendly
 * string for display — no HTTP status codes in user-facing copy. */
export function friendlyFetchError(err: unknown, context: string): string {
  console.error(`[${context}]`, err);
  return "Couldn't reach the companion — is the backend running?";
}

/** Same idea for a specific user action (save/delete/play/...) rather than
 * a background load — logs the real error, always shows the given
 * action-specific fallback rather than a raw `` `status ${code}` `` message. */
export function friendlyActionError(err: unknown, context: string, fallback: string): string {
  console.error(`[${context}]`, err);
  return fallback;
}
