/** Device-local microphone probe + voice-input preference.
 *
 * Hardware presence is enumerated from the browser (same API inside Tauri's
 * webview). The on/off choice is a UI preference, not profile data — it
 * should follow this machine, not the companion identity. */

export type MicHardware = "unknown" | "present" | "absent";
export type MicPermission = "unknown" | "prompt" | "granted" | "denied";

export interface MicrophoneStatus {
  hardware: MicHardware;
  permission: MicPermission;
}

const PREFERENCE_KEY = "companion:voiceInput";

type Listener = () => void;
const listeners = new Set<Listener>();

function notify(): void {
  listeners.forEach((listener) => listener());
}

export function subscribeVoiceInputPreference(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Default on — turning voice off is an explicit choice. */
export function isVoiceInputPreferred(): boolean {
  return localStorage.getItem(PREFERENCE_KEY) !== "false";
}

export function setVoiceInputPreferred(enabled: boolean): void {
  localStorage.setItem(PREFERENCE_KEY, String(enabled));
  notify();
}

export async function probeMicrophone(): Promise<MicrophoneStatus> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
    return { hardware: "absent", permission: "unknown" };
  }

  let permission: MicPermission = "unknown";
  try {
    const status = await navigator.permissions.query({ name: "microphone" as PermissionName });
    if (status.state === "granted" || status.state === "denied" || status.state === "prompt") {
      permission = status.state;
    }
  } catch {
    // Safari / some WebKit builds don't expose microphone in Permissions.
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    return { hardware: inputs.length > 0 ? "present" : "absent", permission };
  } catch {
    return { hardware: "unknown", permission };
  }
}

export function voiceInputUsable(status: MicrophoneStatus, preferred: boolean): boolean {
  return preferred && status.hardware !== "absent" && status.permission !== "denied";
}
