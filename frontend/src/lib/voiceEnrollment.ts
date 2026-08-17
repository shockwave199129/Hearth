/** Voice-enrollment client for POST /api/voice/enrollment.
 *
 * Enrollment stores biometric data (see docs/compliance.md), so the UI that
 * uses this must take explicit consent first — `Settings.tsx` does, and this
 * module deliberately provides no way to enroll without passing recordings a
 * caller had to gather.
 *
 * Wire format matches backend/app/api/voice.py `_decode_samples`: a
 * little-endian uint32 sample count, then one uint32 length per sample, then
 * the float32 PCM runs back to back. Chosen over multipart/JSON so the
 * existing recorder's Float32Array goes out as-is — no base64 inflation and
 * no second encoder to keep in step with the websocket's format. */

import { backendFetch } from "./backendFetch";

export type EnrollmentStatus = {
  model_available: boolean;
  vad_available: boolean;
  required_samples: number;
  min_seconds_per_sample: number;
  /** The single reviewable copy of the consent wording, served from
   * app/config.py so the text shown is exactly the text whose version gets
   * recorded against the profile. Never hardcode it here. */
  consent_text: string;
  consent_recorded: boolean;
  /** False when the wording changed since the user agreed — treated as no
   * consent, so they are asked again rather than inheriting agreement to
   * different text. */
  consent_current: boolean;
  consented_at?: string | null;
  retention_days: number;
  /** When the stored voiceprint is due for automatic destruction. */
  expires_at?: string | null;
  enrolled: boolean;
  sample_count?: number;
  model_id?: string;
  enrolled_at?: string;
  updated_at?: string;
  dimensions?: number;
};

export function encodeSamples(samples: Float32Array[]): ArrayBuffer {
  const total = samples.reduce((sum, s) => sum + s.length, 0);
  const header = 4 + 4 * samples.length;
  const buffer = new ArrayBuffer(header + total * 4);
  const counts = new DataView(buffer);
  counts.setUint32(0, samples.length, true);
  samples.forEach((sample, i) => counts.setUint32(4 + 4 * i, sample.length, true));
  const audio = new Float32Array(buffer, header);
  let offset = 0;
  for (const sample of samples) {
    audio.set(sample, offset);
    offset += sample.length;
  }
  return buffer;
}

export async function getEnrollmentStatus(): Promise<EnrollmentStatus> {
  const response = await backendFetch("/api/voice/enrollment");
  if (!response.ok) throw new Error(`status ${response.status}`);
  return (await response.json()) as EnrollmentStatus;
}

/** Record agreement to the current consent wording. Collects nothing on its
 * own — enrollment stays a separate, explicit act afterwards. */
export async function recordVoiceConsent(): Promise<void> {
  const response = await backendFetch("/api/voice/consent", { method: "POST" });
  if (!response.ok) throw new Error(`status ${response.status}`);
}

export async function enrollVoice(samples: Float32Array[]): Promise<EnrollmentStatus> {
  const response = await backendFetch("/api/voice/enrollment", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: encodeSamples(samples),
  });
  if (!response.ok) {
    // The server returns 422 with an actionable message for the common
    // failures (samples too short, more than one voice). Surface it verbatim
    // rather than a generic error — it tells the user what to do differently.
    let detail = `status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status */
    }
    throw new Error(detail);
  }
  return (await response.json()) as EnrollmentStatus;
}

export async function forgetVoice(): Promise<void> {
  const response = await backendFetch("/api/voice/enrollment", { method: "DELETE" });
  if (!response.ok) throw new Error(`status ${response.status}`);
}
