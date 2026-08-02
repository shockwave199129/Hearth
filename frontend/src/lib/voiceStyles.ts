/** Voice + speaking-style options, shared by Onboarding and Settings so the
 * two never drift. Mirrors backend/app/tts/voice_styles.py — the ids here
 * are what the API validates against, and the copy is deliberately about
 * how it sounds rather than about the TTS model behind it. */

export type PreferredVoice = "female" | "male";
export type VoiceStyleId = "grounded" | "gentle" | "bright";

export const VOICE_OPTIONS: { id: PreferredVoice; label: string }[] = [
  { id: "female", label: "Female voice" },
  { id: "male", label: "Male voice" },
];

export const VOICE_STYLE_OPTIONS: { id: VoiceStyleId; label: string; blurb: string }[] = [
  { id: "grounded", label: "Grounded & even", blurb: "Steady and warm, at a walking pace." },
  { id: "gentle", label: "Gentle & slow", blurb: "Softer and quieter, with more room to breathe." },
  { id: "bright", label: "Warm & lifted", blurb: "A little more lift, still calm." },
];

export const DEFAULT_VOICE: PreferredVoice = "female";
export const DEFAULT_VOICE_STYLE: VoiceStyleId = "grounded";
