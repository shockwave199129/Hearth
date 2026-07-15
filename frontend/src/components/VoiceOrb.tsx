import { CSSProperties } from "react";
import "./VoiceOrb.css";

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

interface VoiceOrbProps {
  state: OrbState;
  amplitude?: number;
  onClick?: () => void;
  disabled?: boolean;
}

const LABELS: Record<OrbState, string> = {
  idle: "Tap to talk",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export function VoiceOrb({ state, amplitude = 0, onClick, disabled }: VoiceOrbProps) {
  const reactiveScale = 1 + Math.min(amplitude, 1) * 0.16;
  const style = { "--reactive-scale": reactiveScale } as CSSProperties;

  return (
    <div className="voice-orb-wrap">
      <button
        type="button"
        className={`voice-orb voice-orb--${state}`}
        style={style}
        onClick={onClick}
        disabled={disabled}
        aria-label={LABELS[state]}
      >
        <span className="voice-orb__halo" aria-hidden />
        <span className="voice-orb__core" aria-hidden />
        <span className="voice-orb__ring" aria-hidden />
      </button>
      <p className="voice-orb__label">{LABELS[state]}</p>
    </div>
  );
}
