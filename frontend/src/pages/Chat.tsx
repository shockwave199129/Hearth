import { useCallback, useMemo, useState } from "react";
import "./Chat.css";
import { VoiceOrb, type OrbState } from "../components/VoiceOrb";
import { TranscriptLog } from "../components/TranscriptLog";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { useCompanionSocket } from "../hooks/useCompanionSocket";
import { useProfile } from "../hooks/useProfile";

function socketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws`;
}

export function Chat() {
  const { profile } = useProfile();
  const { turns, isThinking, isSpeaking, speakingAmplitude, sendUtterance, sendText } = useCompanionSocket(
    useMemo(socketUrl, []),
  );
  const onUtterance = useCallback((audio: Float32Array) => sendUtterance(audio), [sendUtterance]);
  const { state: recorderState, amplitude, error, start, stop } = useAudioRecorder(onUtterance);
  const [draft, setDraft] = useState("");

  const orbState: OrbState =
    recorderState === "listening" ? "listening" : isThinking ? "thinking" : isSpeaking ? "speaking" : "idle";
  const orbAmplitude = recorderState === "listening" ? amplitude : isSpeaking ? speakingAmplitude : 0;
  const busy = orbState === "thinking" || orbState === "speaking";

  const handleOrbClick = () => {
    if (orbState === "idle") void start();
    else if (orbState === "listening") stop();
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    sendText(text);
    setDraft("");
  };

  return (
    <div className="chat-page">
      <div className="chat-page__stage">
        <VoiceOrb
          state={orbState}
          amplitude={orbAmplitude}
          onClick={handleOrbClick}
          disabled={busy}
        />
        {error && <p className="chat-page__error">{error}</p>}
      </div>
      <TranscriptLog turns={turns} companionName={profile?.companion_name ?? "Companion"} />
      <form className="chat-page__text-form" onSubmit={handleTextSubmit}>
        <input
          type="text"
          className="chat-page__text-input"
          placeholder="Or type instead…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={recorderState === "listening"}
        />
        <button type="submit" className="chat-page__text-send" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
