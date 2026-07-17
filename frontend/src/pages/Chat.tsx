import { useCallback, useMemo, useState } from "react";
import "./Chat.css";
import { VoiceOrb, type OrbState } from "../components/VoiceOrb";
import { TranscriptLog } from "../components/TranscriptLog";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { useCompanionSocket, type SocketStatus } from "../hooks/useCompanionSocket";
import { useProfile } from "../hooks/useProfile";
import { wsUrl } from "../lib/backendUrl";

function socketStatusMessage(status: SocketStatus): string | null {
  switch (status) {
    case "connecting":
      return "Connecting to your companion…";
    case "reconnecting":
      return "Connection lost — reconnecting…";
    case "closed":
    case "error":
      return "Offline — trying again…";
    default:
      return null;
  }
}

export function Chat() {
  const { profile } = useProfile();
  const { status, turns, isThinking, isSpeaking, speakingAmplitude, hasMoreHistory, loadingOlder, loadOlderHistory, sendUtterance, sendText } =
    useCompanionSocket(useMemo(() => wsUrl(), []));
  const onUtterance = useCallback((audio: Float32Array) => sendUtterance(audio), [sendUtterance]);
  const { state: recorderState, amplitude, error, start, stop } = useAudioRecorder(onUtterance);
  const [draft, setDraft] = useState("");

  const connected = status === "open";
  const statusMessage = socketStatusMessage(status);

  const orbState: OrbState =
    recorderState === "listening" ? "listening" : isThinking ? "thinking" : isSpeaking ? "speaking" : "idle";
  const orbAmplitude = recorderState === "listening" ? amplitude : isSpeaking ? speakingAmplitude : 0;
  const busy = orbState === "thinking" || orbState === "speaking";

  const handleOrbClick = () => {
    if (!connected) return;
    if (orbState === "idle") void start();
    else if (orbState === "listening") stop();
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy || !connected) return;
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
          disabled={busy || !connected}
        />
        {statusMessage && (
          <p className="chat-page__status" aria-live="polite">
            {statusMessage}
          </p>
        )}
        {error && <p className="chat-page__error">{error}</p>}
      </div>
      <TranscriptLog
        turns={turns}
        companionName={profile?.companion_name ?? "Companion"}
        hasMoreHistory={hasMoreHistory}
        loadingOlder={loadingOlder}
        onLoadOlder={loadOlderHistory}
      />
      <form className="chat-page__text-form" onSubmit={handleTextSubmit}>
        <input
          type="text"
          className="chat-page__text-input"
          placeholder={connected ? "Or type instead…" : "Waiting for connection…"}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={recorderState === "listening" || !connected}
        />
        <button
          type="submit"
          className="chat-page__text-send"
          disabled={busy || !connected || !draft.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}
