import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./Chat.css";
import { VoiceOrb, type OrbState } from "../components/VoiceOrb";
import { TranscriptLog } from "../components/TranscriptLog";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { useCompanionSocket, type SocketStatus } from "../hooks/useCompanionSocket";
import { useMicrophone } from "../hooks/useMicrophone";
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
  const { hardware, permission, preferred, enabled: voiceInputEnabled, setPreferred: setVoiceInputPreferred } =
    useMicrophone();
  const [draft, setDraft] = useState("");
  const textInputRef = useRef<HTMLInputElement>(null);

  const connected = status === "open";
  const statusMessage = socketStatusMessage(status);
  const textOnly = hardware === "absent" || !voiceInputEnabled;

  useEffect(() => {
    if (!voiceInputEnabled && recorderState === "listening") stop();
  }, [voiceInputEnabled, recorderState, stop]);

  useEffect(() => {
    if (textOnly) textInputRef.current?.focus();
  }, [textOnly]);

  const orbState: OrbState =
    recorderState === "listening" ? "listening" : isThinking ? "thinking" : isSpeaking ? "speaking" : "idle";
  const orbAmplitude = recorderState === "listening" ? amplitude : isSpeaking ? speakingAmplitude : 0;
  const busy = orbState === "thinking" || orbState === "speaking";
  const micReady = voiceInputEnabled && permission !== "denied";

  const orbLabel =
    orbState !== "idle"
      ? undefined
      : hardware === "absent"
        ? "No microphone — type below"
        : permission === "denied"
          ? "Microphone blocked"
          : voiceInputEnabled
            ? undefined
            : "Voice input off — type below";

  const handleOrbClick = () => {
    if (!connected || !micReady) return;
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
    <div className={`chat-page${textOnly ? " chat-page--text-only" : ""}`}>
      <div className="chat-page__stage">
        <VoiceOrb
          state={orbState}
          amplitude={orbAmplitude}
          onClick={handleOrbClick}
          disabled={busy || !connected || !micReady}
          label={orbLabel}
        />
        {hardware === "present" && (
          <div className="chat-page__voice-toggle" role="group" aria-label="Voice input">
            <span className="chat-page__voice-toggle-label">Voice input</span>
            <div className="chat-page__voice-toggle-controls">
              {[
                { value: true, label: "On" },
                { value: false, label: "Off" },
              ].map((option) => (
                <button
                  key={String(option.value)}
                  type="button"
                  className={`chat-page__voice-option${preferred === option.value ? " chat-page__voice-option--active" : ""}`}
                  onClick={() => option.value !== preferred && setVoiceInputPreferred(option.value)}
                  disabled={permission === "denied"}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {statusMessage && (
          <p className="chat-page__status" aria-live="polite">
            {statusMessage}
          </p>
        )}
        {hardware === "absent" && (
          <p className="chat-page__status">
            No microphone connected. Type below to chat — voice is optional.
          </p>
        )}
        {permission === "denied" && (
          <p className="chat-page__status">
            Microphone access is blocked. Allow it in your browser or system settings to talk out loud.
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
          ref={textInputRef}
          type="text"
          className="chat-page__text-input"
          placeholder={
            !connected
              ? "Waiting for connection…"
              : textOnly
                ? "Type a message…"
                : "Or type instead…"
          }
          aria-label="Message"
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
