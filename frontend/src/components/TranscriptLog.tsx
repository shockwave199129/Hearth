import { useEffect, useRef, useState } from "react";
import "./TranscriptLog.css";
import type { Turn } from "../hooks/useCompanionSocket";
import { backendFetch } from "../lib/backendFetch";

interface TranscriptLogProps {
  turns: Turn[];
  companionName: string;
}

export function TranscriptLog({ turns, companionName }: TranscriptLogProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  const replay = async (turnDbId: number) => {
    setPlayingId(turnDbId);
    try {
      const res = await backendFetch(`/api/chat_history/${turnDbId}/audio`);
      if (!res.ok) return;
      const url = URL.createObjectURL(await res.blob());
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      await audio.play();
    } finally {
      setPlayingId(null);
    }
  };

  if (turns.length === 0) {
    return (
      <div className="transcript-log transcript-log--empty">
        <p>Nothing said yet — press the orb whenever you're ready.</p>
      </div>
    );
  }

  return (
    <div className="transcript-log">
      {turns.map((turn) => (
        <div className="transcript-turn" key={turn.id}>
          <p className="transcript-turn__line transcript-turn__line--user">{turn.transcript}</p>
          <p className="transcript-turn__line transcript-turn__line--companion">
            <span className="transcript-turn__speaker">{companionName}</span>
            {turn.replyText}
            <button
              type="button"
              className="transcript-turn__replay"
              onClick={() => replay(turn.turnDbId)}
              disabled={playingId === turn.turnDbId}
              aria-label="Replay this reply"
            >
              {playingId === turn.turnDbId ? "…" : "🔊"}
            </button>
          </p>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
