import { useEffect, useLayoutEffect, useRef, useState } from "react";
import "./TranscriptLog.css";
import type { Turn } from "../hooks/useCompanionSocket";
import { backendFetch } from "../lib/backendFetch";

interface TranscriptLogProps {
  turns: Turn[];
  companionName: string;
  hasMoreHistory?: boolean;
  loadingOlder?: boolean;
  onLoadOlder?: () => void | Promise<void>;
}

export function TranscriptLog({
  turns,
  companionName,
  hasMoreHistory = false,
  loadingOlder = false,
  onLoadOlder,
}: TranscriptLogProps) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const prevTurnCountRef = useRef(0);
  const pendingScrollRestoreRef = useRef<{ height: number; top: number } | null>(null);
  const stickToBottomRef = useRef(true);

  // After prepending older turns, keep the same messages under the viewport.
  useLayoutEffect(() => {
    const el = scrollerRef.current;
    const pending = pendingScrollRestoreRef.current;
    if (!el || !pending) return;
    const delta = el.scrollHeight - pending.height;
    el.scrollTop = pending.top + delta;
    pendingScrollRestoreRef.current = null;
  }, [turns]);

  useEffect(() => {
    const grewAtEnd = turns.length > prevTurnCountRef.current;
    prevTurnCountRef.current = turns.length;
    if (!grewAtEnd || !stickToBottomRef.current) return;
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  const onScroll = () => {
    const el = scrollerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 80;

    if (el.scrollTop < 48 && hasMoreHistory && !loadingOlder && onLoadOlder) {
      pendingScrollRestoreRef.current = {
        height: el.scrollHeight,
        top: el.scrollTop,
      };
      void onLoadOlder();
    }
  };

  const replay = async (turnDbId: number) => {
    if (!turnDbId) return;
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
    <div className="transcript-log" ref={scrollerRef} onScroll={onScroll}>
      {hasMoreHistory && (
        <p className="transcript-log__older" aria-live="polite">
          {loadingOlder ? "Loading earlier messages…" : "Scroll up for earlier messages"}
        </p>
      )}
      {turns.map((turn) => (
        <div className="transcript-turn" key={turn.id}>
          {turn.transcript ? (
            <p className="transcript-turn__line transcript-turn__line--user">{turn.transcript}</p>
          ) : null}
          {turn.replyText ? (
            <p className="transcript-turn__line transcript-turn__line--companion">
              <span className="transcript-turn__speaker">{companionName}</span>
              {turn.replyText}
              {turn.turnDbId > 0 && (
                <button
                  type="button"
                  className="transcript-turn__replay"
                  onClick={() => replay(turn.turnDbId)}
                  disabled={playingId === turn.turnDbId}
                  aria-label="Replay this reply"
                >
                  {playingId === turn.turnDbId ? "…" : "🔊"}
                </button>
              )}
            </p>
          ) : null}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
