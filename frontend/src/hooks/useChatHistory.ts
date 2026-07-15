import { useCallback, useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";

export interface ChatHistoryTurn {
  id: number;
  session_id: string;
  turn_id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

interface UseChatHistoryResult {
  turns: ChatHistoryTurn[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
  deleteTurn: (id: number) => Promise<void>;
  playTurn: (id: number) => Promise<void>;
}

/** Backs Settings → Conversation history. Replay re-synthesizes stored text
 * on demand via the normal TTS engine (GET /api/chat_history/{id}/audio) —
 * no audio files are cached anywhere. */
export function useChatHistory(): UseChatHistoryResult {
  const [turns, setTurns] = useState<ChatHistoryTurn[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const refresh = useCallback(() => setRefreshToken((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/chat_history")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<ChatHistoryTurn[]>;
      })
      .then((data) => !cancelled && setTurns(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useChatHistory")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const deleteTurn = useCallback(
    async (id: number) => {
      const res = await fetch(`/api/chat_history/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      refresh();
    },
    [refresh],
  );

  const playTurn = useCallback(async (id: number) => {
    const res = await fetch(`/api/chat_history/${id}/audio`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  }, []);

  return { turns, loading, error, refresh, deleteTurn, playTurn };
}
