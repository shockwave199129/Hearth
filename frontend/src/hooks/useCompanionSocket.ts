import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../lib/backendFetch";

export type SocketStatus = "connecting" | "reconnecting" | "open" | "closed" | "error";

export interface Turn {
  id: string;
  transcript: string;
  replyText: string;
  turnDbId: number;
  /** Smallest chat_history row id in this paired turn — used for pagination. */
  oldestRowId?: number;
  /** Sent, still waiting on the companion's reply. */
  pending?: boolean;
  /** Shown in place of a reply when the turn could not complete. */
  error?: string;
  /** Reply arrived as text because speech synthesis failed. */
  voiceFailed?: boolean;
}

interface ChatHistoryRow {
  id: number;
  session_id: string;
  turn_id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

interface TurnMeta {
  type?: string;
  message?: string;
  transcript: string;
  reply_text: string;
  sample_rate: number;
  turn_db_id: number;
  has_audio: boolean;
  voice_failed?: boolean;
}

interface UseCompanionSocketResult {
  status: SocketStatus;
  turns: Turn[];
  isThinking: boolean;
  isSpeaking: boolean;
  speakingAmplitude: number;
  hasMoreHistory: boolean;
  loadingOlder: boolean;
  loadOlderHistory: () => Promise<void>;
  sendUtterance: (audio: Float32Array) => void;
  sendText: (text: string) => void;
}

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 3000, 5000, 5000, 5000, 5000];
/** ~20 conversation turns (user+assistant ≈ 2 rows each). */
const HISTORY_PAGE_ROWS = 40;

/** Pair persisted chat_history rows into Talk-page Turn objects.
 * API returns newest-first; result is oldest-first. */
export function historyRowsToTurns(rows: ChatHistoryRow[]): Turn[] {
  const map = new Map<
    string,
    {
      transcript?: string;
      replyText?: string;
      turnDbId?: number;
      oldestRowId: number;
      order: number;
    }
  >();
  for (const row of rows) {
    const key = `${row.session_id}:${row.turn_id}`;
    const slot = map.get(key) ?? { oldestRowId: row.id, order: row.id };
    slot.order = Math.min(slot.order, row.id);
    slot.oldestRowId = Math.min(slot.oldestRowId, row.id);
    if (row.role === "user") {
      slot.transcript = row.content;
    } else {
      slot.replyText = row.content;
      slot.turnDbId = row.id;
    }
    map.set(key, slot);
  }
  return [...map.entries()]
    .map(([key, s]) => ({
      id: key,
      transcript: s.transcript ?? "",
      replyText: s.replyText ?? "",
      turnDbId: s.turnDbId ?? 0,
      oldestRowId: s.oldestRowId,
      order: s.order,
    }))
    .filter((t) => t.transcript.length > 0 || t.replyText.length > 0)
    .sort((a, b) => a.order - b.order)
    .map(({ order: _order, ...turn }) => turn);
}

async function fetchHistoryPage(beforeId?: number): Promise<{
  turns: Turn[];
  hasMore: boolean;
  oldestRowId: number | null;
}> {
  const params = new URLSearchParams({ limit: String(HISTORY_PAGE_ROWS) });
  if (beforeId != null) params.set("before_id", String(beforeId));
  const res = await backendFetch(`/api/chat_history?${params}`);
  if (!res.ok) throw new Error(`status ${res.status}`);
  const data = (await res.json()) as { items: ChatHistoryRow[]; has_more: boolean };
  const items = data.items ?? [];
  const turns = historyRowsToTurns(items);
  const oldestRowId = items.length > 0 ? Math.min(...items.map((r) => r.id)) : null;
  return { turns, hasMore: Boolean(data.has_more), oldestRowId };
}

/** Owns the /ws round trip and reply playback. History lives on the Talk
 * page: initial page of ~20 turns, older pages on scroll-up. */
export function useCompanionSocket(url: string): UseCompanionSocketResult {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const [historyTurns, setHistoryTurns] = useState<Turn[]>([]);
  const [liveTurns, setLiveTurns] = useState<Turn[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingAmplitude, setSpeakingAmplitude] = useState(0);
  const [hasMoreHistory, setHasMoreHistory] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const pendingMetaRef = useRef<TurnMeta | null>(null);
  const pendingTurnIdRef = useRef<string | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const attemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const oldestRowIdRef = useRef<number | null>(null);
  const loadingOlderRef = useRef(false);

  const turns = [...historyTurns, ...liveTurns];

  /** Place the turn in the log the moment it is sent. Waiting for a
   * successful reply meant any backend failure erased the message with
   * nothing to show for it, which is indistinguishable from the app
   * ignoring the input — the failure mode packaged builds actually hit. */
  const beginTurn = useCallback((transcript: string) => {
    const id = crypto.randomUUID();
    pendingTurnIdRef.current = id;
    setLiveTurns((prev) => [
      ...prev,
      { id, transcript, replyText: "", turnDbId: 0, pending: true },
    ]);
  }, []);

  /** Rewrite the turn `beginTurn` reserved, or append `fallback` if it is
   * already gone (a reconnect between send and reply drops the slot). */
  const settleTurn = useCallback((settle: (turn: Turn) => Turn, fallback: () => Turn) => {
    const id = pendingTurnIdRef.current;
    pendingTurnIdRef.current = null;
    setLiveTurns((prev) => {
      const index = id === null ? -1 : prev.findIndex((turn) => turn.id === id);
      if (index === -1) return [...prev, fallback()];
      const next = [...prev];
      next[index] = settle(next[index]);
      return next;
    });
  }, []);

  const resolveTurn = useCallback(
    (meta: TurnMeta) => {
      const filled = (turn: Turn): Turn => ({
        ...turn,
        // Voice turns learn their transcript from STT here; typed turns
        // already carried it into beginTurn.
        transcript: meta.transcript || turn.transcript,
        replyText: meta.reply_text,
        turnDbId: meta.turn_db_id,
        oldestRowId: meta.turn_db_id,
        pending: false,
        error: undefined,
        voiceFailed: meta.voice_failed === true,
      });
      settleTurn(filled, () =>
        filled({ id: crypto.randomUUID(), transcript: "", replyText: "", turnDbId: 0 }),
      );
    },
    [settleTurn],
  );

  const failTurn = useCallback(
    (message: string) => {
      settleTurn(
        (turn) => ({ ...turn, pending: false, error: message }),
        () => ({
          id: crypto.randomUUID(),
          transcript: "",
          replyText: "",
          turnDbId: 0,
          error: message,
        }),
      );
    },
    [settleTurn],
  );

  useEffect(() => {
    let cancelled = false;
    fetchHistoryPage()
      .then(({ turns: page, hasMore, oldestRowId }) => {
        if (cancelled) return;
        oldestRowIdRef.current = oldestRowId;
        setHasMoreHistory(hasMore);
        setHistoryTurns(page);
      })
      .catch((err) => {
        console.error("[useCompanionSocket] history hydrate failed", err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const loadOlderHistory = useCallback(async () => {
    if (loadingOlderRef.current || oldestRowIdRef.current == null) return;
    loadingOlderRef.current = true;
    setLoadingOlder(true);
    try {
      const { turns: older, hasMore, oldestRowId } = await fetchHistoryPage(
        oldestRowIdRef.current,
      );
      if (older.length === 0) {
        setHasMoreHistory(false);
        return;
      }
      oldestRowIdRef.current = oldestRowId;
      setHasMoreHistory(hasMore);
      setHistoryTurns((prev) => {
        const seen = new Set(prev.map((t) => t.id));
        const fresh = older.filter((t) => !seen.has(t.id));
        return [...fresh, ...prev];
      });
    } catch (err) {
      console.error("[useCompanionSocket] load older failed", err);
    } finally {
      loadingOlderRef.current = false;
      setLoadingOlder(false);
    }
  }, []);

  const playReply = useCallback(
    (meta: TurnMeta, audio: Float32Array) => {
      setIsThinking(false);
      resolveTurn(meta);

      if (!audio.length || !meta.sample_rate) return;

      if (!playbackCtxRef.current) playbackCtxRef.current = new AudioContext();
      const ctx = playbackCtxRef.current;
      if (ctx.state === "suspended") void ctx.resume();

      const buffer = ctx.createBuffer(1, audio.length, meta.sample_rate);
      buffer.copyToChannel(new Float32Array(audio), 0);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyser.connect(ctx.destination);

      const levels = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteTimeDomainData(levels);
        let sum = 0;
        for (let i = 0; i < levels.length; i++) {
          const centered = (levels[i] - 128) / 128;
          sum += centered * centered;
        }
        setSpeakingAmplitude(Math.min(1, Math.sqrt(sum / levels.length) * 4));
        rafRef.current = requestAnimationFrame(tick);
      };

      setIsSpeaking(true);
      tick();
      source.onended = () => {
        setIsSpeaking(false);
        setSpeakingAmplitude(0);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      };
      try {
        source.start();
      } catch (err) {
        console.error("[useCompanionSocket] reply playback failed", err);
        setIsSpeaking(false);
        setSpeakingAmplitude(0);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      }
    },
    [resolveTurn],
  );

  useEffect(() => {
    let cancelled = false;

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const scheduleReconnect = () => {
      if (cancelled) return;
      clearReconnectTimer();
      setStatus("reconnecting");
      const delay = RECONNECT_DELAYS_MS[Math.min(attemptRef.current, RECONNECT_DELAYS_MS.length - 1)];
      attemptRef.current += 1;
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    const connect = () => {
      if (cancelled) return;
      clearReconnectTimer();

      if (attemptRef.current === 0) {
        setStatus("connecting");
      } else {
        setStatus("reconnecting");
      }

      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        attemptRef.current = 0;
        setStatus("open");
      };

      ws.onclose = () => {
        if (cancelled) return;
        setIsThinking(false);
        pendingMetaRef.current = null;
        if (pendingTurnIdRef.current !== null) {
          failTurn("Lost the connection before a reply came back.");
        }
        setStatus("closed");
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (cancelled) return;
        setIsThinking(false);
        setStatus("error");
      };

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          let meta: TurnMeta;
          try {
            meta = JSON.parse(event.data) as TurnMeta;
          } catch (err) {
            console.error("[useCompanionSocket] bad JSON frame", err);
            return;
          }
          if (meta.type === "error") {
            const stranded = pendingMetaRef.current;
            pendingMetaRef.current = null;
            setIsThinking(false);
            console.error("[useCompanionSocket] turn error", meta.message);
            if (stranded) {
              // The reply itself already arrived — only its audio failed, so
              // keep the text rather than replacing it with the error.
              resolveTurn({ ...stranded, voice_failed: true });
            } else {
              failTurn(meta.message ?? "Something went wrong on that turn.");
            }
            return;
          }
          // Text-only when speak_replies is off, or when synthesis failed and
          // the server fell back to text. When audio is expected, wait for the
          // binary frame so the turn appears with voice.
          if (!meta.has_audio) {
            pendingMetaRef.current = null;
            setIsThinking(false);
            resolveTurn(meta);
            return;
          }
          pendingMetaRef.current = meta;
          return;
        }
        const meta = pendingMetaRef.current;
        pendingMetaRef.current = null;
        if (!meta) return;
        playReply(meta, new Float32Array(event.data as ArrayBuffer));
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) {
        ws.onopen = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
        ws.close();
      }
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      void playbackCtxRef.current?.close();
      playbackCtxRef.current = null;
    };
  }, [url, playReply, resolveTurn, failTurn]);

  // Both senders record the turn first, then check the socket, so a send that
  // loses a readyState race says so in the log instead of vanishing.
  const sendUtterance = useCallback(
    (audio: Float32Array) => {
      beginTurn("");
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        failTurn("Not connected — that recording wasn't sent.");
        return;
      }
      setIsThinking(true);
      const buffer: ArrayBuffer = new Float32Array(audio).buffer as ArrayBuffer;
      wsRef.current.send(buffer);
    },
    [beginTurn, failTurn],
  );

  const sendText = useCallback(
    (text: string) => {
      beginTurn(text);
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        failTurn("Not connected — that message wasn't sent.");
        return;
      }
      setIsThinking(true);
      wsRef.current.send(JSON.stringify({ type: "text", text }));
    },
    [beginTurn, failTurn],
  );

  return {
    status,
    turns,
    isThinking,
    isSpeaking,
    speakingAmplitude,
    hasMoreHistory,
    loadingOlder,
    loadOlderHistory,
    sendUtterance,
    sendText,
  };
}
