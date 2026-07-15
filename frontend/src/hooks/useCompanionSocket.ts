import { useCallback, useEffect, useRef, useState } from "react";

export type SocketStatus = "connecting" | "open" | "closed" | "error";

export interface Turn {
  id: string;
  transcript: string;
  replyText: string;
  turnDbId: number;
}

interface TurnMeta {
  transcript: string;
  reply_text: string;
  sample_rate: number;
  turn_db_id: number;
  has_audio: boolean;
}

interface UseCompanionSocketResult {
  status: SocketStatus;
  turns: Turn[];
  isThinking: boolean;
  isSpeaking: boolean;
  speakingAmplitude: number;
  sendUtterance: (audio: Float32Array) => void;
  sendText: (text: string) => void;
}

/** Owns the /ws round trip (send raw 16kHz PCM, receive JSON metadata then
 * PCM reply audio — protocol defined in backend/app/main.py) and reply
 * playback, so Chat.tsx just reacts to status. */
export function useCompanionSocket(url: string): UseCompanionSocketResult {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingAmplitude, setSpeakingAmplitude] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const pendingMetaRef = useRef<TurnMeta | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);

  const addTurn = useCallback((meta: TurnMeta) => {
    setTurns((prev) => [
      ...prev,
      { id: crypto.randomUUID(), transcript: meta.transcript, replyText: meta.reply_text, turnDbId: meta.turn_db_id },
    ]);
  }, []);

  const playReply = useCallback((meta: TurnMeta, audio: Float32Array) => {
    setIsThinking(false);
    addTurn(meta);

    if (!playbackCtxRef.current) playbackCtxRef.current = new AudioContext();
    const ctx = playbackCtxRef.current;
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
    source.start();
  }, []);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setStatus("error");
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const meta = JSON.parse(event.data) as TurnMeta;
        if (!meta.has_audio) {
          // Typed input with speak_replies off (or any turn where the
          // backend skipped TTS) — no binary frame follows, so resolve
          // the turn immediately instead of waiting for one.
          setIsThinking(false);
          addTurn(meta);
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

    return () => {
      ws.close();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      playbackCtxRef.current?.close();
    };
  }, [url, playReply, addTurn]);

  const sendUtterance = useCallback((audio: Float32Array) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    setIsThinking(true);
    const buffer: ArrayBuffer = new Float32Array(audio).buffer as ArrayBuffer;
    wsRef.current.send(buffer);
  }, []);

  const sendText = useCallback((text: string) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    setIsThinking(true);
    wsRef.current.send(JSON.stringify({ type: "text", text }));
  }, []);

  return { status, turns, isThinking, isSpeaking, speakingAmplitude, sendUtterance, sendText };
}
