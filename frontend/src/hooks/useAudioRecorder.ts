import { useCallback, useEffect, useRef, useState } from "react";
import { TARGET_SAMPLE_RATE, concatFloat32, resampleTo16k, rms } from "../lib/audio";

const SILENCE_RMS_THRESHOLD = 0.02;
const SILENCE_HOLD_S = 1.2;
const MAX_UTTERANCE_S = 30;
const BLOCK_S = 0.1;
const BLOCK_SAMPLES = Math.round(BLOCK_S * TARGET_SAMPLE_RATE);

export type RecorderState = "idle" | "requesting" | "listening" | "error";

interface UseAudioRecorderResult {
  state: RecorderState;
  amplitude: number;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

/** Records mic audio, resamples to 16kHz mono, and auto-ends the utterance
 * on trailing silence — the browser mirror of backend/app/audio_io.py so
 * both entry points (CLI and web) share the same VAD behavior. */
export function useAudioRecorder(onUtterance: (audio: Float32Array) => void): UseAudioRecorderResult {
  const [state, setState] = useState<RecorderState>("idle");
  const [amplitude, setAmplitude] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const blockChunksRef = useRef<Float32Array[]>([]);
  const utteranceChunksRef = useRef<Float32Array[]>([]);
  const silenceBlocksRef = useRef(0);
  const heardSpeechRef = useRef(false);
  const elapsedBlocksRef = useRef(0);
  const activeRef = useRef(false);
  const pendingRef = useRef<Float32Array>(new Float32Array(0));

  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;

  const finish = useCallback(() => {
    activeRef.current = false;
    setState("idle");
    setAmplitude(0);
    const captured = concatFloat32(utteranceChunksRef.current);
    utteranceChunksRef.current = [];
    blockChunksRef.current = [];
    silenceBlocksRef.current = 0;
    heardSpeechRef.current = false;
    elapsedBlocksRef.current = 0;
    if (captured.length > 0) onUtteranceRef.current(captured);
  }, []);

  const handleFrame = useCallback(
    (frame: Float32Array, nativeRate: number) => {
      if (!activeRef.current) return;
      const resampled = resampleTo16k(frame, nativeRate);
      const merged = concatFloat32([pendingRef.current, resampled]);

      let offset = 0;
      while (merged.length - offset >= BLOCK_SAMPLES) {
        const block = merged.subarray(offset, offset + BLOCK_SAMPLES);
        offset += BLOCK_SAMPLES;

        const level = rms(block);
        setAmplitude(Math.min(1, level / 0.12));
        utteranceChunksRef.current.push(block.slice());
        elapsedBlocksRef.current += 1;

        if (level >= SILENCE_RMS_THRESHOLD) {
          heardSpeechRef.current = true;
          silenceBlocksRef.current = 0;
        } else {
          silenceBlocksRef.current += 1;
        }

        const silenceElapsed = silenceBlocksRef.current * BLOCK_S;
        const totalElapsed = elapsedBlocksRef.current * BLOCK_S;
        if (
          (heardSpeechRef.current && silenceElapsed >= SILENCE_HOLD_S) ||
          totalElapsed >= MAX_UTTERANCE_S
        ) {
          finish();
          return;
        }
      }
      pendingRef.current = merged.subarray(offset).slice();
    },
    [finish],
  );

  const start = useCallback(async () => {
    if (activeRef.current) return;
    setError(null);
    setState("requesting");
    try {
      if (!audioCtxRef.current) {
        const ctx = new AudioContext();
        await ctx.audioWorklet.addModule("/recorder-worklet.js");
        audioCtxRef.current = ctx;
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") await ctx.resume();

      if (!streamRef.current) {
        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      }
      if (!sourceRef.current) {
        sourceRef.current = ctx.createMediaStreamSource(streamRef.current);
      }
      if (!workletRef.current) {
        const worklet = new AudioWorkletNode(ctx, "recorder-processor");
        worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
          handleFrame(event.data, ctx.sampleRate);
        };
        sourceRef.current.connect(worklet);
        workletRef.current = worklet;
      }

      pendingRef.current = new Float32Array(0);
      utteranceChunksRef.current = [];
      silenceBlocksRef.current = 0;
      heardSpeechRef.current = false;
      elapsedBlocksRef.current = 0;
      activeRef.current = true;
      setState("listening");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access failed");
      setState("error");
    }
  }, [handleFrame]);

  const stop = useCallback(() => {
    if (activeRef.current) finish();
  }, [finish]);

  useEffect(() => {
    return () => {
      workletRef.current?.disconnect();
      sourceRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      audioCtxRef.current?.close();
    };
  }, []);

  return { state, amplitude, error, start, stop };
}
