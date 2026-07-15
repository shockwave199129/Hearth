export const TARGET_SAMPLE_RATE = 16000;

/** Linear-interpolation resample — good enough for speech-to-STT input;
 * a proper windowed-sinc resampler is overkill for a 1.2B-param LLM loop. */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const outLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcIndex - i0;
    output[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return output;
}

export function rms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

export function concatFloat32(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
