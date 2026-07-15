// Runs on the audio render thread; hands raw mono Float32 frames back to the
// main thread via the port so useAudioRecorder can resample/VAD/buffer them.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      this.port.postMessage(channel.slice(0));
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
