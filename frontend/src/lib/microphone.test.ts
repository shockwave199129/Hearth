import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isVoiceInputPreferred,
  probeMicrophone,
  setVoiceInputPreferred,
  subscribeVoiceInputPreference,
  voiceInputUsable,
} from "./microphone";

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("voice input preference", () => {
  it("defaults to on", () => {
    expect(isVoiceInputPreferred()).toBe(true);
  });

  it("persists off and notifies subscribers", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeVoiceInputPreference(listener);
    setVoiceInputPreferred(false);
    expect(isVoiceInputPreferred()).toBe(false);
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    setVoiceInputPreferred(true);
    expect(listener).toHaveBeenCalledTimes(1);
  });
});

describe("probeMicrophone", () => {
  it("reports absent when mediaDevices is missing", async () => {
    vi.stubGlobal("navigator", {});
    await expect(probeMicrophone()).resolves.toEqual({ hardware: "absent", permission: "unknown" });
  });

  it("reports present when an audioinput device is listed", async () => {
    vi.stubGlobal("navigator", {
      mediaDevices: {
        enumerateDevices: vi.fn().mockResolvedValue([
          { kind: "audioinput", deviceId: "mic-1" },
          { kind: "audiooutput", deviceId: "spk-1" },
        ]),
      },
      permissions: {
        query: vi.fn().mockResolvedValue({ state: "granted" }),
      },
    });
    await expect(probeMicrophone()).resolves.toEqual({ hardware: "present", permission: "granted" });
  });

  it("reports absent when no audioinput devices exist", async () => {
    vi.stubGlobal("navigator", {
      mediaDevices: {
        enumerateDevices: vi.fn().mockResolvedValue([{ kind: "audiooutput", deviceId: "spk-1" }]),
      },
      permissions: {
        query: vi.fn().mockRejectedValue(new Error("unsupported")),
      },
    });
    await expect(probeMicrophone()).resolves.toEqual({ hardware: "absent", permission: "unknown" });
  });
});

describe("voiceInputUsable", () => {
  it("requires a preferred-on setting and does not treat unknown as missing", () => {
    expect(voiceInputUsable({ hardware: "present", permission: "granted" }, true)).toBe(true);
    expect(voiceInputUsable({ hardware: "unknown", permission: "unknown" }, true)).toBe(true);
    expect(voiceInputUsable({ hardware: "present", permission: "prompt" }, false)).toBe(false);
    expect(voiceInputUsable({ hardware: "absent", permission: "unknown" }, true)).toBe(false);
    expect(voiceInputUsable({ hardware: "present", permission: "denied" }, true)).toBe(false);
  });
});
