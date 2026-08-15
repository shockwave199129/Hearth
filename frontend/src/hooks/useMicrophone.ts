import { useCallback, useEffect, useState } from "react";
import {
  isVoiceInputPreferred,
  probeMicrophone,
  setVoiceInputPreferred,
  subscribeVoiceInputPreference,
  voiceInputUsable,
  type MicHardware,
  type MicPermission,
} from "../lib/microphone";

export function useMicrophone() {
  const [hardware, setHardware] = useState<MicHardware>("unknown");
  const [permission, setPermission] = useState<MicPermission>("unknown");
  const [preferred, setPreferred] = useState(true);

  const refresh = useCallback(async () => {
    const status = await probeMicrophone();
    setHardware(status.hardware);
    setPermission(status.permission);
  }, []);

  useEffect(() => {
    setPreferred(isVoiceInputPreferred());
    return subscribeVoiceInputPreference(() => setPreferred(isVoiceInputPreferred()));
  }, []);

  useEffect(() => {
    void refresh();
    const media = navigator.mediaDevices;
    if (!media?.addEventListener) return;
    media.addEventListener("devicechange", refresh);
    return () => media.removeEventListener("devicechange", refresh);
  }, [refresh]);

  const enabled = voiceInputUsable({ hardware, permission }, preferred);

  return {
    hardware,
    permission,
    preferred,
    enabled,
    setPreferred: setVoiceInputPreferred,
    refresh,
  };
}
