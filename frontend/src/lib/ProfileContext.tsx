import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { backendFetch, fetchWithTimeout, retryWithBackoff } from "./backendFetch";
import { friendlyFetchError } from "./errors";

export interface Profile {
  user_id: string;
  name: string;
  age_range: string | null;
  gender: string | null;
  profession: string | null;
  stressors: string[];
  preferred_voice: "female" | "male";
  companion_name: string;
  speak_replies: boolean;
  emergency_contact_consent: boolean;
  emergency_contact_name: string | null;
  emergency_contact_method: "sms" | "email" | null;
  emergency_contact_value: string | null;
  created_at: string;
}

export type OnboardingPayload = Omit<Profile, "created_at" | "user_id">;

export interface ProfileContextValue {
  profile: Profile | null;
  loading: boolean;
  error: string | null;
  submitOnboarding: (payload: OnboardingPayload) => Promise<Profile>;
  setSpeakReplies: (value: boolean) => Promise<void>;
}

const ProfileContext = createContext<ProfileContextValue | null>(null);

/** Single shared profile state for the whole app — App gating, Onboarding,
 * Chat, and Settings all read/write the same active profile so submitting
 * onboarding updates the gate immediately (no stale null in App). */
export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const cancelledRef = { current: false };

    // 404 ("never onboarded") is a real, expected answer — only actual
    // connectivity failures (backend not listening/still booting yet) go
    // through the retry path below. See lib/backendFetch.ts.
    const attempt = async (): Promise<Profile | null> => {
      const res = await fetchWithTimeout("/api/profile");
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`status ${res.status}`);
      return (await res.json()) as Profile;
    };

    retryWithBackoff(attempt, cancelledRef)
      .then((data) => {
        if (!cancelledRef.current) setProfile(data);
      })
      .catch((err) => {
        if (!cancelledRef.current) setError(friendlyFetchError(err, "useProfile"));
      })
      .finally(() => {
        if (!cancelledRef.current) setLoading(false);
      });

    return () => {
      cancelledRef.current = true;
    };
  }, []);

  const submitOnboarding = useCallback(async (payload: OnboardingPayload) => {
    const res = await backendFetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`onboarding failed with status ${res.status}`);
    const saved = (await res.json()) as Profile;
    setProfile(saved);
    setError(null);
    return saved;
  }, []);

  const setSpeakReplies = useCallback(async (value: boolean) => {
    const res = await backendFetch("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speak_replies: value }),
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const updated = (await res.json()) as Profile;
    setProfile(updated);
  }, []);

  const value = useMemo(
    () => ({ profile, loading, error, submitOnboarding, setSpeakReplies }),
    [profile, loading, error, submitOnboarding, setSpeakReplies],
  );

  return <ProfileContext.Provider value={value}>{children}</ProfileContext.Provider>;
}

export function useProfile(): ProfileContextValue {
  const ctx = useContext(ProfileContext);
  if (!ctx) {
    throw new Error("useProfile must be used within ProfileProvider");
  }
  return ctx;
}
