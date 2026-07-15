import { useCallback, useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";
import { fetchWithTimeout, retryWithBackoff } from "../lib/backendFetch";

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

interface UseProfileResult {
  profile: Profile | null;
  loading: boolean;
  error: string | null;
  submitOnboarding: (payload: OnboardingPayload) => Promise<Profile>;
  setSpeakReplies: (value: boolean) => Promise<void>;
}

/** Source of truth for "has this install been onboarded" — a 404 from
 * /api/profile means never onboarded, distinct from an onboarded profile
 * that just has mostly-empty optional fields. */
export function useProfile(): UseProfileResult {
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
    const res = await fetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`onboarding failed with status ${res.status}`);
    const saved = (await res.json()) as Profile;
    setProfile(saved);
    return saved;
  }, []);

  const setSpeakReplies = useCallback(async (value: boolean) => {
    const res = await fetch("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speak_replies: value }),
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const updated = (await res.json()) as Profile;
    setProfile(updated);
  }, []);

  return { profile, loading, error, submitOnboarding, setSpeakReplies };
}
