import { useCallback, useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";
import type { Profile } from "./useProfile";

interface UseProfilesResult {
  profiles: Profile[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
  activateProfile: (userId: string) => Promise<Profile>;
  deleteProfile: (userId: string) => Promise<void>;
}

/** Backs Settings → Profiles. Real multi-profile support — each local
 * install can hold several named profiles, exactly one active at a time
 * (switching is a deliberate action, see backend/app/main.py's Pipeline). */
export function useProfiles(): UseProfilesResult {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const refresh = useCallback(() => setRefreshToken((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/profiles")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<Profile[]>;
      })
      .then((data) => !cancelled && setProfiles(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useProfiles")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const activateProfile = useCallback(
    async (userId: string) => {
      const res = await fetch(`/api/profiles/${userId}/activate`, { method: "POST" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const activated = (await res.json()) as Profile;
      refresh();
      return activated;
    },
    [refresh],
  );

  const deleteProfile = useCallback(
    async (userId: string) => {
      const res = await fetch(`/api/profiles/${userId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      refresh();
    },
    [refresh],
  );

  return { profiles, loading, error, refresh, activateProfile, deleteProfile };
}
