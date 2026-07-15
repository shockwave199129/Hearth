import { useCallback, useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";

export interface MemorySummary {
  id: string;
  category: string;
  label: string;
}

export interface MemoryDetail {
  id: string;
  category: string;
  text: string;
}

interface UseMemoriesResult {
  memories: MemorySummary[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
  getMemory: (id: string) => Promise<MemoryDetail>;
  updateMemory: (id: string, text: string) => Promise<MemoryDetail>;
  deleteMemory: (id: string) => Promise<void>;
}

/** Backs Settings → Memory. Long-term memories are otherwise only ever
 * touched by the companion itself via tool calls (memory/tools.py) — this
 * is the "never actually hidden from them if they go look" surface from
 * project-plan.md §5. */
export function useMemories(): UseMemoriesResult {
  const [memories, setMemories] = useState<MemorySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const refresh = useCallback(() => setRefreshToken((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/memories")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<MemorySummary[]>;
      })
      .then((data) => !cancelled && setMemories(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useMemories")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const getMemory = useCallback(async (id: string) => {
    const res = await fetch(`/api/memories/${id}`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    return (await res.json()) as MemoryDetail;
  }, []);

  const updateMemory = useCallback(
    async (id: string, text: string) => {
      const res = await fetch(`/api/memories/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const updated = (await res.json()) as MemoryDetail;
      refresh();
      return updated;
    },
    [refresh],
  );

  const deleteMemory = useCallback(
    async (id: string) => {
      const res = await fetch(`/api/memories/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      refresh();
    },
    [refresh],
  );

  return { memories, loading, error, refresh, getMemory, updateMemory, deleteMemory };
}
