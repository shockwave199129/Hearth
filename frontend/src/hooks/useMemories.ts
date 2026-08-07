import { useCallback, useEffect, useState } from "react";
import { backendFetch } from "../lib/backendFetch";
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

interface MemoriesPage {
  items: MemorySummary[];
  has_more: boolean;
  limit: number;
  offset: number;
}

const PAGE_SIZE = 50;

interface UseMemoriesResult {
  memories: MemorySummary[];
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  refresh: () => void;
  loadMore: () => Promise<void>;
  getMemory: (id: string) => Promise<MemoryDetail>;
  updateMemory: (id: string, text: string) => Promise<MemoryDetail>;
  deleteMemory: (id: string) => Promise<void>;
}

/** Backs Settings → Memory. Long-term memories are otherwise only ever
 * touched by the companion itself via tool calls (memory/tools.py) — this
 * is the "never actually hidden from them if they go look" surface from
 * docs/project-plan.md §5. */
export function useMemories(): UseMemoriesResult {
  const [memories, setMemories] = useState<MemorySummary[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const refresh = useCallback(() => setRefreshToken((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: "0" });
    backendFetch(`/api/memories?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<MemoriesPage>;
      })
      .then((data) => {
        if (cancelled) return;
        setMemories(data.items ?? []);
        setHasMore(Boolean(data.has_more));
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("[useMemories]", err);
        const msg =
          err instanceof Error && /status \d+/.test(err.message)
            ? "Couldn't load memories right now."
            : friendlyFetchError(err, "useMemories");
        setError(msg);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(memories.length),
      });
      const res = await backendFetch(`/api/memories?${params}`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as MemoriesPage;
      setMemories((prev) => {
        const seen = new Set(prev.map((m) => m.id));
        return [...prev, ...(data.items ?? []).filter((m) => !seen.has(m.id))];
      });
      setHasMore(Boolean(data.has_more));
    } catch (err) {
      console.error("[useMemories.loadMore]", err);
      setError(
        err instanceof Error && /status \d+/.test(err.message)
          ? "Couldn't load more memories."
          : friendlyFetchError(err, "useMemories.loadMore"),
      );
    } finally {
      setLoadingMore(false);
    }
  }, [hasMore, loadingMore, memories.length]);

  const getMemory = useCallback(async (id: string) => {
    const res = await backendFetch(`/api/memories/${id}`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    return (await res.json()) as MemoryDetail;
  }, []);

  const updateMemory = useCallback(
    async (id: string, text: string) => {
      const res = await backendFetch(`/api/memories/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const updated = (await res.json()) as MemoryDetail;
      setMemories((prev) =>
        prev.map((m) => (m.id === id ? { ...m, label: updated.text.slice(0, 40) } : m)),
      );
      return updated;
    },
    [],
  );

  const deleteMemory = useCallback(async (id: string) => {
    const res = await backendFetch(`/api/memories/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`status ${res.status}`);
    setMemories((prev) => prev.filter((m) => m.id !== id));
  }, []);

  return {
    memories,
    loading,
    loadingMore,
    hasMore,
    error,
    refresh,
    loadMore,
    getMemory,
    updateMemory,
    deleteMemory,
  };
}
