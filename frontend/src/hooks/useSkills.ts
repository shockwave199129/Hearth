import { useCallback, useEffect, useState } from "react";
import { friendlyFetchError } from "../lib/errors";

export interface SkillSummary {
  id: string;
  title: string;
  tags: string[];
  summary: string;
}

export interface SkillDetail {
  id: string;
  title: string;
  content: string;
  source: string;
}

interface UseSkillsResult {
  skills: SkillSummary[];
  loading: boolean;
  error: string | null;
  getSkill: (id: string) => Promise<SkillDetail>;
}

/** Backs Settings → Support techniques. The skills library is static
 * reference content the companion draws on via list_skills/get_skill tool
 * calls (skills/tools.py) — read-only here, unlike memory, since there's
 * nothing user-specific to edit. See project-plan.md §6. */
export function useSkills(): UseSkillsResult {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/skills")
      .then((res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json() as Promise<SkillSummary[]>;
      })
      .then((data) => !cancelled && setSkills(data))
      .catch((err) => !cancelled && setError(friendlyFetchError(err, "useSkills")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const getSkill = useCallback(async (id: string) => {
    const res = await fetch(`/api/skills/${id}`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    return (await res.json()) as SkillDetail;
  }, []);

  return { skills, loading, error, getSkill };
}
