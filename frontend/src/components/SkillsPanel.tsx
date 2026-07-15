import { useState } from "react";
import { friendlyActionError } from "../lib/errors";
import "./SkillsPanel.css";
import { useSkills } from "../hooks/useSkills";
import type { SkillDetail } from "../hooks/useSkills";

export function SkillsPanel() {
  const { skills, loading, error, getSkill } = useSkills();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SkillDetail | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const toggleExpand = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setDetailError(null);
    setBusyId(id);
    try {
      const loaded = await getSkill(id);
      setDetail(loaded);
      setExpandedId(id);
    } catch (err) {
      setDetailError(friendlyActionError(err, "SkillsPanel.toggleExpand", "Couldn't load that technique."));
    } finally {
      setBusyId(null);
    }
  };

  if (loading) return <p className="settings__hint">Reading skills library…</p>;
  if (error) return <p className="settings__error">{error}</p>;
  if (skills.length === 0) {
    return <p className="settings__hint">No techniques available yet.</p>;
  }

  return (
    <div className="skills-panel">
      {detailError && <p className="settings__error">{detailError}</p>}
      <ul className="skills-panel__list">
        {skills.map((skill) => (
          <li key={skill.id} className="skills-panel__item">
            <button
              type="button"
              className="skills-panel__row"
              onClick={() => toggleExpand(skill.id)}
              disabled={busyId === skill.id}
            >
              <span className="skills-panel__title">{skill.title}</span>
              <span className="skills-panel__summary">{skill.summary}</span>
              {skill.tags.length > 0 && (
                <span className="skills-panel__tags">{skill.tags.join(", ")}</span>
              )}
            </button>
            {expandedId === skill.id && detail && (
              <div className="skills-panel__detail">
                <pre className="skills-panel__content">{detail.content}</pre>
                {detail.source && <p className="skills-panel__source">Source: {detail.source}</p>}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
