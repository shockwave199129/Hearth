import { useState } from "react";
import { friendlyActionError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import "./MemoryPanel.css";
import { useMemories } from "../hooks/useMemories";

export function MemoryPanel() {
  const { memories, loading, error, getMemory, updateMemory, deleteMemory } = useMemories();
  const { showAlert } = useAlert();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [draftText, setDraftText] = useState("");
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
      const detail = await getMemory(id);
      setDraftText(detail.text);
      setExpandedId(id);
    } catch (err) {
      setDetailError(friendlyActionError(err, "MemoryPanel.toggleExpand", "Couldn't load that memory."));
    } finally {
      setBusyId(null);
    }
  };

  const handleSave = async (id: string) => {
    setBusyId(id);
    setDetailError(null);
    try {
      await updateMemory(id, draftText);
      setExpandedId(null);
      showAlert({ type: "success", message: "Memory updated." });
    } catch (err) {
      const message = friendlyActionError(err, "MemoryPanel.save", "Couldn't save changes.");
      setDetailError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: string) => {
    setBusyId(id);
    setDetailError(null);
    try {
      await deleteMemory(id);
      setExpandedId(null);
      showAlert({ type: "success", message: "Memory deleted." });
    } catch (err) {
      const message = friendlyActionError(err, "MemoryPanel.delete", "Couldn't delete that memory.");
      setDetailError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusyId(null);
    }
  };

  if (loading) return <p className="settings__hint">Reading memory…</p>;
  if (error) return <p className="settings__error">{error}</p>;
  if (memories.length === 0) {
    return (
      <p className="settings__hint">
        Nothing remembered yet — the companion saves things quietly as you talk, and they'll show up here.
      </p>
    );
  }

  return (
    <div className="memory-panel">
      {detailError && <p className="settings__error">{detailError}</p>}
      <ul className="memory-panel__list">
        {memories.map((memory) => (
          <li key={memory.id} className="memory-panel__item">
            <button
              type="button"
              className="memory-panel__row"
              onClick={() => toggleExpand(memory.id)}
              disabled={busyId === memory.id}
            >
              <span className="memory-panel__category">{memory.category}</span>
              <span className="memory-panel__label">{memory.label}</span>
            </button>
            {expandedId === memory.id && (
              <div className="memory-panel__detail">
                <textarea
                  value={draftText}
                  onChange={(e) => setDraftText(e.target.value)}
                  rows={3}
                />
                <div className="memory-panel__actions">
                  <button
                    type="button"
                    className="memory-panel__button memory-panel__button--danger"
                    onClick={() => handleDelete(memory.id)}
                    disabled={busyId === memory.id}
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    className="memory-panel__button memory-panel__button--primary"
                    onClick={() => handleSave(memory.id)}
                    disabled={busyId === memory.id}
                  >
                    {busyId === memory.id ? "Saving…" : "Save"}
                  </button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
