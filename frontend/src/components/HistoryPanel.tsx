import { useState } from "react";
import { friendlyActionError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import "./HistoryPanel.css";
import { useChatHistory } from "../hooks/useChatHistory";

export function HistoryPanel() {
  const { turns, loading, error, deleteTurn, playTurn } = useChatHistory();
  const { showAlert } = useAlert();
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const handlePlay = async (id: number) => {
    setActionError(null);
    setBusyId(id);
    try {
      await playTurn(id);
    } catch (err) {
      const message = friendlyActionError(err, "HistoryPanel.play", "Couldn't play that reply.");
      setActionError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: number) => {
    setActionError(null);
    setBusyId(id);
    try {
      await deleteTurn(id);
      showAlert({ type: "success", message: "Turn deleted." });
    } catch (err) {
      const message = friendlyActionError(err, "HistoryPanel.delete", "Couldn't delete that turn.");
      setActionError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusyId(null);
    }
  };

  if (loading) return <p className="settings__hint">Reading history…</p>;
  if (error) return <p className="settings__error">{error}</p>;
  if (turns.length === 0) {
    return <p className="settings__hint">Nothing said yet this profile — it'll show up here once you talk.</p>;
  }

  return (
    <div className="history-panel">
      {actionError && <p className="settings__error">{actionError}</p>}
      <ul className="history-panel__list">
        {turns.map((turn) => (
          <li key={turn.id} className="history-panel__item">
            <div className="history-panel__row">
              <span className={`history-panel__role history-panel__role--${turn.role}`}>
                {turn.role === "assistant" ? "Companion" : "You"}
              </span>
              <span className="history-panel__text">{turn.content}</span>
            </div>
            <div className="history-panel__actions">
              {turn.role === "assistant" && (
                <button
                  type="button"
                  className="history-panel__button"
                  onClick={() => handlePlay(turn.id)}
                  disabled={busyId === turn.id}
                >
                  {busyId === turn.id ? "Playing…" : "🔊 Replay"}
                </button>
              )}
              <button
                type="button"
                className="history-panel__button history-panel__button--danger"
                onClick={() => handleDelete(turn.id)}
                disabled={busyId === turn.id}
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
