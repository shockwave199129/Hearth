import { useState } from "react";
import { friendlyActionError } from "../lib/errors";
import { Link } from "react-router-dom";
import "./ProfilesPanel.css";
import { useProfiles } from "../hooks/useProfiles";
import { useProfile } from "../hooks/useProfile";

export function ProfilesPanel() {
  const { profile: activeProfile } = useProfile();
  const { profiles, loading, error, activateProfile, deleteProfile } = useProfiles();
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const handleActivate = async (userId: string) => {
    setActionError(null);
    setBusyId(userId);
    try {
      await activateProfile(userId);
      // Every hook (chat socket, memories, checkins...) initializes against
      // whichever profile is active at load time — a full reload is the
      // simplest way to make every one of them pick up the switch cleanly.
      window.location.href = "/chat";
    } catch (err) {
      setActionError(friendlyActionError(err, "ProfilesPanel.activate", "Couldn't switch profiles."));
      setBusyId(null);
    }
  };

  const handleDelete = async (userId: string) => {
    setActionError(null);
    setBusyId(userId);
    try {
      await deleteProfile(userId);
      if (activeProfile?.user_id === userId) {
        window.location.href = "/chat";
        return;
      }
    } catch (err) {
      setActionError(friendlyActionError(err, "ProfilesPanel.delete", "Couldn't delete that profile."));
    } finally {
      setBusyId(null);
    }
  };

  if (loading) return <p className="settings__hint">Reading profiles…</p>;
  if (error) return <p className="settings__error">{error}</p>;

  return (
    <div className="profiles-panel">
      {actionError && <p className="settings__error">{actionError}</p>}
      <ul className="profiles-panel__list">
        {profiles.map((profile) => {
          const isActive = profile.user_id === activeProfile?.user_id;
          return (
            <li key={profile.user_id} className="profiles-panel__item">
              <div className="profiles-panel__row">
                <div>
                  <span className="profiles-panel__name">{profile.name}</span>
                  <span className="profiles-panel__companion"> — talks to {profile.companion_name}</span>
                  {isActive && <span className="profiles-panel__badge">Active</span>}
                </div>
                <div className="profiles-panel__actions">
                  {!isActive && (
                    <button
                      type="button"
                      className="profiles-panel__button profiles-panel__button--primary"
                      onClick={() => handleActivate(profile.user_id)}
                      disabled={busyId === profile.user_id}
                    >
                      {busyId === profile.user_id ? "Switching…" : "Switch"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="profiles-panel__button profiles-panel__button--danger"
                    onClick={() => handleDelete(profile.user_id)}
                    disabled={busyId === profile.user_id}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
      <Link
        to="/onboarding?mode=add"
        className="profiles-panel__button profiles-panel__button--primary profiles-panel__add"
      >
        Add another profile
      </Link>
    </div>
  );
}
