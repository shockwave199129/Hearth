import { useState } from "react";
import "./Settings.css";
import { useTierStatus } from "../hooks/useTierStatus";
import { useProfile } from "../hooks/useProfile";
import { useCheckins } from "../hooks/useCheckins";
import { useSafetyStatus } from "../hooks/useSafetyStatus";
import { MemoryPanel } from "../components/MemoryPanel";
import { SkillsPanel } from "../components/SkillsPanel";
import { ProfilesPanel } from "../components/ProfilesPanel";
import { HistoryPanel } from "../components/HistoryPanel";
import { getStoredTheme, setStoredTheme, type Theme } from "../lib/theme";
import { friendlyActionError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import * as notifications from "../lib/notifications";

export function Settings() {
  const { status, error } = useTierStatus();
  const { profile, error: profileError, setSpeakReplies } = useProfile();
  const { status: checkinStatus, error: checkinError } = useCheckins();
  const { status: safetyStatus, error: safetyError } = useSafetyStatus();
  const { showAlert } = useAlert();
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [speakRepliesBusy, setSpeakRepliesBusy] = useState(false);
  const [speakRepliesError, setSpeakRepliesError] = useState<string | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(notifications.isEnabledPreference);

  const handleThemeChange = (next: Theme) => {
    setTheme(next);
    setStoredTheme(next);
  };

  const handleSpeakRepliesToggle = async () => {
    if (!profile) return;
    setSpeakRepliesBusy(true);
    setSpeakRepliesError(null);
    try {
      await setSpeakReplies(!profile.speak_replies);
      showAlert({ type: "success", message: "Reply voice setting updated." });
    } catch (err) {
      const message = friendlyActionError(err, "Settings.speakReplies", "Couldn't update that setting.");
      setSpeakRepliesError(message);
      showAlert({ type: "error", message });
    } finally {
      setSpeakRepliesBusy(false);
    }
  };

  const handleNotificationsToggle = async (next: boolean) => {
    if (next) {
      const permission = await notifications.requestPermission();
      if (permission !== "granted") {
        showAlert({ type: "error", message: "Notifications weren't allowed — check your browser/OS settings." });
        return;
      }
    }
    notifications.setEnabledPreference(next);
    setNotificationsEnabled(next);
    showAlert({ type: "success", message: next ? "Desktop notifications on." : "Desktop notifications off." });
  };

  return (
    <div className="settings">
      <h1>Settings</h1>

      <section className="settings__section">
        <h2>Hardware &amp; performance</h2>
        {error && <p className="settings__error">{error}</p>}
        {status ? (
          <dl className="settings__grid">
            <div>
              <dt>Tier</dt>
              <dd>{status.tier}</dd>
            </div>
            <div>
              <dt>RAM</dt>
              <dd>{status.hardware.ram_gb} GB</dd>
            </div>
            <div>
              <dt>GPU</dt>
              <dd>{status.hardware.gpu_name ?? "None detected"}</dd>
            </div>
            <div>
              <dt>VRAM</dt>
              <dd>{status.hardware.vram_gb} GB</dd>
            </div>
            <div>
              <dt>Speech engine</dt>
              <dd>{status.tts_engine}</dd>
            </div>
          </dl>
        ) : (
          !error && <p className="settings__hint">Reading hardware…</p>
        )}
      </section>

      <section className="settings__section">
        <h2>Appearance</h2>
        <div className="settings__segmented">
          {(["system", "dark", "light"] as const).map((option) => (
            <button
              key={option}
              className={`settings__segment${theme === option ? " settings__segment--active" : ""}`}
              onClick={() => handleThemeChange(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </section>

      <section className="settings__section">
        <h2>Desktop notifications</h2>
        <p className="settings__hint">
          Local OS notifications only — no external push service, nothing leaves this device. Only
          fires when the app isn't in focus, alongside the in-app alert you'd see either way.
        </p>
        <div className="settings__segmented">
          {[
            { value: true, label: "On" },
            { value: false, label: "Off" },
          ].map((option) => (
            <button
              key={String(option.value)}
              className={`settings__segment${notificationsEnabled === option.value ? " settings__segment--active" : ""}`}
              onClick={() => option.value !== notificationsEnabled && handleNotificationsToggle(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </section>

      <section className="settings__section">
        <h2>Profile</h2>
        {profileError && <p className="settings__error">{profileError}</p>}
        {profile ? (
          <dl className="settings__grid">
            <div>
              <dt>Name</dt>
              <dd>{profile.name || "—"}</dd>
            </div>
            <div>
              <dt>Companion name</dt>
              <dd>{profile.companion_name || "—"}</dd>
            </div>
            <div>
              <dt>Age range</dt>
              <dd>{profile.age_range ?? "—"}</dd>
            </div>
            <div>
              <dt>Voice</dt>
              <dd>{profile.preferred_voice}</dd>
            </div>
            <div>
              <dt>What's on your mind</dt>
              <dd>{profile.stressors.length ? profile.stressors.join(", ") : "—"}</dd>
            </div>
          </dl>
        ) : null}
        {profile && (
          <div className="settings__field">
            <span className="settings__field-label">Speak replies aloud</span>
            <div className="settings__segmented">
              {[
                { value: true, label: "On" },
                { value: false, label: "Off" },
              ].map((option) => (
                <button
                  key={String(option.value)}
                  className={`settings__segment${profile.speak_replies === option.value ? " settings__segment--active" : ""}`}
                  onClick={() => option.value !== profile.speak_replies && handleSpeakRepliesToggle()}
                  disabled={speakRepliesBusy}
                >
                  {option.label}
                </button>
              ))}
            </div>
            {speakRepliesError && <p className="settings__error">{speakRepliesError}</p>}
          </div>
        )}
        {!profile && !profileError && <p className="settings__hint">No profile saved yet.</p>}
      </section>

      <section className="settings__section">
        <h2>Profiles</h2>
        <p className="settings__hint">
          Each profile has its own memory, check-ins, and history. Only one is active at a time.
        </p>
        <ProfilesPanel />
      </section>

      <section className="settings__section">
        <h2>Memory</h2>
        <p className="settings__hint">
          The companion saves things it learns about you quietly, in the background — nothing here is
          ever hidden from you. Browse, correct, or remove anything below.
        </p>
        <MemoryPanel />
      </section>

      <section className="settings__section">
        <h2>Conversation history</h2>
        <p className="settings__hint">
          Past turns are kept, encrypted, so you can replay a previous reply on demand — nothing is
          cached as audio, it's re-spoken fresh each time. Delete anything you don't want kept.
        </p>
        <HistoryPanel />
      </section>

      <section className="settings__section">
        <h2>Support techniques</h2>
        <p className="settings__hint">
          Reference material the companion can draw on mid-conversation — grounding, validation,
          reframing, and more. This library is a starting draft and hasn't yet been reviewed by a
          licensed mental health professional; treat it as psychoeducational, not clinical, guidance.
        </p>
        <SkillsPanel />
      </section>

      <section className="settings__section">
        <h2>Check-ins</h2>
        {checkinError && <p className="settings__error">{checkinError}</p>}
        {checkinStatus ? (
          <dl className="settings__grid">
            <div>
              <dt>Last check-in</dt>
              <dd>
                {checkinStatus.last_checkin_at
                  ? new Date(checkinStatus.last_checkin_at).toLocaleDateString()
                  : "Never yet"}
              </dd>
            </div>
            <div>
              <dt>Days since</dt>
              <dd>
                {checkinStatus.days_since_last_checkin === null
                  ? "—"
                  : checkinStatus.days_since_last_checkin}
              </dd>
            </div>
          </dl>
        ) : (
          !checkinError && <p className="settings__hint">Reading check-in status…</p>
        )}
      </section>

      <section className="settings__section">
        <h2>Safety</h2>
        <p className="settings__hint">
          A crisis detector runs locally before every reply. If it triggers, the companion skips its
          usual reply and shares crisis-line information instead — nothing here is ever hidden from
          you. To change your emergency contact, go through onboarding again.
        </p>
        {profileError && <p className="settings__error">{profileError}</p>}
        {profile && (
          <dl className="settings__grid">
            <div>
              <dt>Emergency contact</dt>
              <dd>{profile.emergency_contact_consent ? "Enabled" : "Not set up"}</dd>
            </div>
            {profile.emergency_contact_consent && (
              <div>
                <dt>Contact</dt>
                <dd>
                  {profile.emergency_contact_name || "—"} (
                  {profile.emergency_contact_method === "email" ? "email" : "text"})
                </dd>
              </div>
            )}
          </dl>
        )}
        {safetyError && <p className="settings__error">{safetyError}</p>}
        {safetyStatus ? (
          <dl className="settings__grid">
            <div>
              <dt>Crisis moments (7 days)</dt>
              <dd>{safetyStatus.recent_crisis_events}</dd>
            </div>
            <div>
              <dt>Last contact made</dt>
              <dd>
                {safetyStatus.last_escalation_at
                  ? new Date(safetyStatus.last_escalation_at).toLocaleDateString()
                  : "Never"}
              </dd>
            </div>
          </dl>
        ) : (
          !safetyError && <p className="settings__hint">Reading safety status…</p>
        )}
      </section>
    </div>
  );
}
