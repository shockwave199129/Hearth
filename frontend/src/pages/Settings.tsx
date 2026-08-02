import { useState } from "react";
import "./Settings.css";
import { useTierStatus } from "../hooks/useTierStatus";
import { useProfile } from "../hooks/useProfile";
import { useCheckins } from "../hooks/useCheckins";
import { useSafetyStatus } from "../hooks/useSafetyStatus";
import { MemoryPanel } from "../components/MemoryPanel";
import { SkillsPanel } from "../components/SkillsPanel";
import { ProfilesPanel } from "../components/ProfilesPanel";
import { getStoredTheme, setStoredTheme, type Theme } from "../lib/theme";
import { friendlyActionError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import * as notifications from "../lib/notifications";
import {
  VOICE_OPTIONS,
  VOICE_STYLE_OPTIONS,
  type PreferredVoice,
  type VoiceStyleId,
} from "../lib/voiceStyles";

export function Settings() {
  const { status, error } = useTierStatus();
  const {
    profile,
    error: profileError,
    setSpeakReplies,
    setCommunicationPreferences,
    setVoicePreferences,
  } = useProfile();
  const { status: checkinStatus, error: checkinError } = useCheckins();
  const { status: safetyStatus, error: safetyError } = useSafetyStatus();
  const { showAlert } = useAlert();
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [speakRepliesBusy, setSpeakRepliesBusy] = useState(false);
  const [speakRepliesError, setSpeakRepliesError] = useState<string | null>(null);
  const [prefsBusy, setPrefsBusy] = useState(false);
  const [prefsError, setPrefsError] = useState<string | null>(null);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
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

  const handleCommunicationPreferencesUpdate = async (
    next: "casual" | "neutral" | "formal",
    responseLength: "short" | "balanced" | "long",
  ) => {
    if (!profile) return;
    setPrefsBusy(true);
    setPrefsError(null);
    try {
      await setCommunicationPreferences({
        communication_formality: next,
        response_length: responseLength,
      });
      showAlert({ type: "success", message: "Communication style updated." });
    } catch (err) {
      const message = friendlyActionError(err, "Settings.communicationPreferences", "Couldn't update that setting.");
      setPrefsError(message);
      showAlert({ type: "error", message });
    } finally {
      setPrefsBusy(false);
    }
  };

  const handleVoicePreferencesUpdate = async (
    preferredVoice: PreferredVoice,
    voiceStyle: VoiceStyleId,
  ) => {
    if (!profile) return;
    setVoiceBusy(true);
    setVoiceError(null);
    try {
      await setVoicePreferences({ preferred_voice: preferredVoice, voice_style: voiceStyle });
      showAlert({ type: "success", message: "Speaking voice updated — you'll hear it on the next reply." });
    } catch (err) {
      const message = friendlyActionError(err, "Settings.voicePreferences", "Couldn't update that setting.");
      setVoiceError(message);
      showAlert({ type: "error", message });
    } finally {
      setVoiceBusy(false);
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
              <dd>
                {profile.preferred_voice} ·{" "}
                {VOICE_STYLE_OPTIONS.find((s) => s.id === profile.voice_style)?.label ??
                  profile.voice_style}
              </dd>
            </div>
            <div>
              <dt>Formality</dt>
              <dd>{profile.communication_formality}</dd>
            </div>
            <div>
              <dt>Response length</dt>
              <dd>{profile.response_length}</dd>
            </div>
            <div>
              <dt>What's on your mind</dt>
              <dd>{profile.stressors.length ? profile.stressors.join(", ") : "—"}</dd>
            </div>
          </dl>
        ) : null}
        {profile && (
          <div className="settings__field">
            <span className="settings__field-label">Communication style</span>
            <div className="settings__segmented">
              {(["casual", "neutral", "formal"] as const).map((option) => (
                <button
                  key={option}
                  className={`settings__segment${profile.communication_formality === option ? " settings__segment--active" : ""}`}
                  onClick={() => option !== profile.communication_formality && handleCommunicationPreferencesUpdate(option, profile.response_length)}
                  disabled={prefsBusy}
                >
                  {option}
                </button>
              ))}
            </div>
            <span className="settings__field-label">Response length</span>
            <div className="settings__segmented">
              {(["short", "balanced", "long"] as const).map((option) => (
                <button
                  key={option}
                  className={`settings__segment${profile.response_length === option ? " settings__segment--active" : ""}`}
                  onClick={() => option !== profile.response_length && handleCommunicationPreferencesUpdate(profile.communication_formality, option)}
                  disabled={prefsBusy}
                >
                  {option}
                </button>
              ))}
            </div>
            {prefsError && <p className="settings__error">{prefsError}</p>}
          </div>
        )}
        {profile && (
          <div className="settings__field">
            <span className="settings__field-label">Speaking voice</span>
            <div className="settings__segmented">
              {VOICE_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  className={`settings__segment${profile.preferred_voice === option.id ? " settings__segment--active" : ""}`}
                  onClick={() =>
                    option.id !== profile.preferred_voice &&
                    handleVoicePreferencesUpdate(option.id, profile.voice_style)
                  }
                  disabled={voiceBusy}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <span className="settings__field-label">Way of speaking</span>
            <div className="settings__segmented">
              {VOICE_STYLE_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  className={`settings__segment${profile.voice_style === option.id ? " settings__segment--active" : ""}`}
                  onClick={() =>
                    option.id !== profile.voice_style &&
                    handleVoicePreferencesUpdate(profile.preferred_voice, option.id)
                  }
                  disabled={voiceBusy}
                  title={option.blurb}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <p className="settings__hint">
              {VOICE_STYLE_OPTIONS.find((s) => s.id === profile.voice_style)?.blurb}
            </p>
            {voiceError && <p className="settings__error">{voiceError}</p>}
          </div>
        )}
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
          Each profile has its own memory and check-ins. Only one is active at a time.
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
