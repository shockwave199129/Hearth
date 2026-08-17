import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./Settings.css";
import { useTierStatus } from "../hooks/useTierStatus";
import { useProfile } from "../hooks/useProfile";
import { useCheckins } from "../hooks/useCheckins";
import { useSafetyStatus } from "../hooks/useSafetyStatus";
import { MemoryPanel } from "../components/MemoryPanel";
import { SkillsPanel } from "../components/SkillsPanel";
import { ProfilesPanel } from "../components/ProfilesPanel";
import { VoiceEnrollmentPanel } from "../components/VoiceEnrollmentPanel";
import { AboutPanel } from "../components/AboutPanel";
import { getStoredTheme, setStoredTheme, type Theme } from "../lib/theme";
import { friendlyActionError } from "../lib/errors";
import { backendFetch } from "../lib/backendFetch";
import { useMicrophone } from "../hooks/useMicrophone";
import { useAlert } from "../lib/alerts";
import type { MicHardware, MicPermission } from "../lib/microphone";
import * as notifications from "../lib/notifications";
import {
  VOICE_OPTIONS,
  VOICE_STYLE_OPTIONS,
  type PreferredVoice,
  type VoiceStyleId,
} from "../lib/voiceStyles";

/** Response of POST /api/data/export — see backend/app/data_export.py.
 * `incomplete` is keyed by store with a human-readable reason, and is empty
 * on a clean export; it is surfaced rather than swallowed because a
 * partial export of "all your data" must not look like a complete one. */
type DataExportResult = {
  path: string;
  counts: {
    long_term_memories: number;
    episodic_memories: number;
    semantic_memories: number;
    transcript_messages: number;
  };
  incomplete: Record<string, string>;
};

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
  const {
    hardware: micHardware,
    permission: micPermission,
    preferred: voiceInputPreferred,
    setPreferred: setVoiceInputPreferred,
  } = useMicrophone();
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [speakRepliesBusy, setSpeakRepliesBusy] = useState(false);
  const [speakRepliesError, setSpeakRepliesError] = useState<string | null>(null);
  const [prefsBusy, setPrefsBusy] = useState(false);
  const [prefsError, setPrefsError] = useState<string | null>(null);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(notifications.isEnabledPreference);
  const [dataResetBusy, setDataResetBusy] = useState(false);
  const [dataResetError, setDataResetError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<DataExportResult | null>(null);

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

  const exportData = async () => {
    // Confirmed rather than one-click: the export is readable by anyone who
    // can read the folder, unlike everything Hearth normally writes.
    if (
      !window.confirm(
        "This writes your profile, memories, and full conversation to plain files in your home folder. They are NOT encrypted — anyone who can open that folder can read them. Continue?",
      )
    ) {
      return;
    }
    setExportBusy(true);
    setExportError(null);
    try {
      const response = await backendFetch("/api/data/export", { method: "POST" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const result = (await response.json()) as DataExportResult;
      setExportResult(result);
      showAlert({ type: "success", message: "Your data was exported." });
    } catch (err) {
      const message = friendlyActionError(err, "Settings.exportData", "Couldn't export your data.");
      setExportError(message);
      showAlert({ type: "error", message });
    } finally {
      setExportBusy(false);
    }
  };

  const resetLocalData = async () => {
    if (
      !window.confirm(
        "This removes downloaded models, Python packages, conversations, memories, and crash logs. Your profile identity and preferences stay. You'll complete setup again. Continue?",
      )
    ) {
      return;
    }
    setDataResetBusy(true);
    setDataResetError(null);
    try {
      const response = await backendFetch("/api/data/reset", { method: "POST" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      window.location.reload();
    } catch (err) {
      const message = friendlyActionError(err, "Settings.resetLocalData", "Couldn't reset local data.");
      setDataResetError(message);
      showAlert({ type: "error", message });
    } finally {
      setDataResetBusy(false);
    }
  };

  const uninstallMacos = async () => {
    if (
      !window.confirm(
        "This removes downloaded models, Python packages, conversations, memories, and crash logs, then moves Hearth to the Trash. Your profile identity and preferences stay for reinstall. Continue?",
      )
    ) {
      return;
    }
    setDataResetBusy(true);
    setDataResetError(null);
    try {
      const response = await backendFetch("/api/data/reset", { method: "POST" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      await invoke("move_macos_app_to_trash");
      window.close();
    } catch (err) {
      const message = friendlyActionError(err, "Settings.uninstallMacos", "Couldn't uninstall Hearth.");
      setDataResetError(message);
      showAlert({ type: "error", message });
      setDataResetBusy(false);
    }
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
            <MicrophoneStatusItems hardware={micHardware} permission={micPermission} />
          </dl>
        ) : (
          <>
            {!error && <p className="settings__hint">Reading hardware…</p>}
            <dl className="settings__grid">
              <MicrophoneStatusItems hardware={micHardware} permission={micPermission} />
            </dl>
          </>
        )}
        <div className="settings__field">
          <span className="settings__field-label">Voice input</span>
          <p className="settings__hint">
            {micHardware === "absent"
              ? "No microphone is available on this device. Typing still works."
              : micPermission === "denied"
                ? "Microphone access is blocked in the browser or OS. Allow it to talk out loud."
                : "Use the microphone on Talk, or turn this off to type only."}
          </p>
          <div className="settings__segmented">
            {[
              { value: true, label: "On" },
              { value: false, label: "Off" },
            ].map((option) => (
              <button
                key={String(option.value)}
                className={`settings__segment${voiceInputPreferred === option.value ? " settings__segment--active" : ""}`}
                onClick={() => option.value !== voiceInputPreferred && setVoiceInputPreferred(option.value)}
                disabled={micHardware === "absent" || micPermission === "denied"}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
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

      <section className="settings__section">
        <h2>Your voice</h2>
        <VoiceEnrollmentPanel />
      </section>

      <section className="settings__section">
        <h2>Your data</h2>
        <p className="settings__hint">
          Export writes your profile, everything {profile?.companion_name ?? "your companion"} remembers,
          and your whole conversation to plain files in your home folder — yours to keep, move, or open
          without Hearth. Nothing is removed from the app, and nothing is uploaded anywhere.
        </p>
        <p className="settings__hint">
          Those files are not encrypted, unlike the copies Hearth keeps. Anyone who can open the folder
          can read them.
        </p>
        <div className="settings__actions">
          <button className="settings__button" onClick={() => void exportData()} disabled={exportBusy}>
            {exportBusy ? "Exporting…" : "Export my data"}
          </button>
        </div>
        {exportError && <p className="settings__error">{exportError}</p>}
        {exportResult && (
          <>
            <div className="settings__field">
              <span className="settings__field-label">Saved to</span>
              <p className="settings__path">{exportResult.path}</p>
            </div>
            <dl className="settings__grid">
              <div>
                <dt>Messages</dt>
                <dd>{exportResult.counts.transcript_messages}</dd>
              </div>
              <div>
                <dt>Memories</dt>
                <dd>
                  {exportResult.counts.long_term_memories +
                    exportResult.counts.episodic_memories +
                    exportResult.counts.semantic_memories}
                </dd>
              </div>
            </dl>
            {Object.entries(exportResult.incomplete).map(([key, reason]) => (
              <p key={key} className="settings__error">
                Part of your data couldn't be exported ({key.replace(/_error$/, "")}): {reason}
              </p>
            ))}
          </>
        )}
      </section>

      <section className="settings__section">
        <h2>About</h2>
        <AboutPanel />
      </section>

      <section className="settings__section settings__section--danger">
        <h2>Local data</h2>
        <p className="settings__hint">
          Reset removes downloaded models, Python packages, conversations, memories, learning data, and
          crash logs. Your profile identity and preferences remain for reinstall.
        </p>
        <div className="settings__actions">
          <button className="settings__danger-button" onClick={() => void resetLocalData()} disabled={dataResetBusy}>
            {dataResetBusy ? "Resetting…" : "Reset local data"}
          </button>
          {navigator.userAgent.includes("Macintosh") && (
            <button className="settings__danger-button" onClick={() => void uninstallMacos()} disabled={dataResetBusy}>
              Uninstall Hearth…
            </button>
          )}
        </div>
        {dataResetError && <p className="settings__error">{dataResetError}</p>}
      </section>
    </div>
  );
}

function MicrophoneStatusItems({
  hardware,
  permission,
}: {
  hardware: MicHardware;
  permission: MicPermission;
}) {
  return (
    <>
      <div>
        <dt>Microphone</dt>
        <dd>
          {hardware === "present" ? "Detected" : hardware === "absent" ? "Not detected" : "Checking…"}
        </dd>
      </div>
      <div>
        <dt>Mic access</dt>
        <dd>
          {permission === "granted"
            ? "Allowed"
            : permission === "denied"
              ? "Blocked"
              : permission === "prompt"
                ? "Not asked yet"
                : "—"}
        </dd>
      </div>
    </>
  );
}
