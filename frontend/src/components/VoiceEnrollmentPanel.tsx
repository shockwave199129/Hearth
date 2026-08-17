import { useCallback, useEffect, useRef, useState } from "react";
import "./VoiceEnrollmentPanel.css";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { TARGET_SAMPLE_RATE } from "../lib/audio";
import { friendlyActionError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import {
  enrollVoice,
  forgetVoice,
  getEnrollmentStatus,
  recordVoiceConsent,
  type EnrollmentStatus,
} from "../lib/voiceEnrollment";

/** Settings → Your voice. Opt-in speaker enrollment.
 *
 * This panel stores biometric data, so it is written to be refusable and
 * reversible at every step: nothing is recorded until the user presses
 * record, nothing is sent until they press save, samples can be discarded
 * before sending, and a stored voiceprint can be deleted on its own without
 * touching the rest of the profile.
 *
 * It also states plainly what the feature does and does not do. The honest
 * version is narrow — an unrecognised voice is still heard, answered and
 * kept; it just does not shape what Hearth remembers — and overselling it as
 * "only you can talk to it" would be a false security claim. */
export function VoiceEnrollmentPanel() {
  const { showAlert } = useAlert();
  const [status, setStatus] = useState<EnrollmentStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [samples, setSamples] = useState<Float32Array[]>([]);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const cancelledRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const next = await getEnrollmentStatus();
      if (!cancelledRef.current) setStatus(next);
    } catch (err) {
      if (!cancelledRef.current) {
        setLoadError(friendlyActionError(err, "VoiceEnrollment.status", "Couldn't read voice settings."));
      }
    }
  }, []);

  useEffect(() => {
    cancelledRef.current = false;
    void refresh();
    return () => {
      cancelledRef.current = true;
    };
  }, [refresh]);

  const onUtterance = useCallback((audio: Float32Array) => {
    setSamples((prev) => [...prev, audio]);
  }, []);
  const { state: recorderState, error: recorderError, start, stop } = useAudioRecorder(onUtterance);

  const required = status?.required_samples ?? 3;
  const minSeconds = status?.min_seconds_per_sample ?? 2;
  const seconds = (sample: Float32Array) => sample.length / TARGET_SAMPLE_RATE;
  // Mirrors the server's per-sample floor so the user finds out here, rather
  // than by having the whole enrollment rejected after recording all of them.
  const usable = samples.filter((s) => seconds(s) >= minSeconds);

  const agree = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await recordVoiceConsent();
      await refresh();
    } catch (err) {
      const message = friendlyActionError(err, "VoiceEnrollment.consent", "Couldn't record your agreement.");
      setActionError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true);
    setActionError(null);
    try {
      const next = await enrollVoice(usable);
      setStatus(next);
      setSamples([]);
      showAlert({ type: "success", message: "Your voice is set up." });
    } catch (err) {
      // enrollVoice surfaces the server's own message for the recoverable
      // cases ("those didn't sound like the same voice") — show it as-is.
      const message = err instanceof Error ? err.message : "Couldn't save your voice.";
      setActionError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusy(false);
    }
  };

  const forget = async () => {
    if (!window.confirm("Delete the voiceprint Hearth stored for you? Your conversations and memories are not affected.")) {
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      await forgetVoice();
      setSamples([]);
      await refresh();
      showAlert({ type: "success", message: "Your voiceprint was deleted." });
    } catch (err) {
      const message = friendlyActionError(err, "VoiceEnrollment.forget", "Couldn't delete your voiceprint.");
      setActionError(message);
      showAlert({ type: "error", message });
    } finally {
      setBusy(false);
    }
  };

  if (loadError) return <p className="settings__error">{loadError}</p>;
  if (!status) return <p className="settings__hint">Reading voice settings…</p>;

  if (!status.model_available) {
    return (
      <>
        <p className="settings__hint">
          Voice recognition isn't installed. It's optional — Hearth works fully without it. To add
          it, run <code>python scripts/fetch_voice_models.py</code> and restart.
        </p>
        {status.vad_available && (
          <p className="settings__hint">
            Background-noise filtering is active, so a television or a fan won't be mistaken for you
            talking.
          </p>
        )}
      </>
    );
  }

  return (
    <>
      <p className="settings__hint">
        If more than one person is around, Hearth can tell whether it's you speaking. When it isn't,
        the conversation still happens normally — it just won't add what it heard to what it
        remembers about you.
      </p>
      <p className="settings__hint">
        This stores a voiceprint: a mathematical summary of your voice, encrypted on this device. It
        never leaves your machine, it isn't included in a data export, and you can delete it here at
        any time. It can't tell whether it's really you with certainty, so don't treat it as a lock.
      </p>

      {status.enrolled ? (
        <>
          <dl className="settings__grid">
            <div>
              <dt>Voiceprint</dt>
              <dd>Stored</dd>
            </div>
            <div>
              <dt>Set up</dt>
              <dd>{status.enrolled_at ? new Date(status.enrolled_at).toLocaleDateString() : "—"}</dd>
            </div>
            <div>
              <dt>Recordings used</dt>
              <dd>{status.sample_count ?? "—"}</dd>
            </div>
            <div>
              {/* The published retention schedule, shown as a date rather than
                  a policy sentence — see docs/compliance.md §6. */}
              <dt>Deleted automatically by</dt>
              <dd>{status.expires_at ? new Date(status.expires_at).toLocaleDateString() : "—"}</dd>
            </div>
          </dl>
          <div className="settings__actions">
            <button className="settings__danger-button" onClick={() => void forget()} disabled={busy}>
              {busy ? "Working…" : "Delete my voiceprint"}
            </button>
          </div>
          <p className="settings__hint">
            Sounding different than usual — a cold, a bad night's sleep — can stop it recognising
            you. Recording again replaces the old voiceprint.
          </p>
        </>
      ) : null}

      {!status.consent_current ? (
        <div className="voice-enroll">
          <p className="settings__field-label">
            {status.consent_recorded
              ? "What you agreed to has changed — please read it again"
              : "Before you set this up"}
          </p>
          {/* Rendered from the server's copy, never a local duplicate: the
              version recorded against the profile has to correspond to the
              exact text shown here. */}
          <p className="voice-enroll__consent">{status.consent_text}</p>
          <div className="settings__actions">
            <button className="settings__button" onClick={() => void agree()} disabled={busy}>
              {busy ? "Saving…" : "I understand — turn this on"}
            </button>
          </div>
          {actionError && <p className="settings__error">{actionError}</p>}
        </div>
      ) : (
      <div className="voice-enroll">
        <p className="settings__field-label">
          {status.enrolled ? "Record again to replace it" : "Set up voice recognition"}
        </p>
        <p className="settings__hint">
          {required} recordings, at least {minSeconds} seconds each. Somewhere quiet, with only you
          speaking — say whatever you like.
        </p>
        <ol className="voice-enroll__list">
          {Array.from({ length: Math.max(required, samples.length) }, (_, i) => {
            const sample = samples[i];
            const tooShort = sample !== undefined && seconds(sample) < minSeconds;
            return (
              <li
                key={i}
                className={`voice-enroll__item${sample ? " voice-enroll__item--done" : ""}${
                  tooShort ? " voice-enroll__item--short" : ""
                }`}
              >
                {sample
                  ? tooShort
                    ? `Recording ${i + 1} — too short (${seconds(sample).toFixed(1)}s), it won't be used`
                    : `Recording ${i + 1} — ${seconds(sample).toFixed(1)}s`
                  : `Recording ${i + 1} — not yet`}
              </li>
            );
          })}
        </ol>
        {recorderError && <p className="settings__error">{recorderError}</p>}
        <div className="settings__actions">
          {recorderState === "listening" ? (
            <button className="settings__button" onClick={stop}>
              Stop recording
            </button>
          ) : (
            <button
              className="settings__button"
              onClick={() => void start()}
              disabled={busy || recorderState === "requesting"}
            >
              {recorderState === "requesting" ? "Starting…" : "Record"}
            </button>
          )}
          <button
            className="settings__button"
            onClick={() => void save()}
            disabled={busy || usable.length < required}
          >
            {busy ? "Saving…" : `Save (${usable.length}/${required})`}
          </button>
          {samples.length > 0 && (
            <button className="settings__button" onClick={() => setSamples([])} disabled={busy}>
              Discard recordings
            </button>
          )}
        </div>
        {actionError && <p className="settings__error">{actionError}</p>}
        {samples.length > 0 && (
          <p className="settings__hint">
            Nothing has been saved yet. These recordings are only in memory until you press Save,
            and discarding them keeps no copy.
          </p>
        )}
        {status.consented_at && (
          <p className="settings__hint">
            You agreed to this on {new Date(status.consented_at).toLocaleDateString()}. Deleting your
            voiceprint also withdraws that agreement.
          </p>
        )}
      </div>
      )}
    </>
  );
}
