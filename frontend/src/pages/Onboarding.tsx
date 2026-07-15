import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Onboarding.css";
import { useProfile } from "../hooks/useProfile";
import { friendlyFetchError } from "../lib/errors";
import { useAlert } from "../lib/alerts";

const AGE_RANGES = ["18–24", "25–34", "35–44", "45–54", "55+"];
const STRESSOR_OPTIONS = ["Work", "Family", "Finances", "Health", "Relationships", "Sleep"];
const VOICES = [
  { id: "female", label: "Warm & even" },
  { id: "male", label: "Calm & low" },
] as const;

interface OnboardingData {
  name: string;
  companionName: string;
  ageRange: string | null;
  profession: string;
  stressors: string[];
  preferredVoice: "female" | "male";
  emergencyContactConsent: boolean;
  emergencyContactName: string;
  emergencyContactMethod: "sms" | "email";
  emergencyContactValue: string;
}

const STEPS = ["Names", "About you", "What's on your mind", "Voice", "Safety"] as const;

export function Onboarding() {
  const navigate = useNavigate();
  const { submitOnboarding } = useProfile();
  const { showAlert } = useAlert();
  const [step, setStep] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [data, setData] = useState<OnboardingData>({
    name: "",
    companionName: "",
    ageRange: null,
    profession: "",
    stressors: [],
    preferredVoice: "female",
    emergencyContactConsent: false,
    emergencyContactName: "",
    emergencyContactMethod: "sms",
    emergencyContactValue: "",
  });

  const isLastStep = step === STEPS.length - 1;

  const toggleStressor = (option: string) => {
    setData((prev) => ({
      ...prev,
      stressors: prev.stressors.includes(option)
        ? prev.stressors.filter((s) => s !== option)
        : [...prev.stressors, option],
    }));
  };

  const finish = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitOnboarding({
        name: data.name.trim() || "friend",
        age_range: data.ageRange,
        gender: null,
        profession: data.profession.trim() || null,
        stressors: data.stressors,
        preferred_voice: data.preferredVoice,
        companion_name: data.companionName.trim() || "Companion",
        speak_replies: true,
        emergency_contact_consent: data.emergencyContactConsent,
        emergency_contact_name: data.emergencyContactConsent ? data.emergencyContactName.trim() || null : null,
        emergency_contact_method: data.emergencyContactConsent ? data.emergencyContactMethod : null,
        emergency_contact_value: data.emergencyContactConsent ? data.emergencyContactValue.trim() || null : null,
      });
      showAlert({ type: "success", message: "Profile ready — welcome in." });
      navigate("/chat", { replace: true });
    } catch (err) {
      setSubmitError(friendlyFetchError(err, "Onboarding.finish"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="onboarding">
      <div className="onboarding__card">
        <div className="onboarding__progress" role="progressbar" aria-valuenow={step + 1} aria-valuemax={STEPS.length}>
          {STEPS.map((label, i) => (
            <span key={label} className={`onboarding__dot${i <= step ? " onboarding__dot--done" : ""}`} />
          ))}
        </div>

        {step === 0 && (
          <div className="onboarding__step">
            <h1>Let's get acquainted</h1>
            <p className="onboarding__hint">So I know how to talk with you — nothing here ever leaves this device.</p>
            <label className="onboarding__field">
              <span>What should I call you?</span>
              <input
                value={data.name}
                onChange={(e) => setData((p) => ({ ...p, name: e.target.value }))}
                placeholder="Your name"
                autoFocus
              />
            </label>
            <label className="onboarding__field">
              <span>What would you like to call me?</span>
              <input
                value={data.companionName}
                onChange={(e) => setData((p) => ({ ...p, companionName: e.target.value }))}
                placeholder="e.g. Sage, River, Companion"
              />
            </label>
          </div>
        )}

        {step === 1 && (
          <div className="onboarding__step">
            <h1>A little about you</h1>
            <p className="onboarding__hint">Optional — skip anything you'd rather not share.</p>
            <div className="onboarding__field">
              <span>Age range</span>
              <div className="onboarding__chip-row">
                {AGE_RANGES.map((range) => (
                  <button
                    key={range}
                    type="button"
                    className={`onboarding__chip${data.ageRange === range ? " onboarding__chip--active" : ""}`}
                    onClick={() => setData((p) => ({ ...p, ageRange: p.ageRange === range ? null : range }))}
                  >
                    {range}
                  </button>
                ))}
              </div>
            </div>
            <label className="onboarding__field">
              <span>What do you do?</span>
              <input
                value={data.profession}
                onChange={(e) => setData((p) => ({ ...p, profession: e.target.value }))}
                placeholder="Optional"
              />
            </label>
          </div>
        )}

        {step === 2 && (
          <div className="onboarding__step">
            <h1>What's been on your mind</h1>
            <p className="onboarding__hint">Pick anything that's felt heavy lately — I'll keep it in mind gently, not bring it up unprompted.</p>
            <div className="onboarding__chip-row">
              {STRESSOR_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`onboarding__chip${data.stressors.includes(option) ? " onboarding__chip--active" : ""}`}
                  onClick={() => toggleStressor(option)}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="onboarding__step">
            <h1>One last thing</h1>
            <p className="onboarding__hint">How should I sound?</p>
            <div className="onboarding__voice-row">
              {VOICES.map((voice) => (
                <button
                  key={voice.id}
                  type="button"
                  className={`onboarding__voice-card${data.preferredVoice === voice.id ? " onboarding__voice-card--active" : ""}`}
                  onClick={() => setData((p) => ({ ...p, preferredVoice: voice.id }))}
                >
                  {voice.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="onboarding__step">
            <h1>One safety net, if you want it</h1>
            <p className="onboarding__hint">
              Completely optional. If things ever get seriously heavy, would you like the companion to
              reach out to someone on your behalf? This is off by default and only ever used alongside
              other safeguards — never as a substitute for emergency services.
            </p>
            <label className="onboarding__field onboarding__field--checkbox">
              <input
                type="checkbox"
                checked={data.emergencyContactConsent}
                onChange={(e) => setData((p) => ({ ...p, emergencyContactConsent: e.target.checked }))}
              />
              <span>If I'm in serious crisis, let the companion notify someone</span>
            </label>
            {data.emergencyContactConsent && (
              <>
                <label className="onboarding__field">
                  <span>Their name</span>
                  <input
                    value={data.emergencyContactName}
                    onChange={(e) => setData((p) => ({ ...p, emergencyContactName: e.target.value }))}
                    placeholder="e.g. Sam"
                  />
                </label>
                <div className="onboarding__field">
                  <span>How should they be contacted?</span>
                  <div className="onboarding__chip-row">
                    {(["sms", "email"] as const).map((method) => (
                      <button
                        key={method}
                        type="button"
                        className={`onboarding__chip${data.emergencyContactMethod === method ? " onboarding__chip--active" : ""}`}
                        onClick={() => setData((p) => ({ ...p, emergencyContactMethod: method }))}
                      >
                        {method === "sms" ? "Text message" : "Email"}
                      </button>
                    ))}
                  </div>
                </div>
                <label className="onboarding__field">
                  <span>{data.emergencyContactMethod === "sms" ? "Their phone number" : "Their email"}</span>
                  <input
                    value={data.emergencyContactValue}
                    onChange={(e) => setData((p) => ({ ...p, emergencyContactValue: e.target.value }))}
                    placeholder={data.emergencyContactMethod === "sms" ? "+1 555 555 5555" : "name@example.com"}
                  />
                </label>
              </>
            )}
          </div>
        )}

        {submitError && <p className="onboarding__error">{submitError}</p>}

        <div className="onboarding__actions">
          <button
            type="button"
            className="onboarding__button onboarding__button--ghost"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0 || submitting}
          >
            Back
          </button>
          <button
            type="button"
            className="onboarding__button onboarding__button--primary"
            onClick={() => (isLastStep ? void finish() : setStep((s) => s + 1))}
            disabled={submitting}
          >
            {isLastStep ? (submitting ? "Saving…" : "Start talking") : "Continue"}
          </button>
        </div>
      </div>
    </div>
  );
}
