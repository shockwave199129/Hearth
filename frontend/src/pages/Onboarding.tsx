import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Onboarding.css";
import { useProfile } from "../hooks/useProfile";
import { friendlyFetchError } from "../lib/errors";
import { useAlert } from "../lib/alerts";
import {
  DEFAULT_VOICE,
  DEFAULT_VOICE_STYLE,
  VOICE_OPTIONS,
  VOICE_STYLE_OPTIONS,
  type PreferredVoice,
  type VoiceStyleId,
} from "../lib/voiceStyles";

const AGE_RANGES = ["18–24", "25–34", "35–44", "45–54", "55+"];
const STRESSOR_OPTIONS = ["Work", "Family", "Finances", "Health", "Relationships", "Sleep"];
const FORMALITY_OPTIONS = [
  { id: "casual", label: "Casual" },
  { id: "neutral", label: "Neutral" },
  { id: "formal", label: "Formal" },
] as const;
const LENGTH_OPTIONS = [
  { id: "short", label: "Short" },
  { id: "balanced", label: "Balanced" },
  { id: "long", label: "Longer" },
] as const;

interface OnboardingData {
  name: string;
  companionName: string;
  ageRange: string | null;
  profession: string;
  stressors: string[];
  preferredVoice: PreferredVoice;
  voiceStyle: VoiceStyleId;
  communicationFormality: "casual" | "neutral" | "formal";
  responseLength: "short" | "balanced" | "long";
  emergencyContactConsent: boolean;
  emergencyContactName: string;
  emergencyContactMethod: "sms" | "email";
  emergencyContactValue: string;
  adultAttested: boolean;
}

// "What this is" is last rather than first on purpose: it's the gate on
// actually starting, and nothing is persisted until the final step submits.
// See docs/compliance.md for why the disclosure exists at all.
const STEPS = ["Names", "About you", "What's on your mind", "Style", "Voice", "Safety", "What this is"] as const;

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
    preferredVoice: DEFAULT_VOICE,
    voiceStyle: DEFAULT_VOICE_STYLE,
    communicationFormality: "casual",
    responseLength: "balanced",
    emergencyContactConsent: false,
    emergencyContactName: "",
    emergencyContactMethod: "sms",
    emergencyContactValue: "",
    adultAttested: false,
  });

  const isLastStep = step === STEPS.length - 1;
  // The backend rejects an unattested profile too (api/profile.py) — this
  // only keeps the user from submitting something it would refuse.
  const blockedOnAttestation = isLastStep && !data.adultAttested;

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
        voice_style: data.voiceStyle,
        companion_name: data.companionName.trim() || "Companion",
        communication_formality: data.communicationFormality,
        response_length: data.responseLength,
        speak_replies: true,
        emergency_contact_consent: data.emergencyContactConsent,
        emergency_contact_name: data.emergencyContactConsent ? data.emergencyContactName.trim() || null : null,
        emergency_contact_method: data.emergencyContactConsent ? data.emergencyContactMethod : null,
        emergency_contact_value: data.emergencyContactConsent ? data.emergencyContactValue.trim() || null : null,
        adult_attested: data.adultAttested,
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
            <h1>How should I write to you?</h1>
            <p className="onboarding__hint">You can change any of this later in Settings.</p>
            <div className="onboarding__field">
              <span>Formality</span>
              <div className="onboarding__chip-row">
                {FORMALITY_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={`onboarding__chip${data.communicationFormality === option.id ? " onboarding__chip--active" : ""}`}
                    onClick={() => setData((p) => ({ ...p, communicationFormality: option.id }))}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="onboarding__field">
              <span>Response length</span>
              <div className="onboarding__chip-row">
                {LENGTH_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={`onboarding__chip${data.responseLength === option.id ? " onboarding__chip--active" : ""}`}
                    onClick={() => setData((p) => ({ ...p, responseLength: option.id }))}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="onboarding__step">
            <h1>And how should I sound?</h1>
            <p className="onboarding__hint">
              This only changes the spoken voice — you can switch it any time in Settings.
            </p>
            <div className="onboarding__field">
              <span>Voice</span>
              <div className="onboarding__voice-row">
                {VOICE_OPTIONS.map((voice) => (
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
            <div className="onboarding__field">
              <span>Way of speaking</span>
              <div className="onboarding__voice-row">
                {VOICE_STYLE_OPTIONS.map((style) => (
                  <button
                    key={style.id}
                    type="button"
                    className={`onboarding__voice-card${data.voiceStyle === style.id ? " onboarding__voice-card--active" : ""}`}
                    onClick={() => setData((p) => ({ ...p, voiceStyle: style.id }))}
                  >
                    <strong>{style.label}</strong>
                    <span className="onboarding__voice-blurb">{style.blurb}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {step === 5 && (
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

        {step === 6 && (
          <div className="onboarding__step">
            <h1>Before we start, plainly</h1>
            <p className="onboarding__hint">
              {data.companionName.trim() || "Your companion"} is an AI — software running on this
              machine. Not a person, and not a therapist. It won't ever claim otherwise, and if you
              ask it directly, it will tell you the same thing.
            </p>
            <ul className="onboarding__disclosure">
              <li>
                <strong>It's for reflection, not treatment.</strong> Talking things through,
                journaling, remembering what matters to you. It doesn't diagnose or treat anything,
                and it isn't a substitute for professional care.
              </li>
              <li>
                <strong>It isn't an emergency service.</strong> If things get serious it will help
                you find real human support — but in an emergency, contact emergency services.
              </li>
              <li>
                <strong>Your conversations stay here.</strong> On this device, encrypted. The one
                exception is the crisis contact you just chose, if you chose one.
              </li>
              <li>
                <strong>It's built for adults.</strong> Hearth is intended for people 18 and over.
              </li>
            </ul>
            <label className="onboarding__field onboarding__field--checkbox">
              <input
                type="checkbox"
                checked={data.adultAttested}
                onChange={(e) => setData((p) => ({ ...p, adultAttested: e.target.checked }))}
              />
              <span>I'm 18 or older, and I understand I'll be talking to software.</span>
            </label>
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
            disabled={submitting || blockedOnAttestation}
            title={blockedOnAttestation ? "Confirm the box above to continue" : undefined}
          >
            {isLastStep ? (submitting ? "Saving…" : "Start talking") : "Continue"}
          </button>
        </div>
      </div>
    </div>
  );
}
