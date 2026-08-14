import { useState } from "react";
import "./DisclosureGate.css";
import { useProfile } from "../hooks/useProfile";
import { friendlyFetchError } from "../lib/errors";

/** Shown when a profile exists but predates the onboarding disclosure step,
 * so it has never seen it (`adult_attested === false`). Blocks the app until
 * accepted — see docs/compliance.md.
 *
 * Deliberately not auto-migrated server-side: a profile created before the
 * gate existed genuinely never saw the disclosure, and recording otherwise
 * would make the attestation audit trail false. So we ask once.
 *
 * The wording here intentionally mirrors the onboarding step rather than
 * being a shorter "just click OK" variant — someone who never saw it should
 * get the same information a new user gets, not an abridged version. */
export function DisclosureGate() {
  const { profile, acceptDisclosure } = useProfile();
  const [checked, setChecked] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const companion = profile?.companion_name?.trim() || "Your companion";

  const accept = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await acceptDisclosure();
    } catch (err) {
      setError(friendlyFetchError(err, "DisclosureGate.accept"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="disclosure-gate" role="dialog" aria-modal="true" aria-labelledby="disclosure-gate-title">
      <div className="disclosure-gate__card">
        <h1 id="disclosure-gate-title">One thing before we carry on</h1>
        <p className="disclosure-gate__hint">
          We've made this explicit rather than assumed. Nothing about your saved conversations or
          memories has changed — this is just information you should have had from the start.
        </p>
        <ul className="disclosure-gate__list">
          <li>
            <strong>{companion} is an AI.</strong> Software running on this machine. Not a person,
            and not a therapist. Ask it directly and it will tell you the same thing.
          </li>
          <li>
            <strong>It's for reflection, not treatment.</strong> It doesn't diagnose or treat
            anything, and it isn't a substitute for professional care.
          </li>
          <li>
            <strong>It isn't an emergency service.</strong> It will help you find real human
            support — but in an emergency, contact emergency services.
          </li>
          <li>
            <strong>It's built for adults.</strong> Hearth is intended for people 18 and over.
          </li>
        </ul>
        <label className="disclosure-gate__confirm">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
          />
          <span>I'm 18 or older, and I understand I'm talking to software.</span>
        </label>
        {error && <p className="disclosure-gate__error">{error}</p>}
        <button
          type="button"
          className="disclosure-gate__button"
          onClick={() => void accept()}
          disabled={!checked || submitting}
          title={!checked ? "Confirm the box above to continue" : undefined}
        >
          {submitting ? "Saving…" : "Continue"}
        </button>
      </div>
    </div>
  );
}
