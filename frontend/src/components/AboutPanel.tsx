import { useEffect, useState } from "react";
import "./AboutPanel.css";
import { backendFetch } from "../lib/backendFetch";
import { friendlyActionError } from "../lib/errors";

type Component = {
  name: string;
  purpose: string;
  license: string;
  url: string;
  attribution?: string;
  optional?: boolean;
  installed?: boolean;
};

type About = {
  version: string;
  components: Component[];
  required_attributions: string[];
};

/** Settings → About. Version plus third-party credits.
 *
 * This is not decoration: the speaker-embedding model is CC-BY-4.0, which
 * requires attribution wherever it is distributed, and until this panel
 * existed there was nowhere in the app to put one (a launch-gate item in
 * docs/compliance.md). `required_attributions` is pre-filtered server-side to
 * what is actually installed, so rendering it satisfies the obligation
 * without this component needing to know which licences demand a credit. */
export function AboutPanel() {
  const [about, setAbout] = useState<About | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const response = await backendFetch("/api/about");
        if (!response.ok) throw new Error(`status ${response.status}`);
        const body = (await response.json()) as About;
        if (!cancelled) setAbout(body);
      } catch (err) {
        if (!cancelled) setError(friendlyActionError(err, "About.load", "Couldn't load credits."));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p className="settings__error">{error}</p>;
  if (!about) return <p className="settings__hint">Loading…</p>;

  const installed = about.components.filter((c) => c.installed !== false);
  const notInstalled = about.components.filter((c) => c.installed === false);

  return (
    <>
      <dl className="settings__grid">
        <div>
          <dt>Version</dt>
          <dd>Hearth {about.version}</dd>
        </div>
      </dl>

      {about.required_attributions.length > 0 && (
        <div className="about__attribution">
          {about.required_attributions.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      )}

      <p className="settings__field-label">Built with</p>
      <ul className="about__list">
        {installed.map((component) => (
          <li key={component.name}>
            <span className="about__name">{component.name}</span>
            <span className="about__purpose">{component.purpose}</span>
            <span className="about__license">{component.license}</span>
          </li>
        ))}
      </ul>

      {notInstalled.length > 0 && (
        <>
          <p className="settings__field-label">Optional, not installed</p>
          <ul className="about__list about__list--muted">
            {notInstalled.map((component) => (
              <li key={component.name}>
                <span className="about__name">{component.name}</span>
                <span className="about__purpose">{component.purpose}</span>
                <span className="about__license">{component.license}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="settings__hint">
        Hearth runs entirely on this machine. It is software for reflection and everyday
        conversation — not therapy, not medical care, and not an emergency service.
      </p>
    </>
  );
}
