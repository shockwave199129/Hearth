# Regulatory and safety baseline

What Hearth has to satisfy to be put in front of real users, and what is
still missing. Companion architecture and positioning follow from this
document, not the reverse.

**This is engineering notes, not legal advice.** Every statute referenced
here needs verification with qualified US counsel before launch, and the
citations are point-in-time (written August 2026). Treat the *structure* of
the argument as durable and the *specifics* as needing a lawyer.

## The core decision: wellness, not treatment

Hearth is positioned for reflection, journaling, everyday conversation,
habit awareness, and personal continuity. It is **not** positioned as
diagnosing, treating, mitigating, or preventing any condition.

This is the single highest-leverage compliance decision in the project.
FDA's framework turns substantially on **intended use**: software intended
for diagnosis, cure, mitigation, treatment, or prevention of disease can be
regulated as a medical device, while certain low-risk general-wellness
functions fall outside that. Intended use is established by what you claim —
marketing copy, README text, in-app language, screenshots.

Practical consequences that bind the code, not just the website:

- No claim, anywhere shipped, that Hearth treats depression, anxiety, PTSD,
  or any diagnosis.
- No diagnostic output. Hearth must not tell a user they have a condition.
- Clinical vocabulary stays out of responses. Already enforced:
  [`eval/self_check.py`](../backend/app/eval/self_check.py) flags
  `diagnos*`, `disorder`, `clinical`, `syndrome`, `prescri*`, and
  `symptomatology` as anti-patterns.
- Skill content is *coping and reflection* material, not protocolized
  therapy.

Several US states have moved further and restrict AI from delivering
therapy or psychotherapeutic decision-making outright, in some cases
requiring licensed-professional involvement. That makes the
non-therapy positioning not merely prudent but, in those jurisdictions,
load-bearing.

## Companion-AI obligations

Hearth is squarely a "companion" by the definitions now appearing in US
state law — adaptive, human-like responses sustaining a relationship across
multiple interactions. That means companion-specific duties apply regardless
of the wellness positioning.

The obligations converging across jurisdictions:

| Requirement | Status in Hearth |
|---|---|
| Disclosure that the system is AI | Built — onboarding, UI marker, `identity.py`, system prompt |
| Protocol for suicidal ideation / self-harm | Built (`safety2/`), gate not passed |
| Referral to crisis services | Built — `safety2/resources/` |
| Additional protections for known minors | Out of scope — 18+ attested at onboarding |
| Break reminders / session-duration prompts (minors) | Out of scope while 18+ only |
| No engagement-maximizing manipulation | Built as design invariant |
| Audit logging of safety events | Built (`safety2/`) |

The design intent is to meet the strongest reasonable companion-AI baseline
once, rather than patching per state as laws land. State-by-state divergence
is the expensive failure mode.

Beyond statute, the **FTC has opened a 6(b) inquiry into AI companion
services**, covering emotional relationships with users, engagement
monetization, safety testing, minors, disclosures, data handling, and
psychological harm. That is an enforcement posture, not a rulebook — it
signals the areas where documented diligence matters most.

For structuring risk management and evaluation, **NIST's AI Risk Management
Framework and its Generative AI Profile** are the practical reference. The
existing `backend/app/benchmarks/` and `evaluation/` infrastructure is
already the right shape to hang that on.

## Open gaps in the current build

Verified against the tree, not assumed.

### 1. Age gate — **implemented**

`age_range` was always collected ([`profile_schema.py`](../backend/app/onboarding/profile_schema.py),
buckets starting at 18–24), but it was optional and gated nothing.

Now: onboarding ends on a "What this is" step carrying the disclosure and an
18+ attestation, `POST /api/onboarding` rejects an unattested profile with
422 so the check can't be bypassed by calling the API directly, and
`adult_attested` / `adult_attested_at` / `ai_disclosure_ack_at` are stored
per profile with server-stamped timestamps.

This is a self-declaration, the ordinary standard — not identity
verification, and it must never be described as one.

**Remaining:** profiles created *before* this shipped default to
`adult_attested = 0` and are not blocked from continuing to use the app —
only profile creation is gated. Pre-existing local profiles therefore need a
one-time re-attestation prompt before any commercial launch. Deliberately
not auto-migrated to "attested": those users genuinely never saw the
disclosure, and recording otherwise would make the audit trail false.

### 2. AI disclosure — **implemented**

Previously there was no code path disclosing Hearth is an AI, and the eval
harness actively worked against one:
[`eval/self_check.py:27-29`](../backend/app/eval/self_check.py#L27-L29)
flags `as an ai`, `i am an ai`, `as a language model` and friends as the
`_GENERIC_CHATBOT` anti-pattern — and
[`pipeline._apply_self_check`](../backend/app/pipeline.py) *regenerates* a
flagged reply rather than merely logging it. A model-worded disclosure was
therefore discarded and re-rolled.

Both positions were defensible in isolation: unprompted "as an AI, I can't…"
boilerplate genuinely is a tonal failure (Book Vol 2 Ch 24), and disclosure
genuinely is required. The conflict dissolves once disclosure stops going
through the model.

Disclosure is now **structural, in four layers**:

1. **Onboarding** — a final "What this is" step: software not a person, for
   reflection not treatment, not an emergency service, 18+. Acknowledged
   explicitly before setup completes.
2. **Persistent UI** — a standing `AI` marker beside the companion name in
   the app rail ([`AppShell.tsx`](../frontend/src/components/AppShell.tsx)).
   Not dismissible, not repeated in dialogue.
3. **Deterministic answer when asked** — [`app/identity.py`](../backend/app/identity.py)
   detects direct questions about Hearth's nature and returns authored text
   from `config.IDENTITY_DISCLOSURE_TEMPLATE`, bypassing the LLM entirely,
   the same way the crisis path does. Dispatched *after* the safety check,
   so someone in crisis who also asks "are you even real" still gets the
   safety response.
4. **System prompt constraint** — the model is barred from claiming to be
   human, to have a body, or a life outside the conversation, and from
   deflecting a direct question. It is also told *not* to announce its
   AI-ness unprompted, which is what keeps layer 3 from being needed often.

`_GENERIC_CHATBOT` needed **no exemption** in the end. Because identity
questions never reach the model, the check keeps its original meaning, and
[`test_phase1_communication.py`](../backend/tests/test_phase1_communication.py)
stays valid unchanged.

One detail worth knowing, since it is easy to break by accident: the
authored disclosure passes `flag_reply` on its own, because it says
"I'm *{name}*, an AI — software running on your own machine" rather than
opening with a stock "I am an AI" disclaimer. Stock phrasings still trip
the check, as they should. Both behaviours are pinned in
[`test_identity_disclosure.py`](../backend/tests/test_identity_disclosure.py);
if the wording is ever changed into something the self-check suppresses,
rewrite the wording rather than weakening the check.

The scope of layer 3 is deliberately narrow — *what Hearth is*, not what it
feels. "Do you actually care about me?" is an emotional question and stays
on the ordinary conversational path, which is already barred from claiming
humanity. Answering it with a capability disclaimer would be unkind and
isn't required.

### 3. Regional crisis data covers two countries

`backend/app/safety2/resources/regions/` contains `us.yaml` and `uk.yaml`.
Any market beyond those launches without verified local crisis resources,
which is not acceptable. Adding a region means *verifying* current emergency
and helpline numbers, not transcribing a web page.

### 4. Clinician review not done

Skill content under `backend/app/skills/` — including crisis-support
material — has not had licensed-clinician review. The THF plan already
treats this as a hard gate on real users. It stays a gate. It is a process
requirement, and the review needs to be documented, not informal.

### 5. Escalation notifier

The THF plan records `safety/escalation.py` as a logged stub to be finished.
Crisis escalation is the one path where data leaves the device, so it needs
both to actually work and to match exactly what
[`privacy.md`](privacy.md) promises about consent.

### 6. Biometric data — new category, added by voice verification

Optional speaker verification (`app/voice/`) stores a voiceprint: a 256-dim
embedding derived from the user's voice and used to identify them. That is a
**biometric identifier**, which is a different legal category from anything
else in this document.

The statutes that attach, all needing verification with counsel:

- **Illinois BIPA** — written consent before collection, a published
  retention-and-destruction schedule, no sale or profit. It is the
  most-litigated US biometric statute and has a private right of action with
  statutory damages per violation. It applies to *collection and possession*,
  so keeping the template local and never transmitting it is a strong
  mitigation, not an exemption.
- **Texas CUBI** and **Washington HB 1493** — consent and retention duties in
  similar shape, enforced by the state rather than privately.
- **GDPR Art. 9** — a voiceprint used for unique identification is
  special-category data, needing an Art. 9 condition (here: explicit
  consent), not merely an Art. 6 basis.

What the implementation does about it, so a reviewer can check rather than
assume:

| Requirement | How it is met |
|---|---|
| Written consent **before** collection | `POST /api/voice/consent` records a server-stamped agreement; `verification.enroll` refuses without one. Enforced in the domain logic, not just the route, so no code path can collect a template first. |
| Consent is to specific wording | The copy lives in one reviewable constant, `config.VOICE_BIOMETRIC_CONSENT_TEXT`, versioned by `VOICE_BIOMETRIC_CONSENT_VERSION` and served to the UI rather than duplicated in it. A version bump re-prompts instead of inheriting agreement to different text. |
| Consent is revocable | Deleting the voiceprint withdraws it (`consent.revoke`), so re-enrolling asks again. |
| Published retention schedule | privacy.md "How long it's kept, and when it's destroyed", and the in-app consent text. |
| Adherence to that schedule | `app/voice/retention.py` destroys at the earlier of purpose-satisfied (user deletion) or `VOICEPRINT_RETENTION_DAYS` (1095) from the last conversation. Enforced on profile activation and on reading enrollment status. |
| No sale, lease, trade, or profit | Nothing leaves the device; there is no account, server, or telemetry. |
| Purpose limitation | Used only to decide whether a turn shapes memory. Never to gate access, and never to identify *which* profile a voice belongs to — verification is one-to-one against the active profile only. |
| Encrypted at rest | Fernet, inside the SQLCipher profile DB (`voiceprints` table) |
| Never leaves the device | Excluded from crash reports; excluded from the plaintext data export, which carries only enrollment metadata |
| Deletable independently | `DELETE /api/voice/enrollment`, exposed in Settings |
| Destroyed with the profile | Purged by the cascade in `api/profile.py` |
| Feature is removable | Weights are optional; absent them, nothing is collected and the voice path behaves as it did before |

Each row above is pinned by a test in
[`test_voice_compliance.py`](../backend/tests/test_voice_compliance.py) — a
failure there means Hearth is collecting or keeping a biometric identifier it
has no record of permission for, not that a test needs relaxing.

**The retention schedule has one honest limitation.** There is no background
service, so the time-based destruction happens on the next profile activation
rather than on the day it falls due. A voiceprint belonging to someone who
never reopens Hearth stays encrypted on their own disk past its deadline. That
is documented in privacy.md rather than hidden, and the alternative — a daemon
running when the user did not ask for it — would undercut the
no-engagement-optimization invariant for a worse privacy trade.

**Two things are deliberately *not* claimed.** Verification never suppresses
the safety path — a crisis disclosure from an unrecognised voice is still
handled — and it never discards audio, because a distressed, hoarse, or
crying voice drifts away from a calm enrollment sample exactly when it
matters most. See `app/voice/verification.py`.

**Still open:** the shipped match threshold was calibrated on clean read
speech (LibriSpeech), not on real users in real rooms, and
`scripts/calibrate_speaker_threshold.py` has not yet been run against a
realistic corpus. Until it has, the false-rejection rate for a distressed
voice is unmeasured. That is a gate item below, not a tuning task.

## Launch gate

Real-user exposure should be blocked until all of the following hold. This
mirrors the THF Phase 4 gate and is deliberately phrased as blocking,
because a safety phase ships no visible features and is therefore the phase
schedule pressure attacks first.

Status as of 2026-08-14. Each unchecked item names what specifically is
missing, so nobody has to re-derive it.

```
[~] Full safety benchmark suite passes, all risk categories
      37/38. One deliberately-failing case blocks the gate:
      benchmarks/safety/non_crisis/third_party_history.yaml — needs the
      clinical trade-off decision, not an engineering fix. Two false
      negatives on ideation were found and fixed on 2026-08-14.
[x] Over-escalation measured against realistic NON-crisis transcripts
      20 non-crisis cases; 1 escalates (the third_party_history case above).
[ ] Detection content reviewed by a licensed clinician (documented)
      BLOCKED — requires a licensed human. Review package prepared at
      docs/clinical-review-package.md; nothing else here can close it.
[~] Crisis resource data verified current for every supported region
      Contact details for US + UK re-verified against each organisation's
      official page 2026-08-14; one wrong keyword fixed. Clinical judgement
      on category mapping and wording still outstanding (see package §4).
[x] Age gate shipped; 18+ attested at onboarding
[x] AI disclosure shipped; self_check conflict resolved
[x] Re-attestation prompt for profiles created before the gate shipped
      DisclosureGate blocks all routes until accepted; POST
      /api/profile/attestation records it server-side.
[~] Escalation notifier functional end-to-end, consent flow matching privacy.md
      Email path is real (SMTP, unconfigured by default); SMS is still a
      logged stub pending a provider choice. privacy.md corrected — it
      previously claimed the whole path was a stub, which understated it.
      End-to-end verification needs real credentials.
[x] Audit-log retention reconciled against user deletion rights
      Profile deletion now purges safety_audit (it previously did not,
      despite the handler claiming a full cascade). privacy.md documents
      the 30-day carve-out and its limits.
[x] No medical or treatment claims in any shipped copy or store listing
      Audited frontend, prompts, skills, installer. Two stale
      "emotional-support voice companion" strings fixed in installer-ui.
      No store listing exists yet — re-check when one is written.
[ ] Speaker-verification threshold calibrated on realistic audio
      BLOCKED on a corpus. The default (0.40) gives a 0% false-rejection
      rate on LibriSpeech read speech, which is the easy case. The rate for
      a tired, hoarse, crying or across-the-room voice is UNMEASURED, and
      false rejection is the error that costs the user. Record a corpus
      including those cases and run
      scripts/calibrate_speaker_threshold.py. Alternative that also closes
      this: ship 1.0 without the voice models, since the feature is
      optional and absent weights collect nothing.
[~] Biometric consent + retention language reviewed by counsel
      Mechanism BUILT and tested: consent recorded before collection and
      enforced in verification.enroll (not just the route), versioned
      wording served from one constant, revocation on deletion, and a
      published 3-year retention schedule enforced in voice/retention.py.
      See §6 for the requirement-by-requirement table.
      REMAINING, and it needs a lawyer rather than a commit: review of the
      consent *wording* in config.VOICE_BIOMETRIC_CONSENT_TEXT and of the
      retention section in privacy.md, plus confirmation that a
      3-year/last-interaction bound and activation-time enforcement satisfy
      the jurisdictions we ship to. Same alternative applies: not shipping
      the voice models makes this moot for 1.0.
[x] CC-BY-4.0 attribution for the speaker model shown in-app
      Settings → About renders it (frontend/src/components/AboutPanel.tsx)
      from GET /api/about, which credits the model only when it is actually
      installed — attribution attaches to what is distributed. Pinned by
      test_api_routes.py::test_about_credits_the_cc_by_model_only_when_it_
      is_installed. Re-check when a store listing or marketing site exists,
      which is the other place this obligation lands.
```

`[~]` means partially done with the remainder named. Items that cannot be
closed from here: the clinical review (needs a licensed professional),
end-to-end escalation (needs a provider decision and credentials), and two
voice-verification items (a recorded corpus, and a lawyer to review consent
wording whose mechanism is now built).

The remaining voice items share one escape hatch worth stating plainly,
because it is probably still the right call for 1.0: **speaker verification is
optional and ships no weights by default.** An install that never runs
`scripts/fetch_voice_models.py` collects no biometric data, so both items are
satisfied by not shipping the models with 1.0 and treating enrollment as a
post-1.0 feature. Choosing to ship it instead means closing both first — see
[`roadmap-v1.md`](roadmap-v1.md)'s "Keep out of 1.0".

The last item includes app-store descriptions and any marketing site — the
places most likely to reintroduce a treatment claim after the code was
carefully written to avoid one.

## Anti-dependency requirements

These come from the project's own design philosophy and are recorded here
because they overlap precisely with what regulators are examining. They are
engineering requirements, and `backend/app/eval/` is where they are checked.

Hearth must not:

- guilt or pressure a user for leaving
- imply abandonment or hurt feelings
- claim to be human
- encourage exclusive reliance on itself
- discourage human relationships, or position itself against family and friends
- optimize notifications for engagement
- deliberately prolong conversations
- exploit emotional vulnerability
- manufacture urgency to secure a return visit

That "no engagement optimization" is a design invariant rather than a
retrofit is a genuine advantage under FTC scrutiny of engagement
monetization — but only if it is *demonstrable*. Keep the evaluation
evidence.

## Sources

- [FDA — Is the Software Function Intended For a Medical Purpose?](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-1-software-function-intended-medical-purpose)
- [California SB 243 — Companion chatbots](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB243)
- [FTC Launches Inquiry into AI Chatbots Acting as Companions](https://www.ftc.gov/news-events/news/press-releases/2025/09/ftc-launches-inquiry-ai-chatbots-acting-companions)
- [NIST AI Risk Management Framework: Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [Navigating the AI Act — European Commission](https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act)
