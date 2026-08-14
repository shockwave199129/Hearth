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
```

`[~]` means partially done with the remainder named. Two items cannot be
closed from here: the clinical review (needs a licensed professional) and
end-to-end escalation (needs a provider decision and credentials).

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
