# Positioning and target market

This document records *what Hearth is sold as*, *who it is for first*, and
why the sequencing is what it is. It exists because those choices constrain
engineering — the difference between "reflection companion" and "AI therapy"
is not marketing copy, it decides which regulations apply (see
[`compliance.md`](compliance.md)).

Written August 2026. Market figures cited here are point-in-time; re-check
before quoting them anywhere external.

## The one-line positioning

> A privacy-first personal AI companion that remembers what matters to you —
> and runs entirely on your own machine.

The emphasis order matters: **private + persistent + local + voice**.
Emotional support is one capability, not the product definition.

Hearth was previously described as "a privacy-first emotional-support voice
companion." That framing was changed deliberately. Leading with
"emotional-support" invites reading the product as mental-health treatment,
which is both a regulatory problem (intended use drives medical-device
classification) and a poor description of what the software actually does
most of the time.

## What Hearth competes on

Not response quality — that tracks whatever local model is current, and any
competitor can license a better one. The defensible combination is:

1. **Nothing leaves the device.** Not a policy promise, an architecture. A
   cloud companion cannot match this without rebuilding.
2. **Memory that persists and is inspectable.** Most companions either
   forget between sessions or remember on someone else's server. Hearth
   remembers locally and lets you read, correct, and delete any of it.
3. **No engagement optimization.** The design forbids prolonging
   conversations, guilt on exit, and dependency-building. Competitors funded
   on engagement metrics cannot credibly copy this.
4. **Longitudinal relationship modelling.** Trust, communication
   preferences, and life context accumulated over months.

### An honest note on the moat

Hearth is Apache-2.0 and public. The *code* is not defensible — anyone may
fork it, close their fork, and ship. What is not published, and is
therefore where durable advantage actually sits:

- `The Book/` — the 10-volume design specification (local-only)
- Evaluation datasets and benchmark cases
- `hearth_ai` training data and annotation schemas

Strategy that depends on the architecture being hard to copy is
mis-specified. Strategy that depends on the specification, the eval corpus,
and accumulated product judgment is not.

## Who Hearth is for, first

Not "everyone who wants a companion." The initial user is someone for whom
the *specific* combination above is worth switching for:

```
Adults 20–45, US
Already uses AI daily; not impressed by another chatbot
Actively values privacy — the kind of person who reads what an app uploads
Journals, reflects, or self-tracks already
Has a machine capable of running a local model (tier S/A)
Wants continuity, not answers
```

The qualifying question is not "who needs a companion?" — many people
plausibly do. It is: **who is dissatisfied enough with cloud companions to
install a desktop app to escape them?**

## Market sequencing

### Tier 1 — United States (launch)

Largest AI-companion market: North America held roughly a third of global
revenue share in 2025, the US being the bulk of it. English-native, high
willingness to pay, and — importantly — the regulatory expectations we are
already designing toward, so compliance work is not wasted.

### Tier 2 — UK, Canada, Australia, Ireland, New Zealand (fast follow)

Chosen for near-zero marginal cost: no new STT/TTS/LLM localization, no new
onboarding translation. Each needs only verified crisis-resource data under
`backend/app/safety2/resources/regions/`. `uk.yaml` already exists;
`ca`/`au`/`ie`/`nz` do not. That file is the gate, and writing it means
verifying live emergency and helpline numbers, not copying a list.

Lighter AI-specific regulation than the EU at present, which is a
sequencing reason, not a permanent one.

### Tier 3 — EU (deliberate, later)

Privacy-first positioning resonates most strongly here, particularly
Germany, the Netherlands, and the Nordics — and local-only processing is a
genuine GDPR advantage, since data that never leaves the device sidesteps
most transfer and retention questions.

Against that: the EU AI Act's 2026 obligations reach wellness and
mental-health-adjacent AI (transparency and AI-labelling duties,
manipulation prohibitions, and a high-risk reading for mental-health
applications), plus per-country localization. Worth entering properly, not
early.

### Tier 4 — highest measured need (long-term, mission)

The countries that need this most are not the ones we can serve first, and
it's worth being explicit rather than quietly ignoring them.

| Signal | Figure |
|---|---|
| Global loneliness | ~24% report feeling very or fairly lonely |
| Highest national rates | Brazil ~50%; Turkey and India ~43–46% |
| Loneliest age band | 19–29, globally |
| Depression treatment gap | ~56% globally; only ~9% get minimally adequate treatment |
| Low/middle-income treatment gap | ~75% |
| High-income treatment gap | still 35–50% |

Note the last row: even the launch market has a large unmet need. Hearth
does not need to serve the hardest markets to be worth building.

The blockers for Tier 4 are concrete, not vague:

- **Mobile-first populations.** Hearth is a desktop app. This alone is
  disqualifying for Brazil and India.
- **Hardware.** Local inference wants tier S/A. Tier B/C works but degrades
  exactly where model quality matters most.
- **Language.** Moonshine, Parler/Kokoro, and LFM2.5 are English-centric
  here. Portuguese, Hindi, and Turkish mean new models and new evals.
- **Willingness to pay.** A subscription priced for the US does not
  transfer.
- **Resources and norms.** Crisis infrastructure differs; safety content
  validated for the US is not automatically safe elsewhere.

Sequencing these behind Tiers 1–3 is a capability judgment. Presenting them
as reachable sooner would be dishonest about the desktop and language
constraints.

## Product scope for a first commercial release

Worth stating plainly, because the codebase already contains more than a
first release needs: `cognitive/`, `workers/`, `memory2/`, `relationship/`,
`intervention/`, `growth/`, `learning/`, `evaluation/`, and `safety2/` are
all implemented. The constraint on shipping is **not** unbuilt features.

It is:

1. The safety gate — see [`compliance.md`](compliance.md)
2. Licensed-clinician review of skill content
3. Age gate and AI disclosure (not yet built)
4. Verified regional resource data
5. Installer and onboarding polish

Adding capability before those are closed increases what has to be reviewed
without moving the release date.

## The growth loop

The loop Hearth is built to validate:

```
talks to Hearth → Hearth remembers something useful → returns later
→ context is still there → feels understood → returns voluntarily
```

Explicitly *not*:

```
notification → habit → engagement → retention
```

This is a product constraint with teeth: any feature justified primarily by
"it increases session length or return rate" is failing the design, not
succeeding at it. `backend/app/eval/` is where that gets measured rather
than asserted.

## Sources

- [Almost a Quarter of the World Feels Lonely — Gallup](https://news.gallup.com/opinion/gallup/512618/almost-quarter-world-feels-lonely.aspx)
- [Loneliness among adults worldwide by country — Statista](https://www.statista.com/statistics/1222815/loneliness-among-adults-by-country/)
- [Loneliness around the world: patterns, predictors, and well-being implications — European Journal of Epidemiology](https://link.springer.com/article/10.1007/s10654-026-01424-z)
- [Mental health care is scarce everywhere — Our World in Data](https://ourworldindata.org/data-insights/mental-health-care-is-scarce-everywhere-but-in-poor-countries-it-barely-exists)
- [Treatment gap for anxiety disorders is global — World Mental Health Surveys, 21 countries](https://repositori.upf.edu/items/d211d81a-b1ca-42a4-ad2a-61cf4e4ec61f)
- [Mental health service coverage and gaps among adults in Europe — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666776225002509)
- [AI Companion Market Size & Research Report — Grand View Research](https://www.grandviewresearch.com/industry-analysis/ai-companion-market-report)
