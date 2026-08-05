"""User-selectable speaking style presets — the thing Parler is actually
good at, since its voice is steered entirely by a natural-language
description rather than by a speaker embedding (see ParlerEngine).

Every preset here has to sound like Hearth: warm, calm, and unhurried, a
steady presence that doesn't perform familiarity (docs/project-plan.md §3's
SYSTEM_PROMPT_TEMPLATE, Book Vol 2 Ch 7). That's a real constraint on the
wording below, not decoration — Parler will happily read "very
expressively" as bubbly ad-copy delivery, which is exactly the register
Hearth must never speak in. So the descriptions vary pitch, pace and
softness, and stop short of animated. `bright` is the ceiling.

The style is a CommunicationPreference in the Book Vol 2 Ch 7 sense: the
user picks it at onboarding, changes it in Settings, and nothing Hearth
learns ever silently overrides it.

Kokoro (tiers B/C) has no description input, so each preset also carries a
voicepack preference order + a speed multiplier — the closest equivalent
knobs it exposes. The order matters because which voicepacks exist depends
on which kokoro repo config.TTS_KOKORO_REPO points at; KokoroEngine picks
the first one actually present in voices.json.
"""
from dataclasses import dataclass, field

DEFAULT_VOICE = "female"
DEFAULT_VOICE_STYLE = "grounded"

VOICES = ("female", "male")


@dataclass(frozen=True)
class VoiceStyle:
    style_id: str
    label: str
    # Shown in onboarding/Settings under the label — plain language about
    # how it sounds, not TTS jargon.
    blurb: str
    # Parler description per voice, keyed "female"/"male". Phrased the way
    # parler-tts's own model card recommends: speaker, pitch, delivery,
    # environment, audio quality, then pace. "very confined sounding
    # environment with very clear audio quality" is the combination that
    # reliably suppresses room reverb and background noise.
    parler: dict[str, str]
    # Kokoro voicepack preference order per voice, most-preferred first.
    kokoro: dict[str, tuple[str, ...]]
    kokoro_speed: float = 1.0
    aliases: tuple[str, ...] = field(default_factory=tuple)


VOICE_STYLES: dict[str, VoiceStyle] = {
    "grounded": VoiceStyle(
        style_id="grounded",
        label="Grounded & even",
        blurb="Steady and warm, at a walking pace. The default.",
        parler={
            "female": (
                "A female speaker with a warm, slightly low-pitched voice delivers "
                "her words in a measured, even tone, in a very confined sounding "
                "environment with very clear audio quality. She speaks at a "
                "moderate pace."
            ),
            "male": (
                "A male speaker with a warm, low-pitched voice delivers his words "
                "in a measured, even tone, in a very confined sounding environment "
                "with very clear audio quality. He speaks at a moderate pace."
            ),
        },
        kokoro={
            "female": ("af_sky", "af_bella", "af_sarah", "af"),
            "male": ("am_adam", "am_michael"),
        },
        kokoro_speed=1.0,
    ),
    "gentle": VoiceStyle(
        style_id="gentle",
        label="Gentle & slow",
        blurb="Softer and quieter, with more room to breathe between words.",
        parler={
            "female": (
                "A female speaker with a soft, low-pitched voice delivers her words "
                "gently and quietly, in a very confined sounding environment with "
                "very clear audio quality. She speaks slowly."
            ),
            "male": (
                "A male speaker with a soft, low-pitched voice delivers his words "
                "gently and quietly, in a very confined sounding environment with "
                "very clear audio quality. He speaks slowly."
            ),
        },
        kokoro={
            "female": ("af_nicole", "af_sarah", "af_sky", "af"),
            "male": ("am_michael", "am_adam"),
        },
        kokoro_speed=0.9,
    ),
    "bright": VoiceStyle(
        style_id="bright",
        label="Warm & lifted",
        blurb="A little more lift and energy, still calm — never chirpy.",
        parler={
            "female": (
                "A female speaker with a warm, slightly high-pitched voice delivers "
                "her words quite expressively, in a very confined sounding "
                "environment with very clear audio quality. She speaks slightly fast."
            ),
            "male": (
                "A male speaker with a warm, moderately pitched voice delivers his "
                "words quite expressively, in a very confined sounding environment "
                "with very clear audio quality. He speaks slightly fast."
            ),
        },
        kokoro={
            "female": ("af_bella", "af_sky", "af"),
            "male": ("am_michael", "am_adam"),
        },
        kokoro_speed=1.1,
    ),
}

VOICE_STYLE_IDS = tuple(VOICE_STYLES)


def normalize_voice(voice: str | None) -> str:
    """Unknown/missing → the default voice rather than an error. Voice
    reaches here straight off a stored profile, and a profile row written by
    an older build must never be the reason a reply goes unspoken."""
    return voice if voice in VOICES else DEFAULT_VOICE


def resolve_style(style_id: str | None) -> VoiceStyle:
    """Same fail-soft contract as normalize_voice — see above. API handlers
    validate strictly (main.py) so bad values are rejected at write time,
    not discovered at synthesis time."""
    if style_id in VOICE_STYLES:
        return VOICE_STYLES[style_id]
    for style in VOICE_STYLES.values():
        if style_id in style.aliases:
            return style
    return VOICE_STYLES[DEFAULT_VOICE_STYLE]


def parler_description(voice: str | None, style_id: str | None) -> str:
    return resolve_style(style_id).parler[normalize_voice(voice)]


def kokoro_voice_ids(voice: str | None, style_id: str | None) -> tuple[str, ...]:
    return resolve_style(style_id).kokoro[normalize_voice(voice)]
