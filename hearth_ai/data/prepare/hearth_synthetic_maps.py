"""Label maps: hearth_relationship_understanding.jsonl → the five locked Hearth label sets.

The synthetic corpus uses its own vocabulary (46 emotional states, 16 intents,
6 relationship signals, 9 stances). Everything here translates those into
``hearth_ai.labels`` so the same heads train on synthetic and HF data.

Confidence tiers matter: ``EMOTION_MAP`` entries listed in
``EMOTION_LOW_CONFIDENCE`` are defensible but unverified, and
``EMOTION_UNMAPPED`` states have no honest GoEmotions equivalent at all
(``tired`` is a body state, ``trusting``/``guarded`` are relational stances).
Emotion rows for unmapped states are dropped rather than forced into
``neutral`` — mislabelled negatives poison a 28-way multi-label head faster
than missing rows do. Those same rows still train the other four heads.
"""

from __future__ import annotations

# --- Emotion: synthetic emotional_state → GoEmotions names (EMOTION_LABELS) ---

EMOTION_MAP: dict[str, list[str]] = {
    # anger family
    "angry": ["anger"],
    "frustrated": ["annoyance"],
    "exasperated": ["annoyance"],
    "irritated": ["annoyance"],
    "annoyed": ["annoyance"],
    # sadness family
    "sad": ["sadness"],
    "hurt": ["sadness"],
    "lonely": ["sadness"],
    "isolated": ["sadness"],
    "empty": ["sadness"],
    "disappointed": ["disappointment"],
    # anxiety family
    "anxious": ["nervousness"],
    "nervous": ["nervousness"],
    "stressed": ["nervousness"],
    "overwhelmed": ["nervousness"],
    "insecure": ["nervousness"],
    # positive family
    "happy": ["joy"],
    "cheerful": ["joy"],
    "excited": ["excitement"],
    "grateful": ["gratitude"],
    "proud": ["pride"],
    "hopeful": ["optimism"],
    "relieved": ["relief"],
    "relaxed": ["relief"],
    "curious": ["curiosity"],
    "affectionate": ["love"],
    "playful": ["amusement"],
    "teasing": ["amusement"],
    # neutral family
    "neutral": ["neutral"],
    "flat": ["neutral"],
    # uncertainty family
    "uncertain": ["confusion"],
    "doubtful": ["confusion"],
    "conflicted": ["confusion"],
    # lower-confidence (see EMOTION_LOW_CONFIDENCE)
    "reflective": ["realization"],
    "content": ["joy"],
    "warm": ["caring"],
    "light": ["amusement"],
    "nostalgic": ["sadness"],
    "wistful": ["sadness"],
    "vulnerable": ["nervousness"],
}

# Defensible but unverified — excluded by ``--strict-emotion-map``.
EMOTION_LOW_CONFIDENCE: frozenset[str] = frozenset(
    {"reflective", "content", "warm", "light", "nostalgic", "wistful", "vulnerable"}
)

# No honest GoEmotions equivalent — always dropped from emotion training only.
EMOTION_UNMAPPED: frozenset[str] = frozenset(
    {"tired", "testing", "open", "trusting", "guarded", "withdrawn"}
)


# --- Intent: synthetic intent → INTENT_LABELS (10 companion needs) ---

INTENT_MAP: dict[str, str] = {
    "venting": "vent",
    "deep_disclosure": "vent",
    "seeking_validation": "validate",
    "seeking_comfort": "comfort",
    "sharing_achievement": "celebrate",
    "seeking_advice": "advise",
    "future_planning": "plan",
    "sharing_update": "small_talk",
    "routine_checkin": "small_talk",
    "reminiscing": "small_talk",
    "playful_bonding": "small_talk",
    "expressing_gratitude": "small_talk",
    "testing_boundaries": "meta",
    "raising_conflict": "meta",
    "expressing_affection": "meta",
    "minimal_disclosure": "unknown",
}

# The synthetic generator produces almost no genuine information-seeking
# questions, so ``inquire`` has to come from DailyDialog / Empathetic v2.
INTENT_UNDERREPRESENTED: frozenset[str] = frozenset({"inquire"})


# --- Memory: topic → MEMORY_TYPES, relationship_signal → importance ---

MEMORY_TYPE_BY_TOPIC: dict[str, str] = {
    "future": "goal",
    "achievement": "episodic",
    "nostalgia": "episodic",
    "daily_life": "episodic",
    "family": "person",
    "health": "semantic",
    "work_stress": "emotional",
    "self_doubt": "emotional",
    "companion_relationship": "boundary",
}

MEMORY_IMPORTANCE_BY_SIGNAL: dict[str, float] = {
    "deepening_intimacy": 0.85,
    "testing_boundaries": 0.80,
    "seeking_validation": 0.65,
    "distancing": 0.65,
    "seeking_release": 0.60,
    "routine_checkin": 0.40,
}

# Signal alone gives only six importance values; topic separates a passing
# comment about the weather from a stated boundary or long-term goal.
MEMORY_IMPORTANCE_BY_TOPIC: dict[str, float] = {
    "future": 0.10,
    "family": 0.05,
    "health": 0.05,
    "companion_relationship": 0.05,
    "achievement": 0.0,
    "self_doubt": 0.0,
    "work_stress": -0.05,
    "nostalgia": -0.05,
    "daily_life": -0.15,
}

# MemoryLoss trains the type head on every row, so negatives must be a single
# consistent class rather than whatever the topic happened to be.
MEMORY_NEGATIVE_TYPE = "other"
MEMORY_NEGATIVE_IMPORTANCE = 0.05


# --- Relationship: signal → [trust_delta, vulnerability, openness, comfort] ---
# Order matches RELATIONSHIP_SIGNALS. Weak labels, not ground truth.

RELATIONSHIP_BY_SIGNAL: dict[str, list[float]] = {
    "deepening_intimacy": [0.25, 0.85, 0.85, 0.70],
    "seeking_validation": [0.10, 0.65, 0.70, 0.45],
    "seeking_release": [0.05, 0.70, 0.75, 0.40],
    "routine_checkin": [0.02, 0.15, 0.35, 0.75],
    "testing_boundaries": [-0.05, 0.45, 0.45, 0.30],
    "distancing": [-0.20, 0.20, 0.15, 0.25],
}

CLOSENESS_TRUST_ADJUSTMENT: dict[str, float] = {
    "increase": 0.10,
    "neutral": 0.0,
    "decrease": -0.10,
}

# Signal + closeness alone yields only ~18 distinct targets, so emotional state
# nudges comfort and vulnerability to spread the regression targets out.
RELATIONSHIP_COMFORT_NUDGE: dict[str, float] = {
    "overwhelmed": -0.15,
    "anxious": -0.10,
    "nervous": -0.10,
    "stressed": -0.10,
    "angry": -0.10,
    "hurt": -0.10,
    "guarded": -0.15,
    "withdrawn": -0.15,
    "relaxed": 0.10,
    "content": 0.10,
    "cheerful": 0.10,
    "warm": 0.10,
    "playful": 0.10,
    "trusting": 0.15,
}

RELATIONSHIP_VULNERABILITY_NUDGE: dict[str, float] = {
    "vulnerable": 0.15,
    "hurt": 0.10,
    "lonely": 0.10,
    "isolated": 0.10,
    "empty": 0.10,
    "insecure": 0.10,
    "overwhelmed": 0.10,
    "guarded": -0.20,
    "withdrawn": -0.20,
    "flat": -0.15,
    "neutral": -0.10,
    "playful": -0.10,
    "teasing": -0.10,
    "light": -0.10,
}


# --- Strategy: suggested_stance → STRATEGY_LABELS (12) ---

STRATEGY_BY_STANCE: dict[str, str] = {
    "gentle_curiosity": "ask_question",
    "validate_without_fixing": "validate",
    "soft_presence": "listen",
    "playful_mirroring": "reflect",
    "warm_reciprocity": "encourage",
    "give_space": "listen",
    "practical_support": "advise",
    "calm_honesty": "reflect",
    "hold_boundary_kindly": "boundary",
}

# Stance alone never yields celebrate/plan/comfort/ground, so intent and
# emotional state override it where they are unambiguous. ``defer_safety``
# stays out — nothing in this corpus is a safety escalation, and the existing
# prepare_strategy.py seed templates already cover it.
STRATEGY_INTENT_OVERRIDE: dict[str, str] = {
    "sharing_achievement": "celebrate",
    "future_planning": "plan",
    "seeking_comfort": "comfort",
}

STRATEGY_GROUND_STATES: frozenset[str] = frozenset(
    {"overwhelmed", "anxious", "stressed", "nervous"}
)
STRATEGY_GROUND_STANCES: frozenset[str] = frozenset({"soft_presence", "give_space"})
STRATEGY_GROUND_LABEL = "ground"


def emotion_names(state: str, *, strict: bool) -> list[str] | None:
    """GoEmotions names for a synthetic emotional_state, or None to drop."""
    if state in EMOTION_UNMAPPED:
        return None
    if strict and state in EMOTION_LOW_CONFIDENCE:
        return None
    return EMOTION_MAP.get(state)


def strategy_label(stance: str, intent: str, state: str) -> str | None:
    """Strategy for one synthetic row: grounding → intent override → stance.

    Grounding wins over the intent override because an overwhelmed user asking
    for comfort needs regulation before reassurance.
    """
    if state in STRATEGY_GROUND_STATES and stance in STRATEGY_GROUND_STANCES:
        return STRATEGY_GROUND_LABEL
    override = STRATEGY_INTENT_OVERRIDE.get(intent)
    if override:
        return override
    return STRATEGY_BY_STANCE.get(stance)


def memory_importance(signal: str, topic: str) -> float | None:
    """Weak importance target in [0, 1], or None when the signal is unmapped."""
    base = MEMORY_IMPORTANCE_BY_SIGNAL.get(signal)
    if base is None:
        return None
    base += MEMORY_IMPORTANCE_BY_TOPIC.get(topic, 0.0)
    return max(0.0, min(1.0, base))


def relationship_signals(signal: str, closeness: str, state: str) -> list[float] | None:
    """Four weak regression targets for one synthetic row."""
    base = RELATIONSHIP_BY_SIGNAL.get(signal)
    if base is None:
        return None
    trust, vulnerability, openness, comfort = base
    trust += CLOSENESS_TRUST_ADJUSTMENT.get(closeness, 0.0)
    vulnerability += RELATIONSHIP_VULNERABILITY_NUDGE.get(state, 0.0)
    comfort += RELATIONSHIP_COMFORT_NUDGE.get(state, 0.0)
    return [
        max(-1.0, min(1.0, trust)),
        max(0.0, min(1.0, vulnerability)),
        max(0.0, min(1.0, openness)),
        max(0.0, min(1.0, comfort)),
    ]
