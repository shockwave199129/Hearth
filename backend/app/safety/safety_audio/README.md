# Safety audio — REPLACE BEFORE RELYING ON THIS

`response.wav` is currently a **generated placeholder tone** (a soft sine
sweep), not a real spoken message. It exists only so the crisis-response
mechanism in `main.py` (skip LLM/TTS, play this file directly) is fully
wired and testable end to end.

Before this is used in any real conversation, replace it with an actual
calm, pre-synthesized or pre-recorded spoken message matching
`config.SAFETY_RESPONSE_TEXT` — e.g. by running that text through the
existing TTS engine (`app/tts/tts_engines.py`) once, offline, and
saving the output here. The response text itself also needs review by a
licensed mental health professional, same as the skills library
(`app/skills/library/*.md`).
