"""Voice-input analysis: speech gating (VAD) and speaker verification.

Kept separate from `app/stt/` (which is transcription) and `app/audio_io.py`
(which is device I/O) because both models here are optional: absent weights
must degrade to today's behaviour, not break the voice path.
"""
