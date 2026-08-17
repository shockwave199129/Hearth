"""About / credits — the in-app surface that discharges attribution duties.

Exists because the speaker-embedding model is CC-BY-4.0, which requires
attribution wherever it is distributed, and Hearth had nowhere to put one
(docs/compliance.md launch gate). Rather than a single hardcoded line, this
serves the whole third-party list so the obligation is discharged for every
component at once and a future addition has an obvious home.

**The backend owns this list, not the frontend.** It sits next to the fetch
scripts that pin these artifacts, and it can report which *optional*
components are actually present — which matters here, because attribution
attaches to what is distributed. A build shipping no voice models is not
distributing the CC-BY model, and the UI can say so accurately instead of
crediting something that is not there.

Keep [`docs/attributions.md`](../../../docs/attributions.md) and this list in
step. The doc is the reviewable record; this is what a user can actually see.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import APP_VERSION, SPEAKER_MODEL_PATH, VAD_MODEL_PATH

router = APIRouter()


def _components() -> list[dict]:
    """Third-party models and binaries, with the licence each one carries.

    `attribution` is set only where the licence actually requires a credit
    line; the UI renders those prominently and lists the rest plainly. Marking
    everything as requiring attribution would bury the one that does.
    """
    return [
        {
            "name": "WeSpeaker voxceleb_resnet34_LM",
            "purpose": "Recognising whether it is you speaking",
            "license": "CC-BY-4.0",
            "url": "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM",
            "attribution": (
                "Speaker embedding model by the WeSpeaker authors, trained on "
                "VoxCeleb, used under CC-BY-4.0."
            ),
            "optional": True,
            "installed": SPEAKER_MODEL_PATH.is_file(),
        },
        {
            "name": "Silero VAD",
            "purpose": "Telling speech apart from background noise",
            "license": "MIT",
            "url": "https://github.com/snakers4/silero-vad",
            "optional": True,
            "installed": VAD_MODEL_PATH.is_file(),
        },
        {
            "name": "Moonshine",
            "purpose": "Turning your speech into text",
            "license": "MIT",
            "url": "https://github.com/moonshine-ai/moonshine",
        },
        {
            "name": "Kokoro ONNX",
            "purpose": "Speaking replies aloud",
            "license": "Apache-2.0",
            "url": "https://huggingface.co/NeuML/kokoro-fp16-onnx",
        },
        {
            "name": "Parler-TTS tiny v1",
            "purpose": "Speaking replies aloud (higher hardware tiers)",
            "license": "Apache-2.0",
            "url": "https://huggingface.co/parler-tts/parler-tts-tiny-v1",
        },
        {
            "name": "EmbeddingGemma 300M",
            "purpose": "Finding relevant memories",
            "license": "Gemma Terms of Use",
            "url": "https://huggingface.co/google/embeddinggemma-300m",
        },
        {
            "name": "llama.cpp",
            "purpose": "Running the language model on your machine",
            "license": "MIT",
            "url": "https://github.com/ggml-org/llama.cpp",
        },
    ]


@router.get("/api/about")
def api_about() -> dict:
    """Version plus third-party credits.

    Unauthenticated-safe in content: it contains no profile data, nothing
    user-specific, and no paths. It still sits behind the same local token
    middleware as every other route.
    """
    components = _components()
    return {
        "version": APP_VERSION,
        "components": components,
        # Pre-filtered so the UI cannot accidentally omit a required credit by
        # rendering the list without checking `installed`.
        "required_attributions": [
            component["attribution"]
            for component in components
            if component.get("attribution") and component.get("installed", True)
        ],
    }
