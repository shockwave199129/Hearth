# Third-party model attributions

Models Hearth downloads at runtime or build time rather than vendoring, with
the licence each one carries. Fetch scripts pin a SHA-256 for every asset;
see `scripts/fetch_*.py`.

This is a record of obligations, not a full dependency list — Python and npm
packages are covered by their own manifests.

## Voice input (`models/voice/`, `scripts/fetch_voice_models.py`)

| Model | Purpose | Licence | Attribution required |
|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) v5.1.2 | Voice activity detection — is a captured buffer speech at all | MIT | No (notice retained) |
| [WeSpeaker `voxceleb_resnet34_LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | Speaker embeddings for voice verification | CC-BY-4.0 | **Yes** |

**CC-BY-4.0 obligation — discharged in-app.** The speaker-embedding model is
licensed CC-BY-4.0, which requires attribution wherever it is distributed.
Credit: *speaker embedding model by the
[WeSpeaker](https://github.com/wenet-e2e/wespeaker) authors, trained on
VoxCeleb, used under CC-BY-4.0.*

That line is shown in **Settings → About**, rendered by
[`AboutPanel.tsx`](../frontend/src/components/AboutPanel.tsx) from
`GET /api/about`. The backend owns the list
([`api/about.py`](../backend/app/api/about.py)) so it sits beside the fetch
scripts that pin these artifacts, and so it can credit the model **only when
it is actually installed** — attribution attaches to what is distributed, and
a build shipping no voice models is not distributing it. Pinned by
`test_api_routes.py::test_about_credits_the_cc_by_model_only_when_it_is_installed`.

Still outstanding for this obligation: **a store listing or marketing site is
the other place attribution lands.** No such surface exists yet; check it
before tagging `1.0` ([`roadmap-v1.md`](roadmap-v1.md)).

When adding a component here, add it to `api/about.py` too — the doc is the
reviewable record, that list is what a user can actually see.

Both models are optional. Absent weights leave the voice path behaving
exactly as it did before verification existed, so a licence problem with
either is a feature-removal decision rather than a blocker.

## Speech, language, and TTS

| Model | Purpose | Licence |
|---|---|---|
| [Moonshine](https://github.com/moonshine-ai/moonshine) (`moonshine-voice`) | Speech-to-text | MIT |
| [Kokoro ONNX](https://huggingface.co/NeuML/kokoro-fp16-onnx) | Text-to-speech (default tiers) | Apache-2.0 |
| [Parler-TTS tiny v1](https://huggingface.co/parler-tts/parler-tts-tiny-v1) | Text-to-speech (higher tiers) | Apache-2.0 |
| [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) | Memory embeddings | Gemma Terms of Use |

`hearth_ai`'s own exported NLP heads are first-party and carry the
repository's licence.

## Binaries

| Artifact | Purpose | Licence |
|---|---|---|
| [`llama.cpp`](https://github.com/ggml-org/llama.cpp) `llama-server` | Local LLM + embedding server | MIT |
| [python-build-standalone](https://github.com/astral-sh/python-build-standalone) | Bundled setup interpreter | PSF / see upstream |

**Before 1.0:** confirm every licence above still matches what upstream
publishes, and confirm the Gemma terms permit the distribution shape Hearth
actually ships. Licences change between releases, and the pinned revisions
here are point-in-time (recorded 2026-08-17).
