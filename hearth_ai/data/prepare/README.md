# HF → JSONL prepare scripts

Build Emotion / Intent training files under ``hearth_ai/data/{emotion,intent}/``.

```bash
cd hearth_ai
pip install datasets   # if needed

# Smoke (small)
python3 -m data.prepare.prepare_emotion --max-per-source 200
python3 -m data.prepare.prepare_intent --max-per-source 200

# Full
python3 -m data.prepare.prepare_emotion
python3 -m data.prepare.prepare_intent
```

## Output formats

**Emotion** (`data/emotion/{train,val,test}.jsonl`)::

    {"text": "...", "labels": [0.0, 1.0, ...], "source": "go_emotions"}

``labels`` length = 28 (GoEmotions + neutral).

**Intent** (`data/intent/{train,val,test}.jsonl`)::

    {"text": "...", "intent": "comfort", "source": "empathetic_dialogues_v2"}

**Memory / Relationship / Strategy** (Phase B — synthetic seeds + annotation schemas)::

```bash
python3 -m data.prepare.prepare_memory
python3 -m data.prepare.prepare_relationship
python3 -m data.prepare.prepare_strategy
```

Schemas for LLM/human annotation: ``labels/annotation_schemas/``.

    {"text": "...", "store": 1, "type": "goal", "importance": 0.8}
    {"text": "...", "signals": [0.2, 0.7, 0.8, 0.4]}
    {"text": "...", "strategy": "validate"}

## Intent empty-behavior heuristic

For Empathetic Dialogues v2 rows with no ``behavior`` / question label: if emotion
is strongly negative → ``vent``; otherwise the row is dropped.
