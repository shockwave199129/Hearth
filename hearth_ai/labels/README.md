# Hearth label schemas (locked)

Canonical YAML for the five `hearth_ai` task heads. Importable constants live in
`hearth_ai.labels` and must stay in sync with these files.

| File | Head | Type | Count |
|---|---|---|---|
| [emotion.yaml](emotion.yaml) | EmotionHead | multi-label BCE | 28 GoEmotions (+ MindState map → 14) |
| [intent.yaml](intent.yaml) | IntentHead | single-label CE | 10 |
| [memory.yaml](memory.yaml) | MemoryHead | store + type + importance | 8 types |
| [relationship.yaml](relationship.yaml) | RelationshipHead | regression | 4 signals |
| [strategy.yaml](strategy.yaml) | StrategyHead | single-label CE | 12 |

Annotation JSON Schemas (Phase B): [annotation_schemas/](annotation_schemas/).

```python
from hearth_ai.labels import INTENT_LABELS, emotion_num_labels, multi_hot
```