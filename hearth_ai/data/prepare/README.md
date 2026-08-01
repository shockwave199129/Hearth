# Dataset preparation

Two corpora feed the same five heads:

| Corpus | Rows | Role |
|---|---|---|
| HF datasets | ~50–100k | Real human text — anchors Emotion + Intent |
| `data/hearth_relationship_understanding.jsonl` | 500k | Generator output — anchors Memory / Relationship / Strategy |

HF sources (all five kept, unchanged):
`google-research-datasets/go_emotions` (simplified), `dair-ai/emotion`,
`Adapting/empathetic_dialogues_v2`, `li2017dailydialog/daily_dialog`,
`tanaos/synthetic-intent-classifier-dataset-v1`.

## One command

```bash
cd hearth_ai
pip install -r data/prepare/requirements.txt
python3 -m data.prepare.prepare_all_full
```

That runs four stages and writes the dirs the train scripts read:

```
data/base/{emotion,intent}/          # stage 1 — HF only
data/synthetic/{5 tasks}/            # stage 2 — 500k converted
data/seed/{memory,relationship,strategy}/   # stage 3 — template seeds
data/{5 tasks}/{train,val,test}.jsonl       # stage 4 — mixed, used for training
```

Useful flags:

```bash
# Faster iteration: cap the synthetic source and reuse an existing HF prepare
python3 -m data.prepare.prepare_all_full --limit 50000 --skip-hf

# Shift the real/synthetic balance (fractions are upper bounds on synthetic)
python3 -m data.prepare.prepare_all_full --emotion-share 0.2 --intent-share 0.4

# Only keep emotion mappings that are unambiguous
python3 -m data.prepare.prepare_all_full --strict-emotion-map
```

## Why the synthetic corpus is capped

500k synthetic rows against ~50k real rows would let the generator's templates
dominate every epoch, and the model would learn the generator rather than the
language. So Emotion defaults to ≤30% synthetic and Intent to ≤50%, while
Memory / Relationship / Strategy run synthetic-heavy because no real corpus
exists for them.

`--synthetic-share` is an **upper bound**, not a target: `--max-per-label` runs
first and can shrink the synthetic pool below the requested share. Trust the
per-split counts the mixer prints.

## Label translation

`hearth_synthetic_maps.py` holds every map, with confidence tiers:

- `EMOTION_MAP` — 46 synthetic states → 28 GoEmotions labels
- `EMOTION_LOW_CONFIDENCE` — defensible but unverified (`nostalgic`, `warm`, …);
  excluded by `--strict-emotion-map`
- `EMOTION_UNMAPPED` — no honest equivalent (`tired`, `trusting`, `guarded`, …).
  Dropped from **emotion only**; those rows still train the other four heads.
  Forcing them to `neutral` would poison a 28-way multi-label head faster than
  the missing rows hurt.
- `INTENT_MAP` — 16 synthetic intents → 10 companion needs. The generator
  produces almost no genuine questions, so `inquire` comes from DailyDialog and
  Empathetic v2 — another reason the HF sources stay.
- Memory / Relationship / Strategy maps are **weak labels**, derived from
  `topic`, `relationship_signal`, `closeness_delta`, and `emotional_state`.
  Validate a hand-reviewed sample before trusting their MAE / F1.

`memory_candidate` is kept as metadata only. `MemoryHead` predicts
store/type/importance — it cannot generate text, so that field would need a
separate extraction model.

## Splits are grouped, not random

All 500k messages are textually unique but built from ~114k generator templates.
A row-level split puts paraphrases of one template in both train and test and
inflates every metric. `common.grouped_split` assigns whole template signatures
(`common.template_signature`) to one split, so val/test stay honest.

HF splits keep their upstream `split_hint`. The mixer additionally writes
`test_real.jsonl` and `test_synthetic.jsonl` per task — report those separately,
and never judge Emotion/Intent on the synthetic slice alone.

## Output formats

```jsonc
// emotion   — 28-dim multi-hot over EMOTION_LABELS
{"text": "...", "labels": [0.0, 1.0, ...], "source": "go_emotions"}
// intent
{"text": "...", "intent": "comfort", "source": "empathetic_dialogues_v2"}
// memory    — negatives are always type "other" / importance 0.05
{"text": "...", "store": 1, "type": "goal", "importance": 0.8}
// relationship — [trust_delta, vulnerability, openness, comfort]
{"text": "...", "signals": [0.2, 0.7, 0.8, 0.4]}
// strategy
{"text": "...", "strategy": "validate"}
```

Schemas for human/LLM annotation: `labels/annotation_schemas/`.

## Individual stages

```bash
# HF only
python3 -m data.prepare.prepare_emotion --out-dir data/base/emotion
python3 -m data.prepare.prepare_intent  --out-dir data/base/intent

# 500k → five task formats (+ drop report)
python3 -m data.prepare.prepare_hearth_synthetic \
    --out-dir data/synthetic --report data/synthetic/conversion_report.json

# Mix one task by hand
python3 -m data.prepare.mix_datasets --task emotion \
    --real data/base/emotion --synthetic data/synthetic/emotion \
    --out data/emotion --synthetic-share 0.3
```

Next: [`../../TRAINING.md`](../../TRAINING.md).
