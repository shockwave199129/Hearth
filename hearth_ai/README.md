# Hearth AI — Shared Encoder + Task Heads

One transformer encoder, five lightweight task heads (emotion, intent,
memory, relationship, strategy). Same architecture discussed in your
planning doc: RoPE + RMSNorm + SwiGLU, 16 layers, ~90M params, encoder-only
(BERT/RoBERTa style, not decoder-only).

```
hearth_ai/
├── config.py                  HearthConfig - all hyperparameters
├── models/
│   ├── embeddings.py          Token embedding
│   ├── norm.py                RMSNorm
│   ├── attention.py           RoPE + multi-head self-attention
│   ├── feed_forward.py        SwiGLU FFN
│   ├── transformer_block.py   Pre-norm attn + FFN block
│   ├── encoder.py             HearthEncoder - stacks 16 blocks, pools output
│   ├── model.py                HearthModel (single-task), HearthMultiTaskModel
│   └── heads/
│       ├── base.py            HearthHead - generic 2-layer MLP head
│       ├── emotion.py         EmotionHead      (multi-label, BCE)
│       ├── intent.py          IntentHead       (single-label, CE)
│       ├── memory.py          MemoryHead       (store/type/importance)
│       ├── relationship.py    RelationshipHead (regression, SmoothL1)
│       └── strategy.py        StrategyHead     (single-label, CE)
├── tokenizer/
│   └── hearth_tokenizer.py    BPE tokenizer training + wrapper
├── trainer/
│   ├── dataset.py             Generic JSONL dataset + collate fn
│   ├── losses.py              Ready-made loss fn per head
│   └── train.py                Generic Trainer (fit/save/load)
├── examples/
│   └── train_intent.py        Full working example, runs in seconds
└── data/
    └── intent_sample.jsonl    12-row toy dataset used by the example
```

## Quickstart

```bash
pip install torch tokenizers datasets --break-system-packages   # or in a venv

cd hearth_ai

# Prepare HF JSONL (smoke sized) then train Emotion → Intent (encoder warm-start)
python3 -m data.prepare.prepare_emotion --max-per-source 500
python3 -m data.prepare.prepare_intent --max-per-source 500
python3 examples/train_emotion_intent.py --smoke

# Phase B heads (memory / relationship / strategy)
python3 examples/train_remaining_heads.py --smoke

# Full ~90M run (after full prepare without --max-per-source)
# python3 examples/train_emotion_intent.py --full
```

This trains a tiny (~1M param) version end-to-end in seconds. Checkpoints land in
`checkpoints/{emotion,intent,memory,relationship,strategy}/`. For the real 90M
model, use `--full` (see `HearthConfig()` defaults).

Toy-only intent demo (12 sample rows, no HF prepare)::

```bash
python3 examples/train_intent.py --sample --smoke
```

## Training your own task

Everything is designed so the **only** things that change between the five
models are the dataset, the head, and the loss function.

1. **Get your labeled data into JSONL.** One example per line. Match the
   format your head expects — see docstrings in `models/heads/*.py` and
   `trainer/dataset.py`. This is the "Real Conversations → Large LLM →
   Structured Labels" pipeline from your notes: use GPT-5.5/Claude/Gemini
   as an annotator to produce these JSONL rows, ideally with a
   multi-model-consensus filter on top.

2. **Train (or reuse) the tokenizer** once on your corpus:
   ```python
   from hearth_ai.tokenizer.hearth_tokenizer import train_tokenizer
   train_tokenizer(["data/all_conversations.txt"], "hearth_ai/tokenizer/hearth_tokenizer.json",
                    vocab_size=32000)
   ```

3. **Build the real-sized model:**
   ```python
   from hearth_ai.config import HearthConfig
   from hearth_ai.models import HearthEncoder, HearthModel, EmotionHead
   from hearth_ai.tokenizer.hearth_tokenizer import HearthTokenizer

   cfg = HearthConfig()   # 16 layers, hidden=512, ~90M params, vocab=32000
   tok = HearthTokenizer("hearth_ai/tokenizer/hearth_tokenizer.json", cfg.max_seq_len)
   encoder = HearthEncoder(cfg)
   model = HearthModel(encoder, EmotionHead(cfg.hidden_size, num_emotions=28))
   ```

4. **Train:**
   ```python
   from hearth_ai.trainer.dataset import HearthDataset
   from hearth_ai.trainer.train import Trainer
   from hearth_ai.trainer.losses import emotion_loss
   import torch

   def label_fn(ex):
       return torch.tensor(ex["labels"], dtype=torch.float)  # multi-hot vector

   train_ds = HearthDataset("data/emotion_train.jsonl", tok, label_fn)
   val_ds = HearthDataset("data/emotion_val.jsonl", tok, label_fn)

   trainer = Trainer(model, train_ds, emotion_loss, val_dataset=val_ds,
                      pad_id=tok.pad_id, checkpoint_dir="checkpoints/emotion")
   trainer.fit(epochs=10, batch_size=32, lr=3e-4)
   ```

Repeat step 3-4 per task, reusing the **same** `HearthEncoder` instance if
you want multi-task weight sharing (recommended - see `HearthMultiTaskModel`
in `models/model.py`), or a fresh `HearthEncoder(cfg)` per task if you'd
rather fine-tune independently.

## Param budget (≤ 200M)

Hard gate: `HearthEncoder`, `HearthModel`, and `HearthMultiTaskModel` assert
`num_parameters() <= 200_000_000` at construction (`MAX_MODEL_PARAMS` in
`hearth_ai.config`).

| Build | Typical size |
|---|---|
| `HearthConfig()` encoder | **~90M** (target) |
| Encoder + all 5 heads | still well under 200M (heads are tiny MLPs) |

```python
from hearth_ai.config import HearthConfig, MAX_MODEL_PARAMS
from hearth_ai.models import build_encoder, build_multitask_model

cfg = HearthConfig()
print(cfg.estimate_param_count() / 1e6, "M (estimate)")
enc = build_encoder(cfg)          # asserts ≤ 200M
print(enc.num_parameters() / 1e6, "M (actual)")
model = build_multitask_model(cfg, encoder=enc)  # asserts again
```

## Sizing knobs (from your 8GB 5060 constraint)

`HearthConfig()` defaults to the ~90M spec. To shrink for faster local
iteration, override fields directly, e.g.:

```python
cfg = HearthConfig(hidden_size=384, num_layers=12, num_heads=6, head_dim=64,
                    ffn_hidden_size=1536)
```

Keep `hidden_size == num_heads * head_dim` (there's an assertion for this).
Use mixed precision (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
around the forward pass) to fit comfortably in 8GB. Oversized configs that
would exceed 200M fail fast at build time.

## Multi-task training

`HearthMultiTaskModel` runs one shared encoder with several heads attached.
In your training loop, alternate batches from each task's dataset, compute
that task's loss against that task's head output, and backprop — the
encoder gets updated by every task, each head only by its own:

```python
from hearth_ai.models import HearthMultiTaskModel, EmotionHead, IntentHead

mt_model = HearthMultiTaskModel(encoder, {
    "emotion": EmotionHead(cfg.hidden_size),
    "intent": IntentHead(cfg.hidden_size),
})
out = mt_model(input_ids, attention_mask, tasks=["emotion"])  # only runs emotion head
loss = emotion_loss(out["emotion"], emotion_labels)
loss.backward()
```

A hand-rolled loop that round-robins task batches (rather than a built-in
`MultiTaskTrainer`) is left to you here, since how you weight/interleave
tasks is very much a design choice worth tuning for Hearth specifically.
