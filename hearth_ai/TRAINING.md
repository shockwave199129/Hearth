# Full training run (Windows 10 + RTX 5060)

Trains all five heads on the ~90M shared encoder from the HF datasets plus the
500k synthetic corpus, then exports ONNX for the backend.

## 0. PyTorch for Blackwell

RTX 50-series is `sm_120`. A CUDA 12.1 wheel has no kernel for it and will either
fall back to CPU or fail with *"no kernel image is available for execution on the
device"*, so install a **CUDA 12.8+** build:

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r ..\backend\requirements-common.txt
pip install -r data\prepare\requirements.txt
pip install -r requirements-export.txt
```

Verify before starting a multi-hour run:

```powershell
.\scripts\train_full_nlp.ps1 -CheckOnly
```

Expect `cuda ok True`, `sm_120`, and `bf16 ok True`.

## 1. One command

From the repo root:

```powershell
.\scripts\train_full_nlp.ps1
```

That runs prepare → tokenizer → 5 heads → ONNX export → golden eval refresh.

If 8 GB VRAM is tight, keep the effective batch at 32 while halving memory:

```powershell
.\scripts\train_full_nlp.ps1 -BatchSize 16 -GradAccum 2
```

## 2. Or step by step

```powershell
cd hearth_ai

# Corpus: HF (5 datasets) + 500k synthetic, mixed and capped
python -m data.prepare.prepare_all_full

# One shared 32k vocab for all five heads — must run before any training
python examples\build_tokenizer.py --full

# Emotion -> Intent -> Memory -> Relationship -> Strategy -> ONNX
python examples\train_all_full.py --batch-size 32 --epochs 3
```

Individual head, if you need to redo just one:

```powershell
python examples\train_strategy.py --full --keep-tokenizer `
    --warm-start checkpoints\relationship\best.pt `
    --epochs 3 --batch-size 32 --amp auto
```

## 3. Why this order

Every head shares one encoder, so ordering is not cosmetic:

1. `build_tokenizer.py` trains **one** vocab over all five tasks. If each
   `train_*.py` built its own, the checkpoints would carry mismatched embeddings
   and warm-start plus ONNX export would silently degrade. Pass
   `--keep-tokenizer` to every head so the vocab is reused verbatim.
2. **Emotion first** — largest real corpus, so the from-scratch encoder learns
   general language before the narrower tasks.
3. Each later head warm-starts from the previous checkpoint
   (`train_all_full.py` wires this automatically), so the shared encoder keeps
   improving instead of five encoders diverging.

## 4. Knobs

| Flag | Default | Notes |
|---|---|---|
| `--amp` | `auto` | bf16 on Blackwell; `off` for a numerically identical baseline |
| `--batch-size` / `--grad-accum` | 32 / 1 | Effective batch is the product |
| `--num-workers` | 0 | Windows uses spawn; raise to 2–4 only after verifying it works |
| `--max-seq` | 128 | Corpus p95 is 28 words, so 128 is already generous |
| `--warmup-ratio` | 0.03 | Linear warmup then cosine decay |
| `--early-stop-patience` | 1 | Stops when val loss stops improving |
| `--max-train-rows` | 0 (all) | Cap per head for a quick end-to-end rehearsal |
| `--strict-gates` | off | Fail the run when a head misses its eval gate |

Rehearse the whole pipeline in a few minutes before committing to the real run:

```powershell
python examples\train_all_full.py --epochs 1 --max-train-rows 2000
```

## 5. Eval gates

From the plan: emotion micro-F1 ≥ 0.45, intent macro-F1 ≥ 0.50, strategy
macro-F1 ≥ 0.45, memory store-F1 ≥ 0.70, relationship MAE under threshold.

Judge Emotion and Intent on `data/<task>/test_real.jsonl`. Scores on
`test_synthetic.jsonl` measure how well the model learned the generator, not the
language, and will look better than they are.

## 6. Ship to the backend

`train_all_full.py` exports automatically; to redo it alone:

```powershell
python examples\export_onnx.py --full --out ..\models\nlp

cd ..\backend
python -m app.eval.nlp_golden --update   # re-lock snapshots for the new weights
python -m app.eval.nlp_golden
python -m pytest tests\ -q
```

The exporter reads the `HearthConfig` embedded in each checkpoint, so it stays
correct even if you changed `--max-seq` or `--vocab-size` at train time.
Installing into `{MODELS_DIR}/nlp` is handled by
`app.setup.nlp_models.ensure_nlp_models` (or set `NLP_MODELS_DIR`).

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no kernel image is available` | cu121 wheel on `sm_120` — reinstall cu128 |
| `cuda ok False` | CPU-only torch build |
| CUDA OOM | Lower `--batch-size`, raise `--grad-accum` |
| DataLoader hangs on Windows | Set `--num-workers 0` |
| Emotion metrics look impossibly good | Judged on `test_synthetic.jsonl` |
| `mixing SYNTHETIC ONLY` warning | `data/base/<task>` missing — rerun without `--skip-hf` |
