# Full training run (Windows 10 + RTX 5060)

Trains all five heads on the ~90M shared encoder from the HF datasets plus the
500k synthetic corpus, then exports ONNX for the backend.

## 0. PyTorch for Blackwell

RTX 50-series is `sm_120`. Two independent constraints decide the wheel:

- **Floor:** Blackwell kernels first shipped in CUDA **12.8**. An older wheel
  (e.g. cu121) either falls back to CPU or fails with *"no kernel image is
  available for execution on the device"*.
- **Ceiling:** the wheel's CUDA major must not exceed what your driver supports.
  Check the `CUDA Version` in `nvidia-smi`'s header.

Pick the newest variant at or below your driver's CUDA version:

| Driver reports | Use | Newest torch there |
|---|---|---|
| 13.2+ | `cu132` | 2.13.0 |
| 13.0–13.1 | `cu130` | 2.13.0 |
| 12.8–12.9 | `cu128` | 2.11.0 |
| below 12.8 | — | update the driver first |

Everything goes into the repo's `.venv`. From the **repo root** — this uses
`cu132`, correct for driver 610.88 / CUDA 13.3; substitute your row above:

```powershell
python -m venv .venv                      # first time only
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -r backend\requirements-common.txt
pip install -r hearth_ai\data\prepare\requirements.txt
pip install -r hearth_ai\requirements-export.txt
```

Don't guess the variant — `-CheckOnly` reads `nvidia-smi` and prints the exact
`pip install` line for your driver, including when a newer variant than the one
you have installed is available.

**If you already have a CPU-only torch** (`-CheckOnly` shows `torch 2.13.0+cpu`
and `CPU-ONLY BUILD`), a plain `pip install torch --index-url ...` will *not* fix
it. pip sees `torch` as already installed and skips the download regardless of the
`+cpu` / `+cu132` tag, so uninstall first:

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

This is the usual outcome of having run a bare `pip install torch` at some point,
which pulls `+cpu` on Windows.

The runner refuses to start without a working GPU, since five heads on CPU takes
days. Pass `-AllowCpu` if you really want that (e.g. a smoke run on a laptop).

You do not strictly need to activate `.venv` to train — the runner looks for
`<repo>\.venv` and uses it regardless, so an un-activated shell can't silently
fall back to a system Python that has no torch. Precedence is `-Python` /
`HEARTH_PYTHON`, then an activated venv, then `<repo>\.venv`, then `python` on
PATH (which warns).

Verify before starting a multi-hour run:

```powershell
.\scripts\train_full_nlp.ps1 -CheckOnly
```

Expect a `venv` line pointing at your `.venv`, a `driver` line, `dep ... ok` for all
five packages, `gpu ... sm_120`, `bf16 yes`, and `kernel launch OK`:

```
driver   610.88  supports CUDA up to 13.3
torch    2.13.0+cu132  cuda build 13.2
gpu      NVIDIA GeForce RTX 5060  sm_120  8.0 GiB
bf16     yes
kernel   launch OK
```

The preflight ends with a real kernel launch because that is the only thing that
proves the wheel actually matches the card — `torch.cuda.is_available()` can return
`True` on a build that still has no Blackwell kernels. Missing `torch` or `tokenizers`
aborts the run; missing `datasets` or `onnx`/`onnxruntime` only warns, since those gate
stages you may be skipping.

## 1. One command

From the repo root:

```powershell
.\scripts\train_full_nlp.ps1
```

That runs prepare → tokenizer → 5 heads → ONNX export → golden eval refresh.

The `.ps1` is only a wrapper; the pipeline itself is `scripts/train_full_nlp.py`, so
you can run it directly on any OS (`python scripts/train_full_nlp.py`). Add `-DryRun`
to print the exact commands for every stage without executing them.

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
| `no kernel image is available` | Wheel below CUDA 12.8 on `sm_120` — reinstall per the table in §0 |
| `CUDA driver version is insufficient` | Wheel CUDA newer than the driver — drop a variant, or update the driver |
| `torch ...+cpu` / `CPU-ONLY BUILD` | CPU wheel installed — `pip uninstall -y torch` first, then reinstall (see §0) |
| `nvidia-smi not found` | Not on PATH; it's normally `C:\Windows\System32\nvidia-smi.exe`. The preflight checks there too, so if it still isn't found, reinstall the driver |
| `cuda NOT AVAILABLE` | CPU-only torch build |
| `venv NONE` / `dep torch MISSING` | Running the system Python instead of `.venv` — activate it, or set `HEARTH_PYTHON` to `<repo>\.venv\Scripts\python.exe` |
| `.ps1` fails to parse: `string is missing the terminator` | The file lost its CRLF line endings. Windows PowerShell 5.1 cannot parse LF-only `.ps1`. `.gitattributes` pins `*.ps1` to `eol=crlf`; re-checkout the file |
| CUDA OOM | Lower `--batch-size`, raise `--grad-accum` |
| DataLoader hangs on Windows | Set `--num-workers 0` |
| Emotion metrics look impossibly good | Judged on `test_synthetic.jsonl` |
| `mixing SYNTHETIC ONLY` warning | `data/base/<task>` missing — rerun without `--skip-hf` |
