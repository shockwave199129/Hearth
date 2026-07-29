<#
.SYNOPSIS
  Full Hearth NLP training run on Windows: prepare -> tokenizer -> 5 heads -> ONNX.

.DESCRIPTION
  Combines the HF datasets (go_emotions, dair-ai/emotion, empathetic_dialogues_v2,
  daily_dialog, tanaos) with data/hearth_relationship_understanding.jsonl, then
  trains all five heads on the shared ~90M encoder and exports ONNX.

  RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8+ PyTorch build. Verify with
  -CheckOnly before starting a multi-hour run; a cu121 wheel will either fall back
  to CPU or fail with "no kernel image is available for execution on the device".

.EXAMPLE
  .\scripts\train_full_nlp.ps1 -CheckOnly
  .\scripts\train_full_nlp.ps1
  .\scripts\train_full_nlp.ps1 -BatchSize 24 -GradAccum 2 -Epochs 3
  .\scripts\train_full_nlp.ps1 -SkipPrepare -Only memory,strategy
#>

[CmdletBinding()]
param(
    [int]    $Epochs        = 3,
    [int]    $BatchSize     = 32,
    [int]    $GradAccum     = 1,
    [double] $Lr            = 3e-4,
    [int]    $MaxSeq        = 128,
    [int]    $VocabSize     = 32000,
    [int]    $NumWorkers    = 0,
    [ValidateSet('off', 'auto', 'bf16', 'fp16')]
    [string] $Amp           = 'auto',
    [int]    $MaxTrainRows  = 0,
    [int]    $SyntheticLimit = 0,
    [double] $EmotionShare  = 0.30,
    [double] $IntentShare   = 0.50,
    [string[]] $Only        = @('emotion', 'intent', 'memory', 'relationship', 'strategy'),
    [switch] $SkipPrepare,
    [switch] $SkipHf,
    [switch] $SkipTokenizer,
    [switch] $SkipExport,
    [switch] $StrictGates,
    [switch] $CheckOnly
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$HearthRoot = Join-Path $RepoRoot 'hearth_ai'

if (-not (Test-Path $HearthRoot)) {
    throw "hearth_ai not found at $HearthRoot"
}

$Python = if ($env:HEARTH_PYTHON) { $env:HEARTH_PYTHON } else { 'python' }

# tokenizers' Rust thread pool oversubscribes the CPU next to the DataLoader.
$env:TOKENIZERS_PARALLELISM = 'false'

Write-Host "=== Environment ===" -ForegroundColor Cyan
& $Python -c @"
import sys
print('python  ', sys.version.split()[0])
try:
    import torch
    print('torch   ', torch.__version__, '| cuda build', torch.version.cuda)
    print('cuda ok ', torch.cuda.is_available())
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'gpu      {name} sm_{cap[0]}{cap[1]} {total:.1f} GiB')
        print('bf16 ok ', torch.cuda.is_bf16_supported())
        if cap[0] >= 12 and int(str(torch.version.cuda or '0').split('.')[0]) < 12:
            print('WARNING: Blackwell GPU with a pre-CUDA-12 torch build')
    else:
        print('WARNING: no CUDA device visible — training will run on CPU')
except ImportError:
    print('torch not installed')
"@
if ($LASTEXITCODE -ne 0) { throw 'environment check failed' }

if ($CheckOnly) {
    Write-Host "`nCheck only — install torch for CUDA 12.8 if the GPU line is missing:" -ForegroundColor Yellow
    Write-Host '  pip install torch --index-url https://download.pytorch.org/whl/cu128'
    exit 0
}

Push-Location $HearthRoot
try {
    if (-not $SkipPrepare) {
        Write-Host "`n=== 1/3 Prepare corpus (HF + 500k synthetic) ===" -ForegroundColor Cyan
        $prepareArgs = @(
            '-m', 'data.prepare.prepare_all_full',
            '--emotion-share', $EmotionShare,
            '--intent-share', $IntentShare
        )
        if ($SyntheticLimit -gt 0) { $prepareArgs += @('--limit', $SyntheticLimit) }
        if ($SkipHf)               { $prepareArgs += '--skip-hf' }
        & $Python @prepareArgs
        if ($LASTEXITCODE -ne 0) { throw 'prepare failed' }
    }
    else {
        Write-Host "`nSkipping prepare (reusing data/*)" -ForegroundColor Yellow
    }

    Write-Host "`n=== 2/3 Train all heads ===" -ForegroundColor Cyan
    $trainArgs = @(
        'examples/train_all_full.py',
        '--epochs', $Epochs,
        '--batch-size', $BatchSize,
        '--grad-accum', $GradAccum,
        '--lr', $Lr,
        '--max-seq', $MaxSeq,
        '--vocab-size', $VocabSize,
        '--num-workers', $NumWorkers,
        '--amp', $Amp,
        '--only'
    ) + $Only
    if ($MaxTrainRows -gt 0) { $trainArgs += @('--max-train-rows', $MaxTrainRows) }
    if ($SkipTokenizer)      { $trainArgs += '--skip-tokenizer' }
    if ($SkipExport)         { $trainArgs += '--skip-export' }
    if ($StrictGates)        { $trainArgs += '--strict-gates' }

    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw 'training failed' }
}
finally {
    Pop-Location
}

if (-not $SkipExport) {
    Write-Host "`n=== 3/3 Refresh golden eval ===" -ForegroundColor Cyan
    Push-Location (Join-Path $RepoRoot 'backend')
    try {
        & $Python -m app.eval.nlp_golden --update
        & $Python -m app.eval.nlp_golden
    }
    finally {
        Pop-Location
    }
}

Write-Host "`nDone." -ForegroundColor Green
