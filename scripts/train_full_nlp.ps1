# Full Hearth NLP training run on Windows.
#
# Thin wrapper around scripts/train_full_nlp.py, which holds all the logic.
# Kept ASCII-only with CRLF line endings on purpose: Windows PowerShell 5.1
# mis-parses here-strings in LF-only files and reads non-ASCII as CP1252.
#
# Examples:
#   .\scripts\train_full_nlp.ps1 -CheckOnly
#   .\scripts\train_full_nlp.ps1
#   .\scripts\train_full_nlp.ps1 -BatchSize 16 -GradAccum 2
#   .\scripts\train_full_nlp.ps1 -SkipPrepare -Only memory,strategy

# Rates are [string], not [double]: PowerShell renders doubles with the current
# culture, so on a comma-decimal locale -Lr would reach Python as '0,0003'.
[CmdletBinding()]
param(
    [int]    $Epochs         = 3,
    [int]    $BatchSize      = 32,
    [int]    $GradAccum      = 1,
    [string] $Lr             = '3e-4',
    [int]    $MaxSeq         = 128,
    [int]    $VocabSize      = 32000,
    [int]    $NumWorkers     = 0,
    [ValidateSet('off', 'auto', 'bf16', 'fp16')]
    [string] $Amp            = 'auto',
    [int]    $MaxTrainRows   = 0,
    [int]    $SyntheticLimit = 0,
    [string] $EmotionShare   = '0.30',
    [string] $IntentShare    = '0.50',
    [string[]] $Only         = @('emotion', 'intent', 'memory', 'relationship', 'strategy'),
    [switch] $SkipPrepare,
    [switch] $SkipHf,
    [switch] $SkipTokenizer,
    [switch] $SkipExport,
    [switch] $SkipEval,
    [switch] $StrictGates,
    [switch] $CheckOnly,
    [switch] $AllowCpu,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner   = Join-Path $PSScriptRoot 'train_full_nlp.py'

if (-not (Test-Path $Runner)) {
    throw "runner not found: $Runner"
}

# Deps live in the repo's .venv. Resolve it explicitly rather than trusting
# PATH, so an un-activated shell doesn't silently run the system Python (which
# has no torch) an hour into the job.
function Resolve-VenvPython {
    param([string] $Root)
    foreach ($rel in @('Scripts\python.exe', 'bin/python')) {
        $candidate = Join-Path $Root $rel
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

if ($env:HEARTH_PYTHON) {
    $Python = $env:HEARTH_PYTHON
    $PythonSource = 'HEARTH_PYTHON'
}
elseif ($env:VIRTUAL_ENV -and (Resolve-VenvPython $env:VIRTUAL_ENV)) {
    $Python = Resolve-VenvPython $env:VIRTUAL_ENV
    $PythonSource = 'activated venv'
}
elseif (Resolve-VenvPython (Join-Path $RepoRoot '.venv')) {
    $Python = Resolve-VenvPython (Join-Path $RepoRoot '.venv')
    $PythonSource = 'repo .venv'
}
else {
    $Python = 'python'
    $PythonSource = 'PATH'
    Write-Warning "No .venv found under $RepoRoot - falling back to 'python' on PATH."
}

$CliArgs = @(
    $Runner
    '--epochs',          $Epochs
    '--batch-size',      $BatchSize
    '--grad-accum',      $GradAccum
    '--lr',              $Lr
    '--max-seq',         $MaxSeq
    '--vocab-size',      $VocabSize
    '--num-workers',     $NumWorkers
    '--amp',             $Amp
    '--emotion-share',   $EmotionShare
    '--intent-share',    $IntentShare
)

if ($MaxTrainRows -gt 0)   { $CliArgs += @('--max-train-rows', $MaxTrainRows) }
if ($SyntheticLimit -gt 0) { $CliArgs += @('--synthetic-limit', $SyntheticLimit) }
if ($SkipPrepare)          { $CliArgs += '--skip-prepare' }
if ($SkipHf)               { $CliArgs += '--skip-hf' }
if ($SkipTokenizer)        { $CliArgs += '--skip-tokenizer' }
if ($SkipExport)           { $CliArgs += '--skip-export' }
if ($SkipEval)             { $CliArgs += '--skip-eval' }
if ($StrictGates)          { $CliArgs += '--strict-gates' }
if ($CheckOnly)            { $CliArgs += '--check-only' }
if ($AllowCpu)             { $CliArgs += '--allow-cpu' }
if ($DryRun)               { $CliArgs += '--dry-run' }

$CliArgs += @('--only')
$CliArgs += $Only

Write-Host "Repo: $RepoRoot"
Write-Host "Python: $Python  [$PythonSource]"

& $Python @CliArgs

if ($LASTEXITCODE -ne 0) {
    throw "training pipeline failed with exit code $LASTEXITCODE"
}
