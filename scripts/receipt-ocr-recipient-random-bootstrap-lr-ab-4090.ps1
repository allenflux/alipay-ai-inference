[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceBootstrapRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$SourceDecision,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Evidence constants, deliberately not caller parameters.  This is a fresh
# single-variable A/B from the original random-root best, never a continuation
# of the failed 1e-4 pilot.
$baselineLearningRate = 0.0001
$candidateLearningRate = 0.0003
$pilotEpochs = 8
$seed = 424242
$recipientContinuationFloor = 0.75
$epoch4To8GainFloor = 0.02
$recipientDeliveryFloor = 0.90

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$trainer = Join-Path $repoRoot "src\transfer_receipt_ai\ocr_unified.py"
$verifier = Join-Path $repoRoot "src\transfer_receipt_ai\recipient_random_bootstrap_lr_ab.py"
$sourceTest = Join-Path $repoRoot "tests\test_recipient_random_bootstrap.py"
$abTest = Join-Path $repoRoot "tests\test_recipient_random_bootstrap_lr_ab.py"
$sourceInputContract = Join-Path $SourceBootstrapRoot "bootstrap-input.contract.json"
$sourceRecoveryDecision = Join-Path $SourceBootstrapRoot "analysis-decision.recovered.json"
$rootCheckpoint = Join-Path $SourceBootstrapRoot "random-root-1e\best.pt"
$inputContract = Join-Path $OutputRoot "lr-ab-input.contract.json"
$pilotOutput = Join-Path $OutputRoot "strict-recipient-lr3e4-8e"
$decisionPath = Join-Path $OutputRoot "lr-ab-decision.json"

function Require-File([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Require-Directory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing ${Description}: $Path"
    }
}

function Invoke-Python([string[]]$CommandArguments, [string]$Description) {
    & $pythonExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Seal-ReadOnlyEvidence([string]$Path, [string]$Description) {
    Require-File $Path $Description
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $item.IsReadOnly = $true
    if (-not (Get-Item -LiteralPath $Path -Force -ErrorAction Stop).IsReadOnly) {
        throw "Unable to seal ${Description} read-only: $Path"
    }
}

if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -lt 1) {
    throw "This fixed LR A/B launcher requires Windows PowerShell 5.1. Observed: $($PSVersionTable.PSVersion)"
}
$SourceBootstrapRoot = [IO.Path]::GetFullPath($SourceBootstrapRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
Require-Directory $SourceBootstrapRoot "immutable source bootstrap root"
foreach ($required in @($pythonExe, $trainer, $verifier, $sourceTest, $abTest, $sourceInputContract, $rootCheckpoint)) {
    Require-File $required "LR A/B dependency"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse LR A/B output: $OutputRoot"
}

$gpuRows = @(& nvidia-smi --query-gpu=index,name --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0 -or [string]$gpuRows[0] -notmatch "^0\s*,.*4090") {
    throw "CUDA device 0 must be an RTX 4090. Observed: $($gpuRows -join '; ')"
}
Invoke-Python @("-m", "transfer_receipt_ai.recipient_random_bootstrap", "probe-cuda") "CUDA:0 probe"

Write-Host "recipient_random_bootstrap_lr_ab_4090 preflight"
Write-Host "  source=$SourceBootstrapRoot"
Write-Host "  restart=original random-root-1e best; failed 1e-4 pilot best/last forbidden"
Write-Host ("  only recipe change: AdamW learning-rate {0} -> {1}" -f $baselineLearningRate, $candidateLearningRate)
Write-Host ("  fixed candidate: epochs={0}; seed={1}; recipient-only; validation every epoch" -f $pilotEpochs, $seed)
Write-Host "  optimizer=train only; checkpoint selection=val only; test=physically excluded"
Write-Host ("  unchanged gates: best recipient>={0:P2}; epoch4-to-8 gain>={1:P2}; delivery target={2:P2}" -f `
    $recipientContinuationFloor, $epoch4To8GainFloor, $recipientDeliveryFloor)
Write-Host "  analysis-only; no ONNX; no parser/runtime/delivery authority"
Write-Host "  output=$OutputRoot"

Invoke-Python @("-m", "pytest", "-q", $sourceTest, $abTest) "LR A/B source-contract tests"
if ([string]::IsNullOrWhiteSpace($SourceDecision)) {
    $SourceDecision = $sourceRecoveryDecision
}
$SourceDecision = [IO.Path]::GetFullPath($SourceDecision)
Require-File $SourceDecision "031004 recovery decision"
if ($SourceDecision -ne [IO.Path]::GetFullPath($sourceRecoveryDecision)) {
    throw "SourceDecision must be the canonical 031004 recovery decision: $sourceRecoveryDecision"
}
if ($CheckOnly) {
    Invoke-Python @(
        "-m", "transfer_receipt_ai.recipient_random_bootstrap_lr_ab", "check-source",
        "--source-bootstrap-root", $SourceBootstrapRoot,
        "--source-decision", $SourceDecision,
        "--runner", $MyInvocation.MyCommand.Path
    ) "LR A/B canonical 031004 source validation"
    Write-Host "recipient_random_bootstrap_lr_ab_4090 preflight=passed"
    exit 0
}

$prepareArgs = @(
    "-m", "transfer_receipt_ai.recipient_random_bootstrap_lr_ab", "prepare",
    "--source-bootstrap-root", $SourceBootstrapRoot,
    "--output-root", $OutputRoot,
    "--runner", $MyInvocation.MyCommand.Path,
    "--verifier", $verifier,
    "--source-decision", $SourceDecision
)
Invoke-Python $prepareArgs "LR A/B immutable input binding"
Require-File $inputContract "LR A/B input contract"

$sourceContract = Get-Content -LiteralPath $sourceInputContract -Raw -Encoding UTF8 | ConvertFrom-Json
$blindRecords = [IO.Path]::GetFullPath([string]$sourceContract.blind_manifest)
$snapshotRoot = [IO.Path]::GetFullPath([string]$sourceContract.snapshot_dataset_root)
Require-File $blindRecords "bound blind train/val manifest"
Require-Directory $snapshotRoot "bound read-only crop snapshot"

$trainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $blindRecords,
    "--dataset-root", $snapshotRoot,
    "--output", $pilotOutput,
    "--device", "cuda:0",
    "--architecture", "v12",
    "--image-height", "80",
    "--image-width", "512",
    "--base-channels", "32",
    "--numeric-hidden-size", "96",
    "--payment-hidden-size", "128",
    "--recipient-hidden-size", "256",
    "--recipient-value-left-trim", "0.30",
    "--recipient-input-height", "128",
    "--recipient-input-width", "1536",
    "--recipient-branch-channels", "24",
    "--recipient-open-text-layers", "2",
    "--recipient-open-text-heads", "8",
    "--recipient-open-text-feedforward", "2048",
    "--recipient-open-text-dropout", "0.0",
    "--recipient-backbone", "legacy_depthwise_gru_v1",
    "--pooled-width", "8",
    "--batch-size", "12",
    "--learning-rate", "$candidateLearningRate",
    "--payment-loss-weight", "1.0",
    "--recipient-loss-weight", "4.0",
    "--recipient-sampling-weight", "1.0",
    "--recipient-rare-character-max-support", "0",
    "--recipient-long-text-min-length", "0",
    "--recipient-low-confidence-threshold", "0.98",
    "--recipient-low-confidence-loss-weight", "0.35",
    "--recipient-confidence-curriculum-epochs", "10",
    "--recipient-tail-rare-character-max-support", "0",
    "--recipient-tail-rare-character-loss-weight", "1.0",
    "--recipient-tail-long-text-min-length", "0",
    "--recipient-tail-long-text-loss-weight", "1.0",
    "--recipient-train-augmentation", "light_v1",
    "--recipient-train-splits", "train",
    "--recipient-only-fine-tune",
    "--init-checkpoint", $rootCheckpoint,
    "--init-checkpoint-mode", "strict",
    "--checkpoint-selection", "balanced",
    "--ctc-loss-weight", "0.75",
    "--structured-loss-weight", "1.0",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--epochs", "$pilotEpochs",
    "--seed", "$seed",
    "--num-workers", "4",
    "--prefetch-factor", "2",
    "--persistent-workers",
    "--train-progress-every", "250",
    "--validation-every", "1",
    "--cuda-tf32",
    "--cudnn-benchmark"
)

Write-Host ""
Write-Host "stage: fresh strict recipient-only LR 3e-4 A/B (8 epochs)"
Invoke-Python $trainArgs "strict recipient LR A/B training"

Seal-ReadOnlyEvidence (Join-Path $pilotOutput "training_summary.json") "LR A/B summary"
Seal-ReadOnlyEvidence (Join-Path $pilotOutput "best.pt") "LR A/B best checkpoint"
Seal-ReadOnlyEvidence (Join-Path $pilotOutput "last.pt") "LR A/B last checkpoint"

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_random_bootstrap_lr_ab", "finalize",
    "--input-contract", $inputContract,
    "--candidate-output", $pilotOutput,
    "--output", $decisionPath
) "LR A/B evidence finalization"

$decision = Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ""
Write-Host "recipient_random_bootstrap_lr_ab_4090 final"
Write-Host ("  source best={0:P2}; candidate best={1:P2}; delta={2:+0.00%;-0.00%;0.00%}" -f `
    [double]$decision.source_observed.best_exact, `
    [double]$decision.candidate_observed.best_exact, `
    [double]$decision.candidate_best_delta)
Write-Host ("  candidate epoch4={0:P2}; epoch8={1:P2}; gain={2:+0.00%;-0.00%;0.00%}" -f `
    [double]$decision.candidate_observed.epoch4_exact, `
    [double]$decision.candidate_observed.epoch8_exact, `
    [double]$decision.candidate_observed.epoch4_to_8_gain)
Write-Host "  DELIVERY=NOT AUTHORIZED; ONNX=NOT EXPORTED; test=UNOPENED"
Write-Host "  decision=$decisionPath"
if ($decision.continuation_16_epoch_authorized -ne $true) {
    Write-Host "  16-epoch continuation=BLOCKED; stop this topology/view instead of burning 80 epochs"
    exit 3
}
Write-Host "  16-epoch continuation=AUTHORIZED FOR SEPARATELY BOUND ANALYSIS ONLY"
exit 0
