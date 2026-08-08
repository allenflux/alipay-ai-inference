[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FullRecords,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$SeedCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(1, 32)]
    [int]$BatchSize = 10,
    [ValidateRange(0.000001, 0.001)]
    [double]$LearningRate = 0.0001,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Deliberately not parameters: this is an eight-epoch analysis pilot, not a
# caller-tunable training or release command.
$pilotEpochs = 8
$recipientStopFloor = 0.75
$epoch4To8GainFloor = 0.02
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$statusTextFloor = 0.90
$requiredInit = "recipient_full_crop_warmstart"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"

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

function Require-FreshNonReparseOutput([string]$Path) {
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing -or (Test-Path -LiteralPath $Path)) {
        throw "Refusing to reuse existing, symlink, or reparse full-crop pilot output: $Path"
    }
    $ancestor = Split-Path -Parent $Path
    while (-not [string]::IsNullOrWhiteSpace($ancestor)) {
        $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Full-crop pilot output must not traverse a symlink/junction/reparse ancestor: $ancestor"
        }
        $next = Split-Path -Parent $ancestor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $ancestor) {
            break
        }
        $ancestor = $next
    }
}

function Invoke-Python([string[]]$CommandArguments, [string]$Description) {
    & $pythonExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Read-Json([string]$Path) {
    Require-File $Path "JSON evidence"
    $normalized = (& $pythonExe $normalizer $Path) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to normalize JSON evidence: $Path"
    }
    return ($normalized | ConvertFrom-Json)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $normalizer "JSON normalizer"
Require-File $FullRecords "full v13 unified manifest"
Require-Directory $DatasetRoot "recipient crop root"
Require-File $SeedCheckpoint "attested analysis-only sanitized 0.30-trim v13 seed checkpoint"
if ([IO.Path]::GetExtension($SeedCheckpoint) -ne ".pt") {
    throw "SeedCheckpoint must be a PyTorch .pt file."
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
Require-FreshNonReparseOutput $OutputRoot

$sourceTests = @(
    (Join-Path $repoRoot "tests\test_recipient_full_crop_seed_sanitizer.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_pilot.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v13.py")
)
foreach ($sourceTest in $sourceTests) {
    Require-File $sourceTest "full-crop source-contract test"
}

$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi did not report a CUDA GPU."
}
if ([string]$gpuRows[0] -notmatch "4090") {
    throw "CUDA device 0 must be an RTX 4090 for this fixed pilot. Observed: $($gpuRows[0])"
}

Write-Host "receipt_recipient_full_crop_pilot_4090 preflight"
Write-Host "  seed requires content-bound sanitizer attestation; top-level train-only claims are insufficient"
Write-Host "  warmstart reopens both sanitizer sources and the complete hash-bound train-only lineage"
Write-Host "  v13 ABI and checkpoint config preserved; only recipient left trim 0.30 -> 0.0"
Write-Host "  optimizer=train only; checkpoint selection=val only; test=physically excluded"
Write-Host ("  fixed pilot: epochs={0}, best recipient>={1:P2}, epoch4-to-8 gain>={2:P2}" -f `
    $pilotEpochs, $recipientStopFloor, $epoch4To8GainFloor)
Write-Host ("  protected floors: amount={0:P2}, time={1:P2}, payment={2:P2}" -f `
    $amountFloor, $timeFloor, $paymentFloor)
Write-Host ("  visible-status floor={0:P2}; non-success->success=0" -f $statusTextFloor)
Write-Host ("  GPU: {0}" -f ($gpuRows -join "; "))

Invoke-Python ((@("-m", "pytest", "-q")) + $sourceTests) "full-crop source-contract tests"
if ($CheckOnly) {
    Write-Host "receipt_recipient_full_crop_pilot_4090 preflight=passed"
    exit 0
}

$blindRoot = Join-Path $OutputRoot "blind-train-val"
$blindRecords = Join-Path $blindRoot "unified_fields.train-val.jsonl"
$blindContractPath = Join-Path $blindRoot "blind.contract.json"
$trainingRoot = Join-Path $OutputRoot "training-full-crop-pilot"
$decisionPath = Join-Path $trainingRoot "pilot_decision.json"

New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
$createdOutput = Get-Item -LiteralPath $OutputRoot -Force -ErrorAction Stop
if (($createdOutput.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Fresh full-crop pilot output unexpectedly became a reparse point: $OutputRoot"
}
Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_blind_manifest",
    "--source", $FullRecords,
    "--output", $blindRecords,
    "--contract", $blindContractPath
) "blind train/val manifest build"

$blind = Read-Json $blindContractPath
$boundSourcePath = [IO.Path]::GetFullPath([string]$blind.source_manifest)
$boundBlindPath = [IO.Path]::GetFullPath([string]$blind.blind_manifest)
$fullRecordsPath = [IO.Path]::GetFullPath($FullRecords)
$blindRecordsPath = [IO.Path]::GetFullPath($blindRecords)
$fullRecordsSha256 = Get-Sha256 $FullRecords
$blindRecordsSha256 = Get-Sha256 $blindRecords
if ([string]$blind.kind -ne "receipt_recipient_blind_train_val_manifest_v1" `
    -or $blind.test_labels_used -ne $false `
    -or $blind.test_metrics_computed -ne $false `
    -or $blind.test_examples_emitted -ne $false `
    -or $boundSourcePath -ne $fullRecordsPath `
    -or $boundBlindPath -ne $blindRecordsPath `
    -or [string]$blind.source_manifest_sha256 -ne $fullRecordsSha256 `
    -or [string]$blind.blind_manifest_sha256 -ne $blindRecordsSha256 `
    -or (($blind.optimizer_supervision_splits -join ",") -ne "train") `
    -or (($blind.checkpoint_selection_splits -join ",") -ne "val") `
    -or (($blind.final_gate_only_splits -join ",") -ne "test") `
    -or [int]$blind.split_counts.train -le 0 `
    -or [int]$blind.split_counts.val -le 0 `
    -or [int]$blind.split_counts.test_excluded -le 0) {
    throw "Blind manifest does not prove train/val/test isolation."
}

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_full_crop_pilot",
    "--records", $blindRecords,
    "--blind-contract", $blindContractPath,
    "--dataset-root", $DatasetRoot,
    "--seed-checkpoint", $SeedCheckpoint,
    "--output", $trainingRoot,
    "--device", "cuda:0",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--num-workers", "$NumWorkers",
    "--prefetch-factor", "$PrefetchFactor",
    "--train-progress-every", "$TrainProgressEvery"
) "fixed full-crop recipient pilot"

$decision = Read-Json $decisionPath
if ([string]$decision.kind -ne "receipt_recipient_full_crop_pilot_v1" `
    -or $decision.analysis_only -ne $true `
    -or $decision.production_route_authorized -ne $false `
    -or [int]$decision.epochs -ne $pilotEpochs `
    -or [double]$decision.target_config.recipient_value_left_trim -ne 0.0 `
    -or [double]$decision.source_config.recipient_value_left_trim -ne 0.30 `
    -or [double]$decision.fixed_gates.minimum_best_recipient_exact -ne $recipientStopFloor `
    -or [double]$decision.fixed_gates.minimum_epoch4_to_8_gain -ne $epoch4To8GainFloor `
    -or [double]$decision.fixed_gates.amount_candidate_exact_floor -ne $amountFloor `
    -or [double]$decision.fixed_gates.time_candidate_exact_floor -ne $timeFloor `
    -or [double]$decision.fixed_gates.payment_candidate_exact_floor -ne $paymentFloor `
    -or [double]$decision.fixed_gates.visible_status_raw_exact_floor -ne $statusTextFloor `
    -or [int]$decision.fixed_gates.status_non_success_to_success_max -ne 0 `
    -or $decision.blind_manifest_contract.test_opened_by_training -ne $false `
    -or (($decision.blind_manifest_contract.optimizer_supervision_splits -join ",") -ne "train") `
    -or (($decision.blind_manifest_contract.checkpoint_selection_splits -join ",") -ne "val") `
    -or [string]$decision.blind_manifest_contract.source_manifest_sha256 -ne [string]$blind.source_manifest_sha256 `
    -or [string]$decision.blind_manifest_contract.blind_manifest_sha256 -ne [string]$blind.blind_manifest_sha256 `
    -or $decision.passed -ne $true) {
    throw "Full-crop pilot decision does not satisfy its fixed analysis contract."
}

Write-Host "PILOT PASS: analysis only; no ONNX was exported and test remains unopened."
Write-Host ("  best={0:P2}; epoch4={1:P2}; epoch8={2:P2}; gain={3:P2}" -f `
    [double]$decision.observed.best_recipient_exact, `
    [double]$decision.observed.epoch4_recipient_exact, `
    [double]$decision.observed.epoch8_recipient_exact, `
    [double]$decision.observed.epoch4_to_8_gain)
Write-Host "  init=$requiredInit; decision=$decisionPath"
