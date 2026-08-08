[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$OutputRoot,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# These are evidence constants, not caller-tunable thresholds.  This runner
# never uses 75% as a delivery floor: it is only the 8 -> 16 epoch analysis
# continuation gate.  The random-root financial branches remain ineligible
# and will be discarded by the later recipient_* sanitizer.
$amountDeliveryFloor = 0.7885
$timeDeliveryFloor = 0.9840
$paymentDeliveryFloor = 0.9325
$recipientDeliveryFloor = 0.90
$recipientContinuationFloor = 0.75
$epoch4To8GainFloor = 0.02
$rootEpochs = 1
$pilotEpochs = 8

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$fullRecords = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
$datasetRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$trainer = Join-Path $repoRoot "src\transfer_receipt_ai\ocr_unified.py"
$blindBuilder = Join-Path $repoRoot "src\transfer_receipt_ai\recipient_blind_manifest.py"
$verifier = Join-Path $repoRoot "src\transfer_receipt_ai\recipient_random_bootstrap.py"
$sourceTest = Join-Path $repoRoot "tests\test_recipient_random_bootstrap.py"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $TeacherRoot ("recipient-random-bootstrap-analysis-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$blindRoot = Join-Path $OutputRoot "blind-train-val"
$blindRecords = Join-Path $blindRoot "unified_fields.train-val.jsonl"
$blindContract = Join-Path $blindRoot "blind.contract.json"
$snapshotRoot = Join-Path $OutputRoot "input-snapshot"
$inputContract = Join-Path $OutputRoot "bootstrap-input.contract.json"
$rootOutput = Join-Path $OutputRoot "random-root-1e"
$pilotOutput = Join-Path $OutputRoot "strict-recipient-warmstart-8e"
$decisionPath = Join-Path $OutputRoot "analysis-decision.json"

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
        throw "Refusing to reuse bootstrap output: $Path"
    }
    $ancestor = Split-Path -Parent $Path
    while (-not [string]::IsNullOrWhiteSpace($ancestor)) {
        $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Bootstrap output must not traverse a symlink/junction/reparse ancestor: $ancestor"
        }
        $next = Split-Path -Parent $ancestor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $ancestor) {
            break
        }
        $ancestor = $next
    }
}

function Require-NonReparseInput([string]$Path, [string]$Description) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $item = Get-Item -LiteralPath $cursor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "${Description} traverses a symlink/junction/reparse point: $cursor"
        }
        $next = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $cursor) {
            break
        }
        $cursor = $next
    }
}

function Invoke-Python([string[]]$CommandArguments, [string]$Description) {
    & $pythonExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-Unchanged([string]$Path, [string]$ExpectedSha256, [string]$Description) {
    Require-File $Path $Description
    $observed = Get-Sha256 $Path
    if ($observed -ne $ExpectedSha256) {
        throw "$Description changed during the run: $Path"
    }
}

function Seal-ReadOnlyEvidence([string]$Path, [string]$Description) {
    Require-File $Path $Description
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $item.IsReadOnly = $true
    $sealed = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $sealed.IsReadOnly) {
        throw "Unable to seal ${Description} read-only: $Path"
    }
}

if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -lt 1) {
    throw "This fixed launcher requires Windows PowerShell 5.1. Observed: $($PSVersionTable.PSVersion)"
}
Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $fullRecords "full v12 r3 manifest"
Require-Directory $datasetRoot "v12 r3 crop root"
Require-File $trainer "unified trainer"
Require-File $blindBuilder "blind manifest builder"
Require-File $verifier "bootstrap verifier"
Require-File $sourceTest "bootstrap source-contract test"
Require-NonReparseInput $fullRecords "full v12 r3 manifest"
Require-NonReparseInput $datasetRoot "v12 r3 crop root"
Require-FreshNonReparseOutput $OutputRoot

$gpuRows = @(& nvidia-smi --query-gpu=index,name --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0 -or [string]$gpuRows[0] -notmatch "^0\s*,.*4090") {
    throw "CUDA device 0 must be an RTX 4090. Observed: $($gpuRows -join '; ')"
}
Invoke-Python @("-m", "transfer_receipt_ai.recipient_random_bootstrap", "probe-cuda") "CUDA:0 probe"

Write-Host "recipient_random_bootstrap_4090 preflight"
Write-Host "  PowerShell=5.1; CUDA=cuda:0 RTX 4090"
Write-Host "  ancestry=completely random v12 root; no legacy checkpoint"
Write-Host "  topology=v12 width1536 layers2; root=1 epoch; strict recipient-only=8 epochs"
Write-Host "  optimizer=train only; checkpoint selection=val only; test=physically excluded"
Write-Host "  validation=every epoch; analysis-only; no ONNX; no production authorization"
Write-Host ("  unchanged delivery floors: amount={0:P2}, time={1:P2}, payment={2:P2}, recipient={3:P2}" -f `
    $amountDeliveryFloor, $timeDeliveryFloor, $paymentDeliveryFloor, $recipientDeliveryFloor)
Write-Host ("  continuation only: best recipient>={0:P2}; epoch4-to-8 gain>={1:P2}" -f `
    $recipientContinuationFloor, $epoch4To8GainFloor)
Write-Host "  random-root amount/time/payment are categorically ineligible for delivery and will be discarded"
Write-Host "  output=$OutputRoot"

Invoke-Python @("-m", "pytest", "-q", $sourceTest) "bootstrap source-contract tests"
if ($CheckOnly) {
    Write-Host "recipient_random_bootstrap_4090 preflight=passed"
    exit 0
}

New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
$createdOutput = Get-Item -LiteralPath $OutputRoot -Force -ErrorAction Stop
if (($createdOutput.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Fresh bootstrap output unexpectedly became a reparse point: $OutputRoot"
}

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_blind_manifest",
    "--source", $fullRecords,
    "--output", $blindRecords,
    "--contract", $blindContract
) "blind train/val manifest build"

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_random_bootstrap", "bind",
    "--source-manifest", $fullRecords,
    "--blind-manifest", $blindRecords,
    "--blind-contract", $blindContract,
    "--dataset-root", $datasetRoot,
    "--snapshot-root", $snapshotRoot,
    "--output", $inputContract,
    "--runner", $MyInvocation.MyCommand.Path,
    "--trainer", $trainer,
    "--blind-builder", $blindBuilder,
    "--verifier", $verifier
) "hash-bound bootstrap input contract"

$fullRecordsSha256 = Get-Sha256 $fullRecords
$blindRecordsSha256 = Get-Sha256 $blindRecords
$blindContractSha256 = Get-Sha256 $blindContract
$inputContractSha256 = Get-Sha256 $inputContract
$runnerSha256 = Get-Sha256 $MyInvocation.MyCommand.Path
$trainerSha256 = Get-Sha256 $trainer
$blindBuilderSha256 = Get-Sha256 $blindBuilder
$verifierSha256 = Get-Sha256 $verifier

$commonTrainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $blindRecords,
    "--dataset-root", $snapshotRoot,
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
    "--learning-rate", "0.0001",
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
    "--checkpoint-selection", "balanced",
    "--ctc-loss-weight", "0.75",
    "--structured-loss-weight", "1.0",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--num-workers", "4",
    "--prefetch-factor", "2",
    "--persistent-workers",
    "--train-progress-every", "250",
    "--validation-every", "1",
    "--cuda-tf32",
    "--cudnn-benchmark"
)

Write-Host ""
Write-Host "stage 1/2: completely random same-topology root (1 epoch)"
$rootArgs = $commonTrainArgs + @(
    "--output", $rootOutput,
    "--epochs", "$rootEpochs",
    "--seed", "424242"
)
Invoke-Python $rootArgs "random same-topology root training"

Seal-ReadOnlyEvidence (Join-Path $rootOutput "training_summary.json") "random-root summary"
Seal-ReadOnlyEvidence (Join-Path $rootOutput "best.pt") "random-root best checkpoint"
Seal-ReadOnlyEvidence (Join-Path $rootOutput "last.pt") "random-root last checkpoint"

Require-Unchanged $fullRecords $fullRecordsSha256 "full v12 r3 manifest"
Require-Unchanged $blindRecords $blindRecordsSha256 "blind manifest"
Require-Unchanged $blindContract $blindContractSha256 "blind contract"
Require-Unchanged $inputContract $inputContractSha256 "bootstrap input contract"
Require-Unchanged $MyInvocation.MyCommand.Path $runnerSha256 "bootstrap runner"
Require-Unchanged $trainer $trainerSha256 "unified trainer"
Require-Unchanged $blindBuilder $blindBuilderSha256 "blind manifest builder"
Require-Unchanged $verifier $verifierSha256 "bootstrap verifier"
$rootCheckpoint = Join-Path $rootOutput "best.pt"
Require-File $rootCheckpoint "random-root best checkpoint"

Write-Host ""
Write-Host "stage 2/2: fresh strict recipient-only warm-start (8 epochs)"
$pilotArgs = $commonTrainArgs + @(
    "--output", $pilotOutput,
    "--epochs", "$pilotEpochs",
    "--seed", "424242",
    "--recipient-only-fine-tune",
    "--init-checkpoint", $rootCheckpoint,
    "--init-checkpoint-mode", "strict"
)
Invoke-Python $pilotArgs "strict recipient-only warm-start training"

Seal-ReadOnlyEvidence (Join-Path $pilotOutput "training_summary.json") "strict warm-start summary"
Seal-ReadOnlyEvidence (Join-Path $pilotOutput "best.pt") "strict warm-start best checkpoint"
Seal-ReadOnlyEvidence (Join-Path $pilotOutput "last.pt") "strict warm-start last checkpoint"

Require-Unchanged $fullRecords $fullRecordsSha256 "full v12 r3 manifest"
Require-Unchanged $blindRecords $blindRecordsSha256 "blind manifest"
Require-Unchanged $blindContract $blindContractSha256 "blind contract"
Require-Unchanged $inputContract $inputContractSha256 "bootstrap input contract"
Require-Unchanged $MyInvocation.MyCommand.Path $runnerSha256 "bootstrap runner"
Require-Unchanged $trainer $trainerSha256 "unified trainer"
Require-Unchanged $blindBuilder $blindBuilderSha256 "blind manifest builder"
Require-Unchanged $verifier $verifierSha256 "bootstrap verifier"

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_random_bootstrap", "finalize",
    "--input-contract", $inputContract,
    "--root-output", $rootOutput,
    "--pilot-output", $pilotOutput,
    "--output", $decisionPath
) "random-root bootstrap evidence finalization"

$decision = (Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json)
Write-Host ""
Write-Host "recipient_random_bootstrap_4090 final"
Write-Host ("  best recipient={0:P2}; epoch4={1:P2}; epoch8={2:P2}; gain={3:P2}" -f `
    [double]$decision.recipient_observed.best_exact, `
    [double]$decision.recipient_observed.epoch4_exact, `
    [double]$decision.recipient_observed.epoch8_exact, `
    [double]$decision.recipient_observed.epoch4_to_8_gain)
Write-Host ("  delivery recipient target {0:P2} reached={1}" -f `
    $recipientDeliveryFloor, [bool]$decision.recipient_delivery_target_reached)
Write-Host "  DELIVERY=NOT AUTHORIZED; ONNX=NOT EXPORTED; nonrecipient random-root branches=INELIGIBLE"
Write-Host "  decision=$decisionPath"
if ($decision.continuation_16_epoch_authorized -ne $true) {
    Write-Host "  16-epoch continuation=BLOCKED by the fixed 75%/epoch4-to-8 analysis gates"
    exit 3
}
Write-Host "  16-epoch continuation=AUTHORIZED FOR ANALYSIS ONLY"
exit 0
