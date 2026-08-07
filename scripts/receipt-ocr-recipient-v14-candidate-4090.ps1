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
    [ValidateRange(1, 100)]
    [int]$Epochs = 60,
    [ValidateRange(1, 32)]
    [int]$BatchSize = 10,
    [ValidateRange(1, 20)]
    [int]$ValidationEvery = 2,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.0003,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$Pilot,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fixed delivery floors. They are constants, not caller parameters.
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusTextFloor = 0.90
$requiredBackbone = "residual_positional_transformer_v2"
$requiredInit = "recipient_visual_context_reinit"
$requiredStatusPolicy = "decode_and_normalize_review_only"
$pilotMinimumBestRecipient = 0.75
$pilotMinimumEpoch4To8Gain = 0.02

if ($Pilot -and $Epochs -ne 8) {
    throw "Pilot mode is fixed to exactly 8 epochs so its stop rule remains comparable."
}

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

function Get-RawExact([object]$Summary, [string]$Field) {
    $fieldProperty = $Summary.by_field.PSObject.Properties[$Field]
    if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
        throw "Validation summary has no $Field field."
    }
    $metric = $fieldProperty.Value.PSObject.Properties["raw_exact_match"]
    if ($null -eq $metric -or $null -eq $metric.Value) {
        throw "Validation summary has no $Field raw_exact_match."
    }
    return [double]$metric.Value
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $normalizer "JSON normalizer"
Require-File $FullRecords "full v13 unified manifest"
Require-Directory $DatasetRoot "recipient crop root"
Require-File $SeedCheckpoint "accepted v13 seed checkpoint"
if ([IO.Path]::GetExtension($SeedCheckpoint) -ne ".pt") {
    throw "SeedCheckpoint must be a PyTorch .pt file."
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse recipient v14 candidate output: $OutputRoot"
}

$sourceTests = @(
    (Join-Path $repoRoot "tests\test_recipient_v14_candidate.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v12.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v13.py")
)
foreach ($sourceTest in $sourceTests) {
    Require-File $sourceTest "recipient v14 contract test"
}
$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi did not report a CUDA GPU."
}

Write-Host "receipt_recipient_v14_candidate preflight"
Write-Host "  architecture=v13 ABI + residual positional Transformer recipient branch"
Write-Host "  optimizer=train only; checkpoint selection=val only; test=physically excluded"
Write-Host ("  fixed floors: amount={0:P2}, time={1:P2}, payment={2:P2}, recipient={3:P2}, status={4:P2}" -f `
    $amountFloor, $timeFloor, $paymentFloor, $recipientFloor, $statusTextFloor)
Write-Host ("  GPU: {0}" -f ($gpuRows -join "; "))

Invoke-Python ((@("-m", "pytest", "-q")) + $sourceTests) "recipient v14 source-contract tests"
if ($CheckOnly) {
    Write-Host "receipt_recipient_v14_candidate preflight=passed"
    exit 0
}

$blindRoot = Join-Path $OutputRoot "blind-train-val"
$blindRecords = Join-Path $blindRoot "unified_fields.train-val.jsonl"
$blindContractPath = Join-Path $blindRoot "blind.contract.json"
$trainingRoot = Join-Path $OutputRoot "training-v14-candidate"
$checkpoint = Join-Path $trainingRoot "best.pt"
$artifactRoot = Join-Path $OutputRoot "artifacts"
$model = Join-Path $artifactRoot "recipient-v14-candidate.onnx"
$validationRoot = Join-Path $OutputRoot "onnx-val-gpu"
$validationSummaryPath = Join-Path $validationRoot "summary.json"
$evidencePath = Join-Path $OutputRoot "recipient_v14_candidate.json"

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_blind_manifest",
    "--source", $FullRecords,
    "--output", $blindRecords,
    "--contract", $blindContractPath
) "blind train/val manifest build"

$blindContract = Read-Json $blindContractPath
if ([string]$blindContract.kind -ne "receipt_recipient_blind_train_val_manifest_v1" `
    -or $blindContract.test_labels_used -ne $false `
    -or $blindContract.test_metrics_computed -ne $false `
    -or (($blindContract.optimizer_supervision_splits -join ",") -ne "train") `
    -or (($blindContract.checkpoint_selection_splits -join ",") -ne "val") `
    -or (($blindContract.final_gate_only_splits -join ",") -ne "test") `
    -or [int]$blindContract.split_counts.train -le 0 `
    -or [int]$blindContract.split_counts.val -le 0 `
    -or [int]$blindContract.split_counts.test_excluded -le 0) {
    throw "Blind manifest does not prove train/val/test isolation."
}

$trainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $blindRecords,
    "--dataset-root", $DatasetRoot,
    "--output", $trainingRoot,
    "--device", "cuda:0",
    "--architecture", "v13",
    "--image-height", "80",
    "--image-width", "512",
    "--base-channels", "32",
    "--numeric-hidden-size", "96",
    "--payment-hidden-size", "128",
    "--recipient-input-height", "128",
    "--recipient-input-width", "1536",
    "--recipient-value-left-trim", "0.30",
    "--recipient-branch-channels", "16",
    "--recipient-hidden-size", "192",
    "--recipient-open-text-layers", "4",
    "--recipient-open-text-heads", "8",
    "--recipient-open-text-feedforward", "1536",
    "--recipient-open-text-dropout", "0.10",
    "--recipient-backbone", $requiredBackbone,
    "--recipient-train-augmentation", "robust_v2",
    "--recipient-train-splits", "train",
    "--recipient-low-confidence-threshold", "0.95",
    "--recipient-low-confidence-loss-weight", "0.50",
    "--recipient-confidence-curriculum-epochs", "10",
    "--recipient-tail-rare-character-max-support", "3",
    "--recipient-tail-rare-character-loss-weight", "1.5",
    "--recipient-tail-long-text-min-length", "9",
    "--recipient-tail-long-text-loss-weight", "1.5",
    "--recipient-only-fine-tune",
    "--init-checkpoint", $SeedCheckpoint,
    "--init-checkpoint-mode", $requiredInit,
    "--checkpoint-selection", "recipient_priority",
    "--checkpoint-min-amount-candidate-exact", "$amountFloor",
    "--checkpoint-min-time-candidate-exact", "$timeFloor",
    "--checkpoint-min-payment-candidate-exact", "$paymentFloor",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--ctc-loss-weight", "1.0",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--validation-every", "$ValidationEvery",
    "--seed", "42",
    "--num-workers", "$NumWorkers",
    "--prefetch-factor", "$PrefetchFactor",
    "--train-progress-every", "$TrainProgressEvery",
    "--cuda-tf32",
    "--cudnn-benchmark"
)
if ($NumWorkers -gt 0) {
    $trainArgs += "--persistent-workers"
}
Invoke-Python $trainArgs "blind recipient v14 CUDA training"

$trainingSummaryPath = Join-Path $trainingRoot "training_summary.json"
$training = Read-Json $trainingSummaryPath
$fineTune = $training.fine_tune_policy
$runtime = $training.training_runtime
$initialization = $training.initialization
$requiredRecipientMap = "fresh_train_only_reinitialized_recipient_v1"
if ([string]$training.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$training.config.architecture_version -ne 13 `
    -or [string]$training.config.recipient_backbone -ne $requiredBackbone `
    -or [int]$training.config.recipient_open_text_layers -ne 4 `
    -or [Math]::Abs([double]$training.config.recipient_open_text_dropout - 0.10) -gt 0.000000001 `
    -or [string]$initialization.mode -ne "parameter_only_recipient_visual_context_reinit" `
    -or [string]$initialization.source_kind -ne "receipt_unified_field_reader_v13" `
    -or [string]$initialization.financial_label_policy.recipient_character_map.mode -ne $requiredRecipientMap `
    -or [string]$fineTune.mode -ne "recipient_only_v13" `
    -or [string]$fineTune.trainable_parameter_prefix -ne "recipient_" `
    -or [string]$fineTune.training_forward -ne "private_recipient_branch_only_v13" `
    -or [string]$training.recipient_train_split_policy.mode -ne "standard_train_only" `
    -or (($training.recipient_train_split_policy.splits -join ",") -ne "train") `
    -or [string]$training.recipient_train_augmentation_policy.mode -ne "robust_v2" `
    -or $runtime.uses_cuda -ne $true `
    -or -not ([string]$runtime.device).StartsWith("cuda", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Training summary does not prove the blind v14 recipient recipe."
}
if ([int]$training.field_counts.recipient_field.test -ne 0 `
    -or [int]$training.recipient_oov_by_split.test.records -ne 0) {
    throw "Training process observed test recipient labels; candidate is invalid."
}
$bestEpoch = [int]$training.best_checkpoint_epoch
$bestRows = @($training.records | Where-Object { [int]$_.epoch -eq $bestEpoch })
if ($bestRows.Count -ne 1 -or $bestRows[0].checkpoint_selection_eligible -ne $true) {
    throw "Training did not select exactly one val-eligible checkpoint."
}
$bestRecipient = [double]$bestRows[0].val_candidate_text_by_field.recipient_field.exact_match
if ($Pilot) {
    $epoch4Rows = @($training.records | Where-Object { [int]$_.epoch -eq 4 })
    $epoch8Rows = @($training.records | Where-Object { [int]$_.epoch -eq 8 })
    if ($epoch4Rows.Count -ne 1 -or $epoch8Rows.Count -ne 1 `
        -or $null -eq $epoch4Rows[0].val_candidate_text_by_field.recipient_field `
        -or $null -eq $epoch8Rows[0].val_candidate_text_by_field.recipient_field) {
        throw "Pilot stop audit requires recipient validation at epochs 4 and 8."
    }
    $epoch4Recipient = [double]$epoch4Rows[0].val_candidate_text_by_field.recipient_field.exact_match
    $epoch8Recipient = [double]$epoch8Rows[0].val_candidate_text_by_field.recipient_field.exact_match
    $pilotGain = $epoch8Recipient - $epoch4Recipient
    if ($bestRecipient -lt $pilotMinimumBestRecipient) {
        throw ("PILOT STOP: best val recipient {0:P2} is below {1:P2}; do not run 60 epochs." -f `
            $bestRecipient, $pilotMinimumBestRecipient)
    }
    if ($pilotGain -lt $pilotMinimumEpoch4To8Gain) {
        throw ("PILOT STOP: epoch4-to-8 gain {0:P2} is below {1:P2}; do not run 60 epochs." -f `
            $pilotGain, $pilotMinimumEpoch4To8Gain)
    }
    Write-Host "PILOT PASS: val trend justifies one fresh 60-epoch train/val-only run."
    Write-Host ("  best={0:P2}; epoch4={1:P2}; epoch8={2:P2}; gain={3:P2}" -f `
        $bestRecipient, $epoch4Recipient, $epoch8Recipient, $pilotGain)
    Write-Host "  test remains unopened; use a new OutputRoot for the full candidate."
    exit 0
}
if ($bestRecipient -lt $recipientFloor) {
    throw ("Val recipient exact {0:P2} is below {1:P2}; test remains unopened." -f $bestRecipient, $recipientFloor)
}
Require-File $checkpoint "best recipient v14 checkpoint"

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "export",
    "--checkpoint", $checkpoint,
    "--output", $model
) "recipient v14 ONNX export"

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "evaluate",
    "--model", $model,
    "--records", $blindRecords,
    "--dataset-root", $DatasetRoot,
    "--split", "val",
    "--output", $validationRoot,
    "--device", "cuda:0",
    "--min-amount-exact-match", "$amountFloor",
    "--min-time-exact-match", "$timeFloor",
    "--min-payment-exact-match", "$paymentFloor",
    "--min-recipient-exact-match", "$recipientFloor",
    "--min-status-exact-match", "$statusTextFloor",
    "--max-non-success-to-success", "0",
    "--progress-every", "250"
) "recipient v14 val-only ONNX gate"

$validation = Read-Json $validationSummaryPath
$statusRaw = [double]$validation.by_field.transfer_status.ctc_raw_exact_match
if ($validation.providers -notcontains "CUDAExecutionProvider" `
    -or $validation.acceptance.passed -ne $true `
    -or [string]$validation.status_text_policy.runtime_policy -ne $requiredStatusPolicy `
    -or (Get-RawExact $validation "amount") -lt $amountFloor `
    -or (Get-RawExact $validation "time") -lt $timeFloor `
    -or (Get-RawExact $validation "payment_method_field") -lt $paymentFloor `
    -or (Get-RawExact $validation "recipient_field") -lt $recipientFloor `
    -or $statusRaw -lt $statusTextFloor `
    -or [int]$validation.by_field.transfer_status.non_success_to_success -ne 0) {
    throw "Val-only ONNX evidence did not pass fixed delivery floors."
}

$modelContractPath = [IO.Path]::ChangeExtension($model, ".contract.json")
$modelLabelsPath = [IO.Path]::ChangeExtension($model, ".labels.json")
Require-File $modelContractPath "candidate ONNX contract"
Require-File $modelLabelsPath "candidate ONNX labels"
$evidence = [ordered]@{
    schema_version = 1
    kind = "receipt_recipient_v14_blind_candidate_v1"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    split_policy = [ordered]@{
        optimizer_supervision = @("train")
        checkpoint_selection = @("val")
        final_gate_only = @("test")
        test_evaluated = $false
        blind_contract = $blindContractPath
        blind_contract_sha256 = Get-Sha256 $blindContractPath
    }
    full_manifest = [IO.Path]::GetFullPath($FullRecords)
    full_manifest_sha256 = Get-Sha256 $FullRecords
    blind_manifest = $blindRecords
    blind_manifest_sha256 = Get-Sha256 $blindRecords
    candidate = [ordered]@{
        checkpoint = $checkpoint
        checkpoint_sha256 = Get-Sha256 $checkpoint
        model = $model
        model_sha256 = Get-Sha256 $model
        contract = $modelContractPath
        contract_sha256 = Get-Sha256 $modelContractPath
        labels = $modelLabelsPath
        labels_sha256 = Get-Sha256 $modelLabelsPath
        architecture_version = 13
        recipe_name = "recipient_v14_residual_positional_transformer"
        backbone = $requiredBackbone
    }
    training = [ordered]@{
        summary = $trainingSummaryPath
        summary_sha256 = Get-Sha256 $trainingSummaryPath
        best_epoch = $bestEpoch
    }
    val_evaluation = [ordered]@{
        summary = $validationSummaryPath
        summary_sha256 = Get-Sha256 $validationSummaryPath
        amount = Get-RawExact $validation "amount"
        time = Get-RawExact $validation "time"
        payment_method_field = Get-RawExact $validation "payment_method_field"
        recipient_field = Get-RawExact $validation "recipient_field"
        visible_transfer_status_cjk_text = $statusRaw
        status_non_success_to_success = [int]$validation.by_field.transfer_status.non_success_to_success
    }
    fixed_floors = [ordered]@{
        amount = $amountFloor
        time = $timeFloor
        payment_method_field = $paymentFloor
        recipient_field = $recipientFloor
        visible_transfer_status_cjk_text = $statusTextFloor
    }
}
$evidence | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $evidencePath -Encoding UTF8

Write-Host ""
Write-Host "PASS: val-selected recipient v14 candidate is sealed; test remains unopened."
Write-Host "  candidate_evidence=$evidencePath"
Write-Host "  model=$model"
Write-Host "  trusted_full_manifest_sha256=$($evidence.full_manifest_sha256)"
Write-Host "  next=scripts\receipt-ocr-recipient-v14-final-gate-4090.ps1"
