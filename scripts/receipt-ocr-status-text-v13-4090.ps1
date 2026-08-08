[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PseudoLabels,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$SeedCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(1, 80)]
    [int]$Epochs = 30,
    [ValidateRange(1, 64)]
    [int]$BatchSize = 12,
    [ValidateRange(1, 80)]
    [int]$ValidationEvery = 4,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.001,
    [ValidateRange(0.0, 1.0)]
    [double]$StatusTextFloor = 0.90,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 32)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# These are the established delivery floors. They are constants rather than
# caller-overridable parameters so a status-head experiment cannot quietly
# weaken any of the four already protected OCR fields.
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$requiredStatusTextPolicy = "decode_and_normalize_review_only"
$requiredReviewValue = "review"
$legacyOutputNames = @(
    "amount_logits",
    "time_logits",
    "payment_logits",
    "status_logits",
    "amount_currency_style_logits",
    "amount_grouped_thousands_logits",
    "amount_sign_position_logits",
    "time_format_logits",
    "time_digit_logits",
    "payment_prefix_logits",
    "payment_bank_prefix_logits",
    "payment_tail_digit_logits",
    "payment_structure_logits",
    "payment_parentheses_logits",
    "recipient_logits"
)

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

function Read-GuardedJson([string]$Path) {
    Require-File $Path "JSON evidence"
    $normalized = (& $pythonExe $normalizer $Path) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to normalize JSON evidence: $Path (exit code $LASTEXITCODE)"
    }
    try {
        return ($normalized | ConvertFrom-Json)
    }
    catch {
        throw "Unable to parse JSON evidence: $Path. $($_.Exception.Message)"
    }
}

function Get-RequiredProperty([object]$Object, [string]$Name, [string]$Description) {
    if ($null -eq $Object) {
        throw "$Description is missing."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Description has no $Name."
    }
    # Keep acceptance.failures=[] as an empty array on Windows PowerShell 5.1.
    return ,$property.Value
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ExactMetric(
    [object]$Summary,
    [string]$Field,
    [string]$Description,
    [string]$MetricName = "raw_exact_match"
) {
    $byFieldProperty = $Summary.PSObject.Properties["by_field"]
    $fieldProperty = if ($null -eq $byFieldProperty -or $null -eq $byFieldProperty.Value) {
        $null
    }
    else {
        $byFieldProperty.Value.PSObject.Properties[$Field]
    }
    if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
        throw "$Description has no $Field metric."
    }
    $metricProperty = $fieldProperty.Value.PSObject.Properties[$MetricName]
    if ($null -eq $metricProperty -or $null -eq $metricProperty.Value) {
        throw "$Description has no $Field $MetricName metric."
    }
    return [double]$metricProperty.Value
}

function Get-StatusOovAudit([object]$Contract, [string]$Split) {
    $oovRootProperty = $Contract.PSObject.Properties["status_text_oov_by_split"]
    $splitProperty = if ($null -eq $oovRootProperty -or $null -eq $oovRootProperty.Value) {
        $null
    }
    else {
        $oovRootProperty.Value.PSObject.Properties[$Split]
    }
    if ($null -eq $splitProperty -or $null -eq $splitProperty.Value) {
        throw "v13 dataset contract has no status-text OOV audit for split '$Split'."
    }
    $audit = $splitProperty.Value
    $recordsProperty = $audit.PSObject.Properties["records"]
    $missingTextProperty = $audit.PSObject.Properties["missing_text_records"]
    $oovRecordsProperty = $audit.PSObject.Properties["oov_records"]
    $oovCharactersProperty = $audit.PSObject.Properties["oov_characters"]
    if ($null -eq $recordsProperty `
        -or $null -eq $missingTextProperty `
        -or $null -eq $oovRecordsProperty `
        -or $null -eq $oovCharactersProperty) {
        throw "v13 dataset contract has an incomplete status-text audit for split '$Split'."
    }
    $records = [int]$audit.records
    $missingTextRecords = [int]$audit.missing_text_records
    $oovRecords = [int]$audit.oov_records
    $oovCharacters = [int]$audit.oov_characters
    if ($records -lt 0 -or $missingTextRecords -lt 0 -or $oovRecords -lt 0 -or $oovCharacters -lt 0) {
        throw "v13 dataset contract has negative status-text counts for split '$Split'."
    }
    $maxPossibleExactMatch = if ($records -gt 0) {
        [double]($records - $oovRecords) / [double]$records
    }
    else {
        $null
    }
    return [ordered]@{
        split = $Split
        visible_status_records = $records
        missing_status_text_records = $missingTextRecords
        total_status_records = $records + $missingTextRecords
        oov_records = $oovRecords
        oov_characters = $oovCharacters
        checked = ($records -gt 0)
        max_possible_exact_match = $maxPossibleExactMatch
        calibration_note = if ($records -gt 0) {
            if ($oovRecords -eq 0 -and $oovCharacters -eq 0) {
                "visible status text present; zero train-charset OOV observed"
            }
            else {
                "held-out train-charset OOV is retained as an error; the exact-match floor remains mandatory"
            }
        }
        else {
            "no visible status text in this split; no status OCR claim is made"
        }
    }
}

function Assert-ArtifactHash([object]$Contract, [string]$ModelPath, [string]$Description) {
    $labelsPath = [IO.Path]::ChangeExtension($ModelPath, ".labels.json")
    Require-File $labelsPath "$Description labels"
    if ([string]$Contract.onnx_file -ne [IO.Path]::GetFileName($ModelPath) `
        -or [string]$Contract.onnx_sha256 -ne (Get-Sha256 $ModelPath) `
        -or [string]$Contract.labels_file -ne [IO.Path]::GetFileName($labelsPath) `
        -or [string]$Contract.labels_sha256 -ne (Get-Sha256 $labelsPath)) {
        throw "$Description ONNX/labels do not match their adjacent contract."
    }
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $normalizer "JSON summary normalizer"
Require-File $PseudoLabels "flat pseudo_labels.jsonl"
Require-Directory $DatasetRoot "pseudo-label crop root"
Require-File $SeedCheckpoint "wide1536 v12 best.pt"
if ([IO.Path]::GetExtension($SeedCheckpoint) -ne ".pt") {
    throw "SeedCheckpoint must be a PyTorch .pt checkpoint: $SeedCheckpoint"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse v13 output root: $OutputRoot"
}

$sourceTests = @(
    (Join-Path $repoRoot "tests\test_ocr_unified_v13.py"),
    (Join-Path $repoRoot "tests\test_receipt_mlnet_status_text_ctc.py"),
    (Join-Path $repoRoot "tests\test_receipt_mlnet_v13_delivery_scripts.py")
)
foreach ($sourceTest in $sourceTests) {
    Require-File $sourceTest "v13 source-contract test"
}

$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi did not report a CUDA GPU; refusing a non-GPU training run."
}

Write-Host "receipt_status_text_v13_4090 preflight"
Write-Host "  python=$pythonExe"
Write-Host "  pseudo-labels=$PseudoLabels"
Write-Host "  dataset-root=$DatasetRoot"
Write-Host "  wide1536-v12-seed=$SeedCheckpoint"
Write-Host "  output=$OutputRoot"
Write-Host ("  floors: amount={0:P2}, time={1:P2}, payment={2:P2}, recipient={3:P2}, status-text={4:P2}" -f `
    $amountFloor, $timeFloor, $paymentFloor, $recipientFloor, $StatusTextFloor)
Write-Host ("  GPU: {0}" -f ($gpuRows -join "; "))

Invoke-Python `
    (@("-m", "pytest", "-q") + $sourceTests) `
    "v13 source-contract tests"

if ($CheckOnly) {
    Write-Host "receipt_status_text_v13_4090 preflight=passed"
    exit 0
}

$manifestRoot = Join-Path $OutputRoot "manifest-v13"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$datasetContractPath = Join-Path $manifestRoot "dataset.contract.json"
$trainingRoot = Join-Path $OutputRoot "training-v13"
$candidateCheckpoint = Join-Path $trainingRoot "best.pt"
$artifactRoot = Join-Path $OutputRoot "artifacts"
$seedModel = Join-Path $artifactRoot "wide1536-v12-seed.onnx"
$candidateModel = Join-Path $artifactRoot "status-text-v13.onnx"
$validationEvidencePath = Join-Path $OutputRoot "v13_status_ocr_validation.json"

New-Item -ItemType Directory -Path $OutputRoot | Out-Null

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified_dataset",
    "--records", $PseudoLabels,
    "--output", $manifestRoot,
    "--architecture", "v13"
) "v13 manifest build"

Require-File $records "v13 unified manifest"
$datasetContract = Read-GuardedJson $datasetContractPath
if ([string]$datasetContract.kind -ne "receipt_unified_field_dataset_v6" `
    -or [string]$datasetContract.architecture -ne "v13" `
    -or [string]$datasetContract.status_text_target -ne "visible_transfer_status_cjk_text" `
    -or [string]$datasetContract.status_text_charset_source -ne "train_only_visible_transfer_status_cjk_text") {
    throw "Built manifest does not carry the strict v13 visible-status contract."
}
$trainStatusAudit = Get-StatusOovAudit $datasetContract "train"
$valStatusAudit = Get-StatusOovAudit $datasetContract "val"
$testStatusAudit = Get-StatusOovAudit $datasetContract "test"
if ([int]$trainStatusAudit.visible_status_records -le 0 `
    -or [int]$valStatusAudit.visible_status_records -le 0 `
    -or [int]$testStatusAudit.visible_status_records -le 0) {
    throw "v13 status-text delivery requires visible transfer status text in train, val, and test."
}
foreach ($statusAudit in @($trainStatusAudit, $valStatusAudit, $testStatusAudit)) {
    if ([int]$statusAudit.missing_status_text_records -ne 0) {
        throw ("v13 {0} contains {1} transfer-status records without teacher-grounded visible Chinese text; " +
            "refusing to shrink the status OCR denominator.") -f `
            $statusAudit.split, [int]$statusAudit.missing_status_text_records
    }
}
if ([int]$trainStatusAudit.oov_records -ne 0 -or [int]$trainStatusAudit.oov_characters -ne 0) {
    throw "v13 train status text must have zero OOV against its own frozen charset."
}
foreach ($heldOutAudit in @($valStatusAudit, $testStatusAudit)) {
    if ([int]$heldOutAudit.visible_status_records -gt 0 `
        -and [double]$heldOutAudit.max_possible_exact_match -lt $StatusTextFloor) {
        throw ("v13 {0} train-charset OOV makes the requested status exact floor impossible: " +
            "max={1:P2}, floor={2:P2}") -f `
            $heldOutAudit.split, [double]$heldOutAudit.max_possible_exact_match, $StatusTextFloor
    }
}

$trainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $records,
    "--dataset-root", $DatasetRoot,
    "--output", $trainingRoot,
    "--device", "cuda:0",
    "--architecture", "v13",
    "--image-height", "80",
    "--image-width", "512",
    "--recipient-input-height", "128",
    "--recipient-input-width", "1536",
    "--recipient-branch-channels", "24",
    "--base-channels", "32",
    "--numeric-hidden-size", "96",
    "--payment-hidden-size", "128",
    "--recipient-hidden-size", "256",
    "--recipient-open-text-layers", "2",
    "--recipient-open-text-heads", "8",
    "--recipient-open-text-feedforward", "2048",
    "--recipient-value-left-trim", "0.30",
    "--pooled-width", "8",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--status-text-loss-weight", "1.0",
    "--status-text-only-fine-tune",
    "--checkpoint-selection", "recipient_priority",
    "--checkpoint-min-amount-candidate-exact", "$amountFloor",
    "--checkpoint-min-time-candidate-exact", "$timeFloor",
    "--checkpoint-min-payment-candidate-exact", "$paymentFloor",
    "--init-checkpoint", $SeedCheckpoint,
    "--init-checkpoint-mode", "strict",
    "--ctc-loss-weight", "1.0",
    "--structured-loss-weight", "1.0",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
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
Invoke-Python $trainArgs "v13 CUDA status-text-only training"

$trainingSummaryPath = Join-Path $trainingRoot "training_summary.json"
$trainingSummary = Read-GuardedJson $trainingSummaryPath
$initialization = $trainingSummary.initialization
$fineTune = $trainingSummary.fine_tune_policy
$runtime = $trainingSummary.training_runtime
if ([string]$trainingSummary.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$trainingSummary.config.architecture_version -ne 13 `
    -or [int]$trainingSummary.config.recipient_input_width -ne 1536 `
    -or [string]$initialization.mode -ne "parameter_only_v12_to_v13_status_text_expansion" `
    -or [string]$initialization.source_kind -ne "receipt_unified_field_reader_v12" `
    -or [int]$initialization.source_config.architecture_version -ne 12 `
    -or [int]$initialization.source_config.recipient_input_width -ne 1536 `
    -or [int]$initialization.frozen_legacy_output_count -ne 15 `
    -or [int]$initialization.copied_legacy_tensor_count -le 0 `
    -or [int]$initialization.new_status_text_tensor_count -le 0 `
    -or [string]$initialization.financial_label_policy.mode -ne "checkpoint_legacy_label_maps_status_text_only_v1" `
    -or [string]$fineTune.mode -ne "status_text_only_v13" `
    -or [string]$fineTune.trainable_parameter_prefix -ne "status_text_" `
    -or [int]$fineTune.frozen_legacy_output_count -ne 15 `
    -or [string]$fineTune.full_validation_schedule -ne "epoch_1_every_n_and_final_epoch" `
    -or [int]$fineTune.validation_every -ne $ValidationEvery `
    -or $runtime.uses_cuda -ne $true `
    -or $runtime.status_text_only_training -ne $true `
    -or [string]$runtime.full_validation_schedule -ne "epoch_1_every_n_and_final_epoch" `
    -or [int]$runtime.validation_every -ne $ValidationEvery `
    -or -not ([string]$runtime.device).StartsWith("cuda", [StringComparison]::OrdinalIgnoreCase) `
    -or [string]$trainingSummary.status_text_runtime_policy -ne $requiredStatusTextPolicy) {
    throw "Training summary does not prove a CUDA-only additive v12-to-v13 status-head fine-tune."
}
Require-File $candidateCheckpoint "best v13 checkpoint"

$protectedMinima = $trainingSummary.checkpoint_selection_policy.protected_minimum_candidate_exact
if ([double]$protectedMinima.amount -ne $amountFloor `
    -or [double]$protectedMinima.time -ne $timeFloor `
    -or [double]$protectedMinima.payment_method_field -ne $paymentFloor) {
    throw "Training checkpoint policy changed a protected field floor."
}
$bestEpoch = [int]$trainingSummary.best_checkpoint_epoch
$bestRecord = @($trainingSummary.records | Where-Object { [int]$_.epoch -eq $bestEpoch })
if ($bestRecord.Count -ne 1 -or $bestRecord[0].checkpoint_selection_eligible -ne $true) {
    throw "Training did not select one floor-eligible best checkpoint."
}
$expectedSelectionMetric = "status_safety_then_transfer_status_raw_ctc_exact_then_recipient_exact_after_protected_candidate_exact_floors"
$expectedStatusSafetyScore = 0.0 - [double]$bestRecord[0].val_status_non_success_to_success
if ($trainingSummary.checkpoint_selection_policy.status_text_ctc_priority -ne $true `
    -or [string]$trainingSummary.checkpoint_selection_policy.selection_metric -ne $expectedSelectionMetric `
    -or $null -eq $bestRecord[0].val_ctc_by_field.transfer_status `
    -or [double]$bestRecord[0].checkpoint_selection_score[0] -ne $expectedStatusSafetyScore `
    -or [double]$bestRecord[0].checkpoint_selection_score[1] -ne `
        [double]$bestRecord[0].val_ctc_by_field.transfer_status.exact_match) {
    throw "Best checkpoint selection does not retain status safety before raw visible-status CTC exact match."
}
$bestMetrics = $bestRecord[0].val_candidate_text_by_field
if ([double]$bestMetrics.amount.exact_match -lt $amountFloor `
    -or [double]$bestMetrics.time.exact_match -lt $timeFloor `
    -or [double]$bestMetrics.payment_method_field.exact_match -lt $paymentFloor `
    -or [double]$bestRecord[0].val_ctc_by_field.transfer_status.exact_match -lt $StatusTextFloor) {
    throw "Best checkpoint does not retain amount/time/payment plus visible-status CTC floors."
}

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "export",
    "--checkpoint", $SeedCheckpoint,
    "--output", $seedModel
) "wide1536 v12 baseline export"
Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "export",
    "--checkpoint", $candidateCheckpoint,
    "--output", $candidateModel
) "v13 candidate export"

$seedContractPath = [IO.Path]::ChangeExtension($seedModel, ".contract.json")
$candidateContractPath = [IO.Path]::ChangeExtension($candidateModel, ".contract.json")
$candidateLabelsPath = [IO.Path]::ChangeExtension($candidateModel, ".labels.json")
$seedContract = Read-GuardedJson $seedContractPath
$candidateContract = Read-GuardedJson $candidateContractPath
Assert-ArtifactHash $seedContract $seedModel "v12 seed"
Assert-ArtifactHash $candidateContract $candidateModel "v13 candidate"
if ([string]$seedContract.kind -ne "receipt_unified_field_reader_v12" `
    -or [int]$seedContract.model.architecture_version -ne 12 `
    -or [int]$seedContract.model.recipient_input_width -ne 1536 `
    -or [string]$candidateContract.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$candidateContract.model.architecture_version -ne 13 `
    -or [int]$candidateContract.model.recipient_input_width -ne 1536) {
    throw "Exported artifacts are not the expected wide1536 v12/v13 pair."
}

$seedOutputProperties = @($seedContract.outputs.PSObject.Properties)
$candidateOutputProperties = @($candidateContract.outputs.PSObject.Properties)
if ($seedOutputProperties.Count -ne 15 -or $candidateOutputProperties.Count -ne 16) {
    throw "Exported v12/v13 output counts are not 15/16."
}
foreach ($outputName in $legacyOutputNames) {
    $seedOutput = $seedContract.outputs.PSObject.Properties[$outputName]
    $candidateOutput = $candidateContract.outputs.PSObject.Properties[$outputName]
    if ($null -eq $seedOutput -or $null -eq $candidateOutput `
        -or (($seedOutput.Value.shape | ConvertTo-Json -Compress) -ne `
            ($candidateOutput.Value.shape | ConvertTo-Json -Compress))) {
        throw "Legacy output ABI parity failed for $outputName."
    }
}
$statusTextOutput = $candidateContract.outputs.PSObject.Properties["status_text_logits"]
if ($null -ne $seedContract.outputs.PSObject.Properties["status_text_logits"] `
    -or $null -eq $statusTextOutput `
    -or [string]$candidateContract.status_text_runtime_policy -ne $requiredStatusTextPolicy `
    -or [string]$statusTextOutput.Value.runtime_policy -ne $requiredStatusTextPolicy `
    -or [string]$statusTextOutput.Value.review_value -ne $requiredReviewValue `
    -or [string]$candidateContract.status_head_policy.runtime_policy -ne "review_only") {
    throw "v13 status text is not an additive, review-only output."
}

$evaluationEvidence = @()
foreach ($statusAudit in @($valStatusAudit, $testStatusAudit)) {
    $split = [string]$statusAudit.split
    if ([int]$statusAudit.visible_status_records -le 0) {
        $evaluationEvidence += [ordered]@{
            split = $split
            evaluated = $false
            visible_status_records = 0
            non_success_truth_records = 0
            non_success_safety_calibrated = $false
            calibration_note = "no visible held-out status text; no status OCR or non-success safety claim is made"
        }
        continue
    }
    $classCountsProperty = $trainingSummary.status_class_counts.PSObject.Properties[$split]
    if ($null -eq $classCountsProperty -or $null -eq $classCountsProperty.Value) {
        throw "Training summary has no status class counts for split '$split'."
    }
    $classCounts = $classCountsProperty.Value
    $nonSuccessTruthRecords = [int]$classCounts.pending + [int]$classCounts.failed
    $evaluationOutput = Join-Path $OutputRoot ("onnx-" + $split + "-gpu")
    $evaluateArgs = @(
        "-m", "transfer_receipt_ai.ocr_unified", "evaluate",
        "--model", $candidateModel,
        "--records", $records,
        "--dataset-root", $DatasetRoot,
        "--split", $split,
        "--output", $evaluationOutput,
        "--device", "cuda:0",
        "--min-amount-exact-match", "$amountFloor",
        "--min-time-exact-match", "$timeFloor",
        "--min-payment-exact-match", "$paymentFloor",
        "--min-recipient-exact-match", "$recipientFloor",
        "--min-status-exact-match", "$StatusTextFloor",
        "--progress-every", "250"
    )
    $nonSuccessSafetyCalibrated = $nonSuccessTruthRecords -gt 0
    if ($nonSuccessSafetyCalibrated) {
        $evaluateArgs += @("--max-non-success-to-success", "0")
    }
    # The v13 base artifact may miss only the recipient floor. The production
    # hybrid PP-OCR route owns the final recipient >=90% gate, while every
    # amount/time/payment/status failure remains fatal here. Python exits 1
    # after writing a structurally complete summary for an acceptance miss, so
    # capture that code and validate the exact failure field below.
    & $pythonExe @evaluateArgs
    $evaluationExitCode = $LASTEXITCODE
    $evaluationSummaryPath = Join-Path $evaluationOutput "summary.json"
    $evaluationSummary = Read-GuardedJson $evaluationSummaryPath
    $acceptance = Get-RequiredProperty $evaluationSummary "acceptance" "v13 $split evaluation"
    $requestedValue = Get-RequiredProperty $acceptance "requested" "v13 $split acceptance"
    $passedValue = Get-RequiredProperty $acceptance "passed" "v13 $split acceptance"
    $rawFailures = Get-RequiredProperty $acceptance "failures" "v13 $split acceptance"
    if ($requestedValue -isnot [bool] `
        -or $requestedValue -ne $true `
        -or $passedValue -isnot [bool] `
        -or $rawFailures -isnot [Array]) {
        throw "v13 $split evaluation has an invalid aggregate acceptance schema."
    }
    $failures = @(
        $rawFailures |
            ForEach-Object {
                if ($_ -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$_)) {
                    throw "v13 $split acceptance failures must be non-empty strings."
                }
                [string]$_
            }
    )
    $nonRecipientFailures = @(
        $failures |
            Where-Object {
                -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal)
            }
    )
    $baseSummaryPassed = [bool]$passedValue
    $aggregateStateValid = (
        ($baseSummaryPassed -and $failures.Count -eq 0) `
        -or (-not $baseSummaryPassed `
            -and $failures.Count -gt 0 `
            -and $nonRecipientFailures.Count -eq 0)
    )
    $expectedEvaluationExitCode = if ($baseSummaryPassed) { 0 } else { 1 }
    $evaluationModel = [IO.Path]::GetFullPath([string]$evaluationSummary.model)
    $evaluationRecords = [IO.Path]::GetFullPath([string]$evaluationSummary.records)
    $statusMetrics = $evaluationSummary.by_field.transfer_status
    $statusCtcRecords = [int]$statusMetrics.ctc_records
    $statusCtcExactMatches = [int]$statusMetrics.ctc_raw_exact_matches
    $statusCtcExactMatch = [double]$statusMetrics.ctc_raw_exact_match
    if (-not $evaluationModel.Equals(
            [IO.Path]::GetFullPath($candidateModel),
            [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$evaluationSummary.model_sha256 -ne (Get-Sha256 $candidateModel) `
        -or -not $evaluationRecords.Equals(
            [IO.Path]::GetFullPath($records),
            [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$evaluationSummary.records_sha256 -ne (Get-Sha256 $records) `
        -or $statusCtcRecords -le 0 `
        -or $statusCtcRecords -ne [int]$statusMetrics.records `
        -or $statusCtcRecords -ne [int]$statusAudit.visible_status_records `
        -or $statusCtcExactMatches -lt 0 `
        -or $statusCtcExactMatches -gt $statusCtcRecords `
        -or [Math]::Abs(
            $statusCtcExactMatch - ([double]$statusCtcExactMatches / [double]$statusCtcRecords)
        ) -gt 0.000000000001 `
        -or $evaluationSummary.providers -notcontains "CUDAExecutionProvider" `
        -or -not $aggregateStateValid `
        -or $nonRecipientFailures.Count -ne 0 `
        -or $evaluationExitCode -ne $expectedEvaluationExitCode `
        -or [string]$evaluationSummary.status_text_policy.runtime_policy -ne $requiredStatusTextPolicy `
        -or [string]$evaluationSummary.status_text_policy.review_value -ne $requiredReviewValue `
        -or (Get-ExactMetric $evaluationSummary "amount" "$split evaluation") -lt $amountFloor `
        -or (Get-ExactMetric $evaluationSummary "time" "$split evaluation") -lt $timeFloor `
        -or (Get-ExactMetric $evaluationSummary "payment_method_field" "$split evaluation") -lt $paymentFloor `
        -or (Get-ExactMetric $evaluationSummary "transfer_status" "$split evaluation" "ctc_raw_exact_match") -lt $StatusTextFloor) {
        throw "v13 $split evaluation did not satisfy its GPU core-field, policy, or exact-match contract."
    }
    $recipientExactMatch = Get-ExactMetric $evaluationSummary "recipient_field" "$split evaluation"
    $requestedRecipientFloor = $acceptance.PSObject.Properties["min_recipient_exact_match"]
    if ($null -eq $requestedRecipientFloor `
        -or $null -eq $requestedRecipientFloor.Value `
        -or [double]$requestedRecipientFloor.Value -lt $recipientFloor) {
        throw "v13 $split evaluation weakens the final hybrid recipient floor."
    }
    $recipientDelegated = $recipientExactMatch -lt $recipientFloor
    if (($recipientDelegated -and ($baseSummaryPassed -or $failures.Count -eq 0)) `
        -or (-not $recipientDelegated -and (-not $baseSummaryPassed -or $failures.Count -ne 0))) {
        throw "v13 $split aggregate acceptance is inconsistent with its recipient metric."
    }
    if ($nonSuccessSafetyCalibrated) {
        $maxSafetyProperty = $acceptance.PSObject.Properties["max_non_success_to_success"]
        if ($null -eq $maxSafetyProperty `
            -or $null -eq $maxSafetyProperty.Value `
            -or [int]$maxSafetyProperty.Value -ne 0 `
            -or [int]$statusMetrics.non_success_to_success -ne 0) {
            throw "v13 $split evaluation did not preserve the zero non-success-to-success safety line."
        }
    }
    $evaluationEvidence += [ordered]@{
        split = $split
        evaluated = $true
        summary_path = $evaluationSummaryPath
        summary_sha256 = Get-Sha256 $evaluationSummaryPath
        visible_status_records = [int]$statusAudit.visible_status_records
        non_success_truth_records = $nonSuccessTruthRecords
        non_success_safety_calibrated = $nonSuccessSafetyCalibrated
        calibration_note = if ($nonSuccessSafetyCalibrated) {
            "pending/failed truth exists; max_non_success_to_success=0 requested and passed"
        }
        else {
            "no pending/failed truth in this split; no non-success safety claim is made"
        }
        status_text_exact_match = $statusCtcExactMatch
        status_non_success_to_success = [int]$statusMetrics.non_success_to_success
        base_summary_passed = $baseSummaryPassed
        base_summary_failures = $failures
        recipient_exact_match = $recipientExactMatch
        recipient_delegated_to_hybrid_formal = $recipientDelegated
        core_amount_time_payment_status_accepted = $true
        accepted = $baseSummaryPassed
    }
}

$recipientDelegationRequired = [bool](@(
        $evaluationEvidence |
            Where-Object { $_.recipient_delegated_to_hybrid_formal -eq $true }
    ).Count -gt 0)
$packagingValidationSummary = Join-Path $OutputRoot "onnx-val-gpu\summary.json"
Require-File $packagingValidationSummary "accepted v13 val ONNX summary for CPU packaging"

$validationEvidence = [ordered]@{
    schema_version = 1
    kind = "receipt_unified_status_text_v13_guarded_validation_v1"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    pseudo_labels = [IO.Path]::GetFullPath($PseudoLabels)
    pseudo_labels_sha256 = Get-Sha256 $PseudoLabels
    dataset_root = [IO.Path]::GetFullPath($DatasetRoot)
    manifest = [ordered]@{
        records = $records
        records_sha256 = Get-Sha256 $records
        contract = $datasetContractPath
        contract_sha256 = Get-Sha256 $datasetContractPath
        status_text_oov = @($trainStatusAudit, $valStatusAudit, $testStatusAudit)
    }
    seed = [ordered]@{
        checkpoint = [IO.Path]::GetFullPath($SeedCheckpoint)
        checkpoint_sha256 = Get-Sha256 $SeedCheckpoint
        exported_model = $seedModel
        exported_model_sha256 = Get-Sha256 $seedModel
        kind = [string]$seedContract.kind
        architecture_version = [int]$seedContract.model.architecture_version
        recipient_input_width = [int]$seedContract.model.recipient_input_width
    }
    candidate = [ordered]@{
        checkpoint = $candidateCheckpoint
        checkpoint_sha256 = Get-Sha256 $candidateCheckpoint
        model = $candidateModel
        model_sha256 = Get-Sha256 $candidateModel
        contract = $candidateContractPath
        contract_sha256 = Get-Sha256 $candidateContractPath
        labels = $candidateLabelsPath
        labels_sha256 = Get-Sha256 $candidateLabelsPath
        kind = [string]$candidateContract.kind
        architecture_version = [int]$candidateContract.model.architecture_version
        status_text_runtime_policy = $requiredStatusTextPolicy
        review_value = $requiredReviewValue
    }
    training = [ordered]@{
        summary = $trainingSummaryPath
        summary_sha256 = Get-Sha256 $trainingSummaryPath
        device = [string]$runtime.device
        cuda_device_name = [string]$runtime.cuda_device_name
        mode = [string]$fineTune.mode
        initialization = [string]$initialization.mode
    }
    legacy_output_parity = [ordered]@{
        passed = $true
        frozen_output_count = 15
        output_names = $legacyOutputNames
        proof = "source exact-tensor test + frozen legacy parameter audit + exported name/shape ABI comparison"
    }
    acceptance_floors = [ordered]@{
        amount = $amountFloor
        time = $timeFloor
        payment_method_field = $paymentFloor
        recipient_field = $recipientFloor
        visible_transfer_status_cjk_text = $StatusTextFloor
    }
    evaluations = $evaluationEvidence
    recipient_delivery_policy = [ordered]@{
        final_floor = $recipientFloor
        base_v13_below_floor_may_only_fail_recipient = $true
        delegated_runtime = "hybrid_ppocr"
        final_gate = "10016-receipt CPU formal"
        final_gate_required = $true
        delegated_in_any_held_out_split = $recipientDelegationRequired
    }
    cpu_packaging = [ordered]@{
        performed = $false
        next_script = "scripts/receipt-mlnet-unified-package-validate-4090.ps1"
        run_directory = [IO.Path]::GetFullPath($OutputRoot)
        unified_model_path = $candidateModel
        unified_model_sha256 = Get-Sha256 $candidateModel
        onnx_validation_summary_path = $packagingValidationSummary
        onnx_validation_summary_sha256 = Get-Sha256 $packagingValidationSummary
        required_runtime_flavor = "cpu"
        required_rectification = "max-side-1600"
        include_device_model = $true
        recipient_delegated_to_hybrid_formal = $recipientDelegationRequired
    }
}
$validationEvidence | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $validationEvidencePath -Encoding UTF8

Write-Host ""
Write-Host "PASS: additive v13 visible-status OCR candidate is ready for existing CPU packaging."
Write-Host "  model=$candidateModel"
Write-Host "  manifest=$records"
Write-Host "  evidence=$validationEvidencePath"
Write-Host "  recipient-delegated-to-hybrid-formal=$recipientDelegationRequired"
Write-Host "  CPU packaging intentionally not run; use scripts\receipt-mlnet-unified-package-validate-4090.ps1 next with:"
Write-Host "    -RunDirectory $OutputRoot"
Write-Host "    -UnifiedModelPath $candidateModel"
Write-Host "    -OnnxValidationSummaryPath $packagingValidationSummary"
