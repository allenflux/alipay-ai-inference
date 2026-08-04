[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 80)]
    [int]$Epochs = 80,
    [string]$RunName,
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [ValidateRange(0.0, 16.0)]
    [double]$RecipientLossWeight = 4.0,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.0001,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

# These are the actual r3/4090 assets, not the old generic r3 directory names.
# The baseline is evaluated again on exactly this manifest before any training
# begins, so the historical floors cannot silently cross dataset splits.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$seedCheckpoint = Join-Path $TeacherRoot "unified-run-v12-120k-r2-recipient-priority\best.pt"
$baselineModel = Join-Path $TeacherRoot "models\receipt_unified_field_reader_v12_120k_r2_recipient24_h256.onnx"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unified-run-v12-r3-4090-recipient-only-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
$output = Join-Path $TeacherRoot $RunName
$baselineOutput = Join-Path $TeacherRoot ("unified-eval-v12-r3-4090-warm-baseline-" + (Get-Date -Format "yyyyMMdd-HHmmss"))

foreach ($required in @(
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot },
    @{ Name = "r2 baseline checkpoint"; Path = $seedCheckpoint },
    @{ Name = "r2 baseline ONNX"; Path = $baselineModel }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Refusing to reuse training output: $output"
}

function Get-ExactDisplay([object]$Metric) {
    if ($null -eq $Metric) {
        return "n/a"
    }
    return ("{0}/{1}={2:P2}" -f $Metric.raw_exact_matches, $Metric.records, $Metric.raw_exact_match)
}

function Get-TrainingExactDisplay([object]$Record, [string]$Field) {
    if ($null -eq $Record -or $null -eq $Record.val_candidate_text_by_field) {
        return "n/a"
    }
    $metric = $Record.val_candidate_text_by_field.PSObject.Properties[$Field].Value
    if ($null -eq $metric) {
        return "n/a"
    }
    return ("{0}/{1}={2:P2}" -f $metric.exact_matches, $metric.records, $metric.exact_match)
}

Write-Host "guarded_4090_recipient_only preflight"
Write-Host "  seed=$seedCheckpoint"
Write-Host "  records=$records"
Write-Host "  output=$output"
Write-Host ("  floors: amount={0:P2}, time={1:P2}, payment={2:P2}" -f $AmountFloor, $TimeFloor, $PaymentFloor)
Write-Host ("  recipe: recipient-only, lr={0}, workers={1}, prefetch={2}, TF32=on, cuDNN-benchmark=on" -f $LearningRate, $NumWorkers, $PrefetchFactor)
try {
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}
catch {
    Write-Host "  nvidia-smi=unavailable"
}
if ($CheckOnly) {
    Write-Host "guarded_4090_recipient_only preflight=passed"
    exit 0
}

# Establish the baseline on this exact r3 validation split.  A comparison
# against a different older split must not authorize an 80-epoch run.
$baselineArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "evaluate",
    "--model", $baselineModel,
    "--records", $records,
    "--dataset-root", $labelsRoot,
    "--split", "val",
    "--output", $baselineOutput,
    "--device", "cuda:0"
)
& python @baselineArgs
if ($LASTEXITCODE -ne 0) {
    throw "Baseline evaluation failed with exit code $LASTEXITCODE"
}
$baseline = Get-Content (Join-Path $baselineOutput "summary.json") -Raw | ConvertFrom-Json
$baselineAmount = $baseline.by_field.amount
$baselineTime = $baseline.by_field.time
$baselinePayment = $baseline.by_field.payment_method_field
$baselineRecipient = $baseline.by_field.recipient_field

Write-Host ""
Write-Host "same_r3_val_baseline"
[pscustomobject]@{
    Amount = Get-ExactDisplay $baselineAmount
    Time = Get-ExactDisplay $baselineTime
    Payment = Get-ExactDisplay $baselinePayment
    Recipient = Get-ExactDisplay $baselineRecipient
    RecipientOov = ("{0}/{1}={2:P2}" -f $baselineRecipient.oov_reference_records, $baselineRecipient.records, $baselineRecipient.oov_reference_rate)
    Providers = ($baseline.providers -join ", ")
} | Format-List

if ($baseline.providers -notcontains "CUDAExecutionProvider") {
    throw "Baseline evaluation did not use CUDAExecutionProvider; fix the 4090 runtime before starting training."
}

if ([double]$baselineAmount.raw_exact_match -lt $AmountFloor -or
    [double]$baselineTime.raw_exact_match -lt $TimeFloor -or
    [double]$baselinePayment.raw_exact_match -lt $PaymentFloor) {
    throw "The r2 baseline does not satisfy the requested floors on this exact r3 val split. Recalibrate metric/split before training."
}
if ([double]$baselineRecipient.oov_reference_rate -gt 0.10) {
    Write-Warning "Recipient val OOV alone makes 90% strict exact impossible without charset/data changes. The run remains diagnostic only."
}

# Freeze every non-recipient v12 parameter and disable receipt-level
# oversampling. This preserves financial fields while the high-resolution
# recipient branch learns; strict warm-start compatibility is checked by the
# trainer before the optimiser is created.
$trainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $records,
    "--dataset-root", $labelsRoot,
    "--output", $output,
    "--device", "cuda:0",
    "--architecture", "v12",
    "--image-height", "80",
    "--image-width", "512",
    "--recipient-input-height", "128",
    "--recipient-input-width", "1024",
    "--recipient-branch-channels", "24",
    "--base-channels", "32",
    "--numeric-hidden-size", "96",
    "--payment-hidden-size", "128",
    "--recipient-hidden-size", "256",
    "--recipient-value-left-trim", "0.30",
    "--epochs", "$Epochs",
    "--batch-size", "12",
    "--learning-rate", "$LearningRate",
    "--payment-loss-weight", "1.0",
    "--recipient-loss-weight", "$RecipientLossWeight",
    "--recipient-sampling-weight", "1.0",
    "--recipient-rare-character-max-support", "0",
    "--recipient-long-text-min-length", "0",
    "--recipient-low-confidence-threshold", "0.98",
    "--recipient-low-confidence-loss-weight", "0.35",
    "--recipient-confidence-curriculum-epochs", "10",
    "--recipient-train-augmentation", "light_v1",
    "--recipient-only-fine-tune",
    "--checkpoint-selection", "recipient_priority",
    "--checkpoint-min-amount-candidate-exact", "$AmountFloor",
    "--checkpoint-min-time-candidate-exact", "$TimeFloor",
    "--checkpoint-min-payment-candidate-exact", "$PaymentFloor",
    "--init-checkpoint", $seedCheckpoint,
    "--ctc-loss-weight", "0.75",
    "--structured-loss-weight", "1.0",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--seed", "42",
    "--num-workers", "$NumWorkers",
    "--prefetch-factor", "$PrefetchFactor",
    "--cuda-tf32",
    "--cudnn-benchmark"
)

$exitCode = 0
try {
    & python @trainArgs
    $exitCode = $LASTEXITCODE
}
finally {
    $summaryPath = Join-Path $output "training_summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $last = @($summary.records | Select-Object -Last 1)[0]
        $best = @($summary.records | Where-Object { $_.epoch -eq $summary.best_checkpoint_epoch })[0]
        $eligible = @($summary.records | Where-Object checkpoint_selection_eligible).Count
        $recipientOov = $summary.recipient_oov_by_split.val

        Write-Host ""
        Write-Host "guarded_4090_recipient_only final_summary"
        [pscustomobject]@{
            BestEpoch = $summary.best_checkpoint_epoch
            EligibleEpochs = $eligible
            TargetRecipientStrictExact = "90.00%"
            BaselineRecipient = Get-ExactDisplay $baselineRecipient
            BestAmount = Get-TrainingExactDisplay $best "amount"
            BestTime = Get-TrainingExactDisplay $best "time"
            BestPayment = Get-TrainingExactDisplay $best "payment_method_field"
            BestRecipient = Get-TrainingExactDisplay $best "recipient_field"
            LastAmount = Get-TrainingExactDisplay $last "amount"
            LastTime = Get-TrainingExactDisplay $last "time"
            LastPayment = Get-TrainingExactDisplay $last "payment_method_field"
            LastRecipient = Get-TrainingExactDisplay $last "recipient_field"
            RecipientValOov = ("{0}/{1}" -f $recipientOov.oov_records, $recipientOov.records)
            AmountFloor = $AmountFloor
            TimeFloor = $TimeFloor
            PaymentFloor = $PaymentFloor
            FineTune = $summary.fine_tune_policy.mode
            Runtime = ("{0}; workers={1}; TF32={2}; cuDNN-benchmark={3}" -f $summary.training_runtime.cuda_device_name, $summary.training_runtime.num_workers, $summary.training_runtime.cuda_tf32_requested, $summary.training_runtime.cudnn_benchmark_requested)
            LastEligible = $last.checkpoint_selection_eligible
            LastFailures = ($last.checkpoint_selection_protection_failures -join "; ")
        } | Format-List
        $summary.records | Select-Object -Last 10 `
            epoch, checkpoint_selection_eligible, checkpoint_selection_protection_failures, `
            @{ n = "amount"; e = { $_.val_candidate_text_by_field.amount.exact_match } }, `
            @{ n = "time"; e = { $_.val_candidate_text_by_field.time.exact_match } }, `
            @{ n = "payment"; e = { $_.val_candidate_text_by_field.payment_method_field.exact_match } }, `
            @{ n = "recipient"; e = { $_.val_candidate_text_by_field.recipient_field.exact_match } } |
            Format-Table -AutoSize
    }
    else {
        Write-Host "guarded_4090_recipient_only final_summary=unavailable (training did not create training_summary.json)"
    }
}

if ($exitCode -ne 0) {
    exit $exitCode
}
