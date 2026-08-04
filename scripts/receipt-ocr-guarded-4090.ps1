[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 80)]
    [int]$Epochs = 80,
    [string]$RunName,
    # Optional audited recipient-only checkpoint for a controlled follow-up
    # pilot.  Leaving this blank preserves the known r2 baseline default.
    [string]$SeedCheckpoint,
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [ValidateRange(0.0, 16.0)]
    [double]$RecipientLossWeight = 4.0,
    [ValidateRange(0, 1000000)]
    [int]$RecipientTailRareCharacterMaxSupport = 0,
    [ValidateRange(1.0, 16.0)]
    [double]$RecipientTailRareCharacterLossWeight = 1.0,
    [ValidateRange(0, 1000000)]
    [int]$RecipientTailLongTextMinLength = 0,
    [ValidateRange(1.0, 16.0)]
    [double]$RecipientTailLongTextLossWeight = 1.0,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.0001,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(1, 80)]
    [int]$ValidationEvery = 5,
    [switch]$DiagnosticOnly,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

# These are the actual r3/4090 assets, not the old generic r3 directory names.
# The baseline is evaluated again on exactly this manifest before any training
# begins, so the historical floors cannot silently cross dataset splits.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$seedCheckpoint = $SeedCheckpoint
if ([string]::IsNullOrWhiteSpace($seedCheckpoint)) {
    $seedCheckpoint = Join-Path $TeacherRoot "unified-run-v12-120k-r2-recipient-priority\best.pt"
}
$baselineModel = Join-Path $TeacherRoot "models\receipt_unified_field_reader_v12_120k_r2_recipient24_h256.onnx"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unified-run-v12-r3-4090-recipient-only-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
$output = Join-Path $TeacherRoot $RunName
$baselineOutput = Join-Path $TeacherRoot ("unified-eval-v12-r3-4090-warm-baseline-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$candidateModel = Join-Path $output "best.onnx"
$candidateValOutput = Join-Path $output "onnx-val"

foreach ($required in @(
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot },
    @{ Name = "recipient-only seed checkpoint"; Path = $seedCheckpoint },
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

function Read-GuardedJson([string]$Path) {
    # Python accepts the NaN/Infinity tokens emitted by Python's JSON encoder,
    # whereas Windows PowerShell's ConvertFrom-Json rejects them. Normalize
    # those values to null before PowerShell reads a training/evaluation summary.
    $normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
    if (-not (Test-Path -LiteralPath $normalizer)) {
        throw "Missing JSON summary normalizer: $normalizer"
    }
    $normalizedJson = (& python $normalizer $Path) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to normalize JSON summary with Python: $Path (exit code $LASTEXITCODE)"
    }
    try {
        return ($normalizedJson | ConvertFrom-Json)
    }
    catch {
        throw "Unable to parse normalized JSON summary: $Path. $($_.Exception.Message)"
    }
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
Write-Host ("  recipe: recipient-only, lr={0}, workers={1}, prefetch={2}, validate-every={3}, TF32=on, cuDNN-benchmark=on" -f $LearningRate, $NumWorkers, $PrefetchFactor, $ValidationEvery)
Write-Host ("  recipient-tail CTC: rare-support<={0} x{1}; long-length>={2} x{3}; max(), no receipt resampling" -f $RecipientTailRareCharacterMaxSupport, $RecipientTailRareCharacterLossWeight, $RecipientTailLongTextMinLength, $RecipientTailLongTextLossWeight)
Write-Host ("  recipient target: 90.00% strict exact ({0})" -f $(if ($DiagnosticOnly) { "diagnostic only" } else { "required" }))
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
$baseline = Read-GuardedJson (Join-Path $baselineOutput "summary.json")
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
    "--recipient-tail-rare-character-max-support", "$RecipientTailRareCharacterMaxSupport",
    "--recipient-tail-rare-character-loss-weight", "$RecipientTailRareCharacterLossWeight",
    "--recipient-tail-long-text-min-length", "$RecipientTailLongTextMinLength",
    "--recipient-tail-long-text-loss-weight", "$RecipientTailLongTextLossWeight",
    "--recipient-train-augmentation", "light_v1",
    "--recipient-only-fine-tune",
    "--checkpoint-selection", "recipient_priority",
    "--checkpoint-min-amount-candidate-exact", "$AmountFloor",
    "--checkpoint-min-time-candidate-exact", "$TimeFloor",
    "--checkpoint-min-payment-candidate-exact", "$PaymentFloor",
    "--init-checkpoint", $seedCheckpoint,
    "--init-checkpoint-mode", "recipient_only_expansion",
    "--ctc-loss-weight", "0.75",
    "--structured-loss-weight", "1.0",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--seed", "42",
    "--num-workers", "$NumWorkers",
    "--prefetch-factor", "$PrefetchFactor",
    "--validation-every", "$ValidationEvery",
    "--cuda-tf32",
    "--cudnn-benchmark"
)

# A successful full run must prove that the exported delivery artifact, not
# just its in-memory PyTorch checkpoint, clears the same r3 guardrails.
# The short diagnostic deliberately skips this second full ONNX pass so it
# can establish convergence speed before committing to the long run.
if (-not $DiagnosticOnly) {
    $trainArgs += @(
        "--onnx-output", $candidateModel
    )
}

$exitCode = 0
try {
    & python @trainArgs
    $exitCode = $LASTEXITCODE
}
finally {
    $summaryPath = Join-Path $output "training_summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        $summary = Read-GuardedJson $summaryPath
        $last = @($summary.records | Select-Object -Last 1)[0]
        $best = @($summary.records | Where-Object { $_.epoch -eq $summary.best_checkpoint_epoch })[0]
        $validatedRecords = @($summary.records | Where-Object { $_.validation_performed -ne $false })
        $eligible = @($summary.records | Where-Object checkpoint_selection_eligible).Count
        $recipientOov = $summary.recipient_oov_by_split.val
        $tailPolicy = $summary.recipient_tail_loss_policy
        $tailMode = if ($null -eq $tailPolicy) { "legacy/absent" } else { $tailPolicy.mode }
        $tailRecipientRecords = if ($null -eq $tailPolicy) { "n/a" } else { $tailPolicy.recipient_train_records }
        $tailRareHits = if ($null -eq $tailPolicy) { "n/a" } else { $tailPolicy.rare_character_train_records }
        $tailLongHits = if ($null -eq $tailPolicy) { "n/a" } else { $tailPolicy.long_text_train_records }
        $tailCombinedHits = if ($null -eq $tailPolicy) { "n/a" } else { $tailPolicy.combined_boost_train_records }
        $bestRecipientMetric = $null
        if ($null -ne $best -and $null -ne $best.val_candidate_text_by_field) {
            $bestRecipientMetric = $best.val_candidate_text_by_field.recipient_field
        }
        $trainingRecipientTargetReached = ($null -ne $bestRecipientMetric -and [double]$bestRecipientMetric.exact_match -ge 0.90)

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
            TrainingRecipient90Reached = $trainingRecipientTargetReached
            LastAmount = Get-TrainingExactDisplay $last "amount"
            LastTime = Get-TrainingExactDisplay $last "time"
            LastPayment = Get-TrainingExactDisplay $last "payment_method_field"
            LastRecipient = Get-TrainingExactDisplay $last "recipient_field"
            RecipientValOov = ("{0}/{1}" -f $recipientOov.oov_records, $recipientOov.records)
            AmountFloor = $AmountFloor
            TimeFloor = $TimeFloor
            PaymentFloor = $PaymentFloor
            FineTune = $summary.fine_tune_policy.mode
            TrainForward = $summary.fine_tune_policy.training_forward
            RecipientTrainRecords = $summary.fine_tune_policy.recipient_train_records
            RecipientTailMode = $tailMode
            RecipientTailTrainRecords = $tailRecipientRecords
            RecipientTailRareHits = $tailRareHits
            RecipientTailLongHits = $tailLongHits
            RecipientTailCombinedBoostHits = $tailCombinedHits
            Initialization = $summary.initialization.mode
            FinancialLabelPolicy = $summary.initialization.financial_label_policy.mode
            Runtime = ("{0}; workers={1}; TF32={2}; cuDNN-benchmark={3}" -f $summary.training_runtime.cuda_device_name, $summary.training_runtime.num_workers, $summary.training_runtime.cuda_tf32_requested, $summary.training_runtime.cudnn_benchmark_requested)
            LastEligible = $last.checkpoint_selection_eligible
            LastFailures = ($last.checkpoint_selection_protection_failures -join "; ")
        } | Format-List
        $validatedRecords | Select-Object -Last 10 `
            epoch, train_seconds, validation_seconds, epoch_seconds, checkpoint_selection_eligible, checkpoint_selection_protection_failures, `
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

if ($DiagnosticOnly) {
    Write-Host "final ONNX validation skipped: diagnostic-only run"
}
elseif ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "guarded_4090_recipient_only final_onnx_validation"
    if (-not (Test-Path -LiteralPath $candidateModel)) {
        Write-Host "final ONNX validation=unavailable (best.onnx was not exported)"
        $exitCode = 3
    }
    elseif (Test-Path -LiteralPath $candidateValOutput) {
        Write-Host "final ONNX validation=unavailable (refusing to reuse output: $candidateValOutput)"
        $exitCode = 3
    }
    else {
        $candidateArgs = @(
            "-m", "transfer_receipt_ai.ocr_unified", "evaluate",
            "--model", $candidateModel,
            "--records", $records,
            "--dataset-root", $labelsRoot,
            "--split", "val",
            "--output", $candidateValOutput,
            "--device", "cuda:0",
            "--min-amount-exact-match", "$AmountFloor",
            "--min-time-exact-match", "$TimeFloor",
            "--min-payment-exact-match", "$PaymentFloor",
            "--min-recipient-exact-match", "0.90"
        )
        & python @candidateArgs
        $candidateExitCode = $LASTEXITCODE
        $candidateSummaryPath = Join-Path $candidateValOutput "summary.json"
        if (Test-Path -LiteralPath $candidateSummaryPath) {
            $candidate = Read-GuardedJson $candidateSummaryPath
            $candidateAmount = $candidate.by_field.amount
            $candidateTime = $candidate.by_field.time
            $candidatePayment = $candidate.by_field.payment_method_field
            $candidateRecipient = $candidate.by_field.recipient_field
            [pscustomobject]@{
                ExportedModel = $candidateModel
                Amount = Get-ExactDisplay $candidateAmount
                Time = Get-ExactDisplay $candidateTime
                Payment = Get-ExactDisplay $candidatePayment
                Recipient = Get-ExactDisplay $candidateRecipient
                Recipient90Reached = ([double]$candidateRecipient.raw_exact_match -ge 0.90)
                RecipientOov = ("{0}/{1}={2:P2}" -f $candidateRecipient.oov_reference_records, $candidateRecipient.records, $candidateRecipient.oov_reference_rate)
                Providers = ($candidate.providers -join ", ")
                Accepted = $candidate.acceptance.passed
                Failures = ($candidate.acceptance.failures -join "; ")
                MeanInferenceMs = $candidate.receipt_latency_ms.mean
            } | Format-List
            if ($candidate.providers -notcontains "CUDAExecutionProvider") {
                Write-Host "Exported ONNX validation did not use CUDAExecutionProvider."
                $candidateExitCode = 3
            }
        }
        else {
            Write-Host "final ONNX validation=unavailable (summary.json was not written)"
            $candidateExitCode = 3
        }
        if ($candidateExitCode -ne 0) {
            $exitCode = $candidateExitCode
        }
    }
}

if ($exitCode -ne 0) {
    exit $exitCode
}
