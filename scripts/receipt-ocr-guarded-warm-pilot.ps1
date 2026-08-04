[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 80)]
    [int]$Epochs = 20,
    [string]$RunName,
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

# Keep this short pilot auditable: it reads the r3 manifest, starts only from
# the independently retained r2 best.pt, and always writes a fresh output
# directory. The floors are the approved pilot guardrails, not a new baseline.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-recipient-curriculum"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$seedCheckpoint = Join-Path $TeacherRoot "unified-run-v12-120k-r2-recipient-priority\best.pt"
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unified-run-v12-r3-warm-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
$output = Join-Path $TeacherRoot $RunName

foreach ($required in @(
    @{ Name = "records"; Path = $records },
    @{ Name = "dataset root"; Path = $labelsRoot },
    @{ Name = "baseline best.pt"; Path = $seedCheckpoint }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Refusing to reuse training output: $output"
}

Write-Host "guarded_warm_pilot preflight"
Write-Host "  seed=$seedCheckpoint"
Write-Host "  records=$records"
Write-Host "  output=$output"
Write-Host ("  floors: amount={0:P2}, time={1:P2}, payment={2:P2}" -f $AmountFloor, $TimeFloor, $PaymentFloor)
if ($CheckOnly) {
    Write-Host "guarded_warm_pilot preflight=passed"
    exit 0
}

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
    "--learning-rate", "0.0004",
    "--payment-loss-weight", "1.0",
    "--recipient-loss-weight", "4.0",
    "--recipient-sampling-weight", "3.0",
    "--recipient-rare-character-max-support", "3",
    "--recipient-rare-character-sampling-weight", "4.0",
    "--recipient-long-text-min-length", "12",
    "--recipient-long-text-sampling-weight", "4.0",
    "--recipient-low-confidence-threshold", "0.98",
    "--recipient-low-confidence-loss-weight", "0.35",
    "--recipient-confidence-curriculum-epochs", "10",
    "--recipient-train-augmentation", "light_v1",
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
    "--num-workers", "0"
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

        function Format-Guard([object]$record, [string]$field) {
            if ($null -eq $record -or $null -eq $record.checkpoint_protection) {
                return "n/a"
            }
            $metric = $record.checkpoint_protection.candidate_exact.PSObject.Properties[$field].Value
            if ($null -eq $metric) {
                return "n/a"
            }
            return ("{0}/{1}={2:P2}" -f $metric.exact_matches, $metric.records, $metric.exact_match)
        }

        Write-Host ""
        Write-Host "guarded_warm_pilot final_summary"
        [pscustomobject]@{
            BestEpoch = $summary.best_checkpoint_epoch
            EligibleEpochs = $eligible
            AmountFloor = $AmountFloor
            TimeFloor = $TimeFloor
            PaymentFloor = $PaymentFloor
            AmountFormatGate = $summary.config.amount_format_min_confidence
            BestAmount = Format-Guard $best "amount"
            BestTime = Format-Guard $best "time"
            BestPayment = Format-Guard $best "payment_method_field"
            LastAmount = Format-Guard $last "amount"
            LastTime = Format-Guard $last "time"
            LastPayment = Format-Guard $last "payment_method_field"
            LastEligible = $last.checkpoint_selection_eligible
            LastFailures = ($last.checkpoint_selection_protection_failures -join "; ")
        } | Format-List
        $summary.records | Select-Object -Last 10 `
            epoch, checkpoint_selection_eligible, checkpoint_selection_protection_failures, `
            @{ n = "amount"; e = { $_.checkpoint_protection.candidate_exact.amount.exact_match } }, `
            @{ n = "time"; e = { $_.checkpoint_protection.candidate_exact.time.exact_match } }, `
            @{ n = "payment"; e = { $_.checkpoint_protection.candidate_exact.payment_method_field.exact_match } } |
            Format-Table -AutoSize
    }
    else {
        Write-Host "guarded_warm_pilot final_summary=unavailable (training did not create training_summary.json)"
    }
}

if ($exitCode -ne 0) {
    exit $exitCode
}
