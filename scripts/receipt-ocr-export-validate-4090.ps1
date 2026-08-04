[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$OutputDirectory,
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [ValidateRange(0.0, 1.0)]
    [double]$RecipientFloor = 0.90
)

$ErrorActionPreference = "Stop"

# Evaluate the delivery artifact on exactly the guarded r3 validation split.
# This never edits the checkpoint and refuses to reuse an existing output so a
# failed prior validation cannot accidentally be reported as a new result.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $TeacherRoot ("unified-export-v12-r3-4090-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
$candidateModel = Join-Path $OutputDirectory "best.onnx"
$candidateValOutput = Join-Path $OutputDirectory "onnx-val"

foreach ($required in @(
    @{ Name = "checkpoint"; Path = $Checkpoint },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse export validation output: $OutputDirectory"
}

function Get-ExactDisplay([object]$Metric) {
    if ($null -eq $Metric) {
        return "n/a"
    }
    return ("{0}/{1}={2:P2}" -f $Metric.raw_exact_matches, $Metric.records, $Metric.raw_exact_match)
}

function Read-GuardedJson([string]$Path) {
    # Python accepts historic NaN/Infinity tokens whereas Windows PowerShell's
    # ConvertFrom-Json rejects them.  Normalise to standards-compliant JSON.
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

Write-Host "guarded_4090_export_validation preflight"
Write-Host "  checkpoint=$Checkpoint"
Write-Host "  records=$records"
Write-Host "  output=$OutputDirectory"
Write-Host ("  floors: amount={0:P2}, time={1:P2}, payment={2:P2}, recipient={3:P2}" -f $AmountFloor, $TimeFloor, $PaymentFloor, $RecipientFloor)

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null

& python -m transfer_receipt_ai.ocr_unified export `
    --checkpoint $Checkpoint `
    --output $candidateModel
if ($LASTEXITCODE -ne 0) {
    throw "ONNX export failed with exit code $LASTEXITCODE"
}

& python -m transfer_receipt_ai.ocr_unified evaluate `
    --model $candidateModel `
    --records $records `
    --dataset-root $labelsRoot `
    --split val `
    --output $candidateValOutput `
    --device cuda:0 `
    --min-amount-exact-match "$AmountFloor" `
    --min-time-exact-match "$TimeFloor" `
    --min-payment-exact-match "$PaymentFloor" `
    --min-recipient-exact-match "$RecipientFloor"
$evaluationExitCode = $LASTEXITCODE
$summaryPath = Join-Path $candidateValOutput "summary.json"
if (-not (Test-Path -LiteralPath $summaryPath)) {
    throw "ONNX validation did not write summary.json: $summaryPath"
}

$summary = Read-GuardedJson $summaryPath
$amount = $summary.by_field.amount
$time = $summary.by_field.time
$payment = $summary.by_field.payment_method_field
$recipient = $summary.by_field.recipient_field
$cudaProviderPresent = $summary.providers -contains "CUDAExecutionProvider"

Write-Host ""
Write-Host "guarded_4090_export_validation final_summary"
[pscustomobject]@{
    ExportedModel = $candidateModel
    Amount = Get-ExactDisplay $amount
    Time = Get-ExactDisplay $time
    Payment = Get-ExactDisplay $payment
    Recipient = Get-ExactDisplay $recipient
    Recipient90Reached = ([double]$recipient.raw_exact_match -ge $RecipientFloor)
    RecipientOov = ("{0}/{1}={2:P2}" -f $recipient.oov_reference_records, $recipient.records, $recipient.oov_reference_rate)
    Providers = ($summary.providers -join ", ")
    CUDAProviderPresent = $cudaProviderPresent
    Accepted = $summary.acceptance.passed
    Failures = ($summary.acceptance.failures -join "; ")
    MeanInferenceMs = $summary.receipt_latency_ms.mean
    LabelTruthWarning = "Paddle teacher labels; not independent production truth"
} | Format-List

if (-not $cudaProviderPresent) {
    Write-Host "Exported ONNX validation did not use CUDAExecutionProvider."
    exit 3
}
if ($evaluationExitCode -ne 0) {
    exit $evaluationExitCode
}
