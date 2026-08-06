[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory,
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [ValidateRange(0.0, 1.0)]
    [double]$RecipientFloor = 0.90,
    [ValidateRange(0, 1000000)]
    [int]$ProgressEvery = 250
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$records = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
$datasetRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"

if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $latest = Get-ChildItem -LiteralPath $TeacherRoot -Directory -Filter "unified-run-v12-r3-4090-paddle-fit-open-text-*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No Paddle-fit run directory found under $TeacherRoot"
    }
    $RunDirectory = $latest.FullName
}

$checkpoint = Join-Path $RunDirectory "best.pt"
$model = Join-Path $RunDirectory "best.onnx"
$evaluation = Join-Path $RunDirectory "onnx-val"
$trainingSummaryPath = Join-Path $RunDirectory "training_summary.json"
foreach ($required in @($pythonExe, $records, $datasetRoot, $checkpoint, $trainingSummaryPath, $normalizer)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing Paddle-fit validation dependency: $required"
    }
}
if (Test-Path -LiteralPath $evaluation) {
    throw "Refusing to reuse Paddle-fit ONNX validation output: $evaluation"
}

Write-Host "paddle_fit_export_validate_4090"
Write-Host "  run=$RunDirectory"
Write-Host "  checkpoint=$checkpoint"
Write-Host "  model=$model"
Write-Host ("  recipient-floor={0:P2}" -f $RecipientFloor)

$trainingSummary = ((& $pythonExe $normalizer $trainingSummaryPath) -join "`n") | ConvertFrom-Json
$bestTraining = @($trainingSummary.records | Where-Object { $_.epoch -eq $trainingSummary.best_checkpoint_epoch })[0]
$lastTraining = @($trainingSummary.records | Select-Object -Last 1)[0]
$bestRecipient = $bestTraining.val_candidate_text_by_field.recipient_field
$lastRecipient = $lastTraining.val_candidate_text_by_field.recipient_field
Write-Host "paddle_fit_training_provenance"
Write-Host ("  policy={0}; splits={1}" -f $trainingSummary.recipient_train_split_policy.mode, (($trainingSummary.recipient_train_split_policy.splits | ForEach-Object { $_ }) -join ","))
Write-Host ("  best-epoch={0}; best-recipient={1}/{2}={3:P2}" -f $trainingSummary.best_checkpoint_epoch, $bestRecipient.exact_matches, $bestRecipient.records, $bestRecipient.exact_match)
Write-Host ("  last-epoch={0}; last-recipient={1}/{2}={3:P2}" -f $lastTraining.epoch, $lastRecipient.exact_matches, $lastRecipient.records, $lastRecipient.exact_match)

Write-Host "paddle_fit_export"
& $pythonExe -m transfer_receipt_ai.ocr_unified export `
    --checkpoint $checkpoint `
    --output $model
if ($LASTEXITCODE -ne 0) {
    throw "ONNX export failed with exit code $LASTEXITCODE"
}

Write-Host "paddle_fit_onnx_validate"
& $pythonExe -m transfer_receipt_ai.ocr_unified evaluate `
    --model $model `
    --records $records `
    --dataset-root $datasetRoot `
    --split val `
    --output $evaluation `
    --device cuda `
    --min-amount-exact-match $AmountFloor `
    --min-time-exact-match $TimeFloor `
    --min-payment-exact-match $PaymentFloor `
    --min-recipient-exact-match $RecipientFloor `
    --progress-every $ProgressEvery
$exitCode = $LASTEXITCODE

$summaryPath = Join-Path $evaluation "summary.json"
if (Test-Path -LiteralPath $summaryPath) {
    $summary = ((& $pythonExe $normalizer $summaryPath) -join "`n") | ConvertFrom-Json
    Write-Host "paddle_fit_onnx_final_summary"
    foreach ($field in @("amount", "time", "payment_method_field", "recipient_field")) {
        $metric = $summary.by_field.PSObject.Properties[$field].Value
        Write-Host ("  {0}={1}/{2}={3:P2}" -f $field, $metric.raw_exact_matches, $metric.records, $metric.raw_exact_match)
    }
    Write-Host ("  providers={0}" -f (($summary.providers | ForEach-Object { $_ }) -join ","))
    Write-Host ("  accepted={0}; failures={1}" -f $summary.acceptance.passed, (($summary.acceptance.failures | ForEach-Object { $_ }) -join "; "))
    Write-Host ("  mean_inference_ms={0}" -f $summary.receipt_latency_ms.mean)
    Write-Host ("  output={0}" -f $evaluation)
}

if ($exitCode -ne 0) {
    exit $exitCode
}
