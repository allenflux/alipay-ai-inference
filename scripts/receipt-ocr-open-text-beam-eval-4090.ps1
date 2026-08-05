[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(2, 32)]
    [int]$BeamWidth = 8,
    [ValidateRange(2, 64)]
    [int]$TokenTopK = 20,
    [ValidateRange(1, 5)]
    [int]$NGramOrder = 3,
    [ValidateRange(0.0, 4.0)]
    [double]$NGramWeight = 0.35
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$checkpoint = Join-Path $RunDirectory "best.pt"
$records = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
$datasetRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$output = Join-Path $RunDirectory ("beam-eval-w{0}-o{1}-a{2}" -f $BeamWidth, $NGramOrder, $NGramWeight)
$model = Join-Path $output "best.onnx"
$evaluation = Join-Path $output "validation"

foreach ($required in @($pythonExe, $checkpoint, $records, $datasetRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing beam evaluation dependency: $required"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Refusing to reuse beam evaluation output: $output"
}

Write-Host "open_text_character_ngram_beam_tests"
& $pythonExe -m pytest -q tests/test_recipient_beam.py tests/test_ocr_unified_v12.py
if ($LASTEXITCODE -ne 0) {
    throw "Beam decoder tests failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Path $output | Out-Null
Write-Host "open_text_character_ngram_beam_export"
& $pythonExe -m transfer_receipt_ai.ocr_unified export `
    --checkpoint $checkpoint `
    --output $model
if ($LASTEXITCODE -ne 0) {
    throw "ONNX export failed with exit code $LASTEXITCODE"
}

Write-Host "open_text_character_ngram_beam_evaluate"
& $pythonExe -m transfer_receipt_ai.ocr_unified evaluate `
    --model $model `
    --records $records `
    --dataset-root $datasetRoot `
    --output $evaluation `
    --split val `
    --device cuda `
    --recipient-beam-width $BeamWidth `
    --recipient-beam-token-top-k $TokenTopK `
    --recipient-ngram-order $NGramOrder `
    --recipient-ngram-weight $NGramWeight
if ($LASTEXITCODE -ne 0) {
    throw "Beam evaluation failed with exit code $LASTEXITCODE"
}

$summaryPath = Join-Path $evaluation "summary.json"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
$summary = ((& $pythonExe $normalizer $summaryPath) -join "`n") | ConvertFrom-Json
Write-Host "open_text_character_ngram_beam_final_summary"
foreach ($field in @("amount", "time", "payment_method_field", "recipient_field")) {
    $metric = $summary.by_field.PSObject.Properties[$field].Value
    Write-Host ("  {0}={1}/{2}={3:P2}" -f $field, $metric.raw_exact_matches, $metric.records, $metric.raw_exact_match)
}
Write-Host ("  decoder={0}; beam={1}; top-k={2}; order={3}; weight={4}" -f $summary.recipient_decoder_policy.mode, $BeamWidth, $TokenTopK, $NGramOrder, $NGramWeight)
Write-Host ("  output={0}" -f $evaluation)
