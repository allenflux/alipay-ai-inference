[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $latest = Get-ChildItem -LiteralPath $TeacherRoot -Directory -Filter "unified-run-v13-status-cjk-ocr-wide1536-*" |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "onnx-test-gpu\comparisons.jsonl") } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No completed v13 test comparison set was found."
    }
    $RunDirectory = $latest.FullName
}

$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$records = Join-Path $run "manifest-v13\unified_fields.jsonl"
$comparisons = Join-Path $run "onnx-test-gpu\comparisons.jsonl"
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$sliceScript = Join-Path $PSScriptRoot "recipient-slice-report.py"
$report = Join-Path $run "recipient-test-geometry-report.json"

foreach ($required in @($pythonExe, $records, $comparisons, $sliceScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing recipient geometry diagnostic input: $required"
    }
}
if (-not (Test-Path -LiteralPath $labelsRoot -PathType Container)) {
    throw "Missing recipient crop root: $labelsRoot"
}
if (Test-Path -LiteralPath $report) {
    throw "Refusing to overwrite recipient geometry diagnostic: $report"
}

Write-Host "recipient_test_geometry_start"
Write-Host "  run=$run"
Write-Host "  mode=read-only image evidence"
& $pythonExe $sliceScript `
    --comparisons $comparisons `
    --manifest $records `
    --dataset-root $labelsRoot `
    --left-trim 0.30 `
    --output $report
if ($LASTEXITCODE -ne 0) {
    throw "Recipient geometry diagnostic failed with exit code $LASTEXITCODE"
}
Write-Host "recipient_test_geometry_complete"
Write-Host "  report=$report"
