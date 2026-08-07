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
$manifestRoot = Join-Path $run "manifest-v13"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$qualityAudit = Join-Path $manifestRoot "recipient_quality_audit.jsonl"
$comparisons = Join-Path $run "onnx-test-gpu\comparisons.jsonl"
$dataAuditScript = Join-Path $PSScriptRoot "recipient-data-audit.py"
$sliceScript = Join-Path $PSScriptRoot "recipient-slice-report.py"
$dataReport = Join-Path $run "recipient-test-data-audit.json"
$sliceReport = Join-Path $run "recipient-test-slice-report.json"

foreach ($required in @(
    @{ Name = "CUDA Python"; Path = $pythonExe },
    @{ Name = "v13 manifest"; Path = $records },
    @{ Name = "v13 test comparisons"; Path = $comparisons },
    @{ Name = "recipient data audit"; Path = $dataAuditScript },
    @{ Name = "recipient slice audit"; Path = $sliceScript }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType Leaf)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
foreach ($output in @($dataReport, $sliceReport)) {
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to overwrite recipient diagnostic: $output"
    }
}

Write-Host "recipient_test_diagnostic"
Write-Host "  run=$run"
Write-Host "  comparisons=$comparisons"
Write-Host "  records=$records"
Write-Host "  mode=read-only; test split is not used for training"

$dataArgs = @(
    $dataAuditScript,
    "--comparisons", $comparisons,
    "--manifest", $records,
    "--target", "0.90",
    "--output", $dataReport
)
if (Test-Path -LiteralPath $qualityAudit -PathType Leaf) {
    $dataArgs += @("--quality-audit", $qualityAudit)
}
& $pythonExe @dataArgs
if ($LASTEXITCODE -ne 0) {
    throw "Recipient data audit failed with exit code $LASTEXITCODE"
}

& $pythonExe $sliceScript `
    --comparisons $comparisons `
    --manifest $records `
    --skip-geometry `
    --output $sliceReport
if ($LASTEXITCODE -ne 0) {
    throw "Recipient slice audit failed with exit code $LASTEXITCODE"
}

Write-Host "recipient_test_diagnostic_complete"
Write-Host "  data-audit=$dataReport"
Write-Host "  slice-audit=$sliceReport"
