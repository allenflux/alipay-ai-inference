[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory,
    [ValidateRange(1, 100)]
    [int]$Top = 15,
    [ValidateRange(0, 10)]
    [int]$ExamplesPerOperation = 2,
    [ValidateRange(0, 100)]
    [int]$RepresentativeLimit = 12,
    [string]$Report
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "recipient-error-forensics.py"

function Get-RecipientExact([string]$Directory) {
    $summaryPath = Join-Path $Directory "onnx-test-gpu\summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        return $null
    }
    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $recipient = $summary.by_field.recipient_field.raw_exact_match
        if ($null -eq $recipient) {
            return $null
        }
        return [double]$recipient
    }
    catch {
        return $null
    }
}

if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $latest = Get-ChildItem -LiteralPath $TeacherRoot -Directory -Filter "unified-run-v13-status-cjk-ocr-wide1536-*" |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName "manifest-v13\unified_fields.jsonl") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "onnx-test-gpu\comparisons.jsonl") -PathType Leaf) -and
            ($null -ne (Get-RecipientExact $_.FullName)) -and
            ((Get-RecipientExact $_.FullName) -lt 0.90)
        } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No completed v13 test comparison with recipient exact below 90% was found."
    }
    $RunDirectory = $latest.FullName
}

$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$records = Join-Path $run "manifest-v13\unified_fields.jsonl"
$comparisons = Join-Path $run "onnx-test-gpu\comparisons.jsonl"
$summaryPath = Join-Path $run "onnx-test-gpu\summary.json"
$reportPath = if ([string]::IsNullOrWhiteSpace($Report)) {
    Join-Path $run "recipient-test-error-forensics.json"
}
elseif ([IO.Path]::IsPathRooted($Report)) {
    $Report
}
else {
    Join-Path $run $Report
}

foreach ($required in @(
    @{ Name = "CUDA Python"; Path = $pythonExe },
    @{ Name = "v13 manifest"; Path = $records },
    @{ Name = "v13 test comparisons"; Path = $comparisons },
    @{ Name = "v13 test summary"; Path = $summaryPath },
    @{ Name = "recipient error forensics script"; Path = $scriptPath }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType Leaf)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $reportPath) {
    throw "Refusing to overwrite recipient error forensics: $reportPath"
}

$recipientExact = Get-RecipientExact $run
Write-Host "recipient_error_forensics_start"
Write-Host "  run=$run"
Write-Host "  recipient-test-exact=$recipientExact"
Write-Host "  comparisons=$comparisons"
Write-Host "  records=$records"
Write-Host "  mode=read-only; no inference, training, crop, checkpoint, or manifest changes"

& $pythonExe $scriptPath `
    --comparisons $comparisons `
    --manifest $records `
    --split test `
    --top $Top `
    --examples-per-operation $ExamplesPerOperation `
    --representative-limit $RepresentativeLimit `
    --output $reportPath
if ($LASTEXITCODE -ne 0) {
    throw "Recipient error forensics failed with exit code $LASTEXITCODE"
}

Write-Host "recipient_error_forensics_complete"
Write-Host "  report=$reportPath"
