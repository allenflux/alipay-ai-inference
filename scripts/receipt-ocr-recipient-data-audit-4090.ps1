[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(0.000001, 1.0)]
    [double]$Target = 0.90,
    [string]$Report
)

$ErrorActionPreference = "Stop"

# This wrapper reads the final CUDA ONNX comparisons and the exact r3 unified
# manifest.  It never invokes a model or changes the run/checkpoint.
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$qualityAudit = Join-Path $manifestRoot "recipient_quality_audit.jsonl"
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$standardComparisons = Join-Path $run "onnx-val\comparisons.jsonl"
$beamComparisons = Join-Path $run "validation\comparisons.jsonl"
$comparisons = if (Test-Path -LiteralPath $standardComparisons) {
    $standardComparisons
} elseif (Test-Path -LiteralPath $beamComparisons) {
    $beamComparisons
} else {
    $standardComparisons
}
$reportPath = if ([string]::IsNullOrWhiteSpace($Report)) {
    Join-Path $run "recipient-data-audit.json"
} elseif ([System.IO.Path]::IsPathRooted($Report)) {
    $Report
} else {
    Join-Path $run $Report
}
$scriptPath = Join-Path $PSScriptRoot "recipient-data-audit.py"

foreach ($required in @(
    @{ Name = "run directory"; Path = $run },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "final ONNX comparisons"; Path = $comparisons },
    @{ Name = "recipient data audit script"; Path = $scriptPath }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $reportPath) {
    throw "Refusing to overwrite recipient data audit: $reportPath"
}

Write-Host "receipt_ocr_recipient_data_audit_4090"
Write-Host "  run=$run"
Write-Host "  comparisons=$comparisons"
Write-Host "  records=$records"
Write-Host "  target=$Target"
Write-Host "  mode=read-only; no model, crop, checkpoint, or manifest is modified"

$arguments = @(
    $scriptPath,
    "--comparisons", $comparisons,
    "--manifest", $records,
    "--target", "$Target",
    "--output", $reportPath
)
if (Test-Path -LiteralPath $qualityAudit) {
    $arguments += "--quality-audit"
    $arguments += $qualityAudit
    Write-Host "  quality-audit=$qualityAudit"
} else {
    Write-Host "  quality-audit=unavailable"
}

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Recipient data audit failed with exit code $LASTEXITCODE"
}

Write-Host "recipient data audit=$reportPath"
