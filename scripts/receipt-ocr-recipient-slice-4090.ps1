[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(0.0, 0.999999)]
    [double]$LeftTrim = 0.30,
    [string]$Report,
    [switch]$SkipGeometry
)

$ErrorActionPreference = "Stop"

# Keep the manually typed RDP command short and free of filenames containing
# underscores.  This wrapper is diagnostic only: it reads an existing final
# ONNX evaluation and the immutable r3 manifest, then writes one new report.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
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
    Join-Path $run "recipient-slice-report.json"
} elseif ([System.IO.Path]::IsPathRooted($Report)) {
    $Report
} else {
    Join-Path $run $Report
}
$scriptPath = Join-Path $PSScriptRoot "recipient-slice-report.py"

foreach ($required in @(
    @{ Name = "run directory"; Path = $run },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot },
    @{ Name = "final ONNX comparisons"; Path = $comparisons },
    @{ Name = "recipient slice script"; Path = $scriptPath }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $reportPath) {
    throw "Refusing to overwrite recipient slice report: $reportPath"
}

Write-Host "receipt_ocr_recipient_slice_4090"
Write-Host "  run=$run"
Write-Host "  comparisons=$comparisons"
Write-Host "  records=$records"
Write-Host "  trim=$LeftTrim"
Write-Host "  geometry=$(if ($SkipGeometry) { 'skipped' } else { 'image-only current trim audit' })"
Write-Host "  mode=read-only; checkpoint and manifest are not modified"

$arguments = @(
    $scriptPath,
    "--comparisons", $comparisons,
    "--manifest", $records,
    "--dataset-root", $labelsRoot,
    "--left-trim", "$LeftTrim",
    "--output", $reportPath
)
if ($SkipGeometry) {
    $arguments += "--skip-geometry"
}

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Recipient slice report failed with exit code $LASTEXITCODE"
}

Write-Host "recipient slice report=$reportPath"
