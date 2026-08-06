[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(0.000001, 1.0)]
    [double]$Target = 0.70,
    [string]$Report
)

$ErrorActionPreference = "Stop"

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
    Join-Path $run "recipient-lexicon-audit.json"
} elseif ([System.IO.Path]::IsPathRooted($Report)) {
    $Report
} else {
    Join-Path $run $Report
}
$scriptPath = Join-Path $PSScriptRoot "recipient-lexicon-audit.py"

foreach ($required in @(
    @{ Name = "run directory"; Path = $run },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "comparisons"; Path = $comparisons },
    @{ Name = "recipient lexicon audit script"; Path = $scriptPath }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $reportPath) {
    throw "Refusing to overwrite recipient lexicon audit: $reportPath"
}

Write-Host "receipt_ocr_recipient_lexicon_audit_4090"
Write-Host "  run=$run"
Write-Host "  comparisons=$comparisons"
Write-Host "  records=$records"
Write-Host "  target=$Target"
Write-Host "  mode=read-only; no model, crop, checkpoint, or manifest is modified"

& python $scriptPath `
    --comparisons $comparisons `
    --manifest $records `
    --target $Target `
    --output $reportPath
if ($LASTEXITCODE -ne 0) {
    throw "Recipient lexicon audit failed with exit code $LASTEXITCODE"
}

Write-Host "recipient lexicon audit=$reportPath"
