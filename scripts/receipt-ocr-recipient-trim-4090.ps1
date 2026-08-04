[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateSet("val", "test")]
    [string]$Split = "val",
    [ValidateRange(0.0, 0.999999)]
    [double[]]$LeftTrims = @(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

# The hyphenated filename keeps the manually typed RDP command simple.  This
# is deliberately a separate diagnostic path: it only reads an already
# exported v12 ONNX plus the immutable r3 manifest and writes a fresh audit.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$model = Join-Path $run "best.onnx"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $run ("recipient-trim-audit-v12-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}

foreach ($required in @(
    @{ Name = "run directory"; Path = $run },
    @{ Name = "v12 ONNX"; Path = $model },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse recipient trim audit output: $OutputDirectory"
}
if ($LeftTrims.Count -eq 0) {
    throw "At least one left trim fraction is required"
}

Write-Host "receipt_ocr_recipient_trim_4090"
Write-Host "  model=$model"
Write-Host "  records=$records"
Write-Host "  split=$Split"
Write-Host ("  trims=" + ($LeftTrims -join ", "))
Write-Host "  output=$OutputDirectory"
Write-Host "  mode=diagnostic only; checkpoint, ONNX, labels, contract, manifest, and standard guard evaluation are not modified"

$arguments = @(
    "-m", "transfer_receipt_ai.ocr_unified", "audit-recipient",
    "--model", $model,
    "--records", $records,
    "--dataset-root", $labelsRoot,
    "--split", $Split,
    "--output", $OutputDirectory,
    "--device", "cuda:0",
    "--require-high-resolution-recipient-input",
    "--left-trims"
)
$arguments += @($LeftTrims | ForEach-Object { "$_" })

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Recipient trim audit failed with exit code $LASTEXITCODE"
}

Write-Host "recipient trim audit=$OutputDirectory"
