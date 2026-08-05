[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateSet("val", "test")]
    [string]$Split = "val",
    [string]$Device = "cuda",
    [ValidateRange(0, 10000000)]
    [int]$Limit = 0,
    [ValidateRange(0.000001, 1.0)]
    [double]$Target = 0.90,
    [ValidateRange(1, 1000000)]
    [int]$ProgressEvery = 25,
    [string]$OutputDirectory,
    [string]$Bundle,
    [switch]$SkipDetection
)

$ErrorActionPreference = "Stop"

# Run native PaddleOCR in a dedicated no-Torch process.  It compares only the
# held-out v12 recipient crops and cannot change the student checkpoint.
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$scriptPath = Join-Path $PSScriptRoot "paddle-recipient-evaluate.py"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $suffix = if ($Limit -gt 0) { "-$Limit" } else { "-full" }
    $OutputDirectory = Join-Path $TeacherRoot ("paddle-recipient-" + $Split + $suffix + "-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}

foreach ($required in @(
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot },
    @{ Name = "Paddle recipient evaluator"; Path = $scriptPath }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse Paddle recipient output: $OutputDirectory"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle) -and -not (Test-Path -LiteralPath $Bundle)) {
    throw "Frozen Paddle bundle not found: $Bundle"
}

Write-Host "receipt_ocr_paddle_recipient_4090"
Write-Host "  records=$records"
Write-Host "  crop-root=$labelsRoot"
Write-Host "  split=$Split; device=$Device; limit=$Limit; target=$Target"
Write-Host "  output=$OutputDirectory"
if ($SkipDetection) {
    Write-Host "  mode=EXPERIMENTAL skip-det cls+rec; det=False; use only for 4090 speed A/B"
} else {
    Write-Host "  mode=full det+cls+rec (default); native PaddleOCR teacher-parity only; no Torch/student model is loaded"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle)) {
    Write-Host "  frozen-bundle=$Bundle"
}

$arguments = @(
    $scriptPath,
    "--manifest", $records,
    "--dataset-root", $labelsRoot,
    "--split", $Split,
    "--device", $Device,
    "--output", $OutputDirectory,
    "--target", "$Target",
    "--progress-every", "$ProgressEvery"
)
if ($Limit -gt 0) {
    $arguments += "--limit"
    $arguments += "$Limit"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle)) {
    $arguments += "--bundle"
    $arguments += $Bundle
}
if ($SkipDetection) {
    $arguments += "--skip-detection"
}

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Paddle recipient evaluation failed with exit code $LASTEXITCODE"
}

Write-Host "Paddle recipient evaluation=$OutputDirectory"
