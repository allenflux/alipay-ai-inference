[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
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

# This launcher is intentionally locked to held-out validation data and CUDA.
# PaddleOCR is used only as a teacher-ceiling diagnostic; it neither trains nor
# changes the pure ONNX + .NET CPU delivery package.
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repositoryRoot ".venv-cu126\Scripts\python.exe"
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$manifestRoot = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$evaluator = Join-Path $PSScriptRoot "paddle-recipient-evaluate.py"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $suffix = if ($Limit -gt 0) { "-$Limit" } else { "-full" }
    $OutputDirectory = Join-Path $TeacherRoot (
        "ppocrv4-recipient-val-ceiling" + $suffix + "-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    )
}

foreach ($required in @(
    @{ Name = "CUDA Python"; Path = $pythonPath },
    @{ Name = "r3 records"; Path = $records },
    @{ Name = "r3 crop root"; Path = $labelsRoot },
    @{ Name = "Paddle recipient evaluator"; Path = $evaluator }
)) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse PP-OCRv4 recipient val output: $OutputDirectory"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle) -and -not (Test-Path -LiteralPath $Bundle)) {
    throw "Frozen Paddle bundle not found: $Bundle"
}

Write-Host "receipt_ocr_ppocrv4_recipient_val_ceiling_4090"
Write-Host "  split=val (hard locked); device=cuda (CPU fallback is rejected)"
Write-Host "  records=$records"
Write-Host "  crop-root=$labelsRoot"
Write-Host "  python=$pythonPath"
Write-Host "  output=$OutputDirectory"
Write-Host "  role=teacher ceiling diagnostic only; delivery remains pure ONNX + .NET CPU"
if ($SkipDetection) {
    Write-Host "  mode=EXPERIMENTAL cls+SVTR_LCNet rec; det=False"
} else {
    Write-Host "  mode=full PP-OCRv4 det+cls+SVTR_LCNet rec"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle)) {
    Write-Host "  frozen-bundle=$Bundle"
}

$arguments = @(
    $evaluator,
    "--manifest", $records,
    "--dataset-root", $labelsRoot,
    "--split", "val",
    "--device", "cuda",
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

& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PP-OCRv4 recipient val ceiling diagnostic failed with exit code $LASTEXITCODE"
}

Write-Host "PP-OCRv4 recipient val ceiling=$OutputDirectory"
