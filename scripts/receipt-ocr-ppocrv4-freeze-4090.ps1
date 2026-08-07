[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$Paddle2Onnx
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repositoryRoot ".venv-cu126\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($Paddle2Onnx)) {
    $Paddle2Onnx = Join-Path $repositoryRoot ".venv-cu126\Scripts\paddle2onnx.exe"
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
foreach ($required in @(
    @{ Name = "CUDA Python"; Path = $pythonExe },
    @{ Name = "paddle2onnx"; Path = $Paddle2Onnx }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType Leaf)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to overwrite frozen PP-OCRv4 audit bundle: $OutputDirectory"
}

Write-Host "receipt_ocr_ppocrv4_freeze_4090"
Write-Host "  source=current pinned PaddleOCR 2.10 default PP-OCRv4 assets"
Write-Host "  output=$OutputDirectory"
Write-Host "  delivery=no (this is the immutable conversion audit bundle)"

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot "src"
    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle snapshot `
        --output $OutputDirectory `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCRv4 snapshot failed with exit code $LASTEXITCODE"
    }
    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle export-onnx `
        --bundle $OutputDirectory `
        --paddle2onnx $Paddle2Onnx `
        --opset-version 11
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCRv4 ONNX export failed with exit code $LASTEXITCODE"
    }
    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle verify `
        --bundle $OutputDirectory `
        --require-onnx
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCRv4 snapshot verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

Write-Host "PP-OCRv4 AUDIT SNAPSHOT + ONNX EXPORT: PASS" -ForegroundColor Green
Write-Host "  bundle=$OutputDirectory"
Write-Host "Next: run a fresh full val with -Bundle bound to this now-immutable exported bundle."
