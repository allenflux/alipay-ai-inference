[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 80)]
    [int]$Epochs = 12,
    [string]$RunName
)

$ErrorActionPreference = "Stop"

# Keep the contract test in the same CUDA virtual environment as the actual
# 4090 training command.  Git Bash can put the system Python first on PATH.
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing CUDA virtual-environment Python: $pythonExe"
}

# This is intentionally a short, reproducible entry point for the 4090 host.
# It first checks the v12 contract, then delegates the guarded run to the
# reusable runner.  The candidate is a new recipient-only input view; the
# established amount, time, and payment guards remain mandatory.
$seedRun = Join-Path $TeacherRoot "unified-run-v12-r3-4090-recipient-only-20260805-074047"
$seedCheckpoint = Join-Path $seedRun "best.pt"
$seedModel = Join-Path $seedRun "best.onnx"
$guardedRunner = Join-Path $PSScriptRoot "receipt-ocr-guarded-4090.ps1"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unified-run-v12-r3-4090-width1536-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

Write-Host "width1536_pilot_tests"
Write-Host "  python=$pythonExe"
& $pythonExe -m pytest -q tests/test_ocr_unified.py tests/test_ocr_unified_v12.py
if ($LASTEXITCODE -ne 0) {
    throw "v12 tests failed with exit code $LASTEXITCODE; refusing to start the Pilot."
}

Write-Host "width1536_pilot_train"
& $guardedRunner `
    -TeacherRoot $TeacherRoot `
    -Epochs $Epochs `
    -RunName $RunName `
    -SeedCheckpoint $seedCheckpoint `
    -SeedModel $seedModel `
    -InitCheckpointMode "recipient_input_width_expansion" `
    -RecipientInputWidth 1536 `
    -RecipientFloor 0.80 `
    -RecipientTailRareCharacterMaxSupport 3 `
    -RecipientTailRareCharacterLossWeight 2.0 `
    -RecipientTailLongTextMinLength 9 `
    -RecipientTailLongTextLossWeight 2.0

exit $LASTEXITCODE
