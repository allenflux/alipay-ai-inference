[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 80)]
    [int]$Epochs = 16,
    [string]$RunName,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 0,
    [switch]$DiagnosticOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing CUDA virtual-environment Python: $pythonExe"
}

$seedRun = Join-Path $TeacherRoot "unified-run-v12-r3-4090-recipient-only-20260805-074047"
$seedCheckpoint = Join-Path $seedRun "best.pt"
$seedModel = Join-Path $seedRun "best.onnx"
$guardedRunner = Join-Path $PSScriptRoot "receipt-ocr-guarded-4090.ps1"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unified-run-v12-r3-4090-capacity32-h384-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

Write-Host "capacity32_h384_pilot_tests"
Write-Host "  python=$pythonExe"
& $pythonExe -m pytest -q tests/test_ocr_unified.py tests/test_ocr_unified_v12.py
if ($LASTEXITCODE -ne 0) {
    throw "v12 tests failed with exit code $LASTEXITCODE; refusing to start the Pilot."
}

Write-Host "capacity32_h384_pilot_train"
& $guardedRunner `
    -TeacherRoot $TeacherRoot `
    -Epochs $Epochs `
    -RunName $RunName `
    -SeedCheckpoint $seedCheckpoint `
    -SeedModel $seedModel `
    -InitCheckpointMode "recipient_capacity_reinit" `
    -RecipientInputWidth 1024 `
    -RecipientBranchChannels 32 `
    -RecipientHiddenSize 384 `
    -RecipientFloor 0.80 `
    -LearningRate 0.0003 `
    -ValidationEvery 4 `
    -RecipientTailRareCharacterMaxSupport 3 `
    -RecipientTailRareCharacterLossWeight 3.0 `
    -RecipientTailLongTextMinLength 9 `
    -RecipientTailLongTextLossWeight 3.0 `
    -NumWorkers $NumWorkers `
    -DiagnosticOnly:$DiagnosticOnly

exit $LASTEXITCODE
