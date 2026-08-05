[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 24)]
    [int]$Epochs = 8,
    [string]$RunName,
    [switch]$UnfreezeLegacy,
    [switch]$FullOnnxValidation
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$guardedRunner = Join-Path $PSScriptRoot "receipt-ocr-guarded-4090.ps1"
$seedRun = Join-Path $TeacherRoot "unified-run-v12-r3-4090-recipient-only-20260805-074047"
$seedCheckpoint = Join-Path $seedRun "best.pt"
$seedModel = Join-Path $seedRun "best.onnx"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $scope = if ($UnfreezeLegacy) { "joint" } else { "adapter-only" }
    $RunName = "unified-run-v12-r3-4090-open-text-transformer2-" + $scope + "-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
foreach ($required in @($pythonExe, $guardedRunner, $seedCheckpoint, $seedModel)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing open-text pilot dependency: $required"
    }
}

Write-Host "open_text_transformer_pilot_tests"
& $pythonExe -m pytest -q tests/test_ocr_unified.py tests/test_ocr_unified_v12.py
if ($LASTEXITCODE -ne 0) {
    throw "OCR tests failed with exit code $LASTEXITCODE; refusing to start the open-text pilot."
}

$runnerArgs = @{
    TeacherRoot = $TeacherRoot
    Epochs = $Epochs
    RunName = $RunName
    SeedCheckpoint = $seedCheckpoint
    SeedModel = $seedModel
    InitCheckpointMode = "recipient_open_text_adapter"
    RecipientInputWidth = 1024
    RecipientBranchChannels = 24
    RecipientHiddenSize = 256
    RecipientOpenTextLayers = 2
    RecipientOpenTextHeads = 8
    RecipientOpenTextFeedforward = 2048
    RecipientFloor = 0.80
    LearningRate = $(if ($UnfreezeLegacy) { 0.00005 } else { 0.0003 })
    ValidationEvery = 2
    RecipientTailRareCharacterMaxSupport = 3
    RecipientTailRareCharacterLossWeight = 2.0
    RecipientTailLongTextMinLength = 9
    RecipientTailLongTextLossWeight = 2.0
}
if ($UnfreezeLegacy) {
    $runnerArgs["RecipientOpenTextUnfreezeLegacy"] = $true
}
if (-not $FullOnnxValidation) {
    $runnerArgs["DiagnosticOnly"] = $true
}

Write-Host "open_text_transformer_pilot_train"
& $guardedRunner @runnerArgs
exit $LASTEXITCODE
