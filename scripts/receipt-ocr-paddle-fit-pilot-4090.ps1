[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(1, 40)]
    [int]$Epochs = 16,
    [string]$RunName,
    [ValidateRange(1024, 2048)]
    [int]$RecipientInputWidth = 1536,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$AdapterOnly,
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
    $scope = if ($AdapterOnly) { "adapter-only" } else { "joint" }
    $RunName = "unified-run-v12-r3-4090-paddle-fit-open-text-" + $scope + "-wide$RecipientInputWidth-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
foreach ($required in @($pythonExe, $guardedRunner, $seedCheckpoint, $seedModel)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing Paddle-fit pilot dependency: $required"
    }
}

Write-Host "paddle_fit_open_text_pilot_tests"
& $pythonExe -m pytest -q tests/test_ocr_unified.py tests/test_ocr_unified_v12.py
if ($LASTEXITCODE -ne 0) {
    throw "OCR tests failed with exit code $LASTEXITCODE; refusing to start the Paddle-fit pilot."
}

$runnerArgs = @{
    TeacherRoot = $TeacherRoot
    Epochs = $Epochs
    RunName = $RunName
    SeedCheckpoint = $seedCheckpoint
    SeedModel = $seedModel
    InitCheckpointMode = "recipient_open_text_adapter"
    RecipientInputWidth = $RecipientInputWidth
    RecipientBranchChannels = 24
    RecipientHiddenSize = 256
    RecipientOpenTextLayers = 2
    RecipientOpenTextHeads = 8
    RecipientOpenTextFeedforward = 2048
    RecipientFloor = 0.90
    LearningRate = $(if ($AdapterOnly) { 0.0003 } else { 0.00008 })
    ValidationEvery = 1
    RecipientLowConfidenceThreshold = 0.0
    RecipientLowConfidenceLossWeight = 1.0
    RecipientConfidenceCurriculumEpochs = 0
    RecipientTailRareCharacterMaxSupport = 3
    RecipientTailRareCharacterLossWeight = 2.0
    RecipientTailLongTextMinLength = 9
    RecipientTailLongTextLossWeight = 2.0
    RecipientTrainSplits = @("train", "val")
    NumWorkers = $NumWorkers
    TrainProgressEvery = $TrainProgressEvery
}
if (-not $AdapterOnly) {
    $runnerArgs["RecipientOpenTextUnfreezeLegacy"] = $true
}
if (-not $FullOnnxValidation) {
    $runnerArgs["DiagnosticOnly"] = $true
}

Write-Host "paddle_fit_open_text_pilot_train"
& $guardedRunner @runnerArgs
exit $LASTEXITCODE
