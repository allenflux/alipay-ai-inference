[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$OutputRoot,
    [ValidateRange(1, 80)]
    [int]$Epochs = 30,
    [ValidateRange(1, 64)]
    [int]$BatchSize = 16,
    [ValidateRange(1, 80)]
    [int]$ValidationEvery = 4,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.001,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 32)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$wrapper = Join-Path $PSScriptRoot "receipt-ocr-status-text-v13-4090.ps1"
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$pseudoLabels = Join-Path $labelsRoot "pseudo_labels.jsonl"
$seedRun = Join-Path $TeacherRoot "unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
$seedCheckpoint = Join-Path $seedRun "best.pt"

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $tag = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $OutputRoot = Join-Path $TeacherRoot "unified-run-v13-status-cjk-ocr-wide1536-$tag"
}

$runnerArgs = @{
    PseudoLabels = $pseudoLabels
    DatasetRoot = $labelsRoot
    SeedCheckpoint = $seedCheckpoint
    OutputRoot = $OutputRoot
    Epochs = $Epochs
    BatchSize = $BatchSize
    ValidationEvery = $ValidationEvery
    LearningRate = $LearningRate
    StatusTextFloor = 0.90
    NumWorkers = $NumWorkers
    PrefetchFactor = $PrefetchFactor
    TrainProgressEvery = $TrainProgressEvery
}
if ($CheckOnly) {
    $runnerArgs.CheckOnly = $true
}

Write-Host "receipt_status_cjk_v13_launcher"
Write-Host "  teacher-root=$TeacherRoot"
Write-Host "  output=$OutputRoot"
Write-Host "  status-text-floor=90.00%"

& $wrapper @runnerArgs
