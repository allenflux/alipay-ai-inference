[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StatusCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$TrainOnlyRecipientCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$OutputCheckpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$sanitizer = Join-Path $PSScriptRoot "receipt-ocr-recipient-full-crop-seed-sanitize.py"
$contractTests = Join-Path $repoRoot "tests\test_recipient_full_crop_seed_sanitizer.py"

function Require-RegularNonReparseFile([string]$Path, [string]$Description) {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -eq $item -or $item.PSIsContainer) {
        throw "Missing ${Description}: $Path"
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "${Description} must not be a symlink, junction, or reparse point: $Path"
    }
}

function Require-FreshNonReparseOutput([string]$Path) {
    if ([IO.Path]::GetExtension($Path) -ne ".pt") {
        throw "OutputCheckpoint must use a .pt extension."
    }
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing -or (Test-Path -LiteralPath $Path)) {
        throw "Refusing to overwrite an existing sanitizer output: $Path"
    }
    $ancestor = Split-Path -Parent $Path
    if ([string]::IsNullOrWhiteSpace($ancestor) -or -not (Test-Path -LiteralPath $ancestor -PathType Container)) {
        throw "OutputCheckpoint parent must already exist: $ancestor"
    }
    while (-not [string]::IsNullOrWhiteSpace($ancestor)) {
        $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "OutputCheckpoint must not traverse a symlink, junction, or reparse point: $ancestor"
        }
        $next = Split-Path -Parent $ancestor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $ancestor) {
            break
        }
        $ancestor = $next
    }
}

Require-RegularNonReparseFile $pythonExe "CUDA virtual-environment Python"
Require-RegularNonReparseFile $sanitizer "seed sanitizer entry point"
Require-RegularNonReparseFile $contractTests "seed sanitizer contract tests"
Require-RegularNonReparseFile $StatusCheckpoint "v13 status checkpoint"
Require-RegularNonReparseFile $TrainOnlyRecipientCheckpoint "train-only wide1536 v12 checkpoint"

$statusPath = [IO.Path]::GetFullPath($StatusCheckpoint)
$trainPath = [IO.Path]::GetFullPath($TrainOnlyRecipientCheckpoint)
$outputPath = [IO.Path]::GetFullPath($OutputCheckpoint)
if ($statusPath -eq $trainPath -or $outputPath -eq $statusPath -or $outputPath -eq $trainPath) {
    throw "StatusCheckpoint, TrainOnlyRecipientCheckpoint, and OutputCheckpoint must be distinct."
}
Require-FreshNonReparseOutput $outputPath

Write-Host "receipt_recipient_full_crop_seed_sanitizer"
Write-Host "  analysis-only; production authorization remains false"
Write-Host "  non-recipient/status tensors and model metadata: v13 status checkpoint"
Write-Host "  recipient_ tensors and recipient metadata: train-only wide1536 v12 checkpoint"
Write-Host "  old status initialization/runtime/metrics: non-operative history only"
Write-Host "  every v12 warmstart ancestor: recorded path/hash/config/epoch, train-only, to random root"
Write-Host "  full-crop warmstart reopens A/B and every B ancestor; a publicly resealed splice is rejected"
Write-Host "  no optimizer restore; no manifest or held-out data lookup"

& $pythonExe -m pytest -q $contractTests
if ($LASTEXITCODE -ne 0) {
    throw "Seed sanitizer contract tests failed with exit code $LASTEXITCODE"
}

& $pythonExe $sanitizer `
    --status-checkpoint $statusPath `
    --train-only-recipient-checkpoint $trainPath `
    --output-checkpoint $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Seed sanitization failed with exit code $LASTEXITCODE"
}

Require-RegularNonReparseFile $outputPath "published sanitized v13 checkpoint"
$outputHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "SANITIZER PASS: analysis-only v13 checkpoint published atomically."
Write-Host "  checkpoint=$outputPath"
Write-Host "  sha256=$outputHash"
