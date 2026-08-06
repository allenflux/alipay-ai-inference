[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputDirectory,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [ValidateSet("all", "flagged", "none")]
    [string]$Annotate = "none",
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

function Require-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-PackageIntegrity([string]$PackageRoot) {
    $hashManifestPath = Join-Path $PackageRoot "SHA256SUMS.json"
    Require-File $hashManifestPath "delivery package hash manifest"
    $hashRows = Get-Content -LiteralPath $hashManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($hashRows.Count -le 0) {
        throw "Delivery package hash manifest is empty."
    }
    foreach ($row in $hashRows) {
        $relativePath = ([string]$row.path).Replace('/', [IO.Path]::DirectorySeparatorChar)
        $segments = @($relativePath.Split([IO.Path]::DirectorySeparatorChar))
        if ([IO.Path]::IsPathRooted($relativePath) -or $segments -contains "..") {
            throw "Unsafe path in delivery package hash manifest: $($row.path)"
        }
        $target = Join-Path $PackageRoot $relativePath
        Require-File $target "delivery package hash target"
        if ((Get-Sha256 $target) -ne ([string]$row.sha256).ToLowerInvariant() `
            -or (Get-Item -LiteralPath $target).Length -ne [long]$row.bytes) {
            throw "Delivery package integrity check failed: $($row.path)"
        }
    }
}

$packageRoot = $PSScriptRoot
Write-Host "Verifying delivery package integrity..." -ForegroundColor DarkGray
Assert-PackageIntegrity $packageRoot
Write-Host "Package: PASS" -ForegroundColor Green
$configPath = Join-Path $packageRoot "evidence\package_config.json"
Require-File $configPath "delivery package configuration"
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$config.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$config.onnx_runtime_flavor -ne "cpu" `
    -or [string]$config.rectification -ne "max-side-1600" `
    -or [string]::IsNullOrWhiteSpace([string]$config.device_model)) {
    throw "This is not an accepted three-model production CPU package."
}

$InputDirectory = [IO.Path]::GetFullPath($InputDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $InputDirectory -PathType Container)) {
    throw "Missing input image directory: $InputDirectory"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to mix a batch with an existing output directory: $OutputDirectory"
}

$executable = Join-Path $packageRoot "app\ReceiptMlNet.Cli.exe"
$detector = Join-Path $packageRoot ("models\" + [string]$config.detector_model)
$deviceModel = Join-Path $packageRoot ("models\" + [string]$config.device_model)
$unifiedModel = Join-Path $packageRoot (([string]$config.unified_model).Replace("/", "\"))
Require-File $executable "ReceiptMlNet executable"
Require-File $detector "receipt detector"
Require-File $deviceModel "device classifier"
Require-File $unifiedModel "unified receipt OCR"

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host " Receipt AI - Windows CPU batch verification" -ForegroundColor Cyan
Write-Host " detector + device classifier + unified receipt OCR" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Input : $InputDirectory"
Write-Host "Output: $OutputDirectory"
Write-Host "JPG   : $Annotate"
if ($Limit -gt 0) { Write-Host "Limit : $Limit" }
Write-Host ""

$arguments = @(
    "--detector", $detector,
    "--device-model", $deviceModel,
    "--ocr", "unified",
    "--ocr-model", $unifiedModel,
    "--input", $InputDirectory,
    "--output", $OutputDirectory,
    "--device", "cpu",
    "--rectification", "max-side-1600",
    "--annotate", $Annotate,
    "--require-complete",
    "--continue-on-error"
)
if ($Limit -gt 0) {
    $arguments += @("--limit", [string]$Limit)
}
& $executable @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Batch CPU inference failed with exit code $LASTEXITCODE."
}

$summaryPath = Join-Path $OutputDirectory "inference_summary.json"
$errorsPath = Join-Path $OutputDirectory "inference_errors.jsonl"
Require-File $summaryPath "batch inference summary"
Require-File $errorsPath "batch inference errors"
$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json

Write-Host ""
Write-Host "BATCH RESULT" -ForegroundColor $(if ([int]$summary.errors -eq 0) { "Green" } else { "Red" })
[pscustomobject]@{
    Input = [int]$summary.input
    Written = [int]$summary.written
    Errors = [int]$summary.errors
    MeanMs = [Math]::Round([double]$summary.inference_latency_ms.mean, 2)
    P50Ms = [Math]::Round([double]$summary.inference_latency_ms.p50, 2)
    P95Ms = [Math]::Round([double]$summary.inference_latency_ms.p95, 2)
    TotalSeconds = [double]$summary.total_seconds
} | Format-List
Write-Host ("Detector mean : {0:N2} ms" -f [double]$summary.stage_latency_ms.detector_inference.mean)
Write-Host ("Unified mean  : {0:N2} ms" -f [double]$summary.stage_latency_ms.unified_ocr_inference.mean)
Write-Host "Results       : $OutputDirectory"
Write-Host "Errors        : $errorsPath"
if ([int]$summary.errors -ne 0) {
    throw "Batch completed with $($summary.errors) error(s); inspect inference_errors.jsonl before delivery."
}
if ([int]$summary.written -ne [int]$summary.input) {
    throw "Batch is incomplete: input=$($summary.input), written=$($summary.written)."
}
Write-Host ""
Write-Host "PASS: the complete three-model CPU pipeline wrote every selected receipt." -ForegroundColor Green
