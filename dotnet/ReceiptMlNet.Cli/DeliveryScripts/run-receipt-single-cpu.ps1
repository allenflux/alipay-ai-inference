[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputImage,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
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

$InputImage = [IO.Path]::GetFullPath($InputImage)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
Require-File $InputImage "input receipt image"
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to mix a single-image validation with an existing output directory: $OutputDirectory"
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
Write-Host " Receipt AI - Windows CPU single-image verification" -ForegroundColor Cyan
Write-Host " detector + device classifier + unified receipt OCR" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Input : $InputImage"
Write-Host "Output: $OutputDirectory"
Write-Host ""

& $executable `
    --detector $detector `
    --device-model $deviceModel `
    --ocr unified `
    --ocr-model $unifiedModel `
    --input $InputImage `
    --output $OutputDirectory `
    --device cpu `
    --rectification max-side-1600 `
    --annotate all `
    --require-complete
if ($LASTEXITCODE -ne 0) {
    throw "Single-image CPU inference failed with exit code $LASTEXITCODE."
}

$manifestPath = Join-Path $OutputDirectory "inference_manifest.json"
$summaryPath = Join-Path $OutputDirectory "inference_summary.json"
Require-File $manifestPath "single-image inference manifest"
Require-File $summaryPath "single-image inference summary"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.Count -ne 1 -or [string]$manifest[0].status -ne "written") {
    throw "Single-image inference did not produce exactly one clean result."
}
$resultPath = [string]$manifest[0].result
Require-File $resultPath "single-image result JSON"
$result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json

$fieldRows = @(
    [pscustomobject]@{ Field = "Amount"; Candidate = [string]$result.fields.amount.candidate; State = [string]$result.fields.amount.state },
    [pscustomobject]@{ Field = "Time"; Candidate = [string]$result.fields.time.candidate; State = [string]$result.fields.time.state },
    [pscustomobject]@{ Field = "Recipient"; Candidate = [string]$result.fields.recipient.candidate; State = [string]$result.fields.recipient.state },
    [pscustomobject]@{ Field = "Payment method"; Candidate = [string]$result.fields.payment_method.candidate; State = [string]$result.fields.payment_method.state },
    [pscustomobject]@{ Field = "Transfer status"; Candidate = [string]$result.fields.transfer_status.candidate; State = [string]$result.fields.transfer_status.state }
)

Write-Host ""
Write-Host "RESULT" -ForegroundColor Green
Write-Host ("Device : {0} ({1}, confidence {2})" -f $result.device.platform_cn, $result.device.platform, $result.device.confidence)
$fieldRows | Format-Table -AutoSize
Write-Host ("CPU latency: {0:N2} ms" -f [double]$summary.inference_latency_ms.mean)
Write-Host "Result JSON : $resultPath"
Write-Host "Annotated   : $($manifest[0].annotated_original)"
Write-Host "Policy      : candidates are shown for verification; business values remain fail-closed as review."
Write-Host ""
Write-Host "PASS: the complete three-model CPU pipeline produced one reviewable receipt." -ForegroundColor Green
