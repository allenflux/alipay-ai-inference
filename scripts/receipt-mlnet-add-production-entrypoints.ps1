[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDeliveryDir,
    [Parameter(Mandatory = $true)]
    [string]$DestinationDeliveryDir
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

function Get-RelativePackagePath([string]$Path, [string]$PackageRoot) {
    $relativePath = $Path.Substring($PackageRoot.Length)
    while ($relativePath.StartsWith("\", [StringComparison]::Ordinal) `
        -or $relativePath.StartsWith("/", [StringComparison]::Ordinal)) {
        $relativePath = $relativePath.Substring(1)
    }
    return $relativePath.Replace('\', '/')
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$singleEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-single-cpu.ps1"
$batchEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-batch-cpu.ps1"
$deliveryReadme = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\README-CPU.md"
$SourceDeliveryDir = [IO.Path]::GetFullPath($SourceDeliveryDir)
$DestinationDeliveryDir = [IO.Path]::GetFullPath($DestinationDeliveryDir)

if (-not (Test-Path -LiteralPath $SourceDeliveryDir -PathType Container)) {
    throw "Missing source delivery package: $SourceDeliveryDir"
}
if ($SourceDeliveryDir.Equals($DestinationDeliveryDir, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source and destination delivery directories must differ."
}
if (Test-Path -LiteralPath $DestinationDeliveryDir) {
    throw "Refusing to overwrite an existing delivery package: $DestinationDeliveryDir"
}
Require-File $singleEntrypoint "single-image production CPU entrypoint"
Require-File $batchEntrypoint "batch production CPU entrypoint"
Require-File $deliveryReadme "production CPU delivery README"

$sourceConfigPath = Join-Path $SourceDeliveryDir "evidence\package_config.json"
$sourceValidationPath = Join-Path $SourceDeliveryDir "evidence\package_validation.json"
$sourceHashesPath = Join-Path $SourceDeliveryDir "SHA256SUMS.json"
Require-File $sourceConfigPath "source package configuration"
Require-File $sourceValidationPath "source package validation"
Require-File $sourceHashesPath "source package hash manifest"

$sourceConfig = Get-Content -LiteralPath $sourceConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceValidation = Get-Content -LiteralPath $sourceValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$sourceConfig.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$sourceConfig.onnx_runtime_flavor -ne "cpu" `
    -or [string]$sourceConfig.runtime_device -ne "cpu" `
    -or [string]$sourceConfig.rectification -ne "max-side-1600" `
    -or [string]::IsNullOrWhiteSpace([string]$sourceConfig.device_model)) {
    throw "Source is not an accepted complete three-model production CPU package."
}
if ([string]$sourceValidation.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$sourceValidation.runtime_flavor -ne "cpu" `
    -or [string]$sourceValidation.runtime_device -ne "cpu" `
    -or [string]$sourceValidation.rectification -ne "max-side-1600" `
    -or [string]$sourceValidation.end_to_end_evaluation.status -ne "accepted" `
    -or $sourceValidation.end_to_end_evaluation.performed -ne $true) {
    throw "Source package validation has not accepted the complete CPU full-val run."
}

$sourceHashRows = Get-Content -LiteralPath $sourceHashesPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($sourceHashRows.Count -le 0) {
    throw "Source package hash manifest is empty."
}
foreach ($row in $sourceHashRows) {
    $relativePath = ([string]$row.path).Replace('/', [IO.Path]::DirectorySeparatorChar)
    if ([IO.Path]::IsPathRooted($relativePath) -or $relativePath.Split([IO.Path]::DirectorySeparatorChar) -contains "..") {
        throw "Unsafe path in source package hash manifest: $($row.path)"
    }
    $sourceFile = Join-Path $SourceDeliveryDir $relativePath
    Require-File $sourceFile "source package hash target"
    if ((Get-Sha256 $sourceFile) -ne ([string]$row.sha256).ToLowerInvariant() `
        -or (Get-Item -LiteralPath $sourceFile).Length -ne [long]$row.bytes) {
        throw "Source package integrity check failed: $($row.path)"
    }
}

$destinationParent = Split-Path -Parent $DestinationDeliveryDir
if ([string]::IsNullOrWhiteSpace($destinationParent)) {
    throw "DestinationDeliveryDir must have a parent directory."
}
New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
$stagingRoot = Join-Path $destinationParent (".receipt-mlnet-entrypoints-staging-" + [Guid]::NewGuid().ToString("N"))
$published = $false

try {
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $SourceDeliveryDir -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination $stagingRoot -Recurse
    }

    Copy-Item -LiteralPath $singleEntrypoint -Destination $stagingRoot -Force
    Copy-Item -LiteralPath $batchEntrypoint -Destination $stagingRoot -Force
    Copy-Item -LiteralPath $deliveryReadme -Destination $stagingRoot -Force

    $stagingConfigPath = Join-Path $stagingRoot "evidence\package_config.json"
    $stagingConfig = Get-Content -LiteralPath $stagingConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $stagingConfig | Add-Member -NotePropertyName production_entrypoints -NotePropertyValue @(
        [IO.Path]::GetFileName($singleEntrypoint),
        [IO.Path]::GetFileName($batchEntrypoint)
    ) -Force
    $stagingConfig | Add-Member -NotePropertyName delivery_readme `
        -NotePropertyValue ([IO.Path]::GetFileName($deliveryReadme)) -Force
    $stagingConfig | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $stagingConfigPath -Encoding UTF8

    $augmentation = [ordered]@{
        schema_version = 1
        kind = "receipt_mlnet_production_entrypoint_augmentation_v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        source_delivery = $SourceDeliveryDir
        source_sha256s_sha256 = Get-Sha256 $sourceHashesPath
        source_package_config_sha256 = Get-Sha256 $sourceConfigPath
        source_package_validation_sha256 = Get-Sha256 $sourceValidationPath
        validation_scope = [string]$sourceValidation.validation_scope
        runtime_flavor = [string]$sourceValidation.runtime_flavor
        runtime_device = [string]$sourceValidation.runtime_device
        rectification = [string]$sourceValidation.rectification
        end_to_end_status = [string]$sourceValidation.end_to_end_evaluation.status
        entrypoints = @(
            [ordered]@{
                path = [IO.Path]::GetFileName($singleEntrypoint)
                sha256 = Get-Sha256 $singleEntrypoint
            },
            [ordered]@{
                path = [IO.Path]::GetFileName($batchEntrypoint)
                sha256 = Get-Sha256 $batchEntrypoint
            },
            [ordered]@{
                path = [IO.Path]::GetFileName($deliveryReadme)
                sha256 = Get-Sha256 $deliveryReadme
            }
        )
    }
    $augmentation | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $stagingRoot "evidence\entrypoint_augmentation.json") -Encoding UTF8

    $stagingHashesPath = Join-Path $stagingRoot "SHA256SUMS.json"
    Remove-Item -LiteralPath $stagingHashesPath -Force
    $hashRows = @(
        Get-ChildItem -LiteralPath $stagingRoot -Recurse -File |
            Sort-Object FullName |
            ForEach-Object {
                [ordered]@{
                    path = Get-RelativePackagePath $_.FullName $stagingRoot
                    sha256 = Get-Sha256 $_.FullName
                    bytes = $_.Length
                }
            }
    )
    ConvertTo-Json -InputObject @($hashRows) -Depth 5 |
        Set-Content -LiteralPath $stagingHashesPath -Encoding UTF8

    if (Test-Path -LiteralPath $DestinationDeliveryDir) {
        throw "Destination appeared during augmentation; refusing to overwrite it: $DestinationDeliveryDir"
    }
    Move-Item -LiteralPath $stagingRoot -Destination $DestinationDeliveryDir
    $published = $true

    Write-Host "PASS: production CPU entrypoints were added after verifying the accepted source package."
    Write-Host "  source=$SourceDeliveryDir"
    Write-Host "  delivery=$DestinationDeliveryDir"
    Write-Host "  single=$(Join-Path $DestinationDeliveryDir ([IO.Path]::GetFileName($singleEntrypoint)))"
    Write-Host "  batch=$(Join-Path $DestinationDeliveryDir ([IO.Path]::GetFileName($batchEntrypoint)))"
    Write-Host "  readme=$(Join-Path $DestinationDeliveryDir ([IO.Path]::GetFileName($deliveryReadme)))"
    Write-Host "  sha256s=$(Join-Path $DestinationDeliveryDir 'SHA256SUMS.json')"
}
finally {
    if (-not $published -and (Test-Path -LiteralPath $stagingRoot)) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
