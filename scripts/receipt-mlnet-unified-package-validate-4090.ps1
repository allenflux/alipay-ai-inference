[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    # Optional direct artifact bindings let additive runs (for example v13's
    # artifacts/status-text-v13.onnx) enter the existing packager without
    # renaming files or rewriting their hash-bound sidecars.  Supply both or
    # neither; the legacy best.onnx + onnx-val/summary.json layout remains the
    # default.
    [string]$UnifiedModelPath,
    [string]$OnnxValidationSummaryPath,
    [Alias("Input")]
    [string]$InputPath,
    [string]$InputList,
    [string]$Records,
    [string]$EndToEndEvaluationDir,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [Parameter(Mandatory = $true)]
    [string]$DeliveryDir,
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,
    [ValidateSet("cpu", "gpu")]
    [string]$RuntimeFlavor = "cpu",
    [ValidateSet("none", "max-side-1600")]
    [string]$Rectification = "max-side-1600",
    [switch]$IncludeDeviceModel,
    [ValidateSet("all", "flagged", "none")]
    [string]$Annotate = "none",
    [ValidateRange(0.0, 1.0)]
    [double]$AmountFloor = 0.7885,
    [ValidateRange(0.0, 1.0)]
    [double]$TimeFloor = 0.9840,
    [ValidateRange(0.0, 1.0)]
    [double]$PaymentFloor = 0.9325,
    [ValidateRange(0.0, 1.0)]
    [double]$RecipientFloor = 0.90,
    [string]$DetectorModel,
    [string]$DeviceModel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq ("ReceiptMlNetPathNativeMethods" -as [type])) {
    Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
using System.Text;

public static class ReceiptMlNetPathNativeMethods
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern uint QueryDosDevice(
        string lpDeviceName,
        StringBuilder lpTargetPath,
        int ucchMax);
}
"@
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
$endToEndScorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$projectFile = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"
$preprocessingContractTestProject = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli.PreprocessingContractTests\ReceiptMlNet.Cli.PreprocessingContractTests.csproj"
$rectificationContractTestProject = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli.RectificationContractTests\ReceiptMlNet.Cli.RectificationContractTests.csproj"
$singleCpuEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-single-cpu.ps1"
$batchCpuEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-batch-cpu.ps1"
$cpuDeliveryReadme = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\README-CPU.md"

if ([string]::IsNullOrWhiteSpace($DetectorModel)) {
    $DetectorModel = Join-Path $repoRoot "artifacts\receipt_lrcnn_v1.onnx"
}
if ([string]::IsNullOrWhiteSpace($DeviceModel)) {
    $DeviceModel = Join-Path $repoRoot "artifacts\statusbar_device_v1.onnx"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-NormalizedTransferStatus([string]$Text) {
    $compact = $Text -replace '\s+', ''
    if ($compact -match '失败|未成功|已撤销') { return "failed" }
    if ($compact -match '处理中|待处理|进行中') { return "pending" }
    if ($compact -match '转账成功|交易成功|付款成功|支付成功|转帐成功') { return "success" }
    return "unknown"
}

function Require-File([string]$Path, [string]$Description) {
    Assert-SafePathSyntax $Path $Description
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
    Assert-NoReparsePointInExistingPath $Path $Description
}

function Assert-SafePathSyntax([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Missing ${Description} path."
    }
    $aliasProbe = $Path.Replace('/', '\')
    foreach ($devicePrefix in @('\\?\', '\\.\', '\??\', '\\??\')) {
        if ($aliasProbe.StartsWith($devicePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "${Description} must not use a Windows device-path alias: $Path"
        }
    }
    if ($aliasProbe.StartsWith('\', [StringComparison]::Ordinal) `
        -and -not $aliasProbe.StartsWith('\\', [StringComparison]::Ordinal)) {
        throw "${Description} must not use a current-drive rooted path: $Path"
    }
    if ($Path -match '^[A-Za-z]:($|[^\\/])') {
        throw "${Description} must not use a drive-relative path: $Path"
    }

    $segments = @($Path.Split([char[]]@('\', '/')))
    for ($index = 0; $index -lt $segments.Count; $index++) {
        $segment = [string]$segments[$index]
        if ([string]::IsNullOrEmpty($segment)) {
            continue
        }
        if ($index -eq 0 -and $segment -match '^[A-Za-z]:$') {
            continue
        }
        if ($segment -in @('.', '..')) {
            continue
        }
        if ($segment.Contains(':')) {
            throw "${Description} must not use an alternate data stream or path alias: $Path"
        }
        $canonicalSegment = $segment.TrimEnd([char[]]@('.', ' '))
        if ($canonicalSegment.Length -ne $segment.Length) {
            throw "${Description} must not contain a trailing dot or space: $Path"
        }
        if ($canonicalSegment -match '^(?i:con|prn|aux|nul|clock\$|conin\$|conout\$|com[1-9]|lpt[1-9])(?:[ .].*)?$') {
            throw "${Description} contains a reserved Windows device name: $Path"
        }
    }
}

function Assert-NoReparsePointInExistingPath([string]$Path, [string]$Description) {
    Assert-SafePathSyntax $Path $Description
    $current = [IO.Path]::GetFullPath($Path)
    if ($current -match '^[A-Za-z]:[\\/]') {
        $driveName = $current.Substring(0, 2)
        $targetBuffer = [Text.StringBuilder]::new(32768)
        $queryLength = [ReceiptMlNetPathNativeMethods]::QueryDosDevice(
            $driveName,
            $targetBuffer,
            $targetBuffer.Capacity)
        if ($queryLength -eq 0) {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            throw "${Description} drive mapping could not be verified: $driveName (Win32 error $errorCode)"
        }
        $driveTarget = $targetBuffer.ToString()
        if ($driveTarget.StartsWith('\??\', [StringComparison]::OrdinalIgnoreCase) `
            -or $driveTarget.StartsWith('\DosDevices\', [StringComparison]::OrdinalIgnoreCase) `
            -or $driveTarget.StartsWith('\GLOBAL??\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "${Description} must not traverse a substituted DOS drive: $driveName -> $driveTarget"
        }
    }
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "${Description} must not traverse a reparse point: $($item.FullName)"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ([string]::IsNullOrWhiteSpace($parent) `
            -or $parent.Equals($current, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $current = $parent
    }
}

function Test-PathWithin([string]$Candidate, [string]$Parent) {
    Assert-SafePathSyntax $Candidate "candidate path"
    Assert-SafePathSyntax $Parent "parent path"
    $Candidate = [IO.Path]::GetFullPath($Candidate)
    $Parent = [IO.Path]::GetFullPath($Parent)
    if ($Candidate.Equals($Parent, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $parentPrefix = $Parent
    if (-not $parentPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $parentPrefix += [IO.Path]::DirectorySeparatorChar
    }
    return $Candidate.StartsWith($parentPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-RelativePackagePath([string]$Path, [string]$PackageRoot) {
    Assert-SafePathSyntax $Path "package payload"
    Assert-SafePathSyntax $PackageRoot "package root"
    $pathFull = [IO.Path]::GetFullPath($Path)
    $rootFull = [IO.Path]::GetFullPath($PackageRoot)
    $rootPrefix = $rootFull
    if (-not $rootPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $rootPrefix += [IO.Path]::DirectorySeparatorChar
    }
    if (-not $pathFull.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the delivery package: $pathFull"
    }
    return $pathFull.Substring($rootPrefix.Length).Replace('\', '/')
}

function Resolve-ContainedPackageFile(
    [string]$PackageRoot,
    [string]$RelativePath,
    [string]$Description
) {
    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        throw "Missing relative path for ${Description}."
    }
    Assert-SafePathSyntax $PackageRoot "package root"
    Assert-SafePathSyntax $RelativePath $Description
    $normalized = $RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $segments = @($normalized.Split([IO.Path]::DirectorySeparatorChar))
    if ([IO.Path]::IsPathRooted($normalized) `
        -or $segments -contains "" `
        -or $segments -contains "." `
        -or $segments -contains "..") {
        throw "Unsafe path for ${Description}: $RelativePath"
    }
    $rootFull = [IO.Path]::GetFullPath($PackageRoot)
    $target = [IO.Path]::GetFullPath((Join-Path $rootFull $normalized))
    if (-not (Test-PathWithin $target $rootFull) `
        -or $target.Equals($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path for ${Description} escapes the delivery package: $RelativePath"
    }
    Require-File $target $Description
    Assert-NoReparsePointInExistingPath $target $Description
    return $target
}

function Get-PackagePayloadFiles([string]$PackageRoot) {
    Assert-SafePathSyntax $PackageRoot "package root"
    $rootFull = [IO.Path]::GetFullPath($PackageRoot)
    if (-not (Test-Path -LiteralPath $rootFull -PathType Container)) {
        throw "Missing delivery package directory: $rootFull"
    }
    Assert-NoReparsePointInExistingPath $rootFull "package root"

    $pending = New-Object System.Collections.Queue
    $pending.Enqueue($rootFull)
    $files = @()
    while ($pending.Count -gt 0) {
        $directory = [string]$pending.Dequeue()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            Assert-SafePathSyntax $item.FullName "package payload"
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Delivery package contains a reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pending.Enqueue($item.FullName)
            }
            else {
                $files += $item
            }
        }
    }
    return @($files)
}

function Assert-PackageIntegrity([string]$PackageRoot) {
    Assert-SafePathSyntax $PackageRoot "package root"
    $PackageRoot = [IO.Path]::GetFullPath($PackageRoot)
    Assert-NoReparsePointInExistingPath $PackageRoot "package root"
    $hashManifestPath = Join-Path $PackageRoot "SHA256SUMS.json"
    Require-File $hashManifestPath "delivery package hash manifest"
    $hashRows = Get-Content -LiteralPath $hashManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $hashRows) {
        throw "Delivery package hash manifest is empty."
    }

    $listedPaths = @{}
    $hashRowCount = 0
    foreach ($row in $hashRows) {
        $hashRowCount++
        $pathProperty = $row.PSObject.Properties["path"]
        $shaProperty = $row.PSObject.Properties["sha256"]
        $bytesProperty = $row.PSObject.Properties["bytes"]
        if ($null -eq $pathProperty -or $null -eq $shaProperty -or $null -eq $bytesProperty) {
            throw "Delivery package hash manifest contains an incomplete row."
        }
        $relativePath = [string]$pathProperty.Value
        $target = Resolve-ContainedPackageFile $PackageRoot $relativePath "delivery package hash target"
        $canonicalPath = Get-RelativePackagePath $target $PackageRoot
        if ($canonicalPath.Equals("SHA256SUMS.json", [StringComparison]::OrdinalIgnoreCase)) {
            throw "SHA256SUMS.json must not contain a self-reference."
        }
        $pathKey = $canonicalPath.ToLowerInvariant()
        if ($listedPaths.ContainsKey($pathKey)) {
            throw "Duplicate path in delivery package hash manifest: $relativePath"
        }

        $expectedHash = ([string]$shaProperty.Value).ToLowerInvariant()
        if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
            throw "Invalid SHA-256 in delivery package hash manifest: $relativePath"
        }
        $expectedBytes = [long]0
        $bytesText = [Convert]::ToString($bytesProperty.Value, [Globalization.CultureInfo]::InvariantCulture)
        if (-not [long]::TryParse(
                $bytesText,
                [Globalization.NumberStyles]::Integer,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$expectedBytes) `
            -or $expectedBytes -lt 0) {
            throw "Invalid byte count in delivery package hash manifest: $relativePath"
        }
        if ((Get-Sha256 $target) -ne $expectedHash `
            -or (Get-Item -LiteralPath $target).Length -ne $expectedBytes) {
            throw "Delivery package integrity check failed: $relativePath"
        }
        $listedPaths[$pathKey] = $canonicalPath
    }
    if ($hashRowCount -le 0) {
        throw "Delivery package hash manifest is empty."
    }

    $actualPaths = @{}
    foreach ($file in Get-PackagePayloadFiles $PackageRoot) {
        if ($file.FullName.Equals($hashManifestPath, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        $canonicalPath = Get-RelativePackagePath $file.FullName $PackageRoot
        $pathKey = $canonicalPath.ToLowerInvariant()
        if ($actualPaths.ContainsKey($pathKey)) {
            throw "Duplicate canonical file path in delivery package: $canonicalPath"
        }
        $actualPaths[$pathKey] = $canonicalPath
    }
    $missingPaths = @($listedPaths.Keys | Where-Object { -not $actualPaths.ContainsKey($_) })
    $extraPaths = @($actualPaths.Keys | Where-Object { -not $listedPaths.ContainsKey($_) })
    if ($missingPaths.Count -ne 0 -or $extraPaths.Count -ne 0) {
        throw "Delivery package hash manifest is not closed: missing=$($missingPaths.Count), extra=$($extraPaths.Count)."
    }
}

function Get-SafeDirectoryFiles([string]$Root, [string]$Description) {
    Assert-SafePathSyntax $Root $Description
    $rootFull = [IO.Path]::GetFullPath($Root)
    if (-not (Test-Path -LiteralPath $rootFull -PathType Container)) {
        throw "Missing ${Description}: $rootFull"
    }
    Assert-NoReparsePointInExistingPath $rootFull $Description
    $pending = New-Object System.Collections.Queue
    $pending.Enqueue($rootFull)
    while ($pending.Count -gt 0) {
        $directory = [string]$pending.Dequeue()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            Assert-SafePathSyntax $item.FullName $Description
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "${Description} contains a reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pending.Enqueue($item.FullName)
            }
            else {
                Write-Output $item
            }
        }
    }
}

function Resolve-ContainedOutputFile(
    [string]$OutputRoot,
    [string]$Path,
    [string]$Description,
    [bool]$RequireExisting = $true
) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Missing ${Description} path."
    }
    Assert-SafePathSyntax $OutputRoot "output root"
    Assert-SafePathSyntax $Path $Description
    $outputRootFull = [IO.Path]::GetFullPath($OutputRoot)
    $target = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $outputRootFull $Path))
    }
    if (-not (Test-PathWithin $target $outputRootFull) `
        -or $target.Equals($outputRootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "${Description} path escapes the output directory: $Path"
    }
    if ($RequireExisting) {
        Require-File $target $Description
    }
    elseif (Test-Path -LiteralPath $target) {
        Require-File $target $Description
    }
    else {
        Assert-NoReparsePointInExistingPath $target $Description
    }
    return $target
}

function Read-NormalizedJson([string]$Path) {
    $json = ((& $pythonExe $normalizer $Path) -join "`n")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not normalize JSON evidence: $Path"
    }
    return $json | ConvertFrom-Json
}

function Assert-StandardModelContract([string]$ModelPath, [string]$ExpectedKind) {
    Assert-SafePathSyntax $ModelPath "$ExpectedKind ONNX"
    $contractPath = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    Require-File $ModelPath "$ExpectedKind ONNX"
    Require-File $contractPath "$ExpectedKind contract"
    Assert-NoReparsePointInExistingPath $ModelPath "$ExpectedKind ONNX"
    Assert-NoReparsePointInExistingPath $contractPath "$ExpectedKind contract"
    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$contract.kind -ne $ExpectedKind) {
        throw "Unexpected model kind in ${contractPath}: $($contract.kind); expected $ExpectedKind"
    }
    $expectedHash = [string]$contract.onnx.sha256
    $actualHash = Get-Sha256 $ModelPath
    if ([string]::IsNullOrWhiteSpace($expectedHash) -or $expectedHash.ToLowerInvariant() -ne $actualHash) {
        throw "ONNX SHA-256 does not match its contract: $ModelPath"
    }
    return $contractPath
}

function Assert-UnifiedBundle([string]$ModelPath) {
    Assert-SafePathSyntax $ModelPath "unified OCR ONNX"
    $labelsPath = [IO.Path]::ChangeExtension($ModelPath, ".labels.json")
    $contractPath = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    Require-File $ModelPath "unified OCR ONNX"
    Require-File $labelsPath "unified OCR labels"
    Require-File $contractPath "unified OCR contract"
    Assert-NoReparsePointInExistingPath $ModelPath "unified OCR ONNX"
    Assert-NoReparsePointInExistingPath $labelsPath "unified OCR labels"
    Assert-NoReparsePointInExistingPath $contractPath "unified OCR contract"

    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $artifactKind = [string]$contract.kind
    $architectureVersion = [int]$contract.model.architecture_version
    $expectedKind = switch ($architectureVersion) {
        12 { "receipt_unified_field_reader_v12" }
        13 { "receipt_unified_field_reader_v13" }
        default {
            throw "Unified OCR contract has an unsupported architecture_version: $architectureVersion"
        }
    }
    if ($artifactKind -ne $expectedKind) {
        throw "Unified OCR contract kind/version mismatch: kind=$artifactKind architecture_version=$architectureVersion"
    }
    if ([string]$contract.onnx_file -ne [IO.Path]::GetFileName($ModelPath)) {
        throw "Unified OCR contract onnx_file does not match the delivered filename."
    }
    if ([string]$contract.labels_file -ne [IO.Path]::GetFileName($labelsPath)) {
        throw "Unified OCR contract labels_file does not match the delivered filename."
    }
    if ([string]$contract.onnx_sha256 -ne (Get-Sha256 $ModelPath)) {
        throw "Unified OCR ONNX SHA-256 does not match its contract."
    }
    if ([string]$contract.labels_sha256 -ne (Get-Sha256 $labelsPath)) {
        throw "Unified OCR labels SHA-256 does not match its contract."
    }
    $statusTextOutputProperty = if ($null -eq $contract.outputs) {
        $null
    }
    else {
        $contract.outputs.PSObject.Properties["status_text_logits"]
    }
    $statusTextDeliveryPolicy = $null
    $statusTextReviewValue = $null
    if ($architectureVersion -eq 13) {
        if ($null -eq $statusTextOutputProperty `
            -or $null -eq $statusTextOutputProperty.Value `
            -or [string]$statusTextOutputProperty.Value.runtime_policy -ne "decode_and_normalize_review_only" `
            -or [string]$statusTextOutputProperty.Value.review_value -ne "review") {
            throw "Unified OCR v13 status-text output is not decode-and-normalize review-only."
        }
        $statusTextDeliveryPolicy = [string]$statusTextOutputProperty.Value.runtime_policy
        $statusTextReviewValue = [string]$statusTextOutputProperty.Value.review_value
    }
    elseif ($null -ne $statusTextOutputProperty) {
        throw "Unified OCR v12 contract must not declare the v13 status_text_logits output."
    }
    return [pscustomobject]@{
        LabelsPath = $labelsPath
        ContractPath = $contractPath
        Kind = $artifactKind
        ArchitectureVersion = $architectureVersion
        StatusTextDeliveryPolicy = $statusTextDeliveryPolicy
        StatusTextReviewValue = $statusTextReviewValue
    }
}

$hasRecords = -not [string]::IsNullOrWhiteSpace($Records)
$hasEndToEndEvaluationDir = -not [string]::IsNullOrWhiteSpace($EndToEndEvaluationDir)
$includeProductionCpuEntrypoints = $hasRecords -and $RuntimeFlavor -eq "cpu"
$requestedRecordsSha256 = $null
$hasExplicitUnifiedModel = -not [string]::IsNullOrWhiteSpace($UnifiedModelPath)
$hasExplicitOnnxValidationSummary = -not [string]::IsNullOrWhiteSpace($OnnxValidationSummaryPath)
$usesExplicitUnifiedArtifactBinding = $hasExplicitUnifiedModel -and $hasExplicitOnnxValidationSummary
$minimumAmountFloor = 0.7885
$minimumTimeFloor = 0.9840
$minimumPaymentFloor = 0.9325
$minimumRecipientFloor = 0.90
$requiredStatusTextFloor = 0.90

if ($hasExplicitUnifiedModel -ne $hasExplicitOnnxValidationSummary) {
    throw "Supply -UnifiedModelPath and -OnnxValidationSummaryPath together, or omit both for the legacy run layout."
}
if ($AmountFloor -lt $minimumAmountFloor `
    -or $TimeFloor -lt $minimumTimeFloor `
    -or $PaymentFloor -lt $minimumPaymentFloor `
    -or $RecipientFloor -lt $minimumRecipientFloor) {
    throw "Delivery floors may be raised but must not be lower than amount=78.85%, time=98.40%, payment=93.25%, recipient=90%."
}

if ([string]::IsNullOrWhiteSpace($InputPath) -eq [string]::IsNullOrWhiteSpace($InputList)) {
    throw "Specify exactly one of -Input or -InputList."
}
if ($hasRecords -ne $hasEndToEndEvaluationDir) {
    throw "Specify -Records and -EndToEndEvaluationDir together, or omit both for candidate smoke only."
}
if ($hasRecords -and [string]::IsNullOrWhiteSpace($InputList)) {
    throw "End-to-end scoring requires -InputList prepared from the same records."
}
if ($hasRecords -and $Limit -ne 0) {
    throw "Formal end-to-end scoring requires the complete val input list; -Limit is smoke-only."
}
if ($hasRecords -and $RuntimeFlavor -ne "cpu") {
    throw "Formal end-to-end delivery validation requires -RuntimeFlavor cpu; GPU is benchmark/smoke only."
}
if ($hasRecords -and $Rectification -ne "max-side-1600") {
    throw "Formal end-to-end delivery validation requires -Rectification max-side-1600."
}
if ($hasRecords -and -not $IncludeDeviceModel) {
    throw "Formal end-to-end delivery validation requires -IncludeDeviceModel for the complete three-model pipeline."
}
$orientationRule = if ($Rectification -eq "max-side-1600") {
    "exif_upright_landscape_clockwise_90"
} else {
    "none"
}

Assert-SafePathSyntax $RunDirectory "RunDirectory"
Assert-SafePathSyntax $Output "Output"
Assert-SafePathSyntax $DeliveryDir "DeliveryDir"
Assert-SafePathSyntax $DetectorModel "DetectorModel"
Assert-SafePathSyntax $DeviceModel "DeviceModel"
if ($usesExplicitUnifiedArtifactBinding) {
    Assert-SafePathSyntax $UnifiedModelPath "UnifiedModelPath"
    Assert-SafePathSyntax $OnnxValidationSummaryPath "OnnxValidationSummaryPath"
}
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
$Output = [IO.Path]::GetFullPath($Output)
$DeliveryDir = [IO.Path]::GetFullPath($DeliveryDir)
$DetectorModel = [IO.Path]::GetFullPath($DetectorModel)
$DeviceModel = [IO.Path]::GetFullPath($DeviceModel)
Assert-NoReparsePointInExistingPath $RunDirectory "RunDirectory"
Assert-NoReparsePointInExistingPath $Output "Output"
Assert-NoReparsePointInExistingPath $DeliveryDir "DeliveryDir"
Assert-NoReparsePointInExistingPath $DetectorModel "DetectorModel"
Assert-NoReparsePointInExistingPath $DeviceModel "DeviceModel"
if ($usesExplicitUnifiedArtifactBinding) {
    $UnifiedModelPath = [IO.Path]::GetFullPath($UnifiedModelPath)
    $OnnxValidationSummaryPath = [IO.Path]::GetFullPath($OnnxValidationSummaryPath)
    Assert-NoReparsePointInExistingPath $UnifiedModelPath "UnifiedModelPath"
    Assert-NoReparsePointInExistingPath $OnnxValidationSummaryPath "OnnxValidationSummaryPath"
    if (-not (Test-PathWithin $UnifiedModelPath $RunDirectory) `
        -or -not (Test-PathWithin $OnnxValidationSummaryPath $RunDirectory)) {
        throw "Explicit unified model and ONNX validation summary must both be contained by RunDirectory."
    }
}

if ($hasRecords) {
    Assert-SafePathSyntax $Records "Records"
    Assert-SafePathSyntax $EndToEndEvaluationDir "EndToEndEvaluationDir"
    $Records = [IO.Path]::GetFullPath($Records)
    $EndToEndEvaluationDir = [IO.Path]::GetFullPath($EndToEndEvaluationDir)
    Assert-NoReparsePointInExistingPath $Records "Records"
    Assert-NoReparsePointInExistingPath $EndToEndEvaluationDir "EndToEndEvaluationDir"
    if (-not (Test-Path -LiteralPath $Records -PathType Leaf)) {
        throw "Missing end-to-end evaluation records: $Records"
    }
    $requestedRecordsSha256 = Get-Sha256 $Records
    if (-not (Test-Path -LiteralPath $endToEndScorer -PathType Leaf)) {
        throw "Missing ML.NET end-to-end scorer: $endToEndScorer"
    }
    if (Test-Path -LiteralPath $EndToEndEvaluationDir) {
        throw "Refusing to reuse an existing end-to-end evaluation directory: $EndToEndEvaluationDir"
    }
}
if (Test-Path -LiteralPath $DeliveryDir) {
    throw "Refusing to overwrite an existing delivery directory: $DeliveryDir"
}
if (Test-Path -LiteralPath $Output) {
    throw "Refusing to mix validation evidence with an existing output path: $Output"
}

Require-File $pythonExe "project Python interpreter"
Require-File $normalizer "JSON normalizer"
Require-File $projectFile "ML.NET project"
Require-File $preprocessingContractTestProject "ML.NET preprocessing contract test project"
Require-File $rectificationContractTestProject "ML.NET rectification contract test project"
if ($includeProductionCpuEntrypoints) {
    Require-File $singleCpuEntrypoint "single-image production CPU entrypoint"
    Require-File $batchCpuEntrypoint "batch production CPU entrypoint"
    Require-File $cpuDeliveryReadme "production CPU delivery README"
}

$unifiedModel = if ($usesExplicitUnifiedArtifactBinding) {
    $UnifiedModelPath
}
else {
    Join-Path $RunDirectory "best.onnx"
}
$unifiedBundle = Assert-UnifiedBundle $unifiedModel
$unifiedLabels = [string]$unifiedBundle.LabelsPath
$unifiedContract = [string]$unifiedBundle.ContractPath
$unifiedKind = [string]$unifiedBundle.Kind
$unifiedArchitectureVersion = [int]$unifiedBundle.ArchitectureVersion
$statusTextDeliveryPolicy = $unifiedBundle.StatusTextDeliveryPolicy
$statusTextReviewValue = $unifiedBundle.StatusTextReviewValue
$unifiedContractPayload = Get-Content -LiteralPath $unifiedContract -Raw -Encoding UTF8 | ConvertFrom-Json
$textDeliveryPolicy = [string]$unifiedContractPayload.text_delivery_policy.runtime_policy
$textReviewValue = [string]$unifiedContractPayload.text_delivery_policy.review_value
if ($textDeliveryPolicy -ne "review_only_pending_independent_human_truth_calibration" -or $textReviewValue -ne "review") {
    throw "Unified OCR text delivery policy is not the required fail-closed review-only policy."
}
$onnxValidationSummary = if ($usesExplicitUnifiedArtifactBinding) {
    $OnnxValidationSummaryPath
}
else {
    Join-Path $RunDirectory "onnx-val\summary.json"
}
Require-File $onnxValidationSummary "final ONNX validation summary"

$detectorContract = Assert-StandardModelContract $DetectorModel "receipt_lrcnn_v1"
$detectorModelSha256 = Get-Sha256 $DetectorModel
$detectorContractSha256 = Get-Sha256 $detectorContract
$deviceContract = $null
$deviceModelSha256 = $null
$deviceContractSha256 = $null
if ($IncludeDeviceModel) {
    $deviceContract = Assert-StandardModelContract $DeviceModel "statusbar_device_v1"
    $deviceModelSha256 = Get-Sha256 $DeviceModel
    $deviceContractSha256 = Get-Sha256 $deviceContract
}

$summary = Read-NormalizedJson $onnxValidationSummary
$unifiedModelSha256 = Get-Sha256 $unifiedModel
$unifiedLabelsSha256 = Get-Sha256 $unifiedLabels
$unifiedContractSha256 = Get-Sha256 $unifiedContract
if ([string]$summary.model_sha256 -ne $unifiedModelSha256) {
    throw "onnx-val summary model_sha256 does not belong to the selected unified ONNX artifact."
}
$providers = @($summary.providers | ForEach-Object { [string]$_ })
if ($providers.Count -eq 0) {
    throw "onnx-val summary has no execution provider evidence."
}
if ($RuntimeFlavor -eq "gpu" -and $providers -notcontains "CUDAExecutionProvider") {
    throw "GPU smoke requires prior CUDA ONNX evidence: $($providers -join ',')"
}
if ($summary.acceptance.requested -ne $true -or $summary.acceptance.passed -ne $true) {
    throw "onnx-val acceptance was not explicitly requested and passed."
}
if ([string]$summary.evaluation_split -ne "val") {
    throw "onnx-val summary is not bound to the val split."
}
if ($hasRecords) {
    $summaryRecords = [IO.Path]::GetFullPath([string]$summary.records)
    if (-not $summaryRecords.Equals($Records, [StringComparison]::OrdinalIgnoreCase)) {
        throw "onnx-val summary records do not match -Records."
    }
}
$priorFailures = @($summary.acceptance.failures | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($priorFailures.Count -ne 0) {
    throw "onnx-val acceptance contains failures: $($priorFailures -join '; ')"
}

$fieldGates = @(
    @{ Field = "amount"; Floor = $AmountFloor; Acceptance = "min_amount_exact_match" },
    @{ Field = "time"; Floor = $TimeFloor; Acceptance = "min_time_exact_match" },
    @{ Field = "payment_method_field"; Floor = $PaymentFloor; Acceptance = "min_payment_exact_match" },
    @{ Field = "recipient_field"; Floor = $RecipientFloor; Acceptance = "min_recipient_exact_match" }
)
$validatedMetrics = [ordered]@{}
foreach ($gate in $fieldGates) {
    $fieldName = [string]$gate.Field
    $floor = [double]$gate.Floor
    $acceptanceName = [string]$gate.Acceptance
    $metricProperty = $summary.by_field.PSObject.Properties[$fieldName]
    $acceptanceProperty = $summary.acceptance.PSObject.Properties[$acceptanceName]
    if ($null -eq $metricProperty -or $null -eq $acceptanceProperty) {
        throw "onnx-val summary is missing the $fieldName metric or $acceptanceName gate."
    }
    $metric = $metricProperty.Value
    $recordCount = [int]$metric.records
    $exactMatch = [double]$metric.raw_exact_match
    $requestedFloor = [double]$acceptanceProperty.Value
    if ($recordCount -le 0 -or [double]::IsNaN($exactMatch) -or [double]::IsInfinity($exactMatch)) {
        throw "onnx-val $fieldName metric is empty or non-finite."
    }
    if ($requestedFloor -lt $floor) {
        throw "onnx-val $fieldName acceptance floor $requestedFloor is below required floor $floor."
    }
    if ($exactMatch -lt $floor) {
        throw "onnx-val $fieldName exact match $exactMatch is below required floor $floor."
    }
    $validatedMetrics[$fieldName] = [ordered]@{
        exact_matches = [int]$metric.raw_exact_matches
        records = $recordCount
        exact_match = $exactMatch
        required_floor = $floor
        requested_floor = $requestedFloor
    }
}

$guardedValidationEvidencePath = $null
$guardedValidationEvidenceSha256 = $null
$guardedTestSummaryPath = $null
$guardedTestSummarySha256 = $null

# v13 adds visible transfer-status CTC.  Its independent held-out exact-match
# evidence is part of the delivery gate, not merely diagnostic output.  Keep
# the 90% floor fixed here so a direct-artifact packaging command cannot omit
# or weaken the status OCR validation performed by the v13 training wrapper.
if ($unifiedArchitectureVersion -eq 13) {
    $v13SummaryRecordsPath = [string]$summary.records
    Assert-SafePathSyntax $v13SummaryRecordsPath "v13 onnx-val records"
    $v13SummaryRecordsPath = [IO.Path]::GetFullPath($v13SummaryRecordsPath)
    Require-File $v13SummaryRecordsPath "v13 onnx-val records"
    if (-not (Test-PathWithin $v13SummaryRecordsPath $RunDirectory) `
        -or [string]$summary.records_sha256 -ne (Get-Sha256 $v13SummaryRecordsPath)) {
        throw "v13 onnx-val records path or SHA-256 does not match the current run manifest."
    }
    $statusMetricProperty = $summary.by_field.PSObject.Properties["transfer_status"]
    $statusAcceptanceProperty = $summary.acceptance.PSObject.Properties["min_status_exact_match"]
    if ($null -eq $statusMetricProperty `
        -or $null -eq $statusMetricProperty.Value `
        -or $null -eq $statusAcceptanceProperty `
        -or $null -eq $statusAcceptanceProperty.Value) {
        throw "v13 onnx-val summary is missing visible transfer-status CTC metrics or its acceptance gate."
    }
    $statusMetric = $statusMetricProperty.Value
    $statusCtcRecords = [int]$statusMetric.ctc_records
    $statusCtcExactMatches = [int]$statusMetric.ctc_raw_exact_matches
    $statusCtcExactMatch = [double]$statusMetric.ctc_raw_exact_match
    $requestedStatusFloor = [double]$statusAcceptanceProperty.Value
    if ($statusCtcRecords -le 0 `
        -or $statusCtcRecords -ne [int]$statusMetric.records `
        -or $statusCtcExactMatches -lt 0 `
        -or $statusCtcExactMatches -gt $statusCtcRecords `
        -or [Math]::Abs(
            $statusCtcExactMatch - ([double]$statusCtcExactMatches / [double]$statusCtcRecords)
        ) -gt 0.000000000001 `
        -or [double]::IsNaN($statusCtcExactMatch) `
        -or [double]::IsInfinity($statusCtcExactMatch) `
        -or $requestedStatusFloor -lt $requiredStatusTextFloor `
        -or $statusCtcExactMatch -lt $requiredStatusTextFloor) {
        throw "v13 onnx-val visible transfer-status CTC did not meet the fixed 90% exact-match floor."
    }
    if ($null -eq $summary.status_text_policy `
        -or [string]$summary.status_text_policy.runtime_policy -ne "decode_and_normalize_review_only" `
        -or [string]$summary.status_text_policy.review_value -ne "review") {
        throw "v13 onnx-val status-text policy is not decode-and-normalize review-only."
    }
    $validatedMetrics["transfer_status"] = [ordered]@{
        exact_matches = $statusCtcExactMatches
        records = $statusCtcRecords
        exact_match = $statusCtcExactMatch
        metric = "ctc_raw_exact_match"
        required_floor = $requiredStatusTextFloor
        requested_floor = $requestedStatusFloor
    }

    # The v13 wrapper writes this only after both independent val and test
    # evaluations pass.  Bind that wrapper evidence to the exact model and val
    # summary selected above, so packaging cannot substitute a different or
    # edited summary merely because it contains the same model hash string.
    $guardedValidationEvidencePath = Join-Path $RunDirectory "v13_status_ocr_validation.json"
    Require-File $guardedValidationEvidencePath "v13 guarded validation evidence"
    $guardedValidationEvidence = Read-NormalizedJson $guardedValidationEvidencePath
    if ([string]$guardedValidationEvidence.kind -ne "receipt_unified_status_text_v13_guarded_validation_v1" `
        -or [string]$guardedValidationEvidence.candidate.kind -ne "receipt_unified_field_reader_v13" `
        -or [int]$guardedValidationEvidence.candidate.architecture_version -ne 13 `
        -or [string]$guardedValidationEvidence.candidate.model_sha256 -ne $unifiedModelSha256) {
        throw "v13 guarded validation evidence does not belong to the selected unified model."
    }

    $evidenceModelPath = [string]$guardedValidationEvidence.candidate.model
    $evidenceContractPath = [string]$guardedValidationEvidence.candidate.contract
    $evidenceLabelsPath = [string]$guardedValidationEvidence.candidate.labels
    $evidenceManifestPath = [string]$guardedValidationEvidence.manifest.records
    Assert-SafePathSyntax $evidenceModelPath "v13 evidence candidate model"
    Assert-SafePathSyntax $evidenceContractPath "v13 evidence candidate contract"
    Assert-SafePathSyntax $evidenceLabelsPath "v13 evidence candidate labels"
    Assert-SafePathSyntax $evidenceManifestPath "v13 evidence manifest"
    $evidenceModelPath = [IO.Path]::GetFullPath($evidenceModelPath)
    $evidenceContractPath = [IO.Path]::GetFullPath($evidenceContractPath)
    $evidenceLabelsPath = [IO.Path]::GetFullPath($evidenceLabelsPath)
    $evidenceManifestPath = [IO.Path]::GetFullPath($evidenceManifestPath)
    if (-not $evidenceModelPath.Equals($unifiedModel, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $evidenceContractPath.Equals($unifiedContract, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $evidenceLabelsPath.Equals($unifiedLabels, [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$guardedValidationEvidence.candidate.contract_sha256 -ne $unifiedContractSha256 `
        -or [string]$guardedValidationEvidence.candidate.labels_sha256 -ne $unifiedLabelsSha256 `
        -or -not $evidenceManifestPath.Equals(
            [IO.Path]::GetFullPath([string]$summary.records),
            [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$guardedValidationEvidence.manifest.records_sha256 -ne `
            (Get-Sha256 $v13SummaryRecordsPath)) {
        throw "v13 guarded validation model, sidecars, or manifest do not match the selected ONNX validation evidence."
    }

    $guardedFloors = $guardedValidationEvidence.acceptance_floors
    if ([double]$guardedFloors.amount -lt $minimumAmountFloor `
        -or [double]$guardedFloors.time -lt $minimumTimeFloor `
        -or [double]$guardedFloors.payment_method_field -lt $minimumPaymentFloor `
        -or [double]$guardedFloors.recipient_field -lt $minimumRecipientFloor `
        -or [double]$guardedFloors.visible_transfer_status_cjk_text -lt $requiredStatusTextFloor) {
        throw "v13 guarded validation evidence weakened a required delivery floor."
    }

    $packagingBinding = $guardedValidationEvidence.cpu_packaging
    $boundModelPath = [string]$packagingBinding.unified_model_path
    $boundSummaryPath = [string]$packagingBinding.onnx_validation_summary_path
    Assert-SafePathSyntax $boundModelPath "v13 packaging evidence model"
    Assert-SafePathSyntax $boundSummaryPath "v13 packaging evidence summary"
    $boundModelPath = [IO.Path]::GetFullPath($boundModelPath)
    $boundSummaryPath = [IO.Path]::GetFullPath($boundSummaryPath)
    if (-not $boundModelPath.Equals($unifiedModel, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $boundSummaryPath.Equals($onnxValidationSummary, [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$packagingBinding.unified_model_sha256 -ne $unifiedModelSha256 `
        -or [string]$packagingBinding.onnx_validation_summary_sha256 -ne (Get-Sha256 $onnxValidationSummary) `
        -or [string]$packagingBinding.required_runtime_flavor -ne "cpu" `
        -or [string]$packagingBinding.required_rectification -ne "max-side-1600" `
        -or $packagingBinding.include_device_model -ne $true) {
        throw "v13 guarded validation packaging binding does not match the requested full CPU pipeline."
    }

    $valEvidence = @(
        $guardedValidationEvidence.evaluations |
            Where-Object { [string]$_.split -eq "val" }
    )
    $testEvidence = @(
        $guardedValidationEvidence.evaluations |
            Where-Object { [string]$_.split -eq "test" }
    )
    if ($valEvidence.Count -ne 1 `
        -or $valEvidence[0].evaluated -ne $true `
        -or $valEvidence[0].accepted -ne $true `
        -or [int]$valEvidence[0].visible_status_records -ne $statusCtcRecords `
        -or [double]$valEvidence[0].status_text_exact_match -ne $statusCtcExactMatch `
        -or [string]$valEvidence[0].summary_sha256 -ne (Get-Sha256 $onnxValidationSummary) `
        -or $testEvidence.Count -ne 1 `
        -or $testEvidence[0].evaluated -ne $true `
        -or $testEvidence[0].accepted -ne $true) {
        throw "v13 guarded validation must contain one accepted val and one accepted test evaluation."
    }
    $valNonSuccessTruthRecords = [int]$valEvidence[0].non_success_truth_records
    $valSafetyCalibrated = $valEvidence[0].non_success_safety_calibrated -eq $true
    $valMaxSafetyProperty = $summary.acceptance.PSObject.Properties["max_non_success_to_success"]
    if ($valSafetyCalibrated -ne ($valNonSuccessTruthRecords -gt 0) `
        -or [int]$valEvidence[0].status_non_success_to_success -ne `
            [int]$statusMetric.non_success_to_success `
        -or ($valNonSuccessTruthRecords -gt 0 `
            -and ($null -eq $valMaxSafetyProperty `
                -or $null -eq $valMaxSafetyProperty.Value `
                -or [int]$valMaxSafetyProperty.Value -ne 0 `
                -or [int]$statusMetric.non_success_to_success -ne 0))) {
        throw "v13 guarded val summary did not preserve the zero non-success-to-success safety line."
    }
    $valEvidenceSummaryPath = [string]$valEvidence[0].summary_path
    $testEvidenceSummaryPath = [string]$testEvidence[0].summary_path
    Assert-SafePathSyntax $valEvidenceSummaryPath "v13 guarded val summary"
    Assert-SafePathSyntax $testEvidenceSummaryPath "v13 guarded test summary"
    $valEvidenceSummaryPath = [IO.Path]::GetFullPath($valEvidenceSummaryPath)
    $testEvidenceSummaryPath = [IO.Path]::GetFullPath($testEvidenceSummaryPath)
    Require-File $testEvidenceSummaryPath "v13 guarded test ONNX summary"
    if (-not (Test-PathWithin $valEvidenceSummaryPath $RunDirectory) `
        -or -not (Test-PathWithin $testEvidenceSummaryPath $RunDirectory) `
        -or -not $valEvidenceSummaryPath.Equals($onnxValidationSummary, [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$testEvidence[0].summary_sha256 -ne (Get-Sha256 $testEvidenceSummaryPath)) {
        throw "v13 guarded val/test summary paths or hashes do not match their evidence."
    }
    $testSummary = Read-NormalizedJson $testEvidenceSummaryPath
    $testFailures = @(
        $testSummary.acceptance.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $testRecordsPath = [string]$testSummary.records
    Assert-SafePathSyntax $testRecordsPath "v13 test summary records"
    $testRecordsPath = [IO.Path]::GetFullPath($testRecordsPath)
    if ([string]$testSummary.model_sha256 -ne $unifiedModelSha256 `
        -or [string]$testSummary.evaluation_split -ne "test" `
        -or [string]$testSummary.records_sha256 -ne (Get-Sha256 $v13SummaryRecordsPath) `
        -or -not $testRecordsPath.Equals($v13SummaryRecordsPath, [StringComparison]::OrdinalIgnoreCase) `
        -or $testSummary.providers -notcontains "CUDAExecutionProvider" `
        -or $testSummary.acceptance.requested -ne $true `
        -or $testSummary.acceptance.passed -ne $true `
        -or $testFailures.Count -ne 0 `
        -or [string]$testSummary.status_text_policy.runtime_policy -ne "decode_and_normalize_review_only" `
        -or [string]$testSummary.status_text_policy.review_value -ne "review") {
        throw "v13 guarded test summary is not a passing, model/records-bound CUDA status-text evaluation."
    }
    $testNonSuccessTruthRecords = [int]$testEvidence[0].non_success_truth_records
    $testSafetyCalibrated = $testEvidence[0].non_success_safety_calibrated -eq $true
    $testMaxSafetyProperty = $testSummary.acceptance.PSObject.Properties["max_non_success_to_success"]
    if ($testSafetyCalibrated -ne ($testNonSuccessTruthRecords -gt 0) `
        -or [int]$testEvidence[0].status_non_success_to_success -ne `
            [int]$testSummary.by_field.transfer_status.non_success_to_success `
        -or ($testNonSuccessTruthRecords -gt 0 `
            -and ($null -eq $testMaxSafetyProperty `
                -or $null -eq $testMaxSafetyProperty.Value `
                -or [int]$testMaxSafetyProperty.Value -ne 0 `
                -or [int]$testSummary.by_field.transfer_status.non_success_to_success -ne 0))) {
        throw "v13 guarded test summary did not preserve the zero non-success-to-success safety line."
    }
    foreach ($testGate in @(
            @{ Field = "amount"; Floor = $minimumAmountFloor; Acceptance = "min_amount_exact_match"; Metric = "raw_exact_match" },
            @{ Field = "time"; Floor = $minimumTimeFloor; Acceptance = "min_time_exact_match"; Metric = "raw_exact_match" },
            @{ Field = "payment_method_field"; Floor = $minimumPaymentFloor; Acceptance = "min_payment_exact_match"; Metric = "raw_exact_match" },
            @{ Field = "recipient_field"; Floor = $minimumRecipientFloor; Acceptance = "min_recipient_exact_match"; Metric = "raw_exact_match" },
            @{ Field = "transfer_status"; Floor = $requiredStatusTextFloor; Acceptance = "min_status_exact_match"; Metric = "ctc_raw_exact_match" }
        )) {
        $testFieldProperty = $testSummary.by_field.PSObject.Properties[[string]$testGate.Field]
        $testFloorProperty = $testSummary.acceptance.PSObject.Properties[[string]$testGate.Acceptance]
        $testMetricProperty = if ($null -eq $testFieldProperty -or $null -eq $testFieldProperty.Value) {
            $null
        }
        else {
            $testFieldProperty.Value.PSObject.Properties[[string]$testGate.Metric]
        }
        if ($null -eq $testFieldProperty `
            -or $null -eq $testFieldProperty.Value `
            -or $null -eq $testFloorProperty `
            -or $null -eq $testFloorProperty.Value `
            -or $null -eq $testMetricProperty `
            -or $null -eq $testMetricProperty.Value) {
            throw "v13 guarded test summary is missing $($testGate.Field) metrics or floor."
        }
        $testExactMatch = [double]$testMetricProperty.Value
        if ([int]$testFieldProperty.Value.records -le 0 `
            -or [double]::IsNaN($testExactMatch) `
            -or [double]::IsInfinity($testExactMatch) `
            -or [double]$testFloorProperty.Value -lt [double]$testGate.Floor `
            -or $testExactMatch -lt [double]$testGate.Floor) {
            throw "v13 guarded test summary did not meet the $($testGate.Field) exact-match floor."
        }
    }
    $testStatusMetric = $testSummary.by_field.transfer_status
    $testStatusCtcRecords = [int]$testStatusMetric.ctc_records
    $testStatusCtcExactMatches = [int]$testStatusMetric.ctc_raw_exact_matches
    $testStatusCtcExactMatch = [double]$testStatusMetric.ctc_raw_exact_match
    if ($testStatusCtcRecords -le 0 `
        -or $testStatusCtcRecords -ne [int]$testStatusMetric.records `
        -or $testStatusCtcRecords -ne [int]$testEvidence[0].visible_status_records `
        -or $testStatusCtcExactMatches -lt 0 `
        -or $testStatusCtcExactMatches -gt $testStatusCtcRecords `
        -or [Math]::Abs(
            $testStatusCtcExactMatch - `
                ([double]$testStatusCtcExactMatches / [double]$testStatusCtcRecords)
        ) -gt 0.000000000001 `
        -or [double]$testEvidence[0].status_text_exact_match -ne $testStatusCtcExactMatch) {
        throw "v13 guarded test status CTC counts or exact-match evidence are inconsistent."
    }
    $guardedTestSummaryPath = $testEvidenceSummaryPath
    $guardedTestSummarySha256 = Get-Sha256 $testEvidenceSummaryPath
    $guardedValidationEvidenceSha256 = Get-Sha256 $guardedValidationEvidencePath
}

$runtimeDevice = if ($RuntimeFlavor -eq "cpu") { "cpu" } else { "cuda:0" }
$requiredRuntimeProvider = if ($RuntimeFlavor -eq "cpu") { "cpu" } else { "cuda:0" }
$torchLib = $null
if ($RuntimeFlavor -eq "gpu") {
    $torchLib = ((& $pythonExe -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))") -join "").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($torchLib)) {
        throw "Could not locate the CUDA PyTorch library directory."
    }
    Require-File (Join-Path $torchLib "cublasLt64_12.dll") "CUDA 12 cublasLt runtime"
    Require-File (Join-Path $torchLib "cudnn64_9.dll") "cuDNN 9 runtime"
    $env:Path = "$torchLib;$env:Path"
}

$resolvedInput = $null
$resolvedInputList = $null
$inputRecords = @()
if (-not [string]::IsNullOrWhiteSpace($InputPath)) {
    Assert-SafePathSyntax $InputPath "Input"
    $resolvedInput = [IO.Path]::GetFullPath($InputPath)
    if (-not (Test-Path -LiteralPath $resolvedInput)) {
        throw "Input does not exist: $resolvedInput"
    }
    Assert-NoReparsePointInExistingPath $resolvedInput "Input"
    $supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if (Test-Path -LiteralPath $resolvedInput -PathType Leaf) {
        if ($supportedExtensions -notcontains [IO.Path]::GetExtension($resolvedInput).ToLowerInvariant()) {
            throw "Input file has an unsupported image extension: $resolvedInput"
        }
        $expectedRecords = 1
    }
    else {
        $availableRecords = @(
            Get-SafeDirectoryFiles $resolvedInput "input image directory" |
                Where-Object { $supportedExtensions -contains $_.Extension.ToLowerInvariant() }
        ).Count
        $expectedRecords = if ($Limit -gt 0) { [Math]::Min($availableRecords, $Limit) } else { $availableRecords }
    }
}
else {
    Assert-SafePathSyntax $InputList "InputList"
    $resolvedInputList = [IO.Path]::GetFullPath($InputList)
    Require-File $resolvedInputList "input list"
    Assert-NoReparsePointInExistingPath $resolvedInputList "InputList"
    $listRoot = Split-Path -Parent $resolvedInputList
    $seenInputRecords = @{}
    $supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
    foreach ($line in Get-Content -LiteralPath $resolvedInputList -Encoding UTF8) {
        $candidate = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($candidate) -or $candidate.StartsWith("#", [StringComparison]::Ordinal)) {
            continue
        }
        Assert-SafePathSyntax $candidate "input-list image"
        if (-not [IO.Path]::IsPathRooted($candidate)) {
            $candidate = Join-Path $listRoot $candidate
        }
        $candidate = [IO.Path]::GetFullPath($candidate)
        Require-File $candidate "input-list image"
        Assert-NoReparsePointInExistingPath $candidate "input-list image"
        if ($supportedExtensions -notcontains [IO.Path]::GetExtension($candidate).ToLowerInvariant()) {
            throw "Input-list file has an unsupported image extension: $candidate"
        }
        if (-not $seenInputRecords.ContainsKey($candidate)) {
            $seenInputRecords[$candidate] = $true
            $inputRecords += $candidate
        }
    }
    if ($Limit -gt 0) {
        $inputRecords = @($inputRecords | Select-Object -First $Limit)
    }
    $expectedRecords = $inputRecords.Count
}
if ($expectedRecords -le 0) {
    throw "No supported validation images were selected."
}
if ((Test-PathWithin $Output $DeliveryDir) -or (Test-PathWithin $DeliveryDir $Output)) {
    throw "Output and DeliveryDir must be separate, non-nested paths."
}
if ($hasRecords) {
    if ((Test-PathWithin $EndToEndEvaluationDir $Output) `
        -or (Test-PathWithin $Output $EndToEndEvaluationDir)) {
        throw "EndToEndEvaluationDir and Output must be separate, non-nested paths."
    }
    if ((Test-PathWithin $EndToEndEvaluationDir $DeliveryDir) -or (Test-PathWithin $DeliveryDir $EndToEndEvaluationDir)) {
        throw "EndToEndEvaluationDir and DeliveryDir must be separate, non-nested paths."
    }
}
if ($null -ne $resolvedInput -and (Test-Path -LiteralPath $resolvedInput -PathType Container)) {
    if ((Test-PathWithin $Output $resolvedInput) -or (Test-PathWithin $DeliveryDir $resolvedInput)) {
        throw "Output and DeliveryDir must be outside the input image directory."
    }
}

$deliveryParent = Split-Path -Parent $DeliveryDir
if ([string]::IsNullOrWhiteSpace($deliveryParent)) {
    throw "DeliveryDir must have a parent directory."
}
New-Item -ItemType Directory -Path $deliveryParent -Force | Out-Null
Assert-NoReparsePointInExistingPath $deliveryParent "delivery parent"
$stagingRoot = Join-Path $deliveryParent (".receipt-mlnet-unified-staging-" + [Guid]::NewGuid().ToString("N"))
$appDirectory = Join-Path $stagingRoot "app"
$modelDirectory = Join-Path $stagingRoot "models"
$unifiedDirectory = Join-Path $modelDirectory "unified"
$evidenceDirectory = Join-Path $stagingRoot "evidence"
$consoleLog = Join-Path $evidenceDirectory "console.log"
$preprocessingContractTestLog = Join-Path $evidenceDirectory "preprocessing-contract-test.log"
$rectificationContractTestLog = Join-Path $evidenceDirectory "rectification-contract-test.log"
$published = $false

try {
    New-Item -ItemType Directory -Path $appDirectory, $modelDirectory, $unifiedDirectory, $evidenceDirectory | Out-Null
    [IO.File]::WriteAllText($consoleLog, "")
    $scoringRecords = $Records
    $recordsSnapshot = $null
    if ($hasRecords) {
        New-Item -ItemType Directory -Path $EndToEndEvaluationDir | Out-Null
        Assert-NoReparsePointInExistingPath $EndToEndEvaluationDir "end-to-end evaluation directory"
        $recordsSnapshot = Join-Path $EndToEndEvaluationDir "bound-unified-fields.jsonl"
        Copy-Item -LiteralPath $Records -Destination $recordsSnapshot
        Require-File $recordsSnapshot "bound end-to-end records snapshot"
        if ((Get-Sha256 $recordsSnapshot) -ne $requestedRecordsSha256) {
            throw "End-to-end records changed while the immutable scoring snapshot was created."
        }
        $scoringRecords = $recordsSnapshot
    }

    $formalExpectedInputList = $null
    if ($hasRecords) {
        $formalExpectedInputList = Join-Path $evidenceDirectory "expected-val-input-list.txt"
        Write-Host "mlnet_unified_prepare_full_val"
        & $pythonExe $endToEndScorer prepare `
            --records $scoringRecords `
            --output $formalExpectedInputList `
            --split val 2>&1 | Tee-Object -FilePath $consoleLog -Append
        $prepareExitCode = $LASTEXITCODE
        if ($prepareExitCode -ne 0) {
            throw "Could not prepare the canonical full-val input list; exit code $prepareExitCode"
        }
        Require-File $formalExpectedInputList "canonical full-val input list"
        $formalExpectedRecords = @()
        $formalExpectedSet = @{}
        $formalListRoot = Split-Path -Parent $formalExpectedInputList
        foreach ($line in Get-Content -LiteralPath $formalExpectedInputList -Encoding UTF8) {
            $candidate = $line.Trim()
            if ([string]::IsNullOrWhiteSpace($candidate) -or $candidate.StartsWith("#", [StringComparison]::Ordinal)) {
                continue
            }
            Assert-SafePathSyntax $candidate "canonical val input"
            if (-not [IO.Path]::IsPathRooted($candidate)) {
                $candidate = Join-Path $formalListRoot $candidate
            }
            $candidate = [IO.Path]::GetFullPath($candidate)
            Require-File $candidate "canonical val input"
            Assert-NoReparsePointInExistingPath $candidate "canonical val input"
            if (-not $formalExpectedSet.ContainsKey($candidate)) {
                $formalExpectedSet[$candidate] = $true
                $formalExpectedRecords += $candidate
            }
        }
        $providedInputSet = @{}
        foreach ($candidate in $inputRecords) {
            $providedInputSet[$candidate] = $true
        }
        $missingValSources = @($formalExpectedRecords | Where-Object { -not $providedInputSet.ContainsKey($_) })
        $extraValSources = @($inputRecords | Where-Object { -not $formalExpectedSet.ContainsKey($_) })
        if ($missingValSources.Count -ne 0 -or $extraValSources.Count -ne 0) {
            throw "InputList is not the canonical complete val source set: missing=$($missingValSources.Count) extra=$($extraValSources.Count)"
        }
        if ($formalExpectedRecords.Count -ne $expectedRecords) {
            throw "InputList count differs from canonical full val: input=$expectedRecords expected=$($formalExpectedRecords.Count)"
        }
    }

    Write-Host "mlnet_unified_publish_$RuntimeFlavor"
    & dotnet restore $projectFile -r win-x64 "-p:OnnxRuntimeFlavor=$RuntimeFlavor"
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet restore failed with exit code $LASTEXITCODE"
    }
    & dotnet publish $projectFile `
        -c Release `
        -r win-x64 `
        --self-contained false `
        "-p:OnnxRuntimeFlavor=$RuntimeFlavor" `
        --no-restore `
        -o $appDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE"
    }

    Write-Host "mlnet_preprocessing_contract_test"
    & dotnet run `
        --project $preprocessingContractTestProject `
        -c Release `
        "-p:OnnxRuntimeFlavor=$RuntimeFlavor" 2>&1 |
        Tee-Object -FilePath $preprocessingContractTestLog |
        Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "ML.NET preprocessing contract test failed with exit code $LASTEXITCODE"
    }

    Write-Host "mlnet_rectification_contract_test"
    & dotnet run `
        --project $rectificationContractTestProject `
        -c Release `
        "-p:OnnxRuntimeFlavor=$RuntimeFlavor" 2>&1 |
        Tee-Object -FilePath $rectificationContractTestLog |
        Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "ML.NET rectification contract test failed with exit code $LASTEXITCODE"
    }

    $deliveryDetector = Join-Path $modelDirectory ([IO.Path]::GetFileName($DetectorModel))
    Copy-Item -LiteralPath $DetectorModel -Destination $deliveryDetector
    Copy-Item -LiteralPath $detectorContract -Destination $modelDirectory
    if ($IncludeDeviceModel) {
        Copy-Item -LiteralPath $DeviceModel -Destination $modelDirectory
        Copy-Item -LiteralPath $deviceContract -Destination $modelDirectory
    }
    $deliveryUnifiedModel = Join-Path $unifiedDirectory ([IO.Path]::GetFileName($unifiedModel))
    Copy-Item -LiteralPath $unifiedModel -Destination $deliveryUnifiedModel
    Copy-Item -LiteralPath $unifiedLabels -Destination $unifiedDirectory
    Copy-Item -LiteralPath $unifiedContract -Destination $unifiedDirectory
    Copy-Item -LiteralPath $onnxValidationSummary -Destination (Join-Path $evidenceDirectory "onnx-validation-summary.json")
    if ($null -ne $guardedValidationEvidencePath) {
        Copy-Item -LiteralPath $guardedValidationEvidencePath -Destination `
            (Join-Path $evidenceDirectory "v13-guarded-validation.json")
        Copy-Item -LiteralPath $guardedTestSummaryPath -Destination `
            (Join-Path $evidenceDirectory "v13-onnx-test-summary.json")
    }
    if ($includeProductionCpuEntrypoints) {
        Copy-Item -LiteralPath $singleCpuEntrypoint -Destination $stagingRoot
        Copy-Item -LiteralPath $batchCpuEntrypoint -Destination $stagingRoot
        Copy-Item -LiteralPath $cpuDeliveryReadme -Destination $stagingRoot
    }

    $executable = Join-Path $appDirectory "ReceiptMlNet.Cli.exe"
    Require-File $executable "published ML.NET executable"
    $deliveryDevice = Join-Path $modelDirectory ([IO.Path]::GetFileName($DeviceModel))

    function Invoke-MlNetValidation {
        $arguments = @(
            "--detector", $deliveryDetector,
            "--ocr", "unified",
            "--ocr-model", $deliveryUnifiedModel,
            "--output", $Output,
            "--device", $runtimeDevice,
            "--rectification", $Rectification,
            "--annotate", $Annotate,
            "--continue-on-error"
        )
        if ($null -ne $resolvedInput) {
            $arguments += @("--input", $resolvedInput)
        }
        else {
            $arguments += @("--input-list", $resolvedInputList)
        }
        if ($IncludeDeviceModel) {
            $arguments += @("--device-model", $deliveryDevice)
        }
        if ($Limit -gt 0) {
            $arguments += @("--limit", [string]$Limit)
        }
        & $executable @arguments 2>&1 | Tee-Object -FilePath $consoleLog -Append
        $inferenceExitCode = $LASTEXITCODE
        if ($inferenceExitCode -ne 0) {
            throw "ML.NET $RuntimeFlavor validation failed with exit code $inferenceExitCode"
        }
    }

    Write-Host "mlnet_unified_${RuntimeFlavor}_validate"
    Invoke-MlNetValidation
    $manifestPath = Join-Path $Output "inference_manifest.json"
    $errorsPath = Join-Path $Output "inference_errors.jsonl"
    $runtimeSummaryPath = Join-Path $Output "inference_summary.json"
    Require-File $manifestPath "ML.NET inference manifest"
    Require-File $errorsPath "ML.NET inference errors"
    Require-File $runtimeSummaryPath "ML.NET inference summary"
    Assert-NoReparsePointInExistingPath $manifestPath "ML.NET inference manifest"
    Assert-NoReparsePointInExistingPath $errorsPath "ML.NET inference errors"
    Assert-NoReparsePointInExistingPath $runtimeSummaryPath "ML.NET inference summary"
    # Windows PowerShell 5.1 emits a JSON top-level array as one pipeline
    # object. Do not wrap the command itself in @(...), which would report a
    # batch of N records as Count=1; retain the decoded array and use its own
    # Count property instead.
    $allManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifestCount = if ($null -eq $allManifest) { 0 } else { [int]$allManifest.Count }
    $runtimeSummary = Get-Content -LiteralPath $runtimeSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $errorText = Get-Content -LiteralPath $errorsPath -Raw -Encoding UTF8
    $errorCount = [int]$runtimeSummary.errors
    $errorsFileEmpty = [string]::IsNullOrWhiteSpace($errorText)
    if (($errorCount -eq 0) -ne $errorsFileEmpty) {
        throw "inference_summary errors and inference_errors.jsonl emptiness disagree."
    }

    $providerMatches = @(Select-String -LiteralPath $consoleLog -Pattern '^Unified OCR ONNX execution provider: (?<provider>[^ ]+)')
    $activeProviders = @(
        $providerMatches |
            ForEach-Object { $_.Matches[0].Groups["provider"].Value } |
            Sort-Object -Unique
    )
    if ($activeProviders.Count -ne 1 -or $activeProviders[0] -ne $requiredRuntimeProvider) {
        throw "Published ML.NET unified OCR did not prove strict $requiredRuntimeProvider execution: $($activeProviders -join ',')"
    }
    $requestedDeviceMatches = @(Select-String -LiteralPath $consoleLog -Pattern ("^Requested ONNX device: " + [regex]::Escape($runtimeDevice) + " "))
    if ($requestedDeviceMatches.Count -eq 0) {
        throw "Published ML.NET validation did not request strict $runtimeDevice execution."
    }

    if ([string]$runtimeSummary.requested_device -ne $runtimeDevice) {
        throw "inference_summary requested_device is not $runtimeDevice."
    }
    if ([string]$runtimeSummary.unified_provider -ne $requiredRuntimeProvider) {
        throw "inference_summary unified_provider is not $requiredRuntimeProvider."
    }
    if ([int]$runtimeSummary.input -ne $expectedRecords) {
        throw "inference_summary input count $($runtimeSummary.input) differs from selected count $expectedRecords."
    }

    $written = @($allManifest | Where-Object { [string]$_.status -eq "written" }).Count
    $skipped = @($allManifest | Where-Object { [string]$_.status -eq "skipped_existing" }).Count
    $unknownStatuses = @($allManifest | Where-Object { [string]$_.status -notin @("written", "skipped_existing") })
    if ($manifestCount + $errorCount -ne $expectedRecords) {
        throw "Validation accounting mismatch: selected=$expectedRecords manifest=$manifestCount errors=$errorCount"
    }
    if ($written -ne $expectedRecords -or $skipped -ne 0 -or $unknownStatuses.Count -ne 0 -or $errorCount -ne 0) {
        throw "Validation was not clean: selected=$expectedRecords written=$written skipped=$skipped errors=$errorCount"
    }
    if ([int]$runtimeSummary.written -ne $written -or [int]$runtimeSummary.skipped -ne $skipped -or [int]$runtimeSummary.errors -ne $errorCount) {
        throw "inference_summary written/skipped/errors do not match manifest evidence."
    }
    if ([int]$runtimeSummary.inference_latency_ms.count -ne $written) {
        throw "inference_summary latency count does not match written results."
    }
    $totalSeconds = [double]$runtimeSummary.total_seconds
    if ([double]::IsNaN($totalSeconds) -or [double]::IsInfinity($totalSeconds) -or $totalSeconds -lt 0.0) {
        throw "inference_summary total_seconds is invalid."
    }
    $runtimeLatencies = [ordered]@{}
    foreach ($latencyName in @("mean", "p50", "p95")) {
        $latencyProperty = $runtimeSummary.inference_latency_ms.PSObject.Properties[$latencyName]
        if ($null -eq $latencyProperty -or $null -eq $latencyProperty.Value) {
            throw "inference_summary is missing inference_latency_ms.$latencyName."
        }
        $latencyValue = [double]$latencyProperty.Value
        if ([double]::IsNaN($latencyValue) -or [double]::IsInfinity($latencyValue) -or $latencyValue -lt 0.0) {
            throw "inference_summary inference_latency_ms.$latencyName is invalid."
        }
        $runtimeLatencies[$latencyName] = $latencyValue
    }
    if ($runtimeLatencies.p95 -lt $runtimeLatencies.p50) {
        throw "inference_summary p95 latency is below p50."
    }

    $candidateComplete = 0
    $candidateByField = [ordered]@{
        amount = 0
        time = 0
        recipient = 0
        payment_method = 0
    }
    $manifestSourceSet = @{}
    $resultEvidenceRows = @()
    foreach ($manifestRecord in $allManifest) {
        Assert-SafePathSyntax ([string]$manifestRecord.source) "inference manifest source"
        $manifestSource = [IO.Path]::GetFullPath([string]$manifestRecord.source)
        Require-File $manifestSource "inference manifest source"
        Assert-NoReparsePointInExistingPath $manifestSource "inference manifest source"
        if ($null -ne $resolvedInput) {
            if ((Test-Path -LiteralPath $resolvedInput -PathType Leaf) `
                -and -not $manifestSource.Equals($resolvedInput, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Inference manifest source does not match the requested input file: $manifestSource"
            }
            if ((Test-Path -LiteralPath $resolvedInput -PathType Container) `
                -and (-not (Test-PathWithin $manifestSource $resolvedInput) `
                    -or $manifestSource.Equals($resolvedInput, [StringComparison]::OrdinalIgnoreCase))) {
                throw "Inference manifest source escapes the requested input directory: $manifestSource"
            }
        }
        if ($manifestSourceSet.ContainsKey($manifestSource)) {
            throw "Inference manifest contains a duplicate source: $manifestSource"
        }
        $manifestSourceSet[$manifestSource] = $true
        $inferenceMs = [double]$manifestRecord.inference_ms
        if ([double]::IsNaN($inferenceMs) -or [double]::IsInfinity($inferenceMs) -or $inferenceMs -lt 0.0) {
            throw "Manifest inference_ms is invalid for source: $manifestSource"
        }
        $resultPath = Resolve-ContainedOutputFile `
            $Output ([string]$manifestRecord.result) "ML.NET receipt result"
        foreach ($annotation in @(
                @{ Property = "annotated_rectified"; Description = "ML.NET rectified annotation" },
                @{ Property = "annotated_original"; Description = "ML.NET original annotation" }
            )) {
            $annotationPropertyName = [string]$annotation.Property
            $annotationProperty = $manifestRecord.PSObject.Properties[$annotationPropertyName]
            $annotationValue = if ($null -eq $annotationProperty) { $null } else { [string]$annotationProperty.Value }
            $null = Resolve-ContainedOutputFile `
                $Output $annotationValue ([string]$annotation.Description) ($Annotate -eq "all")
        }
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $resultSourceProperty = if ($null -eq $result) { $null } else { $result.PSObject.Properties["source"] }
        if ($null -eq $resultSourceProperty `
            -or [string]::IsNullOrWhiteSpace([string]$resultSourceProperty.Value)) {
            throw "Result has no source path: $resultPath"
        }
        Assert-SafePathSyntax ([string]$resultSourceProperty.Value) "result source"
        $resultSource = [IO.Path]::GetFullPath([string]$resultSourceProperty.Value)
        Assert-NoReparsePointInExistingPath $resultSource "result source"
        if (-not $resultSource.Equals($manifestSource, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Result source does not match its inference manifest source: $resultPath"
        }
        if ([string]$result.inference_engine -ne "mlnet") {
            throw "Unexpected inference engine in result: $resultPath"
        }
        if ([string]$result.model_contracts.unified_ocr_model_sha256 -ne $unifiedModelSha256) {
            throw "Result does not reference the delivered unified OCR model: $resultPath"
        }
        if ([string]$result.model_contracts.detector_sha256 -ne $detectorModelSha256 `
            -or [string]$result.model_contracts.detector_contract_sha256 -ne $detectorContractSha256 `
            -or [string]$result.model_contracts.unified_ocr_labels_sha256 -ne $unifiedLabelsSha256 `
            -or [string]$result.model_contracts.unified_ocr_contract_sha256 -ne $unifiedContractSha256) {
            throw "Result contains mixed detector or unified sidecar provenance: $resultPath"
        }
        if ($IncludeDeviceModel) {
            if ([string]$result.model_contracts.device_sha256 -ne $deviceModelSha256 `
                -or [string]$result.model_contracts.device_contract_sha256 -ne $deviceContractSha256 `
                -or $null -eq $result.PSObject.Properties["device"] `
                -or $null -eq $result.device) {
                throw "Result does not prove execution of the delivered device model: $resultPath"
            }
        }
        $geometryProperty = $result.PSObject.Properties["geometry"]
        if ($null -eq $geometryProperty -or $null -eq $geometryProperty.Value) {
            throw "Result has no geometry evidence: $resultPath"
        }
        $geometry = $geometryProperty.Value
        if ([string]$geometry.rectification -ne $Rectification) {
            throw "Result rectification does not match the requested production mode ${Rectification}: $resultPath"
        }
        foreach ($sizeName in @("source_size", "rectified_size")) {
            $sizeProperty = $geometry.PSObject.Properties[$sizeName]
            if ($null -eq $sizeProperty `
                -or $null -eq $sizeProperty.Value `
                -or [int]$sizeProperty.Value.width -le 0 `
                -or [int]$sizeProperty.Value.height -le 0) {
                throw "Result geometry has an invalid ${sizeName}: $resultPath"
            }
        }
        if ($Rectification -eq "max-side-1600") {
            $rotationProperty = $geometry.PSObject.Properties["rotation_degrees"]
            $screenDetectedProperty = $geometry.PSObject.Properties["screen_detected"]
            if ($null -eq $rotationProperty `
                -or $null -eq $rotationProperty.Value `
                -or $null -eq $screenDetectedProperty `
                -or $null -eq $screenDetectedProperty.Value `
                -or $screenDetectedProperty.Value -isnot [bool]) {
                throw "Result geometry omits typed rotation/screen evidence: $resultPath"
            }
            $rectifiedMaximumSide = [Math]::Max(
                [int]$geometry.rectified_size.width,
                [int]$geometry.rectified_size.height)
            $expectedRotationDegrees = if (
                [int]$geometry.source_size.width -gt [int]$geometry.source_size.height
            ) { 90 } else { 0 }
            $expectedWidth = if ($expectedRotationDegrees -eq 90) {
                [int]$geometry.source_size.height
            } else {
                [int]$geometry.source_size.width
            }
            $expectedHeight = if ($expectedRotationDegrees -eq 90) {
                [int]$geometry.source_size.width
            } else {
                [int]$geometry.source_size.height
            }
            $expectedMaximumSide = [Math]::Max($expectedWidth, $expectedHeight)
            if ($expectedMaximumSide -gt 1600) {
                $scale = 1600.0 / $expectedMaximumSide
                $expectedWidth = [Math]::Max(
                    2,
                    [int][Math]::Round($expectedWidth * $scale, [MidpointRounding]::ToEven))
                $expectedHeight = [Math]::Max(
                    2,
                    [int][Math]::Round($expectedHeight * $scale, [MidpointRounding]::ToEven))
            }
            if ($rectifiedMaximumSide -gt 1600 `
                -or [int]$rotationProperty.Value -ne $expectedRotationDegrees `
                -or [bool]$screenDetectedProperty.Value `
                -or [int]$geometry.rectified_size.width -ne $expectedWidth `
                -or [int]$geometry.rectified_size.height -ne $expectedHeight) {
                throw "Result geometry is not the portrait-oriented fail-closed max-side-1600 full-image contract: $resultPath"
            }
            foreach ($matrixName in @("H_original_to_rectified", "H_rectified_to_original")) {
                $matrixProperty = $geometry.PSObject.Properties[$matrixName]
                if ($null -eq $matrixProperty -or @($matrixProperty.Value).Count -ne 3) {
                    throw "Result geometry is missing a 3x3 ${matrixName}: $resultPath"
                }
                foreach ($matrixRow in @($matrixProperty.Value)) {
                    if (@($matrixRow).Count -ne 3) {
                        throw "Result geometry has a malformed ${matrixName}: $resultPath"
                    }
                    foreach ($matrixValue in @($matrixRow)) {
                        if ($null -eq $matrixValue) {
                            throw "Result geometry has a null ${matrixName} value: $resultPath"
                        }
                        $numericMatrixValue = [double]$matrixValue
                        if ([double]::IsNaN($numericMatrixValue) `
                            -or [double]::IsInfinity($numericMatrixValue)) {
                            throw "Result geometry has a non-finite ${matrixName} value: $resultPath"
                        }
                    }
                }
            }
            $forwardMatrix = @($geometry.PSObject.Properties["H_original_to_rectified"].Value)
            $inverseMatrix = @($geometry.PSObject.Properties["H_rectified_to_original"].Value)
            for ($matrixRowIndex = 0; $matrixRowIndex -lt 3; $matrixRowIndex++) {
                for ($matrixColumnIndex = 0; $matrixColumnIndex -lt 3; $matrixColumnIndex++) {
                    $matrixProduct = 0.0
                    for ($matrixInnerIndex = 0; $matrixInnerIndex -lt 3; $matrixInnerIndex++) {
                        $matrixProduct += `
                            [double]$forwardMatrix[$matrixRowIndex][$matrixInnerIndex] * `
                            [double]$inverseMatrix[$matrixInnerIndex][$matrixColumnIndex]
                    }
                    $expectedMatrixProduct = if ($matrixRowIndex -eq $matrixColumnIndex) { 1.0 } else { 0.0 }
                    if ([Math]::Abs($matrixProduct - $expectedMatrixProduct) -gt 0.0001) {
                        throw "Result geometry homographies are not mutual inverses: $resultPath"
                    }
                }
            }
        }
        $receiptCandidateComplete = $true
        foreach ($fieldName in @("amount", "time", "recipient", "payment_method")) {
            $field = $result.fields.PSObject.Properties[$fieldName]
            if ($null -eq $field) {
                throw "Result has no $fieldName field object: $resultPath"
            }
            if ([string]$field.Value.delivery_policy -ne $textDeliveryPolicy) {
                throw "Result $fieldName has the wrong delivery policy: $resultPath"
            }
            $candidateProperty = $field.Value.PSObject.Properties["candidate"]
            $candidate = if ($null -eq $candidateProperty) { $null } else { [string]$candidateProperty.Value }
            $valueProperty = $field.Value.PSObject.Properties["value"]
            $fieldValue = if ($null -eq $valueProperty) { $null } else { $valueProperty.Value }
            $deliveryValueProperty = $field.Value.PSObject.Properties["delivery_value"]
            $fieldDeliveryValue = if ($null -eq $deliveryValueProperty) { $null } else { $deliveryValueProperty.Value }
            if ([string]::IsNullOrWhiteSpace($candidate)) {
                $receiptCandidateComplete = $false
                if ([string]$field.Value.state -notin @("absent", "unreadable") `
                    -or ($null -ne $fieldValue -and [string]$fieldValue -ne $textReviewValue) `
                    -or ($null -ne $fieldDeliveryValue -and [string]$fieldDeliveryValue -ne $textReviewValue)) {
                    throw "Result $fieldName has an invalid fail-closed missing-candidate state: $resultPath"
                }
                continue
            }
            $candidateByField[$fieldName]++
            if ([string]$fieldDeliveryValue -ne $textReviewValue `
                -or [string]$fieldValue -ne $textReviewValue `
                -or [string]$field.Value.state -ne "review") {
                throw "Result $fieldName candidate escaped the required review-only policy: $resultPath"
            }
        }
        if ($unifiedArchitectureVersion -eq 13) {
            $statusProperty = $result.fields.PSObject.Properties["transfer_status"]
            if ($null -eq $statusProperty -or $null -eq $statusProperty.Value) {
                throw "Result has no transfer_status field object for unified OCR v13: $resultPath"
            }
            $statusField = $statusProperty.Value
            $statusValueProperty = $statusField.PSObject.Properties["value"]
            $statusValue = if ($null -eq $statusValueProperty) { $null } else { $statusValueProperty.Value }
            $statusDeliveryValueProperty = $statusField.PSObject.Properties["delivery_value"]
            $statusRawProperty = $statusField.PSObject.Properties["raw"]
            $statusCandidateProperty = $statusField.PSObject.Properties["candidate"]
            $statusCtcCandidateProperty = $statusField.PSObject.Properties["ctc_candidate"]
            $statusNormalizedProperty = $statusField.PSObject.Properties["normalized"]
            $statusDeliveryValue = if ($null -eq $statusDeliveryValueProperty) {
                $null
            }
            else {
                $statusDeliveryValueProperty.Value
            }
            if ([string]$statusField.delivery_policy -ne [string]$statusTextDeliveryPolicy) {
                throw "Result transfer_status has the wrong v13 status-text delivery policy: $resultPath"
            }
            if ([string]$statusField.state -eq "absent") {
                throw "Result transfer_status is absent; formal v13 delivery requires visible OCR text: $resultPath"
            }
            else {
                $statusRaw = if ($null -eq $statusRawProperty) { "" } else { [string]$statusRawProperty.Value }
                $statusCandidate = if ($null -eq $statusCandidateProperty) { "" } else { [string]$statusCandidateProperty.Value }
                $statusCtcCandidate = if ($null -eq $statusCtcCandidateProperty) { "" } else { [string]$statusCtcCandidateProperty.Value }
                $statusNormalized = if ($null -eq $statusNormalizedProperty) { "" } else { [string]$statusNormalizedProperty.Value }
                if ([string]::IsNullOrWhiteSpace($statusRaw) `
                    -or $statusRaw -ne $statusCandidate `
                    -or $statusRaw -ne $statusCtcCandidate `
                    -or [string]::IsNullOrWhiteSpace($statusNormalized) `
                    -or $statusNormalized -ne (Get-NormalizedTransferStatus $statusRaw)) {
                    throw "Result transfer_status has incomplete or inconsistent v13 OCR text evidence: $resultPath"
                }
                if ([string]$statusField.state -ne "review" `
                    -or [string]$statusValue -ne [string]$statusTextReviewValue `
                    -or [string]$statusDeliveryValue -ne [string]$statusTextReviewValue) {
                    throw "Result transfer_status escaped the v13 review-only delivery policy: $resultPath"
                }
            }
        }
        $resultEvidenceRows += [ordered]@{
            source = $manifestSource
            result = [IO.Path]::GetFullPath($resultPath)
            result_sha256 = Get-Sha256 $resultPath
            result_bytes = (Get-Item -LiteralPath $resultPath).Length
        }
        if ($receiptCandidateComplete) {
            $candidateComplete++
        }
    }

    if ($null -ne $resolvedInputList) {
        $selectedInputSet = @{}
        foreach ($candidate in $inputRecords) {
            $selectedInputSet[$candidate] = $true
        }
        $missingManifestSources = @($inputRecords | Where-Object { -not $manifestSourceSet.ContainsKey($_) })
        $extraManifestSources = @($manifestSourceSet.Keys | Where-Object { -not $selectedInputSet.ContainsKey($_) })
        if ($missingManifestSources.Count -ne 0 -or $extraManifestSources.Count -ne 0) {
            throw "Manifest source set differs from InputList: missing=$($missingManifestSources.Count) extra=$($extraManifestSources.Count)"
        }
    }

    $endToEndSummaryPath = $null
    $endToEndComparisonsPath = $null
    if ($hasRecords) {
        Write-Host "mlnet_unified_end_to_end_score"
        $scoreArguments = @(
            "score",
            "--records", $scoringRecords,
            "--results", $Output,
            "--model", $unifiedModel,
            "--output", $EndToEndEvaluationDir,
            "--split", "val",
            "--amount-floor", [Convert]::ToString($AmountFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--time-floor", [Convert]::ToString($TimeFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--payment-floor", [Convert]::ToString($PaymentFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--recipient-floor", [Convert]::ToString($RecipientFloor, [Globalization.CultureInfo]::InvariantCulture)
        )
        if ($unifiedArchitectureVersion -eq 13) {
            $scoreArguments += @(
                "--status-floor",
                [Convert]::ToString($requiredStatusTextFloor, [Globalization.CultureInfo]::InvariantCulture)
            )
        }
        & $pythonExe $endToEndScorer @scoreArguments 2>&1 | Tee-Object -FilePath $consoleLog -Append
        $scoreExitCode = $LASTEXITCODE
        $endToEndSummaryPath = Join-Path $EndToEndEvaluationDir "summary.json"
        $endToEndComparisonsPath = Join-Path $EndToEndEvaluationDir "comparisons.jsonl"
        Require-File $endToEndSummaryPath "ML.NET end-to-end evaluation summary"
        Require-File $endToEndComparisonsPath "ML.NET end-to-end comparisons"
        Assert-NoReparsePointInExistingPath `
            $endToEndSummaryPath "ML.NET end-to-end evaluation summary"
        Assert-NoReparsePointInExistingPath `
            $endToEndComparisonsPath "ML.NET end-to-end comparisons"
        $endToEndSummary = Read-NormalizedJson $endToEndSummaryPath
        $scoreFailures = @($endToEndSummary.failures | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($scoreExitCode -ne 0) {
            throw "ML.NET end-to-end scorer failed with exit code ${scoreExitCode}: $($scoreFailures -join '; ')"
        }
        if ([string]$endToEndSummary.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
            -or [string]$endToEndSummary.evaluation_split -ne "val") {
            throw "ML.NET end-to-end scorer wrote an unexpected summary kind or split."
        }
        if ([string]$endToEndSummary.model_sha256 -ne $unifiedModelSha256) {
            throw "ML.NET end-to-end score is not bound to the delivered unified ONNX artifact."
        }
        if ([string]$endToEndSummary.records_sha256 -ne $requestedRecordsSha256 `
            -or [string]$endToEndSummary.manifest_sha256 -ne (Get-Sha256 $manifestPath)) {
            throw "ML.NET end-to-end score is not bound to the requested records and inference manifest."
        }
        if ($endToEndSummary.accepted -ne $true -or $endToEndSummary.acceptance.passed -ne $true -or $scoreFailures.Count -ne 0) {
            throw "ML.NET end-to-end score did not pass: $($scoreFailures -join '; ')"
        }
        if ($endToEndSummary.artifact_audit.all_results_match_model -ne $true) {
            throw "ML.NET end-to-end evidence contains missing or mixed model hashes."
        }
        if ([int]$endToEndSummary.coverage.expected_receipts -ne $expectedRecords `
            -or [int]$endToEndSummary.coverage.matched_result_receipts -ne $expectedRecords `
            -or [int]$endToEndSummary.coverage.fully_scored_receipts -ne $expectedRecords) {
            throw "ML.NET end-to-end score does not cover the canonical complete val receipt set."
        }
        $validatedEndToEndMetrics = [ordered]@{}
        foreach ($gate in $fieldGates) {
            $fieldName = [string]$gate.Field
            $floor = [double]$gate.Floor
            $scoreMetricProperty = $endToEndSummary.by_field.PSObject.Properties[$fieldName]
            $scoreFloorProperty = $endToEndSummary.floors.PSObject.Properties[$fieldName]
            if ($null -eq $scoreMetricProperty -or $null -eq $scoreFloorProperty) {
                throw "ML.NET end-to-end score is missing $fieldName metrics or floor."
            }
            $scoreMetric = $scoreMetricProperty.Value
            $scoreExactMatch = [double]$scoreMetric.raw_exact_match
            $scoreCandidateCoverage = [double]$scoreMetric.candidate_coverage
            if ([int]$scoreMetric.records -ne [int]$validatedMetrics[$fieldName].records) {
                throw "ML.NET end-to-end $fieldName records do not match the bound onnx-val summary."
            }
            if ([double]$scoreFloorProperty.Value -lt $floor `
                -or [double]::IsNaN($scoreExactMatch) `
                -or [double]::IsInfinity($scoreExactMatch) `
                -or $scoreExactMatch -lt $floor `
                -or $scoreCandidateCoverage -ne 1.0) {
                throw "ML.NET end-to-end $fieldName did not meet exact-match or candidate-coverage gates."
            }
            $validatedEndToEndMetrics[$fieldName] = [ordered]@{
                exact_matches = [int]$scoreMetric.raw_exact_matches
                records = [int]$scoreMetric.records
                exact_match = $scoreExactMatch
                candidate_coverage = $scoreCandidateCoverage
                required_floor = $floor
            }
        }
        if ($unifiedArchitectureVersion -eq 13) {
            $statusScoreMetricProperty = $endToEndSummary.by_field.PSObject.Properties["transfer_status"]
            $statusScoreFloorProperty = $endToEndSummary.floors.PSObject.Properties["transfer_status"]
            if ($null -eq $statusScoreMetricProperty `
                -or $null -eq $statusScoreMetricProperty.Value `
                -or $null -eq $statusScoreFloorProperty `
                -or $null -eq $statusScoreFloorProperty.Value) {
                throw "ML.NET end-to-end score is missing v13 visible transfer-status metrics or floor."
            }
            $statusScoreMetric = $statusScoreMetricProperty.Value
            $statusScoreExactMatch = [double]$statusScoreMetric.raw_exact_match
            $statusScoreCandidateCoverage = [double]$statusScoreMetric.candidate_coverage
            $statusScoreMaxSafetyProperty = `
                $endToEndSummary.acceptance.PSObject.Properties["max_non_success_to_success"]
            if ([int]$statusScoreMetric.records -ne [int]$validatedMetrics["transfer_status"].records `
                -or [int]$statusScoreMetric.non_success_truth_records -ne $valNonSuccessTruthRecords `
                -or ($statusScoreMetric.non_success_safety_calibrated -eq $true) -ne $valSafetyCalibrated `
                -or [int]$statusScoreMetric.non_success_to_success -ne 0 `
                -or ($valNonSuccessTruthRecords -gt 0 `
                    -and ($null -eq $statusScoreMaxSafetyProperty `
                        -or $null -eq $statusScoreMaxSafetyProperty.Value `
                        -or [int]$statusScoreMaxSafetyProperty.Value -ne 0)) `
                -or [double]$statusScoreFloorProperty.Value -lt $requiredStatusTextFloor `
                -or [double]::IsNaN($statusScoreExactMatch) `
                -or [double]::IsInfinity($statusScoreExactMatch) `
                -or $statusScoreExactMatch -lt $requiredStatusTextFloor `
                -or $statusScoreCandidateCoverage -ne 1.0) {
                throw "ML.NET end-to-end visible transfer-status OCR did not meet exact-match or candidate-coverage gates."
            }
            $validatedEndToEndMetrics["transfer_status"] = [ordered]@{
                exact_matches = [int]$statusScoreMetric.raw_exact_matches
                records = [int]$statusScoreMetric.records
                exact_match = $statusScoreExactMatch
                candidate_coverage = $statusScoreCandidateCoverage
                non_success_truth_records = [int]$statusScoreMetric.non_success_truth_records
                non_success_to_success = [int]$statusScoreMetric.non_success_to_success
                non_success_safety_calibrated = $statusScoreMetric.non_success_safety_calibrated -eq $true
                required_floor = $requiredStatusTextFloor
            }
        }
        $validationScope = "full_val_end_to_end_scored_cpu"
        $endToEndEvidence = [ordered]@{
            performed = $true
            status = "accepted"
            records = $Records
            records_sha256 = $requestedRecordsSha256
            records_snapshot = "evidence/bound-unified-fields.jsonl"
            records_snapshot_sha256 = Get-Sha256 $recordsSnapshot
            evaluation = $EndToEndEvaluationDir
            summary_sha256 = Get-Sha256 $endToEndSummaryPath
            comparisons_sha256 = Get-Sha256 $endToEndComparisonsPath
            manifest_sha256 = Get-Sha256 $manifestPath
            model_sha256 = [string]$endToEndSummary.model_sha256
            expected_receipts = [int]$endToEndSummary.coverage.expected_receipts
            metrics = $validatedEndToEndMetrics
        }
    }
    else {
        $validationScope = "candidate_smoke_only"
        Write-Warning "Records were not supplied: this run can prove $RuntimeFlavor package wiring only and is not a formal end-to-end delivery gate."
        $endToEndEvidence = [ordered]@{
            performed = $false
            status = "candidate_smoke_only"
            reason = "Records and EndToEndEvaluationDir were not supplied; no end-to-end reference scoring was performed."
        }
    }

    $unifiedArtifactEvidence = [ordered]@{
        kind = $unifiedKind
        architecture_version = $unifiedArchitectureVersion
        model_path = "models/unified/$([IO.Path]::GetFileName($unifiedModel))"
        model_sha256 = $unifiedModelSha256
        labels_path = "models/unified/$([IO.Path]::GetFileName($unifiedLabels))"
        labels_sha256 = $unifiedLabelsSha256
        contract_path = "models/unified/$([IO.Path]::GetFileName($unifiedContract))"
        contract_sha256 = $unifiedContractSha256
        text_delivery_policy = $textDeliveryPolicy
        review_value = $textReviewValue
    }
    if ($unifiedArchitectureVersion -eq 13) {
        $unifiedArtifactEvidence["status_text_delivery_policy"] = [string]$statusTextDeliveryPolicy
        $unifiedArtifactEvidence["status_text_review_value"] = [string]$statusTextReviewValue
    }
    $modelArtifactEvidence = [ordered]@{
        detector = [ordered]@{
            kind = "receipt_lrcnn_v1"
            model_path = "models/$([IO.Path]::GetFileName($DetectorModel))"
            model_sha256 = $detectorModelSha256
            contract_path = "models/$([IO.Path]::GetFileName($detectorContract))"
            contract_sha256 = $detectorContractSha256
        }
        device = if ($IncludeDeviceModel) {
            [ordered]@{
                kind = "statusbar_device_v1"
                model_path = "models/$([IO.Path]::GetFileName($DeviceModel))"
                model_sha256 = $deviceModelSha256
                contract_path = "models/$([IO.Path]::GetFileName($deviceContract))"
                contract_sha256 = $deviceContractSha256
            }
        }
        else {
            $null
        }
        unified_ocr = $unifiedArtifactEvidence
    }

    $packageValidation = [ordered]@{
        schema_version = 1
        kind = "receipt_mlnet_unified_package_validation_v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        validation_scope = $validationScope
        input_mode = if ($null -ne $resolvedInput) { "input" } else { "input_list" }
        candidate_complete = $candidateComplete
        candidates_by_field = $candidateByField
        output = $Output
        include_device_model = [bool]$IncludeDeviceModel
        annotate = $Annotate
        model_sha256 = $unifiedModelSha256
        unified_artifact_source = [ordered]@{
            binding = if ($usesExplicitUnifiedArtifactBinding) { "explicit_run_contained" } else { "legacy_run_layout" }
            run_directory = $RunDirectory
            model = $unifiedModel
            model_sha256 = $unifiedModelSha256
            labels = $unifiedLabels
            labels_sha256 = $unifiedLabelsSha256
            contract = $unifiedContract
            contract_sha256 = $unifiedContractSha256
            onnx_validation_summary = $onnxValidationSummary
            onnx_validation_summary_sha256 = Get-Sha256 $onnxValidationSummary
            guarded_validation_evidence = $guardedValidationEvidencePath
            guarded_validation_evidence_sha256 = $guardedValidationEvidenceSha256
            guarded_test_summary = $guardedTestSummaryPath
            guarded_test_summary_sha256 = $guardedTestSummarySha256
        }
        unified_ocr_kind = $unifiedKind
        unified_ocr_architecture_version = $unifiedArchitectureVersion
        model_artifacts = $modelArtifactEvidence
        runtime_flavor = $RuntimeFlavor
        runtime_device = $runtimeDevice
        rectification = $Rectification
        orientation_rule = $orientationRule
        geometry_audit = [ordered]@{
            requested_mode = $Rectification
            orientation_rule = $orientationRule
            checked_results = $written
            matching_results = $written
            matrices_valid = $true
            source_sizes_valid = $true
        }
        contract_tests = [ordered]@{
            preprocessing = [ordered]@{
                status = "passed"
                log_sha256 = Get-Sha256 $preprocessingContractTestLog
            }
            rectification = [ordered]@{
                status = "passed"
                log_sha256 = Get-Sha256 $rectificationContractTestLog
            }
        }
        inference_summary = $runtimeSummary
        end_to_end_evaluation = $endToEndEvidence
        onnx_validation = [ordered]@{
            providers = $providers
            accepted = [bool]$summary.acceptance.passed
            summary_sha256 = Get-Sha256 $onnxValidationSummary
            fields = $validatedMetrics
        }
    }
    if ($unifiedArchitectureVersion -eq 13) {
        $packageValidation["status_text_delivery_policy"] = [string]$statusTextDeliveryPolicy
        $packageValidation["status_text_review_value"] = [string]$statusTextReviewValue
    }
    $packageValidation | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath (Join-Path $evidenceDirectory "package_validation.json") -Encoding UTF8
    ConvertTo-Json -InputObject @($resultEvidenceRows) -Depth 6 |
        Set-Content -LiteralPath (Join-Path $evidenceDirectory "result_evidence_sha256.json") -Encoding UTF8

    Copy-Item -LiteralPath $manifestPath -Destination $evidenceDirectory
    Copy-Item -LiteralPath $errorsPath -Destination $evidenceDirectory
    Copy-Item -LiteralPath $runtimeSummaryPath -Destination $evidenceDirectory
    if ($null -ne $resolvedInputList) {
        Copy-Item -LiteralPath $resolvedInputList -Destination (Join-Path $evidenceDirectory "validation-input-list.txt")
    }
    if ($null -ne $endToEndSummaryPath) {
        Copy-Item -LiteralPath $endToEndSummaryPath -Destination (Join-Path $evidenceDirectory "end-to-end-evaluation-summary.json")
        Copy-Item -LiteralPath $endToEndComparisonsPath -Destination (Join-Path $evidenceDirectory "end-to-end-comparisons.jsonl")
        Copy-Item -LiteralPath $recordsSnapshot -Destination (Join-Path $evidenceDirectory "bound-unified-fields.jsonl")
    }
    $packageConfig = [ordered]@{
        schema_version = 1
        kind = if ($hasRecords) { "receipt_mlnet_unified_delivery_package_v1" } else { "receipt_mlnet_unified_candidate_smoke_package_v1" }
        framework = "net8.0"
        runtime_identifier = "win-x64"
        self_contained = $false
        onnx_runtime_flavor = $RuntimeFlavor
        runtime_device = $runtimeDevice
        rectification = $Rectification
        orientation_rule = $orientationRule
        prerequisites = if ($RuntimeFlavor -eq "cpu") {
            @("Microsoft.NETCore.App 8.x")
        }
        else {
            @("Microsoft.NETCore.App 8.x", "NVIDIA CUDA 12.x", "NVIDIA cuDNN 9.x")
        }
        validation_scope = $validationScope
        run_directory = $RunDirectory
        unified_artifact_source_binding = if ($usesExplicitUnifiedArtifactBinding) {
            "explicit_run_contained"
        }
        else {
            "legacy_run_layout"
        }
        unified_artifact_source_model = $unifiedModel
        onnx_validation_summary_source = $onnxValidationSummary
        guarded_validation_evidence_source = $guardedValidationEvidencePath
        guarded_validation_evidence_sha256 = $guardedValidationEvidenceSha256
        guarded_test_summary_source = $guardedTestSummaryPath
        guarded_test_summary_sha256 = $guardedTestSummarySha256
        input = $resolvedInput
        input_list = $resolvedInputList
        records = if ($hasRecords) { $Records } else { $null }
        records_sha256 = if ($hasRecords) { $requestedRecordsSha256 } else { $null }
        records_snapshot = if ($hasRecords) { "evidence/bound-unified-fields.jsonl" } else { $null }
        end_to_end_evaluation = if ($hasRecords) { $EndToEndEvaluationDir } else { $null }
        limit = $Limit
        detector_model = [IO.Path]::GetFileName($DetectorModel)
        device_model = if ($IncludeDeviceModel) { [IO.Path]::GetFileName($DeviceModel) } else { $null }
        unified_model = "models/unified/$([IO.Path]::GetFileName($unifiedModel))"
        unified_ocr_kind = $unifiedKind
        unified_ocr_architecture_version = $unifiedArchitectureVersion
        text_delivery_policy = $textDeliveryPolicy
        text_review_value = $textReviewValue
        model_artifacts = $modelArtifactEvidence
    }
    if ($unifiedArchitectureVersion -eq 13) {
        $packageConfig["status_text_delivery_policy"] = [string]$statusTextDeliveryPolicy
        $packageConfig["status_text_review_value"] = [string]$statusTextReviewValue
    }
    if ($includeProductionCpuEntrypoints) {
        $packageConfig["production_entrypoints"] = @(
            [IO.Path]::GetFileName($singleCpuEntrypoint),
            [IO.Path]::GetFileName($batchCpuEntrypoint)
        )
        $packageConfig["delivery_readme"] = [IO.Path]::GetFileName($cpuDeliveryReadme)
    }
    $packageConfig | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $evidenceDirectory "package_config.json") -Encoding UTF8

    if ($hasRecords -and (Get-Sha256 $Records) -ne $requestedRecordsSha256) {
        throw "End-to-end records changed during CPU validation; refusing atomic publication."
    }

    $hashRows = @(
        Get-PackagePayloadFiles $stagingRoot |
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
        Set-Content -LiteralPath (Join-Path $stagingRoot "SHA256SUMS.json") -Encoding UTF8
    Assert-PackageIntegrity $stagingRoot

    if (Test-Path -LiteralPath $DeliveryDir) {
        throw "Delivery directory appeared during validation; refusing to overwrite it: $DeliveryDir"
    }
    Assert-NoReparsePointInExistingPath $stagingRoot "staging delivery package"
    Assert-NoReparsePointInExistingPath $DeliveryDir "DeliveryDir"
    [IO.Directory]::Move($stagingRoot, $DeliveryDir)
    $published = $true

    Write-Host "inference_summary"
    Write-Host "  runtime-flavor=$RuntimeFlavor"
    Write-Host "  requested-device=$runtimeDevice"
    Write-Host "  provider=$($activeProviders[0])"
    Write-Host "  selected=$expectedRecords"
    Write-Host "  written=$written"
    Write-Host "  errors=$errorCount"
    Write-Host "  candidate-complete=$candidateComplete"
    Write-Host "  candidates-by-field=$($candidateByField | ConvertTo-Json -Compress)"
    Write-Host "  mean-ms=$($runtimeLatencies.mean)"
    Write-Host "  p50-ms=$($runtimeLatencies.p50)"
    Write-Host "  p95-ms=$($runtimeLatencies.p95)"
    Write-Host "  validation-scope=$validationScope"
    Write-Host "  output=$Output"
    if ($hasRecords) {
        Write-Host "  end-to-end-evaluation=$EndToEndEvaluationDir"
    }
    Write-Host "  delivery=$DeliveryDir"
    Write-Host "  evidence=$(Join-Path $DeliveryDir 'evidence\inference_summary.json')"
    Write-Host "  executable=$(Join-Path $DeliveryDir 'app\ReceiptMlNet.Cli.exe')"
}
finally {
    if (-not $published -and (Test-Path -LiteralPath $stagingRoot)) {
        $cleanupSafe = $true
        try {
            Assert-NoReparsePointInExistingPath $stagingRoot "staging delivery package cleanup"
        }
        catch {
            $cleanupSafe = $false
            Write-Warning "Refusing to recurse into an unsafe staging cleanup path: $($_.Exception.Message)"
        }
        if ($cleanupSafe) {
            Remove-Item -LiteralPath $stagingRoot -Recurse -Force
        }
    }
}
