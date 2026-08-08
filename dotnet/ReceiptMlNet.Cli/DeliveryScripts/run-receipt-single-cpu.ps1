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

function Require-File([string]$Path, [string]$Description) {
    Assert-SafePathSyntax $Path $Description
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
    Assert-NoReparsePointInExistingPath $Path $Description
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-NormalizedTransferStatus([string]$Text) {
    $compact = $Text -replace '\s+', ''
    # Keep this file ASCII so Windows PowerShell 5.1 does not misparse a
    # UTF-8-without-BOM script. .NET Regex expands the Unicode escapes.
    if ($compact -match '\u5931\u8d25|\u672a\u6210\u529f|\u5df2\u64a4\u9500') { return "failed" }
    if ($compact -match '\u5904\u7406\u4e2d|\u5f85\u5904\u7406|\u8fdb\u884c\u4e2d') { return "pending" }
    $successPattern = [regex]'\u8f6c\u8d26\u6210\u529f|\u4ea4\u6613\u6210\u529f|\u4ed8\u6b3e\u6210\u529f|\u652f\u4ed8\u6210\u529f|\u8f6c\u5e10\u6210\u529f'
    if ($successPattern.IsMatch($compact)) {
        if ($compact -match '\u672a|\u4e0d|\u975e|\u65e0|\u5426|\u6ca1|\u6ca1\u6709|\u672a\u80fd|\u4e0d\u662f|\u5e76\u672a|\u5c1a\u672a|\u4e0d\u80fd|\u65e0\u6cd5|\u6ca1\u80fd|\u672a\u66fe|\u4ece\u672a|\u5e76\u975e|\u5417|\u4e48|\u5f85\u786e\u8ba4|\u5f85\u6838\u5b9e|\u672a\u77e5|\u4e0d\u786e\u5b9a|\u7591\u4f3c') { return "unknown" }
        return "success"
    }
    return "unknown"
}

function Assert-CurrentResultSemantics([object]$Result, [string]$ResultPath) {
    $schemaProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["result_schema_version"] }
    $semanticsProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["result_semantics_version"] }
    if ($null -eq $schemaProperty `
        -or (($schemaProperty.Value -isnot [int]) -and ($schemaProperty.Value -isnot [long])) `
        -or [long]$schemaProperty.Value -ne 1 `
        -or $null -eq $semanticsProperty `
        -or $semanticsProperty.Value -isnot [string] `
        -or [string]$semanticsProperty.Value -ne "status-review-only-visible-text-negation-v2") {
        throw "Result uses stale or malformed runtime semantics: $ResultPath"
    }
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

    $segments = @($Path.Split([char[]] @('\', '/')))
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
        $canonicalSegment = $segment.TrimEnd([char[]] @('.', ' '))
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

function Resolve-ContainedPackageDirectory(
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
        -or $target.Equals($rootFull, [StringComparison]::OrdinalIgnoreCase) `
        -or -not (Test-Path -LiteralPath $target -PathType Container)) {
        throw "Missing, unsafe, or non-contained ${Description}: $RelativePath"
    }
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

function Read-StandardModelEvidence(
    [string]$PackageRoot,
    [string]$ModelPath,
    [string]$ExpectedKind,
    [string]$Description
) {
    $contractCandidate = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    $contractRelative = Get-RelativePackagePath $contractCandidate $PackageRoot
    $contractPath = Resolve-ContainedPackageFile $PackageRoot $contractRelative "$Description contract"
    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $onnxProperty = if ($null -eq $contract) { $null } else { $contract.PSObject.Properties["onnx"] }
    $shaProperty = if ($null -eq $onnxProperty -or $null -eq $onnxProperty.Value) {
        $null
    }
    else {
        $onnxProperty.Value.PSObject.Properties["sha256"]
    }
    $modelSha256 = Get-Sha256 $ModelPath
    if ($null -eq $contract `
        -or [string]$contract.kind -ne $ExpectedKind `
        -or $null -eq $shaProperty `
        -or [string]$shaProperty.Value -ne $modelSha256) {
        throw "$Description model and adjacent contract do not form a verified $ExpectedKind bundle."
    }
    return [pscustomobject]@{
        ModelPath = $ModelPath
        ModelFileName = [IO.Path]::GetFileName($ModelPath)
        ModelSha256 = $modelSha256
        ContractPath = $contractPath
        ContractFileName = [IO.Path]::GetFileName($contractPath)
        ContractSha256 = Get-Sha256 $contractPath
        Kind = $ExpectedKind
    }
}

function Read-UnifiedModelEvidence(
    [string]$PackageRoot,
    [string]$ModelPath,
    [string]$RequiredTextPolicy,
    [string]$RequiredReviewValue
) {
    $labelsCandidate = [IO.Path]::ChangeExtension($ModelPath, ".labels.json")
    $contractCandidate = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    $labelsPath = Resolve-ContainedPackageFile `
        $PackageRoot (Get-RelativePackagePath $labelsCandidate $PackageRoot) "unified OCR labels"
    $contractPath = Resolve-ContainedPackageFile `
        $PackageRoot (Get-RelativePackagePath $contractCandidate $PackageRoot) "unified OCR contract"
    $labels = Get-Content -LiteralPath $labelsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $modelSha256 = Get-Sha256 $ModelPath
    $labelsSha256 = Get-Sha256 $labelsPath
    if ($null -eq $labels `
        -or $null -eq $contract `
        -or [int]$labels.schema_version -ne 1 `
        -or [int]$contract.schema_version -ne 1) {
        throw "Unified OCR labels or contract schema is inconsistent."
    }
    $artifactKind = [string]$contract.kind
    $architectureVersion = [int]$contract.model.architecture_version
    $expectedKind = switch ($architectureVersion) {
        12 { "receipt_unified_field_reader_v12" }
        13 { "receipt_unified_field_reader_v13" }
        default {
            throw "Unified OCR contract has an unsupported architecture_version: $architectureVersion"
        }
    }
    if ($artifactKind -ne $expectedKind `
        -or [string]$contract.onnx_file -ne [IO.Path]::GetFileName($ModelPath) `
        -or [string]$contract.labels_file -ne [IO.Path]::GetFileName($labelsPath) `
        -or [string]$contract.onnx_sha256 -ne $modelSha256 `
        -or [string]$contract.labels_sha256 -ne $labelsSha256 `
        -or [string]$contract.text_delivery_policy.runtime_policy -ne $RequiredTextPolicy `
        -or [string]$contract.text_delivery_policy.review_value -ne $RequiredReviewValue) {
        throw "Unified OCR model, labels, contract, or fail-closed text policy is inconsistent."
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
            -or [string]$statusTextOutputProperty.Value.review_value -ne $RequiredReviewValue) {
            throw "Unified OCR v13 status-text output is not decode-and-normalize review-only."
        }
        $statusTextDeliveryPolicy = [string]$statusTextOutputProperty.Value.runtime_policy
        $statusTextReviewValue = [string]$statusTextOutputProperty.Value.review_value
    }
    elseif ($null -ne $statusTextOutputProperty) {
        throw "Unified OCR v12 contract must not declare the v13 status_text_logits output."
    }
    return [pscustomobject]@{
        ModelPath = $ModelPath
        ModelFileName = [IO.Path]::GetFileName($ModelPath)
        ModelSha256 = $modelSha256
        LabelsPath = $labelsPath
        LabelsFileName = [IO.Path]::GetFileName($labelsPath)
        LabelsSha256 = $labelsSha256
        ContractPath = $contractPath
        ContractFileName = [IO.Path]::GetFileName($contractPath)
        ContractSha256 = Get-Sha256 $contractPath
        Kind = $artifactKind
        ArchitectureVersion = $architectureVersion
        TextDeliveryPolicy = $RequiredTextPolicy
        ReviewValue = $RequiredReviewValue
        StatusTextDeliveryPolicy = $statusTextDeliveryPolicy
        StatusTextReviewValue = $statusTextReviewValue
    }
}

function Read-PaddleDeliveryFileEvidence(
    [string]$PackageRoot,
    [string]$BundlePath,
    [object]$Record,
    [string]$Description
) {
    $pathProperty = if ($null -eq $Record) { $null } else { $Record.PSObject.Properties["path"] }
    $shaProperty = if ($null -eq $Record) { $null } else { $Record.PSObject.Properties["sha256"] }
    $bytesProperty = if ($null -eq $Record) { $null } else { $Record.PSObject.Properties["size_bytes"] }
    if ($null -eq $pathProperty -or $null -eq $shaProperty -or $null -eq $bytesProperty) {
        throw "Paddle OCR delivery contract has an incomplete ${Description} record."
    }
    $relativeWithinBundle = [string]$pathProperty.Value
    $bundleRelative = Get-RelativePackagePath $BundlePath $PackageRoot
    $target = Resolve-ContainedPackageFile `
        $PackageRoot ($bundleRelative + "/" + $relativeWithinBundle) $Description
    if (-not (Test-PathWithin $target $BundlePath) `
        -or $target.Equals($BundlePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Paddle OCR ${Description} escapes its contained delivery bundle."
    }
    $expectedSha256 = ([string]$shaProperty.Value).ToLowerInvariant()
    $expectedBytes = [long]0
    $bytesText = [Convert]::ToString($bytesProperty.Value, [Globalization.CultureInfo]::InvariantCulture)
    if ($expectedSha256 -notmatch '^[0-9a-f]{64}$' `
        -or -not [long]::TryParse(
            $bytesText,
            [Globalization.NumberStyles]::Integer,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$expectedBytes) `
        -or $expectedBytes -le 0 `
        -or (Get-Sha256 $target) -ne $expectedSha256 `
        -or (Get-Item -LiteralPath $target).Length -ne $expectedBytes) {
        throw "Paddle OCR ${Description} does not match its delivery contract."
    }
    return [pscustomobject]@{
        Path = $target
        RelativePath = Get-RelativePackagePath $target $PackageRoot
        Sha256 = $expectedSha256
        SizeBytes = $expectedBytes
    }
}

function Read-PaddleRecipientEvidence(
    [string]$PackageRoot,
    [object]$Declaration
) {
    if ($null -eq $Declaration `
        -or [string]$Declaration.kind -ne "paddle_ocr_v2_delivery" `
        -or [string]$Declaration.bundle_path -ne "models/recipient-ppocr" `
        -or [string]$Declaration.contract_path -ne "models/recipient-ppocr/paddle_ocr_delivery.contract.json") {
        throw "Package does not declare the fixed recipient-only PP-OCR delivery bundle."
    }
    $bundlePath = Resolve-ContainedPackageDirectory `
        $PackageRoot ([string]$Declaration.bundle_path) "recipient PP-OCR bundle"
    $contractPath = Resolve-ContainedPackageFile `
        $PackageRoot ([string]$Declaration.contract_path) "recipient PP-OCR delivery contract"
    if (-not [IO.Path]::GetDirectoryName($contractPath).Equals(
            $bundlePath,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Recipient PP-OCR contract is not directly inside its declared bundle."
    }
    $contractSha256 = Get-Sha256 $contractPath
    if ([string]$Declaration.contract_sha256 -ne $contractSha256) {
        throw "Recipient PP-OCR config contract hash does not match the delivered contract."
    }
    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $modelProperties = if ($null -eq $contract -or $null -eq $contract.models) {
        @()
    }
    else {
        @($contract.models.PSObject.Properties)
    }
    if ([int]$contract.schema_version -ne 1 `
        -or [string]$contract.kind -ne "paddle_ocr_v2_delivery" `
        -or $modelProperties.Count -ne 3 `
        -or $null -eq $contract.models.PSObject.Properties["det"] `
        -or $null -eq $contract.models.PSObject.Properties["cls"] `
        -or $null -eq $contract.models.PSObject.Properties["rec"]) {
        throw "Recipient PP-OCR delivery contract is not a complete det/cls/rec v1 bundle."
    }
    $forbiddenDependencies = @(
        $contract.forbidden_runtime_dependencies |
            ForEach-Object { [string]$_ }
    )
    foreach ($dependency in @("Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files")) {
        if ($forbiddenDependencies -notcontains $dependency) {
            throw "Recipient PP-OCR contract does not forbid runtime dependency: $dependency"
        }
    }
    $models = [ordered]@{}
    $packageSizeBytes = [long]0
    foreach ($role in @("det", "cls", "rec")) {
        $models[$role] = Read-PaddleDeliveryFileEvidence `
            $PackageRoot $bundlePath $contract.models.PSObject.Properties[$role].Value `
            "recipient PP-OCR $role model"
        $packageSizeBytes += [long]$models[$role].SizeBytes
    }
    $dictionary = Read-PaddleDeliveryFileEvidence `
        $PackageRoot $bundlePath $contract.dictionary "recipient PP-OCR dictionary"
    $packageSizeBytes += [long]$dictionary.SizeBytes
    $declaredPackageSize = [long]0
    $packageSizeText = [Convert]::ToString(
        $contract.package_size_bytes,
        [Globalization.CultureInfo]::InvariantCulture)
    if (-not [long]::TryParse(
            $packageSizeText,
            [Globalization.NumberStyles]::Integer,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$declaredPackageSize) `
        -or $declaredPackageSize -le 0 `
        -or $declaredPackageSize -ne $packageSizeBytes) {
        throw "Recipient PP-OCR delivery contract package_size_bytes is inconsistent."
    }
    if ((Get-Sha256 $contractPath) -ne $contractSha256) {
        throw "Recipient PP-OCR delivery contract changed during verification."
    }
    $expectedBundlePaths = @{
        "paddle_ocr_delivery.contract.json" = $true
    }
    foreach ($record in @($models.det, $models.cls, $models.rec, $dictionary)) {
        $expectedBundlePaths[(Get-RelativePackagePath $record.Path $bundlePath)] = $true
    }
    $actualBundlePaths = @{}
    foreach ($file in Get-PackagePayloadFiles $bundlePath) {
        $relativePath = Get-RelativePackagePath $file.FullName $bundlePath
        $actualBundlePaths[$relativePath] = $true
    }
    $missingBundlePaths = @(
        $expectedBundlePaths.Keys | Where-Object { -not $actualBundlePaths.ContainsKey($_) }
    )
    $extraBundlePaths = @(
        $actualBundlePaths.Keys | Where-Object { -not $expectedBundlePaths.ContainsKey($_) }
    )
    if ($missingBundlePaths.Count -ne 0 -or $extraBundlePaths.Count -ne 0) {
        throw "Recipient PP-OCR bundle is not closed over its contract, det/cls/rec ONNX files, and dictionary."
    }
    return [pscustomobject]@{
        Kind = "paddle_ocr_v2_delivery"
        BundlePath = $bundlePath
        BundleRelativePath = Get-RelativePackagePath $bundlePath $PackageRoot
        ContractPath = $contractPath
        ContractRelativePath = Get-RelativePackagePath $contractPath $PackageRoot
        ContractFileName = [IO.Path]::GetFileName($contractPath)
        ContractSha256 = $contractSha256
        Models = $models
        Dictionary = $dictionary
        PackageSizeBytes = $packageSizeBytes
    }
}

function Assert-DeclaredPaddleRecipientArtifact(
    [object]$Declaration,
    [object]$PaddleEvidence,
    [string]$Description
) {
    $declaredModelProperties = if ($null -eq $Declaration -or $null -eq $Declaration.models) {
        @()
    }
    else {
        @($Declaration.models.PSObject.Properties)
    }
    if ($null -eq $Declaration `
        -or [string]$Declaration.kind -ne [string]$PaddleEvidence.Kind `
        -or [string]$Declaration.bundle_path -ne [string]$PaddleEvidence.BundleRelativePath `
        -or [string]$Declaration.contract_path -ne [string]$PaddleEvidence.ContractRelativePath `
        -or [string]$Declaration.contract_sha256 -ne [string]$PaddleEvidence.ContractSha256 `
        -or $declaredModelProperties.Count -ne 3) {
        throw "$Description recipient PP-OCR declaration is incomplete or inconsistent."
    }
    foreach ($role in @("det", "cls", "rec")) {
        $declaredProperty = $Declaration.models.PSObject.Properties[$role]
        $expected = $PaddleEvidence.Models[$role]
        if ($null -eq $declaredProperty `
            -or $null -eq $declaredProperty.Value `
            -or [string]$declaredProperty.Value.path -ne [string]$expected.RelativePath `
            -or [string]$declaredProperty.Value.sha256 -ne [string]$expected.Sha256 `
            -or [long]$declaredProperty.Value.size_bytes -ne [long]$expected.SizeBytes) {
            throw "$Description recipient PP-OCR $role declaration does not match the delivered file."
        }
    }
    if ($null -eq $Declaration.dictionary `
        -or [string]$Declaration.dictionary.path -ne [string]$PaddleEvidence.Dictionary.RelativePath `
        -or [string]$Declaration.dictionary.sha256 -ne [string]$PaddleEvidence.Dictionary.Sha256 `
        -or [long]$Declaration.dictionary.size_bytes -ne [long]$PaddleEvidence.Dictionary.SizeBytes) {
        throw "$Description recipient PP-OCR dictionary declaration does not match the delivered file."
    }
}

function Assert-DeclaredModelArtifacts(
    [object]$Declaration,
    [string]$PackageRoot,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
    [object]$PaddleEvidence,
    [string]$Description
) {
    $versionedUnifiedBindingInvalid = $true
    $unifiedDeclarationProperty = if ($null -eq $Declaration) {
        $null
    }
    else {
        $Declaration.PSObject.Properties["unified_ocr"]
    }
    if ($null -ne $unifiedDeclarationProperty -and $null -ne $unifiedDeclarationProperty.Value) {
        $unifiedDeclaration = $unifiedDeclarationProperty.Value
        $architectureProperty = $unifiedDeclaration.PSObject.Properties["architecture_version"]
        $statusPolicyProperty = $unifiedDeclaration.PSObject.Properties["status_text_delivery_policy"]
        $statusReviewProperty = $unifiedDeclaration.PSObject.Properties["status_text_review_value"]
        if ([int]$UnifiedEvidence.ArchitectureVersion -eq 13) {
            $versionedUnifiedBindingInvalid = (
                $null -eq $architectureProperty `
                -or [int]$architectureProperty.Value -ne 13 `
                -or $null -eq $statusPolicyProperty `
                -or [string]$statusPolicyProperty.Value -ne [string]$UnifiedEvidence.StatusTextDeliveryPolicy `
                -or $null -eq $statusReviewProperty `
                -or [string]$statusReviewProperty.Value -ne [string]$UnifiedEvidence.StatusTextReviewValue)
        }
        else {
            # Legacy v12 declarations did not record architecture_version.
            $versionedUnifiedBindingInvalid = (
                ($null -ne $architectureProperty -and [int]$architectureProperty.Value -ne 12) `
                -or $null -ne $statusPolicyProperty `
                -or $null -ne $statusReviewProperty)
        }
    }
    if ($null -eq $Declaration `
        -or [string]$Declaration.detector.kind -ne [string]$DetectorEvidence.Kind `
        -or [string]$Declaration.detector.model_path -ne (Get-RelativePackagePath $DetectorEvidence.ModelPath $PackageRoot) `
        -or [string]$Declaration.detector.model_sha256 -ne [string]$DetectorEvidence.ModelSha256 `
        -or [string]$Declaration.detector.contract_path -ne (Get-RelativePackagePath $DetectorEvidence.ContractPath $PackageRoot) `
        -or [string]$Declaration.detector.contract_sha256 -ne [string]$DetectorEvidence.ContractSha256 `
        -or [string]$Declaration.device.kind -ne [string]$DeviceEvidence.Kind `
        -or [string]$Declaration.device.model_path -ne (Get-RelativePackagePath $DeviceEvidence.ModelPath $PackageRoot) `
        -or [string]$Declaration.device.model_sha256 -ne [string]$DeviceEvidence.ModelSha256 `
        -or [string]$Declaration.device.contract_path -ne (Get-RelativePackagePath $DeviceEvidence.ContractPath $PackageRoot) `
        -or [string]$Declaration.device.contract_sha256 -ne [string]$DeviceEvidence.ContractSha256 `
        -or [string]$Declaration.unified_ocr.kind -ne [string]$UnifiedEvidence.Kind `
        -or [string]$Declaration.unified_ocr.model_path -ne (Get-RelativePackagePath $UnifiedEvidence.ModelPath $PackageRoot) `
        -or [string]$Declaration.unified_ocr.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$Declaration.unified_ocr.labels_path -ne (Get-RelativePackagePath $UnifiedEvidence.LabelsPath $PackageRoot) `
        -or [string]$Declaration.unified_ocr.labels_sha256 -ne [string]$UnifiedEvidence.LabelsSha256 `
        -or [string]$Declaration.unified_ocr.contract_path -ne (Get-RelativePackagePath $UnifiedEvidence.ContractPath $PackageRoot) `
        -or [string]$Declaration.unified_ocr.contract_sha256 -ne [string]$UnifiedEvidence.ContractSha256 `
        -or [string]$Declaration.unified_ocr.text_delivery_policy -ne [string]$UnifiedEvidence.TextDeliveryPolicy `
        -or [string]$Declaration.unified_ocr.review_value -ne [string]$UnifiedEvidence.ReviewValue `
        -or $versionedUnifiedBindingInvalid) {
        throw "$Description model artifact declaration does not match the delivered models and sidecars."
    }
    $paddleDeclaration = $Declaration.PSObject.Properties["recipient_ppocr"]
    if ($null -eq $paddleDeclaration -or $null -eq $paddleDeclaration.Value) {
        throw "$Description does not bind the recipient PP-OCR delivery bundle."
    }
    Assert-DeclaredPaddleRecipientArtifact `
        $paddleDeclaration.Value $PaddleEvidence $Description
}

function Assert-ContainedCliAppClosure(
    [string]$PackageRoot,
    [string]$ManifestPath,
    [string]$ExpectedManifestSha256,
    [int]$ExpectedFileCount
) {
    $appRoot = Resolve-ContainedPackageDirectory $PackageRoot "app" "CLI app closure"
    if ($ExpectedManifestSha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or (Get-Sha256 $ManifestPath) -ne $ExpectedManifestSha256) {
        throw "CLI app closure manifest does not match its lowercase SHA-256 binding."
    }
    $rows = @(Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    if ($rows.Count -le 0 -or $rows.Count -ne $ExpectedFileCount) {
        throw "CLI app closure manifest file_count is empty or inconsistent."
    }
    $listed = @{}
    $requiredBasenames = @{
        "receiptmlnet.cli.exe" = $false
        "receiptmlnet.cli.dll" = $false
        "receiptmlnet.cli.deps.json" = $false
        "receiptmlnet.cli.runtimeconfig.json" = $false
        "microsoft.ml.onnxruntime.dll" = $false
        "onnxruntime.dll" = $false
        "opencvsharp.dll" = $false
        "opencvsharpextern.dll" = $false
    }
    $previousSortKey = $null
    foreach ($row in $rows) {
        $propertyNames = @($row.PSObject.Properties.Name)
        $sizeProperty = $row.PSObject.Properties["size_bytes"]
        if ($propertyNames.Count -ne 3 `
            -or $propertyNames -notcontains "path" `
            -or $propertyNames -notcontains "sha256" `
            -or $null -eq $sizeProperty `
            -or $null -eq $sizeProperty.Value `
            -or (($sizeProperty.Value -isnot [int]) -and ($sizeProperty.Value -isnot [long])) `
            -or [long]$sizeProperty.Value -lt 0 `
            -or [string]$row.sha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw "CLI app closure rows must contain exactly path, lowercase sha256, and non-negative integer size_bytes."
        }
        $target = Resolve-ContainedPackageFile $appRoot ([string]$row.path) "CLI app closure file"
        $relative = Get-RelativePackagePath $target $appRoot
        if ($relative -cne [string]$row.path) {
            throw "CLI app closure path is not canonical: $($row.path)"
        }
        $key = $relative.ToLowerInvariant()
        if ($listed.ContainsKey($key)) {
            throw "Duplicate case-insensitive CLI app closure path: $relative"
        }
        $sortKey = $key + [char]0 + $relative
        if ($null -ne $previousSortKey `
            -and [StringComparer]::Ordinal.Compare($previousSortKey, $sortKey) -ge 0) {
            throw "CLI app closure manifest paths are not canonically sorted."
        }
        $previousSortKey = $sortKey
        $item = Get-Item -LiteralPath $target
        if ([long]$item.Length -ne [long]$sizeProperty.Value `
            -or (Get-Sha256 $target) -ne [string]$row.sha256) {
            throw "CLI app closure file differs from its hash/size binding: $relative"
        }
        $listed[$key] = $relative
        $basename = [IO.Path]::GetFileName($relative).ToLowerInvariant()
        if ($requiredBasenames.ContainsKey($basename)) {
            $requiredBasenames[$basename] = $true
        }
    }
    $actual = @{}
    foreach ($file in @(Get-PackagePayloadFiles $appRoot)) {
        $relative = Get-RelativePackagePath $file.FullName $appRoot
        $key = $relative.ToLowerInvariant()
        if ($actual.ContainsKey($key)) {
            throw "Duplicate case-insensitive delivered CLI app path: $relative"
        }
        $actual[$key] = $relative
    }
    $missing = @($listed.Keys | Where-Object { -not $actual.ContainsKey($_) })
    $extra = @($actual.Keys | Where-Object { -not $listed.ContainsKey($_) })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        throw "Delivered CLI app closure is not exact: missing=$($missing.Count), extra=$($extra.Count)."
    }
    $missingRequired = @($requiredBasenames.Keys | Where-Object { $requiredBasenames[$_] -ne $true })
    if ($missingRequired.Count -ne 0) {
        throw "Delivered CLI app closure lacks required managed/native payload: $($missingRequired -join ',')."
    }
    if ((Get-Sha256 $ManifestPath) -ne $ExpectedManifestSha256) {
        throw "CLI app closure manifest changed during verification."
    }
    return [pscustomobject]@{
        AppRoot = $appRoot
        FileCount = $rows.Count
        ClosureSha256 = $ExpectedManifestSha256
    }
}

function Assert-HybridFormalEvidence(
    [string]$PackageRoot,
    [object]$Config,
    [object]$Validation,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
    [object]$PaddleEvidence
) {
    $requiredRecords = 10016
    $configBindingProperty = $Config.PSObject.Properties["hybrid_ab_evidence"]
    $validationBindingProperty = $Validation.PSObject.Properties["hybrid_formal_ab"]
    if ($null -eq $configBindingProperty `
        -or $null -eq $configBindingProperty.Value `
        -or $null -eq $validationBindingProperty `
        -or $null -eq $validationBindingProperty.Value) {
        throw "Package config and validation must both bind the formal hybrid CPU A/B evidence."
    }
    $configBinding = $configBindingProperty.Value
    $validationBinding = $validationBindingProperty.Value
    foreach ($binding in @($configBinding, $validationBinding)) {
        if ($binding.performed -ne $true `
            -or [string]$binding.status -ne "accepted" `
            -or [string]$binding.mode -ne "formal") {
            throw "Hybrid CPU A/B binding is not an accepted formal run."
        }
    }

    $requiredFiles = @(
        @{
            Key = "ComparisonSummary"
            PathProperty = "comparison_summary"
            HashProperty = "comparison_summary_sha256"
            RelativePath = "evidence/hybrid-formal-ab-summary.json"
        },
        @{
            Key = "ComparisonRows"
            PathProperty = "comparison_comparisons"
            HashProperty = "comparison_comparisons_sha256"
            RelativePath = "evidence/hybrid-formal-ab-comparisons.jsonl"
        },
        @{
            Key = "AccuracySummary"
            PathProperty = "accuracy_summary"
            HashProperty = "accuracy_summary_sha256"
            RelativePath = "evidence/hybrid-formal-accuracy-summary.json"
        },
        @{
            Key = "AccuracyRows"
            PathProperty = "accuracy_comparisons"
            HashProperty = "accuracy_comparisons_sha256"
            RelativePath = "evidence/hybrid-formal-accuracy-comparisons.jsonl"
        },
        @{
            Key = "InputManifest"
            PathProperty = "input_manifest"
            HashProperty = "input_manifest_sha256"
            RelativePath = "evidence/hybrid-formal-fixed-inputs.txt"
        },
        @{
            Key = "BaselineManifest"
            PathProperty = "baseline_inference_manifest"
            HashProperty = "baseline_inference_manifest_sha256"
            RelativePath = "evidence/hybrid-formal-baseline-inference-manifest.json"
        },
        @{
            Key = "HybridManifest"
            PathProperty = "hybrid_inference_manifest"
            HashProperty = "hybrid_inference_manifest_sha256"
            RelativePath = "evidence/hybrid-formal-hybrid-inference-manifest.json"
        },
        @{
            Key = "BaselineRuntimeSummary"
            PathProperty = "baseline_runtime_summary"
            HashProperty = "baseline_runtime_summary_sha256"
            RelativePath = "evidence/hybrid-formal-baseline-inference-summary.json"
        },
        @{
            Key = "HybridRuntimeSummary"
            PathProperty = "hybrid_runtime_summary"
            HashProperty = "hybrid_runtime_summary_sha256"
            RelativePath = "evidence/hybrid-formal-hybrid-inference-summary.json"
        },
        @{
            Key = "CliAppClosure"
            PathProperty = "cli_app_closure_manifest"
            HashProperty = "cli_app_closure_manifest_sha256"
            RelativePath = "evidence/hybrid-formal-cli-app-closure.json"
        }
    )
    $evidencePaths = @{}
    foreach ($file in $requiredFiles) {
        $pathName = [string]$file.PathProperty
        $hashName = [string]$file.HashProperty
        $configPathProperty = $configBinding.PSObject.Properties[$pathName]
        $configHashProperty = $configBinding.PSObject.Properties[$hashName]
        $validationPathProperty = $validationBinding.PSObject.Properties[$pathName]
        $validationHashProperty = $validationBinding.PSObject.Properties[$hashName]
        if ($null -eq $configPathProperty `
            -or $null -eq $configHashProperty `
            -or $null -eq $validationPathProperty `
            -or $null -eq $validationHashProperty `
            -or [string]$configPathProperty.Value -ne [string]$file.RelativePath `
            -or [string]$validationPathProperty.Value -ne [string]$file.RelativePath `
            -or [string]$configHashProperty.Value -cnotmatch '^[0-9a-f]{64}$' `
            -or [string]$validationHashProperty.Value -cne [string]$configHashProperty.Value) {
            throw "Package config/validation formal A/B file bindings disagree or are incomplete."
        }
        $resolvedPath = Resolve-ContainedPackageFile `
            $PackageRoot ([string]$file.RelativePath) "formal hybrid CPU A/B evidence"
        if ((Get-Item -LiteralPath $resolvedPath).Length -le 0 `
            -or (Get-Sha256 $resolvedPath) -cne [string]$configHashProperty.Value) {
            throw "Formal hybrid CPU A/B evidence does not match its contained hash binding."
        }
        $evidencePaths[[string]$file.Key] = $resolvedPath
    }

    $comparisonSummary = Get-Content `
        -LiteralPath $evidencePaths["ComparisonSummary"] -Raw -Encoding UTF8 | ConvertFrom-Json
    $comparisonFailures = @(
        $comparisonSummary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $p95Overhead = [double]$comparisonSummary.cpu.p95_overhead_ms
    $p95Ceiling = [double]$comparisonSummary.cpu.max_p95_overhead_ms
    $baselineP95 = [double]$comparisonSummary.cpu.baseline_inference_latency_ms.p95
    $hybridP95 = [double]$comparisonSummary.cpu.hybrid_inference_latency_ms.p95
    if ([int]$comparisonSummary.schema_version -ne 2 `
        -or [string]$comparisonSummary.kind -ne "receipt_mlnet_hybrid_recipient_cpu_ab_v1" `
        -or [string]$comparisonSummary.evaluation_mode -ne "formal" `
        -or $comparisonSummary.accepted -ne $true `
        -or $comparisonSummary.input_set_identical -ne $true `
        -or $comparisonSummary.cli_summary_counts_verified -ne $true `
        -or $comparisonFailures.Count -ne 0 `
        -or [int]$comparisonSummary.records -ne $requiredRecords `
        -or [int]$comparisonSummary.invariant_records -ne $requiredRecords `
        -or [int]$comparisonSummary.input_set.records -ne $requiredRecords `
        -or [int]$comparisonSummary.input_set.input_manifest.records -ne $requiredRecords `
        -or [int]$comparisonSummary.run_manifests.baseline.records -ne $requiredRecords `
        -or [int]$comparisonSummary.run_manifests.hybrid.records -ne $requiredRecords `
        -or [string]$comparisonSummary.input_set.normalized_source_set_sha256 -notmatch '^[0-9a-f]{64}$' `
        -or [string]$comparisonSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$comparisonSummary.input_set.input_manifest.normalized_source_set_sha256 `
        -or [string]$comparisonSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$comparisonSummary.run_manifests.baseline.normalized_source_set_sha256 `
        -or [string]$comparisonSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$comparisonSummary.run_manifests.hybrid.normalized_source_set_sha256 `
        -or [string]$comparisonSummary.input_set.input_manifest.sha256 -notmatch '^[0-9a-f]{64}$' `
        -or [long]$comparisonSummary.input_set.input_manifest.size_bytes -le 0 `
        -or [string]$comparisonSummary.run_manifests.baseline.sha256 -notmatch '^[0-9a-f]{64}$' `
        -or [long]$comparisonSummary.run_manifests.baseline.size_bytes -le 0 `
        -or [string]$comparisonSummary.run_manifests.hybrid.sha256 -notmatch '^[0-9a-f]{64}$' `
        -or [long]$comparisonSummary.run_manifests.hybrid.size_bytes -le 0 `
        -or [string]$comparisonSummary.cli_build.assembly.sha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or [long]$comparisonSummary.cli_build.assembly.size_bytes -le 0 `
        -or [double]$comparisonSummary.recipient_candidate_coverage -ne 1.0 `
        -or [double]::IsNaN($p95Overhead) `
        -or [double]::IsInfinity($p95Overhead) `
        -or [double]::IsNaN($p95Ceiling) `
        -or [double]::IsInfinity($p95Ceiling) `
        -or [double]::IsNaN($baselineP95) `
        -or [double]::IsInfinity($baselineP95) `
        -or [double]::IsNaN($hybridP95) `
        -or [double]::IsInfinity($hybridP95) `
        -or $p95Ceiling -lt 0.0 `
        -or $p95Ceiling -gt 250.0 `
        -or $p95Overhead -gt $p95Ceiling `
        -or $baselineP95 -lt 0.0 `
        -or $hybridP95 -lt 0.0) {
        throw "Formal hybrid CPU A/B comparison is not a clean 10016-record pass within the fixed p95 ceiling."
    }
    $baselineRuntimePath = $evidencePaths["BaselineRuntimeSummary"]
    $hybridRuntimePath = $evidencePaths["HybridRuntimeSummary"]
    $baselineRuntimeRecord = $comparisonSummary.run_summaries.baseline
    $hybridRuntimeRecord = $comparisonSummary.run_summaries.hybrid
    $baselineRuntimeSizeProperty = $baselineRuntimeRecord.PSObject.Properties["size_bytes"]
    $hybridRuntimeSizeProperty = $hybridRuntimeRecord.PSObject.Properties["size_bytes"]
    $baselineRuntimeItem = Get-Item -LiteralPath $baselineRuntimePath
    $hybridRuntimeItem = Get-Item -LiteralPath $hybridRuntimePath
    if ([string]::IsNullOrWhiteSpace([string]$baselineRuntimeRecord.path) `
        -or [string]::IsNullOrWhiteSpace([string]$hybridRuntimeRecord.path) `
        -or $null -eq $baselineRuntimeSizeProperty `
        -or $null -eq $baselineRuntimeSizeProperty.Value `
        -or (($baselineRuntimeSizeProperty.Value -isnot [int]) `
            -and ($baselineRuntimeSizeProperty.Value -isnot [long])) `
        -or $null -eq $hybridRuntimeSizeProperty `
        -or $null -eq $hybridRuntimeSizeProperty.Value `
        -or (($hybridRuntimeSizeProperty.Value -isnot [int]) `
            -and ($hybridRuntimeSizeProperty.Value -isnot [long])) `
        -or [string]$baselineRuntimeRecord.sha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or [long]$baselineRuntimeSizeProperty.Value -le 0 `
        -or [string]$hybridRuntimeRecord.sha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or [long]$hybridRuntimeSizeProperty.Value -le 0 `
        -or (Get-Sha256 $baselineRuntimePath) -ne [string]$baselineRuntimeRecord.sha256 `
        -or (Get-Sha256 $hybridRuntimePath) -ne [string]$hybridRuntimeRecord.sha256 `
        -or [long]$baselineRuntimeItem.Length -ne [long]$baselineRuntimeRecord.size_bytes `
        -or [long]$hybridRuntimeItem.Length -ne [long]$hybridRuntimeRecord.size_bytes) {
        throw "Formal hybrid CPU A/B runtime summaries do not match their hash/size bindings."
    }
    $baselineRuntime = Get-Content -LiteralPath $baselineRuntimePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $hybridRuntime = Get-Content -LiteralPath $hybridRuntimePath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($runtime in @(
            [pscustomobject]@{ Name = "baseline"; Summary = $baselineRuntime; Paddle = $null },
            [pscustomobject]@{ Name = "hybrid"; Summary = $hybridRuntime; Paddle = "cpu" }
        )) {
        $paddleProviderProperty = $runtime.Summary.PSObject.Properties["paddle_ocr_provider"]
        $runtimeP95 = [double]$runtime.Summary.inference_latency_ms.p95
        if ([string]$runtime.Summary.requested_device -ne "cpu" `
            -or [string]$runtime.Summary.unified_provider -ne "cpu" `
            -or [int]$runtime.Summary.input -ne $requiredRecords `
            -or [int]$runtime.Summary.written -ne $requiredRecords `
            -or [int]$runtime.Summary.skipped -ne 0 `
            -or [int]$runtime.Summary.errors -ne 0 `
            -or [int]$runtime.Summary.inference_latency_ms.count -ne $requiredRecords `
            -or [double]::IsNaN($runtimeP95) `
            -or [double]::IsInfinity($runtimeP95) `
            -or $runtimeP95 -lt 0.0 `
            -or $null -eq $paddleProviderProperty `
            -or ($null -eq $runtime.Paddle -and $null -ne $paddleProviderProperty.Value) `
            -or ($null -ne $runtime.Paddle -and [string]$paddleProviderProperty.Value -ne "cpu")) {
            throw "Formal hybrid CPU A/B $($runtime.Name) runtime summary is not a complete CPU run."
        }
    }
    $rawBaselineP95 = [double]$baselineRuntime.inference_latency_ms.p95
    $rawHybridP95 = [double]$hybridRuntime.inference_latency_ms.p95
    $rawP95Overhead = $rawHybridP95 - $rawBaselineP95
    if ($baselineP95 -ne $rawBaselineP95 `
        -or $hybridP95 -ne $rawHybridP95 `
        -or $p95Overhead -ne $rawP95Overhead `
        -or $rawP95Overhead -gt $p95Ceiling) {
        throw "Formal hybrid CPU A/B p95 values do not equal the contained runtime summaries."
    }

    $appClosure = $comparisonSummary.cli_build.app_closure
    $closureManifestRecord = $appClosure.manifest
    $closureFileCountProperty = $appClosure.PSObject.Properties["file_count"]
    $closureManifestSizeProperty = $closureManifestRecord.PSObject.Properties["size_bytes"]
    $closureManifestPath = $evidencePaths["CliAppClosure"]
    $closureManifestItem = Get-Item -LiteralPath $closureManifestPath
    if ([string]::IsNullOrWhiteSpace([string]$appClosure.root) `
        -or [string]::IsNullOrWhiteSpace([string]$closureManifestRecord.path) `
        -or $null -eq $closureFileCountProperty `
        -or $null -eq $closureFileCountProperty.Value `
        -or (($closureFileCountProperty.Value -isnot [int]) `
            -and ($closureFileCountProperty.Value -isnot [long])) `
        -or $null -eq $closureManifestSizeProperty `
        -or $null -eq $closureManifestSizeProperty.Value `
        -or (($closureManifestSizeProperty.Value -isnot [int]) `
            -and ($closureManifestSizeProperty.Value -isnot [long])) `
        -or [string]$appClosure.closure_sha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or [string]$closureManifestRecord.sha256 -cnotmatch '^[0-9a-f]{64}$' `
        -or [string]$appClosure.closure_sha256 -ne [string]$closureManifestRecord.sha256 `
        -or [string]$appClosure.closure_sha256 -ne (Get-Sha256 $closureManifestPath) `
        -or [long]$closureManifestSizeProperty.Value -ne [long]$closureManifestItem.Length `
        -or [int]$closureFileCountProperty.Value -le 0) {
        throw "Formal hybrid CPU A/B CLI app closure digest, size, or file_count is inconsistent."
    }
    $verifiedClosure = Assert-ContainedCliAppClosure `
        $PackageRoot $closureManifestPath ([string]$appClosure.closure_sha256) `
        ([int]$closureFileCountProperty.Value)
    if ([string]$comparisonSummary.artifact_hashes.detector_sha256 -ne [string]$DetectorEvidence.ModelSha256 `
        -or [string]$comparisonSummary.artifact_hashes.detector_contract_sha256 -ne [string]$DetectorEvidence.ContractSha256 `
        -or [string]$comparisonSummary.artifact_hashes.device_sha256 -ne [string]$DeviceEvidence.ModelSha256 `
        -or [string]$comparisonSummary.artifact_hashes.device_contract_sha256 -ne [string]$DeviceEvidence.ContractSha256 `
        -or [string]$comparisonSummary.artifact_hashes.unified_ocr_model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$comparisonSummary.artifact_hashes.unified_ocr_labels_sha256 -ne [string]$UnifiedEvidence.LabelsSha256 `
        -or [string]$comparisonSummary.artifact_hashes.unified_ocr_contract_sha256 -ne [string]$UnifiedEvidence.ContractSha256 `
        -or [string]$comparisonSummary.paddle_delivery.contract_sha256 -ne [string]$PaddleEvidence.ContractSha256 `
        -or [long]$comparisonSummary.paddle_delivery.package_size_bytes -ne `
            [long]$PaddleEvidence.PackageSizeBytes) {
        throw "Formal hybrid CPU A/B comparison is not bound to every delivered model."
    }
    $fixedInputRows = @(
        Get-Content -LiteralPath $evidencePaths["InputManifest"] -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#") }
    )
    $baselineManifest = Get-Content `
        -LiteralPath $evidencePaths["BaselineManifest"] -Raw -Encoding UTF8 | ConvertFrom-Json
    $hybridManifest = Get-Content `
        -LiteralPath $evidencePaths["HybridManifest"] -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($fixedInputRows.Count -ne $requiredRecords `
        -or @($fixedInputRows | Sort-Object -Unique).Count -ne $requiredRecords `
        -or @($baselineManifest).Count -ne $requiredRecords `
        -or @($hybridManifest).Count -ne $requiredRecords `
        -or (Get-Sha256 $evidencePaths["InputManifest"]) -ne `
            [string]$comparisonSummary.input_set.input_manifest.sha256 `
        -or (Get-Sha256 $evidencePaths["BaselineManifest"]) -ne `
            [string]$comparisonSummary.run_manifests.baseline.sha256 `
        -or (Get-Sha256 $evidencePaths["HybridManifest"]) -ne `
            [string]$comparisonSummary.run_manifests.hybrid.sha256) {
        throw "Formal hybrid CPU A/B source and run manifests are not the contained 10016-record evidence."
    }
    $deliveredAssembly = Resolve-ContainedPackageFile `
        $PackageRoot "app/ReceiptMlNet.Cli.dll" "ReceiptMlNet assembly bound by formal A/B"
    $deliveredAssemblyItem = Get-Item -LiteralPath $deliveredAssembly
    $deliveredAssemblySize = [long]$deliveredAssemblyItem.Length
    if ([string]$comparisonSummary.cli_build.assembly.sha256 -cne (Get-Sha256 $deliveredAssembly) `
        -or [long]$comparisonSummary.cli_build.assembly.size_bytes -ne `
            $deliveredAssemblySize) {
        throw "Formal hybrid CPU A/B comparison was not produced by the delivered CLI build."
    }
    foreach ($role in @("det", "cls", "rec")) {
        $comparisonModel = $comparisonSummary.paddle_delivery.models.PSObject.Properties[$role].Value
        $deliveredModel = $PaddleEvidence.Models[$role]
        if ([string]$comparisonModel.path -ne (Get-RelativePackagePath $deliveredModel.Path $PaddleEvidence.BundlePath) `
            -or [string]$comparisonModel.sha256 -ne [string]$deliveredModel.Sha256 `
            -or [long]$comparisonModel.size_bytes -ne [long]$deliveredModel.SizeBytes) {
            throw "Formal hybrid CPU A/B comparison PP-OCR $role binding is inconsistent."
        }
    }
    if ([string]$comparisonSummary.paddle_delivery.dictionary.path -ne `
            (Get-RelativePackagePath $PaddleEvidence.Dictionary.Path $PaddleEvidence.BundlePath) `
        -or [string]$comparisonSummary.paddle_delivery.dictionary.sha256 -ne `
            [string]$PaddleEvidence.Dictionary.Sha256 `
        -or [long]$comparisonSummary.paddle_delivery.dictionary.size_bytes -ne `
            [long]$PaddleEvidence.Dictionary.SizeBytes) {
        throw "Formal hybrid CPU A/B comparison PP-OCR dictionary binding is inconsistent."
    }

    $accuracySummary = Get-Content `
        -LiteralPath $evidencePaths["AccuracySummary"] -Raw -Encoding UTF8 | ConvertFrom-Json
    $accuracyFailures = @(
        $accuracySummary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $requestedLimitProperty = $accuracySummary.evaluation_scope.PSObject.Properties["requested_limit"]
    $maxStatusSafetyProperty = $accuracySummary.acceptance.PSObject.Properties["max_non_success_to_success"]
    if ([int]$accuracySummary.schema_version -ne 1 `
        -or [int]$accuracySummary.coverage_contract_version -ne 2 `
        -or [string]$accuracySummary.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
        -or [string]$accuracySummary.evaluation_split -ne "val" `
        -or [string]$accuracySummary.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$accuracySummary.manifest_sha256 -ne `
            [string]$comparisonSummary.run_manifests.hybrid.sha256 `
        -or $accuracySummary.formal_delivery_gate -ne $true `
        -or $accuracySummary.accepted -ne $true `
        -or $accuracySummary.acceptance.passed -ne $true `
        -or $accuracySummary.acceptance.formal_delivery_gate -ne $true `
        -or $accuracyFailures.Count -ne 0 `
        -or [string]$accuracySummary.evaluation_scope.kind -ne "full_split" `
        -or $null -eq $requestedLimitProperty `
        -or $null -ne $requestedLimitProperty.Value `
        -or [int]$accuracySummary.evaluation_scope.evaluated_expected_receipts -ne $requiredRecords `
        -or [int]$accuracySummary.evaluation_scope.full_split_expected_receipts -ne $requiredRecords `
        -or $accuracySummary.artifact_audit.all_results_match_model -ne $true `
        -or [int]$accuracySummary.coverage.expected_receipts -ne $requiredRecords `
        -or [int]$accuracySummary.coverage.matched_result_receipts -ne $requiredRecords `
        -or [int]$accuracySummary.coverage.fully_scored_receipts -ne $requiredRecords `
        -or [double]$accuracySummary.coverage.result_coverage -ne 1.0 `
        -or [double]$accuracySummary.coverage.fully_scored_coverage -ne 1.0 `
        -or [int]$accuracySummary.coverage.coverage_contract_version -ne 2 `
        -or [string]$accuracySummary.coverage.candidate_coverage_domain -ne "all_expected_receipts" `
        -or [int]$accuracySummary.coverage.fully_candidate_covered_receipts -ne $requiredRecords `
        -or [double]$accuracySummary.coverage.all_field_candidate_coverage -ne 1.0 `
        -or $accuracySummary.input_selection.hash_bound -ne $true `
        -or [int]$accuracySummary.input_selection.records -ne $requiredRecords `
        -or [string]$accuracySummary.input_selection.sha256 -ne `
            [string]$comparisonSummary.input_set.input_manifest.sha256 `
        -or $accuracySummary.accuracy_denominators.hash_bound -ne $true `
        -or [string]$accuracySummary.accuracy_denominators.source -ne "input_selection.field_reference_counts" `
        -or [string]$accuracySummary.all_receipt_candidate_coverage.scope -ne "all_selected_receipts" `
        -or [int]$accuracySummary.all_receipt_candidate_coverage.expected_receipts -ne $requiredRecords `
        -or [int]$accuracySummary.all_receipt_candidate_coverage.complete_receipts -ne $requiredRecords `
        -or [int]$accuracySummary.all_receipt_candidate_coverage.missing_complete_receipts -ne 0 `
        -or [double]$accuracySummary.all_receipt_candidate_coverage.complete_coverage -ne 1.0 `
        -or $null -eq $maxStatusSafetyProperty `
        -or $null -eq $maxStatusSafetyProperty.Value `
        -or (($maxStatusSafetyProperty.Value -isnot [int]) `
            -and ($maxStatusSafetyProperty.Value -isnot [long])) `
        -or [int]$maxStatusSafetyProperty.Value -ne 0) {
        throw "Formal hybrid CPU A/B accuracy is not a complete accepted 10016-record full-split score."
    }
    $fixedFloors = @(
        @{ Field = "amount"; Floor = 0.7885 },
        @{ Field = "time"; Floor = 0.9840 },
        @{ Field = "payment_method_field"; Floor = 0.9325 },
        @{ Field = "recipient_field"; Floor = 0.90 },
        @{ Field = "transfer_status"; Floor = 0.90 }
    )
    foreach ($gate in $fixedFloors) {
        $fieldName = [string]$gate.Field
        $requiredFloor = [double]$gate.Floor
        $metricProperty = $accuracySummary.by_field.PSObject.Properties[$fieldName]
        $floorProperty = $accuracySummary.floors.PSObject.Properties[$fieldName]
        $referenceCountProperty = `
            $accuracySummary.input_selection.field_reference_counts.PSObject.Properties[$fieldName]
        $denominatorProperty = `
            $accuracySummary.accuracy_denominators.by_field.PSObject.Properties[$fieldName]
        $candidateProperty = `
            $accuracySummary.all_receipt_candidate_coverage.by_field.PSObject.Properties[$fieldName]
        if ($null -eq $metricProperty `
            -or $null -eq $floorProperty `
            -or $null -eq $referenceCountProperty `
            -or $null -eq $denominatorProperty `
            -or $null -eq $candidateProperty `
            -or [double]$floorProperty.Value -ne $requiredFloor `
            -or [int]$referenceCountProperty.Value -le 0 `
            -or [int]$metricProperty.Value.records -ne [int]$referenceCountProperty.Value `
            -or [int]$denominatorProperty.Value -ne [int]$referenceCountProperty.Value `
            -or [int]$candidateProperty.Value.expected_receipts -ne $requiredRecords `
            -or [int]$candidateProperty.Value.candidate_records -ne $requiredRecords `
            -or [int]$candidateProperty.Value.missing_candidate_records -ne 0 `
            -or [double]$candidateProperty.Value.candidate_coverage -ne 1.0 `
            -or [double]$metricProperty.Value.raw_exact_match -lt $requiredFloor) {
            throw "Formal hybrid CPU A/B accuracy did not pass the fixed $fieldName reference denominator/floor with all-receipt candidate coverage."
        }
    }
    if ([int]$accuracySummary.by_field.transfer_status.non_success_to_success -ne 0) {
        throw "Formal hybrid CPU A/B status evidence crossed the zero non-success-to-success safety line."
    }
    $bindingProperties = @(
        "source",
        "records_sha256",
        "expected_receipts",
        "normalized_source_set_sha256",
        "input_manifest_sha256",
        "baseline_inference_manifest_sha256",
        "hybrid_inference_manifest_sha256",
        "score_manifest_sha256",
        "cli_assembly",
        "cli_assembly_sha256",
        "cli_assembly_size_bytes",
        "cli_app_closure_sha256",
        "cli_app_closure_file_count",
        "paddle_contract_sha256",
        "paddle_package_size_bytes",
        "invariant_records",
        "recipient_candidate_coverage",
        "cpu_p95_overhead_ms",
        "max_cpu_p95_overhead_ms",
        "recipient_exact_match"
    )
    foreach ($propertyName in $bindingProperties) {
        $configProperty = $configBinding.PSObject.Properties[$propertyName]
        $validationProperty = $validationBinding.PSObject.Properties[$propertyName]
        if ($null -eq $configProperty `
            -or $null -eq $validationProperty `
            -or [string]$configProperty.Value -ne [string]$validationProperty.Value) {
            throw "Package config/validation formal A/B scalar bindings disagree: $propertyName"
        }
    }
    if ([string]$configBinding.records_sha256 -ne [string]$Config.records_sha256 `
        -or [string]$configBinding.records_sha256 -ne [string]$Validation.end_to_end_evaluation.records_sha256 `
        -or [string]$configBinding.records_sha256 -ne [string]$accuracySummary.records_sha256 `
        -or [int]$configBinding.expected_receipts -ne $requiredRecords `
        -or [string]$configBinding.normalized_source_set_sha256 -ne `
            [string]$comparisonSummary.input_set.normalized_source_set_sha256 `
        -or [string]$configBinding.input_manifest_sha256 -ne `
            [string]$comparisonSummary.input_set.input_manifest.sha256 `
        -or [string]$configBinding.baseline_inference_manifest_sha256 -ne `
            [string]$comparisonSummary.run_manifests.baseline.sha256 `
        -or [string]$configBinding.hybrid_inference_manifest_sha256 -ne `
            [string]$comparisonSummary.run_manifests.hybrid.sha256 `
        -or [string]$configBinding.baseline_runtime_summary_sha256 -ne `
            [string]$comparisonSummary.run_summaries.baseline.sha256 `
        -or [string]$configBinding.hybrid_runtime_summary_sha256 -ne `
            [string]$comparisonSummary.run_summaries.hybrid.sha256 `
        -or [string]$configBinding.score_manifest_sha256 -ne [string]$accuracySummary.manifest_sha256 `
        -or [string]$configBinding.cli_assembly -ne "app/ReceiptMlNet.Cli.dll" `
        -or [string]$configBinding.cli_assembly_sha256 -ne (Get-Sha256 $deliveredAssembly) `
        -or [long]$configBinding.cli_assembly_size_bytes -ne $deliveredAssemblySize `
        -or [string]$configBinding.cli_app_closure_manifest_sha256 -cne `
            [string]$verifiedClosure.ClosureSha256 `
        -or [string]$configBinding.cli_app_closure_sha256 -cne `
            [string]$verifiedClosure.ClosureSha256 `
        -or [int]$configBinding.cli_app_closure_file_count -ne [int]$verifiedClosure.FileCount `
        -or [string]$validationBinding.cli_app_closure_sha256 -cne `
            [string]$verifiedClosure.ClosureSha256 `
        -or [int]$validationBinding.cli_app_closure_file_count -ne [int]$verifiedClosure.FileCount `
        -or [string]$configBinding.paddle_contract_sha256 -ne [string]$PaddleEvidence.ContractSha256 `
        -or [long]$configBinding.paddle_package_size_bytes -ne [long]$PaddleEvidence.PackageSizeBytes `
        -or [int]$configBinding.invariant_records -ne $requiredRecords `
        -or [double]$configBinding.recipient_candidate_coverage -ne 1.0 `
        -or [double]$configBinding.cpu_p95_overhead_ms -ne $p95Overhead `
        -or [double]$configBinding.max_cpu_p95_overhead_ms -ne $p95Ceiling `
        -or [double]$configBinding.recipient_exact_match -ne `
            [double]$accuracySummary.by_field.recipient_field.raw_exact_match) {
        throw "Formal hybrid CPU A/B bindings do not agree with their scored and runtime evidence."
    }
}

function Assert-FreshFormalAccuracyEvidence(
    [object]$Summary,
    [string]$ExpectedModelSha256,
    [string]$ExpectedRecordsSha256,
    [string]$ExpectedManifestSha256,
    [string]$ExpectedInputManifestSha256
) {
    $requiredRecords = 10016
    $failures = @(
        $Summary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $acceptanceFailures = @(
        $Summary.acceptance.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $requestedLimitProperty = $Summary.evaluation_scope.PSObject.Properties["requested_limit"]
    $maxStatusSafetyProperty = $Summary.acceptance.PSObject.Properties["max_non_success_to_success"]
    if ([int]$Summary.schema_version -ne 1 `
        -or [int]$Summary.coverage_contract_version -ne 2 `
        -or [string]$Summary.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
        -or [string]$Summary.evaluation_split -ne "val" `
        -or [string]$Summary.model_sha256 -ne $ExpectedModelSha256 `
        -or [string]$Summary.records_sha256 -ne $ExpectedRecordsSha256 `
        -or [string]$Summary.manifest_sha256 -ne $ExpectedManifestSha256 `
        -or $Summary.formal_delivery_gate -ne $true `
        -or $Summary.accepted -ne $true `
        -or $Summary.acceptance.passed -ne $true `
        -or $Summary.acceptance.formal_delivery_gate -ne $true `
        -or $failures.Count -ne 0 `
        -or $acceptanceFailures.Count -ne 0 `
        -or [string]$Summary.evaluation_scope.kind -ne "full_split" `
        -or $null -eq $requestedLimitProperty `
        -or $null -ne $requestedLimitProperty.Value `
        -or [int]$Summary.evaluation_scope.evaluated_expected_receipts -ne $requiredRecords `
        -or [int]$Summary.evaluation_scope.full_split_expected_receipts -ne $requiredRecords `
        -or $Summary.artifact_audit.all_results_match_model -ne $true `
        -or [int]$Summary.coverage.expected_receipts -ne $requiredRecords `
        -or [int]$Summary.coverage.matched_result_receipts -ne $requiredRecords `
        -or [int]$Summary.coverage.fully_scored_receipts -ne $requiredRecords `
        -or [double]$Summary.coverage.result_coverage -ne 1.0 `
        -or [double]$Summary.coverage.fully_scored_coverage -ne 1.0 `
        -or [int]$Summary.coverage.coverage_contract_version -ne 2 `
        -or [string]$Summary.coverage.candidate_coverage_domain -ne "all_expected_receipts" `
        -or [int]$Summary.coverage.fully_candidate_covered_receipts -ne $requiredRecords `
        -or [double]$Summary.coverage.all_field_candidate_coverage -ne 1.0 `
        -or $Summary.input_selection.hash_bound -ne $true `
        -or [int]$Summary.input_selection.records -ne $requiredRecords `
        -or [string]$Summary.input_selection.sha256 -ne $ExpectedInputManifestSha256 `
        -or $Summary.accuracy_denominators.hash_bound -ne $true `
        -or [string]$Summary.accuracy_denominators.source -ne "input_selection.field_reference_counts" `
        -or [string]$Summary.all_receipt_candidate_coverage.scope -ne "all_selected_receipts" `
        -or [int]$Summary.all_receipt_candidate_coverage.expected_receipts -ne $requiredRecords `
        -or [int]$Summary.all_receipt_candidate_coverage.complete_receipts -ne $requiredRecords `
        -or [int]$Summary.all_receipt_candidate_coverage.missing_complete_receipts -ne 0 `
        -or [double]$Summary.all_receipt_candidate_coverage.complete_coverage -ne 1.0 `
        -or $null -eq $maxStatusSafetyProperty `
        -or $null -eq $maxStatusSafetyProperty.Value `
        -or (($maxStatusSafetyProperty.Value -isnot [int]) `
            -and ($maxStatusSafetyProperty.Value -isnot [long])) `
        -or [int]$maxStatusSafetyProperty.Value -ne 0) {
        throw "Fresh package accuracy evidence is not an accepted 10016-record formal full-split result."
    }

    $fixedFloors = @(
        @{ Field = "amount"; Floor = 0.7885 },
        @{ Field = "time"; Floor = 0.9840 },
        @{ Field = "payment_method_field"; Floor = 0.9325 },
        @{ Field = "recipient_field"; Floor = 0.90 },
        @{ Field = "transfer_status"; Floor = 0.90 }
    )
    foreach ($gate in $fixedFloors) {
        $fieldName = [string]$gate.Field
        $requiredFloor = [double]$gate.Floor
        $metricProperty = $Summary.by_field.PSObject.Properties[$fieldName]
        $floorProperty = $Summary.floors.PSObject.Properties[$fieldName]
        if ($null -eq $metricProperty -or $null -eq $floorProperty) {
            throw "Fresh package accuracy evidence is missing $fieldName metrics or its fixed floor."
        }
        $records = [int]$metricProperty.Value.records
        $referenceCountProperty = `
            $Summary.input_selection.field_reference_counts.PSObject.Properties[$fieldName]
        $denominatorProperty = `
            $Summary.accuracy_denominators.by_field.PSObject.Properties[$fieldName]
        $candidateProperty = `
            $Summary.all_receipt_candidate_coverage.by_field.PSObject.Properties[$fieldName]
        if ($null -eq $referenceCountProperty `
            -or $null -eq $denominatorProperty `
            -or $null -eq $candidateProperty) {
            throw "Fresh package accuracy evidence is missing $fieldName denominator or all-receipt candidate coverage."
        }
        $coverage = [double]$candidateProperty.Value.candidate_coverage
        $exactMatch = [double]$metricProperty.Value.raw_exact_match
        if ([double]$floorProperty.Value -ne $requiredFloor `
            -or [int]$referenceCountProperty.Value -le 0 `
            -or $records -ne [int]$referenceCountProperty.Value `
            -or [int]$denominatorProperty.Value -ne [int]$referenceCountProperty.Value `
            -or [double]::IsNaN($coverage) `
            -or [double]::IsInfinity($coverage) `
            -or [int]$candidateProperty.Value.expected_receipts -ne $requiredRecords `
            -or [int]$candidateProperty.Value.candidate_records -ne $requiredRecords `
            -or [int]$candidateProperty.Value.missing_candidate_records -ne 0 `
            -or $coverage -ne 1.0 `
            -or [double]::IsNaN($exactMatch) `
            -or [double]::IsInfinity($exactMatch) `
            -or $exactMatch -lt $requiredFloor) {
            throw "Fresh package accuracy evidence did not pass the fixed $fieldName reference denominator/floor with all-receipt candidate coverage."
        }
    }
    $statusSafetyProperty = `
        $Summary.by_field.transfer_status.PSObject.Properties["non_success_to_success"]
    if ($null -eq $statusSafetyProperty `
        -or $null -eq $statusSafetyProperty.Value `
        -or (($statusSafetyProperty.Value -isnot [int]) `
            -and ($statusSafetyProperty.Value -isnot [long])) `
        -or [int]$statusSafetyProperty.Value -ne 0) {
        throw "Fresh package transfer-status evidence crossed the zero non-success-to-success safety line."
    }
}

function Assert-AcceptedPackageBinding(
    [string]$PackageRoot,
    [object]$Config,
    [object]$Validation,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
    [object]$PaddleEvidence
) {
    $requiredRecords = 10016
    $configDeclaration = $Config.PSObject.Properties["model_artifacts"]
    $validationDeclaration = $Validation.PSObject.Properties["model_artifacts"]
    $reviewValueProperty = $Config.PSObject.Properties["text_review_value"]
    $candidateByFieldProperty = $Validation.PSObject.Properties["candidates_by_field"]
    if ($null -eq $configDeclaration `
        -or $null -eq $validationDeclaration `
        -or $null -eq $reviewValueProperty `
        -or $null -eq $candidateByFieldProperty `
        -or $null -eq $candidateByFieldProperty.Value `
        -or [int]$Validation.candidate_complete -ne $requiredRecords `
        -or [string]$reviewValueProperty.Value -ne [string]$UnifiedEvidence.ReviewValue) {
        throw "Package lacks an explicit model/review binding or all-receipt five-field candidate evidence."
    }
    foreach ($fieldName in @("amount", "time", "recipient", "payment_method", "transfer_status")) {
        $candidateCountProperty = $candidateByFieldProperty.Value.PSObject.Properties[$fieldName]
        if ($null -eq $candidateCountProperty `
            -or [int]$candidateCountProperty.Value -ne $requiredRecords) {
            throw "Package validation does not prove $requiredRecords $fieldName candidates."
        }
    }
    Assert-DeclaredModelArtifacts `
        $configDeclaration.Value $PackageRoot `
        $DetectorEvidence $DeviceEvidence $UnifiedEvidence $PaddleEvidence "Package config"
    Assert-DeclaredModelArtifacts `
        $validationDeclaration.Value $PackageRoot `
        $DetectorEvidence $DeviceEvidence $UnifiedEvidence $PaddleEvidence "Package validation"
    Assert-HybridFormalEvidence `
        $PackageRoot $Config $Validation `
        $DetectorEvidence $DeviceEvidence $UnifiedEvidence $PaddleEvidence

    $onnxValidationPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/onnx-validation-summary.json" "ONNX validation summary"
    $endToEndSummaryPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-evaluation-summary.json" "end-to-end evaluation summary"
    $endToEndComparisonsPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-comparisons.jsonl" "end-to-end comparisons"
    $inferenceManifestPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/inference_manifest.json" "inference manifest"
    $validationInputListPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/validation-input-list.txt" "formal validation input list"
    $onnxValidation = Get-Content -LiteralPath $onnxValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $endToEndSummary = Get-Content -LiteralPath $endToEndSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $onnxFailures = @(
        $onnxValidation.acceptance.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $nonRecipientOnnxFailures = @(
        $onnxFailures |
            Where-Object { -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal) }
    )
    $endToEndFailures = @(
        $endToEndSummary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $manifestSha256 = Get-Sha256 $inferenceManifestPath
    Assert-FreshFormalAccuracyEvidence `
        $endToEndSummary ([string]$UnifiedEvidence.ModelSha256) `
        ([string]$Config.records_sha256) $manifestSha256 (Get-Sha256 $validationInputListPath)
    if ([int]$onnxValidation.schema_version -ne 1 `
        -or [int]$endToEndSummary.schema_version -ne 1 `
        -or [string]$Validation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$Validation.end_to_end_evaluation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$Validation.onnx_validation.summary_sha256 -ne (Get-Sha256 $onnxValidationPath) `
        -or [string]$Validation.end_to_end_evaluation.summary_sha256 -ne (Get-Sha256 $endToEndSummaryPath) `
        -or [string]$Validation.end_to_end_evaluation.comparisons_sha256 -ne (Get-Sha256 $endToEndComparisonsPath) `
        -or [string]$Validation.end_to_end_evaluation.manifest_sha256 -ne $manifestSha256 `
        -or [string]$Validation.end_to_end_evaluation.input_manifest_sha256 -ne `
            (Get-Sha256 $validationInputListPath) `
        -or [string]$onnxValidation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$onnxValidation.evaluation_split -ne "val" `
        -or $onnxValidation.acceptance.requested -ne $true `
        -or $nonRecipientOnnxFailures.Count -ne 0 `
        -or ($onnxValidation.acceptance.passed -eq $true -and $onnxFailures.Count -ne 0) `
        -or ($onnxValidation.acceptance.passed -ne $true -and $onnxFailures.Count -eq 0) `
        -or [string]$endToEndSummary.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
        -or [string]$endToEndSummary.evaluation_split -ne "val" `
        -or [string]$endToEndSummary.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$endToEndSummary.records_sha256 -ne [string]$Validation.end_to_end_evaluation.records_sha256 `
        -or [string]$endToEndSummary.manifest_sha256 -ne $manifestSha256 `
        -or $endToEndSummary.artifact_audit.all_results_match_model -ne $true `
        -or $endToEndSummary.accepted -ne $true `
        -or $endToEndSummary.acceptance.passed -ne $true `
        -or $endToEndFailures.Count -ne 0) {
        throw "Accepted package evidence is not cryptographically bound to the delivered artifacts."
    }
}

function Assert-ProductionGeometry([object]$Geometry, [string]$ResultPath) {
    $sourceSizeProperty = if ($null -eq $Geometry) { $null } else { $Geometry.PSObject.Properties["source_size"] }
    $rectifiedSizeProperty = if ($null -eq $Geometry) { $null } else { $Geometry.PSObject.Properties["rectified_size"] }
    $rotationProperty = if ($null -eq $Geometry) { $null } else { $Geometry.PSObject.Properties["rotation_degrees"] }
    $screenProperty = if ($null -eq $Geometry) { $null } else { $Geometry.PSObject.Properties["screen_detected"] }
    if ($null -eq $sourceSizeProperty `
        -or $null -eq $rectifiedSizeProperty `
        -or $null -eq $rotationProperty `
        -or $null -eq $screenProperty `
        -or $null -eq $sourceSizeProperty.Value `
        -or $null -eq $rectifiedSizeProperty.Value `
        -or $null -eq $rotationProperty.Value `
        -or $null -eq $screenProperty.Value `
        -or $screenProperty.Value -isnot [bool] `
        -or [int]$sourceSizeProperty.Value.width -lt 2 `
        -or [int]$sourceSizeProperty.Value.height -lt 2 `
        -or [int]$rectifiedSizeProperty.Value.width -lt 2 `
        -or [int]$rectifiedSizeProperty.Value.height -lt 2) {
        throw "Result has incomplete max-side-1600 geometry evidence: $ResultPath"
    }
    $expectedRotationDegrees = if (
        [int]$sourceSizeProperty.Value.width -gt [int]$sourceSizeProperty.Value.height
    ) { 90 } else { 0 }
    $expectedWidth = if ($expectedRotationDegrees -eq 90) {
        [int]$sourceSizeProperty.Value.height
    } else {
        [int]$sourceSizeProperty.Value.width
    }
    $expectedHeight = if ($expectedRotationDegrees -eq 90) {
        [int]$sourceSizeProperty.Value.width
    } else {
        [int]$sourceSizeProperty.Value.height
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
    $rectifiedMaximumSide = [Math]::Max(
        [int]$rectifiedSizeProperty.Value.width,
        [int]$rectifiedSizeProperty.Value.height)
    if ([int]$rotationProperty.Value -ne $expectedRotationDegrees `
        -or [bool]$screenProperty.Value `
        -or $rectifiedMaximumSide -gt 1600 `
        -or [int]$rectifiedSizeProperty.Value.width -ne $expectedWidth `
        -or [int]$rectifiedSizeProperty.Value.height -ne $expectedHeight) {
        throw "Result does not use the portrait-oriented full-image geometry contract: $ResultPath"
    }
}

function Assert-ResultProvenanceAndPolicy(
    [object]$Result,
    [string]$ResultPath,
    [string]$ExpectedSource,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
    [object]$PaddleEvidence
) {
    Assert-CurrentResultSemantics $Result $ResultPath
    $sourceProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["source"] }
    $contractsProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["model_contracts"] }
    $geometryProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["geometry"] }
    $deviceProperty = if ($null -eq $Result) { $null } else { $Result.PSObject.Properties["device"] }
    if ($null -eq $Result `
        -or [string]$Result.inference_engine -ne "mlnet" `
        -or $null -eq $geometryProperty `
        -or $null -eq $geometryProperty.Value `
        -or [string]$geometryProperty.Value.rectification -ne "max-side-1600" `
        -or $null -eq $deviceProperty `
        -or $null -eq $deviceProperty.Value `
        -or $null -eq $contractsProperty `
        -or $null -eq $contractsProperty.Value) {
        throw "Result does not prove the complete detector/device/v13/PP-OCR CPU path: $ResultPath"
    }
    Assert-ProductionGeometry $geometryProperty.Value $ResultPath
    if ($null -eq $sourceProperty -or [string]::IsNullOrWhiteSpace([string]$sourceProperty.Value)) {
        throw "Result has no source path: $ResultPath"
    }
    Assert-SafePathSyntax ([string]$sourceProperty.Value) "result source"
    $resultSource = [IO.Path]::GetFullPath([string]$sourceProperty.Value)
    $expectedSourceFull = [IO.Path]::GetFullPath($ExpectedSource)
    Assert-NoReparsePointInExistingPath $resultSource "result source"
    if (-not $resultSource.Equals($expectedSourceFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Result source does not match its manifest source: $ResultPath"
    }

    $contracts = $contractsProperty.Value
    $expectedContracts = [ordered]@{
        detector = [string]$DetectorEvidence.ContractFileName
        detector_sha256 = [string]$DetectorEvidence.ModelSha256
        detector_contract_sha256 = [string]$DetectorEvidence.ContractSha256
        device = [string]$DeviceEvidence.ContractFileName
        device_sha256 = [string]$DeviceEvidence.ModelSha256
        device_contract_sha256 = [string]$DeviceEvidence.ContractSha256
        ocr_bundle = [string]$PaddleEvidence.ContractFileName
        ocr_bundle_contract_sha256 = [string]$PaddleEvidence.ContractSha256
        unified_ocr_model = [string]$UnifiedEvidence.ModelFileName
        unified_ocr_contract = [string]$UnifiedEvidence.ContractFileName
        unified_ocr_model_sha256 = [string]$UnifiedEvidence.ModelSha256
        unified_ocr_labels_sha256 = [string]$UnifiedEvidence.LabelsSha256
        unified_ocr_contract_sha256 = [string]$UnifiedEvidence.ContractSha256
    }
    foreach ($propertyName in $expectedContracts.Keys) {
        $property = $contracts.PSObject.Properties[$propertyName]
        if ($null -eq $property -or [string]$property.Value -ne [string]$expectedContracts[$propertyName]) {
            throw "Result has mixed or missing model provenance ($propertyName): $ResultPath"
        }
    }

    $fieldsProperty = $Result.PSObject.Properties["fields"]
    if ($null -eq $fieldsProperty -or $null -eq $fieldsProperty.Value) {
        throw "Result has no receipt field evidence: $ResultPath"
    }
    foreach ($fieldName in @("amount", "time", "recipient", "payment_method")) {
        $fieldProperty = $fieldsProperty.Value.PSObject.Properties[$fieldName]
        if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
            throw "Result has no $fieldName field object: $ResultPath"
        }
        $field = $fieldProperty.Value
        $policyProperty = $field.PSObject.Properties["delivery_policy"]
        if ($null -eq $policyProperty `
            -or [string]$policyProperty.Value -ne [string]$UnifiedEvidence.TextDeliveryPolicy) {
            throw "Result $fieldName has the wrong fail-closed delivery policy: $ResultPath"
        }
        $candidateProperty = $field.PSObject.Properties["candidate"]
        $candidate = if ($null -eq $candidateProperty) { $null } else { [string]$candidateProperty.Value }
        $valueProperty = $field.PSObject.Properties["value"]
        $fieldValue = if ($null -eq $valueProperty) { $null } else { $valueProperty.Value }
        $deliveryValueProperty = $field.PSObject.Properties["delivery_value"]
        $deliveryValue = if ($null -eq $deliveryValueProperty) { $null } else { $deliveryValueProperty.Value }
        $stateProperty = $field.PSObject.Properties["state"]
        if ($null -eq $stateProperty) {
            throw "Result $fieldName has no state: $ResultPath"
        }
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            if ([string]$stateProperty.Value -notin @("absent", "unreadable") `
                -or ($null -ne $fieldValue -and [string]$fieldValue -ne [string]$UnifiedEvidence.ReviewValue) `
                -or ($null -ne $deliveryValue -and [string]$deliveryValue -ne [string]$UnifiedEvidence.ReviewValue)) {
                throw "Result $fieldName has an invalid fail-closed missing-candidate state: $ResultPath"
            }
        }
        elseif ([string]$stateProperty.Value -ne "review" `
            -or [string]$fieldValue -ne [string]$UnifiedEvidence.ReviewValue `
            -or [string]$deliveryValue -ne [string]$UnifiedEvidence.ReviewValue) {
            throw "Result $fieldName candidate escaped the review-only policy: $ResultPath"
        }
    }

    if ([int]$UnifiedEvidence.ArchitectureVersion -eq 13) {
        $statusProperty = $fieldsProperty.Value.PSObject.Properties["transfer_status"]
        if ($null -eq $statusProperty -or $null -eq $statusProperty.Value) {
            throw "V13 result has no transfer_status field object: $ResultPath"
        }
        $status = $statusProperty.Value
        $policyProperty = $status.PSObject.Properties["delivery_policy"]
        $stateProperty = $status.PSObject.Properties["state"]
        $valueProperty = $status.PSObject.Properties["value"]
        $deliveryValueProperty = $status.PSObject.Properties["delivery_value"]
        $rawProperty = $status.PSObject.Properties["raw"]
        $candidateProperty = $status.PSObject.Properties["candidate"]
        $ctcCandidateProperty = $status.PSObject.Properties["ctc_candidate"]
        $normalizedProperty = $status.PSObject.Properties["normalized"]
        $statusValue = if ($null -eq $valueProperty) { $null } else { $valueProperty.Value }
        $statusDeliveryValue = if ($null -eq $deliveryValueProperty) { $null } else { $deliveryValueProperty.Value }
        if ($null -eq $policyProperty `
            -or [string]$policyProperty.Value -ne [string]$UnifiedEvidence.StatusTextDeliveryPolicy `
            -or $null -eq $stateProperty) {
            throw "V13 result transfer_status has incomplete review-only policy evidence: $ResultPath"
        }
        if ([string]$stateProperty.Value -eq "absent") {
            throw "V13 result transfer_status is absent; the complete delivery path requires visible OCR text: $ResultPath"
        }
        else {
            $rawStatus = if ($null -eq $rawProperty) { "" } else { [string]$rawProperty.Value }
            $candidateStatus = if ($null -eq $candidateProperty) { "" } else { [string]$candidateProperty.Value }
            $ctcCandidateStatus = if ($null -eq $ctcCandidateProperty) { "" } else { [string]$ctcCandidateProperty.Value }
            $normalizedStatus = if ($null -eq $normalizedProperty) { "" } else { [string]$normalizedProperty.Value }
            if ([string]::IsNullOrWhiteSpace($rawStatus) `
                -or $rawStatus -ne $candidateStatus `
                -or $rawStatus -ne $ctcCandidateStatus `
                -or [string]::IsNullOrWhiteSpace($normalizedStatus) `
                -or $normalizedStatus -ne (Get-NormalizedTransferStatus $rawStatus)) {
                throw "V13 result transfer_status has incomplete or inconsistent OCR text evidence: $ResultPath"
            }
            if ([string]$stateProperty.Value -ne "review" `
                -or [string]$statusValue -ne [string]$UnifiedEvidence.StatusTextReviewValue `
                -or [string]$statusDeliveryValue -ne [string]$UnifiedEvidence.StatusTextReviewValue) {
                throw "V13 result transfer_status escaped the review-only policy: $ResultPath"
            }
        }
    }
}

function Resolve-ContainedOutputFile([string]$OutputRoot, [string]$Path, [string]$Description) {
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
    Require-File $target $Description
    Assert-NoReparsePointInExistingPath $target $Description
    return $target
}

function Get-NonNegativeFiniteDouble([object]$Value, [string]$Description) {
    if ($null -eq $Value) {
        throw "Missing ${Description}."
    }
    $number = [double]$Value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt 0.0) {
        throw "Invalid ${Description}: $Value"
    }
    return $number
}

$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
Write-Host "Verifying delivery package integrity..." -ForegroundColor DarkGray
Assert-PackageIntegrity $packageRoot
Write-Host "Package: PASS" -ForegroundColor Green

$configPath = Join-Path $packageRoot "evidence\package_config.json"
$validationPath = Join-Path $packageRoot "evidence\package_validation.json"
Require-File $configPath "delivery package configuration"
Require-File $validationPath "delivery package validation"
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$validation = Get-Content -LiteralPath $validationPath -Raw -Encoding UTF8 | ConvertFrom-Json
$requiredTextPolicy = "review_only_pending_independent_human_truth_calibration"
$requiredReviewValue = "review"
if ([int]$config.schema_version -ne 1 `
    -or [string]$config.kind -ne "receipt_mlnet_hybrid_recipient_delivery_package_v1" `
    -or [string]$config.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$config.onnx_runtime_flavor -ne "cpu" `
    -or [string]$config.runtime_device -ne "cpu" `
    -or [string]$config.rectification -ne "max-side-1600" `
    -or [string]$config.orientation_rule -ne "exif_upright_landscape_clockwise_90" `
    -or [string]$config.ocr_mode -ne "hybrid-recipient" `
    -or [string]::IsNullOrWhiteSpace([string]$config.device_model) `
    -or [string]$config.text_delivery_policy -ne $requiredTextPolicy) {
    throw "This is not an accepted hybrid-recipient production CPU package."
}
if ([int]$validation.schema_version -ne 1 `
    -or [string]$validation.kind -ne "receipt_mlnet_hybrid_recipient_package_validation_v1" `
    -or [string]$validation.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$validation.runtime_flavor -ne "cpu" `
    -or [string]$validation.runtime_device -ne "cpu" `
    -or [string]$validation.rectification -ne "max-side-1600" `
    -or [string]$validation.orientation_rule -ne "exif_upright_landscape_clockwise_90" `
    -or [string]$validation.ocr_mode -ne "hybrid-recipient" `
    -or $validation.include_device_model -ne $true `
    -or [string]$validation.inference_summary.requested_device -ne "cpu" `
    -or [string]$validation.inference_summary.unified_provider -ne "cpu" `
    -or [string]$validation.inference_summary.paddle_ocr_provider -ne "cpu" `
    -or $validation.end_to_end_evaluation.performed -ne $true `
    -or [string]$validation.end_to_end_evaluation.status -ne "accepted") {
    throw "Delivery package evidence has not accepted the complete CPU full-val run."
}

Assert-SafePathSyntax $InputImage "InputImage"
Assert-SafePathSyntax $OutputDirectory "OutputDirectory"
$InputImage = [IO.Path]::GetFullPath($InputImage)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
Require-File $InputImage "input receipt image"
Assert-NoReparsePointInExistingPath $InputImage "InputImage"
Assert-NoReparsePointInExistingPath $OutputDirectory "OutputDirectory"
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to mix a single-image validation with an existing output directory: $OutputDirectory"
}
if (Test-PathWithin $OutputDirectory $packageRoot) {
    throw "OutputDirectory must be outside the immutable delivery package: $OutputDirectory"
}

$detectorName = [string]$config.detector_model
$deviceModelName = [string]$config.device_model
if ([IO.Path]::GetFileName($detectorName) -ne $detectorName `
    -or [IO.Path]::GetFileName($deviceModelName) -ne $deviceModelName) {
    throw "Detector and device model configuration must contain leaf filenames only."
}
$executable = Resolve-ContainedPackageFile $packageRoot "app/ReceiptMlNet.Cli.exe" "ReceiptMlNet executable"
$detector = Resolve-ContainedPackageFile $packageRoot ("models/" + $detectorName) "receipt detector"
$deviceModel = Resolve-ContainedPackageFile $packageRoot ("models/" + $deviceModelName) "device classifier"
$unifiedModel = Resolve-ContainedPackageFile $packageRoot ([string]$config.unified_model) "unified receipt OCR"
$detectorEvidence = Read-StandardModelEvidence `
    $packageRoot $detector "receipt_lrcnn_v1" "receipt detector"
$deviceEvidence = Read-StandardModelEvidence `
    $packageRoot $deviceModel "statusbar_device_v1" "device classifier"
$unifiedEvidence = Read-UnifiedModelEvidence `
    $packageRoot $unifiedModel $requiredTextPolicy $requiredReviewValue
if ([int]$unifiedEvidence.ArchitectureVersion -ne 13) {
    throw "This production entrypoint requires architecture-v13 visible transfer-status OCR. The package contains a legacy v12 status classifier; use a v13 delivery package."
}
$recipientDeclaration = $config.model_artifacts.PSObject.Properties["recipient_ppocr"]
if ($null -eq $recipientDeclaration -or $null -eq $recipientDeclaration.Value) {
    throw "Package config does not contain the recipient PP-OCR artifact declaration."
}
$paddleEvidence = Read-PaddleRecipientEvidence $packageRoot $recipientDeclaration.Value
Assert-AcceptedPackageBinding `
    $packageRoot $config $validation `
    $detectorEvidence $deviceEvidence $unifiedEvidence $paddleEvidence

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host " Receipt AI - Windows CPU single-image verification" -ForegroundColor Cyan
Write-Host " detector + device classifier + v13 OCR + recipient PP-OCR (pure ONNX)" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Input : $InputImage"
Write-Host "Output: $OutputDirectory"
Write-Host ""

& $executable `
    --detector $detector `
    --device-model $deviceModel `
    --ocr hybrid-recipient `
    --ocr-model $unifiedModel `
    --ocr-bundle $paddleEvidence.BundlePath `
    --input $InputImage `
    --output $OutputDirectory `
    --device cpu `
    --rectification max-side-1600 `
    --annotate all `
    --require-complete
if ($LASTEXITCODE -ne 0) {
    throw "Single-image CPU inference failed with exit code $LASTEXITCODE."
}
Assert-NoReparsePointInExistingPath $OutputDirectory "OutputDirectory"

$manifestPath = Join-Path $OutputDirectory "inference_manifest.json"
$summaryPath = Join-Path $OutputDirectory "inference_summary.json"
$errorsPath = Join-Path $OutputDirectory "inference_errors.jsonl"
Require-File $manifestPath "single-image inference manifest"
Require-File $summaryPath "single-image inference summary"
Require-File $errorsPath "single-image inference errors"
Assert-NoReparsePointInExistingPath $manifestPath "single-image inference manifest"
Assert-NoReparsePointInExistingPath $summaryPath "single-image inference summary"
Assert-NoReparsePointInExistingPath $errorsPath "single-image inference errors"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$errorText = Get-Content -LiteralPath $errorsPath -Raw -Encoding UTF8
if ($null -eq $manifest) {
    throw "Single-image inference manifest is empty."
}
$manifestCount = 0
$singleManifestRecord = $null
foreach ($record in $manifest) {
    $manifestCount++
    $singleManifestRecord = $record
}
if ($manifestCount -ne 1 -or [string]$singleManifestRecord.status -ne "written") {
    throw "Single-image inference did not produce exactly one clean result."
}
if ([string]$summary.requested_device -ne "cpu" `
    -or [string]$summary.unified_provider -ne "cpu" `
    -or [string]$summary.paddle_ocr_provider -ne "cpu" `
    -or [int]$summary.input -ne 1 `
    -or [int]$summary.written -ne 1 `
    -or [int]$summary.skipped -ne 0 `
    -or [int]$summary.errors -ne 0 `
    -or [int]$summary.inference_latency_ms.count -ne 1 `
    -or -not [string]::IsNullOrWhiteSpace($errorText)) {
    throw "Single-image inference summary, provider, or error evidence is inconsistent."
}
$meanLatencyMs = Get-NonNegativeFiniteDouble $summary.inference_latency_ms.mean "mean CPU latency"
$p50LatencyMs = Get-NonNegativeFiniteDouble $summary.inference_latency_ms.p50 "p50 CPU latency"
$p95LatencyMs = Get-NonNegativeFiniteDouble $summary.inference_latency_ms.p95 "p95 CPU latency"
if ($p95LatencyMs -lt $p50LatencyMs) {
    throw "Single-image inference p95 latency is below p50."
}

$resultPath = Resolve-ContainedOutputFile $OutputDirectory ([string]$singleManifestRecord.result) "single-image result JSON"
$annotatedRectifiedPath = Resolve-ContainedOutputFile `
    $OutputDirectory ([string]$singleManifestRecord.annotated_rectified) "rectified annotation"
$annotatedOriginalPath = Resolve-ContainedOutputFile `
    $OutputDirectory ([string]$singleManifestRecord.annotated_original) "original annotation"
$manifestSourceValue = [string]$singleManifestRecord.source
Assert-SafePathSyntax $manifestSourceValue "single-image manifest source"
$manifestSource = [IO.Path]::GetFullPath($manifestSourceValue)
Require-File $manifestSource "single-image manifest source"
if (-not $manifestSource.Equals($InputImage, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Single-image manifest source does not match InputImage."
}
$result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-ResultProvenanceAndPolicy `
    $result $resultPath $manifestSource `
    $detectorEvidence $deviceEvidence $unifiedEvidence $paddleEvidence

$fieldRows = @(
    [pscustomobject]@{ Field = "Amount"; Candidate = [string]$result.fields.amount.candidate; State = [string]$result.fields.amount.state },
    [pscustomobject]@{ Field = "Time"; Candidate = [string]$result.fields.time.candidate; State = [string]$result.fields.time.state },
    [pscustomobject]@{ Field = "Recipient"; Candidate = [string]$result.fields.recipient.candidate; State = [string]$result.fields.recipient.state },
    [pscustomobject]@{ Field = "Payment method"; Candidate = [string]$result.fields.payment_method.candidate; State = [string]$result.fields.payment_method.state }
)

Write-Host ""
Write-Host "RESULT" -ForegroundColor Green
Write-Host ("Device : {0} ({1}, confidence {2})" -f $result.device.platform_cn, $result.device.platform, $result.device.confidence)
$fieldRows | Format-Table -AutoSize
Write-Host ""
Write-Host "TRANSFER STATUS OCR" -ForegroundColor Cyan
[pscustomobject]@{
    "Raw OCR" = [string]$result.fields.transfer_status.raw
    "Normalized" = [string]$result.fields.transfer_status.normalized
    # Legacy label: "Decision" = [string]$result.fields.transfer_status.state
    "Review state" = [string]$result.fields.transfer_status.state
} | Format-List
Write-Host ("CPU latency: {0:N2} ms" -f $meanLatencyMs)
Write-Host "Result JSON : $resultPath"
Write-Host "Annotated   : $annotatedOriginalPath"
Write-Host "Rectified   : $annotatedRectifiedPath"
Write-Host "Policy      : candidates are shown for verification; business values remain fail-closed as review."
Write-Host ""
Write-Host "PASS: the complete pure-ONNX detector/device/v13/PP-OCR CPU pipeline produced one reviewable receipt." -ForegroundColor Green
