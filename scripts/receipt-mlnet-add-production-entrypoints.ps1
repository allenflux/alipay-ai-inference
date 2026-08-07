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
        -or [int]$contract.schema_version -ne 1 `
        -or [string]$contract.kind -ne "receipt_unified_field_reader_v12" `
        -or [int]$contract.model.architecture_version -ne 12 `
        -or [string]$contract.onnx_file -ne [IO.Path]::GetFileName($ModelPath) `
        -or [string]$contract.labels_file -ne [IO.Path]::GetFileName($labelsPath) `
        -or [string]$contract.onnx_sha256 -ne $modelSha256 `
        -or [string]$contract.labels_sha256 -ne $labelsSha256 `
        -or [string]$contract.text_delivery_policy.runtime_policy -ne $RequiredTextPolicy `
        -or [string]$contract.text_delivery_policy.review_value -ne $RequiredReviewValue) {
        throw "Unified OCR model, labels, contract, or fail-closed text policy is inconsistent."
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
        Kind = "receipt_unified_field_reader_v12"
        TextDeliveryPolicy = $RequiredTextPolicy
        ReviewValue = $RequiredReviewValue
    }
}

function Assert-DeclaredModelArtifacts(
    [object]$Declaration,
    [string]$PackageRoot,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
    [string]$Description
) {
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
        -or [string]$Declaration.unified_ocr.review_value -ne [string]$UnifiedEvidence.ReviewValue) {
        throw "$Description model artifact declaration does not match the delivered models and sidecars."
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
        throw "Legacy result has incomplete max-side-1600 geometry evidence: $ResultPath"
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
        throw "Legacy result does not use the portrait-oriented full-image geometry contract: $ResultPath"
    }
}

function Assert-ResultProvenanceAndPolicy(
    [object]$Result,
    [string]$ResultPath,
    [string]$ExpectedSource,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence
) {
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
        -or $null -eq $contractsProperty.Value `
        -or $null -eq $sourceProperty `
        -or [string]::IsNullOrWhiteSpace([string]$sourceProperty.Value)) {
        throw "Legacy result does not prove the complete three-model ML.NET CPU path: $ResultPath"
    }
    Assert-ProductionGeometry $geometryProperty.Value $ResultPath
    Assert-SafePathSyntax ([string]$sourceProperty.Value) "legacy result source"
    $resultSource = [IO.Path]::GetFullPath([string]$sourceProperty.Value)
    $expectedSourceFull = [IO.Path]::GetFullPath($ExpectedSource)
    Assert-NoReparsePointInExistingPath $resultSource "legacy result source"
    if (-not $resultSource.Equals($expectedSourceFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Legacy result source does not match result_evidence_sha256: $ResultPath"
    }

    $contracts = $contractsProperty.Value
    $expectedContracts = [ordered]@{
        detector = [string]$DetectorEvidence.ContractFileName
        detector_sha256 = [string]$DetectorEvidence.ModelSha256
        detector_contract_sha256 = [string]$DetectorEvidence.ContractSha256
        device = [string]$DeviceEvidence.ContractFileName
        device_sha256 = [string]$DeviceEvidence.ModelSha256
        device_contract_sha256 = [string]$DeviceEvidence.ContractSha256
        unified_ocr_model = [string]$UnifiedEvidence.ModelFileName
        unified_ocr_contract = [string]$UnifiedEvidence.ContractFileName
        unified_ocr_model_sha256 = [string]$UnifiedEvidence.ModelSha256
        unified_ocr_labels_sha256 = [string]$UnifiedEvidence.LabelsSha256
        unified_ocr_contract_sha256 = [string]$UnifiedEvidence.ContractSha256
    }
    foreach ($propertyName in $expectedContracts.Keys) {
        $property = $contracts.PSObject.Properties[$propertyName]
        if ($null -eq $property -or [string]$property.Value -ne [string]$expectedContracts[$propertyName]) {
            throw "Legacy result has mixed or missing model provenance ($propertyName): $ResultPath"
        }
    }

    $fieldsProperty = $Result.PSObject.Properties["fields"]
    if ($null -eq $fieldsProperty -or $null -eq $fieldsProperty.Value) {
        throw "Legacy result has no receipt field evidence: $ResultPath"
    }
    foreach ($fieldName in @("amount", "time", "recipient", "payment_method")) {
        $fieldProperty = $fieldsProperty.Value.PSObject.Properties[$fieldName]
        if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
            throw "Legacy result has no $fieldName field object: $ResultPath"
        }
        $field = $fieldProperty.Value
        $policyProperty = $field.PSObject.Properties["delivery_policy"]
        $candidateProperty = $field.PSObject.Properties["candidate"]
        $candidate = if ($null -eq $candidateProperty) { $null } else { [string]$candidateProperty.Value }
        $valueProperty = $field.PSObject.Properties["value"]
        $fieldValue = if ($null -eq $valueProperty) { $null } else { $valueProperty.Value }
        $deliveryValueProperty = $field.PSObject.Properties["delivery_value"]
        $deliveryValue = if ($null -eq $deliveryValueProperty) { $null } else { $deliveryValueProperty.Value }
        $stateProperty = $field.PSObject.Properties["state"]
        if ($null -eq $policyProperty `
            -or [string]$policyProperty.Value -ne [string]$UnifiedEvidence.TextDeliveryPolicy `
            -or $null -eq $stateProperty) {
            throw "Legacy result $fieldName has incomplete fail-closed policy evidence: $ResultPath"
        }
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            if ([string]$stateProperty.Value -notin @("absent", "unreadable") `
                -or ($null -ne $fieldValue -and [string]$fieldValue -ne [string]$UnifiedEvidence.ReviewValue) `
                -or ($null -ne $deliveryValue -and [string]$deliveryValue -ne [string]$UnifiedEvidence.ReviewValue)) {
                throw "Legacy result $fieldName has an invalid missing-candidate state: $ResultPath"
            }
        }
        elseif ([string]$stateProperty.Value -ne "review" `
            -or [string]$fieldValue -ne [string]$UnifiedEvidence.ReviewValue `
            -or [string]$deliveryValue -ne [string]$UnifiedEvidence.ReviewValue) {
            throw "Legacy result $fieldName candidate escaped the review-only policy: $ResultPath"
        }
    }
}

function Read-LegacyResultEvidence(
    [string]$ResultEvidencePath,
    [string]$InferenceManifestPath,
    [object]$Validation,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence
) {
    $rows = Get-Content -LiteralPath $ResultEvidencePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $rows) {
        throw "Legacy result evidence is empty."
    }
    $resultSet = @{}
    $sourceSet = @{}
    $evidencePairs = @{}
    $rowCount = 0
    foreach ($row in $rows) {
        $rowCount++
        foreach ($propertyName in @("source", "result", "result_sha256", "result_bytes")) {
            if ($null -eq $row.PSObject.Properties[$propertyName]) {
                throw "Legacy result evidence contains an incomplete row."
            }
        }
        $sourceValue = [string]$row.source
        $resultValue = [string]$row.result
        Assert-SafePathSyntax $sourceValue "legacy result source"
        Assert-SafePathSyntax $resultValue "legacy result path"
        $sourcePath = [IO.Path]::GetFullPath($sourceValue)
        $resultPath = [IO.Path]::GetFullPath($resultValue)
        Require-File $sourcePath "legacy result source"
        Require-File $resultPath "legacy result JSON"
        if ($sourceSet.ContainsKey($sourcePath) -or $resultSet.ContainsKey($resultPath)) {
            throw "Legacy result evidence contains a duplicate source or result path."
        }
        $sourceSet[$sourcePath] = $true
        $resultSet[$resultPath] = $true
        $evidencePairs[($sourcePath + "|" + $resultPath)] = $true

        $expectedHash = ([string]$row.result_sha256).ToLowerInvariant()
        $expectedBytes = [long]0
        $bytesText = [Convert]::ToString($row.result_bytes, [Globalization.CultureInfo]::InvariantCulture)
        if ($expectedHash -notmatch '^[0-9a-f]{64}$' `
            -or -not [long]::TryParse(
                $bytesText,
                [Globalization.NumberStyles]::Integer,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$expectedBytes) `
            -or $expectedBytes -lt 0 `
            -or (Get-Sha256 $resultPath) -ne $expectedHash `
            -or (Get-Item -LiteralPath $resultPath).Length -ne $expectedBytes) {
            throw "Legacy result evidence hash or byte count is invalid: $resultPath"
        }
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Assert-ResultProvenanceAndPolicy `
            $result $resultPath $sourcePath $DetectorEvidence $DeviceEvidence $UnifiedEvidence
    }
    $expectedResults = [int]$Validation.inference_summary.written
    if ($rowCount -le 0 `
        -or $rowCount -ne $expectedResults `
        -or $rowCount -ne [int]$Validation.end_to_end_evaluation.expected_receipts) {
        throw "Legacy result evidence does not cover every accepted receipt: results=$rowCount expected=$expectedResults"
    }

    $manifest = Get-Content -LiteralPath $InferenceManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $manifest) {
        throw "Packaged legacy inference manifest is empty."
    }
    $manifestPairs = @{}
    $manifestCount = 0
    foreach ($record in $manifest) {
        $manifestCount++
        if ([string]$record.status -ne "written") {
            throw "Packaged legacy inference manifest contains a non-written record."
        }
        $manifestSourceValue = [string]$record.source
        $manifestResultValue = [string]$record.result
        Assert-SafePathSyntax $manifestSourceValue "legacy manifest source"
        Assert-SafePathSyntax $manifestResultValue "legacy manifest result"
        $manifestSource = [IO.Path]::GetFullPath($manifestSourceValue)
        $manifestResult = if ([IO.Path]::IsPathRooted($manifestResultValue)) {
            [IO.Path]::GetFullPath($manifestResultValue)
        }
        else {
            [IO.Path]::GetFullPath((Join-Path ([string]$Validation.output) $manifestResultValue))
        }
        $pairKey = $manifestSource + "|" + $manifestResult
        if ($manifestPairs.ContainsKey($pairKey)) {
            throw "Packaged legacy inference manifest contains a duplicate source/result pair."
        }
        $manifestPairs[$pairKey] = $true
    }
    $missingManifestPairs = @($evidencePairs.Keys | Where-Object { -not $manifestPairs.ContainsKey($_) })
    $extraManifestPairs = @($manifestPairs.Keys | Where-Object { -not $evidencePairs.ContainsKey($_) })
    if ($manifestCount -ne $rowCount `
        -or $missingManifestPairs.Count -ne 0 `
        -or $extraManifestPairs.Count -ne 0) {
        throw "Legacy result evidence and packaged inference manifest differ: missing=$($missingManifestPairs.Count) extra=$($extraManifestPairs.Count)"
    }
    return $rowCount
}

function Read-AcceptedProductionPackage([string]$PackageRoot) {
    $configPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/package_config.json" "package configuration"
    $validationPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/package_validation.json" "package validation"
    $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $validation = Get-Content -LiteralPath $validationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $requiredTextPolicy = "review_only_pending_independent_human_truth_calibration"
    $requiredReviewValue = "review"
    if ([string]$config.kind -ne "receipt_mlnet_unified_delivery_package_v1" `
        -or [string]$config.validation_scope -ne "full_val_end_to_end_scored_cpu" `
        -or [string]$config.onnx_runtime_flavor -ne "cpu" `
        -or [string]$config.runtime_device -ne "cpu" `
        -or [string]$config.rectification -ne "max-side-1600" `
        -or [string]$config.orientation_rule -ne "exif_upright_landscape_clockwise_90" `
        -or [string]::IsNullOrWhiteSpace([string]$config.device_model) `
        -or [string]$config.text_delivery_policy -ne $requiredTextPolicy) {
        throw "Package is not an accepted complete three-model production CPU package."
    }
    if ([string]$validation.kind -ne "receipt_mlnet_unified_package_validation_v1" `
        -or [string]$validation.validation_scope -ne "full_val_end_to_end_scored_cpu" `
        -or [string]$validation.runtime_flavor -ne "cpu" `
        -or [string]$validation.runtime_device -ne "cpu" `
        -or [string]$validation.rectification -ne "max-side-1600" `
        -or [string]$validation.orientation_rule -ne "exif_upright_landscape_clockwise_90" `
        -or $validation.include_device_model -ne $true `
        -or $validation.end_to_end_evaluation.performed -ne $true `
        -or [string]$validation.end_to_end_evaluation.status -ne "accepted") {
        throw "Package validation has not accepted the complete CPU full-val run."
    }

    $detectorName = [string]$config.detector_model
    $deviceModelName = [string]$config.device_model
    if ([IO.Path]::GetFileName($detectorName) -ne $detectorName `
        -or [IO.Path]::GetFileName($deviceModelName) -ne $deviceModelName) {
        throw "Detector and device model configuration must contain leaf filenames only."
    }
    $null = Resolve-ContainedPackageFile $PackageRoot "app/ReceiptMlNet.Cli.exe" "ReceiptMlNet executable"
    $detectorPath = Resolve-ContainedPackageFile `
        $PackageRoot ("models/" + $detectorName) "receipt detector"
    $devicePath = Resolve-ContainedPackageFile `
        $PackageRoot ("models/" + $deviceModelName) "device classifier"
    $unifiedPath = Resolve-ContainedPackageFile `
        $PackageRoot ([string]$config.unified_model) "unified receipt OCR"
    $detectorEvidence = Read-StandardModelEvidence `
        $PackageRoot $detectorPath "receipt_lrcnn_v1" "receipt detector"
    $deviceEvidence = Read-StandardModelEvidence `
        $PackageRoot $devicePath "statusbar_device_v1" "device classifier"
    $unifiedEvidence = Read-UnifiedModelEvidence `
        $PackageRoot $unifiedPath $requiredTextPolicy $requiredReviewValue
    $configDeclarationProperty = $config.PSObject.Properties["model_artifacts"]
    $validationDeclarationProperty = $validation.PSObject.Properties["model_artifacts"]
    $modelBindingMode = "declared_model_artifacts"
    if (($null -eq $configDeclarationProperty) -xor ($null -eq $validationDeclarationProperty)) {
        throw "Package config and validation must either both declare model_artifacts or both use the legacy package-level binding."
    }
    if ($null -ne $configDeclarationProperty) {
        Assert-DeclaredModelArtifacts `
            $configDeclarationProperty.Value $PackageRoot `
            $detectorEvidence $deviceEvidence $unifiedEvidence "Package config"
        Assert-DeclaredModelArtifacts `
            $validationDeclarationProperty.Value $PackageRoot `
            $detectorEvidence $deviceEvidence $unifiedEvidence "Package validation"
    }
    else {
        $modelBindingMode = "legacy_result_reverification"
    }
    $reviewValueProperty = $config.PSObject.Properties["text_review_value"]
    if ($null -ne $reviewValueProperty -and [string]$reviewValueProperty.Value -ne $requiredReviewValue) {
        throw "Package config text_review_value conflicts with the unified OCR contract."
    }

    $onnxValidationPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/onnx-validation-summary.json" "ONNX validation summary"
    $endToEndSummaryPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-evaluation-summary.json" "end-to-end evaluation summary"
    $endToEndComparisonsPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-comparisons.jsonl" "end-to-end comparisons"
    $inferenceManifestPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/inference_manifest.json" "inference manifest"
    $resultEvidencePath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/result_evidence_sha256.json" "result evidence hashes"
    $onnxValidation = Get-Content -LiteralPath $onnxValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $endToEndSummary = Get-Content -LiteralPath $endToEndSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $onnxFailures = @(
        $onnxValidation.acceptance.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $endToEndFailures = @(
        $endToEndSummary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $actualSummarySha256 = Get-Sha256 $endToEndSummaryPath
    $actualComparisonsSha256 = Get-Sha256 $endToEndComparisonsPath
    $actualManifestSha256 = Get-Sha256 $inferenceManifestPath
    $actualOnnxValidationSha256 = Get-Sha256 $onnxValidationPath
    if ([string]$validation.model_sha256 -ne [string]$unifiedEvidence.ModelSha256 `
        -or [string]$validation.end_to_end_evaluation.model_sha256 -ne [string]$unifiedEvidence.ModelSha256 `
        -or [string]$validation.end_to_end_evaluation.summary_sha256 -ne $actualSummarySha256 `
        -or [string]$validation.end_to_end_evaluation.comparisons_sha256 -ne $actualComparisonsSha256 `
        -or [string]$validation.end_to_end_evaluation.manifest_sha256 -ne $actualManifestSha256 `
        -or [string]$endToEndSummary.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
        -or [string]$endToEndSummary.evaluation_split -ne "val" `
        -or [string]$endToEndSummary.model_sha256 -ne [string]$unifiedEvidence.ModelSha256 `
        -or [string]$endToEndSummary.records_sha256 -ne [string]$validation.end_to_end_evaluation.records_sha256 `
        -or [string]$endToEndSummary.manifest_sha256 -ne $actualManifestSha256 `
        -or $endToEndSummary.artifact_audit.all_results_match_model -ne $true `
        -or $endToEndSummary.accepted -ne $true `
        -or $endToEndSummary.acceptance.passed -ne $true `
        -or $endToEndFailures.Count -ne 0) {
        throw "Package validation, end-to-end evidence, and delivered unified OCR artifacts are not cryptographically bound."
    }
    if ([string]$onnxValidation.model_sha256 -ne [string]$unifiedEvidence.ModelSha256 `
        -or [string]$onnxValidation.evaluation_split -ne "val" `
        -or $onnxValidation.acceptance.requested -ne $true `
        -or $onnxValidation.acceptance.passed -ne $true `
        -or $validation.onnx_validation.accepted -ne $true `
        -or $onnxFailures.Count -ne 0) {
        throw "Packaged ONNX validation is not accepted and bound to the delivered unified OCR model."
    }
    $onnxSummaryShaProperty = $validation.onnx_validation.PSObject.Properties["summary_sha256"]
    if (($null -ne $configDeclarationProperty -and $null -eq $onnxSummaryShaProperty) `
        -or ($null -ne $onnxSummaryShaProperty `
            -and [string]$onnxSummaryShaProperty.Value -ne $actualOnnxValidationSha256)) {
        throw "Package validation onnx_validation.summary_sha256 does not match the packaged ONNX summary."
    }

    foreach ($fieldName in @("amount", "time", "payment_method_field", "recipient_field")) {
        $validationMetricProperty = $validation.end_to_end_evaluation.metrics.PSObject.Properties[$fieldName]
        $summaryMetricProperty = $endToEndSummary.by_field.PSObject.Properties[$fieldName]
        if ($null -eq $validationMetricProperty -or $null -eq $summaryMetricProperty) {
            throw "End-to-end evidence is missing the $fieldName metric."
        }
        $validationMetric = $validationMetricProperty.Value
        $summaryMetric = $summaryMetricProperty.Value
        if ([int]$validationMetric.records -ne [int]$summaryMetric.records `
            -or [double]$validationMetric.exact_match -ne [double]$summaryMetric.raw_exact_match `
            -or [double]$validationMetric.candidate_coverage -ne 1.0 `
            -or [double]$summaryMetric.candidate_coverage -ne 1.0) {
            throw "Package validation and end-to-end $fieldName metrics do not match."
        }
    }
    $legacyResultCount = 0
    if ($null -eq $configDeclarationProperty) {
        $legacyResultCount = Read-LegacyResultEvidence `
            $resultEvidencePath $inferenceManifestPath $validation `
            $detectorEvidence $deviceEvidence $unifiedEvidence
    }

    return [pscustomobject]@{
        Config = $config
        Validation = $validation
        ConfigPath = $configPath
        ValidationPath = $validationPath
        Detector = $detectorEvidence
        Device = $deviceEvidence
        Unified = $unifiedEvidence
        OnnxValidationPath = $onnxValidationPath
        EndToEndSummaryPath = $endToEndSummaryPath
        EndToEndComparisonsPath = $endToEndComparisonsPath
        InferenceManifestPath = $inferenceManifestPath
        ResultEvidencePath = $resultEvidencePath
        ModelBindingMode = $modelBindingMode
        LegacyResultCount = $legacyResultCount
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$singleEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-single-cpu.ps1"
$batchEntrypoint = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\run-receipt-batch-cpu.ps1"
$deliveryReadme = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\DeliveryScripts\README-CPU.md"
Assert-SafePathSyntax $SourceDeliveryDir "SourceDeliveryDir"
Assert-SafePathSyntax $DestinationDeliveryDir "DestinationDeliveryDir"
$SourceDeliveryDir = [IO.Path]::GetFullPath($SourceDeliveryDir)
$DestinationDeliveryDir = [IO.Path]::GetFullPath($DestinationDeliveryDir)
Assert-NoReparsePointInExistingPath $SourceDeliveryDir "SourceDeliveryDir"
Assert-NoReparsePointInExistingPath $DestinationDeliveryDir "DestinationDeliveryDir"

if (-not (Test-Path -LiteralPath $SourceDeliveryDir -PathType Container)) {
    throw "Missing source delivery package: $SourceDeliveryDir"
}
if ((Test-PathWithin $DestinationDeliveryDir $SourceDeliveryDir) `
    -or (Test-PathWithin $SourceDeliveryDir $DestinationDeliveryDir)) {
    throw "Source and destination delivery directories must be separate, non-nested paths."
}
if (Test-Path -LiteralPath $DestinationDeliveryDir) {
    throw "Refusing to overwrite an existing delivery package: $DestinationDeliveryDir"
}
Require-File $singleEntrypoint "single-image production CPU entrypoint"
Require-File $batchEntrypoint "batch production CPU entrypoint"
Require-File $deliveryReadme "production CPU delivery README"
Assert-NoReparsePointInExistingPath $singleEntrypoint "single-image production CPU entrypoint"
Assert-NoReparsePointInExistingPath $batchEntrypoint "batch production CPU entrypoint"
Assert-NoReparsePointInExistingPath $deliveryReadme "production CPU delivery README"

$sourceHashesPath = Join-Path $SourceDeliveryDir "SHA256SUMS.json"
Require-File $sourceHashesPath "source package hash manifest"
Assert-PackageIntegrity $SourceDeliveryDir
$null = Read-AcceptedProductionPackage $SourceDeliveryDir
$sourceHashesSha256 = Get-Sha256 $sourceHashesPath

$destinationParent = Split-Path -Parent $DestinationDeliveryDir
if ([string]::IsNullOrWhiteSpace($destinationParent)) {
    throw "DestinationDeliveryDir must have a parent directory."
}
New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
Assert-NoReparsePointInExistingPath $destinationParent "destination parent"
$stagingRoot = Join-Path $destinationParent (".receipt-mlnet-entrypoints-staging-" + [Guid]::NewGuid().ToString("N"))
$published = $false

try {
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $SourceDeliveryDir -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination $stagingRoot -Recurse -Force
    }

    $stagingHashesPath = Join-Path $stagingRoot "SHA256SUMS.json"
    Require-File $stagingHashesPath "copied source package hash manifest"
    if ((Get-Sha256 $stagingHashesPath) -ne $sourceHashesSha256) {
        throw "Source package hash manifest changed while the staging snapshot was copied."
    }
    Assert-PackageIntegrity $stagingRoot
    $stagingSourceEvidence = Read-AcceptedProductionPackage $stagingRoot
    $sourcePackageConfigSha256 = Get-Sha256 $stagingSourceEvidence.ConfigPath
    $sourcePackageValidationSha256 = Get-Sha256 $stagingSourceEvidence.ValidationPath

    Copy-Item -LiteralPath $singleEntrypoint -Destination $stagingRoot -Force
    Copy-Item -LiteralPath $batchEntrypoint -Destination $stagingRoot -Force
    Copy-Item -LiteralPath $deliveryReadme -Destination $stagingRoot -Force

    $modelArtifactDeclaration = [ordered]@{
        detector = [ordered]@{
            kind = [string]$stagingSourceEvidence.Detector.Kind
            model_path = Get-RelativePackagePath $stagingSourceEvidence.Detector.ModelPath $stagingRoot
            model_sha256 = [string]$stagingSourceEvidence.Detector.ModelSha256
            contract_path = Get-RelativePackagePath $stagingSourceEvidence.Detector.ContractPath $stagingRoot
            contract_sha256 = [string]$stagingSourceEvidence.Detector.ContractSha256
        }
        device = [ordered]@{
            kind = [string]$stagingSourceEvidence.Device.Kind
            model_path = Get-RelativePackagePath $stagingSourceEvidence.Device.ModelPath $stagingRoot
            model_sha256 = [string]$stagingSourceEvidence.Device.ModelSha256
            contract_path = Get-RelativePackagePath $stagingSourceEvidence.Device.ContractPath $stagingRoot
            contract_sha256 = [string]$stagingSourceEvidence.Device.ContractSha256
        }
        unified_ocr = [ordered]@{
            kind = [string]$stagingSourceEvidence.Unified.Kind
            model_path = Get-RelativePackagePath $stagingSourceEvidence.Unified.ModelPath $stagingRoot
            model_sha256 = [string]$stagingSourceEvidence.Unified.ModelSha256
            labels_path = Get-RelativePackagePath $stagingSourceEvidence.Unified.LabelsPath $stagingRoot
            labels_sha256 = [string]$stagingSourceEvidence.Unified.LabelsSha256
            contract_path = Get-RelativePackagePath $stagingSourceEvidence.Unified.ContractPath $stagingRoot
            contract_sha256 = [string]$stagingSourceEvidence.Unified.ContractSha256
            text_delivery_policy = [string]$stagingSourceEvidence.Unified.TextDeliveryPolicy
            review_value = [string]$stagingSourceEvidence.Unified.ReviewValue
        }
    }
    $stagingConfigPath = $stagingSourceEvidence.ConfigPath
    $stagingConfig = $stagingSourceEvidence.Config
    $stagingConfig | Add-Member -NotePropertyName production_entrypoints -NotePropertyValue @(
        [IO.Path]::GetFileName($singleEntrypoint),
        [IO.Path]::GetFileName($batchEntrypoint)
    ) -Force
    $stagingConfig | Add-Member -NotePropertyName delivery_readme `
        -NotePropertyValue ([IO.Path]::GetFileName($deliveryReadme)) -Force
    $stagingConfig | Add-Member -NotePropertyName text_review_value `
        -NotePropertyValue ([string]$stagingSourceEvidence.Unified.ReviewValue) -Force
    $stagingConfig | Add-Member -NotePropertyName model_artifacts `
        -NotePropertyValue $modelArtifactDeclaration -Force
    $stagingConfig | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $stagingConfigPath -Encoding UTF8
    $augmentedPackageConfigSha256 = Get-Sha256 $stagingConfigPath

    $sourceValidation = $stagingSourceEvidence.Validation
    $sourceValidation | Add-Member -NotePropertyName model_artifacts `
        -NotePropertyValue $modelArtifactDeclaration -Force
    $sourceValidation.onnx_validation | Add-Member -NotePropertyName summary_sha256 `
        -NotePropertyValue (Get-Sha256 $stagingSourceEvidence.OnnxValidationPath) -Force
    $sourceValidation | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $stagingSourceEvidence.ValidationPath -Encoding UTF8
    $augmentedPackageValidationSha256 = Get-Sha256 $stagingSourceEvidence.ValidationPath
    $augmentation = [ordered]@{
        schema_version = 1
        kind = "receipt_mlnet_production_entrypoint_augmentation_v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        source_delivery = $SourceDeliveryDir
        source_sha256s_sha256 = $sourceHashesSha256
        source_package_config_sha256 = $sourcePackageConfigSha256
        augmented_package_config_sha256 = $augmentedPackageConfigSha256
        source_package_validation_sha256 = $sourcePackageValidationSha256
        augmented_package_validation_sha256 = $augmentedPackageValidationSha256
        source_model_binding_mode = [string]$stagingSourceEvidence.ModelBindingMode
        legacy_results_reverified = [int]$stagingSourceEvidence.LegacyResultCount
        validation_scope = [string]$sourceValidation.validation_scope
        runtime_flavor = [string]$sourceValidation.runtime_flavor
        runtime_device = [string]$sourceValidation.runtime_device
        rectification = [string]$sourceValidation.rectification
        orientation_rule = [string]$sourceValidation.orientation_rule
        end_to_end_status = [string]$sourceValidation.end_to_end_evaluation.status
        models = $modelArtifactDeclaration
        source_evidence = [ordered]@{
            onnx_validation = [ordered]@{
                path = Get-RelativePackagePath $stagingSourceEvidence.OnnxValidationPath $stagingRoot
                sha256 = Get-Sha256 $stagingSourceEvidence.OnnxValidationPath
                model_sha256 = [string]$stagingSourceEvidence.Unified.ModelSha256
            }
            end_to_end_summary = [ordered]@{
                path = Get-RelativePackagePath $stagingSourceEvidence.EndToEndSummaryPath $stagingRoot
                sha256 = Get-Sha256 $stagingSourceEvidence.EndToEndSummaryPath
                records_sha256 = [string]$sourceValidation.end_to_end_evaluation.records_sha256
                model_sha256 = [string]$sourceValidation.end_to_end_evaluation.model_sha256
            }
            end_to_end_comparisons = [ordered]@{
                path = Get-RelativePackagePath $stagingSourceEvidence.EndToEndComparisonsPath $stagingRoot
                sha256 = Get-Sha256 $stagingSourceEvidence.EndToEndComparisonsPath
            }
            inference_manifest = [ordered]@{
                path = Get-RelativePackagePath $stagingSourceEvidence.InferenceManifestPath $stagingRoot
                sha256 = Get-Sha256 $stagingSourceEvidence.InferenceManifestPath
            }
            result_hashes = [ordered]@{
                path = Get-RelativePackagePath $stagingSourceEvidence.ResultEvidencePath $stagingRoot
                sha256 = Get-Sha256 $stagingSourceEvidence.ResultEvidencePath
            }
        }
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

    Remove-Item -LiteralPath $stagingHashesPath -Force
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
        Set-Content -LiteralPath $stagingHashesPath -Encoding UTF8
    Assert-PackageIntegrity $stagingRoot
    $null = Read-AcceptedProductionPackage $stagingRoot

    if (Test-Path -LiteralPath $DestinationDeliveryDir) {
        throw "Destination appeared during augmentation; refusing to overwrite it: $DestinationDeliveryDir"
    }
    Assert-NoReparsePointInExistingPath $stagingRoot "staging delivery package"
    Assert-NoReparsePointInExistingPath $DestinationDeliveryDir "DestinationDeliveryDir"
    [IO.Directory]::Move($stagingRoot, $DestinationDeliveryDir)
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
