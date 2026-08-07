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

function Assert-DeclaredModelArtifacts(
    [object]$Declaration,
    [string]$PackageRoot,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence,
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
}

function Assert-AcceptedPackageBinding(
    [string]$PackageRoot,
    [object]$Config,
    [object]$Validation,
    [object]$DetectorEvidence,
    [object]$DeviceEvidence,
    [object]$UnifiedEvidence
) {
    $configDeclaration = $Config.PSObject.Properties["model_artifacts"]
    $validationDeclaration = $Validation.PSObject.Properties["model_artifacts"]
    $reviewValueProperty = $Config.PSObject.Properties["text_review_value"]
    if ($null -eq $configDeclaration `
        -or $null -eq $validationDeclaration `
        -or $null -eq $reviewValueProperty `
        -or [string]$reviewValueProperty.Value -ne [string]$UnifiedEvidence.ReviewValue) {
        throw "Package lacks an explicit config/validation model and review-policy binding."
    }
    Assert-DeclaredModelArtifacts `
        $configDeclaration.Value $PackageRoot `
        $DetectorEvidence $DeviceEvidence $UnifiedEvidence "Package config"
    Assert-DeclaredModelArtifacts `
        $validationDeclaration.Value $PackageRoot `
        $DetectorEvidence $DeviceEvidence $UnifiedEvidence "Package validation"

    $onnxValidationPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/onnx-validation-summary.json" "ONNX validation summary"
    $endToEndSummaryPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-evaluation-summary.json" "end-to-end evaluation summary"
    $endToEndComparisonsPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/end-to-end-comparisons.jsonl" "end-to-end comparisons"
    $inferenceManifestPath = Resolve-ContainedPackageFile `
        $PackageRoot "evidence/inference_manifest.json" "inference manifest"
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
    $manifestSha256 = Get-Sha256 $inferenceManifestPath
    if ([string]$Validation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$Validation.end_to_end_evaluation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$Validation.onnx_validation.summary_sha256 -ne (Get-Sha256 $onnxValidationPath) `
        -or [string]$Validation.end_to_end_evaluation.summary_sha256 -ne (Get-Sha256 $endToEndSummaryPath) `
        -or [string]$Validation.end_to_end_evaluation.comparisons_sha256 -ne (Get-Sha256 $endToEndComparisonsPath) `
        -or [string]$Validation.end_to_end_evaluation.manifest_sha256 -ne $manifestSha256 `
        -or [string]$onnxValidation.model_sha256 -ne [string]$UnifiedEvidence.ModelSha256 `
        -or [string]$onnxValidation.evaluation_split -ne "val" `
        -or $onnxValidation.acceptance.requested -ne $true `
        -or $onnxValidation.acceptance.passed -ne $true `
        -or $onnxFailures.Count -ne 0 `
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
        -or $null -eq $contractsProperty.Value) {
        throw "Result does not prove the complete three-model ML.NET CPU path: $ResultPath"
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
        $statusValue = if ($null -eq $valueProperty) { $null } else { $valueProperty.Value }
        $statusDeliveryValue = if ($null -eq $deliveryValueProperty) { $null } else { $deliveryValueProperty.Value }
        if ($null -eq $policyProperty `
            -or [string]$policyProperty.Value -ne [string]$UnifiedEvidence.StatusTextDeliveryPolicy `
            -or $null -eq $stateProperty) {
            throw "V13 result transfer_status has incomplete review-only policy evidence: $ResultPath"
        }
        if ([string]$stateProperty.Value -eq "absent") {
            if ($null -ne $statusValue -or $null -ne $statusDeliveryValue) {
                throw "V13 result transfer_status delivered a value while absent: $ResultPath"
            }
        }
        elseif ([string]$stateProperty.Value -ne "review" `
            -or [string]$statusValue -ne [string]$UnifiedEvidence.StatusTextReviewValue `
            -or [string]$statusDeliveryValue -ne [string]$UnifiedEvidence.StatusTextReviewValue) {
            throw "V13 result transfer_status escaped the review-only policy: $ResultPath"
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
if ([string]$config.kind -ne "receipt_mlnet_unified_delivery_package_v1" `
    -or [string]$config.validation_scope -ne "full_val_end_to_end_scored_cpu" `
    -or [string]$config.onnx_runtime_flavor -ne "cpu" `
    -or [string]$config.runtime_device -ne "cpu" `
    -or [string]$config.rectification -ne "max-side-1600" `
    -or [string]$config.orientation_rule -ne "exif_upright_landscape_clockwise_90" `
    -or [string]::IsNullOrWhiteSpace([string]$config.device_model) `
    -or [string]$config.text_delivery_policy -ne $requiredTextPolicy) {
    throw "This is not an accepted three-model production CPU package."
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
Assert-AcceptedPackageBinding `
    $packageRoot $config $validation $detectorEvidence $deviceEvidence $unifiedEvidence

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
    $result $resultPath $manifestSource $detectorEvidence $deviceEvidence $unifiedEvidence

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
Write-Host ("CPU latency: {0:N2} ms" -f $meanLatencyMs)
Write-Host "Result JSON : $resultPath"
Write-Host "Annotated   : $annotatedOriginalPath"
Write-Host "Rectified   : $annotatedRectifiedPath"
Write-Host "Policy      : candidates are shown for verification; business values remain fail-closed as review."
Write-Host ""
Write-Host "PASS: the complete three-model CPU pipeline produced one reviewable receipt." -ForegroundColor Green
