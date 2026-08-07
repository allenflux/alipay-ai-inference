[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [Parameter(Mandatory = $true)]
    [string]$PaddleDeliveryBundle,
    [string]$HybridAbEvidence,
    [string]$DotnetExe,
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

if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    $dotnetCommand = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($null -ne $dotnetCommand) {
        $DotnetExe = $dotnetCommand.Source
    }
    else {
        $portableDotnet = Join-Path $repoRoot "artifacts\dotnet8\dotnet.exe"
        if (Test-Path -LiteralPath $portableDotnet -PathType Leaf) {
            $DotnetExe = $portableDotnet
        }
    }
}
if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    throw "Missing .NET 8 host. Install dotnet, place the portable host at artifacts\dotnet8\dotnet.exe, or pass -DotnetExe."
}

if ([string]::IsNullOrWhiteSpace($DetectorModel)) {
    $DetectorModel = Join-Path $repoRoot "artifacts\receipt_lrcnn_v1.onnx"
}
if ([string]::IsNullOrWhiteSpace($DeviceModel)) {
    $DeviceModel = Join-Path $repoRoot "artifacts\statusbar_device_v1.onnx"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RequiredJsonProperty([object]$Object, [string]$Name, [string]$Description) {
    if ($null -eq $Object) {
        throw "$Description must be a JSON object."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Description is missing required property '$Name'."
    }
    return $property.Value
}

function Assert-JsonInteger([object]$Value, [string]$Description, [long]$Minimum = 0) {
    if (($Value -isnot [sbyte]) -and ($Value -isnot [byte]) `
        -and ($Value -isnot [int16]) -and ($Value -isnot [uint16]) `
        -and ($Value -isnot [int32]) -and ($Value -isnot [uint32]) `
        -and ($Value -isnot [int64]) -and ($Value -isnot [uint64])) {
        throw "$Description must be a JSON integer."
    }
    if ([decimal]$Value -lt [decimal]$Minimum) {
        throw "$Description must be >= $Minimum."
    }
}

function Assert-JsonNumber([object]$Value, [string]$Description) {
    if ($Value -is [bool] -or $Value -is [string] -or $null -eq $Value) {
        throw "$Description must be a finite JSON number."
    }
    $number = [double]0
    if (-not [double]::TryParse(
            [Convert]::ToString($Value, [Globalization.CultureInfo]::InvariantCulture),
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$number) `
        -or [double]::IsNaN($number) `
        -or [double]::IsInfinity($number)) {
        throw "$Description must be a finite JSON number."
    }
}

function Assert-JsonString([object]$Value, [string]$Description) {
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        throw "$Description must be a non-empty JSON string."
    }
}

function Assert-JsonBoolean([object]$Value, [string]$Description) {
    if ($Value -isnot [bool]) {
        throw "$Description must be a JSON boolean."
    }
}

function Assert-JsonSha256([object]$Value, [string]$Description) {
    Assert-JsonString $Value $Description
    if ([string]$Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be exactly 64 lowercase hexadecimal characters."
    }
}

function Assert-JsonArray([object]$Value, [string]$Description) {
    if ($Value -isnot [Array]) {
        throw "$Description must be a JSON array."
    }
}

function Assert-RequiredJsonArray([object]$Object, [string]$Name, [string]$Description) {
    if ($null -eq $Object) {
        throw "$Description must be a JSON object."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $property.Value -isnot [Array]) {
        throw "$Description.$Name must be a JSON array."
    }
}

function Assert-HybridAbComparisonSchema([object]$Document) {
    Assert-JsonInteger (Get-RequiredJsonProperty $Document "schema_version" "Hybrid A/B summary") `
        "Hybrid A/B summary schema_version" 1
    Assert-JsonString (Get-RequiredJsonProperty $Document "kind" "Hybrid A/B summary") `
        "Hybrid A/B summary kind"
    Assert-JsonString (Get-RequiredJsonProperty $Document "evaluation_mode" "Hybrid A/B summary") `
        "Hybrid A/B summary evaluation_mode"
    foreach ($name in @("records", "invariant_records")) {
        Assert-JsonInteger (Get-RequiredJsonProperty $Document $name "Hybrid A/B summary") `
            "Hybrid A/B summary $name" 1
    }
    foreach ($name in @("input_set_identical", "cli_summary_counts_verified", "accepted")) {
        Assert-JsonBoolean (Get-RequiredJsonProperty $Document $name "Hybrid A/B summary") `
            "Hybrid A/B summary $name"
    }
    Assert-JsonNumber (Get-RequiredJsonProperty $Document "recipient_candidate_coverage" "Hybrid A/B summary") `
        "Hybrid A/B summary recipient_candidate_coverage"
    Assert-RequiredJsonArray $Document "failures" "Hybrid A/B summary"

    $inputSet = Get-RequiredJsonProperty $Document "input_set" "Hybrid A/B summary"
    Assert-JsonInteger (Get-RequiredJsonProperty $inputSet "records" "Hybrid A/B input_set") `
        "Hybrid A/B input_set records" 1
    Assert-JsonSha256 (Get-RequiredJsonProperty $inputSet "normalized_source_set_sha256" "Hybrid A/B input_set") `
        "Hybrid A/B normalized source-set SHA-256"
    $inputManifest = Get-RequiredJsonProperty $inputSet "input_manifest" "Hybrid A/B input_set"
    foreach ($name in @("path")) {
        Assert-JsonString (Get-RequiredJsonProperty $inputManifest $name "Hybrid A/B input manifest") `
            "Hybrid A/B input manifest $name"
    }
    Assert-JsonSha256 (Get-RequiredJsonProperty $inputManifest "sha256" "Hybrid A/B input manifest") `
        "Hybrid A/B input manifest SHA-256"
    foreach ($name in @("size_bytes", "records")) {
        Assert-JsonInteger (Get-RequiredJsonProperty $inputManifest $name "Hybrid A/B input manifest") `
            "Hybrid A/B input manifest $name" 1
    }
    Assert-JsonSha256 `
        (Get-RequiredJsonProperty $inputManifest "normalized_source_set_sha256" "Hybrid A/B input manifest") `
        "Hybrid A/B input manifest normalized source-set SHA-256"

    $runManifests = Get-RequiredJsonProperty $Document "run_manifests" "Hybrid A/B summary"
    foreach ($variant in @("baseline", "hybrid")) {
        $manifest = Get-RequiredJsonProperty $runManifests $variant "Hybrid A/B run_manifests"
        Assert-JsonString (Get-RequiredJsonProperty $manifest "path" "Hybrid A/B $variant manifest") `
            "Hybrid A/B $variant manifest path"
        Assert-JsonSha256 (Get-RequiredJsonProperty $manifest "sha256" "Hybrid A/B $variant manifest") `
            "Hybrid A/B $variant manifest SHA-256"
        foreach ($name in @("size_bytes", "records")) {
            Assert-JsonInteger (Get-RequiredJsonProperty $manifest $name "Hybrid A/B $variant manifest") `
                "Hybrid A/B $variant manifest $name" 1
        }
        Assert-JsonSha256 `
            (Get-RequiredJsonProperty $manifest "normalized_source_set_sha256" "Hybrid A/B $variant manifest") `
            "Hybrid A/B $variant manifest normalized source-set SHA-256"
    }
    $runSummaries = Get-RequiredJsonProperty $Document "run_summaries" "Hybrid A/B summary"
    foreach ($variant in @("baseline", "hybrid")) {
        $runtimeSummary = Get-RequiredJsonProperty $runSummaries $variant "Hybrid A/B run_summaries"
        Assert-JsonString (Get-RequiredJsonProperty $runtimeSummary "path" "Hybrid A/B $variant runtime summary") `
            "Hybrid A/B $variant runtime summary path"
        Assert-JsonSha256 (Get-RequiredJsonProperty $runtimeSummary "sha256" "Hybrid A/B $variant runtime summary") `
            "Hybrid A/B $variant runtime summary SHA-256"
        Assert-JsonInteger (Get-RequiredJsonProperty $runtimeSummary "size_bytes" "Hybrid A/B $variant runtime summary") `
            "Hybrid A/B $variant runtime summary size_bytes" 1
    }
    $cliBuild = Get-RequiredJsonProperty $Document "cli_build" "Hybrid A/B summary"
    $assembly = Get-RequiredJsonProperty $cliBuild "assembly" "Hybrid A/B cli_build"
    Assert-JsonString (Get-RequiredJsonProperty $assembly "path" "Hybrid A/B CLI assembly") `
        "Hybrid A/B CLI assembly path"
    Assert-JsonSha256 (Get-RequiredJsonProperty $assembly "sha256" "Hybrid A/B CLI assembly") `
        "Hybrid A/B CLI assembly SHA-256"
    Assert-JsonInteger (Get-RequiredJsonProperty $assembly "size_bytes" "Hybrid A/B CLI assembly") `
        "Hybrid A/B CLI assembly size_bytes" 1
    $appClosure = Get-RequiredJsonProperty $cliBuild "app_closure" "Hybrid A/B cli_build"
    Assert-JsonString (Get-RequiredJsonProperty $appClosure "root" "Hybrid A/B CLI app closure") `
        "Hybrid A/B CLI app closure root"
    Assert-JsonSha256 (Get-RequiredJsonProperty $appClosure "closure_sha256" "Hybrid A/B CLI app closure") `
        "Hybrid A/B CLI app closure digest"
    Assert-JsonInteger (Get-RequiredJsonProperty $appClosure "file_count" "Hybrid A/B CLI app closure") `
        "Hybrid A/B CLI app closure file_count" 1
    $closureManifest = Get-RequiredJsonProperty $appClosure "manifest" "Hybrid A/B CLI app closure"
    Assert-JsonString (Get-RequiredJsonProperty $closureManifest "path" "Hybrid A/B CLI closure manifest") `
        "Hybrid A/B CLI closure manifest path"
    Assert-JsonSha256 (Get-RequiredJsonProperty $closureManifest "sha256" "Hybrid A/B CLI closure manifest") `
        "Hybrid A/B CLI closure manifest SHA-256"
    Assert-JsonInteger (Get-RequiredJsonProperty $closureManifest "size_bytes" "Hybrid A/B CLI closure manifest") `
        "Hybrid A/B CLI closure manifest size_bytes" 1

    $artifactHashes = Get-RequiredJsonProperty $Document "artifact_hashes" "Hybrid A/B summary"
    foreach ($name in @(
            "detector_sha256", "detector_contract_sha256", "device_sha256", "device_contract_sha256",
            "unified_ocr_model_sha256", "unified_ocr_labels_sha256", "unified_ocr_contract_sha256"
        )) {
        Assert-JsonSha256 (Get-RequiredJsonProperty $artifactHashes $name "Hybrid A/B artifact_hashes") `
            "Hybrid A/B artifact_hashes.$name"
    }
    $paddle = Get-RequiredJsonProperty $Document "paddle_delivery" "Hybrid A/B summary"
    foreach ($name in @("directory", "contract")) {
        Assert-JsonString (Get-RequiredJsonProperty $paddle $name "Hybrid A/B paddle_delivery") `
            "Hybrid A/B paddle_delivery.$name"
    }
    Assert-JsonSha256 (Get-RequiredJsonProperty $paddle "contract_sha256" "Hybrid A/B paddle_delivery") `
        "Hybrid A/B paddle_delivery.contract_sha256"
    Assert-JsonInteger (Get-RequiredJsonProperty $paddle "package_size_bytes" "Hybrid A/B paddle_delivery") `
        "Hybrid A/B paddle_delivery.package_size_bytes" 1
    $paddleModels = Get-RequiredJsonProperty $paddle "models" "Hybrid A/B paddle_delivery"
    foreach ($role in @("det", "cls", "rec")) {
        $record = Get-RequiredJsonProperty $paddleModels $role "Hybrid A/B paddle_delivery.models"
        Assert-JsonString (Get-RequiredJsonProperty $record "path" "Hybrid A/B PP-OCR $role") `
            "Hybrid A/B PP-OCR $role path"
        Assert-JsonSha256 (Get-RequiredJsonProperty $record "sha256" "Hybrid A/B PP-OCR $role") `
            "Hybrid A/B PP-OCR $role SHA-256"
        Assert-JsonInteger (Get-RequiredJsonProperty $record "size_bytes" "Hybrid A/B PP-OCR $role") `
            "Hybrid A/B PP-OCR $role size_bytes" 1
    }
    $dictionary = Get-RequiredJsonProperty $paddle "dictionary" "Hybrid A/B paddle_delivery"
    Assert-JsonString (Get-RequiredJsonProperty $dictionary "path" "Hybrid A/B PP-OCR dictionary") `
        "Hybrid A/B PP-OCR dictionary path"
    Assert-JsonSha256 (Get-RequiredJsonProperty $dictionary "sha256" "Hybrid A/B PP-OCR dictionary") `
        "Hybrid A/B PP-OCR dictionary SHA-256"
    Assert-JsonInteger (Get-RequiredJsonProperty $dictionary "size_bytes" "Hybrid A/B PP-OCR dictionary") `
        "Hybrid A/B PP-OCR dictionary size_bytes" 1
    $cpu = Get-RequiredJsonProperty $Document "cpu" "Hybrid A/B summary"
    foreach ($name in @("p95_overhead_ms", "max_p95_overhead_ms")) {
        Assert-JsonNumber (Get-RequiredJsonProperty $cpu $name "Hybrid A/B cpu") "Hybrid A/B cpu.$name"
    }
    foreach ($name in @("baseline_inference_latency_ms", "hybrid_inference_latency_ms")) {
        $latency = Get-RequiredJsonProperty $cpu $name "Hybrid A/B cpu"
        Assert-JsonNumber (Get-RequiredJsonProperty $latency "p95" "Hybrid A/B cpu.$name") `
            "Hybrid A/B cpu.$name.p95"
    }
}

function Assert-HybridAbScoreSchema([object]$Document) {
    Assert-JsonInteger (Get-RequiredJsonProperty $Document "schema_version" "Hybrid A/B score") `
        "Hybrid A/B score schema_version" 1
    foreach ($name in @("kind", "records", "records_sha256", "results_root", "manifest", "manifest_sha256", "evaluation_split", "model", "model_sha256")) {
        $value = Get-RequiredJsonProperty $Document $name "Hybrid A/B score"
        if ($name.EndsWith("sha256", [StringComparison]::Ordinal)) {
            Assert-JsonSha256 $value "Hybrid A/B score $name"
        }
        else {
            Assert-JsonString $value "Hybrid A/B score $name"
        }
    }
    foreach ($name in @("formal_delivery_gate", "accepted")) {
        Assert-JsonBoolean (Get-RequiredJsonProperty $Document $name "Hybrid A/B score") `
            "Hybrid A/B score $name"
    }
    Assert-RequiredJsonArray $Document "failures" "Hybrid A/B score"
    $scope = Get-RequiredJsonProperty $Document "evaluation_scope" "Hybrid A/B score"
    Assert-JsonString (Get-RequiredJsonProperty $scope "kind" "Hybrid A/B score evaluation_scope") `
        "Hybrid A/B score evaluation_scope.kind"
    foreach ($name in @("evaluated_expected_receipts", "full_split_expected_receipts")) {
        Assert-JsonInteger (Get-RequiredJsonProperty $scope $name "Hybrid A/B score evaluation_scope") `
            "Hybrid A/B score evaluation_scope.$name" 1
    }
    Assert-JsonBoolean (Get-RequiredJsonProperty $scope "formal_delivery_gate" "Hybrid A/B score evaluation_scope") `
        "Hybrid A/B score evaluation_scope.formal_delivery_gate"
    $coverage = Get-RequiredJsonProperty $Document "coverage" "Hybrid A/B score"
    foreach ($name in @("expected_receipts", "matched_result_receipts", "fully_scored_receipts")) {
        Assert-JsonInteger (Get-RequiredJsonProperty $coverage $name "Hybrid A/B score coverage") `
            "Hybrid A/B score coverage.$name" 1
    }
    foreach ($name in @("result_coverage", "fully_scored_coverage")) {
        Assert-JsonNumber (Get-RequiredJsonProperty $coverage $name "Hybrid A/B score coverage") `
            "Hybrid A/B score coverage.$name"
    }
    $acceptance = Get-RequiredJsonProperty $Document "acceptance" "Hybrid A/B score"
    foreach ($name in @("passed", "formal_delivery_gate")) {
        Assert-JsonBoolean (Get-RequiredJsonProperty $acceptance $name "Hybrid A/B score acceptance") `
            "Hybrid A/B score acceptance.$name"
    }
    $audit = Get-RequiredJsonProperty $Document "artifact_audit" "Hybrid A/B score"
    Assert-JsonBoolean (Get-RequiredJsonProperty $audit "all_results_match_model" "Hybrid A/B score artifact_audit") `
        "Hybrid A/B score artifact_audit.all_results_match_model"
    $byField = Get-RequiredJsonProperty $Document "by_field" "Hybrid A/B score"
    $floors = Get-RequiredJsonProperty $Document "floors" "Hybrid A/B score"
    foreach ($field in @("amount", "time", "payment_method_field", "recipient_field", "transfer_status")) {
        $metric = Get-RequiredJsonProperty $byField $field "Hybrid A/B score by_field"
        Assert-JsonInteger (Get-RequiredJsonProperty $metric "records" "Hybrid A/B score $field") `
            "Hybrid A/B score $field.records" 1
        foreach ($name in @("raw_exact_match", "candidate_coverage")) {
            Assert-JsonNumber (Get-RequiredJsonProperty $metric $name "Hybrid A/B score $field") `
                "Hybrid A/B score $field.$name"
        }
        Assert-JsonNumber (Get-RequiredJsonProperty $floors $field "Hybrid A/B score floors") `
            "Hybrid A/B score floor $field"
    }
    Assert-JsonInteger `
        (Get-RequiredJsonProperty `
            (Get-RequiredJsonProperty $byField "transfer_status" "Hybrid A/B score by_field") `
            "non_success_to_success" "Hybrid A/B score transfer_status") `
        "Hybrid A/B score transfer_status.non_success_to_success" 0
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

function Assert-CliAppClosure(
    [string]$AppRoot,
    [string]$ManifestPath,
    [string]$ExpectedManifestSha256
) {
    Assert-SafePathSyntax $AppRoot "CLI app closure root"
    Assert-SafePathSyntax $ManifestPath "CLI app closure manifest"
    $rootFull = [IO.Path]::GetFullPath($AppRoot)
    $manifestFull = [IO.Path]::GetFullPath($ManifestPath)
    if (-not (Test-Path -LiteralPath $rootFull -PathType Container)) {
        throw "Missing CLI app closure root: $rootFull"
    }
    Require-File $manifestFull "CLI app closure manifest"
    Assert-JsonSha256 $ExpectedManifestSha256 "CLI app closure manifest SHA-256"
    if ((Get-Sha256 $manifestFull) -ne $ExpectedManifestSha256) {
        throw "CLI app closure manifest SHA-256 does not match its binding."
    }
    $rows = @(Get-Content -LiteralPath $manifestFull -Raw -Encoding UTF8 | ConvertFrom-Json)
    if ($rows.Count -le 0) {
        throw "CLI app closure manifest is empty."
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
        if ($propertyNames.Count -ne 3 `
            -or $propertyNames -notcontains "path" `
            -or $propertyNames -notcontains "sha256" `
            -or $propertyNames -notcontains "size_bytes") {
            throw "CLI app closure rows must contain exactly path, sha256 and size_bytes."
        }
        Assert-JsonString $row.path "CLI app closure path"
        Assert-JsonSha256 $row.sha256 "CLI app closure file SHA-256"
        Assert-JsonInteger $row.size_bytes "CLI app closure file size_bytes" 0
        $target = Resolve-ContainedPackageFile $rootFull ([string]$row.path) "CLI app closure file"
        $relative = Get-RelativePackagePath $target $rootFull
        if ($relative -cne [string]$row.path) {
            throw "CLI app closure path is not canonical: $($row.path)"
        }
        $key = $relative.ToLowerInvariant()
        if ($listed.ContainsKey($key)) {
            throw "Duplicate CLI app closure path: $relative"
        }
        $sortKey = $key + [char]0 + $relative
        if ($null -ne $previousSortKey `
            -and [StringComparer]::Ordinal.Compare($previousSortKey, $sortKey) -ge 0) {
            throw "CLI app closure manifest paths are not canonically sorted."
        }
        $previousSortKey = $sortKey
        $item = Get-Item -LiteralPath $target
        if ([long]$item.Length -ne [long]$row.size_bytes `
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
    foreach ($file in @(Get-SafeDirectoryFiles $rootFull "CLI app closure")) {
        $relative = Get-RelativePackagePath $file.FullName $rootFull
        $key = $relative.ToLowerInvariant()
        if ($actual.ContainsKey($key)) {
            throw "Duplicate case-insensitive CLI app closure path: $relative"
        }
        $actual[$key] = $relative
    }
    $missing = @($listed.Keys | Where-Object { -not $actual.ContainsKey($_) })
    $extra = @($actual.Keys | Where-Object { -not $listed.ContainsKey($_) })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        throw "CLI app closure is not exact: missing=$($missing.Count), extra=$($extra.Count)."
    }
    $missingRequired = @($requiredBasenames.Keys | Where-Object { $requiredBasenames[$_] -ne $true })
    if ($missingRequired.Count -ne 0) {
        throw "CLI app closure lacks required managed/native runtime payload: $($missingRequired -join ',')."
    }
    if ((Get-Sha256 $manifestFull) -ne $ExpectedManifestSha256) {
        throw "CLI app closure manifest changed during verification."
    }
    return [pscustomobject]@{
        Root = $rootFull
        ManifestPath = $manifestFull
        ClosureSha256 = $ExpectedManifestSha256
        FileCount = $rows.Count
    }
}

function Assert-PaddleDeliveryFileRecord(
    [string]$BundleRoot,
    [object]$Record,
    [string]$Description,
    [bool]$RequireOnnx
) {
    if ($null -eq $Record) {
        throw "Paddle delivery contract has no ${Description} record."
    }
    $pathProperty = $Record.PSObject.Properties["path"]
    $shaProperty = $Record.PSObject.Properties["sha256"]
    $bytesProperty = $Record.PSObject.Properties["size_bytes"]
    if ($null -eq $pathProperty -or $null -eq $shaProperty -or $null -eq $bytesProperty) {
        throw "Paddle delivery ${Description} record is incomplete."
    }
    $relativePath = [string]$pathProperty.Value
    $target = Resolve-ContainedPackageFile $BundleRoot $relativePath "Paddle delivery $Description"
    if ($RequireOnnx -and [IO.Path]::GetExtension($target) -ne ".onnx") {
        throw "Paddle delivery ${Description} is not an ONNX model: $relativePath"
    }
    if (-not $RequireOnnx -and [IO.Path]::GetExtension($target) -eq ".onnx") {
        throw "Paddle delivery dictionary must not be an ONNX model: $relativePath"
    }
    $expectedHash = ([string]$shaProperty.Value).ToLowerInvariant()
    if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
        throw "Paddle delivery ${Description} has an invalid SHA-256."
    }
    $expectedBytes = [long]0
    $bytesText = [Convert]::ToString($bytesProperty.Value, [Globalization.CultureInfo]::InvariantCulture)
    if (-not [long]::TryParse(
            $bytesText,
            [Globalization.NumberStyles]::Integer,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$expectedBytes) `
        -or $expectedBytes -le 0) {
        throw "Paddle delivery ${Description} has an invalid size_bytes."
    }
    if ((Get-Sha256 $target) -ne $expectedHash `
        -or (Get-Item -LiteralPath $target).Length -ne $expectedBytes) {
        throw "Paddle delivery ${Description} does not match its contract hash/size: $relativePath"
    }
    return [pscustomobject]@{
        RelativePath = (Get-RelativePackagePath $target $BundleRoot)
        SourcePath = $target
        Sha256 = $expectedHash
        SizeBytes = $expectedBytes
    }
}

function Assert-PaddleDeliveryBundle([string]$BundleRoot) {
    Assert-SafePathSyntax $BundleRoot "PaddleDeliveryBundle"
    $bundleFull = [IO.Path]::GetFullPath($BundleRoot)
    if (-not (Test-Path -LiteralPath $bundleFull -PathType Container)) {
        throw "Missing pure-ONNX Paddle delivery bundle: $bundleFull"
    }
    Assert-NoReparsePointInExistingPath $bundleFull "PaddleDeliveryBundle"
    $contractPath = Join-Path $bundleFull "paddle_ocr_delivery.contract.json"
    Require-File $contractPath "Paddle delivery contract"
    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $modelNames = if ($null -eq $contract.models) {
        @()
    }
    else {
        @($contract.models.PSObject.Properties.Name)
    }
    $paddleSchemaVersion = Get-RequiredJsonProperty $contract "schema_version" "Paddle delivery contract"
    Assert-JsonInteger $paddleSchemaVersion "Paddle delivery contract schema_version" 1
    if ([long]$paddleSchemaVersion -ne 1 `
        -or [string]$contract.kind -ne "paddle_ocr_v2_delivery" `
        -or $modelNames.Count -ne 3 `
        -or $modelNames -notcontains "det" `
        -or $modelNames -notcontains "cls" `
        -or $modelNames -notcontains "rec") {
        throw "Paddle delivery contract is not a complete paddle_ocr_v2_delivery det/cls/rec package."
    }
    $forbiddenDependencies = @(
        $contract.forbidden_runtime_dependencies |
            ForEach-Object { [string]$_ }
    )
    foreach ($dependency in @("Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files")) {
        if ($forbiddenDependencies -notcontains $dependency) {
            throw "Paddle delivery contract does not forbid runtime dependency: $dependency"
        }
    }
    $models = [ordered]@{}
    foreach ($role in @("det", "cls", "rec")) {
        $models[$role] = Assert-PaddleDeliveryFileRecord `
            $bundleFull $contract.models.PSObject.Properties[$role].Value "$role model" $true
    }
    $dictionary = Assert-PaddleDeliveryFileRecord `
        $bundleFull $contract.dictionary "dictionary" $false
    $packageSizeProperty = $contract.PSObject.Properties["package_size_bytes"]
    if ($null -eq $packageSizeProperty) {
        throw "Paddle delivery contract is missing package_size_bytes."
    }
    Assert-JsonInteger $packageSizeProperty.Value "Paddle delivery package_size_bytes" 1
    $packageSizeBytes = [long]$packageSizeProperty.Value
    $verifiedPackageSizeBytes = [long]$dictionary.SizeBytes
    foreach ($role in @("det", "cls", "rec")) {
        $verifiedPackageSizeBytes += [long]$models[$role].SizeBytes
    }
    if ($packageSizeBytes -le 0 `
        -or $packageSizeBytes -ne $verifiedPackageSizeBytes) {
        throw "Paddle delivery package_size_bytes must be positive and exactly equal the verified det/cls/rec/dictionary payload."
    }

    # A production delivery is deliberately closed over exactly the contract,
    # three ONNX models and one dictionary.  Audit Paddle/Python assets or
    # unbound files must never be smuggled into the atomic package.
    $expectedPaths = @{}
    $expectedPaths["paddle_ocr_delivery.contract.json"] = $true
    foreach ($record in @($models["det"], $models["cls"], $models["rec"], $dictionary)) {
        $expectedPaths[[string]$record.RelativePath] = $true
    }
    $payload = @(Get-SafeDirectoryFiles $bundleFull "PaddleDeliveryBundle")
    $actualPaths = @{}
    foreach ($file in $payload) {
        $relative = Get-RelativePackagePath $file.FullName $bundleFull
        $actualPaths[$relative] = $true
    }
    $missing = @($expectedPaths.Keys | Where-Object { -not $actualPaths.ContainsKey($_) })
    $extra = @($actualPaths.Keys | Where-Object { -not $expectedPaths.ContainsKey($_) })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        throw "Paddle delivery payload is not closed over pure ONNX det/cls/rec + dictionary: missing=$($missing.Count), extra=$($extra.Count)."
    }
    return [pscustomobject]@{
        Root = $bundleFull
        ContractPath = $contractPath
        ContractSha256 = Get-Sha256 $contractPath
        PackageSizeBytes = $packageSizeBytes
        Models = $models
        Dictionary = $dictionary
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
$hasHybridAbEvidence = -not [string]::IsNullOrWhiteSpace($HybridAbEvidence)
$productionCpuEntrypointsRequested = $hasRecords -and $RuntimeFlavor -eq "cpu"
$includeProductionCpuEntrypoints = $false
$requestedRecordsSha256 = $null
$hasExplicitUnifiedModel = -not [string]::IsNullOrWhiteSpace($UnifiedModelPath)
$hasExplicitOnnxValidationSummary = -not [string]::IsNullOrWhiteSpace($OnnxValidationSummaryPath)
$usesExplicitUnifiedArtifactBinding = $hasExplicitUnifiedModel -and $hasExplicitOnnxValidationSummary
$minimumAmountFloor = 0.7885
$minimumTimeFloor = 0.9840
$minimumPaymentFloor = 0.9325
$minimumRecipientFloor = 0.90
$requiredStatusTextFloor = 0.90
$requiredFormalReceiptCount = 10016

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
if ($RuntimeFlavor -ne "cpu") {
    throw "Hybrid recipient delivery packaging is CPU-only; both unified and PP-OCR providers must be cpu."
}
if ($hasRecords -and $Rectification -ne "max-side-1600") {
    throw "Formal end-to-end delivery validation requires -Rectification max-side-1600."
}
if ($hasRecords -and -not $IncludeDeviceModel) {
    throw "Formal end-to-end delivery validation requires -IncludeDeviceModel for the complete three-model pipeline."
}
if (-not $IncludeDeviceModel) {
    throw "Hybrid recipient delivery validation must not skip the device model. Pass -IncludeDeviceModel."
}
if ($hasRecords -and -not $hasHybridAbEvidence) {
    throw "Formal hybrid delivery requires -HybridAbEvidence from the complete CPU formal A/B run."
}
if ($hasHybridAbEvidence -and -not $hasRecords) {
    throw "HybridAbEvidence is formal-only and requires -Records plus -EndToEndEvaluationDir."
}
$orientationRule = if ($Rectification -eq "max-side-1600") {
    "exif_upright_landscape_clockwise_90"
} else {
    "none"
}

Assert-SafePathSyntax $RunDirectory "RunDirectory"
Assert-SafePathSyntax $Output "Output"
Assert-SafePathSyntax $DeliveryDir "DeliveryDir"
Assert-SafePathSyntax $PaddleDeliveryBundle "PaddleDeliveryBundle"
Assert-SafePathSyntax $DotnetExe "DotnetExe"
Assert-SafePathSyntax $DetectorModel "DetectorModel"
Assert-SafePathSyntax $DeviceModel "DeviceModel"
if ($hasHybridAbEvidence) {
    Assert-SafePathSyntax $HybridAbEvidence "HybridAbEvidence"
}
if ($usesExplicitUnifiedArtifactBinding) {
    Assert-SafePathSyntax $UnifiedModelPath "UnifiedModelPath"
    Assert-SafePathSyntax $OnnxValidationSummaryPath "OnnxValidationSummaryPath"
}
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
$Output = [IO.Path]::GetFullPath($Output)
$DeliveryDir = [IO.Path]::GetFullPath($DeliveryDir)
$PaddleDeliveryBundle = [IO.Path]::GetFullPath($PaddleDeliveryBundle)
$DotnetExe = [IO.Path]::GetFullPath($DotnetExe)
$DetectorModel = [IO.Path]::GetFullPath($DetectorModel)
$DeviceModel = [IO.Path]::GetFullPath($DeviceModel)
Assert-NoReparsePointInExistingPath $RunDirectory "RunDirectory"
Assert-NoReparsePointInExistingPath $Output "Output"
Assert-NoReparsePointInExistingPath $DeliveryDir "DeliveryDir"
Assert-NoReparsePointInExistingPath $PaddleDeliveryBundle "PaddleDeliveryBundle"
Assert-NoReparsePointInExistingPath $DotnetExe "DotnetExe"
Assert-NoReparsePointInExistingPath $DetectorModel "DetectorModel"
Assert-NoReparsePointInExistingPath $DeviceModel "DeviceModel"
if ($hasHybridAbEvidence) {
    $HybridAbEvidence = [IO.Path]::GetFullPath($HybridAbEvidence)
    Assert-NoReparsePointInExistingPath $HybridAbEvidence "HybridAbEvidence"
    if (-not (Test-Path -LiteralPath $HybridAbEvidence -PathType Container)) {
        throw "Missing hybrid CPU A/B evidence directory: $HybridAbEvidence"
    }
}
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
Require-File $DotnetExe ".NET 8 host"
Require-File $normalizer "JSON normalizer"
Require-File $projectFile "ML.NET project"
Require-File $preprocessingContractTestProject "ML.NET preprocessing contract test project"
Require-File $rectificationContractTestProject "ML.NET rectification contract test project"

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
if ($unifiedArchitectureVersion -ne 13) {
    throw "Hybrid recipient delivery requires an architecture-v13 unified model for amount/time/payment/status OCR."
}
$paddleDelivery = Assert-PaddleDeliveryBundle $PaddleDeliveryBundle
$paddleDeliveryContract = [string]$paddleDelivery.ContractPath
$paddleDeliveryContractSha256 = [string]$paddleDelivery.ContractSha256
$includeProductionCpuEntrypoints = `
    $productionCpuEntrypointsRequested -and $unifiedArchitectureVersion -eq 13
if ($includeProductionCpuEntrypoints) {
    Require-File $singleCpuEntrypoint "single-image production CPU entrypoint"
    Require-File $batchCpuEntrypoint "batch production CPU entrypoint"
    Require-File $cpuDeliveryReadme "production CPU delivery README"
}
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
if ($summary.acceptance.requested -ne $true) {
    throw "onnx-val acceptance was not explicitly requested."
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
$nonRecipientPriorFailures = @(
    $priorFailures |
        Where-Object { -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal) }
)
if ($nonRecipientPriorFailures.Count -ne 0 `
    -or ($summary.acceptance.passed -ne $true -and $priorFailures.Count -eq 0)) {
    throw "onnx-val acceptance contains non-recipient failures: $($nonRecipientPriorFailures -join '; ')"
}

# The v13 artifact is the production source for amount/time/payment/status.
# Its historical recipient head is intentionally not a release pre-gate:
# recipient is replaced by the hash-bound PP-OCR route and is guarded at 90%
# by both the formal hybrid A/B evidence and this package's fresh full scorer.
$preFieldGates = @(
    @{ Field = "amount"; Floor = $AmountFloor; Acceptance = "min_amount_exact_match" },
    @{ Field = "time"; Floor = $TimeFloor; Acceptance = "min_time_exact_match" },
    @{ Field = "payment_method_field"; Floor = $PaymentFloor; Acceptance = "min_payment_exact_match" }
)
$validatedMetrics = [ordered]@{}
foreach ($gate in $preFieldGates) {
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
$hybridAbComparisonSummaryPath = $null
$hybridAbComparisonComparisonsPath = $null
$hybridAbScoreSummaryPath = $null
$hybridAbScoreComparisonsPath = $null
$hybridAbInputManifestPath = $null
$hybridAbBaselineManifestPath = $null
$hybridAbHybridManifestPath = $null
$hybridAbBaselineRuntimeSummaryPath = $null
$hybridAbHybridRuntimeSummaryPath = $null
$hybridAbCliAssemblyPath = $null
$hybridAbCliAssemblySha256 = $null
$hybridAbCliClosureManifestPath = $null
$hybridAbCliClosureSha256 = $null
$hybridAbCliClosureFileCount = $null
$hybridAbScoreExpectedRecords = $null
$hybridAbEvidenceBinding = $null

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
        -or [int]$valEvidence[0].visible_status_records -ne $statusCtcRecords `
        -or [double]$valEvidence[0].status_text_exact_match -ne $statusCtcExactMatch `
        -or [string]$valEvidence[0].summary_sha256 -ne (Get-Sha256 $onnxValidationSummary) `
        -or $testEvidence.Count -ne 1 `
        -or $testEvidence[0].evaluated -ne $true) {
        throw "v13 guarded validation must contain one bound val and one bound test evaluation."
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
    $testNonRecipientFailures = @(
        $testFailures |
            Where-Object { -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal) }
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
        -or $testNonRecipientFailures.Count -ne 0 `
        -or ($testSummary.acceptance.passed -ne $true -and $testFailures.Count -eq 0) `
        -or [string]$testSummary.status_text_policy.runtime_policy -ne "decode_and_normalize_review_only" `
        -or [string]$testSummary.status_text_policy.review_value -ne "review") {
        throw "v13 guarded test summary is not a model/records-bound CUDA amount/time/payment/status evaluation."
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

if ($hasHybridAbEvidence) {
    $hybridAbComparisonSummaryPath = Join-Path $HybridAbEvidence "comparison\summary.json"
    $hybridAbComparisonComparisonsPath = Join-Path $HybridAbEvidence "comparison\comparisons.jsonl"
    $hybridAbScoreSummaryPath = Join-Path $HybridAbEvidence "hybrid-val-score\summary.json"
    $hybridAbScoreComparisonsPath = Join-Path $HybridAbEvidence "hybrid-val-score\comparisons.jsonl"
    foreach ($evidenceFile in @(
            $hybridAbComparisonSummaryPath,
            $hybridAbComparisonComparisonsPath,
            $hybridAbScoreSummaryPath,
            $hybridAbScoreComparisonsPath
        )) {
        Require-File $evidenceFile "formal hybrid CPU A/B evidence"
        if (-not (Test-PathWithin $evidenceFile $HybridAbEvidence)) {
            throw "Formal hybrid CPU A/B evidence escaped its declared root: $evidenceFile"
        }
    }
    $hybridAbSummary = Read-NormalizedJson $hybridAbComparisonSummaryPath
    Assert-HybridAbComparisonSchema $hybridAbSummary
    $hybridAbFailures = @(
        $hybridAbSummary.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $hybridAbRecords = [int]$hybridAbSummary.records
    $hybridAbRecipientCoverage = [double]$hybridAbSummary.recipient_candidate_coverage
    $hybridAbP95Overhead = [double]$hybridAbSummary.cpu.p95_overhead_ms
    $hybridAbP95Ceiling = [double]$hybridAbSummary.cpu.max_p95_overhead_ms
    $hybridBaselineP95 = [double]$hybridAbSummary.cpu.baseline_inference_latency_ms.p95
    $hybridCandidateP95 = [double]$hybridAbSummary.cpu.hybrid_inference_latency_ms.p95
    if ([int]$hybridAbSummary.schema_version -ne 2 `
        -or [string]$hybridAbSummary.kind -ne "receipt_mlnet_hybrid_recipient_cpu_ab_v1" `
        -or [string]$hybridAbSummary.evaluation_mode -ne "formal" `
        -or $hybridAbSummary.accepted -ne $true `
        -or $hybridAbFailures.Count -ne 0 `
        -or $hybridAbRecords -ne $requiredFormalReceiptCount `
        -or [int]$hybridAbSummary.invariant_records -ne $hybridAbRecords `
        -or [int]$hybridAbSummary.input_set.records -ne $hybridAbRecords `
        -or [int]$hybridAbSummary.input_set.input_manifest.records -ne $hybridAbRecords `
        -or [int]$hybridAbSummary.run_manifests.baseline.records -ne $hybridAbRecords `
        -or [int]$hybridAbSummary.run_manifests.hybrid.records -ne $hybridAbRecords `
        -or [string]$hybridAbSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$hybridAbSummary.input_set.input_manifest.normalized_source_set_sha256 `
        -or [string]$hybridAbSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$hybridAbSummary.run_manifests.baseline.normalized_source_set_sha256 `
        -or [string]$hybridAbSummary.input_set.normalized_source_set_sha256 -ne `
            [string]$hybridAbSummary.run_manifests.hybrid.normalized_source_set_sha256 `
        -or $hybridAbRecipientCoverage -ne 1.0 `
        -or [double]::IsNaN($hybridAbP95Overhead) `
        -or [double]::IsInfinity($hybridAbP95Overhead) `
        -or [double]::IsNaN($hybridAbP95Ceiling) `
        -or [double]::IsInfinity($hybridAbP95Ceiling) `
        -or [double]::IsNaN($hybridBaselineP95) `
        -or [double]::IsInfinity($hybridBaselineP95) `
        -or [double]::IsNaN($hybridCandidateP95) `
        -or [double]::IsInfinity($hybridCandidateP95) `
        -or $hybridAbP95Ceiling -lt 0.0 `
        -or $hybridAbP95Ceiling -gt 250.0 `
        -or $hybridAbP95Overhead -gt $hybridAbP95Ceiling `
        -or $hybridBaselineP95 -lt 0.0 `
        -or $hybridCandidateP95 -lt 0.0) {
        throw "Hybrid A/B evidence is not a clean, invariant, complete CPU formal pass within the fixed 250 ms p95 overhead ceiling."
    }
    $hybridBoundFiles = @(
        [pscustomobject]@{
            Description = "Hybrid A/B fixed input manifest"
            Record = $hybridAbSummary.input_set.input_manifest
            Role = "input"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B baseline inference manifest"
            Record = $hybridAbSummary.run_manifests.baseline
            Role = "baseline"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B hybrid inference manifest"
            Record = $hybridAbSummary.run_manifests.hybrid
            Role = "hybrid"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B baseline runtime summary"
            Record = $hybridAbSummary.run_summaries.baseline
            Role = "baseline-summary"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B hybrid runtime summary"
            Record = $hybridAbSummary.run_summaries.hybrid
            Role = "hybrid-summary"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B ReceiptMlNet.Cli assembly"
            Record = $hybridAbSummary.cli_build.assembly
            Role = "assembly"
        },
        [pscustomobject]@{
            Description = "Hybrid A/B CLI app closure manifest"
            Record = $hybridAbSummary.cli_build.app_closure.manifest
            Role = "closure-manifest"
        }
    )
    foreach ($binding in $hybridBoundFiles) {
        $boundRawPath = [string]$binding.Record.path
        Assert-SafePathSyntax $boundRawPath ([string]$binding.Description)
        $boundPath = [IO.Path]::GetFullPath($boundRawPath)
        Require-File $boundPath ([string]$binding.Description)
        if (-not (Test-PathWithin $boundPath $HybridAbEvidence)) {
            throw "$($binding.Description) escaped the formal HybridAbEvidence root."
        }
        $boundItem = Get-Item -LiteralPath $boundPath
        if ([long]$boundItem.Length -ne [long]$binding.Record.size_bytes `
            -or (Get-Sha256 $boundPath) -ne [string]$binding.Record.sha256) {
            throw "$($binding.Description) differs from its comparison-summary hash/size binding."
        }
        switch ([string]$binding.Role) {
            "input" { $hybridAbInputManifestPath = $boundPath }
            "baseline" { $hybridAbBaselineManifestPath = $boundPath }
            "hybrid" { $hybridAbHybridManifestPath = $boundPath }
            "baseline-summary" { $hybridAbBaselineRuntimeSummaryPath = $boundPath }
            "hybrid-summary" { $hybridAbHybridRuntimeSummaryPath = $boundPath }
            "assembly" {
                $hybridAbCliAssemblyPath = $boundPath
                $hybridAbCliAssemblySha256 = [string]$binding.Record.sha256
            }
            "closure-manifest" { $hybridAbCliClosureManifestPath = $boundPath }
        }
    }
    if ([string]$hybridAbSummary.cli_build.app_closure.closure_sha256 -ne `
        [string]$hybridAbSummary.cli_build.app_closure.manifest.sha256) {
        throw "Hybrid A/B CLI app closure digest differs from its manifest SHA-256."
    }
    Assert-SafePathSyntax ([string]$hybridAbSummary.cli_build.app_closure.root) "Hybrid A/B CLI app closure root"
    $hybridAbCliAppRoot = [IO.Path]::GetFullPath([string]$hybridAbSummary.cli_build.app_closure.root)
    if (-not (Test-PathWithin $hybridAbCliAppRoot $HybridAbEvidence) `
        -or -not (Test-PathWithin $hybridAbCliAssemblyPath $hybridAbCliAppRoot)) {
        throw "Hybrid A/B CLI app closure or assembly escaped its formal evidence root."
    }
    $hybridAbCliClosureSha256 = [string]$hybridAbSummary.cli_build.app_closure.closure_sha256
    $verifiedAbCliClosure = Assert-CliAppClosure `
        $hybridAbCliAppRoot $hybridAbCliClosureManifestPath $hybridAbCliClosureSha256
    $hybridAbCliClosureFileCount = [int]$verifiedAbCliClosure.FileCount
    if ($hybridAbCliClosureFileCount -ne [int]$hybridAbSummary.cli_build.app_closure.file_count) {
        throw "Hybrid A/B CLI app closure file_count differs from the exact manifest."
    }
    if (-not (Split-Path -Parent $hybridAbBaselineRuntimeSummaryPath).Equals(
            (Split-Path -Parent $hybridAbBaselineManifestPath),
            [StringComparison]::OrdinalIgnoreCase) `
        -or -not (Split-Path -Parent $hybridAbHybridRuntimeSummaryPath).Equals(
            (Split-Path -Parent $hybridAbHybridManifestPath),
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Hybrid A/B runtime summaries do not belong to their bound inference-manifest runs."
    }
    $hybridAbRawBaselineSummary = Read-NormalizedJson $hybridAbBaselineRuntimeSummaryPath
    $hybridAbRawCandidateSummary = Read-NormalizedJson $hybridAbHybridRuntimeSummaryPath
    foreach ($runtimeBinding in @(
            [pscustomobject]@{ Name = "baseline"; Summary = $hybridAbRawBaselineSummary; Paddle = $null },
            [pscustomobject]@{ Name = "hybrid"; Summary = $hybridAbRawCandidateSummary; Paddle = "cpu" }
        )) {
        foreach ($name in @("input", "written", "skipped", "errors")) {
            Assert-JsonInteger `
                (Get-RequiredJsonProperty $runtimeBinding.Summary $name "Hybrid A/B $($runtimeBinding.Name) runtime summary") `
                "Hybrid A/B $($runtimeBinding.Name) runtime summary $name" 0
        }
        $runtimeLatency = Get-RequiredJsonProperty `
            $runtimeBinding.Summary "inference_latency_ms" "Hybrid A/B runtime summary"
        Assert-JsonInteger `
            (Get-RequiredJsonProperty $runtimeLatency "count" "Hybrid A/B runtime summary inference_latency_ms") `
            "Hybrid A/B $($runtimeBinding.Name) runtime summary inference_latency_ms.count" 1
        Assert-JsonNumber `
            (Get-RequiredJsonProperty $runtimeLatency "p95" "Hybrid A/B runtime summary inference_latency_ms") `
            "Hybrid A/B $($runtimeBinding.Name) runtime summary p95"
        $runtimePaddleProperty = $runtimeBinding.Summary.PSObject.Properties["paddle_ocr_provider"]
        if ([string]$runtimeBinding.Summary.requested_device -ne "cpu" `
            -or [string]$runtimeBinding.Summary.unified_provider -ne "cpu" `
            -or [int]$runtimeBinding.Summary.input -ne $requiredFormalReceiptCount `
            -or [int]$runtimeBinding.Summary.written -ne $requiredFormalReceiptCount `
            -or [int]$runtimeBinding.Summary.skipped -ne 0 `
            -or [int]$runtimeBinding.Summary.errors -ne 0 `
            -or [int]$runtimeLatency.count -ne $requiredFormalReceiptCount `
            -or $null -eq $runtimePaddleProperty `
            -or ($null -eq $runtimeBinding.Paddle -and $null -ne $runtimePaddleProperty.Value) `
            -or ($null -ne $runtimeBinding.Paddle -and [string]$runtimePaddleProperty.Value -ne "cpu")) {
            throw "Hybrid A/B $($runtimeBinding.Name) runtime summary is not a complete strict CPU formal run."
        }
    }
    $rawBaselineP95 = [double]$hybridAbRawBaselineSummary.inference_latency_ms.p95
    $rawCandidateP95 = [double]$hybridAbRawCandidateSummary.inference_latency_ms.p95
    $rawP95Overhead = $rawCandidateP95 - $rawBaselineP95
    if ($hybridBaselineP95 -ne $rawBaselineP95 `
        -or $hybridCandidateP95 -ne $rawCandidateP95 `
        -or $hybridAbP95Overhead -ne $rawP95Overhead) {
        throw "Hybrid A/B p95 evidence does not exactly equal the hash-bound raw runtime summaries."
    }
    if ([string]$hybridAbSummary.artifact_hashes.detector_sha256 -ne $detectorModelSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.detector_contract_sha256 -ne $detectorContractSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.device_sha256 -ne $deviceModelSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.device_contract_sha256 -ne $deviceContractSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.unified_ocr_model_sha256 -ne $unifiedModelSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.unified_ocr_labels_sha256 -ne $unifiedLabelsSha256 `
        -or [string]$hybridAbSummary.artifact_hashes.unified_ocr_contract_sha256 -ne $unifiedContractSha256) {
        throw "Hybrid A/B evidence is not bound to the selected detector, device and unified v13 artifacts."
    }
    Assert-SafePathSyntax ([string]$hybridAbSummary.paddle_delivery.directory) "Hybrid A/B Paddle directory"
    Assert-SafePathSyntax ([string]$hybridAbSummary.paddle_delivery.contract) "Hybrid A/B Paddle contract"
    $hybridAbPaddleDirectory = [IO.Path]::GetFullPath([string]$hybridAbSummary.paddle_delivery.directory)
    $hybridAbPaddleContract = [IO.Path]::GetFullPath([string]$hybridAbSummary.paddle_delivery.contract)
    if (-not $hybridAbPaddleDirectory.Equals($PaddleDeliveryBundle, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $hybridAbPaddleContract.Equals($paddleDeliveryContract, [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$hybridAbSummary.paddle_delivery.contract_sha256 -ne $paddleDeliveryContractSha256 `
        -or [long]$hybridAbSummary.paddle_delivery.package_size_bytes -ne [long]$paddleDelivery.PackageSizeBytes) {
        throw "Hybrid A/B evidence is not bound to the selected pure-ONNX PP-OCR delivery contract."
    }
    foreach ($role in @("det", "cls", "rec")) {
        $abModel = $hybridAbSummary.paddle_delivery.models.PSObject.Properties[$role].Value
        $deliveryModel = $paddleDelivery.Models[$role]
        if ([string]$abModel.path -ne [string]$deliveryModel.RelativePath `
            -or [string]$abModel.sha256 -ne [string]$deliveryModel.Sha256 `
            -or [long]$abModel.size_bytes -ne [long]$deliveryModel.SizeBytes) {
            throw "Hybrid A/B evidence PP-OCR $role model differs from the selected delivery bundle."
        }
    }
    if ([string]$hybridAbSummary.paddle_delivery.dictionary.path -ne [string]$paddleDelivery.Dictionary.RelativePath `
        -or [string]$hybridAbSummary.paddle_delivery.dictionary.sha256 -ne [string]$paddleDelivery.Dictionary.Sha256 `
        -or [long]$hybridAbSummary.paddle_delivery.dictionary.size_bytes -ne [long]$paddleDelivery.Dictionary.SizeBytes) {
        throw "Hybrid A/B evidence PP-OCR dictionary differs from the selected delivery bundle."
    }

    $hybridAbScore = Read-NormalizedJson $hybridAbScoreSummaryPath
    Assert-HybridAbScoreSchema $hybridAbScore
    $hybridAbScoreFailures = @(
        $hybridAbScore.failures |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $hybridAbScoreExpectedRecords = [int]$hybridAbScore.coverage.expected_receipts
    Assert-SafePathSyntax ([string]$hybridAbScore.manifest) "Hybrid A/B score manifest"
    Assert-SafePathSyntax ([string]$hybridAbScore.results_root) "Hybrid A/B score results_root"
    Assert-SafePathSyntax ([string]$hybridAbScore.records) "Hybrid A/B score records"
    $hybridAbScoreManifest = [IO.Path]::GetFullPath([string]$hybridAbScore.manifest)
    $hybridAbScoreResultsRoot = [IO.Path]::GetFullPath([string]$hybridAbScore.results_root)
    $hybridAbScoreRecords = [IO.Path]::GetFullPath([string]$hybridAbScore.records)
    Require-File $hybridAbScoreManifest "Hybrid A/B score inference manifest"
    Require-File $hybridAbScoreRecords "Hybrid A/B score records"
    if (-not (Test-Path -LiteralPath $hybridAbScoreResultsRoot -PathType Container)) {
        throw "Missing Hybrid A/B score results_root: $hybridAbScoreResultsRoot"
    }
    Assert-NoReparsePointInExistingPath $hybridAbScoreResultsRoot "Hybrid A/B score results_root"
    if (-not (Test-PathWithin $hybridAbScoreResultsRoot $HybridAbEvidence)) {
        throw "Hybrid A/B score results_root escaped the formal HybridAbEvidence root."
    }
    if (-not $hybridAbScoreManifest.Equals($hybridAbHybridManifestPath, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $hybridAbScoreResultsRoot.Equals(
            (Split-Path -Parent $hybridAbHybridManifestPath),
            [StringComparison]::OrdinalIgnoreCase) `
        -or -not $hybridAbScoreRecords.Equals($Records, [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$hybridAbScore.manifest_sha256 -ne [string]$hybridAbSummary.run_manifests.hybrid.sha256 `
        -or [string]$hybridAbScore.manifest_sha256 -ne (Get-Sha256 $hybridAbScoreManifest)) {
        throw "Hybrid A/B score is not bound to the comparison's exact hybrid inference manifest/results source."
    }
    if ([int]$hybridAbScore.schema_version -ne 1 `
        -or [string]$hybridAbScore.kind -ne "receipt_mlnet_unified_candidate_evaluation_v1" `
        -or [string]$hybridAbScore.evaluation_split -ne "val" `
        -or [string]$hybridAbScore.model_sha256 -ne $unifiedModelSha256 `
        -or [string]$hybridAbScore.records_sha256 -ne $requestedRecordsSha256 `
        -or $hybridAbScore.formal_delivery_gate -ne $true `
        -or $hybridAbScore.accepted -ne $true `
        -or $hybridAbScore.acceptance.passed -ne $true `
        -or $hybridAbScore.acceptance.formal_delivery_gate -ne $true `
        -or $hybridAbScoreFailures.Count -ne 0 `
        -or [string]$hybridAbScore.evaluation_scope.kind -ne "full_split" `
        -or $null -ne $hybridAbScore.evaluation_scope.requested_limit `
        -or [int]$hybridAbScore.evaluation_scope.evaluated_expected_receipts -ne $hybridAbScoreExpectedRecords `
        -or [int]$hybridAbScore.evaluation_scope.full_split_expected_receipts -ne $hybridAbScoreExpectedRecords `
        -or $hybridAbScore.artifact_audit.all_results_match_model -ne $true `
        -or $hybridAbScoreExpectedRecords -ne $requiredFormalReceiptCount `
        -or $hybridAbScoreExpectedRecords -ne $hybridAbRecords `
        -or [int]$hybridAbScore.coverage.matched_result_receipts -ne [int]$hybridAbScore.coverage.expected_receipts `
        -or [int]$hybridAbScore.coverage.fully_scored_receipts -ne [int]$hybridAbScore.coverage.expected_receipts `
        -or [double]$hybridAbScore.coverage.result_coverage -ne 1.0 `
        -or [double]$hybridAbScore.coverage.fully_scored_coverage -ne 1.0) {
        throw "Hybrid A/B accuracy evidence is not a complete, accepted, model/records-bound formal val score."
    }
    $hybridFixedGates = @(
        @{ Field = "amount"; Floor = $minimumAmountFloor },
        @{ Field = "time"; Floor = $minimumTimeFloor },
        @{ Field = "payment_method_field"; Floor = $minimumPaymentFloor },
        @{ Field = "recipient_field"; Floor = $minimumRecipientFloor },
        @{ Field = "transfer_status"; Floor = $requiredStatusTextFloor }
    )
    foreach ($gate in $hybridFixedGates) {
        $fieldName = [string]$gate.Field
        $floor = [double]$gate.Floor
        $metricProperty = $hybridAbScore.by_field.PSObject.Properties[$fieldName]
        $floorProperty = $hybridAbScore.floors.PSObject.Properties[$fieldName]
        if ($null -eq $metricProperty -or $null -eq $floorProperty) {
            throw "Hybrid A/B accuracy evidence is missing $fieldName metrics or its fixed floor."
        }
        $metric = $metricProperty.Value
        if ([double]$floorProperty.Value -ne $floor `
            -or [int]$metric.records -ne [int]$hybridAbScore.coverage.expected_receipts `
            -or [double]$metric.candidate_coverage -ne 1.0 `
            -or [double]$metric.raw_exact_match -lt $floor) {
            throw "Hybrid A/B accuracy evidence did not pass the fixed $fieldName floor and full candidate coverage."
        }
    }
    if ([int]$hybridAbScore.by_field.transfer_status.non_success_to_success -ne 0) {
        throw "Hybrid A/B formal status evidence crossed the zero non-success-to-success safety line."
    }
    $hybridAbEvidenceBinding = [ordered]@{
        performed = $true
        status = "accepted"
        mode = "formal"
        source = $HybridAbEvidence
        comparison_summary = "evidence/hybrid-formal-ab-summary.json"
        comparison_summary_sha256 = Get-Sha256 $hybridAbComparisonSummaryPath
        comparison_comparisons = "evidence/hybrid-formal-ab-comparisons.jsonl"
        comparison_comparisons_sha256 = Get-Sha256 $hybridAbComparisonComparisonsPath
        accuracy_summary = "evidence/hybrid-formal-accuracy-summary.json"
        accuracy_summary_sha256 = Get-Sha256 $hybridAbScoreSummaryPath
        accuracy_comparisons = "evidence/hybrid-formal-accuracy-comparisons.jsonl"
        accuracy_comparisons_sha256 = Get-Sha256 $hybridAbScoreComparisonsPath
        records_sha256 = $requestedRecordsSha256
        expected_receipts = $requiredFormalReceiptCount
        normalized_source_set_sha256 = [string]$hybridAbSummary.input_set.normalized_source_set_sha256
        input_manifest = "evidence/hybrid-formal-fixed-inputs.txt"
        input_manifest_sha256 = [string]$hybridAbSummary.input_set.input_manifest.sha256
        baseline_inference_manifest = "evidence/hybrid-formal-baseline-inference-manifest.json"
        baseline_inference_manifest_sha256 = [string]$hybridAbSummary.run_manifests.baseline.sha256
        hybrid_inference_manifest = "evidence/hybrid-formal-hybrid-inference-manifest.json"
        hybrid_inference_manifest_sha256 = [string]$hybridAbSummary.run_manifests.hybrid.sha256
        baseline_runtime_summary = "evidence/hybrid-formal-baseline-inference-summary.json"
        baseline_runtime_summary_sha256 = [string]$hybridAbSummary.run_summaries.baseline.sha256
        hybrid_runtime_summary = "evidence/hybrid-formal-hybrid-inference-summary.json"
        hybrid_runtime_summary_sha256 = [string]$hybridAbSummary.run_summaries.hybrid.sha256
        score_manifest_sha256 = [string]$hybridAbScore.manifest_sha256
        cli_assembly = "app/ReceiptMlNet.Cli.dll"
        cli_assembly_sha256 = $hybridAbCliAssemblySha256
        cli_assembly_size_bytes = [long]$hybridAbSummary.cli_build.assembly.size_bytes
        cli_app_closure_manifest = "evidence/hybrid-formal-cli-app-closure.json"
        cli_app_closure_manifest_sha256 = $hybridAbCliClosureSha256
        cli_app_closure_sha256 = $hybridAbCliClosureSha256
        cli_app_closure_file_count = $hybridAbCliClosureFileCount
        paddle_contract_sha256 = $paddleDeliveryContractSha256
        paddle_package_size_bytes = [long]$paddleDelivery.PackageSizeBytes
        invariant_records = $hybridAbRecords
        recipient_candidate_coverage = $hybridAbRecipientCoverage
        cpu_p95_overhead_ms = $hybridAbP95Overhead
        max_cpu_p95_overhead_ms = $hybridAbP95Ceiling
        recipient_exact_match = [double]$hybridAbScore.by_field.recipient_field.raw_exact_match
    }
}
else {
    $hybridAbEvidenceBinding = [ordered]@{
        performed = $false
        status = "candidate_smoke_only"
        reason = "No formal hybrid CPU A/B evidence was supplied; this package cannot be formal delivery evidence."
    }
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
if ($hasRecords -and (
        $expectedRecords -ne $requiredFormalReceiptCount `
        -or $hybridAbRecords -ne $expectedRecords `
        -or $hybridAbScoreExpectedRecords -ne $expectedRecords
    )) {
    throw "Formal delivery requires fresh input count == A/B records == A/B score expected == $requiredFormalReceiptCount."
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
$immutableSourceRoots = @($PaddleDeliveryBundle)
if ($hasHybridAbEvidence) {
    $immutableSourceRoots += $HybridAbEvidence
}
foreach ($immutableSourceRoot in $immutableSourceRoots) {
    if ([string]::IsNullOrWhiteSpace([string]$immutableSourceRoot)) {
        continue
    }
    if ((Test-PathWithin $Output $immutableSourceRoot) `
        -or (Test-PathWithin $DeliveryDir $immutableSourceRoot) `
        -or ($hasRecords -and (Test-PathWithin $EndToEndEvaluationDir $immutableSourceRoot))) {
        throw "Validation/output paths must not be nested inside an immutable bundle or A/B evidence source: $immutableSourceRoot"
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
$recipientPaddleDirectory = Join-Path $modelDirectory "recipient-ppocr"
$evidenceDirectory = Join-Path $stagingRoot "evidence"
$consoleLog = Join-Path $evidenceDirectory "console.log"
$preprocessingContractTestLog = Join-Path $evidenceDirectory "preprocessing-contract-test.log"
$rectificationContractTestLog = Join-Path $evidenceDirectory "rectification-contract-test.log"
$published = $false

try {
    New-Item -ItemType Directory -Path `
        $appDirectory, $modelDirectory, $unifiedDirectory, $recipientPaddleDirectory, $evidenceDirectory | Out-Null
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
        if ($formalExpectedRecords.Count -ne $requiredFormalReceiptCount) {
            throw "Canonical full val must contain exactly $requiredFormalReceiptCount receipts."
        }
        if ((Get-Sha256 $formalExpectedInputList) -ne `
            [string]$hybridAbSummary.input_set.input_manifest.sha256) {
            throw "Fresh canonical full-val input manifest does not match the hash-bound formal A/B input manifest."
        }
    }

    Write-Host "mlnet_unified_publish_$RuntimeFlavor"
    & $DotnetExe restore $projectFile -r win-x64 "-p:OnnxRuntimeFlavor=$RuntimeFlavor"
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet restore failed with exit code $LASTEXITCODE"
    }
    & $DotnetExe publish $projectFile `
        -c Release `
        -r win-x64 `
        --self-contained false `
        "-p:OnnxRuntimeFlavor=$RuntimeFlavor" `
        --no-restore `
        -o $appDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE"
    }
    if ($hasHybridAbEvidence) {
        $publishedCliAssembly = Join-Path $appDirectory "ReceiptMlNet.Cli.dll"
        Require-File $publishedCliAssembly "published ReceiptMlNet.Cli assembly"
        if ((Get-Sha256 $publishedCliAssembly) -ne $hybridAbCliAssemblySha256) {
            throw "Published ReceiptMlNet.Cli assembly does not match the hash-bound formal A/B build."
        }
        $verifiedPublishedCliClosure = Assert-CliAppClosure `
            $appDirectory $hybridAbCliClosureManifestPath $hybridAbCliClosureSha256
        if ([int]$verifiedPublishedCliClosure.FileCount -ne $hybridAbCliClosureFileCount) {
            throw "Published CLI app closure file_count differs from the formal A/B build."
        }
    }

    Write-Host "mlnet_preprocessing_contract_test"
    & $DotnetExe run `
        --project $preprocessingContractTestProject `
        -c Release `
        "-p:OnnxRuntimeFlavor=$RuntimeFlavor" 2>&1 |
        Tee-Object -FilePath $preprocessingContractTestLog |
        Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "ML.NET preprocessing contract test failed with exit code $LASTEXITCODE"
    }

    Write-Host "mlnet_rectification_contract_test"
    & $DotnetExe run `
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
    Copy-Item -LiteralPath $paddleDeliveryContract -Destination $recipientPaddleDirectory
    foreach ($record in @(
            $paddleDelivery.Models["det"],
            $paddleDelivery.Models["cls"],
            $paddleDelivery.Models["rec"],
            $paddleDelivery.Dictionary
        )) {
        $destination = Join-Path `
            $recipientPaddleDirectory ([string]$record.RelativePath).Replace('/', '\')
        $destinationParent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
        Copy-Item -LiteralPath ([string]$record.SourcePath) -Destination $destination
    }
    $stagedPaddleDelivery = Assert-PaddleDeliveryBundle $recipientPaddleDirectory
    if ([string]$stagedPaddleDelivery.ContractSha256 -ne $paddleDeliveryContractSha256) {
        throw "Staged PP-OCR delivery contract differs from the verified source bundle."
    }
    Copy-Item -LiteralPath $onnxValidationSummary -Destination (Join-Path $evidenceDirectory "onnx-validation-summary.json")
    if ($null -ne $guardedValidationEvidencePath) {
        Copy-Item -LiteralPath $guardedValidationEvidencePath -Destination `
            (Join-Path $evidenceDirectory "v13-guarded-validation.json")
        Copy-Item -LiteralPath $guardedTestSummaryPath -Destination `
            (Join-Path $evidenceDirectory "v13-onnx-test-summary.json")
    }
    if ($hasHybridAbEvidence) {
        Copy-Item -LiteralPath $hybridAbComparisonSummaryPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-ab-summary.json")
        Copy-Item -LiteralPath $hybridAbComparisonComparisonsPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-ab-comparisons.jsonl")
        Copy-Item -LiteralPath $hybridAbScoreSummaryPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-accuracy-summary.json")
        Copy-Item -LiteralPath $hybridAbScoreComparisonsPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-accuracy-comparisons.jsonl")
        Copy-Item -LiteralPath $hybridAbInputManifestPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-fixed-inputs.txt")
        Copy-Item -LiteralPath $hybridAbBaselineManifestPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-baseline-inference-manifest.json")
        Copy-Item -LiteralPath $hybridAbHybridManifestPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-hybrid-inference-manifest.json")
        Copy-Item -LiteralPath $hybridAbBaselineRuntimeSummaryPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-baseline-inference-summary.json")
        Copy-Item -LiteralPath $hybridAbHybridRuntimeSummaryPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-hybrid-inference-summary.json")
        Copy-Item -LiteralPath $hybridAbCliClosureManifestPath -Destination `
            (Join-Path $evidenceDirectory "hybrid-formal-cli-app-closure.json")
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
            "--ocr", "hybrid-recipient",
            "--ocr-model", $deliveryUnifiedModel,
            "--ocr-bundle", $recipientPaddleDirectory,
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

    Write-Host "mlnet_hybrid_recipient_${RuntimeFlavor}_validate"
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
    $paddleProviderMatches = @(
        Select-String -LiteralPath $consoleLog -Pattern '^OCR ONNX execution provider: (?<provider>[^ ]+) \(det/cls/rec\)'
    )
    $activePaddleProviders = @(
        $paddleProviderMatches |
            ForEach-Object { $_.Matches[0].Groups["provider"].Value } |
            Sort-Object -Unique
    )
    if ($activePaddleProviders.Count -ne 1 -or $activePaddleProviders[0] -ne "cpu") {
        throw "Published ML.NET PP-OCR did not prove strict cpu det/cls/rec execution: $($activePaddleProviders -join ',')"
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
    if ([string]$runtimeSummary.paddle_ocr_provider -ne "cpu") {
        throw "inference_summary paddle_ocr_provider is not cpu."
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
        Assert-CurrentResultSemantics $result $resultPath
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
        if ([string]$result.model_contracts.ocr_bundle -ne "paddle_ocr_delivery.contract.json" `
            -or [string]$result.model_contracts.ocr_bundle_contract_sha256 -ne $paddleDeliveryContractSha256) {
            throw "Result does not reference the delivered pure-ONNX PP-OCR recipient bundle: $resultPath"
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
        if ([double]$endToEndSummary.coverage.result_coverage -ne 1.0 `
            -or [double]$endToEndSummary.coverage.fully_scored_coverage -ne 1.0 `
            -or $endToEndSummary.formal_delivery_gate -ne $true `
            -or $endToEndSummary.acceptance.formal_delivery_gate -ne $true `
            -or [string]$endToEndSummary.evaluation_scope.kind -ne "full_split" `
            -or $null -ne $endToEndSummary.evaluation_scope.requested_limit) {
            throw "ML.NET end-to-end score is not an unbounded, full-coverage formal delivery gate."
        }
        $finalFieldGates = @(
            @{ Field = "amount"; Floor = $AmountFloor },
            @{ Field = "time"; Floor = $TimeFloor },
            @{ Field = "payment_method_field"; Floor = $PaymentFloor },
            @{ Field = "recipient_field"; Floor = $RecipientFloor }
        )
        $validatedEndToEndMetrics = [ordered]@{}
        foreach ($gate in $finalFieldGates) {
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
            if ([int]$scoreMetric.records -ne [int]$endToEndSummary.coverage.expected_receipts) {
                throw "ML.NET end-to-end $fieldName records do not match the complete formal receipt set."
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
    $paddleModelEvidence = [ordered]@{}
    foreach ($role in @("det", "cls", "rec")) {
        $record = $paddleDelivery.Models[$role]
        $paddleModelEvidence[$role] = [ordered]@{
            path = "models/recipient-ppocr/$([string]$record.RelativePath)"
            sha256 = [string]$record.Sha256
            size_bytes = [long]$record.SizeBytes
        }
    }
    $paddleDictionaryEvidence = [ordered]@{
        path = "models/recipient-ppocr/$([string]$paddleDelivery.Dictionary.RelativePath)"
        sha256 = [string]$paddleDelivery.Dictionary.Sha256
        size_bytes = [long]$paddleDelivery.Dictionary.SizeBytes
    }
    $recipientPaddleArtifactEvidence = [ordered]@{
        kind = "paddle_ocr_v2_delivery"
        bundle_path = "models/recipient-ppocr"
        contract_path = "models/recipient-ppocr/paddle_ocr_delivery.contract.json"
        contract_sha256 = $paddleDeliveryContractSha256
        package_size_bytes = [long]$paddleDelivery.PackageSizeBytes
        models = $paddleModelEvidence
        dictionary = $paddleDictionaryEvidence
        runtime_dependencies = @("ONNX Runtime", "OpenCV-compatible image processing for the OCR adapter")
        forbidden_runtime_dependencies = @("Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files")
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
        recipient_ppocr = $recipientPaddleArtifactEvidence
    }

    $packageValidation = [ordered]@{
        schema_version = 1
        kind = "receipt_mlnet_hybrid_recipient_package_validation_v1"
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
        recipient_ppocr_source = [ordered]@{
            bundle = $PaddleDeliveryBundle
            contract = $paddleDeliveryContract
            contract_sha256 = $paddleDeliveryContractSha256
        }
        unified_ocr_kind = $unifiedKind
        unified_ocr_architecture_version = $unifiedArchitectureVersion
        ocr_mode = "hybrid-recipient"
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
        hybrid_formal_ab = $hybridAbEvidenceBinding
        end_to_end_evaluation = $endToEndEvidence
        onnx_validation = [ordered]@{
            providers = $providers
            accepted = $true
            source_aggregate_accepted = [bool]$summary.acceptance.passed
            legacy_recipient_head_is_not_a_pre_gate = $true
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
        kind = if ($hasRecords) {
            "receipt_mlnet_hybrid_recipient_delivery_package_v1"
        }
        else {
            "receipt_mlnet_hybrid_recipient_candidate_smoke_package_v1"
        }
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
        paddle_delivery_bundle_source = $PaddleDeliveryBundle
        paddle_delivery_contract_sha256 = $paddleDeliveryContractSha256
        hybrid_ab_evidence_source = if ($hasHybridAbEvidence) { $HybridAbEvidence } else { $null }
        hybrid_ab_evidence = $hybridAbEvidenceBinding
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
        ocr_mode = "hybrid-recipient"
        recipient_ocr_bundle = "models/recipient-ppocr"
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
    $finalPaddleDelivery = Assert-PaddleDeliveryBundle $PaddleDeliveryBundle
    if ([string]$finalPaddleDelivery.ContractSha256 -ne $paddleDeliveryContractSha256) {
        throw "PP-OCR delivery bundle changed during CPU validation; refusing atomic publication."
    }
    if ($hasHybridAbEvidence) {
        if ((Get-Sha256 $hybridAbComparisonSummaryPath) -ne [string]$hybridAbEvidenceBinding.comparison_summary_sha256 `
            -or (Get-Sha256 $hybridAbComparisonComparisonsPath) -ne [string]$hybridAbEvidenceBinding.comparison_comparisons_sha256 `
            -or (Get-Sha256 $hybridAbScoreSummaryPath) -ne [string]$hybridAbEvidenceBinding.accuracy_summary_sha256 `
            -or (Get-Sha256 $hybridAbScoreComparisonsPath) -ne [string]$hybridAbEvidenceBinding.accuracy_comparisons_sha256 `
            -or (Get-Sha256 $hybridAbInputManifestPath) -ne [string]$hybridAbEvidenceBinding.input_manifest_sha256 `
            -or (Get-Sha256 $hybridAbBaselineManifestPath) -ne [string]$hybridAbEvidenceBinding.baseline_inference_manifest_sha256 `
            -or (Get-Sha256 $hybridAbHybridManifestPath) -ne [string]$hybridAbEvidenceBinding.hybrid_inference_manifest_sha256 `
            -or (Get-Sha256 $hybridAbBaselineRuntimeSummaryPath) -ne [string]$hybridAbEvidenceBinding.baseline_runtime_summary_sha256 `
            -or (Get-Sha256 $hybridAbHybridRuntimeSummaryPath) -ne [string]$hybridAbEvidenceBinding.hybrid_runtime_summary_sha256 `
            -or (Get-Sha256 $hybridAbCliClosureManifestPath) -ne [string]$hybridAbEvidenceBinding.cli_app_closure_manifest_sha256 `
            -or (Get-Sha256 $hybridAbCliAssemblyPath) -ne [string]$hybridAbEvidenceBinding.cli_assembly_sha256) {
            throw "Formal hybrid CPU A/B evidence changed during package validation; refusing atomic publication."
        }
        $finalPublishedCliClosure = Assert-CliAppClosure `
            $appDirectory $hybridAbCliClosureManifestPath $hybridAbCliClosureSha256
        if ([int]$finalPublishedCliClosure.FileCount -ne $hybridAbCliClosureFileCount) {
            throw "Published CLI app closure changed during package validation; refusing atomic publication."
        }
    }
    $null = Assert-PaddleDeliveryBundle $recipientPaddleDirectory

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
    Write-Host "  unified-provider=$($activeProviders[0])"
    Write-Host "  paddle-ocr-provider=$($activePaddleProviders[0])"
    Write-Host "  ocr-mode=hybrid-recipient"
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
