[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineExecutable,
    [Parameter(Mandatory = $true)]
    [string]$CandidateExecutable,
    [Parameter(Mandatory = $true)]
    [string]$DetectorModel,
    [Parameter(Mandatory = $true)]
    [string]$DeviceModel,
    [Parameter(Mandatory = $true)]
    [string]$UnifiedModel,
    [Parameter(Mandatory = $true)]
    [string]$InputList,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(0, 10000)]
    [int]$InputLimit = 0,
    [ValidateRange(1, 20)]
    [int]$WarmupRuns = 1,
    [ValidateRange(1, 10000)]
    [int]$WarmupImages = 8,
    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(0, 256)]
    [int]$CandidateDetectorIntraOpThreads = 0,
    [string]$PythonExecutable,
    [ValidateSet("unified", "hybrid-recipient")]
    [string]$OcrMode = "unified",
    [string]$PaddleOcrBundle,
    [ValidateRange(0, 1000000)]
    [int]$ExpectedInputCount = 0,
    [switch]$AllowIncompleteRecipientEquivalence,
    [switch]$AllowPreexistingIncompleteEquivalence,
    [switch]$EquivalenceOnlyOnce,
    [string]$PerformanceEvidenceReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

$repoRoot = Split-Path -Parent $PSScriptRoot
$analyzer = Join-Path $PSScriptRoot "receipt_mlnet_cpu_ab_compare.py"
$supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".webp")

function Require-File([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Missing ${Description} path."
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Get-ProviderFullPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Cannot resolve an empty filesystem path."
    }
    # [IO.Path]::GetFullPath resolves relative paths from the host process
    # working directory, which can remain on C: even when PowerShell's current
    # location is the D: repository. Resolve through PowerShell's provider so
    # CLI arguments follow the directory shown by Get-Location/the prompt.
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-PathWithin([string]$Candidate, [string]$Parent) {
    $candidateFull = Get-ProviderFullPath $Candidate
    $parentFull = Get-ProviderFullPath $Parent
    if ($candidateFull.Equals($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if (-not $parentFull.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $parentFull += [IO.Path]::DirectorySeparatorChar
    }
    return $candidateFull.StartsWith($parentFull, [StringComparison]::OrdinalIgnoreCase)
}

function Get-FileEvidence([string]$Path, [string]$Description) {
    $fullPath = Get-ProviderFullPath $Path
    Require-File $fullPath $Description
    $item = Get-Item -LiteralPath $fullPath -Force
    return [ordered]@{
        path = $fullPath
        sha256 = Get-Sha256 $fullPath
        bytes = [long]$item.Length
    }
}

function Get-AppPayloadFiles([string]$AppRoot, [string]$Description) {
    $root = Get-ProviderFullPath $AppRoot
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "Missing ${Description} directory: $root"
    }
    $pending = [Collections.Queue]::new()
    $files = [Collections.Generic.List[object]]::new()
    $pending.Enqueue($root)
    while ($pending.Count -gt 0) {
        $directory = [string]$pending.Dequeue()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "${Description} contains a reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pending.Enqueue($item.FullName)
            }
            else {
                $files.Add($item)
            }
        }
    }
    return @($files | Sort-Object { $_.FullName.ToUpperInvariant() })
}

function Freeze-AppPayload(
    [string]$Executable,
    [string]$Variant,
    [string]$EvidenceRoot
) {
    $executableEvidence = Get-FileEvidence $Executable "$Variant executable"
    $appRoot = Get-ProviderFullPath (Split-Path -Parent $Executable)
    $executableName = [IO.Path]::GetFileName($Executable)
    $managedName = [IO.Path]::ChangeExtension($executableName, ".dll")
    $depsName = [IO.Path]::ChangeExtension($executableName, ".deps.json")
    $runtimeConfigName = [IO.Path]::ChangeExtension($executableName, ".runtimeconfig.json")
    foreach ($requiredName in @($executableName, $managedName, $depsName, $runtimeConfigName)) {
        Require-File (Join-Path $appRoot $requiredName) "$Variant app payload $requiredName"
    }

    $rootPrefix = $appRoot
    if (-not $rootPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $rootPrefix += [IO.Path]::DirectorySeparatorChar
    }
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($file in @(Get-AppPayloadFiles $appRoot "$Variant app payload")) {
        $fullPath = Get-ProviderFullPath ([string]$file.FullName)
        if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Variant app payload file escapes its root: $fullPath"
        }
        $relativePath = $fullPath.Substring($rootPrefix.Length).Replace('\', '/')
        $rows.Add([ordered]@{
            path = $relativePath
            sha256 = Get-Sha256 $fullPath
            bytes = [long]$file.Length
        })
    }
    if ($rows.Count -le 0) {
        throw "$Variant app payload is empty: $appRoot"
    }
    $dllCount = @($rows | Where-Object { ([string]$_.path).EndsWith(".dll", [StringComparison]::OrdinalIgnoreCase) }).Count
    if ($dllCount -lt 2) {
        throw "$Variant app payload must include ReceiptMlNet.Cli.dll and its runtime/native DLL payload."
    }
    $manifestPath = Join-Path $EvidenceRoot ("$Variant-app-payload.json")
    Write-JsonNoBom $manifestPath $rows 5
    return [ordered]@{
        app_root = $appRoot
        executable = $executableEvidence
        app_payload = Get-FileEvidence $manifestPath "$Variant app payload manifest"
        managed_entrypoint_sha256 = Get-Sha256 (Join-Path $appRoot $managedName)
        executable_relative_path = $executableName.Replace('\', '/')
        managed_entrypoint_relative_path = $managedName.Replace('\', '/')
        deps_json_relative_path = $depsName.Replace('\', '/')
        runtimeconfig_json_relative_path = $runtimeConfigName.Replace('\', '/')
    }
}

function Freeze-DirectoryPayload(
    [string]$Directory,
    [string]$Description,
    [string]$EvidenceRoot,
    [string]$ManifestName
) {
    $root = Get-ProviderFullPath $Directory
    $rootPrefix = $root
    if (-not $rootPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $rootPrefix += [IO.Path]::DirectorySeparatorChar
    }
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($file in @(Get-AppPayloadFiles $root $Description)) {
        $fullPath = Get-ProviderFullPath ([string]$file.FullName)
        if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Description file escapes its root: $fullPath"
        }
        $rows.Add([ordered]@{
            path = $fullPath.Substring($rootPrefix.Length).Replace('\', '/')
            sha256 = Get-Sha256 $fullPath
            bytes = [long]$file.Length
        })
    }
    if ($rows.Count -le 0) {
        throw "$Description is empty: $root"
    }
    $manifestPath = Join-Path $EvidenceRoot $ManifestName
    Write-JsonNoBom $manifestPath $rows 5
    return [ordered]@{
        bundle_root = $root
        bundle_payload = Get-FileEvidence $manifestPath "$Description payload manifest"
        contract_relative_path = "paddle_ocr_delivery.contract.json"
    }
}

function Write-JsonNoBom([string]$Path, [object]$Payload, [int]$Depth = 12) {
    $json = $Payload | ConvertTo-Json -Depth $Depth
    [IO.File]::WriteAllText(
        $Path,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false))
}

function Resolve-Python([string]$Requested) {
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        if (Test-Path -LiteralPath $Requested -PathType Leaf) {
            return (Get-ProviderFullPath $Requested)
        }
        $requestedCommand = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($null -eq $requestedCommand) {
            throw "Python executable was not found: $Requested"
        }
        return [string]$requestedCommand.Source
    }
    foreach ($candidate in @(
        (Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"),
        (Join-Path $repoRoot ".venv\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Get-ProviderFullPath $candidate)
        }
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        $pythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if ($null -eq $pythonCommand) {
        throw "Python 3 is required for the CPU A/B deep comparison."
    }
    return [string]$pythonCommand.Source
}

function Read-FixedInputs([string]$Path) {
    $fullListPath = Get-ProviderFullPath $Path
    Require-File $fullListPath "input list"
    $listDirectory = Split-Path -Parent $fullListPath
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $inputs = [Collections.Generic.List[string]]::new()
    $lineNumber = 0
    foreach ($rawLine in Get-Content -LiteralPath $fullListPath -Encoding UTF8) {
        $lineNumber++
        $entry = ([string]$rawLine).Trim()
        if ([string]::IsNullOrWhiteSpace($entry) `
            -or $entry.StartsWith("#", [StringComparison]::Ordinal)) {
            continue
        }
        $candidate = if ([IO.Path]::IsPathRooted($entry)) {
            $entry
        }
        else {
            Join-Path $listDirectory $entry
        }
        try {
            $fullPath = Get-ProviderFullPath $candidate
        }
        catch {
            throw "Invalid input path at ${fullListPath}:${lineNumber}: $($_.Exception.Message)"
        }
        Require-File $fullPath "input image at ${fullListPath}:$lineNumber"
        if ($supportedExtensions -notcontains [IO.Path]::GetExtension($fullPath).ToLowerInvariant()) {
            throw "Unsupported image extension at ${fullListPath}:${lineNumber}: $fullPath"
        }
        if ($seen.Add($fullPath)) {
            $inputs.Add($fullPath)
        }
    }
    if ($inputs.Count -le 0) {
        throw "Input list contains no supported image paths: $fullListPath"
    }
    return $inputs
}

function Assert-CompletedCpuRun(
    [string]$OutputDirectory,
    [int]$ExpectedCount,
    [string]$RunId,
    [string]$ExpectedOcrMode,
    [AllowNull()]
    [object]$ExpectedDetectorIntraOpThreads
) {
    $summaryPath = Join-Path $OutputDirectory "inference_summary.json"
    $manifestPath = Join-Path $OutputDirectory "inference_manifest.json"
    $errorsPath = Join-Path $OutputDirectory "inference_errors.jsonl"
    Require-File $summaryPath "$RunId summary"
    Require-File $manifestPath "$RunId manifest"
    Require-File $errorsPath "$RunId errors file"
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $errorText = Get-Content -LiteralPath $errorsPath -Raw -Encoding UTF8
    $detectorThreadsProperty = $summary.PSObject.Properties["detector_intra_op_threads"]
    $actualDetectorIntraOpThreads = if ($null -eq $detectorThreadsProperty) {
        $null
    }
    else {
        $detectorThreadsProperty.Value
    }
    $paddleProviderProperty = $summary.PSObject.Properties["paddle_ocr_provider"]
    $actualPaddleProvider = if ($null -eq $paddleProviderProperty) {
        $null
    }
    else {
        $paddleProviderProperty.Value
    }
    $paddleProviderMismatch = if ($ExpectedOcrMode -eq "hybrid-recipient") {
        [string]$actualPaddleProvider -ne "cpu"
    }
    else {
        $null -ne $actualPaddleProvider
    }
    $detectorThreadsMismatch = if ($null -eq $ExpectedDetectorIntraOpThreads) {
        $null -ne $actualDetectorIntraOpThreads
    }
    else {
        $null -eq $actualDetectorIntraOpThreads `
            -or $actualDetectorIntraOpThreads -is [bool] `
            -or [int]$actualDetectorIntraOpThreads -ne [int]$ExpectedDetectorIntraOpThreads
    }
    if ([string]$summary.requested_device -ne "cpu" `
        -or [string]$summary.unified_provider -ne "cpu" `
        -or $paddleProviderMismatch `
        -or [int]$summary.input -ne $ExpectedCount `
        -or [int]$summary.written -ne $ExpectedCount `
        -or [int]$summary.skipped -ne 0 `
        -or [int]$summary.errors -ne 0 `
        -or $detectorThreadsMismatch `
        -or -not [string]::IsNullOrWhiteSpace($errorText)) {
        throw "$RunId did not complete the fixed protected CPU workload without errors."
    }
}

function Invoke-CpuRun(
    [object]$Descriptor,
    [object]$Variants,
    [string]$FixedInputList,
    [string]$Detector,
    [string]$Device,
    [string]$Unified,
    [int]$CandidateDetectorThreads,
    [string]$SelectedOcrMode,
    [AllowNull()]
    [string]$PaddleBundle,
    [bool]$AllowIncompleteRecipient
) {
    $variant = [string]$Descriptor.variant
    $executable = [string]$Variants[$variant]["executable"]["path"]
    $outputDirectory = [string]$Descriptor.output_directory
    $consoleLog = [string]$Descriptor.console_log
    $wallClockEvidence = [string]$Descriptor.wall_clock_evidence
    if (Test-Path -LiteralPath $outputDirectory) {
        throw "Refusing to reuse A/B output directory: $outputDirectory"
    }
    if (Test-Path -LiteralPath $wallClockEvidence) {
        throw "Refusing to reuse A/B wall-clock evidence: $wallClockEvidence"
    }
    $runContainer = Split-Path -Parent $outputDirectory
    New-Item -ItemType Directory -Path $runContainer -Force | Out-Null
    $arguments = @(
        "--detector", $Detector,
        "--device-model", $Device,
        "--ocr", $SelectedOcrMode,
        "--ocr-model", $Unified,
        "--input-list", $FixedInputList,
        "--output", $outputDirectory,
        "--device", "cpu",
        "--score-threshold", "0.50",
        "--rectification", "max-side-1600",
        "--annotate", "none"
    )
    if (-not $AllowIncompleteRecipient) {
        $arguments += "--require-complete"
    }
    if ($SelectedOcrMode -eq "hybrid-recipient") {
        $arguments += @("--ocr-bundle", $PaddleBundle)
    }
    if ([string]$Descriptor.phase -eq "warmup") {
        $arguments += @("--limit", [string]$Descriptor.expected_count)
    }
    $expectedDetectorThreads = $null
    if ($variant -eq "candidate" -and $CandidateDetectorThreads -gt 0) {
        $expectedDetectorThreads = $CandidateDetectorThreads
        $arguments += @(
            "--detector-intra-op-threads",
            [string]$CandidateDetectorThreads)
    }

    Write-Host (
        "[{0:D2}] {1} {2} iteration {3}; images={4}" -f `
            [int]$Descriptor.execution_order,
            ([string]$Descriptor.phase).ToUpperInvariant(),
            $variant.ToUpperInvariant(),
            [int]$Descriptor.iteration,
            [int]$Descriptor.expected_count)
    $startedUtc = (Get-Date).ToUniversalTime().ToString("o")
    $wallClock = [Diagnostics.Stopwatch]::StartNew()
    & $executable @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
    $wallClock.Stop()
    $finishedUtc = (Get-Date).ToUniversalTime().ToString("o")
    Write-JsonNoBom $wallClockEvidence ([ordered]@{
        schema_version = 1
        kind = "receipt_mlnet_cpu_ab_wall_clock_v1"
        run_id = [string]$Descriptor.id
        phase = [string]$Descriptor.phase
        variant = $variant
        iteration = [int]$Descriptor.iteration
        expected_count = [int]$Descriptor.expected_count
        exit_code = [int]$exitCode
        started_utc = $startedUtc
        finished_utc = $finishedUtc
        elapsed_seconds = [Math]::Round($wallClock.Elapsed.TotalSeconds, 6)
    }) 5
    if ($exitCode -ne 0) {
        throw "$($Descriptor.id) failed with exit code $exitCode; see $consoleLog"
    }
    Assert-CompletedCpuRun `
        $outputDirectory ([int]$Descriptor.expected_count) ([string]$Descriptor.id) `
        $SelectedOcrMode $expectedDetectorThreads
}

Require-File $analyzer "CPU A/B analyzer"
$pythonExe = Resolve-Python $PythonExecutable
$BaselineExecutable = Get-ProviderFullPath $BaselineExecutable
$CandidateExecutable = Get-ProviderFullPath $CandidateExecutable
$DetectorModel = Get-ProviderFullPath $DetectorModel
$DeviceModel = Get-ProviderFullPath $DeviceModel
$UnifiedModel = Get-ProviderFullPath $UnifiedModel
$InputList = Get-ProviderFullPath $InputList
$OutputRoot = Get-ProviderFullPath $OutputRoot
if ($AllowIncompleteRecipientEquivalence -and $AllowPreexistingIncompleteEquivalence) {
    throw "Choose exactly one incomplete-equivalence policy; recipient-only and preexisting-all-fields are mutually exclusive."
}
$allowIncompleteEquivalence = [bool](
    $AllowIncompleteRecipientEquivalence -or $AllowPreexistingIncompleteEquivalence)
$performanceEvidence = $null
if ($EquivalenceOnlyOnce) {
    if ([string]::IsNullOrWhiteSpace($PerformanceEvidenceReport)) {
        throw "-EquivalenceOnlyOnce requires -PerformanceEvidenceReport."
    }
    if ($Repetitions -ne 1) {
        throw "-EquivalenceOnlyOnce requires -Repetitions 1."
    }
    if ($WarmupRuns -ne 1) {
        throw "-EquivalenceOnlyOnce requires -WarmupRuns 1."
    }
    if ($InputLimit -ne 0 -or $ExpectedInputCount -ne 10016) {
        throw "-EquivalenceOnlyOnce requires the complete formal set: -InputLimit 0 -ExpectedInputCount 10016."
    }
    if ($OcrMode -ne "hybrid-recipient" `
        -or -not $AllowPreexistingIncompleteEquivalence `
        -or $AllowIncompleteRecipientEquivalence) {
        throw "-EquivalenceOnlyOnce requires hybrid-recipient OCR with -AllowPreexistingIncompleteEquivalence."
    }

    $PerformanceEvidenceReport = Get-ProviderFullPath $PerformanceEvidenceReport
    Require-File $PerformanceEvidenceReport "accepted 332-image CPU performance report"
    if (Test-PathWithin $PerformanceEvidenceReport $OutputRoot) {
        throw "The external performance report must be outside the new OutputRoot."
    }
    $performanceReportPayload = Get-Content -LiteralPath $PerformanceEvidenceReport -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$performanceReportPayload.schema_version -ne 2 `
        -or [string]$performanceReportPayload.kind -ne "receipt_mlnet_cpu_ab_report_v2" `
        -or $performanceReportPayload.accepted -ne $true `
        -or $performanceReportPayload.prediction_consistency.accepted -ne $true `
        -or [int]$performanceReportPayload.prediction_consistency.difference_count -ne 0 `
        -or $performanceReportPayload.route_consistency.accepted -ne $true `
        -or $performanceReportPayload.performance.accepted -ne $true `
        -or [int]$performanceReportPayload.input_count -ne 332 `
        -or [int]$performanceReportPayload.measured_repetitions_per_variant -ne 3) {
        throw "-PerformanceEvidenceReport must be an accepted standard 332-image, three-repetition CPU A/B report."
    }
    if ($null -ne $performanceReportPayload.PSObject.Properties["execution_mode"]) {
        throw "External performance evidence cannot itself use an equivalence-only-once report."
    }
    $performancePlanRaw = [string]$performanceReportPayload.plan
    if ([string]::IsNullOrWhiteSpace($performancePlanRaw)) {
        throw "The external performance report does not identify its frozen plan."
    }
    if ([IO.Path]::IsPathRooted($performancePlanRaw)) {
        $performancePlanPath = Get-ProviderFullPath $performancePlanRaw
    }
    else {
        $performancePlanPath = Get-ProviderFullPath (Join-Path (Split-Path -Parent $PerformanceEvidenceReport) $performancePlanRaw)
    }
    Require-File $performancePlanPath "332-image CPU performance plan"
    if ((Test-PathWithin $performancePlanPath $OutputRoot) `
        -or $performancePlanPath.Equals($PerformanceEvidenceReport, [StringComparison]::OrdinalIgnoreCase)) {
        throw "The external performance plan must be a distinct file outside the new OutputRoot."
    }
    if ([string]$performanceReportPayload.plan_sha256 -ne (Get-Sha256 $performancePlanPath)) {
        throw "The external performance report is not bound to the referenced plan SHA-256."
    }
    $performancePlanPayload = Get-Content -LiteralPath $performancePlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$performancePlanPayload.schema_version -ne 2 `
        -or [string]$performancePlanPayload.kind -ne "receipt_mlnet_cpu_ab_plan_v2" `
        -or [int]$performancePlanPayload.input_count -ne 332 `
        -or [int]$performancePlanPayload.repetitions -ne 3 `
        -or $null -ne $performancePlanPayload.PSObject.Properties["execution_mode"]) {
        throw "The external performance plan must be one standard 332-image, three-repetition v2 plan."
    }
    $performanceCliContract = $performancePlanPayload.cli_contract
    $performanceAllowedIncompleteFields = @(
        $performanceCliContract.allowed_incomplete_fields | ForEach-Object { [string]$_ })
    if ($performanceCliContract.require_complete -ne $false `
        -or $performanceCliContract.equivalence_only -ne $true `
        -or $null -ne $performanceCliContract.allowed_incomplete_field `
        -or $performanceCliContract.preexisting_incomplete_equivalence -ne $true `
        -or ($performanceAllowedIncompleteFields -join "|") -ne "time|amount|transfer_status|recipient|payment_method" `
        -or [string]$performanceCliContract.missing_set_rule -ne "baseline_and_candidate_exact_per_field") {
        throw "The external 332-image plan must use the same explicit preexisting-incomplete equivalence contract."
    }
    $performanceEvidence = [ordered]@{
        expected_input_count = 332
        expected_repetitions_per_variant = 3
        measurement = "external_process_wall_clock"
        report = Get-FileEvidence $PerformanceEvidenceReport "332-image CPU performance report"
        plan = Get-FileEvidence $performancePlanPath "332-image CPU performance plan"
    }
}
else {
    if ($Repetitions -lt 3) {
        throw "Standard CPU A/B requires -Repetitions 3 or greater."
    }
    if (-not [string]::IsNullOrWhiteSpace($PerformanceEvidenceReport)) {
        throw "-PerformanceEvidenceReport is valid only with -EquivalenceOnlyOnce."
    }
}
$paddleContractPath = $null
if ($OcrMode -eq "hybrid-recipient") {
    if ([string]::IsNullOrWhiteSpace($PaddleOcrBundle)) {
        throw "-PaddleOcrBundle is required when -OcrMode hybrid-recipient."
    }
    $PaddleOcrBundle = Get-ProviderFullPath $PaddleOcrBundle
    if (-not (Test-Path -LiteralPath $PaddleOcrBundle -PathType Container)) {
        throw "Missing Paddle OCR delivery bundle: $PaddleOcrBundle"
    }
    $paddleContractPath = Join-Path $PaddleOcrBundle "paddle_ocr_delivery.contract.json"
    Require-File $paddleContractPath "Paddle OCR delivery contract"
    $paddleContract = Get-Content -LiteralPath $paddleContractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$paddleContract.schema_version -ne 1 `
        -or [string]$paddleContract.kind -ne "paddle_ocr_v2_delivery" `
        -or $null -eq $paddleContract.models.det `
        -or $null -eq $paddleContract.models.cls `
        -or $null -eq $paddleContract.models.rec) {
        throw "Paddle OCR delivery bundle must be a complete det/cls/rec ONNX package."
    }
}
elseif (-not [string]::IsNullOrWhiteSpace($PaddleOcrBundle)) {
    throw "-PaddleOcrBundle is valid only when -OcrMode hybrid-recipient."
}
if ($AllowIncompleteRecipientEquivalence -and $OcrMode -ne "hybrid-recipient") {
    throw "-AllowIncompleteRecipientEquivalence is valid only when -OcrMode hybrid-recipient."
}
if ($AllowPreexistingIncompleteEquivalence -and $OcrMode -ne "hybrid-recipient") {
    throw "-AllowPreexistingIncompleteEquivalence is valid only when -OcrMode hybrid-recipient."
}
$baselineAppRoot = Get-ProviderFullPath (Split-Path -Parent $BaselineExecutable)
$candidateAppRoot = Get-ProviderFullPath (Split-Path -Parent $CandidateExecutable)

foreach ($required in @(
    @{ Path = $BaselineExecutable; Description = "baseline executable" },
    @{ Path = $CandidateExecutable; Description = "candidate executable" },
    @{ Path = $DetectorModel; Description = "detector model" },
    @{ Path = $DeviceModel; Description = "device model" },
    @{ Path = $UnifiedModel; Description = "unified OCR model" },
    @{ Path = $InputList; Description = "input list" }
)) {
    Require-File ([string]$required.Path) ([string]$required.Description)
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to mix CPU A/B evidence with an existing output root: $OutputRoot"
}
if ((Test-PathWithin $baselineAppRoot $candidateAppRoot) `
    -or (Test-PathWithin $candidateAppRoot $baselineAppRoot)) {
    throw "Baseline and candidate app roots must be separate, non-nested publish directories."
}
foreach ($appRoot in @($baselineAppRoot, $candidateAppRoot)) {
    if ((Test-PathWithin $OutputRoot $appRoot) `
        -or (Test-PathWithin $appRoot $OutputRoot)) {
        throw "OutputRoot and each immutable app payload must be separate, non-nested directories."
    }
}
if ($OcrMode -eq "hybrid-recipient" `
    -and ((Test-PathWithin $OutputRoot $PaddleOcrBundle) `
        -or (Test-PathWithin $PaddleOcrBundle $OutputRoot))) {
    throw "OutputRoot and the immutable Paddle OCR bundle must be separate, non-nested directories."
}
$outputParent = Split-Path -Parent $OutputRoot
if ([string]::IsNullOrWhiteSpace($outputParent)) {
    throw "OutputRoot must have a parent directory."
}
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$availableInputs = @(Read-FixedInputs $InputList)
$selectedInputCount = if ($InputLimit -gt 0) {
    [Math]::Min($InputLimit, $availableInputs.Count)
}
else {
    $availableInputs.Count
}
$inputs = [Collections.Generic.List[string]]::new()
for ($inputIndex = 0; $inputIndex -lt $selectedInputCount; $inputIndex++) {
    $inputs.Add([string]$availableInputs[$inputIndex])
}
if ($ExpectedInputCount -gt 0 -and $inputs.Count -ne $ExpectedInputCount) {
    throw "Selected input count differs from -ExpectedInputCount: expected $ExpectedInputCount, got $($inputs.Count)."
}
$warmupLimit = [Math]::Min($WarmupImages, $inputs.Count)
$fixedInputListPath = Join-Path $OutputRoot "fixed-inputs.txt"
[IO.File]::WriteAllLines(
    $fixedInputListPath,
    [string[]]$inputs,
    [Text.UTF8Encoding]::new($false))

Write-Host "Freezing $($inputs.Count) input image(s) by SHA-256..."
$inputEvidenceRows = [Collections.Generic.List[object]]::new()
foreach ($inputPath in $inputs) {
    $item = Get-Item -LiteralPath $inputPath -Force
    $inputEvidenceRows.Add([ordered]@{
        source = [string]$inputPath
        sha256 = Get-Sha256 $inputPath
        bytes = [long]$item.Length
    })
}
$inputEvidencePath = Join-Path $OutputRoot "input-evidence.json"
Write-JsonNoBom $inputEvidencePath $inputEvidenceRows 5

$detectorContract = [IO.Path]::ChangeExtension($DetectorModel, ".contract.json")
$deviceContract = [IO.Path]::ChangeExtension($DeviceModel, ".contract.json")
$unifiedLabels = [IO.Path]::ChangeExtension($UnifiedModel, ".labels.json")
$unifiedContract = [IO.Path]::ChangeExtension($UnifiedModel, ".contract.json")
$artifacts = [ordered]@{
    detector = Get-FileEvidence $DetectorModel "detector model"
    detector_contract = Get-FileEvidence $detectorContract "detector contract"
    device = Get-FileEvidence $DeviceModel "device model"
    device_contract = Get-FileEvidence $deviceContract "device contract"
    unified_ocr = Get-FileEvidence $UnifiedModel "unified OCR model"
    unified_labels = Get-FileEvidence $unifiedLabels "unified OCR labels"
    unified_contract = Get-FileEvidence $unifiedContract "unified OCR contract"
}
if ($OcrMode -eq "hybrid-recipient") {
    $artifacts["paddle_ocr_bundle"] = Freeze-DirectoryPayload `
        $PaddleOcrBundle "Paddle OCR delivery bundle" $OutputRoot `
        "paddle-ocr-bundle-payload.json"
}
$variants = [ordered]@{
    baseline = Freeze-AppPayload $BaselineExecutable "baseline" $OutputRoot
    candidate = Freeze-AppPayload $CandidateExecutable "candidate" $OutputRoot
}
$baselinePayloadSha256 = [string]$variants["baseline"]["app_payload"]["sha256"]
$candidatePayloadSha256 = [string]$variants["candidate"]["app_payload"]["sha256"]
if ($baselinePayloadSha256 -eq $candidatePayloadSha256) {
    throw "Baseline and candidate app payloads are byte-identical; refusing a meaningless A/B."
}
$baselineManagedSha256 = [string]$variants["baseline"]["managed_entrypoint_sha256"]
$candidateManagedSha256 = [string]$variants["candidate"]["managed_entrypoint_sha256"]
if ($baselineManagedSha256 -eq $candidateManagedSha256) {
    throw "Baseline and candidate ReceiptMlNet.Cli.dll files are byte-identical; refusing a meaningless A/B."
}

$runPlan = [Collections.Generic.List[object]]::new()
$executionOrder = 0
foreach ($phaseSpec in @(
    [pscustomobject]@{ Phase = "warmup"; Count = $WarmupRuns; Expected = $warmupLimit },
    [pscustomobject]@{ Phase = "measured"; Count = $Repetitions; Expected = $inputs.Count }
)) {
    for ($iteration = 1; $iteration -le [int]$phaseSpec.Count; $iteration++) {
        $variantOrder = if (($iteration % 2) -eq 1) {
            @("baseline", "candidate")
        }
        else {
            @("candidate", "baseline")
        }
        foreach ($variant in $variantOrder) {
            $executionOrder++
            $runId = "{0}-{1:D2}-{2}" -f [string]$phaseSpec.Phase, $iteration, $variant
            $runContainer = Join-Path $OutputRoot ("runs\{0}-{1:D2}\{2}" -f `
                [string]$phaseSpec.Phase, $iteration, $variant)
            $runPlan.Add([pscustomobject][ordered]@{
                id = $runId
                phase = [string]$phaseSpec.Phase
                variant = $variant
                iteration = $iteration
                execution_order = $executionOrder
                expected_count = [int]$phaseSpec.Expected
                detector_intra_op_threads = if ($variant -eq "candidate" -and $CandidateDetectorIntraOpThreads -gt 0) {
                    $CandidateDetectorIntraOpThreads
                }
                else {
                    $null
                }
                output_directory = Join-Path $runContainer "output"
                console_log = Join-Path $runContainer "console.log"
                wall_clock_evidence = Join-Path $runContainer "wall-clock.json"
            })
        }
    }
}

$planPath = Join-Path $OutputRoot "ab-plan.json"
$plan = [ordered]@{
    schema_version = 2
    kind = "receipt_mlnet_cpu_ab_plan_v2"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    output_root = $OutputRoot
    input_count = $inputs.Count
    input_selection = [ordered]@{
        rule = "deduplicate_in_order_then_first_n"
        source_input_list = Get-FileEvidence $InputList "source input list"
        input_limit_requested = $InputLimit
        available_count = $availableInputs.Count
        selected_count = $inputs.Count
        expected_input_count = if ($ExpectedInputCount -gt 0) {
            $ExpectedInputCount
        }
        else {
            $null
        }
    }
    fixed_input_list = Get-FileEvidence $fixedInputListPath "fixed input list"
    input_evidence = Get-FileEvidence $inputEvidencePath "input evidence"
    warmup_runs = $WarmupRuns
    warmup_limit = $warmupLimit
    repetitions = $Repetitions
    cli_contract = [ordered]@{
        device = "cpu"
        unified_provider = "cpu"
        paddle_ocr_provider = if ($OcrMode -eq "hybrid-recipient") { "cpu" } else { $null }
        ocr = $OcrMode
        score_threshold = [double]0.50
        rectification = "max-side-1600"
        annotate = "none"
        require_complete = -not $allowIncompleteEquivalence
        equivalence_only = $allowIncompleteEquivalence
        allowed_incomplete_field = if ($AllowIncompleteRecipientEquivalence) {
            "recipient"
        }
        else {
            $null
        }
        continue_on_error = $false
        skip_existing = $false
        includes_device_model = $true
        includes_paddle_ocr_bundle = ($OcrMode -eq "hybrid-recipient")
        detector_intra_op_threads = [ordered]@{
            baseline = $null
            candidate = if ($CandidateDetectorIntraOpThreads -gt 0) {
                $CandidateDetectorIntraOpThreads
            }
            else {
                $null
            }
        }
    }
    performance_gate = [ordered]@{
        minimum_throughput_gain_percent = [double]2.0
        maximum_p50_regression_percent = [double]0.0
        maximum_p95_regression_percent = [double]0.0
    }
    artifacts = $artifacts
    variants = $variants
    runs = $runPlan
}
if ($AllowPreexistingIncompleteEquivalence) {
    $plan["cli_contract"]["preexisting_incomplete_equivalence"] = $true
    $plan["cli_contract"]["allowed_incomplete_fields"] = @(
        "time",
        "amount",
        "transfer_status",
        "recipient",
        "payment_method")
    $plan["cli_contract"]["missing_set_rule"] = "baseline_and_candidate_exact_per_field"
}
if ($EquivalenceOnlyOnce) {
    $plan["execution_mode"] = "equivalence_only_once"
    $plan["performance_evidence"] = $performanceEvidence
}
Write-JsonNoBom $planPath $plan 12
$frozenPlanSha256 = Get-Sha256 $planPath

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host " Receipt ML.NET - CPU performance/consistency A/B" -ForegroundColor Cyan
Write-Host " fixed detector + device + $OcrMode OCR; CPU only" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Inputs      : $($inputs.Count)"
if ($InputLimit -gt 0) {
    Write-Host "Selection   : first $($inputs.Count) of $($availableInputs.Count) canonical input(s)"
}
Write-Host "Warmup      : $WarmupRuns x $warmupLimit image(s) per variant"
Write-Host "Measured    : $Repetitions full repeat(s) per variant"
if ($EquivalenceOnlyOnce) {
    Write-Host "Evidence    : accepted external 332-image x3 performance report"
}
Write-Host "OCR mode    : $OcrMode"
$completenessDescription = if ($AllowPreexistingIncompleteEquivalence) {
    "preexisting baseline missing sets must remain exactly equal"
}
elseif ($AllowIncompleteRecipientEquivalence) {
    "recipient-only incomplete equivalence"
}
else {
    "require-complete"
}
Write-Host "Completeness: $completenessDescription"
Write-Host "Throughput  : external process wall clock"
$candidateThreadDescription = if ($CandidateDetectorIntraOpThreads -gt 0) {
    [string]$CandidateDetectorIntraOpThreads
}
else {
    "default"
}
Write-Host "Threads     : baseline=default; candidate detector intra-op=$candidateThreadDescription"
Write-Host "Output root : $OutputRoot"
Write-Host ""

foreach ($descriptor in @($runPlan | Sort-Object execution_order)) {
    Invoke-CpuRun `
        $descriptor $variants $fixedInputListPath `
        $DetectorModel $DeviceModel $UnifiedModel `
        $CandidateDetectorIntraOpThreads $OcrMode $PaddleOcrBundle `
        $allowIncompleteEquivalence
}
if ((Get-Sha256 $planPath) -ne $frozenPlanSha256) {
    throw "CPU A/B plan changed while the runs were executing."
}

$reportPath = Join-Path $OutputRoot "ab-report.json"
$differencesPath = Join-Path $OutputRoot "prediction-differences.jsonl"
& $pythonExe $analyzer `
    --plan $planPath `
    --report $reportPath `
    --differences $differencesPath
$analysisExitCode = $LASTEXITCODE
Require-File $reportPath "CPU A/B report"
Require-File $differencesPath "CPU A/B prediction differences"
if ($analysisExitCode -ne 0) {
    throw "CPU A/B analysis failed; report=$reportPath differences=$differencesPath"
}

$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($report.accepted -ne $true `
    -or $report.prediction_consistency.accepted -ne $true `
    -or [int]$report.prediction_consistency.difference_count -ne 0 `
    -or $report.route_consistency.accepted -ne $true `
    -or $report.performance.accepted -ne $true) {
    throw "CPU A/B prediction consistency or performance improvement was not accepted."
}
if ($AllowPreexistingIncompleteEquivalence `
    -and ($report.preexisting_incomplete_consistency.accepted -ne $true `
        -or [string]$report.preexisting_incomplete_consistency.policy -ne "baseline_and_candidate_exact_per_field")) {
    throw "CPU A/B preexisting missing-source sets were not exactly preserved."
}
if ($EquivalenceOnlyOnce `
    -and ([string]$report.execution_mode -ne "equivalence_only_once" `
        -or [string]$report.performance.evidence_source -ne "hash_bound_external_332_three_repetition_report" `
        -or $report.external_performance_evidence.accepted -ne $true `
        -or $report.performance.full_once_observation.acceptance_applicable -ne $false)) {
    throw "Equivalence-only-once report did not retain its external performance provenance."
}

Write-Host ""
Write-Host "CPU A/B AGGREGATE" -ForegroundColor Green
@(
    [pscustomobject]@{
        Variant = "baseline"
        MeanMs = $report.performance.baseline.inference_latency_ms.mean
        P50Ms = $report.performance.baseline.inference_latency_ms.p50
        P95Ms = $report.performance.baseline.inference_latency_ms.p95
        ImagesPerSecond = $report.performance.baseline.throughput_images_per_second.aggregate
    },
    [pscustomobject]@{
        Variant = "candidate"
        MeanMs = $report.performance.candidate.inference_latency_ms.mean
        P50Ms = $report.performance.candidate.inference_latency_ms.p50
        P95Ms = $report.performance.candidate.inference_latency_ms.p95
        ImagesPerSecond = $report.performance.candidate.throughput_images_per_second.aggregate
    }
) | Format-Table -AutoSize

Write-Host "STAGE MEAN/P50/P95 (ms)" -ForegroundColor Cyan
$stageRows = foreach ($stage in @(
    "image_load",
    "device",
    "detector_preprocess",
    "detector_inference",
    "detector_postprocess",
    "paddle_ocr",
    "unified_ocr_preprocess",
    "unified_ocr_inference",
    "unified_ocr_postprocess",
    "result_assembly"
)) {
    $baselineStage = $report.performance.baseline.stage_latency_ms.PSObject.Properties[$stage].Value
    $candidateStage = $report.performance.candidate.stage_latency_ms.PSObject.Properties[$stage].Value
    [pscustomobject]@{
        Stage = $stage
        BaselineMean = $baselineStage.mean
        BaselineP50 = $baselineStage.p50
        BaselineP95 = $baselineStage.p95
        CandidateMean = $candidateStage.mean
        CandidateP50 = $candidateStage.p50
        CandidateP95 = $candidateStage.p95
    }
}
$stageRows | Format-Table -AutoSize
Write-Host "Report      : $reportPath"
Write-Host "Differences : $differencesPath"
Write-Host "PASS: candidate predictions and route counts exactly match every baseline/warmup/repeat result." -ForegroundColor Green
