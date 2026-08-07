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
    [ValidateRange(3, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(0, 256)]
    [int]$CandidateDetectorIntraOpThreads = 0,
    [string]$PythonExecutable
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
        -or [int]$summary.input -ne $ExpectedCount `
        -or [int]$summary.written -ne $ExpectedCount `
        -or [int]$summary.skipped -ne 0 `
        -or [int]$summary.errors -ne 0 `
        -or $detectorThreadsMismatch `
        -or -not [string]::IsNullOrWhiteSpace($errorText)) {
        throw "$RunId did not complete the fixed three-model CPU workload without errors."
    }
}

function Invoke-CpuRun(
    [object]$Descriptor,
    [object]$Variants,
    [string]$FixedInputList,
    [string]$Detector,
    [string]$Device,
    [string]$Unified,
    [int]$CandidateDetectorThreads
) {
    $variant = [string]$Descriptor.variant
    $executable = [string]$Variants[$variant]["executable"]["path"]
    $outputDirectory = [string]$Descriptor.output_directory
    $consoleLog = [string]$Descriptor.console_log
    if (Test-Path -LiteralPath $outputDirectory) {
        throw "Refusing to reuse A/B output directory: $outputDirectory"
    }
    $runContainer = Split-Path -Parent $outputDirectory
    New-Item -ItemType Directory -Path $runContainer -Force | Out-Null
    $arguments = @(
        "--detector", $Detector,
        "--device-model", $Device,
        "--ocr", "unified",
        "--ocr-model", $Unified,
        "--input-list", $FixedInputList,
        "--output", $outputDirectory,
        "--device", "cpu",
        "--score-threshold", "0.50",
        "--rectification", "max-side-1600",
        "--annotate", "none",
        "--require-complete"
    )
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
    & $executable @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$($Descriptor.id) failed with exit code $exitCode; see $consoleLog"
    }
    Assert-CompletedCpuRun `
        $outputDirectory ([int]$Descriptor.expected_count) ([string]$Descriptor.id) `
        $expectedDetectorThreads
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
            })
        }
    }
}

$planPath = Join-Path $OutputRoot "ab-plan.json"
$plan = [ordered]@{
    schema_version = 1
    kind = "receipt_mlnet_cpu_ab_plan_v1"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    output_root = $OutputRoot
    input_count = $inputs.Count
    input_selection = [ordered]@{
        rule = "deduplicate_in_order_then_first_n"
        source_input_list = Get-FileEvidence $InputList "source input list"
        input_limit_requested = $InputLimit
        available_count = $availableInputs.Count
        selected_count = $inputs.Count
    }
    fixed_input_list = Get-FileEvidence $fixedInputListPath "fixed input list"
    input_evidence = Get-FileEvidence $inputEvidencePath "input evidence"
    warmup_runs = $WarmupRuns
    warmup_limit = $warmupLimit
    repetitions = $Repetitions
    cli_contract = [ordered]@{
        device = "cpu"
        unified_provider = "cpu"
        ocr = "unified"
        score_threshold = [double]0.50
        rectification = "max-side-1600"
        annotate = "none"
        require_complete = $true
        continue_on_error = $false
        skip_existing = $false
        includes_device_model = $true
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
Write-JsonNoBom $planPath $plan 12
$frozenPlanSha256 = Get-Sha256 $planPath

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host " Receipt ML.NET - CPU performance/consistency A/B" -ForegroundColor Cyan
Write-Host " fixed detector + device + unified OCR; CPU only" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Inputs      : $($inputs.Count)"
if ($InputLimit -gt 0) {
    Write-Host "Selection   : first $($inputs.Count) of $($availableInputs.Count) canonical input(s)"
}
Write-Host "Warmup      : $WarmupRuns x $warmupLimit image(s) per variant"
Write-Host "Measured    : $Repetitions full repeat(s) per variant"
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
        $CandidateDetectorIntraOpThreads
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
    -or $report.performance.accepted -ne $true) {
    throw "CPU A/B prediction consistency or performance improvement was not accepted."
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
Write-Host "PASS: candidate predictions exactly match every baseline/warmup/repeat result." -ForegroundColor Green
