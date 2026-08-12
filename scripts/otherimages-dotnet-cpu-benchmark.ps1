[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [Parameter(Mandatory = $true)]
    [string]$DeviceModel,
    [Parameter(Mandatory = $true)]
    [string]$OcrBundle,
    [Parameter(Mandatory = $true)]
    [string]$WhiteStudentBundle,
    [Parameter(Mandatory = $true)]
    [string]$InputList,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(1, 20)]
    [int]$WarmupRuns = 1,
    [ValidateRange(1, 10000)]
    [int]$WarmupImages = 8,
    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(50, 5000)]
    [int]$PollIntervalMilliseconds = 200,
    [string]$BaselineEvidence,
    [ValidateRange(0.0, 100.0)]
    [double]$MaxThroughputRegressionPercent = 5.0,
    [ValidateRange(0.0, 100.0)]
    [double]$MaxLatencyRegressionPercent = 5.0,
    [ValidateRange(0.0, 100.0)]
    [double]$MaxMemoryRegressionPercent = 10.0,
    [ValidateRange(0, 1048576)]
    [int]$MaxMemoryAbsoluteIncreaseMiB = 128,
    [double]$MaxP50LatencyMilliseconds,
    [double]$MaxP95LatencyMilliseconds,
    [double]$MinThroughputImagesPerSecond,
    [double]$MaxPeakWorkingSetMiB,
    [double]$MaxPeakPrivateBytesMiB,
    [double]$MaxPackageSizeMiB
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredFile([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).ProviderPath)
}

function Resolve-RequiredDirectory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing ${Description}: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).ProviderPath)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-FileEvidence([string]$Path, [string]$Description) {
    $resolved = Resolve-RequiredFile $Path $Description
    $item = Get-Item -LiteralPath $resolved
    return [ordered]@{
        path = $resolved
        sha256 = Get-Sha256 $resolved
        size_bytes = [long]$item.Length
    }
}

function Get-TextSha256([string]$Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Value)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $hasher.Dispose()
    }
}

function Get-DirectoryPayloadEvidence([string]$Path, [string]$Description) {
    $resolved = Resolve-RequiredDirectory $Path $Description
    $files = @(Get-ChildItem -LiteralPath $resolved -Recurse -File | Sort-Object FullName)
    $closureLines = foreach ($file in $files) {
        $relative = $file.FullName.Substring($resolved.TrimEnd('\').Length).TrimStart('\').Replace('\', '/')
        $digest = Get-Sha256 $file.FullName
        "{0}`t{1}`t{2}" -f $relative, $digest, [long]$file.Length
    }
    return [ordered]@{
        path = $resolved
        file_count = $files.Count
        size_bytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
        closure_format = "relative_path_tab_sha256_tab_bytes_lf_v1"
        closure_sha256 = Get-TextSha256 (($closureLines -join "`n") + "`n")
    }
}

function Write-JsonNoBom([string]$Path, [object]$Value, [int]$Depth = 20) {
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth $Depth
    [IO.File]::WriteAllText($Path, $json + "`n", [Text.UTF8Encoding]::new($false))
}

function ConvertTo-WindowsCommandLineArgument([string]$Value) {
    if ($null -eq $Value) {
        return '""'
    }
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $slashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (2 * $slashes + 1)))
            [void]$builder.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) {
            [void]$builder.Append(('\' * $slashes))
            $slashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($slashes -gt 0) {
        [void]$builder.Append(('\' * (2 * $slashes)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Get-PathKey([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\').ToUpperInvariant()
}

function Assert-FiniteNonnegative([object]$Value, [string]$Description) {
    try {
        $number = [double]$Value
    }
    catch {
        throw "$Description must be numeric."
    }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt 0) {
        throw "$Description must be finite and non-negative."
    }
    return $number
}

function Assert-FinitePositive([object]$Value, [string]$Description) {
    $number = Assert-FiniteNonnegative $Value $Description
    if ($number -le 0) {
        throw "$Description must be positive."
    }
    return $number
}

function Get-UniqueDeliveryPayloadEvidence([string[]]$Directories, [string[]]$Files) {
    $uniquePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $totalBytes = [long]0
    foreach ($directory in $Directories) {
        foreach ($item in Get-ChildItem -LiteralPath $directory -Recurse -File) {
            $resolved = [IO.Path]::GetFullPath($item.FullName)
            if ($uniquePaths.Add($resolved)) {
                $length = [long]$item.Length
                if ($totalBytes -gt [long]::MaxValue - $length) {
                    throw "Delivery package size exceeds Int64 accounting capacity."
                }
                $totalBytes += $length
            }
        }
    }
    foreach ($file in $Files) {
        $resolved = Resolve-RequiredFile $file "delivery package file"
        if ($uniquePaths.Add($resolved)) {
            $length = [long](Get-Item -LiteralPath $resolved).Length
            if ($totalBytes -gt [long]::MaxValue - $length) {
                throw "Delivery package size exceeds Int64 accounting capacity."
            }
            $totalBytes += $length
        }
    }
    return [ordered]@{
        accounting = "unique_resolved_file_paths_case_insensitive_v1"
        file_count = $uniquePaths.Count
        size_bytes = $totalBytes
        size_mib = [Math]::Round($totalBytes / 1MB, 6)
    }
}

function Get-Percentile([double[]]$Values, [double]$Quantile) {
    if ($Values.Count -eq 0) {
        return $null
    }
    $sorted = @($Values | Sort-Object)
    $position = ($sorted.Count - 1) * $Quantile
    $lower = [int][Math]::Floor($position)
    $upper = [int][Math]::Ceiling($position)
    if ($lower -eq $upper) {
        return [double]$sorted[$lower]
    }
    return [double]$sorted[$lower] + ([double]$sorted[$upper] - [double]$sorted[$lower]) * ($position - $lower)
}

function Get-MetricSummary([double[]]$Values) {
    if ($Values.Count -eq 0) {
        return [ordered]@{ count = 0; mean = $null; p50 = $null; p95 = $null; maximum = $null }
    }
    return [ordered]@{
        count = $Values.Count
        mean = [Math]::Round(($Values | Measure-Object -Average).Average, 6)
        p50 = [Math]::Round((Get-Percentile $Values 0.50), 6)
        p95 = [Math]::Round((Get-Percentile $Values 0.95), 6)
        maximum = [Math]::Round(($Values | Measure-Object -Maximum).Maximum, 6)
    }
}

function Get-SystemEvidence() {
    $evidence = [ordered]@{
        machine_name = [Environment]::MachineName
        operating_system = [Environment]::OSVersion.VersionString
        process_architecture = if ([Environment]::Is64BitProcess) { "x64" } else { "x86" }
        os_architecture = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
        logical_processor_count = [Environment]::ProcessorCount
        processor = $null
        windows = $null
    }
    try {
        $processor = Get-CimInstance Win32_Processor | Select-Object -First 1
        $evidence.processor = [ordered]@{
            name = [string]$processor.Name
            manufacturer = [string]$processor.Manufacturer
            physical_cores = [int]$processor.NumberOfCores
            logical_processors = [int]$processor.NumberOfLogicalProcessors
            max_clock_mhz = [int]$processor.MaxClockSpeed
        }
        $windows = Get-CimInstance Win32_OperatingSystem
        $evidence.windows = [ordered]@{
            caption = [string]$windows.Caption
            version = [string]$windows.Version
            build_number = [string]$windows.BuildNumber
        }
    }
    catch {
        $evidence["cim_warning"] = $_.Exception.Message
    }
    return $evidence
}

function Read-WhiteRun(
    [string]$RunId,
    [string]$OutputDirectory,
    [int]$ExpectedCount,
    [object]$ProcessEvidence,
    [string[]]$ExpectedSources,
    [string]$ExpectedDeviceSha256,
    [string]$ExpectedOcrContractSha256,
    [object]$ExpectedStudentBundle,
    [string]$StdoutPath,
    [string]$StderrPath
) {
    if ((Get-Item -LiteralPath $StderrPath).Length -ne 0) {
        throw "$RunId wrote stderr; see $StderrPath"
    }
    $stdout = Get-Content -LiteralPath $StdoutPath -Raw -Encoding UTF8
    if ($stdout -notmatch '(?m)^OCR ONNX execution provider: cpu \(det/cls/rec\)\s*$') {
        throw "$RunId did not print strict CPU PP-OCR provider evidence."
    }
    $summaryPath = Join-Path $OutputDirectory "inference_summary.json"
    $manifestPath = Join-Path $OutputDirectory "inference_manifest.json"
    $errorsPath = Join-Path $OutputDirectory "inference_errors.jsonl"
    $summary = Get-Content -LiteralPath (Resolve-RequiredFile $summaryPath "$RunId summary") -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifestValue = Get-Content -LiteralPath (Resolve-RequiredFile $manifestPath "$RunId manifest") -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifest = @($manifestValue)
    $errors = Get-Content -LiteralPath (Resolve-RequiredFile $errorsPath "$RunId errors") -Raw -Encoding UTF8
    if ([string]$summary.document_type -ne "white" `
        -or [string]$summary.requested_device -ne "cpu" `
        -or [string]$summary.paddle_ocr_provider -ne "cpu" `
        -or [string]$summary.white_student_provider -ne "cpu" `
        -or [int]$summary.input -ne $ExpectedCount `
        -or [int]$summary.written -ne $ExpectedCount `
        -or [int]$summary.skipped -ne 0 `
        -or [int]$summary.errors -ne 0 `
        -or -not [string]::IsNullOrWhiteSpace($errors)) {
        throw "$RunId did not complete the exact zero-error white CPU workload."
    }
    if ($manifest.Count -ne $ExpectedCount) {
        throw "$RunId manifest count $($manifest.Count) differs from expected $ExpectedCount."
    }
    $expectedKeys = @($ExpectedSources | ForEach-Object { Get-PathKey $_ })
    $observedKeys = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $latencies = [Collections.Generic.List[double]]::new()
    $stageNames = @("image_load", "device", "paddle_ocr", "result_assembly")
    $stageValues = [ordered]@{}
    foreach ($stage in $stageNames) {
        $stageValues[$stage] = [Collections.Generic.List[double]]::new()
    }
    foreach ($record in $manifest) {
        if ([string]$record.status -ne "written") {
            throw "$RunId contains a non-written manifest record."
        }
        $source = [string]$record.source
        [void]$observedKeys.Add((Get-PathKey $source))
        $latencies.Add((Assert-FiniteNonnegative $record.inference_ms "$RunId inference_ms"))
        foreach ($stage in $stageNames) {
            $property = $record.stage_latency_ms.PSObject.Properties[$stage]
            if ($null -eq $property) {
                throw "$RunId manifest omitted stage_latency_ms.$stage"
            }
            $stageValues[$stage].Add((Assert-FiniteNonnegative $property.Value "$RunId $stage"))
        }
        $resultPath = [string]$record.result
        if (-not [IO.Path]::IsPathRooted($resultPath)) {
            $resultPath = Join-Path $OutputDirectory $resultPath
        }
        $result = Get-Content -LiteralPath (Resolve-RequiredFile $resultPath "$RunId white result") -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$result.document_type -ne "white" `
            -or [string]$result.inference_engine -ne "dotnet_onnxruntime_cpu" `
            -or [string]$result.ocr.provider -ne "cpu" `
            -or [string]$result.ocr.delivery_policy -ne "review_only" `
            -or [string]$result.ocr.student_model_status -ne "integrated_review_only" `
            -or [string]$result.ocr.student_provider -ne "cpu" `
            -or [string]$result.ocr.student_crop_source -ne "same_paddle_db_cls_oriented_crop" `
            -or $result.route.review_required -ne $true `
            -or [string]$result.model_contracts.runtime_source -ne "immutable_verified_bytes" `
            -or $result.model_contracts.reopened_paths_after_verification -ne $false `
            -or [string]$result.model_contracts.device_sha256 -ne $ExpectedDeviceSha256 `
            -or [string]$result.model_contracts.ocr_bundle_contract_sha256 -ne $ExpectedOcrContractSha256 `
            -or [string]$result.model_contracts.white_student_model -ne [string]$ExpectedStudentBundle.ModelName `
            -or [string]$result.model_contracts.white_student_model_sha256 -ne [string]$ExpectedStudentBundle.ModelSha256 `
            -or [long]$result.model_contracts.white_student_model_snapshot_size_bytes -ne [long]$ExpectedStudentBundle.ModelSizeBytes `
            -or [string]$result.model_contracts.white_student_charset -ne [string]$ExpectedStudentBundle.CharsetName `
            -or [string]$result.model_contracts.white_student_charset_sha256 -ne [string]$ExpectedStudentBundle.CharsetSha256 `
            -or [long]$result.model_contracts.white_student_charset_snapshot_size_bytes -ne [long]$ExpectedStudentBundle.CharsetSizeBytes `
            -or [string]$result.model_contracts.white_student_contract -ne [string]$ExpectedStudentBundle.ContractName `
            -or [string]$result.model_contracts.white_student_contract_sha256 -ne [string]$ExpectedStudentBundle.ContractSha256 `
            -or [long]$result.model_contracts.white_student_contract_snapshot_size_bytes -ne [long]$ExpectedStudentBundle.ContractSizeBytes `
            -or [string]$result.model_contracts.white_student_runtime_source -ne "immutable_verified_bytes" `
            -or $result.model_contracts.white_student_reopened_paths_after_verification -ne $false) {
            throw "$RunId result does not prove the strict white CPU/immutable/review-only contract: $resultPath"
        }
        $lines = @($result.lines)
        if ([int]$result.ocr.student_comparison_line_count -ne $lines.Count) {
            throw "$RunId student did not cover every PP-OCR DB line: $resultPath"
        }
        foreach ($line in $lines) {
            if ($null -eq $line.student `
                -or [string]$line.student.provider -ne "cpu" `
                -or [string]$line.student.delivery_policy -ne "review_only" `
                -or [string]$line.student.crop_source -ne "same_paddle_db_cls_oriented_crop") {
                throw "$RunId line does not prove the white student CPU/same-crop contract: $resultPath"
            }
        }
    }
    if ($observedKeys.Count -ne $expectedKeys.Count) {
        throw "$RunId observed source cardinality differs from expected."
    }
    foreach ($key in $expectedKeys) {
        if (-not $observedKeys.Contains($key)) {
            throw "$RunId omitted expected source: $key"
        }
    }
    $stages = [ordered]@{}
    foreach ($stage in $stageNames) {
        $stages[$stage] = Get-MetricSummary $stageValues[$stage].ToArray()
    }
    return [ordered]@{
        run_id = $RunId
        expected_count = $ExpectedCount
        process = $ProcessEvidence
        runtime_summary = Get-FileEvidence $summaryPath "$RunId summary"
        inference_latency_ms = Get-MetricSummary $latencies.ToArray()
        stage_latency_ms = $stages
        throughput_images_per_second = [Math]::Round($ExpectedCount / [double]$ProcessEvidence.elapsed_seconds, 6)
    }
}

function Invoke-WhiteCpuRun(
    [string]$RunId,
    [string]$OutputDirectory,
    [int]$ExpectedCount,
    [int]$Limit,
    [string[]]$ExpectedSources,
    [string]$ExpectedDeviceSha256,
    [string]$ExpectedOcrContractSha256,
    [object]$ExpectedStudentBundle
) {
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw "Refusing to reuse output directory: $OutputDirectory"
    }
    $runRoot = Split-Path -Parent $OutputDirectory
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    $stdoutPath = Join-Path $runRoot "stdout.log"
    $stderrPath = Join-Path $runRoot "stderr.log"
    $arguments = @(
        "--document-type", "white",
        "--device-model", $DeviceModel,
        "--ocr", "onnx",
        "--ocr-bundle", $OcrBundle,
        "--white-student-bundle", $WhiteStudentBundle,
        "--input-list", $InputList,
        "--output", $OutputDirectory,
        "--device", "cpu",
        "--rectification", "none",
        "--annotate", "none"
    )
    if ($Limit -gt 0) {
        $arguments += @("--limit", [string]$Limit)
    }
    $argumentLine = ($arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument ([string]$_ }) -join " ")
    $process = Start-Process `
        -FilePath $Executable `
        -ArgumentList $argumentLine `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru `
        -NoNewWindow
    $sampleCount = 0
    $sampledPeakWorkingSet = [long]0
    $sampledPeakPrivateBytes = [long]0
    while (-not $process.HasExited) {
        try {
            $process.Refresh()
            $sampledPeakWorkingSet = [Math]::Max($sampledPeakWorkingSet, [long]$process.WorkingSet64)
            $sampledPeakPrivateBytes = [Math]::Max($sampledPeakPrivateBytes, [long]$process.PrivateMemorySize64)
            $sampleCount++
        }
        catch {
            # The process may naturally exit between HasExited and Refresh.
        }
        if (-not $process.HasExited) {
            Start-Sleep -Milliseconds $PollIntervalMilliseconds
        }
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = [int]$process.ExitCode
    $startedUtc = $process.StartTime.ToUniversalTime()
    $finishedUtc = $process.ExitTime.ToUniversalTime()
    $elapsedSeconds = ($finishedUtc - $startedUtc).TotalSeconds
    $osPeakWorkingSet = [long]$process.PeakWorkingSet64
    $cpuSeconds = $process.TotalProcessorTime.TotalSeconds
    if ($exitCode -ne 0) {
        throw "$RunId failed with exit code $exitCode; stdout=$stdoutPath stderr=$stderrPath"
    }
    if ($sampleCount -eq 0 -or $elapsedSeconds -le 0) {
        throw "$RunId produced no valid process lifecycle/memory samples."
    }
    $processEvidence = [ordered]@{
        exit_code = $exitCode
        started_utc = $startedUtc.ToString("o")
        finished_utc = $finishedUtc.ToString("o")
        elapsed_seconds = [Math]::Round($elapsedSeconds, 6)
        total_processor_seconds = [Math]::Round($cpuSeconds, 6)
        normalized_cpu_utilization_percent = [Math]::Round(
            100.0 * $cpuSeconds / $elapsedSeconds / [Environment]::ProcessorCount, 4)
        logical_processor_count = [Environment]::ProcessorCount
        memory_poll_interval_ms = $PollIntervalMilliseconds
        memory_samples = $sampleCount
        sampled_peak_working_set_bytes = $sampledPeakWorkingSet
        os_peak_working_set_bytes = $osPeakWorkingSet
        peak_working_set_bytes = [Math]::Max($sampledPeakWorkingSet, $osPeakWorkingSet)
        sampled_peak_private_bytes = $sampledPeakPrivateBytes
        peak_private_bytes = $sampledPeakPrivateBytes
        private_bytes_measurement = "polled_PrivateMemorySize64"
        stdout = Get-FileEvidence $stdoutPath "$RunId stdout"
        stderr = Get-FileEvidence $stderrPath "$RunId stderr"
    }
    return Read-WhiteRun `
        $RunId $OutputDirectory $ExpectedCount $processEvidence $ExpectedSources `
        $ExpectedDeviceSha256 $ExpectedOcrContractSha256 $ExpectedStudentBundle $stdoutPath $stderrPath
}

$absoluteBudgetParameterNames = @(
    "MaxP50LatencyMilliseconds",
    "MaxP95LatencyMilliseconds",
    "MinThroughputImagesPerSecond",
    "MaxPeakWorkingSetMiB",
    "MaxPeakPrivateBytesMiB",
    "MaxPackageSizeMiB"
)
$providedAbsoluteBudgetParameterNames = @(
    $absoluteBudgetParameterNames | Where-Object { $PSBoundParameters.ContainsKey($_) }
)
$missingAbsoluteBudgetParameterNames = @(
    $absoluteBudgetParameterNames | Where-Object { -not $PSBoundParameters.ContainsKey($_) }
)
$absoluteBudgetComplete = $providedAbsoluteBudgetParameterNames.Count -eq $absoluteBudgetParameterNames.Count
$absoluteBudgetPartiallyProvided = `
    $providedAbsoluteBudgetParameterNames.Count -gt 0 -and -not $absoluteBudgetComplete
$hasBaselineEvidence = -not [string]::IsNullOrWhiteSpace($BaselineEvidence)
$absoluteBudgetValues = [ordered]@{
    max_p50_latency_ms = if ($PSBoundParameters.ContainsKey("MaxP50LatencyMilliseconds")) {
        Assert-FinitePositive $MaxP50LatencyMilliseconds "-MaxP50LatencyMilliseconds"
    } else { $null }
    max_p95_latency_ms = if ($PSBoundParameters.ContainsKey("MaxP95LatencyMilliseconds")) {
        Assert-FinitePositive $MaxP95LatencyMilliseconds "-MaxP95LatencyMilliseconds"
    } else { $null }
    min_throughput_images_per_second = if ($PSBoundParameters.ContainsKey("MinThroughputImagesPerSecond")) {
        Assert-FinitePositive $MinThroughputImagesPerSecond "-MinThroughputImagesPerSecond"
    } else { $null }
    max_peak_working_set_mib = if ($PSBoundParameters.ContainsKey("MaxPeakWorkingSetMiB")) {
        Assert-FinitePositive $MaxPeakWorkingSetMiB "-MaxPeakWorkingSetMiB"
    } else { $null }
    max_peak_private_bytes_mib = if ($PSBoundParameters.ContainsKey("MaxPeakPrivateBytesMiB")) {
        Assert-FinitePositive $MaxPeakPrivateBytesMiB "-MaxPeakPrivateBytesMiB"
    } else { $null }
    max_package_size_mib = if ($PSBoundParameters.ContainsKey("MaxPackageSizeMiB")) {
        Assert-FinitePositive $MaxPackageSizeMiB "-MaxPackageSizeMiB"
    } else { $null }
}
if ($absoluteBudgetComplete `
    -and [double]$absoluteBudgetValues.max_p50_latency_ms -gt [double]$absoluteBudgetValues.max_p95_latency_ms) {
    throw "-MaxP50LatencyMilliseconds may not exceed -MaxP95LatencyMilliseconds."
}
$configurationFailures = [Collections.Generic.List[string]]::new()
$diagnosticOnly = $false
if (-not $absoluteBudgetComplete) {
    $diagnosticOnly = $true
    $configurationFailures.Add(
        "Formal acceptance requires all absolute CPU budget parameters; baseline evidence is comparative only. Missing: " +
        (($missingAbsoluteBudgetParameterNames | ForEach-Object { "-$_" }) -join ", ")
    )
}

$Executable = Resolve-RequiredFile $Executable "published .NET executable"
$DeviceModel = Resolve-RequiredFile $DeviceModel "status-bar device model"
$OcrBundle = Resolve-RequiredDirectory $OcrBundle "PP-OCR delivery bundle"
$WhiteStudentBundle = Resolve-RequiredDirectory $WhiteStudentBundle "white line student bundle"
$InputList = Resolve-RequiredFile $InputList "held-out input list"
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse benchmark output root: $OutputRoot"
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$deviceContract = [IO.Path]::ChangeExtension($DeviceModel, ".contract.json")
$deviceModelEvidence = Get-FileEvidence $DeviceModel "status-bar device model"
$deviceContractEvidence = Get-FileEvidence $deviceContract "status-bar device contract"
$ocrContract = Join-Path $OcrBundle "paddle_ocr_delivery.contract.json"
$ocrContractEvidence = Get-FileEvidence $ocrContract "PP-OCR delivery contract"
$ocrContractPayload = Get-Content -LiteralPath $ocrContract -Raw -Encoding UTF8 | ConvertFrom-Json
$declaredOcrBytes = [long]$ocrContractPayload.package_size_bytes
if ($declaredOcrBytes -le 0) {
    throw "PP-OCR delivery contract package_size_bytes must be positive."
}
$studentContracts = @(Get-ChildItem -LiteralPath $WhiteStudentBundle -File -Filter "*.contract.json")
if ($studentContracts.Count -ne 1) {
    throw "White line student bundle must contain exactly one *.contract.json."
}
$studentContract = $studentContracts[0].FullName
$studentContractPayload = Get-Content -LiteralPath $studentContract -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$studentContractPayload.kind -ne "receipt_ocr_ctc_v1") {
    throw "White line student contract kind must be receipt_ocr_ctc_v1."
}
$studentModel = Resolve-RequiredFile `
    (Join-Path $WhiteStudentBundle ([string]$studentContractPayload.onnx_file)) "white student ONNX"
$studentCharset = Resolve-RequiredFile `
    (Join-Path $WhiteStudentBundle ([string]$studentContractPayload.charset_file)) "white student charset"
$studentBundleEvidence = [ordered]@{
    ModelName = [IO.Path]::GetFileName($studentModel)
    ModelSha256 = Get-Sha256 $studentModel
    ModelSizeBytes = [long](Get-Item -LiteralPath $studentModel).Length
    CharsetName = [IO.Path]::GetFileName($studentCharset)
    CharsetSha256 = Get-Sha256 $studentCharset
    CharsetSizeBytes = [long](Get-Item -LiteralPath $studentCharset).Length
    ContractName = [IO.Path]::GetFileName($studentContract)
    ContractSha256 = Get-Sha256 $studentContract
    ContractSizeBytes = [long](Get-Item -LiteralPath $studentContract).Length
}
if ([string]$studentContractPayload.onnx_sha256 -ne [string]$studentBundleEvidence.ModelSha256 `
    -or [string]$studentContractPayload.charset_sha256 -ne [string]$studentBundleEvidence.CharsetSha256) {
    throw "White line student model/charset identity differs from its contract."
}

$appRoot = Split-Path -Parent $Executable
$depsPath = Join-Path $appRoot (([IO.Path]::GetFileNameWithoutExtension($Executable)) + ".deps.json")
$depsEvidence = Get-FileEvidence $depsPath "published app deps.json"
$depsText = Get-Content -LiteralPath $depsPath -Raw -Encoding UTF8
if ($depsText -notmatch 'Microsoft\.ML\.OnnxRuntime/' -or $depsText -match 'Microsoft\.ML\.OnnxRuntime\.Gpu/') {
    throw "Published app dependency closure is not strict CPU ONNX Runtime."
}
$forbiddenGpuFiles = @(
    Get-ChildItem -LiteralPath $appRoot -Recurse -File |
        Where-Object { $_.Name -match '(?i)(cuda|cudnn|tensorrt|onnxruntime_providers_cuda)' }
)
if ($forbiddenGpuFiles.Count -gt 0) {
    throw "Published CPU app contains forbidden GPU runtime files: $($forbiddenGpuFiles.FullName -join ', ')"
}

$inputListDirectory = Split-Path -Parent $InputList
$sourceLines = @(
    foreach ($rawLine in Get-Content -LiteralPath $InputList -Encoding UTF8) {
        $entry = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($entry) -or $entry.StartsWith('#')) {
            continue
        }
        $candidate = if ([IO.Path]::IsPathRooted($entry)) {
            $entry
        }
        else {
            Join-Path $inputListDirectory $entry
        }
        Resolve-RequiredFile $candidate "held-out source image"
    }
)
if ($sourceLines.Count -eq 0) {
    throw "Held-out input list is empty."
}
$sourceEvidence = [Collections.Generic.List[object]]::new()
$sourceKeys = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($source in $sourceLines) {
    $evidence = Get-FileEvidence $source "held-out source image"
    if (-not $sourceKeys.Add((Get-PathKey ([string]$evidence.path)))) {
        throw "Held-out input list contains a duplicate source: $source"
    }
    $sourceEvidence.Add($evidence)
}
$sourceEvidencePath = Join-Path $OutputRoot "source-evidence.json"
Write-JsonNoBom $sourceEvidencePath $sourceEvidence 6
$warmupCount = [Math]::Min($WarmupImages, $sourceLines.Count)
$deliveryPackagePayloadEvidence = Get-UniqueDeliveryPayloadEvidence `
    @($appRoot, $OcrBundle, $WhiteStudentBundle) @($DeviceModel, $deviceContract)

$artifactsBefore = [ordered]@{
    executable = Get-FileEvidence $Executable "published .NET executable"
    deps_json = $depsEvidence
    app_payload = Get-DirectoryPayloadEvidence $appRoot "published app payload"
    device_model = $deviceModelEvidence
    device_contract = $deviceContractEvidence
    ocr_contract = $ocrContractEvidence
    ocr_declared_model_payload_bytes = $declaredOcrBytes
    ocr_bundle = Get-DirectoryPayloadEvidence $OcrBundle "PP-OCR delivery bundle"
    white_student_bundle = Get-DirectoryPayloadEvidence $WhiteStudentBundle "white line student bundle"
    white_student_model = Get-FileEvidence $studentModel "white student ONNX"
    white_student_charset = Get-FileEvidence $studentCharset "white student charset"
    white_student_contract = Get-FileEvidence $studentContract "white student contract"
    delivery_package_payload = $deliveryPackagePayloadEvidence
    input_list = Get-FileEvidence $InputList "held-out input list"
    source_evidence = Get-FileEvidence $sourceEvidencePath "held-out source evidence"
    cpu_provider = [ordered]@{
        onnxruntime_flavor = "cpu"
        deps_contains_cpu_onnxruntime = $true
        deps_contains_gpu_onnxruntime = $false
        forbidden_gpu_runtime_file_count = 0
    }
}

$warmupResults = [Collections.Generic.List[object]]::new()
for ($iteration = 1; $iteration -le $WarmupRuns; $iteration++) {
    $runId = "warmup-{0:D2}" -f $iteration
    $runRoot = Join-Path $OutputRoot ("runs\" + $runId)
    $warmupResults.Add((Invoke-WhiteCpuRun `
        $runId (Join-Path $runRoot "output") $warmupCount $warmupCount `
        $sourceLines[0..($warmupCount - 1)] `
        ([string]$deviceModelEvidence.sha256) ([string]$ocrContractEvidence.sha256) `
        $studentBundleEvidence))
}

$measuredResults = [Collections.Generic.List[object]]::new()
for ($iteration = 1; $iteration -le $Repetitions; $iteration++) {
    $runId = "measured-{0:D2}" -f $iteration
    $runRoot = Join-Path $OutputRoot ("runs\" + $runId)
    $measuredResults.Add((Invoke-WhiteCpuRun `
        $runId (Join-Path $runRoot "output") $sourceLines.Count 0 $sourceLines `
        ([string]$deviceModelEvidence.sha256) ([string]$ocrContractEvidence.sha256) `
        $studentBundleEvidence))
}

# Re-hash every frozen input and model contract after the measured runs.
if ((Get-Sha256 $Executable) -ne [string]$artifactsBefore.executable.sha256 `
    -or (Get-Sha256 $DeviceModel) -ne [string]$deviceModelEvidence.sha256 `
    -or (Get-Sha256 $deviceContract) -ne [string]$deviceContractEvidence.sha256 `
    -or (Get-Sha256 $ocrContract) -ne [string]$ocrContractEvidence.sha256 `
    -or (Get-Sha256 $studentModel) -ne [string]$studentBundleEvidence.ModelSha256 `
    -or (Get-Sha256 $studentCharset) -ne [string]$studentBundleEvidence.CharsetSha256 `
    -or (Get-Sha256 $studentContract) -ne [string]$studentBundleEvidence.ContractSha256 `
    -or (Get-Sha256 $InputList) -ne [string]$artifactsBefore.input_list.sha256) {
    throw "A benchmark artifact changed while the CPU runs were executing."
}
$appPayloadAfter = Get-DirectoryPayloadEvidence $appRoot "published app payload after benchmark"
$ocrBundleAfter = Get-DirectoryPayloadEvidence $OcrBundle "PP-OCR delivery bundle after benchmark"
$studentBundleAfter = Get-DirectoryPayloadEvidence $WhiteStudentBundle "white student bundle after benchmark"
$deliveryPackagePayloadAfter = Get-UniqueDeliveryPayloadEvidence `
    @($appRoot, $OcrBundle, $WhiteStudentBundle) @($DeviceModel, $deviceContract)
if ([string]$appPayloadAfter.closure_sha256 -ne [string]$artifactsBefore.app_payload.closure_sha256 `
    -or [long]$appPayloadAfter.size_bytes -ne [long]$artifactsBefore.app_payload.size_bytes `
    -or [string]$ocrBundleAfter.closure_sha256 -ne [string]$artifactsBefore.ocr_bundle.closure_sha256 `
    -or [long]$ocrBundleAfter.size_bytes -ne [long]$artifactsBefore.ocr_bundle.size_bytes `
    -or [string]$studentBundleAfter.closure_sha256 -ne [string]$artifactsBefore.white_student_bundle.closure_sha256 `
    -or [long]$studentBundleAfter.size_bytes -ne [long]$artifactsBefore.white_student_bundle.size_bytes `
    -or [long]$deliveryPackagePayloadAfter.size_bytes -ne [long]$deliveryPackagePayloadEvidence.size_bytes `
    -or [int]$deliveryPackagePayloadAfter.file_count -ne [int]$deliveryPackagePayloadEvidence.file_count) {
    throw "Published app, PP-OCR, or white student bundle closure changed while the CPU runs were executing."
}
foreach ($evidence in $sourceEvidence) {
    if ((Get-Sha256 ([string]$evidence.path)) -ne [string]$evidence.sha256) {
        throw "Held-out source image changed during CPU benchmark: $($evidence.path)"
    }
}

$allInference = [Collections.Generic.List[double]]::new()
$workingSets = [Collections.Generic.List[double]]::new()
$privateBytes = [Collections.Generic.List[double]]::new()
$cpuSeconds = [Collections.Generic.List[double]]::new()
$elapsedSeconds = [Collections.Generic.List[double]]::new()
$stageNames = @("image_load", "device", "paddle_ocr", "result_assembly")
$pooledStages = [ordered]@{}
foreach ($stage in $stageNames) {
    $pooledStages[$stage] = [Collections.Generic.List[double]]::new()
}
foreach ($run in $measuredResults) {
    $workingSets.Add([double]$run.process.peak_working_set_bytes)
    $privateBytes.Add([double]$run.process.peak_private_bytes)
    $cpuSeconds.Add([double]$run.process.total_processor_seconds)
    $elapsedSeconds.Add([double]$run.process.elapsed_seconds)
    $manifestPath = Join-Path (Split-Path -Parent ([string]$run.runtime_summary.path)) "inference_manifest.json"
    $manifest = @(Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    foreach ($record in $manifest) {
        $allInference.Add([double]$record.inference_ms)
        foreach ($stage in $stageNames) {
            $pooledStages[$stage].Add([double]$record.stage_latency_ms.$stage)
        }
    }
}
$stageSummary = [ordered]@{}
foreach ($stage in $stageNames) {
    $stageSummary[$stage] = Get-MetricSummary $pooledStages[$stage].ToArray()
}
$sumElapsed = ($elapsedSeconds | Measure-Object -Sum).Sum
$measured = [ordered]@{
    repetitions = $Repetitions
    input_images_per_run = $sourceLines.Count
    total_images = $sourceLines.Count * $Repetitions
    total_process_elapsed_seconds = [Math]::Round($sumElapsed, 6)
    throughput_images_per_second = [Math]::Round(($sourceLines.Count * $Repetitions) / $sumElapsed, 6)
    inference_latency_ms = Get-MetricSummary $allInference.ToArray()
    stage_latency_ms = $stageSummary
    peak_working_set_bytes = Get-MetricSummary $workingSets.ToArray()
    peak_private_bytes = Get-MetricSummary $privateBytes.ToArray()
    delivery_package_payload_bytes = [long]$deliveryPackagePayloadEvidence.size_bytes
    delivery_package_payload_mib = [double]$deliveryPackagePayloadEvidence.size_mib
    total_processor_seconds_per_run = Get-MetricSummary $cpuSeconds.ToArray()
    process_elapsed_seconds_per_run = Get-MetricSummary $elapsedSeconds.ToArray()
    runs = $measuredResults
}

$baselineGateFailures = [Collections.Generic.List[string]]::new()
$absoluteGateFailures = [Collections.Generic.List[string]]::new()
$baselineBinding = $null
if ($hasBaselineEvidence) {
    $BaselineEvidence = Resolve-RequiredFile $BaselineEvidence "baseline CPU benchmark evidence"
    $baseline = Get-Content -LiteralPath $BaselineEvidence -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$baseline.kind -ne "otherimages_dotnet_white_cpu_benchmark_v1" `
        -or $baseline.accepted -ne $true) {
        throw "Baseline evidence is not an accepted white CPU benchmark v1 report."
    }
    $baselineBinding = Get-FileEvidence $BaselineEvidence "baseline CPU benchmark evidence"
    $minimumThroughput = [double]$baseline.measured.throughput_images_per_second * (1.0 - $MaxThroughputRegressionPercent / 100.0)
    if ([double]$measured.throughput_images_per_second -lt $minimumThroughput) {
        $baselineGateFailures.Add("throughput regressed more than $MaxThroughputRegressionPercent percent")
    }
    foreach ($latencyName in @("p50", "p95")) {
        $baselineLatency = [double]$baseline.measured.inference_latency_ms.$latencyName
        $candidateLatency = [double]$measured.inference_latency_ms.$latencyName
        if ($candidateLatency -gt $baselineLatency * (1.0 + $MaxLatencyRegressionPercent / 100.0)) {
            $baselineGateFailures.Add("inference $latencyName regressed more than $MaxLatencyRegressionPercent percent")
        }
    }
    $absoluteMemoryBytes = [long]$MaxMemoryAbsoluteIncreaseMiB * 1024L * 1024L
    foreach ($memoryName in @("peak_working_set_bytes", "peak_private_bytes")) {
        $baselineMemory = [double]$baseline.measured.$memoryName.maximum
        $candidateMemory = [double]$measured.$memoryName.maximum
        if ($candidateMemory -gt $baselineMemory * (1.0 + $MaxMemoryRegressionPercent / 100.0)) {
            $baselineGateFailures.Add("$memoryName regressed more than $MaxMemoryRegressionPercent percent")
        }
        if ($candidateMemory - $baselineMemory -gt $absoluteMemoryBytes) {
            $baselineGateFailures.Add("$memoryName increased more than $MaxMemoryAbsoluteIncreaseMiB MiB")
        }
    }
}

if ($absoluteBudgetComplete) {
    if ([double]$measured.inference_latency_ms.p50 -gt [double]$absoluteBudgetValues.max_p50_latency_ms) {
        $absoluteGateFailures.Add(
            "inference p50 exceeds absolute budget $($absoluteBudgetValues.max_p50_latency_ms) ms"
        )
    }
    if ([double]$measured.inference_latency_ms.p95 -gt [double]$absoluteBudgetValues.max_p95_latency_ms) {
        $absoluteGateFailures.Add(
            "inference p95 exceeds absolute budget $($absoluteBudgetValues.max_p95_latency_ms) ms"
        )
    }
    if ([double]$measured.throughput_images_per_second `
        -lt [double]$absoluteBudgetValues.min_throughput_images_per_second) {
        $absoluteGateFailures.Add(
            "throughput is below absolute budget $($absoluteBudgetValues.min_throughput_images_per_second) images/s"
        )
    }
    $maxWorkingSetBytes = [double]$absoluteBudgetValues.max_peak_working_set_mib * 1MB
    if ([double]$measured.peak_working_set_bytes.maximum -gt $maxWorkingSetBytes) {
        $absoluteGateFailures.Add(
            "peak working set exceeds absolute budget $($absoluteBudgetValues.max_peak_working_set_mib) MiB"
        )
    }
    $maxPrivateBytes = [double]$absoluteBudgetValues.max_peak_private_bytes_mib * 1MB
    if ([double]$measured.peak_private_bytes.maximum -gt $maxPrivateBytes) {
        $absoluteGateFailures.Add(
            "peak private bytes exceeds absolute budget $($absoluteBudgetValues.max_peak_private_bytes_mib) MiB"
        )
    }
    $maxPackageBytes = [double]$absoluteBudgetValues.max_package_size_mib * 1MB
    if ([double]$measured.delivery_package_payload_bytes -gt $maxPackageBytes) {
        $absoluteGateFailures.Add(
            "delivery package payload exceeds absolute budget $($absoluteBudgetValues.max_package_size_mib) MiB"
        )
    }
}

$gateFailures = [Collections.Generic.List[string]]::new()
foreach ($failure in $configurationFailures) { $gateFailures.Add($failure) }
foreach ($failure in $baselineGateFailures) { $gateFailures.Add($failure) }
foreach ($failure in $absoluteGateFailures) { $gateFailures.Add($failure) }
$formalGateConfigured = $absoluteBudgetComplete
$reportAccepted = $formalGateConfigured -and -not $diagnosticOnly -and $gateFailures.Count -eq 0
$selectedExitCode = if ($reportAccepted) { 0 } elseif ($diagnosticOnly) { 3 } else { 2 }
$gateMode = if ($diagnosticOnly) {
    "diagnostic_only_incomplete_budget_configuration"
}
elseif ($null -ne $baselineBinding -and $absoluteBudgetComplete) {
    "baseline_regression_and_absolute_budget"
}
else {
    "absolute_budget"
}

$report = [ordered]@{
    schema_version = 1
    kind = "otherimages_dotnet_white_cpu_benchmark_v1"
    accepted = $reportAccepted
    diagnostic_only = $diagnosticOnly
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    artifacts = $artifactsBefore
    system = Get-SystemEvidence
    workload = [ordered]@{
        document_type = "white"
        requested_device = "cpu"
        paddle_ocr_provider = "cpu"
        white_student_provider = "cpu"
        input_count = $sourceLines.Count
        warmup_runs = $WarmupRuns
        warmup_images = $warmupCount
        measured_repetitions = $Repetitions
        memory_poll_interval_ms = $PollIntervalMilliseconds
        output_root = $OutputRoot
    }
    warmup = $warmupResults
    measured = $measured
    efficiency_gate = [ordered]@{
        mode = $gateMode
        formal_gate_configured = $formalGateConfigured
        baseline_regression = [ordered]@{
            baseline = $baselineBinding
            applicable = ($null -ne $baselineBinding)
            max_throughput_regression_percent = $MaxThroughputRegressionPercent
            max_latency_regression_percent = $MaxLatencyRegressionPercent
            max_memory_regression_percent = $MaxMemoryRegressionPercent
            max_memory_absolute_increase_mib = $MaxMemoryAbsoluteIncreaseMiB
            accepted = ($null -ne $baselineBinding -and $baselineGateFailures.Count -eq 0)
            failures = $baselineGateFailures
        }
        absolute_budget = [ordered]@{
            required_without_baseline = $true
            applicable = $absoluteBudgetComplete
            complete = $absoluteBudgetComplete
            provided_parameters = $providedAbsoluteBudgetParameterNames
            missing_parameters = $missingAbsoluteBudgetParameterNames
            limits = $absoluteBudgetValues
            measured_package_payload_bytes = [long]$measured.delivery_package_payload_bytes
            accepted = ($absoluteBudgetComplete -and $absoluteGateFailures.Count -eq 0)
            failures = $absoluteGateFailures
        }
        diagnostic_only = $diagnosticOnly
        accepted = $reportAccepted
        failures = $gateFailures
    }
    exit_semantics = [ordered]@{
        accepted_exit_code = 0
        rejected_exit_code = 2
        diagnostic_only_exit_code = 3
        selected_exit_code = $selectedExitCode
    }
}
$reportPath = Join-Path $OutputRoot "white-cpu-benchmark.json"
Write-JsonNoBom $reportPath $report 30
Write-Host "White CPU benchmark accepted=$($report.accepted) diagnostic_only=$($report.diagnostic_only)"
Write-Host "  throughput=$($measured.throughput_images_per_second) images/s"
Write-Host "  p50=$($measured.inference_latency_ms.p50) ms; p95=$($measured.inference_latency_ms.p95) ms"
Write-Host "  peak WorkingSet=$([Math]::Round($measured.peak_working_set_bytes.maximum / 1MB, 2)) MiB"
Write-Host "  peak PrivateBytes=$([Math]::Round($measured.peak_private_bytes.maximum / 1MB, 2)) MiB"
Write-Host "  delivery package=$([Math]::Round($measured.delivery_package_payload_bytes / 1MB, 2)) MiB"
Write-Host "  evidence=$reportPath"
if ($diagnosticOnly) {
    Write-Host "White CPU benchmark is diagnostic-only: $($configurationFailures -join '; ')"
    exit 3
}
if (-not $reportAccepted) {
    Write-Host "White CPU efficiency gate rejected the candidate: $($gateFailures -join '; ')"
    exit 2
}
