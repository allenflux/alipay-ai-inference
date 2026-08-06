[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
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

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
$endToEndScorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$projectFile = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"

if ([string]::IsNullOrWhiteSpace($DetectorModel)) {
    $DetectorModel = Join-Path $repoRoot "artifacts\receipt_lrcnn_v1.onnx"
}
if ([string]::IsNullOrWhiteSpace($DeviceModel)) {
    $DeviceModel = Join-Path $repoRoot "artifacts\statusbar_device_v1.onnx"
}

$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
$Output = [IO.Path]::GetFullPath($Output)
$DeliveryDir = [IO.Path]::GetFullPath($DeliveryDir)
$DetectorModel = [IO.Path]::GetFullPath($DetectorModel)
$DeviceModel = [IO.Path]::GetFullPath($DeviceModel)
$hasRecords = -not [string]::IsNullOrWhiteSpace($Records)
$hasEndToEndEvaluationDir = -not [string]::IsNullOrWhiteSpace($EndToEndEvaluationDir)

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
if ($hasRecords) {
    $Records = [IO.Path]::GetFullPath($Records)
    $EndToEndEvaluationDir = [IO.Path]::GetFullPath($EndToEndEvaluationDir)
    if (-not (Test-Path -LiteralPath $Records -PathType Leaf)) {
        throw "Missing end-to-end evaluation records: $Records"
    }
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

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Test-PathWithin([string]$Candidate, [string]$Parent) {
    if ($Candidate.Equals($Parent, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $parentPrefix = $Parent
    if (-not $parentPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
        $parentPrefix += [IO.Path]::DirectorySeparatorChar
    }
    return $Candidate.StartsWith($parentPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Read-NormalizedJson([string]$Path) {
    $json = ((& $pythonExe $normalizer $Path) -join "`n")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not normalize JSON evidence: $Path"
    }
    return $json | ConvertFrom-Json
}

function Assert-StandardModelContract([string]$ModelPath, [string]$ExpectedKind) {
    $contractPath = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    Require-File $ModelPath "$ExpectedKind ONNX"
    Require-File $contractPath "$ExpectedKind contract"
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
    $labelsPath = [IO.Path]::ChangeExtension($ModelPath, ".labels.json")
    $contractPath = [IO.Path]::ChangeExtension($ModelPath, ".contract.json")
    Require-File $ModelPath "unified OCR ONNX"
    Require-File $labelsPath "unified OCR labels"
    Require-File $contractPath "unified OCR contract"

    $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$contract.kind -ne "receipt_unified_field_reader_v12") {
        throw "Unified OCR contract is not architecture v12: $contractPath"
    }
    if ([int]$contract.model.architecture_version -ne 12) {
        throw "Unified OCR contract has an unsupported architecture_version: $($contract.model.architecture_version)"
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
    return @($labelsPath, $contractPath)
}

Require-File $pythonExe "project Python interpreter"
Require-File $normalizer "JSON normalizer"
Require-File $projectFile "ML.NET project"

$unifiedModel = Join-Path $RunDirectory "best.onnx"
$unifiedSidecars = Assert-UnifiedBundle $unifiedModel
$unifiedLabels = $unifiedSidecars[0]
$unifiedContract = $unifiedSidecars[1]
$unifiedContractPayload = Get-Content -LiteralPath $unifiedContract -Raw -Encoding UTF8 | ConvertFrom-Json
$textDeliveryPolicy = [string]$unifiedContractPayload.text_delivery_policy.runtime_policy
$textReviewValue = [string]$unifiedContractPayload.text_delivery_policy.review_value
if ($textDeliveryPolicy -ne "review_only_pending_independent_human_truth_calibration" -or $textReviewValue -ne "review") {
    throw "Unified OCR text delivery policy is not the required fail-closed review-only policy."
}
$onnxValidationSummary = Join-Path $RunDirectory "onnx-val\summary.json"
Require-File $onnxValidationSummary "final ONNX validation summary"

$detectorContract = Assert-StandardModelContract $DetectorModel "receipt_lrcnn_v1"
$deviceContract = $null
if ($IncludeDeviceModel) {
    $deviceContract = Assert-StandardModelContract $DeviceModel "statusbar_device_v1"
}

$summary = Read-NormalizedJson $onnxValidationSummary
$unifiedModelSha256 = Get-Sha256 $unifiedModel
if ([string]$summary.model_sha256 -ne $unifiedModelSha256) {
    throw "onnx-val summary model_sha256 does not belong to best.onnx."
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
    $resolvedInput = [IO.Path]::GetFullPath($InputPath)
    if (-not (Test-Path -LiteralPath $resolvedInput)) {
        throw "Input does not exist: $resolvedInput"
    }
    $supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if (Test-Path -LiteralPath $resolvedInput -PathType Leaf) {
        if ($supportedExtensions -notcontains [IO.Path]::GetExtension($resolvedInput).ToLowerInvariant()) {
            throw "Input file has an unsupported image extension: $resolvedInput"
        }
        $expectedRecords = 1
    }
    else {
        $availableRecords = @(
            Get-ChildItem -LiteralPath $resolvedInput -Recurse -File |
                Where-Object { $supportedExtensions -contains $_.Extension.ToLowerInvariant() }
        ).Count
        $expectedRecords = if ($Limit -gt 0) { [Math]::Min($availableRecords, $Limit) } else { $availableRecords }
    }
}
else {
    $resolvedInputList = [IO.Path]::GetFullPath($InputList)
    Require-File $resolvedInputList "input list"
    $listRoot = Split-Path -Parent $resolvedInputList
    $seenInputRecords = @{}
    $supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
    foreach ($line in Get-Content -LiteralPath $resolvedInputList -Encoding UTF8) {
        $candidate = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($candidate) -or $candidate.StartsWith("#", [StringComparison]::Ordinal)) {
            continue
        }
        if (-not [IO.Path]::IsPathRooted($candidate)) {
            $candidate = Join-Path $listRoot $candidate
        }
        $candidate = [IO.Path]::GetFullPath($candidate)
        Require-File $candidate "input-list image"
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
    if ($EndToEndEvaluationDir.Equals($Output, [StringComparison]::OrdinalIgnoreCase)) {
        throw "EndToEndEvaluationDir must not be the same path as Output."
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
$stagingRoot = Join-Path $deliveryParent (".receipt-mlnet-unified-staging-" + [Guid]::NewGuid().ToString("N"))
$appDirectory = Join-Path $stagingRoot "app"
$modelDirectory = Join-Path $stagingRoot "models"
$unifiedDirectory = Join-Path $modelDirectory "unified"
$evidenceDirectory = Join-Path $stagingRoot "evidence"
$consoleLog = Join-Path $evidenceDirectory "console.log"
$published = $false

try {
    New-Item -ItemType Directory -Path $appDirectory, $modelDirectory, $unifiedDirectory, $evidenceDirectory | Out-Null
    [IO.File]::WriteAllText($consoleLog, "")

    $formalExpectedInputList = $null
    if ($hasRecords) {
        $formalExpectedInputList = Join-Path $evidenceDirectory "expected-val-input-list.txt"
        Write-Host "mlnet_unified_prepare_full_val"
        & $pythonExe $endToEndScorer prepare `
            --records $Records `
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
            if (-not [IO.Path]::IsPathRooted($candidate)) {
                $candidate = Join-Path $formalListRoot $candidate
            }
            $candidate = [IO.Path]::GetFullPath($candidate)
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
        $manifestSource = [IO.Path]::GetFullPath([string]$manifestRecord.source)
        if ($manifestSourceSet.ContainsKey($manifestSource)) {
            throw "Inference manifest contains a duplicate source: $manifestSource"
        }
        $manifestSourceSet[$manifestSource] = $true
        $inferenceMs = [double]$manifestRecord.inference_ms
        if ([double]::IsNaN($inferenceMs) -or [double]::IsInfinity($inferenceMs) -or $inferenceMs -lt 0.0) {
            throw "Manifest inference_ms is invalid for source: $manifestSource"
        }
        $resultPath = [string]$manifestRecord.result
        Require-File $resultPath "ML.NET receipt result"
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$result.inference_engine -ne "mlnet") {
            throw "Unexpected inference engine in result: $resultPath"
        }
        if ([string]$result.model_contracts.unified_ocr_model_sha256 -ne $unifiedModelSha256) {
            throw "Result does not reference the delivered unified OCR model: $resultPath"
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
            $candidate = [string]$field.Value.candidate
            if ([string]::IsNullOrWhiteSpace($candidate)) {
                $receiptCandidateComplete = $false
                if ([string]$field.Value.state -notin @("absent", "unreadable") `
                    -or ($null -ne $field.Value.value -and [string]$field.Value.value -ne $textReviewValue) `
                    -or ($null -ne $field.Value.delivery_value -and [string]$field.Value.delivery_value -ne $textReviewValue)) {
                    throw "Result $fieldName has an invalid fail-closed missing-candidate state: $resultPath"
                }
                continue
            }
            $candidateByField[$fieldName]++
            if ([string]$field.Value.delivery_value -ne $textReviewValue `
                -or [string]$field.Value.value -ne $textReviewValue `
                -or [string]$field.Value.state -ne "review") {
                throw "Result $fieldName candidate escaped the required review-only policy: $resultPath"
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
            "--records", $Records,
            "--results", $Output,
            "--model", $unifiedModel,
            "--output", $EndToEndEvaluationDir,
            "--split", "val",
            "--amount-floor", [Convert]::ToString($AmountFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--time-floor", [Convert]::ToString($TimeFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--payment-floor", [Convert]::ToString($PaymentFloor, [Globalization.CultureInfo]::InvariantCulture),
            "--recipient-floor", [Convert]::ToString($RecipientFloor, [Globalization.CultureInfo]::InvariantCulture)
        )
        & $pythonExe $endToEndScorer @scoreArguments 2>&1 | Tee-Object -FilePath $consoleLog -Append
        $scoreExitCode = $LASTEXITCODE
        $endToEndSummaryPath = Join-Path $EndToEndEvaluationDir "summary.json"
        $endToEndComparisonsPath = Join-Path $EndToEndEvaluationDir "comparisons.jsonl"
        Require-File $endToEndSummaryPath "ML.NET end-to-end evaluation summary"
        Require-File $endToEndComparisonsPath "ML.NET end-to-end comparisons"
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
            throw "ML.NET end-to-end score is not bound to the delivered best.onnx."
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
        $validationScope = "full_val_end_to_end_scored_cpu"
        $endToEndEvidence = [ordered]@{
            performed = $true
            status = "accepted"
            records = $Records
            records_sha256 = Get-Sha256 $Records
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
        runtime_flavor = $RuntimeFlavor
        runtime_device = $runtimeDevice
        inference_summary = $runtimeSummary
        end_to_end_evaluation = $endToEndEvidence
        onnx_validation = [ordered]@{
            providers = $providers
            accepted = [bool]$summary.acceptance.passed
            fields = $validatedMetrics
        }
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
    }
    $packageConfig = [ordered]@{
        schema_version = 1
        kind = if ($hasRecords) { "receipt_mlnet_unified_delivery_package_v1" } else { "receipt_mlnet_unified_candidate_smoke_package_v1" }
        framework = "net8.0"
        runtime_identifier = "win-x64"
        self_contained = $false
        onnx_runtime_flavor = $RuntimeFlavor
        runtime_device = $runtimeDevice
        prerequisites = if ($RuntimeFlavor -eq "cpu") {
            @("Microsoft.NETCore.App 8.x")
        }
        else {
            @("Microsoft.NETCore.App 8.x", "NVIDIA CUDA 12.x", "NVIDIA cuDNN 9.x")
        }
        validation_scope = $validationScope
        run_directory = $RunDirectory
        input = $resolvedInput
        input_list = $resolvedInputList
        records = if ($hasRecords) { $Records } else { $null }
        end_to_end_evaluation = if ($hasRecords) { $EndToEndEvaluationDir } else { $null }
        limit = $Limit
        detector_model = [IO.Path]::GetFileName($DetectorModel)
        device_model = if ($IncludeDeviceModel) { [IO.Path]::GetFileName($DeviceModel) } else { $null }
        unified_model = "models/unified/$([IO.Path]::GetFileName($unifiedModel))"
        text_delivery_policy = $textDeliveryPolicy
    }
    $packageConfig | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $evidenceDirectory "package_config.json") -Encoding UTF8

    $hashRows = @(
        Get-ChildItem -LiteralPath $stagingRoot -Recurse -File |
            Sort-Object FullName |
            ForEach-Object {
                $relativePath = $_.FullName.Substring($stagingRoot.Length)
                while ($relativePath.StartsWith("\", [StringComparison]::Ordinal) -or $relativePath.StartsWith("/", [StringComparison]::Ordinal)) {
                    $relativePath = $relativePath.Substring(1)
                }
                [ordered]@{
                    path = $relativePath.Replace('\', '/')
                    sha256 = Get-Sha256 $_.FullName
                    bytes = $_.Length
                }
            }
    )
    ConvertTo-Json -InputObject @($hashRows) -Depth 5 |
        Set-Content -LiteralPath (Join-Path $stagingRoot "SHA256SUMS.json") -Encoding UTF8

    if (Test-Path -LiteralPath $DeliveryDir) {
        throw "Delivery directory appeared during validation; refusing to overwrite it: $DeliveryDir"
    }
    Move-Item -LiteralPath $stagingRoot -Destination $DeliveryDir
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
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
