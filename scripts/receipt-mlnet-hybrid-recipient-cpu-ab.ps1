[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Records,
    [Parameter(Mandatory = $true)]
    [string]$UnifiedModel,
    [Parameter(Mandatory = $true)]
    [string]$PaddleDeliveryBundle,
    [string]$DetectorModel,
    [string]$DeviceModel,
    [ValidateSet("pilot", "formal")]
    [string]$Mode = "pilot",
    [ValidateRange(0, 1000000)]
    [int]$Limit = 200,
    [ValidateRange(0, 1024)]
    [int]$DetectorIntraOpThreads = 0,
    [ValidateRange(0.0, 250.0)]
    [double]$MaxP95OverheadMs = 250.0,
    [string]$OutputDirectory,
    [string]$DotnetExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$modeName = $Mode.ToLowerInvariant()
$requiredFormalReceipts = 10016
$expectedSelectionOrder = if ($modeName -eq "pilot") {
    "deterministic_min16_field_quota_then_records_manifest_order"
}
else {
    "first_unique_source_in_records_manifest_order"
}
if ($modeName -eq "formal" -and $Limit -ne 0) {
    throw "Formal CPU A/B must use -Limit 0 so the complete validation split is evaluated."
}
if ($modeName -eq "pilot" -and $Limit -le 0) {
    throw "Pilot CPU A/B requires a positive -Limit; use -Mode formal -Limit 0 for the complete split."
}

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repositoryRoot ".venv-cu126\Scripts\python.exe"
$project = Join-Path $repositoryRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"
$parserContractProject = Join-Path $repositoryRoot (
    "dotnet\ReceiptMlNet.Cli.PaddleRecipientContractTests\ReceiptMlNet.Cli.PaddleRecipientContractTests.csproj"
)
$scorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$comparator = Join-Path $PSScriptRoot "receipt-mlnet-hybrid-recipient-ab.py"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-CliAppClosureManifest([string]$AppRoot, [string]$ManifestPath) {
    $root = [IO.Path]::GetFullPath($AppRoot)
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "Missing frozen CLI app directory: $root"
    }
    $rootPrefix = $root
    if (-not $rootPrefix.EndsWith([IO.Path]::DirectorySeparatorChar.ToString())) {
        $rootPrefix += [IO.Path]::DirectorySeparatorChar
    }
    $pending = New-Object System.Collections.Queue
    $pending.Enqueue($root)
    $rows = @()
    while ($pending.Count -gt 0) {
        $directory = [string]$pending.Dequeue()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Frozen CLI app contains a reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pending.Enqueue($item.FullName)
                continue
            }
            $fullPath = [IO.Path]::GetFullPath($item.FullName)
            if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Frozen CLI app file escaped its root: $fullPath"
            }
            $rows += [pscustomobject][ordered]@{
                path = $fullPath.Substring($rootPrefix.Length).Replace('\', '/')
                sha256 = Get-Sha256 $fullPath
                size_bytes = [long]$item.Length
            }
        }
    }
    $rows = @($rows | Sort-Object @{ Expression = { $_.path.ToLowerInvariant() } }, @{ Expression = { $_.path } })
    if ($rows.Count -le 0) {
        throw "Frozen CLI app closure is empty."
    }
    $json = ConvertTo-Json -InputObject @($rows) -Depth 5
    [IO.File]::WriteAllText(
        $ManifestPath,
        $json + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false)))
}

if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    $dotnetCommand = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($null -ne $dotnetCommand) {
        $DotnetExe = $dotnetCommand.Source
    } else {
        $portableDotnet = Join-Path $repositoryRoot "artifacts\dotnet8\dotnet.exe"
        if (Test-Path -LiteralPath $portableDotnet -PathType Leaf) {
            $DotnetExe = $portableDotnet
        }
    }
}
if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    throw "Missing .NET 8 host. Install dotnet, place the portable host at artifacts\dotnet8\dotnet.exe, or pass -DotnetExe."
}
$DotnetExe = [IO.Path]::GetFullPath($DotnetExe)

if ([string]::IsNullOrWhiteSpace($DetectorModel)) {
    $DetectorModel = Join-Path $repositoryRoot "artifacts\receipt_lrcnn_v1.onnx"
}
if ([string]::IsNullOrWhiteSpace($DeviceModel)) {
    $DeviceModel = Join-Path $repositoryRoot "artifacts\statusbar_device_v1.onnx"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path (Split-Path -Parent ([IO.Path]::GetFullPath($Records))) (
        "mlnet-hybrid-recipient-cpu-$modeName-ab-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    )
}

foreach ($required in @(
    @{ Name = "project Python"; Path = $pythonExe; Kind = "Leaf" },
    @{ Name = ".NET host"; Path = $DotnetExe; Kind = "Leaf" },
    @{ Name = "ML.NET project"; Path = $project; Kind = "Leaf" },
    @{ Name = "recipient parser contract project"; Path = $parserContractProject; Kind = "Leaf" },
    @{ Name = "val records"; Path = $Records; Kind = "Leaf" },
    @{ Name = "v13 unified model"; Path = $UnifiedModel; Kind = "Leaf" },
    @{ Name = "Paddle delivery bundle"; Path = $PaddleDeliveryBundle; Kind = "Container" },
    @{ Name = "receipt detector"; Path = $DetectorModel; Kind = "Leaf" },
    @{ Name = "device classifier"; Path = $DeviceModel; Kind = "Leaf" },
    @{ Name = "val input preparer"; Path = $scorer; Kind = "Leaf" },
    @{ Name = "A/B comparator"; Path = $comparator; Kind = "Leaf" }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType $required.Kind)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
$recordsSha256 = Get-Sha256 $Records

$unifiedContract = [IO.Path]::ChangeExtension([IO.Path]::GetFullPath($UnifiedModel), ".contract.json")
$unifiedLabels = [IO.Path]::ChangeExtension([IO.Path]::GetFullPath($UnifiedModel), ".labels.json")
foreach ($sidecar in @($unifiedContract, $unifiedLabels)) {
    if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
        throw "Missing unified model sidecar: $sidecar"
    }
}
$contract = Get-Content -LiteralPath $unifiedContract -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$contract.model.architecture_version -ne 13 `
    -or [string]$contract.kind -ne "receipt_unified_field_reader_v13") {
    throw "Hybrid recipient CPU A/B requires an architecture-v13 unified model."
}

$paddleContract = Join-Path $PaddleDeliveryBundle "paddle_ocr_delivery.contract.json"
if (-not (Test-Path -LiteralPath $paddleContract -PathType Leaf)) {
    throw "Missing Paddle delivery contract: $paddleContract"
}
$paddle = Get-Content -LiteralPath $paddleContract -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$paddle.kind -ne "paddle_ocr_v2_delivery" `
    -or $null -eq $paddle.models.det `
    -or $null -eq $paddle.models.cls `
    -or $null -eq $paddle.models.rec) {
    throw "Paddle delivery bundle is not a complete det/cls/rec ONNX package."
}

$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse hybrid recipient CPU A/B output: $OutputDirectory"
}
$outputParent = Split-Path -Parent $OutputDirectory
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$preparedInputList = Join-Path $OutputDirectory "prepared-full-val-inputs.txt"
$inputList = Join-Path $OutputDirectory "fixed-selected-inputs.txt"
$baselineOutput = Join-Path $OutputDirectory "baseline-v13"
$hybridOutput = Join-Path $OutputDirectory "hybrid-recipient"
$reportOutput = Join-Path $OutputDirectory "comparison"
$scoreOutput = Join-Path $OutputDirectory "hybrid-val-score"
$cliPublishDirectory = Join-Path $OutputDirectory "cli-app"
$cliAssembly = Join-Path $cliPublishDirectory "ReceiptMlNet.Cli.dll"
$cliClosureManifest = Join-Path $OutputDirectory "cli-app-closure.json"

Write-Host "receipt_mlnet_hybrid_recipient_cpu_ab"
Write-Host "  split=val (prepared by scorer; held-out test is never opened)"
Write-Host "  mode=$modeName; limit=$Limit; device=cpu; detector/device enabled"
Write-Host "  fixed-p95-overhead-ceiling=$MaxP95OverheadMs ms (release maximum=250 ms)"
Write-Host "  baseline=v13; candidate=v13 + PP-OCR det/cls/SVTR_LCNet recipient-only"
Write-Host "  output=$OutputDirectory"

& $DotnetExe run --project $parserContractProject -c Release
if ($LASTEXITCODE -ne 0) {
    throw "PP-OCR recipient parser contract tests failed with exit code $LASTEXITCODE"
}

& $pythonExe $scorer prepare --records $Records --output $preparedInputList --split val
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare fixed val input list; exit code $LASTEXITCODE"
}
$preparedInputs = @(
    Get-Content -LiteralPath $preparedInputList -Encoding UTF8 |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#") }
)
if ($preparedInputs.Count -le 0) {
    throw "Prepared fixed val input list is empty."
}
if (@($preparedInputs | Sort-Object -Unique).Count -ne $preparedInputs.Count) {
    throw "Prepared fixed val input list contains duplicate sources."
}
if ($modeName -eq "formal" -and $preparedInputs.Count -ne $requiredFormalReceipts) {
    throw "Formal CPU A/B requires exactly $requiredFormalReceipts prepared val receipts; got $($preparedInputs.Count)."
}
if ($modeName -eq "pilot") {
    & $pythonExe $scorer prepare `
        --records $Records `
        --output $inputList `
        --split val `
        --limit $Limit
    if ($LASTEXITCODE -ne 0) {
        throw "Could not prepare deterministic five-field pilot input list; exit code $LASTEXITCODE"
    }
}
else {
    Copy-Item -LiteralPath $preparedInputList -Destination $inputList
}
[string[]]$selectedInputs = @(
    Get-Content -LiteralPath $inputList -Encoding UTF8 |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#") }
)
if ($selectedInputs.Count -le 0) {
    throw "Fixed selected input manifest would be empty."
}
if (@($selectedInputs | Sort-Object -Unique).Count -ne $selectedInputs.Count) {
    throw "Fixed selected input manifest contains duplicate sources."
}
if ($modeName -eq "pilot" -and $selectedInputs.Count -ne $Limit) {
    throw "Pilot input preparation must produce exactly $Limit receipts; got $($selectedInputs.Count)."
}
if ($modeName -eq "formal" -and $selectedInputs.Count -ne $requiredFormalReceipts) {
    throw "Formal input manifest must contain exactly $requiredFormalReceipts receipts."
}
$inputManifestSha256 = Get-Sha256 $inputList

Write-Host "mlnet_hybrid_ab_publish_frozen_cli"
& $DotnetExe publish $project `
    -c Release `
    -r win-x64 `
    --self-contained false `
    "-p:OnnxRuntimeFlavor=cpu" `
    -o $cliPublishDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Could not publish the frozen hybrid CPU A/B CLI; exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $cliAssembly -PathType Leaf)) {
    throw "Published ReceiptMlNet.Cli assembly is missing: $cliAssembly"
}
$cliAssemblySha256 = (Get-FileHash -LiteralPath $cliAssembly -Algorithm SHA256).Hash.ToLowerInvariant()
Write-CliAppClosureManifest $cliPublishDirectory $cliClosureManifest
$cliClosureManifestSha256 = Get-Sha256 $cliClosureManifest

$common = @(
    "--detector", $DetectorModel,
    "--device-model", $DeviceModel,
    "--input-list", $inputList,
    "--device", "cpu",
    "--rectification", "max-side-1600",
    "--annotate", "none"
)
if ($DetectorIntraOpThreads -gt 0) {
    $common += "--detector-intra-op-threads"
    $common += "$DetectorIntraOpThreads"
}

Write-Host "mlnet_hybrid_ab_baseline"
$baselineArguments = @(
    "--ocr", "unified", "--ocr-model", $UnifiedModel,
    "--output", $baselineOutput
) + $common
& $DotnetExe $cliAssembly @baselineArguments
if ($LASTEXITCODE -ne 0) {
    throw "v13 CPU baseline failed with exit code $LASTEXITCODE"
}

Write-Host "mlnet_hybrid_ab_candidate"
$hybridArguments = @(
    "--ocr", "hybrid-recipient",
    "--ocr-model", $UnifiedModel,
    "--ocr-bundle", $PaddleDeliveryBundle,
    "--output", $hybridOutput
) + $common
& $DotnetExe $cliAssembly @hybridArguments
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid recipient CPU candidate failed with exit code $LASTEXITCODE"
}
if ((Get-FileHash -LiteralPath $inputList -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputManifestSha256) {
    throw "Fixed val input manifest changed during hybrid CPU A/B execution."
}
if ((Get-FileHash -LiteralPath $cliAssembly -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cliAssemblySha256) {
    throw "Frozen ReceiptMlNet.Cli assembly changed during hybrid CPU A/B execution."
}
if ((Get-Sha256 $cliClosureManifest) -ne $cliClosureManifestSha256) {
    throw "Frozen CLI app closure manifest changed during hybrid CPU A/B execution."
}

$comparisonArguments = @(
    $comparator,
    "--baseline", $baselineOutput,
    "--hybrid", $hybridOutput,
    "--delivery", $PaddleDeliveryBundle,
    "--output", $reportOutput,
    "--mode", $modeName,
    "--input-manifest", $inputList,
    "--input-manifest-sha256", $inputManifestSha256,
    "--cli-assembly", $cliAssembly,
    "--cli-assembly-sha256", $cliAssemblySha256,
    "--cli-app", $cliPublishDirectory,
    "--cli-closure-manifest", $cliClosureManifest,
    "--cli-closure-manifest-sha256", $cliClosureManifestSha256,
    "--max-p95-overhead-ms", "$MaxP95OverheadMs"
)
& $pythonExe @comparisonArguments
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid recipient CPU A/B comparison failed with exit code $LASTEXITCODE"
}

$scoreArguments = @(
    $scorer, "score",
    "--records", $Records,
    "--results", $hybridOutput,
    "--manifest", (Join-Path $hybridOutput "inference_manifest.json"),
    "--input-list", $inputList,
    "--input-list-sha256", $inputManifestSha256,
    "--model", $UnifiedModel,
    "--output", $scoreOutput,
    "--split", "val",
    "--amount-floor", "0.7885",
    "--time-floor", "0.9840",
    "--payment-floor", "0.9325",
    "--recipient-floor", "0.90",
    "--status-floor", "0.90",
    "--limit", "$Limit"
)
& $pythonExe @scoreArguments
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid recipient fixed-val accuracy gate failed with exit code $LASTEXITCODE"
}
if ((Get-Sha256 $inputList) -ne $inputManifestSha256) {
    throw "Fixed val input manifest changed during hash-bound scoring."
}
if ((Get-Sha256 $Records) -ne $recordsSha256) {
    throw "Validation records changed during hash-bound scoring."
}

$summary = Get-Content -LiteralPath (Join-Path $reportOutput "summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$score = Get-Content -LiteralPath (Join-Path $scoreOutput "summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$hybridManifestPath = [IO.Path]::GetFullPath((Join-Path $hybridOutput "inference_manifest.json"))
$scoreManifestPath = [IO.Path]::GetFullPath([string]$score.manifest)
$hybridManifestSha256 = (Get-FileHash -LiteralPath $hybridManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$baselineRuntimeSummaryPath = [IO.Path]::GetFullPath((Join-Path $baselineOutput "inference_summary.json"))
$hybridRuntimeSummaryPath = [IO.Path]::GetFullPath((Join-Path $hybridOutput "inference_summary.json"))
$baselineRuntimeSummarySha256 = Get-Sha256 $baselineRuntimeSummaryPath
$hybridRuntimeSummarySha256 = Get-Sha256 $hybridRuntimeSummaryPath
if ([int]$summary.schema_version -ne 2 `
    -or [string]$summary.input_set.input_manifest.sha256 -ne $inputManifestSha256 `
    -or [string]$summary.cli_build.assembly.sha256 -ne $cliAssemblySha256 `
    -or [string]$summary.cli_build.app_closure.closure_sha256 -ne $cliClosureManifestSha256 `
    -or [string]$summary.run_summaries.baseline.sha256 -ne $baselineRuntimeSummarySha256 `
    -or [string]$summary.run_summaries.hybrid.sha256 -ne $hybridRuntimeSummarySha256 `
    -or -not ([IO.Path]::GetFullPath([string]$summary.run_manifests.hybrid.path)).Equals(
        $hybridManifestPath,
        [StringComparison]::OrdinalIgnoreCase) `
    -or [string]$summary.run_manifests.hybrid.sha256 -ne $hybridManifestSha256 `
    -or -not $scoreManifestPath.Equals($hybridManifestPath, [StringComparison]::OrdinalIgnoreCase) `
    -or [string]$score.manifest_sha256 -ne $hybridManifestSha256 `
    -or [string]$score.records_sha256 -ne $recordsSha256 `
    -or -not ([IO.Path]::GetFullPath([string]$score.input_selection.path)).Equals(
        ([IO.Path]::GetFullPath($inputList)),
        [StringComparison]::OrdinalIgnoreCase) `
    -or [string]$score.input_selection.sha256 -ne $inputManifestSha256 `
    -or $score.input_selection.hash_bound -ne $true `
    -or [int]$score.input_selection.records -ne $selectedInputs.Count `
    -or [string]$score.input_selection.selection_order -ne $expectedSelectionOrder `
    -or [string]$score.evaluation_scope.selection_order -ne $expectedSelectionOrder `
    -or [string]$score.evaluation_scope.input_list_sha256 -ne $inputManifestSha256 `
    -or $score.accuracy_denominators.hash_bound -ne $true `
    -or [string]$score.accuracy_denominators.source -ne "input_selection.field_reference_counts") {
    throw "Hybrid CPU A/B comparison and score are not schema/hash-bound to the frozen inputs, CLI and hybrid manifest."
}
$scoreFields = @(
    "amount",
    "time",
    "payment_method_field",
    "recipient_field",
    "transfer_status"
)
$fixedFloors = @{
    amount = 0.7885
    time = 0.9840
    payment_method_field = 0.9325
    recipient_field = 0.90
    transfer_status = 0.90
}
$candidateCoverageEvidence = $score.PSObject.Properties["all_receipt_candidate_coverage"]
$referenceCountEvidence = $score.input_selection.PSObject.Properties["field_reference_counts"]
if ($null -eq $candidateCoverageEvidence `
    -or $null -eq $candidateCoverageEvidence.Value `
    -or $null -eq $referenceCountEvidence `
    -or $null -eq $referenceCountEvidence.Value `
    -or [string]$candidateCoverageEvidence.Value.scope -ne "all_selected_receipts" `
    -or [int]$score.coverage_contract_version -ne 2 `
    -or [int]$score.coverage.coverage_contract_version -ne 2 `
    -or [string]$score.coverage.candidate_coverage_domain -ne "all_expected_receipts" `
    -or [int]$score.coverage.fully_candidate_covered_receipts -ne $selectedInputs.Count `
    -or [double]$score.coverage.all_field_candidate_coverage -ne 1.0 `
    -or [int]$candidateCoverageEvidence.Value.expected_receipts -ne $selectedInputs.Count `
    -or [int]$candidateCoverageEvidence.Value.complete_receipts -ne $selectedInputs.Count `
    -or [int]$candidateCoverageEvidence.Value.missing_complete_receipts -ne 0 `
    -or [double]$candidateCoverageEvidence.Value.complete_coverage -ne 1.0) {
    throw "Hybrid CPU A/B score lacks complete all-receipt five-field candidate evidence."
}
foreach ($fieldName in $scoreFields) {
    $metricProperty = $score.by_field.PSObject.Properties[$fieldName]
    $referenceCountProperty = $referenceCountEvidence.Value.PSObject.Properties[$fieldName]
    $denominatorProperty = $score.accuracy_denominators.by_field.PSObject.Properties[$fieldName]
    $candidateProperty = $candidateCoverageEvidence.Value.by_field.PSObject.Properties[$fieldName]
    $floorProperty = $score.floors.PSObject.Properties[$fieldName]
    $requiredFloor = [double]$fixedFloors[$fieldName]
    if ($null -eq $metricProperty `
        -or $null -eq $referenceCountProperty `
        -or $null -eq $denominatorProperty `
        -or $null -eq $floorProperty `
        -or [int]$referenceCountProperty.Value -le 0 `
        -or [int]$metricProperty.Value.records -ne [int]$referenceCountProperty.Value `
        -or [int]$denominatorProperty.Value -ne [int]$referenceCountProperty.Value `
        -or [double]$floorProperty.Value -ne $requiredFloor `
        -or [double]::IsNaN([double]$metricProperty.Value.raw_exact_match) `
        -or [double]::IsInfinity([double]$metricProperty.Value.raw_exact_match) `
        -or [double]$metricProperty.Value.raw_exact_match -lt $requiredFloor `
        -or $null -eq $candidateProperty `
        -or [int]$candidateProperty.Value.expected_receipts -ne $selectedInputs.Count `
        -or [int]$candidateProperty.Value.candidate_records -ne $selectedInputs.Count `
        -or [int]$candidateProperty.Value.missing_candidate_records -ne 0 `
        -or [double]$candidateProperty.Value.candidate_coverage -ne 1.0) {
        throw "Hybrid CPU A/B $fieldName accuracy denominator or all-receipt candidate coverage is incomplete."
    }
}
$formalGateProperty = $score.PSObject.Properties["formal_delivery_gate"]
if ($modeName -eq "formal") {
    if ([int]$summary.records -ne $requiredFormalReceipts `
        -or [int]$score.coverage.expected_receipts -ne $requiredFormalReceipts `
        -or [int]$score.evaluation_scope.full_split_expected_receipts -ne $requiredFormalReceipts `
        -or $null -eq $formalGateProperty `
        -or $formalGateProperty.Value -ne $true) {
        throw "Formal CPU A/B must schema-bind exactly $requiredFormalReceipts receipts and declare formal_delivery_gate=true."
    }
    if ($score.accepted -ne $true -or $score.acceptance.passed -ne $true) {
        throw "Formal CPU A/B scorer did not accept the complete validation split."
    }
    $passLabel = "FORMAL PASS"
}
else {
    $pilotThresholdProperty = $score.PSObject.Properties["pilot_thresholds_passed"]
    if ($null -eq $formalGateProperty -or $formalGateProperty.Value -ne $false `
        -or $null -eq $pilotThresholdProperty -or $pilotThresholdProperty.Value -ne $true `
        -or $score.accepted -ne $false `
        -or [int]$score.input_selection.records -ne $Limit `
        -or [int]$score.by_field.amount.records -le 0 `
        -or [int]$score.by_field.time.records -le 0 `
        -or [int]$score.by_field.payment_method_field.records -le 0 `
        -or [int]$score.by_field.recipient_field.records -le 0 `
        -or [int]$score.by_field.transfer_status.records -le 0) {
        throw "Pilot scorer evidence is not explicitly partial/non-formal or its fixed thresholds failed."
    }
    $passLabel = "PILOT PASS"
}
Write-Host ""
Write-Host "HYBRID RECIPIENT CPU A/B: $passLabel" -ForegroundColor Green
Write-Host ("  invariant fields: {0}/{1}" -f $summary.invariant_records, $summary.records)
Write-Host ("  recipient exact: {0}/{1}={2:P2}" -f `
    [int]$score.by_field.recipient_field.raw_exact_matches, `
    [int]$score.by_field.recipient_field.records, `
    [double]$score.by_field.recipient_field.raw_exact_match)
Write-Host ("  baseline CPU p95: {0:N2} ms" -f [double]$summary.cpu.baseline_inference_latency_ms.p95)
Write-Host ("  hybrid CPU p95: {0:N2} ms" -f [double]$summary.cpu.hybrid_inference_latency_ms.p95)
Write-Host ("  CPU p95 overhead: {0:N2} ms" -f [double]$summary.cpu.p95_overhead_ms)
Write-Host "  report=$reportOutput"
Write-Host "  accuracy=$scoreOutput"
