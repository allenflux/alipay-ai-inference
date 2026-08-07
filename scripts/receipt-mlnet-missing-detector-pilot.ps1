[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory,
    [string]$FormalTag = "20260806-165128"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectFile = Join-Path $repoRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"
$detector = Join-Path $repoRoot "artifacts\receipt_lrcnn_v1.onnx"
$deviceModel = Join-Path $repoRoot "artifacts\statusbar_device_v1.onnx"
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $RunDirectory = Join-Path $TeacherRoot "unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
}
$unifiedModel = Join-Path $RunDirectory "best.onnx"
$dataRoot = Split-Path -Parent $TeacherRoot
$formalEvaluation = Join-Path $dataRoot "delivery-validation\mlnet-wide1536-cpu-full-e2e-$FormalTag"
$formalComparisons = Join-Path $formalEvaluation "comparisons.jsonl"

foreach ($required in @($projectFile, $detector, $deviceModel, $unifiedModel, $formalComparisons)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing detector pilot dependency: $required"
    }
}

$cases = @(
    [pscustomobject]@{
        Field = "amount"
        ResultField = "amount"
        Source = "D:\download\TempFakeImages\s3_voucher_GWCZ2072762506148974592_20260703032240.jpg"
        MatchToken = "2072762506148974592"
    },
    [pscustomobject]@{
        Field = "payment_method_field"
        ResultField = "payment_method"
        Source = "D:\download\TempFakeImages\s3_voucher_GWCZ2072894140638695424_20260703120459.jpg"
        MatchToken = "2072894140638695424"
    }
)
foreach ($case in $cases) {
    if (-not (Test-Path -LiteralPath $case.Source -PathType Leaf)) {
        throw "Missing detector pilot source: $($case.Source)"
    }
    $comparison = @(
        Select-String -LiteralPath $formalComparisons -SimpleMatch $case.MatchToken |
            ForEach-Object { $_.Line | ConvertFrom-Json } |
            Where-Object { [string]$_.field -eq [string]$case.Field }
    )
    if ($comparison.Count -ne 1) {
        throw "Expected one formal comparison for $($case.Field), found $($comparison.Count)."
    }
    Add-Member -InputObject $case -NotePropertyName Reference -NotePropertyValue ([string]$comparison[0].reference_text)
}

$tag = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$pilotRoot = Join-Path $dataRoot "delivery-validation\mlnet-missing-detector-pilot-$tag"
if (Test-Path -LiteralPath $pilotRoot) {
    throw "Refusing to overwrite detector pilot: $pilotRoot"
}
$appDirectory = Join-Path $pilotRoot "app"
$inputList = Join-Path $pilotRoot "inputs.txt"
New-Item -ItemType Directory -Path $appDirectory | Out-Null
[IO.File]::WriteAllLines(
    $inputList,
    [string[]]@($cases | ForEach-Object { $_.Source }),
    [Text.UTF8Encoding]::new($false))

Write-Host "receipt_mlnet_missing_detector_publish_cpu"
& dotnet restore $projectFile -r win-x64 "-p:OnnxRuntimeFlavor=cpu"
if ($LASTEXITCODE -ne 0) {
    throw "dotnet restore failed with exit code $LASTEXITCODE"
}
& dotnet publish $projectFile `
    -c Release `
    -r win-x64 `
    --self-contained false `
    "-p:OnnxRuntimeFlavor=cpu" `
    --no-restore `
    -o $appDirectory
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE"
}
$executable = Join-Path $appDirectory "ReceiptMlNet.Cli.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Published detector pilot executable is missing: $executable"
}

function Get-OptionalPropertyValue($Object, [string]$Name) {
    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Invoke-ThresholdPilot([string]$Name, [double]$Threshold) {
    $output = Join-Path $pilotRoot $Name
    $arguments = @(
        "--detector", $detector,
        "--device-model", $deviceModel,
        "--ocr", "unified",
        "--ocr-model", $unifiedModel,
        "--input-list", $inputList,
        "--output", $output,
        "--device", "cpu",
        "--score-threshold", $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--rectification", "max-side-1600",
        "--annotate", "all"
    )
    Write-Host "receipt_mlnet_missing_detector_$Name"
    $consoleOutput = @(& $executable @arguments 2>&1)
    $inferenceExitCode = $LASTEXITCODE
    foreach ($line in $consoleOutput) {
        Write-Host $line
    }
    if ($inferenceExitCode -ne 0) {
        throw "Detector pilot $Name failed with exit code $inferenceExitCode"
    }
    $summaryPath = Join-Path $output "inference_summary.json"
    $manifestPath = Join-Path $output "inference_manifest.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf) `
        -or -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Detector pilot $Name did not write complete runtime evidence."
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$summary.requested_device -ne "cpu" `
        -or [string]$summary.unified_provider -ne "cpu" `
        -or [int]$summary.input -ne $cases.Count `
        -or [int]$summary.written -ne $cases.Count `
        -or [int]$summary.skipped -ne 0 `
        -or [int]$summary.errors -ne 0) {
        throw "Detector pilot $Name runtime summary is not a complete CPU run."
    }
    $manifest = @(Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    if ($manifest.Count -ne $cases.Count) {
        throw "Detector pilot $Name manifest count differs from input count."
    }
    $observations = @()
    foreach ($case in $cases) {
        $manifestRecord = @($manifest | Where-Object {
            [IO.Path]::GetFullPath([string]$_.source).Equals(
                [IO.Path]::GetFullPath([string]$case.Source),
                [StringComparison]::OrdinalIgnoreCase)
        })
        if ($manifestRecord.Count -ne 1 -or [string]$manifestRecord[0].status -ne "written") {
            throw "Detector pilot $Name has no unique written result for $($case.Source)."
        }
        $resultPath = [string]$manifestRecord[0].result
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($null -eq $result.device) {
            throw "Detector pilot $Name omitted the device model result for $($case.Source)."
        }
        $field = $result.fields.PSObject.Properties[[string]$case.ResultField].Value
        $detection = @($result.detections | Where-Object { [string]$_.label -eq [string]$case.Field })
        if ($detection.Count -gt 1) {
            throw "Detector pilot $Name emitted duplicate $($case.Field) detections."
        }
        $candidate = Get-OptionalPropertyValue $field "candidate"
        $ctcCandidate = Get-OptionalPropertyValue $field "ctc_candidate"
        $structuredCandidate = Get-OptionalPropertyValue $field "structured_candidate"
        $observation = [ordered]@{
            threshold = $Threshold
            field = [string]$case.Field
            source = [string]$case.Source
            reference = [string]$case.Reference
            state = [string]$field.state
            candidate = $candidate
            ctc_candidate = $ctcCandidate
            structured_candidate = $structuredCandidate
            exact = $null -ne $candidate -and [string]$candidate -ceq [string]$case.Reference
            detection_score = if ($detection.Count -eq 1) { [double]$detection[0].score } else { $null }
            detection_bbox = if ($detection.Count -eq 1) { @($detection[0].bbox_image) } else { $null }
            result = $resultPath
        }
        $observations += [pscustomobject]$observation
        Write-Host (
            "  threshold={0}; field={1}; state={2}; score={3}; candidate={4}; reference={5}; exact={6}" -f `
                $Threshold, $case.Field, $field.state, $observation.detection_score, `
                $candidate, $case.Reference, $observation.exact)
    }
    return [pscustomobject]@{
        name = $Name
        output = $output
        summary = $summary
        observations = $observations
    }
}

$baseline = Invoke-ThresholdPilot "threshold-050" 0.50
foreach ($observation in $baseline.observations) {
    if ([string]$observation.state -ne "absent" `
        -or $null -ne $observation.candidate `
        -or $null -ne $observation.detection_score) {
        throw "Threshold 0.50 did not reproduce the formal missing detection for $($observation.field)."
    }
}
$zeroThreshold = Invoke-ThresholdPilot "threshold-000" 0.0

$report = [ordered]@{
    schema_version = 1
    kind = "receipt_mlnet_missing_detector_pilot_v1"
    formal_tag = $FormalTag
    formal_evaluation = $formalEvaluation
    pilot_root = $pilotRoot
    runtime = "cpu"
    rectification = "max-side-1600"
    includes_device_model = $true
    baseline = $baseline
    zero_threshold = $zeroThreshold
}
$reportPath = Join-Path $pilotRoot "report.json"
[IO.File]::WriteAllText(
    $reportPath,
    ($report | ConvertTo-Json -Depth 20),
    [Text.UTF8Encoding]::new($false))

Write-Host "receipt_mlnet_missing_detector_pilot_complete"
Write-Host "  root=$pilotRoot"
Write-Host "  report=$reportPath"
Write-Host "  threshold-050=$($baseline.output)"
Write-Host "  threshold-000=$($zeroThreshold.output)"
