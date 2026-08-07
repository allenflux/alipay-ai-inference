[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1"
)

# From the repository root:
# powershell -ExecutionPolicy Bypass -File .\scripts\v13-cpu.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This launcher intentionally exposes no floor, model, limit, or runtime
# switches. The fixed values below are the production delivery contract.
$pilotCount = 100
$formalCount = 10016
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusFloor = 0.90

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
$scorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$packager = Join-Path $PSScriptRoot "receipt-mlnet-unified-package-validate-4090.ps1"
$detectorModel = Join-Path $repoRoot "artifacts\receipt_lrcnn_v1.onnx"
$deviceModel = Join-Path $repoRoot "artifacts\statusbar_device_v1.onnx"

function Require-File([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Require-Directory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing ${Description}: $Path"
    }
}

function Require-NewPath([string]$Path, [string]$Description) {
    if (Test-Path -LiteralPath $Path) {
        throw "$Description already exists; refusing result reuse: $Path"
    }
}

function Read-GuardedJson([string]$Path) {
    Require-File $Path "JSON evidence"
    $normalized = (& $pythonExe $normalizer $Path) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to normalize JSON evidence: $Path"
    }
    try {
        return ($normalized | ConvertFrom-Json)
    }
    catch {
        throw "Unable to parse JSON evidence: $Path. $($_.Exception.Message)"
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-PathWithin([string]$Candidate, [string]$Parent) {
    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd([char[]]@('\', '/'))
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd([char[]]@('\', '/'))
    if ($candidateFull.Equals($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
    return $candidateFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-PassedGpuSummary(
    [object]$Summary,
    [string]$Split,
    [string]$ModelSha256,
    [string]$RecordsSha256
) {
    if ([string]$Summary.model_sha256 -ne $ModelSha256 `
        -or [string]$Summary.records_sha256 -ne $RecordsSha256 `
        -or [string]$Summary.evaluation_split -ne $Split `
        -or @($Summary.providers) -notcontains "CUDAExecutionProvider" `
        -or $Summary.acceptance.requested -ne $true `
        -or $Summary.acceptance.passed -ne $true `
        -or @($Summary.acceptance.failures).Count -ne 0 `
        -or [string]$Summary.status_text_policy.runtime_policy -ne "decode_and_normalize_review_only" `
        -or [string]$Summary.status_text_policy.review_value -ne "review") {
        throw "$Split GPU summary is not accepted v13 evidence."
    }
    foreach ($gate in @(
            @{ Field = "amount"; Metric = "raw_exact_match"; Acceptance = "min_amount_exact_match"; Floor = $amountFloor },
            @{ Field = "time"; Metric = "raw_exact_match"; Acceptance = "min_time_exact_match"; Floor = $timeFloor },
            @{ Field = "payment_method_field"; Metric = "raw_exact_match"; Acceptance = "min_payment_exact_match"; Floor = $paymentFloor },
            @{ Field = "recipient_field"; Metric = "raw_exact_match"; Acceptance = "min_recipient_exact_match"; Floor = $recipientFloor },
            @{ Field = "transfer_status"; Metric = "ctc_raw_exact_match"; Acceptance = "min_status_exact_match"; Floor = $statusFloor }
        )) {
        $fieldProperty = $Summary.by_field.PSObject.Properties[[string]$gate.Field]
        if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
            throw "$Split GPU summary has no $($gate.Field) metric."
        }
        $metricProperty = $fieldProperty.Value.PSObject.Properties[[string]$gate.Metric]
        if ($null -eq $metricProperty `
            -or $null -eq $metricProperty.Value `
            -or [int]$fieldProperty.Value.records -le 0) {
            throw "$Split GPU summary has no $($gate.Field) $($gate.Metric) metric."
        }
        $acceptanceProperty = $Summary.acceptance.PSObject.Properties[[string]$gate.Acceptance]
        if ($null -eq $acceptanceProperty `
            -or $null -eq $acceptanceProperty.Value `
            -or [double]$acceptanceProperty.Value -lt [double]$gate.Floor) {
            throw "$Split GPU summary weakens the fixed $($gate.Field) acceptance floor."
        }
        $metric = [double]$metricProperty.Value
        if ([double]::IsNaN($metric) -or [double]::IsInfinity($metric) -or $metric -lt [double]$gate.Floor) {
            throw "$Split GPU summary does not meet the fixed $($gate.Field) floor."
        }
    }
    $statusMetric = $Summary.by_field.transfer_status
    if ([int]$statusMetric.ctc_records -le 0 `
        -or [int]$statusMetric.ctc_records -ne [int]$statusMetric.records) {
        throw "$Split GPU summary has no visible status OCR records."
    }
    $nonSuccessTruthRecords = `
        [int]$Summary.status_reference_class_counts.pending + `
        [int]$Summary.status_reference_class_counts.failed
    if ($nonSuccessTruthRecords -gt 0) {
        $safetyProperty = $Summary.acceptance.PSObject.Properties["max_non_success_to_success"]
        if ($null -eq $safetyProperty `
            -or $null -eq $safetyProperty.Value `
            -or [int]$safetyProperty.Value -ne 0 `
            -or [int]$statusMetric.non_success_to_success -ne 0) {
            throw "$Split GPU summary does not preserve the non-success safety line."
        }
    }
}

function Read-PassedV13Run([IO.FileInfo]$EvidenceFile) {
    $runDirectory = $EvidenceFile.Directory.FullName
    $evidence = Read-GuardedJson $EvidenceFile.FullName
    if ([string]$evidence.kind -ne "receipt_unified_status_text_v13_guarded_validation_v1" `
        -or [string]$evidence.candidate.kind -ne "receipt_unified_field_reader_v13" `
        -or [int]$evidence.candidate.architecture_version -ne 13 `
        -or [string]$evidence.candidate.status_text_runtime_policy -ne "decode_and_normalize_review_only" `
        -or [string]$evidence.candidate.review_value -ne "review") {
        throw "Evidence is not a passed v13 visible-status OCR candidate."
    }

    $floors = $evidence.acceptance_floors
    if ([double]$floors.amount -lt $amountFloor `
        -or [double]$floors.time -lt $timeFloor `
        -or [double]$floors.payment_method_field -lt $paymentFloor `
        -or [double]$floors.recipient_field -lt $recipientFloor `
        -or [double]$floors.visible_transfer_status_cjk_text -lt $statusFloor) {
        throw "Evidence weakens a fixed delivery floor."
    }

    $valEvidence = @(
        $evidence.evaluations |
            Where-Object { [string]$_.split -eq "val" }
    )
    $testEvidence = @(
        $evidence.evaluations |
            Where-Object { [string]$_.split -eq "test" }
    )
    if ($valEvidence.Count -ne 1 `
        -or $valEvidence[0].evaluated -ne $true `
        -or $valEvidence[0].accepted -ne $true `
        -or $testEvidence.Count -ne 1 `
        -or $testEvidence[0].evaluated -ne $true `
        -or $testEvidence[0].accepted -ne $true `
        -or [double]$valEvidence[0].status_text_exact_match -lt $statusFloor `
        -or [double]$testEvidence[0].status_text_exact_match -lt $statusFloor) {
        throw "Evidence does not contain accepted val and test status OCR results."
    }

    $binding = $evidence.cpu_packaging
    if ([string]$binding.required_runtime_flavor -ne "cpu" `
        -or [string]$binding.required_rectification -ne "max-side-1600" `
        -or $binding.include_device_model -ne $true) {
        throw "Evidence is not bound to the complete production CPU pipeline."
    }

    $records = [IO.Path]::GetFullPath([string]$evidence.manifest.records)
    $unifiedModel = [IO.Path]::GetFullPath([string]$binding.unified_model_path)
    $validationSummary = [IO.Path]::GetFullPath([string]$binding.onnx_validation_summary_path)
    $testSummaryPath = [IO.Path]::GetFullPath([string]$testEvidence[0].summary_path)
    $candidateContract = [IO.Path]::GetFullPath([string]$evidence.candidate.contract)
    $candidateLabels = [IO.Path]::GetFullPath([string]$evidence.candidate.labels)
    foreach ($boundPath in @(
            $records,
            $unifiedModel,
            $validationSummary,
            $testSummaryPath,
            $candidateContract,
            $candidateLabels
        )) {
        if (-not (Test-PathWithin $boundPath $runDirectory)) {
            throw "Evidence path escapes its v13 run: $boundPath"
        }
        Require-File $boundPath "v13 bound artifact"
    }

    $recordsSha256 = Get-Sha256 $records
    $unifiedModelSha256 = Get-Sha256 $unifiedModel
    $candidateContractSha256 = Get-Sha256 $candidateContract
    $candidateLabelsSha256 = Get-Sha256 $candidateLabels
    $validationSummarySha256 = Get-Sha256 $validationSummary
    $testSummarySha256 = Get-Sha256 $testSummaryPath
    if (-not $unifiedModel.Equals(
            [IO.Path]::GetFullPath([string]$evidence.candidate.model),
            [StringComparison]::OrdinalIgnoreCase) `
        -or -not $validationSummary.Equals(
            [IO.Path]::GetFullPath([string]$valEvidence[0].summary_path),
            [StringComparison]::OrdinalIgnoreCase) `
        -or [string]$evidence.manifest.records_sha256 -ne $recordsSha256 `
        -or [string]$evidence.candidate.model_sha256 -ne $unifiedModelSha256 `
        -or [string]$evidence.candidate.contract_sha256 -ne $candidateContractSha256 `
        -or [string]$evidence.candidate.labels_sha256 -ne $candidateLabelsSha256 `
        -or [string]$binding.unified_model_sha256 -ne $unifiedModelSha256 `
        -or [string]$binding.onnx_validation_summary_sha256 -ne $validationSummarySha256 `
        -or [string]$valEvidence[0].summary_sha256 -ne $validationSummarySha256 `
        -or [string]$testEvidence[0].summary_sha256 -ne $testSummarySha256) {
        throw "Evidence hashes or explicit v13 artifact bindings do not match."
    }

    $valSummary = Read-GuardedJson $validationSummary
    $testSummary = Read-GuardedJson $testSummaryPath
    Assert-PassedGpuSummary $valSummary "val" $unifiedModelSha256 $recordsSha256
    Assert-PassedGpuSummary $testSummary "test" $unifiedModelSha256 $recordsSha256
    if ([int]$valEvidence[0].visible_status_records -ne `
            [int]$valSummary.by_field.transfer_status.ctc_records `
        -or [int]$testEvidence[0].visible_status_records -ne `
            [int]$testSummary.by_field.transfer_status.ctc_records `
        -or [double]$valEvidence[0].status_text_exact_match -ne `
            [double]$valSummary.by_field.transfer_status.ctc_raw_exact_match `
        -or [double]$testEvidence[0].status_text_exact_match -ne `
            [double]$testSummary.by_field.transfer_status.ctc_raw_exact_match) {
        throw "Guarded status exact metrics do not match val/test GPU summaries."
    }

    return [pscustomobject]@{
        RunDirectory = $runDirectory
        EvidencePath = $EvidenceFile.FullName
        Evidence = $evidence
        Records = $records
        UnifiedModel = $unifiedModel
        ValidationSummary = $validationSummary
    }
}

function Find-LatestPassedV13Run {
    $evidenceFiles = @(
        Get-ChildItem -LiteralPath $TeacherRoot -Directory -Filter "unified-run-v13-*" |
            ForEach-Object {
                $candidate = Join-Path $_.FullName "v13_status_ocr_validation.json"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    Get-Item -LiteralPath $candidate
                }
            } |
            Sort-Object LastWriteTimeUtc -Descending
    )
    if ($evidenceFiles.Count -eq 0) {
        throw "No v13_status_ocr_validation.json was found under $TeacherRoot"
    }
    $rejections = @()
    foreach ($evidenceFile in $evidenceFiles) {
        try {
            return (Read-PassedV13Run $evidenceFile)
        }
        catch {
            $detail = "$($evidenceFile.Directory.FullName): $($_.Exception.Message)"
            $rejections += $detail
            Write-Warning "Skipping invalid v13 run $detail"
        }
    }
    throw ("No passed v13_status_ocr_validation.json was found under $TeacherRoot. " +
        "Rejected candidates: " + ($rejections -join " | "))
}

function Get-RawExactMetric([object]$Summary, [string]$Field) {
    $property = $Summary.by_field.PSObject.Properties[$Field]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "Formal evaluation has no $Field metric."
    }
    $value = [double]$property.Value.raw_exact_match
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
        throw "Formal evaluation has a non-finite $Field raw exact metric."
    }
    return $value
}

Require-Directory $TeacherRoot "teacher root"
foreach ($required in @(
        @{ Path = $pythonExe; Description = "CUDA environment Python used for evidence tooling" },
        @{ Path = $normalizer; Description = "JSON normalizer" },
        @{ Path = $scorer; Description = "ML.NET scorer" },
        @{ Path = $packager; Description = "ML.NET package validator" },
        @{ Path = $detectorModel; Description = "receipt detector ONNX" },
        @{ Path = $deviceModel; Description = "device classifier ONNX" }
    )) {
    Require-File ([string]$required.Path) ([string]$required.Description)
}

$selected = Find-LatestPassedV13Run
Write-Host "v13_cpu_delivery_start"
Write-Host "  run=$($selected.RunDirectory)"
Write-Host "  evidence=$($selected.EvidencePath)"
Write-Host "  unified-model=$($selected.UnifiedModel)"
Write-Host "  detector-model=$detectorModel"
Write-Host "  device-model=$deviceModel"

$dataRoot = Split-Path -Parent ([IO.Path]::GetFullPath($TeacherRoot))
$validationRoot = Join-Path $dataRoot "delivery-validation"
$deliveryRoot = Join-Path $dataRoot "delivery"
New-Item -ItemType Directory -Path $validationRoot, $deliveryRoot -Force | Out-Null

$tag = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmssfff") + "-$PID"
$inputList = Join-Path $validationRoot "v13-val-$tag.txt"
$pilotOutput = Join-Path $validationRoot "v13-cpu-pilot-100-$tag"
$pilotDelivery = Join-Path $deliveryRoot "ReceiptMlNet-v13-cpu-pilot-100-$tag"
$formalOutput = Join-Path $validationRoot "v13-cpu-formal-$tag"
$formalEvaluation = Join-Path $validationRoot "v13-cpu-formal-e2e-$tag"
$formalDelivery = Join-Path $deliveryRoot "ReceiptMlNet-v13-cpu-production-$tag"
foreach ($fresh in @(
        @{ Path = $inputList; Description = "input list" },
        @{ Path = $pilotOutput; Description = "pilot output" },
        @{ Path = $pilotDelivery; Description = "pilot delivery" },
        @{ Path = $formalOutput; Description = "formal output" },
        @{ Path = $formalEvaluation; Description = "formal evaluation" },
        @{ Path = $formalDelivery; Description = "formal delivery" }
    )) {
    Require-NewPath ([string]$fresh.Path) ([string]$fresh.Description)
}

& $pythonExe $scorer prepare `
    --records $selected.Records `
    --output $inputList `
    --split val
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare the fresh canonical v13 val input list."
}
Require-File $inputList "fresh canonical v13 val input list"
$inputLines = @(
    Get-Content -LiteralPath $inputList -Encoding UTF8 |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#") }
)
if ($inputLines.Count -ne $formalCount) {
    throw "Fresh canonical v13 val input count is $($inputLines.Count), expected $formalCount."
}

$commonArguments = @{
    RunDirectory = $selected.RunDirectory
    UnifiedModelPath = $selected.UnifiedModel
    OnnxValidationSummaryPath = $selected.ValidationSummary
    InputList = $inputList
    RuntimeFlavor = "cpu"
    Rectification = "max-side-1600"
    IncludeDeviceModel = $true
    Annotate = "none"
    AmountFloor = $amountFloor
    TimeFloor = $timeFloor
    PaymentFloor = $paymentFloor
    RecipientFloor = $recipientFloor
    DetectorModel = $detectorModel
    DeviceModel = $deviceModel
}

$pilotArguments = @{}
foreach ($entry in $commonArguments.GetEnumerator()) {
    $pilotArguments[$entry.Key] = $entry.Value
}
$pilotArguments["Output"] = $pilotOutput
$pilotArguments["DeliveryDir"] = $pilotDelivery
$pilotArguments["Limit"] = $pilotCount
Write-Host "v13_cpu_pilot_start"
& $packager @pilotArguments

$pilotSummary = Read-GuardedJson (Join-Path $pilotOutput "inference_summary.json")
$pilotValidation = Read-GuardedJson (Join-Path $pilotDelivery "evidence\package_validation.json")
if ([int]$pilotSummary.input -ne $pilotCount `
    -or [int]$pilotSummary.written -ne $pilotCount `
    -or [int]$pilotSummary.errors -ne 0 `
    -or [string]$pilotSummary.requested_device -ne "cpu" `
    -or [string]$pilotSummary.unified_provider -ne "cpu" `
    -or [string]$pilotValidation.validation_scope -ne "candidate_smoke_only" `
    -or [string]$pilotValidation.unified_artifact_source.binding -ne "explicit_run_contained" `
    -or [int]$pilotValidation.unified_ocr_architecture_version -ne 13 `
    -or $pilotValidation.include_device_model -ne $true `
    -or -not (Test-Path -LiteralPath (Join-Path $pilotDelivery "SHA256SUMS.json") -PathType Leaf)) {
    throw "The 100-image complete three-model CPU pilot did not pass. Formal was not started."
}
Write-Host "v13_cpu_pilot_pass"
Write-Host "  selected=$pilotCount"
Write-Host "  errors=0"
Write-Host "  output=$pilotOutput"
Write-Host "  delivery=$pilotDelivery"

$formalArguments = @{}
foreach ($entry in $commonArguments.GetEnumerator()) {
    $formalArguments[$entry.Key] = $entry.Value
}
$formalArguments["Records"] = $selected.Records
$formalArguments["EndToEndEvaluationDir"] = $formalEvaluation
$formalArguments["Output"] = $formalOutput
$formalArguments["DeliveryDir"] = $formalDelivery
Write-Host "v13_cpu_formal_start"
Write-Host "  selected=$formalCount"
& $packager @formalArguments

$formalSummary = Read-GuardedJson (Join-Path $formalOutput "inference_summary.json")
$scoreSummary = Read-GuardedJson (Join-Path $formalEvaluation "summary.json")
$packageValidation = Read-GuardedJson (Join-Path $formalDelivery "evidence\package_validation.json")
$packageConfig = Read-GuardedJson (Join-Path $formalDelivery "evidence\package_config.json")
$amountExact = Get-RawExactMetric $scoreSummary "amount"
$timeExact = Get-RawExactMetric $scoreSummary "time"
$paymentExact = Get-RawExactMetric $scoreSummary "payment_method_field"
$recipientExact = Get-RawExactMetric $scoreSummary "recipient_field"
$statusExact = Get-RawExactMetric $scoreSummary "transfer_status"
$p50 = [double]$formalSummary.inference_latency_ms.p50
$p95 = [double]$formalSummary.inference_latency_ms.p95
$errors = [int]$formalSummary.errors
$atomicPublished = (
    (Test-Path -LiteralPath $formalDelivery -PathType Container) `
    -and (Test-Path -LiteralPath (Join-Path $formalDelivery "SHA256SUMS.json") -PathType Leaf) `
    -and [string]$packageValidation.validation_scope -eq "full_val_end_to_end_scored_cpu" `
    -and [string]$packageValidation.end_to_end_evaluation.status -eq "accepted" `
    -and [string]$packageValidation.unified_artifact_source.binding -eq "explicit_run_contained" `
    -and [string]$packageValidation.runtime_flavor -eq "cpu" `
    -and [int]$packageValidation.unified_ocr_architecture_version -eq 13 `
    -and $packageValidation.include_device_model -eq $true `
    -and [string]$packageConfig.kind -eq "receipt_mlnet_unified_delivery_package_v1"
)

if ([int]$formalSummary.input -ne $formalCount `
    -or [int]$formalSummary.written -ne $formalCount `
    -or $errors -ne 0 `
    -or [string]$formalSummary.requested_device -ne "cpu" `
    -or [string]$formalSummary.unified_provider -ne "cpu" `
    -or $scoreSummary.accepted -ne $true `
    -or $amountExact -lt $amountFloor `
    -or $timeExact -lt $timeFloor `
    -or $paymentExact -lt $paymentFloor `
    -or $recipientExact -lt $recipientFloor `
    -or $statusExact -lt $statusFloor `
    -or [double]$scoreSummary.by_field.transfer_status.candidate_coverage -ne 1.0 `
    -or [int]$scoreSummary.by_field.transfer_status.non_success_to_success -ne 0 `
    -or [double]::IsNaN($p50) `
    -or [double]::IsInfinity($p50) `
    -or [double]::IsNaN($p95) `
    -or [double]::IsInfinity($p95) `
    -or $p50 -lt 0.0 `
    -or $p95 -lt $p50 `
    -or -not $atomicPublished) {
    throw "Fresh 10016-image v13 CPU formal did not satisfy the fixed delivery contract."
}

Write-Host ""
Write-Host "V13_CPU_DELIVERY_PASS"
Write-Host "  selected=$formalCount"
Write-Host ("  amount_raw_exact={0:P2}" -f $amountExact)
Write-Host ("  time_raw_exact={0:P2}" -f $timeExact)
Write-Host ("  payment_raw_exact={0:P2}" -f $paymentExact)
Write-Host ("  recipient_raw_exact={0:P2}" -f $recipientExact)
Write-Host ("  status_raw_exact={0:P2}" -f $statusExact)
Write-Host "  cpu_p50_ms=$p50"
Write-Host "  cpu_p95_ms=$p95"
Write-Host "  errors=$errors"
Write-Host "  output=$formalOutput"
Write-Host "  evaluation=$formalEvaluation"
Write-Host "  delivery=$formalDelivery"
Write-Host "  atomic_published=$atomicPublished"
