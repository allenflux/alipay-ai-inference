[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This script only attests an already complete v13 run. It never trains or
# evaluates. It performs temporary deterministic re-exports solely to bind the
# recorded checkpoints to the already evaluated artifacts, and never weakens a
# delivery floor.
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusTextFloor = 0.90
$requiredStatusTextPolicy = "decode_and_normalize_review_only"
$requiredReviewValue = "review"
$legacyOutputNames = @(
    "amount_logits",
    "time_logits",
    "payment_logits",
    "status_logits",
    "amount_currency_style_logits",
    "amount_grouped_thousands_logits",
    "amount_sign_position_logits",
    "time_format_logits",
    "time_digit_logits",
    "payment_prefix_logits",
    "payment_bank_prefix_logits",
    "payment_tail_digit_logits",
    "payment_structure_logits",
    "payment_parentheses_logits",
    "recipient_logits"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"

function Require-File([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) `
        -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Description}: $Path"
    }
}

function Require-Directory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) `
        -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing ${Description}: $Path"
    }
}

function Require-NewPath([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or (Test-Path -LiteralPath $Path)) {
        throw "$Description already exists; refusing reuse or overwrite: $Path"
    }
}

function Read-GuardedJson([string]$Path, [string]$Description) {
    Require-File $Path $Description
    try {
        $raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
        $trimmed = $raw.Trim()
        if (-not $trimmed.StartsWith("{", [StringComparison]::Ordinal) `
            -or -not $trimmed.EndsWith("}", [StringComparison]::Ordinal)) {
            throw "$Description must contain one top-level JSON object."
        }
        $document = ConvertFrom-Json -InputObject $trimmed
    }
    catch {
        throw "Unable to parse ${Description}: $Path. $($_.Exception.Message)"
    }
    if ($null -eq $document -or $document -is [Array] `
        -or $document -is [string] -or $document -is [ValueType]) {
        throw "$Description must contain one JSON object: $Path"
    }
    return $document
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-FileHash(
    [string]$Path,
    [string]$ExpectedSha256,
    [string]$Description
) {
    if ((Get-Sha256 $Path) -cne $ExpectedSha256) {
        throw "$Description changed during v13 evidence recovery."
    }
}

function Require-Sha256([object]$Value, [string]$Description) {
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be one lowercase SHA-256 string."
    }
    return [string]$Value
}

function Get-RequiredProperty([object]$Object, [string]$Name, [string]$Description) {
    if ($null -eq $Object) {
        throw "$Description is missing."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Description has no $Name."
    }
    # Preserve empty JSON arrays instead of allowing the PowerShell pipeline to
    # enumerate them into no output (notably acceptance.failures=[]).
    return ,$property.Value
}

function Test-JsonIntegerEqual([object]$Value, [long]$Expected) {
    $isInteger = (
        ($Value -is [sbyte]) -or ($Value -is [byte]) `
        -or ($Value -is [int16]) -or ($Value -is [uint16]) `
        -or ($Value -is [int32]) -or ($Value -is [uint32]) `
        -or ($Value -is [int64]) -or ($Value -is [uint64])
    )
    return $isInteger -and [decimal]$Value -eq [decimal]$Expected
}

function Get-JsonInteger([object]$Value, [string]$Description, [long]$Minimum = 0) {
    $converted = [long]0
    try {
        $converted = [long]$Value
    }
    catch {
        throw "$Description must be a JSON integer."
    }
    if (-not (Test-JsonIntegerEqual $Value $converted) -or $converted -lt $Minimum) {
        throw "$Description must be a JSON integer greater than or equal to $Minimum."
    }
    return $converted
}

function Get-FiniteNumber([object]$Value, [string]$Description) {
    if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string]) {
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
    return $number
}

function Resolve-RecordedFile([object]$Value, [string]$Description) {
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        throw "$Description must be a non-empty recorded path."
    }
    try {
        $path = [IO.Path]::GetFullPath([string]$Value)
    }
    catch {
        throw "$Description is not a valid recorded path: $Value"
    }
    Require-File $path $Description
    return $path
}

function Resolve-RecordedDirectory([object]$Value, [string]$Description) {
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        throw "$Description must be a non-empty recorded path."
    }
    try {
        $path = [IO.Path]::GetFullPath([string]$Value)
    }
    catch {
        throw "$Description is not a valid recorded path: $Value"
    }
    Require-Directory $path $Description
    return $path
}

function Invoke-Python([string[]]$Arguments, [string]$Description) {
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-ExactMetric(
    [object]$Summary,
    [string]$Field,
    [string]$Description,
    [string]$MetricName = "raw_exact_match"
) {
    $fieldProperty = $Summary.by_field.PSObject.Properties[$Field]
    if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
        throw "$Description has no $Field metric."
    }
    $metricProperty = $fieldProperty.Value.PSObject.Properties[$MetricName]
    if ($null -eq $metricProperty -or $null -eq $metricProperty.Value) {
        throw "$Description has no $Field $MetricName metric."
    }
    return Get-FiniteNumber $metricProperty.Value "$Description $Field $MetricName"
}

function Get-StatusOovAudit([object]$Contract, [string]$Split) {
    $root = $Contract.PSObject.Properties["status_text_oov_by_split"]
    $property = if ($null -eq $root -or $null -eq $root.Value) {
        $null
    }
    else {
        $root.Value.PSObject.Properties[$Split]
    }
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "v13 dataset contract has no status-text OOV audit for split '$Split'."
    }
    $audit = $property.Value
    foreach ($name in @("records", "missing_text_records", "oov_records", "oov_characters")) {
        if ($null -eq $audit.PSObject.Properties[$name]) {
            throw "v13 dataset contract has an incomplete status-text audit for split '$Split'."
        }
    }
    $records = Get-JsonInteger $audit.records "$Split status-text records"
    $missing = Get-JsonInteger $audit.missing_text_records "$Split missing status-text records"
    $oovRecords = Get-JsonInteger $audit.oov_records "$Split status-text OOV records"
    $oovCharacters = Get-JsonInteger $audit.oov_characters "$Split status-text OOV characters"
    if ($oovRecords -gt $records) {
        throw "v13 dataset contract has invalid status-text counts for split '$Split'."
    }
    $maximum = if ($records -gt 0) {
        [double]($records - $oovRecords) / [double]$records
    }
    else {
        $null
    }
    return [ordered]@{
        split = $Split
        visible_status_records = $records
        missing_status_text_records = $missing
        total_status_records = $records + $missing
        oov_records = $oovRecords
        oov_characters = $oovCharacters
        checked = ($records -gt 0)
        max_possible_exact_match = $maximum
        calibration_note = if ($records -gt 0) {
            if ($oovRecords -eq 0 -and $oovCharacters -eq 0) {
                "visible status text present; zero train-charset OOV observed"
            }
            else {
                "held-out train-charset OOV is retained as an error; the exact-match floor remains mandatory"
            }
        }
        else {
            "no visible status text in this split; no status OCR claim is made"
        }
    }
}

function Assert-ArtifactHash([object]$Contract, [string]$ModelPath, [string]$Description) {
    $labelsPath = [IO.Path]::ChangeExtension($ModelPath, ".labels.json")
    Require-File $labelsPath "$Description labels"
    $onnxSha = Require-Sha256 $Contract.onnx_sha256 "$Description contract onnx_sha256"
    $labelsSha = Require-Sha256 $Contract.labels_sha256 "$Description contract labels_sha256"
    if ([string]$Contract.onnx_file -ne [IO.Path]::GetFileName($ModelPath) `
        -or $onnxSha -cne (Get-Sha256 $ModelPath) `
        -or [string]$Contract.labels_file -ne [IO.Path]::GetFileName($labelsPath) `
        -or $labelsSha -cne (Get-Sha256 $labelsPath)) {
        throw "$Description ONNX/labels do not match their adjacent contract."
    }
}

function Get-CanonicalJson([object]$Value) {
    return ($Value | ConvertTo-Json -Depth 30 -Compress)
}

function Get-StatusOovProjection([object]$Root, [string]$Description) {
    if ($null -eq $Root) {
        throw "$Description is missing."
    }
    $projection = [ordered]@{}
    foreach ($split in @("train", "val", "test")) {
        $splitProperty = $Root.PSObject.Properties[$split]
        if ($null -eq $splitProperty -or $null -eq $splitProperty.Value) {
            throw "$Description has no '$split' split."
        }
        $audit = $splitProperty.Value
        foreach ($name in @("records", "oov_records", "oov_characters", "examples")) {
            if ($null -eq $audit.PSObject.Properties[$name]) {
                throw "$Description has an incomplete '$split' split."
            }
        }
        $projection[$split] = [ordered]@{
            records = Get-JsonInteger $audit.records "$Description $split records"
            oov_records = Get-JsonInteger $audit.oov_records "$Description $split OOV records"
            oov_characters = Get-JsonInteger $audit.oov_characters "$Description $split OOV characters"
            examples = @($audit.examples)
        }
    }
    return [pscustomobject]$projection
}

function Assert-EvaluationSummary(
    [string]$SummaryPath,
    [string]$Split,
    [object]$StatusAudit,
    [object]$TrainingSummary,
    [string]$CandidateModel,
    [string]$CandidateModelSha256,
    [string]$Records,
    [string]$RecordsSha256,
    [string]$SummarySha256
) {
    if ((Get-Sha256 $SummaryPath) -cne $SummarySha256) {
        throw "$Split GPU summary changed before validation."
    }
    $summary = Read-GuardedJson $SummaryPath "$Split GPU evaluation summary"
    $summaryModel = Resolve-RecordedFile $summary.model "$Split GPU summary model"
    $summaryRecords = Resolve-RecordedFile $summary.records "$Split GPU summary records"
    $acceptance = Get-RequiredProperty $summary "acceptance" "$Split GPU summary"
    $basePassedValue = Get-RequiredProperty $acceptance "passed" "$Split GPU acceptance"
    $requestedValue = Get-RequiredProperty $acceptance "requested" "$Split GPU acceptance"
    $rawFailures = Get-RequiredProperty $acceptance "failures" "$Split GPU acceptance"
    if ($basePassedValue -isnot [bool] `
        -or $requestedValue -isnot [bool] `
        -or $requestedValue -ne $true `
        -or $rawFailures -isnot [Array]) {
        throw "$Split GPU summary has an invalid aggregate acceptance schema."
    }
    $failures = @(
        $rawFailures |
            ForEach-Object {
                if ($_ -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$_)) {
                    throw "$Split GPU summary acceptance failures must be non-empty strings."
                }
                [string]$_
            }
    )
    $nonRecipientFailures = @(
        $failures |
            Where-Object {
                -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal)
            }
    )
    $baseSummaryPassed = [bool]$basePassedValue
    $aggregateStateValid = (
        ($baseSummaryPassed -and $failures.Count -eq 0) `
        -or (-not $baseSummaryPassed `
            -and $failures.Count -gt 0 `
            -and $nonRecipientFailures.Count -eq 0)
    )
    if ([string]$summary.kind -ne "receipt_unified_field_reader_teacher_parity_v1" `
        -or ([string]$summary.warning).IndexOf(
            "not independently verified business truth",
            [StringComparison]::Ordinal) -lt 0 `
        -or [string]$summary.evaluation_split -ne $Split `
        -or -not $summaryModel.Equals($CandidateModel, [StringComparison]::OrdinalIgnoreCase) `
        -or -not $summaryRecords.Equals($Records, [StringComparison]::OrdinalIgnoreCase) `
        -or (Require-Sha256 $summary.model_sha256 "$Split model_sha256") -cne $CandidateModelSha256 `
        -or (Require-Sha256 $summary.records_sha256 "$Split records_sha256") -cne $RecordsSha256 `
        -or $summary.providers -isnot [Array] `
        -or @($summary.providers) -notcontains "CUDAExecutionProvider" `
        -or -not $aggregateStateValid `
        -or $nonRecipientFailures.Count -ne 0 `
        -or [string]$summary.status_text_policy.runtime_policy -ne $requiredStatusTextPolicy `
        -or [string]$summary.status_text_policy.review_value -ne $requiredReviewValue) {
        throw "$Split GPU summary is not accepted amount/time/payment/status CUDA evidence for this model and manifest."
    }

    foreach ($gate in @(
            @{ Field = "amount"; Metric = "raw_exact_match"; Acceptance = "min_amount_exact_match"; Floor = $amountFloor },
            @{ Field = "time"; Metric = "raw_exact_match"; Acceptance = "min_time_exact_match"; Floor = $timeFloor },
            @{ Field = "payment_method_field"; Metric = "raw_exact_match"; Acceptance = "min_payment_exact_match"; Floor = $paymentFloor },
            @{ Field = "transfer_status"; Metric = "ctc_raw_exact_match"; Acceptance = "min_status_exact_match"; Floor = $statusTextFloor }
        )) {
        $metric = Get-ExactMetric $summary ([string]$gate.Field) "$Split evaluation" ([string]$gate.Metric)
        $fieldProperty = $summary.by_field.PSObject.Properties[[string]$gate.Field]
        $fieldRecords = if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
            throw "$Split GPU summary has no $($gate.Field) metric."
        }
        else {
            Get-JsonInteger `
                (Get-RequiredProperty $fieldProperty.Value "records" "$Split $($gate.Field) metric") `
                "$Split $($gate.Field) records" 1
        }
        if ($fieldRecords -le 0) {
            throw "$Split GPU summary has no $($gate.Field) records."
        }
        $requestedProperty = $acceptance.PSObject.Properties[[string]$gate.Acceptance]
        if ($null -eq $requestedProperty -or $null -eq $requestedProperty.Value `
            -or (Get-FiniteNumber $requestedProperty.Value "$Split requested $($gate.Field) floor") -lt [double]$gate.Floor `
            -or $metric -lt [double]$gate.Floor) {
            throw "$Split GPU summary weakens or misses the fixed $($gate.Field) floor."
        }
    }
    $recipientMetric = Get-ExactMetric $summary "recipient_field" "$Split evaluation"
    $recipientField = Get-RequiredProperty $summary.by_field "recipient_field" "$Split GPU summary by_field"
    $recipientRecords = Get-JsonInteger `
        (Get-RequiredProperty $recipientField "records" "$Split recipient metric") `
        "$Split recipient records" 1
    $recipientRequested = $acceptance.PSObject.Properties["min_recipient_exact_match"]
    if ($null -eq $recipientRequested -or $null -eq $recipientRequested.Value `
        -or (Get-FiniteNumber $recipientRequested.Value "$Split requested recipient floor") -lt $recipientFloor) {
        throw "$Split GPU summary weakens the final hybrid recipient floor."
    }
    $recipientDelegated = $recipientMetric -lt $recipientFloor
    if (($recipientDelegated -and ($baseSummaryPassed -or $failures.Count -eq 0)) `
        -or (-not $recipientDelegated -and (-not $baseSummaryPassed -or $failures.Count -ne 0))) {
        throw "$Split GPU summary aggregate state is inconsistent with its recipient metric."
    }

    $statusMetrics = Get-RequiredProperty $summary.by_field "transfer_status" "$Split GPU summary by_field"
    $statusRecords = Get-JsonInteger `
        (Get-RequiredProperty $statusMetrics "records" "$Split status metric") `
        "$Split status records" 1
    $ctcRecords = Get-JsonInteger `
        (Get-RequiredProperty $statusMetrics "ctc_records" "$Split status metric") `
        "$Split status CTC records" 1
    $ctcExactMatches = Get-JsonInteger `
        (Get-RequiredProperty $statusMetrics "ctc_raw_exact_matches" "$Split status metric") `
        "$Split status CTC exact matches"
    $ctcExact = Get-FiniteNumber $statusMetrics.ctc_raw_exact_match "$Split status CTC exact match"
    if ($statusRecords -le 0 -or $ctcRecords -ne $statusRecords `
        -or $ctcRecords -ne [int]$StatusAudit.visible_status_records `
        -or $ctcExactMatches -lt 0 -or $ctcExactMatches -gt $ctcRecords `
        -or [Math]::Abs($ctcExact - ([double]$ctcExactMatches / [double]$ctcRecords)) -gt 0.000000000001) {
        throw "$Split GPU summary has an invalid or incomplete visible-status denominator."
    }

    $classCountsProperty = $TrainingSummary.status_class_counts.PSObject.Properties[$Split]
    if ($null -eq $classCountsProperty -or $null -eq $classCountsProperty.Value) {
        throw "Training summary has no status class counts for split '$Split'."
    }
    $classCounts = $classCountsProperty.Value
    $summaryClassCounts = Get-RequiredProperty `
        $summary "status_reference_class_counts" "$Split GPU summary"
    $validatedClassCounts = [ordered]@{}
    foreach ($className in @("success", "pending", "failed")) {
        $trainingClassCount = Get-JsonInteger `
            (Get-RequiredProperty $classCounts $className "$Split training status counts") `
            "$Split training $className count"
        $summaryClassCount = Get-JsonInteger `
            (Get-RequiredProperty $summaryClassCounts $className "$Split summary status counts") `
            "$Split summary $className count"
        if ($summaryClassCount -ne $trainingClassCount) {
            throw "$Split GPU summary status class counts differ from training evidence."
        }
        $validatedClassCounts[$className] = $trainingClassCount
    }
    if ([long]$validatedClassCounts.success + [long]$validatedClassCounts.pending `
        + [long]$validatedClassCounts.failed -ne $statusRecords) {
        throw "$Split GPU summary status class counts do not cover the status denominator."
    }
    $nonSuccess = [long]$validatedClassCounts.pending + [long]$validatedClassCounts.failed
    $safetyCalibrated = $nonSuccess -gt 0
    $unsafeCount = Get-JsonInteger `
        (Get-RequiredProperty $statusMetrics "non_success_to_success" "$Split status metric") `
        "$Split non-success-to-success count"
    if ($safetyCalibrated) {
        $maxUnsafe = $acceptance.PSObject.Properties["max_non_success_to_success"]
        if ($null -eq $maxUnsafe -or $null -eq $maxUnsafe.Value `
            -or (Get-JsonInteger $maxUnsafe.Value "$Split maximum unsafe count") -ne 0 `
            -or $unsafeCount -ne 0) {
            throw "$Split GPU summary did not preserve zero non-success-to-success errors."
        }
    }

    if ((Get-Sha256 $SummaryPath) -cne $SummarySha256) {
        throw "$Split GPU summary changed during validation."
    }

    return [ordered]@{
        split = $Split
        evaluated = $true
        summary_path = $SummaryPath
        summary_sha256 = $SummarySha256
        visible_status_records = [int]$StatusAudit.visible_status_records
        non_success_truth_records = $nonSuccess
        non_success_safety_calibrated = $safetyCalibrated
        calibration_note = if ($safetyCalibrated) {
            "pending/failed truth exists; max_non_success_to_success=0 requested and passed"
        }
        else {
            "no pending/failed truth in this split; no non-success safety claim is made"
        }
        status_text_exact_match = $ctcExact
        status_non_success_to_success = $unsafeCount
        base_summary_passed = $baseSummaryPassed
        base_summary_failures = $failures
        recipient_exact_match = $recipientMetric
        recipient_delegated_to_hybrid_formal = $recipientDelegated
        core_amount_time_payment_status_accepted = $true
        accepted = $baseSummaryPassed
    }
}

Require-File $pythonExe "CUDA virtual-environment Python used for provenance tooling"
$checkpointAttestor = Join-Path $PSScriptRoot "receipt_ocr_v13_recovery_attest.py"
Require-File $checkpointAttestor "v13 recovery checkpoint attestor"
$checkpointAttestorSha256 = Get-Sha256 $checkpointAttestor
$sidecarAttestor = Join-Path $PSScriptRoot "receipt_ocr_v13_sidecar_attest.py"
Require-File $sidecarAttestor "v13 recovery sidecar attestor"
$sidecarAttestorSha256 = Get-Sha256 $sidecarAttestor
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
Require-Directory $RunDirectory "existing v13 run directory"

$evidencePath = Join-Path $RunDirectory "v13_status_ocr_validation.json"
Require-NewPath $evidencePath "guarded v13 evidence"
$manifestRoot = Join-Path $RunDirectory "manifest-v13"
$records = Join-Path $manifestRoot "unified_fields.jsonl"
$datasetContractPath = Join-Path $manifestRoot "dataset.contract.json"
$trainingRoot = Join-Path $RunDirectory "training-v13"
$trainingSummaryPath = Join-Path $trainingRoot "training_summary.json"
$candidateCheckpoint = Join-Path $trainingRoot "best.pt"
$artifactRoot = Join-Path $RunDirectory "artifacts"
$seedModel = Join-Path $artifactRoot "wide1536-v12-seed.onnx"
$candidateModel = Join-Path $artifactRoot "status-text-v13.onnx"
$seedContractPath = [IO.Path]::ChangeExtension($seedModel, ".contract.json")
$candidateContractPath = [IO.Path]::ChangeExtension($candidateModel, ".contract.json")
$seedLabelsPath = [IO.Path]::ChangeExtension($seedModel, ".labels.json")
$candidateLabelsPath = [IO.Path]::ChangeExtension($candidateModel, ".labels.json")
$valSummaryPath = Join-Path $RunDirectory "onnx-val-gpu\summary.json"
$testSummaryPath = Join-Path $RunDirectory "onnx-test-gpu\summary.json"

foreach ($required in @(
        @{ Path = $records; Description = "v13 unified manifest" },
        @{ Path = $datasetContractPath; Description = "v13 dataset contract" },
        @{ Path = $trainingSummaryPath; Description = "v13 training summary" },
        @{ Path = $candidateCheckpoint; Description = "best v13 checkpoint" },
        @{ Path = $seedModel; Description = "original v12 seed export" },
        @{ Path = $seedContractPath; Description = "original v12 seed export contract" },
        @{ Path = $seedLabelsPath; Description = "original v12 seed export labels" },
        @{ Path = $candidateModel; Description = "v13 candidate ONNX" },
        @{ Path = $candidateContractPath; Description = "v13 candidate contract" },
        @{ Path = $candidateLabelsPath; Description = "v13 candidate labels" },
        @{ Path = $valSummaryPath; Description = "v13 val GPU summary" },
        @{ Path = $testSummaryPath; Description = "v13 test GPU summary" }
    )) {
    Require-File ([string]$required.Path) ([string]$required.Description)
}

# Freeze every path known before its first read. Derived external provenance
# paths are frozen immediately after their guarded parent record is parsed.
$recordsSha256 = Get-Sha256 $records
$datasetContractSha256 = Get-Sha256 $datasetContractPath
$trainingSummarySha256 = Get-Sha256 $trainingSummaryPath
$candidateCheckpointSha256 = Get-Sha256 $candidateCheckpoint
$seedModelSha256 = Get-Sha256 $seedModel
$seedContractSha256 = Get-Sha256 $seedContractPath
$seedLabelsSha256 = Get-Sha256 $seedLabelsPath
$candidateModelSha256 = Get-Sha256 $candidateModel
$candidateContractSha256 = Get-Sha256 $candidateContractPath
$candidateLabelsSha256 = Get-Sha256 $candidateLabelsPath
$valSummarySha256 = Get-Sha256 $valSummaryPath
$testSummarySha256 = Get-Sha256 $testSummaryPath

$datasetContract = Read-GuardedJson $datasetContractPath "v13 dataset contract"
Assert-FileHash $datasetContractPath $datasetContractSha256 "v13 dataset contract"
if ([string]$datasetContract.kind -ne "receipt_unified_field_dataset_v6" `
    -or [string]$datasetContract.architecture -ne "v13" `
    -or [string]$datasetContract.status_text_target -ne "visible_transfer_status_cjk_text" `
    -or [string]$datasetContract.status_text_charset_source -ne "train_only_visible_transfer_status_cjk_text" `
    -or ([string]$datasetContract.warning).IndexOf(
        "teacher labels, not independent business truth",
        [StringComparison]::Ordinal) -lt 0) {
    throw "Existing manifest does not carry the strict teacher-derived v13 visible-status contract."
}
$pseudoLabels = Resolve-RecordedFile $datasetContract.source_records "dataset contract source pseudo-labels"
$datasetRoot = Resolve-RecordedDirectory $datasetContract.dataset_root "dataset contract crop root"
$pseudoLabelsSha256 = Get-Sha256 $pseudoLabels

$trainStatusAudit = Get-StatusOovAudit $datasetContract "train"
$valStatusAudit = Get-StatusOovAudit $datasetContract "val"
$testStatusAudit = Get-StatusOovAudit $datasetContract "test"
foreach ($audit in @($trainStatusAudit, $valStatusAudit, $testStatusAudit)) {
    if ([int]$audit.visible_status_records -le 0 `
        -or [int]$audit.missing_status_text_records -ne 0) {
        throw "v13 status-text evidence requires a complete visible-status denominator in every split."
    }
}
if ([int]$trainStatusAudit.oov_records -ne 0 `
    -or [int]$trainStatusAudit.oov_characters -ne 0) {
    throw "v13 train status text is not closed over its frozen charset."
}
foreach ($audit in @($valStatusAudit, $testStatusAudit)) {
    if ((Get-FiniteNumber $audit.max_possible_exact_match "$($audit.split) OOV ceiling") -lt $statusTextFloor) {
        throw "v13 $($audit.split) OOV ceiling is below the fixed visible-status floor."
    }
}

# Prove that the currently recorded source manifest still deterministically
# produces the exact existing v13 manifest and contract. This is data
# attestation only; it performs no teacher inference and no model evaluation.
$attestationRoot = Join-Path $RunDirectory (".v13-evidence-recovery-" + [Guid]::NewGuid().ToString("N"))
Require-NewPath $attestationRoot "temporary manifest attestation directory"
try {
    Invoke-Python @(
        "-m", "transfer_receipt_ai.ocr_unified_dataset",
        "--records", $pseudoLabels,
        "--output", $attestationRoot,
        "--architecture", "v13"
    ) "deterministic v13 manifest attestation"
    $attestedRecords = Join-Path $attestationRoot "unified_fields.jsonl"
    $attestedContract = Join-Path $attestationRoot "dataset.contract.json"
    Require-File $attestedRecords "attested v13 records"
    Require-File $attestedContract "attested v13 dataset contract"
    Assert-FileHash $pseudoLabels $pseudoLabelsSha256 "source pseudo-labels"
    Assert-FileHash $records $recordsSha256 "existing v13 manifest"
    Assert-FileHash $datasetContractPath $datasetContractSha256 "v13 dataset contract"
    if ((Get-Sha256 $attestedRecords) -cne $recordsSha256 `
        -or (Get-Sha256 $attestedContract) -cne $datasetContractSha256) {
        throw "Recorded pseudo-labels no longer reproduce the existing v13 manifest and contract."
    }
}
finally {
    if (Test-Path -LiteralPath $attestationRoot -PathType Container) {
        Remove-Item -LiteralPath $attestationRoot -Recurse -Force
    }
}

$trainingSummary = Read-GuardedJson $trainingSummaryPath "v13 training summary"
Assert-FileHash $trainingSummaryPath $trainingSummarySha256 "v13 training summary"
$initialization = $trainingSummary.initialization
$fineTune = $trainingSummary.fine_tune_policy
$runtime = $trainingSummary.training_runtime
$validationEvery = [int]$fineTune.validation_every
if ([string]$trainingSummary.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$trainingSummary.config.architecture_version -ne 13 `
    -or [int]$trainingSummary.config.recipient_input_width -ne 1536 `
    -or [string]$initialization.mode -ne "parameter_only_v12_to_v13_status_text_expansion" `
    -or [string]$initialization.source_kind -ne "receipt_unified_field_reader_v12" `
    -or [int]$initialization.source_config.architecture_version -ne 12 `
    -or [int]$initialization.source_config.recipient_input_width -ne 1536 `
    -or [int]$initialization.frozen_legacy_output_count -ne 15 `
    -or [int]$initialization.copied_legacy_tensor_count -le 0 `
    -or [int]$initialization.new_status_text_tensor_count -le 0 `
    -or [string]$initialization.financial_label_policy.mode -ne "checkpoint_legacy_label_maps_status_text_only_v1" `
    -or [string]$fineTune.mode -ne "status_text_only_v13" `
    -or [string]$fineTune.trainable_parameter_prefix -ne "status_text_" `
    -or [int]$fineTune.frozen_legacy_output_count -ne 15 `
    -or [string]$fineTune.full_validation_schedule -ne "epoch_1_every_n_and_final_epoch" `
    -or $validationEvery -le 0 `
    -or $runtime.uses_cuda -ne $true `
    -or $runtime.status_text_only_training -ne $true `
    -or [string]$runtime.full_validation_schedule -ne "epoch_1_every_n_and_final_epoch" `
    -or [int]$runtime.validation_every -ne $validationEvery `
    -or -not ([string]$runtime.device).StartsWith("cuda", [StringComparison]::OrdinalIgnoreCase) `
    -or [string]::IsNullOrWhiteSpace([string]$runtime.cuda_device_name) `
    -or [string]$trainingSummary.status_text_runtime_policy -ne $requiredStatusTextPolicy) {
    throw "Training summary does not prove the guarded CUDA-only v12-to-v13 status-head fine-tune."
}

$seedCheckpoint = Resolve-RecordedFile $initialization.checkpoint_path "recorded v12 seed checkpoint"
$seedCheckpointSha256 = Get-Sha256 $seedCheckpoint
$seedCheckpointExpectedSha256 = Require-Sha256 `
    $initialization.checkpoint_sha256 `
    "training initialization checkpoint_sha256"
if ([IO.Path]::GetExtension($seedCheckpoint) -ne ".pt" `
    -or $seedCheckpointSha256 -cne $seedCheckpointExpectedSha256) {
    throw "Recorded seed checkpoint no longer matches the training initialization provenance."
}

$protectedMinima = $trainingSummary.checkpoint_selection_policy.protected_minimum_candidate_exact
if ((Get-FiniteNumber $protectedMinima.amount "protected amount floor") -ne $amountFloor `
    -or (Get-FiniteNumber $protectedMinima.time "protected time floor") -ne $timeFloor `
    -or (Get-FiniteNumber $protectedMinima.payment_method_field "protected payment floor") -ne $paymentFloor) {
    throw "Training checkpoint policy changed a protected field floor."
}
$bestEpoch = [int]$trainingSummary.best_checkpoint_epoch
$bestRecord = @($trainingSummary.records | Where-Object { [int]$_.epoch -eq $bestEpoch })
if ($bestEpoch -le 0 -or $bestRecord.Count -ne 1 `
    -or $bestRecord[0].checkpoint_selection_eligible -ne $true) {
    throw "Training did not select exactly one floor-eligible best checkpoint."
}
$expectedSelectionMetric = "status_safety_then_transfer_status_raw_ctc_exact_then_recipient_exact_after_protected_candidate_exact_floors"
$expectedStatusSafetyScore = 0.0 - [double]$bestRecord[0].val_status_non_success_to_success
if ($trainingSummary.checkpoint_selection_policy.status_text_ctc_priority -ne $true `
    -or [string]$trainingSummary.checkpoint_selection_policy.selection_metric -ne $expectedSelectionMetric `
    -or $null -eq $bestRecord[0].val_ctc_by_field.transfer_status `
    -or [double]$bestRecord[0].checkpoint_selection_score[0] -ne $expectedStatusSafetyScore `
    -or [double]$bestRecord[0].checkpoint_selection_score[1] -ne `
        [double]$bestRecord[0].val_ctc_by_field.transfer_status.exact_match) {
    throw "Best checkpoint selection did not prioritize status safety and raw status CTC exact match."
}
$bestMetrics = $bestRecord[0].val_candidate_text_by_field
if ([double]$bestMetrics.amount.exact_match -lt $amountFloor `
    -or [double]$bestMetrics.time.exact_match -lt $timeFloor `
    -or [double]$bestMetrics.payment_method_field.exact_match -lt $paymentFloor `
    -or [double]$bestRecord[0].val_ctc_by_field.transfer_status.exact_match -lt $statusTextFloor) {
    throw "Best checkpoint does not retain amount/time/payment and visible-status CTC floors."
}

$seedContract = Read-GuardedJson $seedContractPath "v12 seed ONNX contract"
$candidateContract = Read-GuardedJson $candidateContractPath "v13 candidate ONNX contract"
$candidateLabels = Read-GuardedJson $candidateLabelsPath "v13 candidate labels"
Assert-ArtifactHash $seedContract $seedModel "v12 seed"
Assert-ArtifactHash $candidateContract $candidateModel "v13 candidate"
foreach ($artifactBinding in @(
        @{ Path = $seedModel; Sha256 = $seedModelSha256; Description = "seed ONNX" },
        @{ Path = $seedContractPath; Sha256 = $seedContractSha256; Description = "seed ONNX contract" },
        @{ Path = $seedLabelsPath; Sha256 = $seedLabelsSha256; Description = "seed ONNX labels" },
        @{ Path = $candidateModel; Sha256 = $candidateModelSha256; Description = "candidate ONNX" },
        @{ Path = $candidateContractPath; Sha256 = $candidateContractSha256; Description = "candidate contract" },
        @{ Path = $candidateLabelsPath; Sha256 = $candidateLabelsSha256; Description = "candidate labels" }
    )) {
    Assert-FileHash `
        ([string]$artifactBinding.Path) `
        ([string]$artifactBinding.Sha256) `
        ([string]$artifactBinding.Description)
}
if ([string]$seedContract.kind -ne "receipt_unified_field_reader_v12" `
    -or [int]$seedContract.model.architecture_version -ne 12 `
    -or [int]$seedContract.model.recipient_input_width -ne 1536 `
    -or [string]$candidateContract.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$candidateContract.model.architecture_version -ne 13 `
    -or [int]$candidateContract.model.recipient_input_width -ne 1536) {
    throw "Existing artifacts are not the expected wide1536 v12/v13 pair."
}
if ((Get-CanonicalJson $candidateContract.training_initialization) -cne `
        (Get-CanonicalJson $initialization) `
    -or (Get-CanonicalJson $candidateLabels.initialization) -cne `
        (Get-CanonicalJson $initialization) `
    -or (Get-CanonicalJson $candidateContract.checkpoint_selection_policy) -cne `
        (Get-CanonicalJson $trainingSummary.checkpoint_selection_policy)) {
    throw "Candidate sidecars do not preserve the training and checkpoint-selection provenance."
}
$trainingStatusOov = Get-StatusOovProjection `
    $trainingSummary.status_text_oov_by_split "training status-text OOV audit"
$contractStatusOov = Get-StatusOovProjection `
    $candidateContract.status_text_oov_by_split "candidate contract status-text OOV audit"
$labelsStatusOov = Get-StatusOovProjection `
    $candidateLabels.status_text_oov_by_split "candidate labels status-text OOV audit"
$datasetStatusOov = Get-StatusOovProjection `
    $datasetContract.status_text_oov_by_split "dataset contract status-text OOV audit"
if ((Get-CanonicalJson $contractStatusOov) -cne (Get-CanonicalJson $trainingStatusOov) `
    -or (Get-CanonicalJson $labelsStatusOov) -cne (Get-CanonicalJson $trainingStatusOov) `
    -or (Get-CanonicalJson $datasetStatusOov) -cne (Get-CanonicalJson $trainingStatusOov) `
    -or [string]$candidateContract.status_text_target -ne [string]$datasetContract.status_text_target `
    -or [string]$candidateLabels.status_text_target -ne [string]$datasetContract.status_text_target `
    -or [string]$trainingSummary.status_text_target -ne [string]$datasetContract.status_text_target `
    -or [string]$candidateContract.status_text_charset_source -ne [string]$datasetContract.status_text_charset_source `
    -or [string]$candidateLabels.status_text_charset_source -ne [string]$datasetContract.status_text_charset_source `
    -or [string]$trainingSummary.status_text_charset_source -ne [string]$datasetContract.status_text_charset_source `
    -or (Require-Sha256 $candidateContract.status_text_charset_sha256 "candidate status charset SHA-256") -cne `
        (Require-Sha256 $datasetContract.status_text_charset_sha256 "dataset status charset SHA-256") `
    -or (Require-Sha256 $candidateLabels.status_text_charset_sha256 "candidate labels status charset SHA-256") -cne `
        (Require-Sha256 $datasetContract.status_text_charset_sha256 "dataset status charset SHA-256") `
    -or (Require-Sha256 $trainingSummary.status_text_charset_sha256 "training status charset SHA-256") -cne `
        (Require-Sha256 $datasetContract.status_text_charset_sha256 "dataset status charset SHA-256")) {
    throw "Candidate, training, and dataset status-text provenance do not match."
}

$seedOutputs = @($seedContract.outputs.PSObject.Properties)
$candidateOutputs = @($candidateContract.outputs.PSObject.Properties)
if ($seedOutputs.Count -ne 15 -or $candidateOutputs.Count -ne 16) {
    throw "Existing v12/v13 output counts are not 15/16."
}
foreach ($outputName in $legacyOutputNames) {
    $seedOutput = $seedContract.outputs.PSObject.Properties[$outputName]
    $candidateOutput = $candidateContract.outputs.PSObject.Properties[$outputName]
    if ($null -eq $seedOutput -or $null -eq $candidateOutput `
        -or (Get-CanonicalJson $seedOutput.Value.shape) -cne `
            (Get-CanonicalJson $candidateOutput.Value.shape)) {
        throw "Legacy output ABI parity failed for $outputName."
    }
}
$statusTextOutput = $candidateContract.outputs.PSObject.Properties["status_text_logits"]
if ($null -ne $seedContract.outputs.PSObject.Properties["status_text_logits"] `
    -or $null -eq $statusTextOutput `
    -or [string]$candidateContract.status_text_runtime_policy -ne $requiredStatusTextPolicy `
    -or [string]$statusTextOutput.Value.runtime_policy -ne $requiredStatusTextPolicy `
    -or [string]$statusTextOutput.Value.review_value -ne $requiredReviewValue `
    -or [string]$candidateContract.status_head_policy.runtime_policy -ne "review_only") {
    throw "v13 status text is not an additive review-only output."
}

# Restore the causal proof that the original one-process producer obtained by
# training and immediately exporting. The helper directly compares every
# frozen seed/best tensor with torch.equal. Fresh ONNX graphs must then be
# byte-identical. Sidecars must be identical except for the strictly attested
# 17bc8af legacy-default metadata additions and their derived labels hash.
$artifactAttestationRoot = Join-Path `
    $RunDirectory `
    (".v13-artifact-attestation-" + [Guid]::NewGuid().ToString("N"))
Require-NewPath $artifactAttestationRoot "temporary artifact attestation directory"
$checkpointAttestation = $null
$sidecarAttestation = $null
try {
    New-Item -ItemType Directory -Path $artifactAttestationRoot | Out-Null
    $checkpointAttestationPath = Join-Path $artifactAttestationRoot "checkpoint-attestation.json"
    Invoke-Python @(
        $checkpointAttestor,
        "--seed-checkpoint", $seedCheckpoint,
        "--candidate-checkpoint", $candidateCheckpoint,
        "--training-summary", $trainingSummaryPath,
        "--output", $checkpointAttestationPath
    ) "direct v12/v13 checkpoint tensor attestation"
    $checkpointAttestation = Read-GuardedJson `
        $checkpointAttestationPath `
        "direct v12/v13 checkpoint tensor attestation"
    if ([string]$checkpointAttestation.kind -ne `
            "receipt_unified_v13_recovery_checkpoint_attestation_v1" `
        -or $checkpointAttestation.passed -isnot [bool] `
        -or $checkpointAttestation.passed -ne $true `
        -or (Require-Sha256 `
                $checkpointAttestation.seed_checkpoint_sha256 `
                "checkpoint attestation seed SHA-256") -cne $seedCheckpointSha256 `
        -or (Require-Sha256 `
                $checkpointAttestation.candidate_checkpoint_sha256 `
                "checkpoint attestation candidate SHA-256") -cne $candidateCheckpointSha256 `
        -or (Require-Sha256 `
                $checkpointAttestation.training_summary_sha256 `
                "checkpoint attestation training SHA-256") -cne $trainingSummarySha256 `
        -or (Get-JsonInteger `
                $checkpointAttestation.candidate_epoch `
                "checkpoint attestation candidate epoch" 1) -ne $bestEpoch `
        -or (Get-JsonInteger `
                $checkpointAttestation.legacy_tensor_count `
                "checkpoint attestation legacy tensor count" 1) -ne `
            [long]$initialization.copied_legacy_tensor_count `
        -or (Get-JsonInteger `
                $checkpointAttestation.new_status_text_tensor_count `
                "checkpoint attestation new status tensor count" 1) -ne `
            [long]$initialization.new_status_text_tensor_count `
        -or [string]$checkpointAttestation.comparison -ne `
            "torch.equal_cpu_for_every_non_status_text_tensor") {
        throw "Direct v12/v13 checkpoint tensor attestation did not pass."
    }

    $attestedArtifactRoot = Join-Path $artifactAttestationRoot "artifacts"
    New-Item -ItemType Directory -Path $attestedArtifactRoot | Out-Null
    $attestedSeedModel = Join-Path $attestedArtifactRoot "wide1536-v12-seed.onnx"
    $attestedCandidateModel = Join-Path $attestedArtifactRoot "status-text-v13.onnx"
    Invoke-Python @(
        "-m", "transfer_receipt_ai.ocr_unified", "export",
        "--checkpoint", $seedCheckpoint,
        "--output", $attestedSeedModel
    ) "deterministic v12 seed attestation re-export"
    Invoke-Python @(
        "-m", "transfer_receipt_ai.ocr_unified", "export",
        "--checkpoint", $candidateCheckpoint,
        "--output", $attestedCandidateModel
    ) "deterministic v13 candidate attestation re-export"

    # The original run predates 17bc8af, which added two defaulted legacy
    # recipient config fields plus one repeated legacy-backbone metadata key.
    # ONNX bytes must still match exactly. Sidecars may differ only at those
    # fixed paths/values and at the contract hash derived from the changed
    # labels bytes. Every other path or value is rejected by the helper.
    $sidecarAttestationPath = Join-Path $artifactAttestationRoot "sidecar-attestation.json"
    Assert-FileHash $sidecarAttestor $sidecarAttestorSha256 "sidecar attestor"
    Invoke-Python @(
        $sidecarAttestor,
        "--existing-seed-model", $seedModel,
        "--fresh-seed-model", $attestedSeedModel,
        "--existing-candidate-model", $candidateModel,
        "--fresh-candidate-model", $attestedCandidateModel,
        "--output", $sidecarAttestationPath
    ) "strict v12/v13 sidecar compatibility attestation"
    $sidecarAttestation = Read-GuardedJson `
        $sidecarAttestationPath `
        "strict v12/v13 sidecar compatibility attestation"
    if ([string]$sidecarAttestation.kind -ne `
            "receipt_unified_v13_recovery_sidecar_attestation_v1" `
        -or $sidecarAttestation.passed -isnot [bool] `
        -or $sidecarAttestation.passed -ne $true `
        -or [string]$sidecarAttestation.policy -ne `
            "legacy_recipient_sidecar_defaults_added_by_17bc8af_v1" `
        -or [string]$sidecarAttestation.compatibility_commit -ne `
            "17bc8afca6f0a1a95b0f3a45d603d016638fbbdb" `
        -or $sidecarAttestation.all_onnx_byte_identical -isnot [bool] `
        -or $sidecarAttestation.all_onnx_byte_identical -ne $true `
        -or $sidecarAttestation.all_sidecars_semantically_equivalent -isnot [bool] `
        -or $sidecarAttestation.all_sidecars_semantically_equivalent -ne $true `
        -or $sidecarAttestation.all_sidecars_byte_identical -isnot [bool] `
        -or $sidecarAttestation.comparisons.seed.passed -isnot [bool] `
        -or $sidecarAttestation.comparisons.seed.passed -ne $true `
        -or $sidecarAttestation.comparisons.seed.onnx_byte_identical -ne $true `
        -or $sidecarAttestation.comparisons.candidate.passed -isnot [bool] `
        -or $sidecarAttestation.comparisons.candidate.passed -ne $true `
        -or $sidecarAttestation.comparisons.candidate.onnx_byte_identical -ne $true) {
        throw "Strict v12/v13 sidecar compatibility attestation did not pass."
    }

    # Retain an independent PowerShell byte-hash assertion for the two ONNX
    # graphs. JSON sidecars are validated semantically under the fixed helper
    # allowlist above; they are deliberately not claimed byte-identical.
    foreach ($reexportBinding in @(
            @{ Actual = $attestedSeedModel; Expected = $seedModelSha256; Description = "seed ONNX" },
            @{ Actual = $attestedCandidateModel; Expected = $candidateModelSha256; Description = "candidate ONNX" }
        )) {
        Require-File ([string]$reexportBinding.Actual) "re-exported $($reexportBinding.Description)"
        if ((Get-Sha256 ([string]$reexportBinding.Actual)) -cne `
                [string]$reexportBinding.Expected) {
            throw "Deterministic re-export differs from existing $($reexportBinding.Description)."
        }
    }
    foreach ($sourceBinding in @(
            @{ Path = $seedCheckpoint; Sha256 = $seedCheckpointSha256; Description = "seed checkpoint" },
            @{ Path = $candidateCheckpoint; Sha256 = $candidateCheckpointSha256; Description = "candidate checkpoint" },
            @{ Path = $trainingSummaryPath; Sha256 = $trainingSummarySha256; Description = "training summary" },
            @{ Path = $checkpointAttestor; Sha256 = $checkpointAttestorSha256; Description = "checkpoint attestor" },
            @{ Path = $sidecarAttestor; Sha256 = $sidecarAttestorSha256; Description = "sidecar attestor" }
        )) {
        Assert-FileHash `
            ([string]$sourceBinding.Path) `
            ([string]$sourceBinding.Sha256) `
            ([string]$sourceBinding.Description)
    }
}
finally {
    if (Test-Path -LiteralPath $artifactAttestationRoot -PathType Container) {
        Remove-Item -LiteralPath $artifactAttestationRoot -Recurse -Force
    }
}
if ($null -eq $checkpointAttestation -or $null -eq $sidecarAttestation) {
    throw "Checkpoint and artifact attestation produced no evidence."
}

$valEvidence = Assert-EvaluationSummary `
    $valSummaryPath "val" $valStatusAudit $trainingSummary `
    $candidateModel $candidateModelSha256 $records $recordsSha256 $valSummarySha256
$testEvidence = Assert-EvaluationSummary `
    $testSummaryPath "test" $testStatusAudit $trainingSummary `
    $candidateModel $candidateModelSha256 $records $recordsSha256 $testSummarySha256

$validationEvidence = [ordered]@{
    schema_version = 1
    kind = "receipt_unified_status_text_v13_guarded_validation_v1"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    pseudo_labels = $pseudoLabels
    pseudo_labels_sha256 = $pseudoLabelsSha256
    dataset_root = $datasetRoot
    manifest = [ordered]@{
        records = $records
        records_sha256 = $recordsSha256
        contract = $datasetContractPath
        contract_sha256 = $datasetContractSha256
        status_text_oov = @($trainStatusAudit, $valStatusAudit, $testStatusAudit)
    }
    seed = [ordered]@{
        checkpoint = $seedCheckpoint
        checkpoint_sha256 = $seedCheckpointSha256
        exported_model = $seedModel
        exported_model_sha256 = $seedModelSha256
        kind = [string]$seedContract.kind
        architecture_version = [int]$seedContract.model.architecture_version
        recipient_input_width = [int]$seedContract.model.recipient_input_width
    }
    candidate = [ordered]@{
        checkpoint = $candidateCheckpoint
        checkpoint_sha256 = $candidateCheckpointSha256
        model = $candidateModel
        model_sha256 = $candidateModelSha256
        contract = $candidateContractPath
        contract_sha256 = $candidateContractSha256
        labels = $candidateLabelsPath
        labels_sha256 = $candidateLabelsSha256
        kind = [string]$candidateContract.kind
        architecture_version = [int]$candidateContract.model.architecture_version
        status_text_runtime_policy = $requiredStatusTextPolicy
        review_value = $requiredReviewValue
    }
    training = [ordered]@{
        summary = $trainingSummaryPath
        summary_sha256 = $trainingSummarySha256
        device = [string]$runtime.device
        cuda_device_name = [string]$runtime.cuda_device_name
        mode = [string]$fineTune.mode
        initialization = [string]$initialization.mode
    }
    checkpoint_attestation = [ordered]@{
        passed = $true
        candidate_epoch = [long]$checkpointAttestation.candidate_epoch
        legacy_tensor_count = [long]$checkpointAttestation.legacy_tensor_count
        new_status_text_tensor_count = [long]$checkpointAttestation.new_status_text_tensor_count
        comparison = [string]$checkpointAttestation.comparison
        deterministic_reexport_onnx_byte_identical = $true
        deterministic_reexport_sidecars_byte_identical = `
            [bool]$sidecarAttestation.all_sidecars_byte_identical
        deterministic_reexport_sidecars_semantically_equivalent = $true
        sidecar_compatibility_policy = [string]$sidecarAttestation.policy
        sidecar_compatibility_commit = [string]$sidecarAttestation.compatibility_commit
        sidecar_allowed_fresh_only_defaults = $sidecarAttestation.allowed_fresh_only_defaults
        sidecar_allowed_derived_differences = $sidecarAttestation.allowed_derived_differences
        sidecar_comparisons = $sidecarAttestation.comparisons
        attestor = $checkpointAttestor
        attestor_sha256 = $checkpointAttestorSha256
        sidecar_attestor = $sidecarAttestor
        sidecar_attestor_sha256 = $sidecarAttestorSha256
    }
    legacy_output_parity = [ordered]@{
        passed = $true
        frozen_output_count = 15
        output_names = $legacyOutputNames
        proof = "direct seed/best torch.equal legacy-tensor audit + byte-identical ONNX re-export + strict allowlisted sidecar semantic attestation + exported name/shape ABI comparison"
    }
    acceptance_floors = [ordered]@{
        amount = $amountFloor
        time = $timeFloor
        payment_method_field = $paymentFloor
        recipient_field = $recipientFloor
        visible_transfer_status_cjk_text = $statusTextFloor
    }
    evaluations = @($valEvidence, $testEvidence)
    recipient_delivery_policy = [ordered]@{
        final_floor = $recipientFloor
        base_v13_below_floor_may_only_fail_recipient = $true
        delegated_runtime = "hybrid_ppocr"
        final_gate = "10016-receipt CPU formal"
        final_gate_required = $true
        delegated_in_any_held_out_split = (
            $valEvidence.recipient_delegated_to_hybrid_formal -eq $true `
            -or $testEvidence.recipient_delegated_to_hybrid_formal -eq $true
        )
    }
    cpu_packaging = [ordered]@{
        performed = $false
        next_script = "scripts/receipt-mlnet-unified-package-validate-4090.ps1"
        run_directory = $RunDirectory
        unified_model_path = $candidateModel
        unified_model_sha256 = $candidateModelSha256
        onnx_validation_summary_path = $valSummaryPath
        onnx_validation_summary_sha256 = $valSummarySha256
        required_runtime_flavor = "cpu"
        required_rectification = "max-side-1600"
        include_device_model = $true
        recipient_delegated_to_hybrid_formal = (
            $valEvidence.recipient_delegated_to_hybrid_formal -eq $true `
            -or $testEvidence.recipient_delegated_to_hybrid_formal -eq $true
        )
    }
}

# Close the time-of-check/time-of-use window before publishing the attestation.
$sourceBindings = @(
        @{ Path = $pseudoLabels; Sha256 = $pseudoLabelsSha256; Description = "pseudo-labels" },
        @{ Path = $records; Sha256 = $recordsSha256; Description = "manifest records" },
        @{ Path = $datasetContractPath; Sha256 = $datasetContractSha256; Description = "dataset contract" },
        @{ Path = $seedCheckpoint; Sha256 = $seedCheckpointSha256; Description = "seed checkpoint" },
        @{ Path = $seedModel; Sha256 = $seedModelSha256; Description = "seed ONNX" },
        @{ Path = $seedContractPath; Sha256 = $seedContractSha256; Description = "seed ONNX contract" },
        @{ Path = $seedLabelsPath; Sha256 = $seedLabelsSha256; Description = "seed ONNX labels" },
        @{ Path = $candidateCheckpoint; Sha256 = $candidateCheckpointSha256; Description = "candidate checkpoint" },
        @{ Path = $candidateModel; Sha256 = $candidateModelSha256; Description = "candidate ONNX" },
        @{ Path = $candidateContractPath; Sha256 = $candidateContractSha256; Description = "candidate contract" },
        @{ Path = $candidateLabelsPath; Sha256 = $candidateLabelsSha256; Description = "candidate labels" },
        @{ Path = $trainingSummaryPath; Sha256 = $trainingSummarySha256; Description = "training summary" },
        @{ Path = $valSummaryPath; Sha256 = $valSummarySha256; Description = "val GPU summary" },
        @{ Path = $testSummaryPath; Sha256 = $testSummarySha256; Description = "test GPU summary" },
        @{ Path = $checkpointAttestor; Sha256 = $checkpointAttestorSha256; Description = "checkpoint attestor" },
        @{ Path = $sidecarAttestor; Sha256 = $sidecarAttestorSha256; Description = "sidecar attestor" }
    )
foreach ($binding in $sourceBindings) {
    Assert-FileHash `
        ([string]$binding.Path) `
        ([string]$binding.Sha256) `
        ([string]$binding.Description)
}

$temporaryEvidence = $evidencePath + ".tmp-" + [Guid]::NewGuid().ToString("N")
Require-NewPath $temporaryEvidence "temporary guarded v13 evidence"
try {
    $json = $validationEvidence | ConvertTo-Json -Depth 12
    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [IO.File]::WriteAllText(
        $temporaryEvidence,
        $json + [Environment]::NewLine,
        $utf8NoBom)
    $written = Read-GuardedJson $temporaryEvidence "temporary guarded v13 evidence"
    if ([int]$written.schema_version -ne 1 `
        -or [string]$written.kind -ne "receipt_unified_status_text_v13_guarded_validation_v1" `
        -or @($written.evaluations).Count -ne 2) {
        throw "Temporary guarded evidence failed its own schema check."
    }
    Require-NewPath $evidencePath "guarded v13 evidence"
    # The JSON serialization/self-parse above is intentionally outside the
    # first hash sweep. Recheck immediately adjacent to the atomic rename so
    # evidence can never publish after one of its bound sources changed.
    foreach ($binding in $sourceBindings) {
        Assert-FileHash `
            ([string]$binding.Path) `
            ([string]$binding.Sha256) `
            ([string]$binding.Description)
    }
    Move-Item -LiteralPath $temporaryEvidence -Destination $evidencePath
}
finally {
    if (Test-Path -LiteralPath $temporaryEvidence -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryEvidence -Force
    }
}

Write-Host ""
Write-Host "V13_GUARDED_EVIDENCE_RECOVERY_PASS"
Write-Host "  run=$RunDirectory"
Write-Host "  pseudo-labels=$pseudoLabels"
Write-Host "  dataset-root=$datasetRoot"
Write-Host "  seed-checkpoint=$seedCheckpoint"
Write-Host "  candidate-model=$candidateModel"
Write-Host "  val-summary=$valSummaryPath"
Write-Host "  test-summary=$testSummaryPath"
Write-Host "  evidence=$evidencePath"
Write-Host ("  recipient-delegated-to-hybrid-formal={0}" -f `
    [bool]$validationEvidence.recipient_delivery_policy.delegated_in_any_held_out_split)
Write-Host "  temporary-attestation-reexport-performed=true"
Write-Host "  deterministic-reexport-onnx-byte-identical=true"
Write-Host ("  deterministic-reexport-sidecars-byte-identical={0}" -f `
    [bool]$sidecarAttestation.all_sidecars_byte_identical)
Write-Host "  deterministic-reexport-sidecars-semantically-equivalent=true"
Write-Host "  training-or-evaluation-performed=false"
