[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$V13EvidencePath,
    [Parameter(Mandatory = $true)]
    [string]$PaddleDeliveryBundle,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$DotnetExe
)

# From the repository root:
# powershell -ExecutionPolicy Bypass -File .\scripts\receipt-mlnet-hybrid-recipient-formal-ab.ps1 `
#   -V13EvidencePath D:\path\to\v13_status_ocr_validation.json `
#   -PaddleDeliveryBundle D:\path\to\ppocr-delivery `
#   -OutputDirectory D:\path\to\fresh-formal-output

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusFloor = 0.90

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

function Get-RequiredProperty([object]$Object, [string]$Name, [string]$Description) {
    if ($null -eq $Object) {
        throw "$Description is missing."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Description has no required $Name property."
    }
    # The unary comma preserves empty JSON arrays through PowerShell's output
    # pipeline instead of silently converting them to $null.
    return ,$property.Value
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
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

function Get-FiniteJsonNumber([object]$Value, [string]$Description) {
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

function Read-GuardedJson([string]$Path, [string]$Description) {
    Require-File $Path $Description
    try {
        $rawJson = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
        $trimmedJson = $rawJson.Trim()
        # Windows PowerShell 5.1 enumerates a top-level JSON array as pipeline
        # output. Reject non-object JSON before ConvertFrom-Json so a one-item
        # array cannot masquerade as the required evidence object.
        if (-not $trimmedJson.StartsWith("{", [StringComparison]::Ordinal) `
            -or -not $trimmedJson.EndsWith("}", [StringComparison]::Ordinal)) {
            throw "$Description must contain one top-level JSON object: $Path"
        }
        $document = ConvertFrom-Json -InputObject $trimmedJson
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

function Test-PathWithin([string]$Candidate, [string]$Parent) {
    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd([char[]]@('\', '/'))
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd([char[]]@('\', '/'))
    if ($candidateFull.Equals($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
    return $candidateFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Resolve-BoundFile(
    [object]$RawPath,
    [string]$EvidenceDirectory,
    [string]$Description
) {
    if ($RawPath -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$RawPath)) {
        throw "V13 evidence $Description path must be a non-empty JSON string."
    }
    $rawPathText = [string]$RawPath
    try {
        $candidate = if ([IO.Path]::IsPathRooted($rawPathText)) {
            [IO.Path]::GetFullPath($rawPathText)
        }
        else {
            [IO.Path]::GetFullPath((Join-Path $EvidenceDirectory $rawPathText))
        }
    }
    catch {
        throw "Invalid $Description path in v13 evidence: $rawPathText"
    }
    if (-not (Test-PathWithin $candidate $EvidenceDirectory)) {
        throw "V13 evidence $Description path escapes its evidence directory: $candidate"
    }
    Require-File $candidate "v13 evidence $Description"
    return $candidate
}

function Require-Sha256([object]$Value, [string]$Description) {
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be one lowercase SHA-256 string."
    }
    return [string]$Value
}

function Assert-PassedGpuSummary(
    [object]$Summary,
    [string]$Split,
    [string]$ModelSha256,
    [string]$RecordsSha256
) {
    $summaryModelSha256 = Require-Sha256 `
        (Get-RequiredProperty $Summary "model_sha256" "$Split GPU summary") `
        "$Split GPU summary model_sha256"
    $summaryRecordsSha256 = Require-Sha256 `
        (Get-RequiredProperty $Summary "records_sha256" "$Split GPU summary") `
        "$Split GPU summary records_sha256"
    $summarySplit = Get-RequiredProperty $Summary "evaluation_split" "$Split GPU summary"
    $providers = Get-RequiredProperty $Summary "providers" "$Split GPU summary"
    $acceptance = Get-RequiredProperty $Summary "acceptance" "$Split GPU summary"
    $requested = Get-RequiredProperty $acceptance "requested" "$Split GPU summary acceptance"
    $passed = Get-RequiredProperty $acceptance "passed" "$Split GPU summary acceptance"
    $rawFailures = Get-RequiredProperty $acceptance "failures" "$Split GPU summary acceptance"
    if ($summarySplit -isnot [string] `
        -or [string]$summarySplit -ne $Split `
        -or $providers -isnot [Array] `
        -or @($providers) -notcontains "CUDAExecutionProvider" `
        -or $requested -isnot [bool] -or $requested -ne $true `
        -or $passed -isnot [bool] `
        -or $rawFailures -isnot [Array]) {
        throw "$Split GPU summary has invalid evaluation/provider/acceptance schema."
    }
    $failures = @(
        $rawFailures |
            ForEach-Object {
                if ($_ -isnot [string]) {
                    throw "$Split GPU summary acceptance failures must contain only strings."
                }
                [string]$_
            } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $nonRecipientFailures = @(
        $failures |
            Where-Object { -not $_.StartsWith("recipient_field:", [StringComparison]::Ordinal) }
    )
    $aggregateStateValid = (
        ($passed -eq $true -and $failures.Count -eq 0) `
        -or ($passed -eq $false `
            -and $failures.Count -gt 0 `
            -and $nonRecipientFailures.Count -eq 0)
    )
    if ($summaryModelSha256 -cne $ModelSha256 `
        -or $summaryRecordsSha256 -cne $RecordsSha256 `
        -or -not $aggregateStateValid `
        -or $nonRecipientFailures.Count -ne 0) {
        throw "$Split GPU summary is not accepted amount/time/payment/status v13 evidence."
    }

    $byField = Get-RequiredProperty $Summary "by_field" "$Split GPU summary"
    foreach ($gate in @(
            @{ Field = "amount"; Metric = "raw_exact_match"; Acceptance = "min_amount_exact_match"; Floor = $amountFloor },
            @{ Field = "time"; Metric = "raw_exact_match"; Acceptance = "min_time_exact_match"; Floor = $timeFloor },
            @{ Field = "payment_method_field"; Metric = "raw_exact_match"; Acceptance = "min_payment_exact_match"; Floor = $paymentFloor },
            @{ Field = "transfer_status"; Metric = "ctc_raw_exact_match"; Acceptance = "min_status_exact_match"; Floor = $statusFloor }
        )) {
        $field = Get-RequiredProperty $byField ([string]$gate.Field) "$Split GPU summary by_field"
        $recordsValue = Get-RequiredProperty $field "records" "$Split $($gate.Field) metric"
        if (-not (Test-JsonIntegerEqual $recordsValue ([long]$recordsValue)) `
            -or [long]$recordsValue -le 0) {
            throw "$Split GPU summary has invalid $($gate.Field) records."
        }
        $metric = Get-FiniteJsonNumber `
            (Get-RequiredProperty $field ([string]$gate.Metric) "$Split $($gate.Field) metric") `
            "$Split $($gate.Field) $($gate.Metric)"
        $requestedFloor = Get-FiniteJsonNumber `
            (Get-RequiredProperty $acceptance ([string]$gate.Acceptance) "$Split acceptance") `
            "$Split $($gate.Acceptance)"
        if ($metric -lt [double]$gate.Floor -or $requestedFloor -lt [double]$gate.Floor) {
            throw "$Split GPU summary weakens or misses the fixed $($gate.Field) floor."
        }
    }

    $statusMetric = Get-RequiredProperty $byField "transfer_status" "$Split GPU summary by_field"
    $statusRecords = Get-RequiredProperty $statusMetric "records" "$Split transfer_status metric"
    $ctcRecords = Get-RequiredProperty $statusMetric "ctc_records" "$Split transfer_status metric"
    if (-not (Test-JsonIntegerEqual $statusRecords ([long]$statusRecords)) `
        -or -not (Test-JsonIntegerEqual $ctcRecords ([long]$ctcRecords)) `
        -or [long]$ctcRecords -le 0 `
        -or [long]$ctcRecords -ne [long]$statusRecords) {
        throw "$Split GPU summary has no complete visible-status OCR records."
    }
    $statusPolicy = Get-RequiredProperty $Summary "status_text_policy" "$Split GPU summary"
    if ([string](Get-RequiredProperty $statusPolicy "runtime_policy" "$Split status policy") -ne "decode_and_normalize_review_only" `
        -or [string](Get-RequiredProperty $statusPolicy "review_value" "$Split status policy") -ne "review") {
        throw "$Split GPU summary has an invalid visible-status runtime policy."
    }
    $statusCounts = Get-RequiredProperty $Summary "status_reference_class_counts" "$Split GPU summary"
    $pending = Get-RequiredProperty $statusCounts "pending" "$Split status counts"
    $failed = Get-RequiredProperty $statusCounts "failed" "$Split status counts"
    if (-not (Test-JsonIntegerEqual $pending ([long]$pending)) `
        -or -not (Test-JsonIntegerEqual $failed ([long]$failed)) `
        -or [long]$pending -lt 0 -or [long]$failed -lt 0) {
        throw "$Split GPU summary has invalid status reference counts."
    }
    if ([long]$pending + [long]$failed -gt 0) {
        $maxUnsafe = Get-RequiredProperty $acceptance "max_non_success_to_success" "$Split acceptance"
        $unsafe = Get-RequiredProperty $statusMetric "non_success_to_success" "$Split transfer_status metric"
        if (-not (Test-JsonIntegerEqual $maxUnsafe 0) `
            -or -not (Test-JsonIntegerEqual $unsafe 0)) {
            throw "$Split GPU summary does not preserve the zero non-success safety line."
        }
    }
}

$launcher = Join-Path $PSScriptRoot "receipt-mlnet-hybrid-recipient-cpu-ab.ps1"
Require-File $launcher "hybrid recipient CPU A/B launcher"

if ([string]::IsNullOrWhiteSpace($V13EvidencePath)) {
    throw "V13EvidencePath must not be empty."
}
$V13EvidencePath = [IO.Path]::GetFullPath($V13EvidencePath)
Require-File $V13EvidencePath "v13_status_ocr_validation.json"
if (-not [IO.Path]::GetFileName($V13EvidencePath).Equals(
        "v13_status_ocr_validation.json",
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "V13EvidencePath must name v13_status_ocr_validation.json."
}
$evidenceDirectory = [IO.Path]::GetFullPath((Split-Path -Parent $V13EvidencePath))
$evidenceSha256 = Get-Sha256 $V13EvidencePath
$evidence = Read-GuardedJson $V13EvidencePath "v13 evidence JSON"

$schemaVersion = Get-RequiredProperty $evidence "schema_version" "v13 evidence"
$kind = Get-RequiredProperty $evidence "kind" "v13 evidence"
$manifest = Get-RequiredProperty $evidence "manifest" "v13 evidence"
$candidate = Get-RequiredProperty $evidence "candidate" "v13 evidence"
$cpuPackaging = Get-RequiredProperty $evidence "cpu_packaging" "v13 evidence"
$candidateKind = Get-RequiredProperty $candidate "kind" "v13 candidate"
$candidateArchitecture = Get-RequiredProperty $candidate "architecture_version" "v13 candidate"
$statusRuntimePolicy = Get-RequiredProperty $candidate "status_text_runtime_policy" "v13 candidate"
$reviewValue = Get-RequiredProperty $candidate "review_value" "v13 candidate"
if (-not (Test-JsonIntegerEqual $schemaVersion 1) `
    -or $kind -isnot [string] `
    -or [string]$kind -ne "receipt_unified_status_text_v13_guarded_validation_v1" `
    -or $candidateKind -isnot [string] `
    -or [string]$candidateKind -ne "receipt_unified_field_reader_v13" `
    -or -not (Test-JsonIntegerEqual $candidateArchitecture 13) `
    -or $statusRuntimePolicy -isnot [string] `
    -or [string]$statusRuntimePolicy -ne "decode_and_normalize_review_only" `
    -or $reviewValue -isnot [string] `
    -or [string]$reviewValue -ne "review") {
    throw "Evidence is not schema-version-1 guarded v13 visible-status OCR evidence."
}
$runtimeFlavor = Get-RequiredProperty $cpuPackaging "required_runtime_flavor" "v13 cpu_packaging"
$rectification = Get-RequiredProperty $cpuPackaging "required_rectification" "v13 cpu_packaging"
$includeDeviceModel = Get-RequiredProperty $cpuPackaging "include_device_model" "v13 cpu_packaging"
if ($runtimeFlavor -isnot [string] `
    -or [string]$runtimeFlavor -ne "cpu" `
    -or $rectification -isnot [string] `
    -or [string]$rectification -ne "max-side-1600" `
    -or $includeDeviceModel -isnot [bool] `
    -or $includeDeviceModel -ne $true) {
    throw "V13 evidence is not bound to the complete production CPU/device pipeline."
}

$floors = Get-RequiredProperty $evidence "acceptance_floors" "v13 evidence"
foreach ($floor in @(
        @{ Name = "amount"; Minimum = $amountFloor },
        @{ Name = "time"; Minimum = $timeFloor },
        @{ Name = "payment_method_field"; Minimum = $paymentFloor },
        @{ Name = "recipient_field"; Minimum = $recipientFloor },
        @{ Name = "visible_transfer_status_cjk_text"; Minimum = $statusFloor }
    )) {
    $value = Get-FiniteJsonNumber `
        (Get-RequiredProperty $floors ([string]$floor.Name) "v13 acceptance_floors") `
        "v13 $($floor.Name) acceptance floor"
    if ($value -lt [double]$floor.Minimum) {
        throw "V13 evidence weakens the fixed $($floor.Name) acceptance floor."
    }
}

$evaluations = Get-RequiredProperty $evidence "evaluations" "v13 evidence"
if ($evaluations -isnot [Array]) {
    throw "V13 evidence evaluations must be a JSON array."
}
$valEvidence = @($evaluations | Where-Object { [string]$_.split -eq "val" })
$testEvidence = @($evaluations | Where-Object { [string]$_.split -eq "test" })
if ($valEvidence.Count -ne 1 -or $testEvidence.Count -ne 1) {
    throw "V13 evidence must contain exactly one val and one test evaluation."
}
foreach ($entry in @($valEvidence[0], $testEvidence[0])) {
    $split = [string](Get-RequiredProperty $entry "split" "v13 evaluation")
    $evaluated = Get-RequiredProperty $entry "evaluated" "v13 $split evaluation"
    $statusExact = Get-FiniteJsonNumber `
        (Get-RequiredProperty $entry "status_text_exact_match" "v13 $split evaluation") `
        "v13 $split status_text_exact_match"
    if ($evaluated -isnot [bool] -or $evaluated -ne $true `
        -or $statusExact -lt $statusFloor) {
        throw "V13 $split evaluation is not a completed visible-status result."
    }
}

$records = Resolve-BoundFile `
    (Get-RequiredProperty $manifest "records" "v13 manifest") `
    $evidenceDirectory `
    "manifest records"
$unifiedModel = Resolve-BoundFile `
    (Get-RequiredProperty $cpuPackaging "unified_model_path" "v13 cpu_packaging") `
    $evidenceDirectory `
    "CPU packaging unified model"
$candidateModel = Resolve-BoundFile `
    (Get-RequiredProperty $candidate "model" "v13 candidate") `
    $evidenceDirectory `
    "candidate model"
$candidateContract = Resolve-BoundFile `
    (Get-RequiredProperty $candidate "contract" "v13 candidate") `
    $evidenceDirectory `
    "candidate contract"
$candidateLabels = Resolve-BoundFile `
    (Get-RequiredProperty $candidate "labels" "v13 candidate") `
    $evidenceDirectory `
    "candidate labels"
$validationSummary = Resolve-BoundFile `
    (Get-RequiredProperty $cpuPackaging "onnx_validation_summary_path" "v13 cpu_packaging") `
    $evidenceDirectory `
    "CPU packaging ONNX validation summary"
$valSummaryPath = Resolve-BoundFile `
    (Get-RequiredProperty $valEvidence[0] "summary_path" "v13 val evaluation") `
    $evidenceDirectory `
    "val evaluation summary"
$testSummaryPath = Resolve-BoundFile `
    (Get-RequiredProperty $testEvidence[0] "summary_path" "v13 test evaluation") `
    $evidenceDirectory `
    "test evaluation summary"
if (-not $unifiedModel.Equals($candidateModel, [StringComparison]::OrdinalIgnoreCase)) {
    throw "V13 CPU packaging unified model does not equal candidate.model."
}
if (-not $validationSummary.Equals($valSummaryPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw "V13 CPU packaging validation summary does not equal the val evaluation summary."
}
$adjacentContract = [IO.Path]::ChangeExtension($unifiedModel, ".contract.json")
$adjacentLabels = [IO.Path]::ChangeExtension($unifiedModel, ".labels.json")
if (-not $candidateContract.Equals($adjacentContract, [StringComparison]::OrdinalIgnoreCase) `
    -or -not $candidateLabels.Equals($adjacentLabels, [StringComparison]::OrdinalIgnoreCase)) {
    throw "V13 evidence sidecars do not equal the contract/labels files loaded beside the unified model."
}

$recordsExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $manifest "records_sha256" "v13 manifest") `
    "v13 manifest records_sha256"
$candidateExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $candidate "model_sha256" "v13 candidate") `
    "v13 candidate model_sha256"
$packagingExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $cpuPackaging "unified_model_sha256" "v13 cpu_packaging") `
    "v13 cpu_packaging unified_model_sha256"
$contractExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $candidate "contract_sha256" "v13 candidate") `
    "v13 candidate contract_sha256"
$labelsExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $candidate "labels_sha256" "v13 candidate") `
    "v13 candidate labels_sha256"
$validationExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $cpuPackaging "onnx_validation_summary_sha256" "v13 cpu_packaging") `
    "v13 cpu_packaging onnx_validation_summary_sha256"
$valExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $valEvidence[0] "summary_sha256" "v13 val evaluation") `
    "v13 val summary_sha256"
$testExpectedSha256 = Require-Sha256 `
    (Get-RequiredProperty $testEvidence[0] "summary_sha256" "v13 test evaluation") `
    "v13 test summary_sha256"
$recordsSha256 = Get-Sha256 $records
$unifiedModelSha256 = Get-Sha256 $unifiedModel
$contractSha256 = Get-Sha256 $candidateContract
$labelsSha256 = Get-Sha256 $candidateLabels
$validationSummarySha256 = Get-Sha256 $validationSummary
$testSummarySha256 = Get-Sha256 $testSummaryPath
if ($recordsExpectedSha256 -cne $recordsSha256 `
    -or $candidateExpectedSha256 -cne $unifiedModelSha256 `
    -or $packagingExpectedSha256 -cne $unifiedModelSha256 `
    -or $contractExpectedSha256 -cne $contractSha256 `
    -or $labelsExpectedSha256 -cne $labelsSha256 `
    -or $validationExpectedSha256 -cne $validationSummarySha256 `
    -or $valExpectedSha256 -cne $validationSummarySha256 `
    -or $testExpectedSha256 -cne $testSummarySha256) {
    throw "V13 records/model/sidecar/evaluation hashes do not match their evidence bindings."
}

$unifiedContract = Read-GuardedJson $candidateContract "v13 unified model contract"
if ([string](Get-RequiredProperty $unifiedContract "onnx_file" "v13 unified model contract") `
        -ne [IO.Path]::GetFileName($unifiedModel) `
    -or [string](Get-RequiredProperty $unifiedContract "labels_file" "v13 unified model contract") `
        -ne [IO.Path]::GetFileName($candidateLabels) `
    -or [string](Get-RequiredProperty $unifiedContract "onnx_sha256" "v13 unified model contract") `
        -cne $unifiedModelSha256 `
    -or [string](Get-RequiredProperty $unifiedContract "labels_sha256" "v13 unified model contract") `
        -cne $labelsSha256) {
    throw "V13 unified model contract is not hash/name-bound to its adjacent ONNX and labels."
}

$valSummary = Read-GuardedJson $validationSummary "v13 val GPU summary"
$testSummary = Read-GuardedJson $testSummaryPath "v13 test GPU summary"
Assert-PassedGpuSummary $valSummary "val" $unifiedModelSha256 $recordsSha256
Assert-PassedGpuSummary $testSummary "test" $unifiedModelSha256 $recordsSha256
foreach ($binding in @(
        @{ Entry = $valEvidence[0]; Summary = $valSummary; Split = "val" },
        @{ Entry = $testEvidence[0]; Summary = $testSummary; Split = "test" }
    )) {
    $visibleRecords = Get-RequiredProperty $binding.Entry "visible_status_records" "v13 $($binding.Split) evaluation"
    $summaryVisibleRecords = Get-RequiredProperty `
        (Get-RequiredProperty $binding.Summary.by_field "transfer_status" "v13 $($binding.Split) summary") `
        "ctc_records" `
        "v13 $($binding.Split) transfer_status"
    $evidenceExact = Get-FiniteJsonNumber `
        (Get-RequiredProperty $binding.Entry "status_text_exact_match" "v13 $($binding.Split) evaluation") `
        "v13 $($binding.Split) evidence status exact"
    $summaryExact = Get-FiniteJsonNumber `
        (Get-RequiredProperty $binding.Summary.by_field.transfer_status "ctc_raw_exact_match" "v13 $($binding.Split) transfer_status") `
        "v13 $($binding.Split) summary status exact"
    if (-not (Test-JsonIntegerEqual $visibleRecords ([long]$summaryVisibleRecords)) `
        -or [long]$visibleRecords -ne [long]$summaryVisibleRecords `
        -or $evidenceExact -ne $summaryExact) {
        throw "V13 $($binding.Split) guarded status metrics do not match its GPU summary."
    }
}

$PaddleDeliveryBundle = [IO.Path]::GetFullPath($PaddleDeliveryBundle)
Require-Directory $PaddleDeliveryBundle "Paddle recipient delivery bundle"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Formal A/B output already exists; refusing result reuse: $OutputDirectory"
}

$arguments = @{
    Records = $records
    UnifiedModel = $unifiedModel
    PaddleDeliveryBundle = $PaddleDeliveryBundle
    OutputDirectory = $OutputDirectory
    Mode = "formal"
    Limit = 0
}
if (-not [string]::IsNullOrWhiteSpace($DotnetExe)) {
    $DotnetExe = [IO.Path]::GetFullPath($DotnetExe)
    Require-File $DotnetExe ".NET host"
    $arguments["DotnetExe"] = $DotnetExe
}

Write-Host "receipt_mlnet_hybrid_recipient_formal_ab_from_v13"
Write-Host "  evidence=$V13EvidencePath"
Write-Host "  evidence-root=$evidenceDirectory"
Write-Host "  records=$records"
Write-Host "  records-sha256=$recordsSha256"
Write-Host "  unified-model=$unifiedModel"
Write-Host "  unified-model-sha256=$unifiedModelSha256"
Write-Host "  unified-contract=$candidateContract"
Write-Host "  unified-labels=$candidateLabels"
Write-Host "  val-summary=$validationSummary"
Write-Host "  test-summary=$testSummaryPath"
Write-Host "  recipient-ppocr=$PaddleDeliveryBundle"
Write-Host "  output=$OutputDirectory"
Write-Host "  mode=formal; limit=0; delegated-device=cpu; detector/device enabled"
if (-not [string]::IsNullOrWhiteSpace($DotnetExe)) {
    Write-Host "  dotnet=$DotnetExe"
}

& $launcher @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Formal hybrid recipient CPU A/B failed with exit code $LASTEXITCODE"
}
if ((Get-Sha256 $V13EvidencePath) -cne $evidenceSha256 `
    -or (Get-Sha256 $records) -cne $recordsSha256 `
    -or (Get-Sha256 $unifiedModel) -cne $unifiedModelSha256 `
    -or (Get-Sha256 $candidateContract) -cne $contractSha256 `
    -or (Get-Sha256 $candidateLabels) -cne $labelsSha256 `
    -or (Get-Sha256 $validationSummary) -cne $validationSummarySha256 `
    -or (Get-Sha256 $testSummaryPath) -cne $testSummarySha256) {
    throw "V13 evidence or a bound records/model/sidecar/evaluation file changed during formal A/B."
}

Write-Host "receipt_mlnet_hybrid_recipient_formal_ab_launcher_pass"
Write-Host "  output=$OutputDirectory"
