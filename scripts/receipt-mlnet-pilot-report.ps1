[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvaluationDir,
    [string]$ResultsDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$EvaluationDir = (Resolve-Path -LiteralPath $EvaluationDir).Path
$summaryPath = Join-Path $EvaluationDir "summary.json"
$comparisonsPath = Join-Path $EvaluationDir "comparisons.jsonl"
foreach ($required in @($summaryPath, $comparisonsPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing pilot evidence: $required"
    }
}

$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Get-OptionalPropertyValue {
    param(
        [object]$InputObject,
        [string]$Name
    )
    if ($null -eq $InputObject) {
        return $null
    }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Format-DiagnosticValue {
    param([object]$Value)
    if ($null -eq $Value) {
        return "null"
    }
    if ($Value -is [string]) {
        return $Value
    }
    return ($Value | ConvertTo-Json -Compress -Depth 8)
}

Write-Host "mlnet_cpu_pilot_metrics"
foreach ($fieldName in @("amount", "time", "payment_method_field", "recipient_field")) {
    $metric = $summary.by_field.PSObject.Properties[$fieldName].Value
    Write-Host ("  {0}={1}/{2}={3:P2}" -f $fieldName, $metric.raw_exact_matches, $metric.records, $metric.raw_exact_match)
}
$amountSemantic = Get-OptionalPropertyValue $summary "amount_semantic"
if ($null -ne $amountSemantic) {
    Write-Host ("  amount_semantic={0}/{1}={2:P2}; diagnostic-only={3}; affects-acceptance={4}" -f `
        $amountSemantic.exact_matches, $amountSemantic.records, $amountSemantic.exact_match, `
        $amountSemantic.diagnostic_only, $amountSemantic.affects_acceptance)
}
Write-Host "  formal-delivery-gate=$($summary.formal_delivery_gate)"
Write-Host "  pilot-thresholds-passed=$($summary.pilot_thresholds_passed)"

$mismatches = @(
    Get-Content -LiteralPath $comparisonsPath -Encoding UTF8 |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.raw_exact -ne $true }
)
Write-Host "mlnet_cpu_pilot_mismatches count=$($mismatches.Count)"
$mismatchFormat = "  field={0}; reference=[{1}]; candidate=[{2}]; amount-semantic={3}; " + `
    "reference-decimal=[{4}]; candidate-decimal=[{5}]; ctc=[{6}]; structured=[{7}]; " + `
    "bbox={8}; detector-score={9}; geometry={10}; teacher-bbox={11}; teacher-detector-score={12}; " + `
    "teacher-geometry={13}; teacher-diagnostic-error={14}; teacher-result={15}; result={16}; source={17}"
foreach ($row in $mismatches) {
    $semanticExact = Get-OptionalPropertyValue $row "amount_semantic_exact"
    $referenceDecimal = Get-OptionalPropertyValue $row "reference_amount_decimal"
    $candidateDecimal = Get-OptionalPropertyValue $row "candidate_amount_decimal"
    $ctcCandidate = Get-OptionalPropertyValue $row "ctc_candidate_text"
    $structuredCandidate = Get-OptionalPropertyValue $row "structured_candidate_text"
    $bbox = Get-OptionalPropertyValue $row "detection_bbox_image"
    $detectorScore = Get-OptionalPropertyValue $row "detection_score"
    $geometry = Get-OptionalPropertyValue $row "result_geometry"
    $teacherResult = Get-OptionalPropertyValue $row "teacher_result_json"
    $teacherBbox = $null
    $teacherDetectorScore = $null
    $teacherGeometry = $null
    $teacherDiagnosticError = $null
    if ($teacherResult -is [string] -and
        -not [string]::IsNullOrWhiteSpace($teacherResult) -and
        (Test-Path -LiteralPath $teacherResult -PathType Leaf)) {
        try {
            $teacherPayload = Get-Content -LiteralPath $teacherResult -Raw -Encoding UTF8 | ConvertFrom-Json
            $teacherGeometry = Get-OptionalPropertyValue $teacherPayload "geometry"
            $teacherDetections = Get-OptionalPropertyValue $teacherPayload "detections"
            foreach ($teacherDetection in @($teacherDetections)) {
                if ($null -eq $teacherDetection -or
                    (Get-OptionalPropertyValue $teacherDetection "label") -ne $row.field) {
                    continue
                }
                $teacherBbox = Get-OptionalPropertyValue $teacherDetection "bbox_rectified"
                $teacherDetectorScore = Get-OptionalPropertyValue $teacherDetection "score"
                break
            }
        }
        catch {
            $teacherDiagnosticError = $_.Exception.Message
        }
    }
    Write-Host ($mismatchFormat -f `
        $row.field, $row.reference_text, $row.candidate_text, (Format-DiagnosticValue $semanticExact), `
        (Format-DiagnosticValue $referenceDecimal), (Format-DiagnosticValue $candidateDecimal), `
        (Format-DiagnosticValue $ctcCandidate), (Format-DiagnosticValue $structuredCandidate), `
        (Format-DiagnosticValue $bbox), (Format-DiagnosticValue $detectorScore), `
        (Format-DiagnosticValue $geometry), (Format-DiagnosticValue $teacherBbox), `
        (Format-DiagnosticValue $teacherDetectorScore), (Format-DiagnosticValue $teacherGeometry), `
        (Format-DiagnosticValue $teacherDiagnosticError), (Format-DiagnosticValue $teacherResult), `
        $row.result_json, $row.source)
}

if (-not [string]::IsNullOrWhiteSpace($ResultsDir)) {
    $ResultsDir = (Resolve-Path -LiteralPath $ResultsDir).Path
    $runtimeSummaryPath = Join-Path $ResultsDir "inference_summary.json"
    if (-not (Test-Path -LiteralPath $runtimeSummaryPath -PathType Leaf)) {
        throw "Missing runtime summary: $runtimeSummaryPath"
    }
    $runtime = Get-Content -LiteralPath $runtimeSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "mlnet_cpu_stage_mean_ms"
    foreach ($property in $runtime.stage_latency_ms.PSObject.Properties) {
        Write-Host ("  {0}={1}" -f $property.Name, $property.Value.mean)
    }
    Write-Host "  total-inference=$($runtime.inference_latency_ms.mean)"
}
