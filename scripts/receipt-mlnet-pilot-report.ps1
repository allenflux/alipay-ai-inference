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
Write-Host "mlnet_cpu_pilot_metrics"
foreach ($fieldName in @("amount", "time", "payment_method_field", "recipient_field")) {
    $metric = $summary.by_field.PSObject.Properties[$fieldName].Value
    Write-Host ("  {0}={1}/{2}={3:P2}" -f $fieldName, $metric.raw_exact_matches, $metric.records, $metric.raw_exact_match)
}
Write-Host "  formal-delivery-gate=$($summary.formal_delivery_gate)"
Write-Host "  pilot-thresholds-passed=$($summary.pilot_thresholds_passed)"

$mismatches = @(
    Get-Content -LiteralPath $comparisonsPath -Encoding UTF8 |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.raw_exact -ne $true }
)
Write-Host "mlnet_cpu_pilot_mismatches count=$($mismatches.Count)"
foreach ($row in $mismatches) {
    Write-Host ("  field={0}; reference=[{1}]; candidate=[{2}]; source={3}" -f `
        $row.field, $row.reference_text, $row.candidate_text, $row.source)
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
