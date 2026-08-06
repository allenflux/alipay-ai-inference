[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsDir,
    [Parameter(Mandatory = $true)]
    [string]$EvaluationDir,
    [ValidateRange(1, 10000)]
    [int]$Limit = 500,
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$scorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$report = Join-Path $PSScriptRoot "receipt-mlnet-pilot-report.ps1"
$records = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $RunDirectory = Join-Path $TeacherRoot "unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
}
$model = Join-Path $RunDirectory "best.onnx"

foreach ($required in @($pythonExe, $scorer, $report, $records, $model, $ResultsDir)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing pilot diagnostic dependency: $required"
    }
}
if (Test-Path -LiteralPath $EvaluationDir) {
    throw "Refusing to overwrite an existing pilot evaluation directory: $EvaluationDir"
}

& $pythonExe $scorer score `
    --records $records `
    --results $ResultsDir `
    --model $model `
    --output $EvaluationDir `
    --split val `
    --limit $Limit
$scoreExitCode = $LASTEXITCODE

$summary = Join-Path $EvaluationDir "summary.json"
if (-not (Test-Path -LiteralPath $summary)) {
    throw "Pilot scorer did not write summary.json; exit code $scoreExitCode"
}

& $report -EvaluationDir $EvaluationDir -ResultsDir $ResultsDir
if ($LASTEXITCODE -ne 0) {
    throw "Pilot diagnostic report failed with exit code $LASTEXITCODE"
}

Write-Host "mlnet_cpu_pilot_rescore_complete"
Write-Host "  score-exit=$scoreExitCode"
Write-Host "  evaluation=$EvaluationDir"
exit $scoreExitCode
