[CmdletBinding()]
param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$Mode = "formal",
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [string]$RunDirectory,
    [ValidateRange(1, 1000)]
    [int]$SmokeLimit = 1,
    [ValidateRange(1, 10000)]
    [int]$PilotLimit = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$scorer = Join-Path $PSScriptRoot "receipt_mlnet_unified_evaluate.py"
$packager = Join-Path $PSScriptRoot "receipt-mlnet-unified-package-validate-4090.ps1"
$records = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $RunDirectory = Join-Path $TeacherRoot "unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
}

foreach ($required in @($pythonExe, $scorer, $packager, $records, (Join-Path $RunDirectory "best.onnx"))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing CPU delivery dependency: $required"
    }
}

$dataRoot = Split-Path -Parent $TeacherRoot
$validationRoot = Join-Path $dataRoot "delivery-validation"
$deliveryRoot = Join-Path $dataRoot "delivery"
New-Item -ItemType Directory -Path $validationRoot, $deliveryRoot -Force | Out-Null

$tag = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$inputList = Join-Path $validationRoot "mlnet-wide1536-val-inputs-$tag.txt"

Write-Host "mlnet_production_cpu_prepare"
& $pythonExe $scorer prepare `
    --records $records `
    --output $inputList `
    --split val
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare the canonical val input list; exit code $LASTEXITCODE"
}

if ($Mode -eq "smoke") {
    $output = Join-Path $validationRoot "mlnet-wide1536-cpu-smoke-$tag"
    $delivery = Join-Path $deliveryRoot "ReceiptMlNet-wide1536-cpu-smoke-$tag"
    Write-Host "mlnet_production_cpu_smoke"
    & $packager `
        -RunDirectory $RunDirectory `
        -InputList $inputList `
        -Output $output `
        -DeliveryDir $delivery `
        -Limit $SmokeLimit `
        -RuntimeFlavor cpu `
        -IncludeDeviceModel `
        -Annotate all
}
elseif ($Mode -eq "pilot") {
    $output = Join-Path $validationRoot "mlnet-wide1536-cpu-pilot-$PilotLimit-$tag"
    $evaluation = Join-Path $validationRoot "mlnet-wide1536-cpu-pilot-$PilotLimit-e2e-$tag"
    $delivery = Join-Path $deliveryRoot "ReceiptMlNet-wide1536-cpu-pilot-$PilotLimit-$tag"
    Write-Host "mlnet_production_cpu_pilot"
    & $packager `
        -RunDirectory $RunDirectory `
        -InputList $inputList `
        -Output $output `
        -DeliveryDir $delivery `
        -Limit $PilotLimit `
        -RuntimeFlavor cpu `
        -IncludeDeviceModel `
        -Annotate none
    & $pythonExe $scorer score `
        --records $records `
        --results $output `
        --model (Join-Path $RunDirectory "best.onnx") `
        --output $evaluation `
        --split val `
        --limit $PilotLimit
    $pilotScoreExitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath (Join-Path $evaluation "summary.json"))) {
        throw "CPU pilot scorer did not write summary.json; exit code $pilotScoreExitCode"
    }
    Write-Host "  pilot-score-exit=$pilotScoreExitCode"
}
else {
    $output = Join-Path $validationRoot "mlnet-wide1536-cpu-full-$tag"
    $evaluation = Join-Path $validationRoot "mlnet-wide1536-cpu-full-e2e-$tag"
    $delivery = Join-Path $deliveryRoot "ReceiptMlNet-wide1536-cpu-production-$tag"
    Write-Host "mlnet_production_cpu_formal"
    & $packager `
        -RunDirectory $RunDirectory `
        -InputList $inputList `
        -Records $records `
        -EndToEndEvaluationDir $evaluation `
        -Output $output `
        -DeliveryDir $delivery `
        -RuntimeFlavor cpu `
        -IncludeDeviceModel `
        -Annotate none
}

Write-Host "mlnet_production_cpu_complete"
Write-Host "  mode=$Mode"
Write-Host "  input-list=$inputList"
Write-Host "  output=$output"
if ($Mode -in @("pilot", "formal")) {
    Write-Host "  evaluation=$evaluation"
}
Write-Host "  delivery=$delivery"
