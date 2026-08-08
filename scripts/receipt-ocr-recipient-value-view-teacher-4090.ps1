[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1",
    [ValidateRange(0, 6788)]
    [int]$Limit = 0,
    [string]$OutputDirectory,
    [string]$Bundle
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$labelsRoot = Join-Path $TeacherRoot "paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"
$records = Join-Path $TeacherRoot "unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
$scriptPath = Join-Path $PSScriptRoot "receipt-ocr-recipient-value-view-teacher.py"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $tag = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $OutputDirectory = Join-Path $TeacherRoot "recipient-value-view-teacher-val-$tag"
}

foreach ($required in @($pythonExe, $labelsRoot, $records, $scriptPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing recipient value-view teacher dependency: $required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Refusing to reuse recipient value-view teacher output: $OutputDirectory"
}
if (-not [string]::IsNullOrWhiteSpace($Bundle) -and -not (Test-Path -LiteralPath $Bundle)) {
    throw "Frozen Paddle bundle not found: $Bundle"
}
if ($Limit -eq 0 -and [string]::IsNullOrWhiteSpace($Bundle)) {
    throw "A full 6789-record val ceiling must bind the immutable Paddle audit bundle with -Bundle."
}
if (-not [string]::IsNullOrWhiteSpace($Bundle)) {
    $Bundle = [IO.Path]::GetFullPath($Bundle)
}

$arguments = @(
    $scriptPath,
    "--manifest", $records,
    "--dataset-root", $labelsRoot,
    "--output", $OutputDirectory,
    "--device", "cuda:0",
    "--progress-every", "25"
)
if ($Limit -gt 0) {
    $arguments += @("--limit", "$Limit")
}
if (-not [string]::IsNullOrWhiteSpace($Bundle)) {
    $arguments += @("--bundle", $Bundle)
}

Write-Host "recipient_value_view_teacher_4090"
Write-Host "  fixed route=left trim 30% -> Paddle cls+rec; confidence>=0.80; det/parser/full-layout disabled"
Write-Host "  split=val (hard locked); device=cuda:0; limit=$Limit; target=90.00%"
Write-Host "  output=$OutputDirectory"
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Recipient value-view teacher ceiling failed with exit code $LASTEXITCODE"
}
