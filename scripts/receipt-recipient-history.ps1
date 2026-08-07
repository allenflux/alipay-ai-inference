[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$history = Join-Path $PSScriptRoot "recipient-run-history.py"
foreach ($required in @($pythonExe, $history)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing recipient history dependency: $required"
    }
}
Write-Host "recipient_history_read_only"
& $pythonExe $history --root $TeacherRoot
if ($LASTEXITCODE -ne 0) {
    throw "Recipient history failed with exit code $LASTEXITCODE"
}
