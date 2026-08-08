[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$DecisionName = "analysis-decision.recovered.json",
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$recoveryVerifier = Join-Path $repoRoot "src\transfer_receipt_ai\recipient_random_bootstrap_recovery.py"
$recoveryTest = Join-Path $repoRoot "tests\test_recipient_random_bootstrap_recovery.py"
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$inputContract = Join-Path $OutputRoot "bootstrap-input.contract.json"
$rootOutput = Join-Path $OutputRoot "random-root-1e"
$pilotOutput = Join-Path $OutputRoot "strict-recipient-warmstart-8e"
$decisionPath = Join-Path $OutputRoot $DecisionName

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

function Require-NonReparse([string]$Path, [string]$Description) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $item = Get-Item -LiteralPath $cursor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "${Description} traverses a symlink/junction/reparse point: $cursor"
        }
        $next = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $cursor) {
            break
        }
        $cursor = $next
    }
}

function Invoke-Python([string[]]$CommandArguments, [string]$Description) {
    & $pythonExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $recoveryVerifier "recovery verifier"
Require-File $recoveryTest "recovery source-contract test"
Require-File $MyInvocation.MyCommand.Path "recovery launcher"
Require-NonReparse $recoveryVerifier "recovery verifier"
Require-NonReparse $MyInvocation.MyCommand.Path "recovery launcher"

Invoke-Python @("-m", "pytest", "-q", $recoveryTest) "recovery source-contract tests"
if ($CheckOnly) {
    Write-Host "recipient_random_bootstrap_recovery preflight=passed"
    exit 0
}

Require-Directory $OutputRoot "completed bootstrap output"
Require-File $inputContract "bootstrap input contract"
Require-Directory $rootOutput "random-root output"
Require-Directory $pilotOutput "strict warm-start output"
Require-NonReparse $OutputRoot "completed bootstrap output"
if (Test-Path -LiteralPath $decisionPath) {
    throw "Refusing to overwrite recovery decision: $decisionPath"
}

Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_random_bootstrap_recovery",
    "--input-contract", $inputContract,
    "--root-output", $rootOutput,
    "--pilot-output", $pilotOutput,
    "--output", $decisionPath,
    "--recovery-verifier", $recoveryVerifier,
    "--recovery-launcher", $MyInvocation.MyCommand.Path
) "random-root bootstrap recovery finalization"

$decisionItem = Get-Item -LiteralPath $decisionPath -Force -ErrorAction Stop
$decisionItem.IsReadOnly = $true
if (-not (Get-Item -LiteralPath $decisionPath -Force).IsReadOnly) {
    throw "Unable to seal recovery decision read-only: $decisionPath"
}
$decision = Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ""
Write-Host "recipient_random_bootstrap_recovery final"
Write-Host ("  best recipient={0:P2}; epoch4={1:P2}; epoch8={2:P2}; gain={3:P2}" -f `
    [double]$decision.recipient_observed.best_exact, `
    [double]$decision.recipient_observed.epoch4_exact, `
    [double]$decision.recipient_observed.epoch8_exact, `
    [double]$decision.recipient_observed.epoch4_to_8_gain)
Write-Host ("  candidate denominators: amount={0}; time={1}; payment={2}; recipient={3}" -f `
    [int]$decision.candidate_denominator_evidence.candidate_val_denominators.amount, `
    [int]$decision.candidate_denominator_evidence.candidate_val_denominators.time, `
    [int]$decision.candidate_denominator_evidence.candidate_val_denominators.payment_method_field, `
    [int]$decision.candidate_denominator_evidence.candidate_val_denominators.recipient_field)
Write-Host "  DELIVERY=NOT AUTHORIZED; ONNX=NOT EXPORTED; recovery=ANALYSIS ONLY"
Write-Host "  decision=$decisionPath"
if ($decision.continuation_16_epoch_authorized -ne $true) {
    Write-Host "  16-epoch continuation=BLOCKED by unchanged 75%/epoch4-to-8 analysis gates"
    exit 3
}
Write-Host "  16-epoch continuation=AUTHORIZED FOR ANALYSIS ONLY"
exit 0
