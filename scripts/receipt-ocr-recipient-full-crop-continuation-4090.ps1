[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PilotRoot,
    [Parameter(Mandatory = $true)]
    [string]$FullRecords,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# These are intentionally not caller parameters.  B8 is a fixed, analysis-only
# continuation pilot and cannot be stretched into a 24/80-epoch route.
$epochs = 8
$batchSize = 10
$learningRate = 0.0001
$seed = 42
$augmentation = "robust_v2"
$recipientDenominator = 6789
$sourceMatches = 5468
$minimumBestMatches = 5790
$minimumGainMatches = 136
$maximumTailGapMatches = 67
$finalTargetMatches = 6111
$authorization = "fixed_8_epoch_legacy_trim0_continuation_pilot_only"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"

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

function Require-FreshNonReparseOutput([string]$Path) {
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing -or (Test-Path -LiteralPath $Path)) {
        throw "Refusing to reuse continuation output: $Path"
    }
    $ancestor = Split-Path -Parent $Path
    while (-not [string]::IsNullOrWhiteSpace($ancestor)) {
        $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Continuation output traverses a symlink/junction/reparse ancestor: $ancestor"
        }
        $next = Split-Path -Parent $ancestor
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $ancestor) { break }
        $ancestor = $next
    }
}

function Invoke-Python([string[]]$CommandArguments, [string]$Description) {
    & $pythonExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Read-Json([string]$Path) {
    Require-File $Path "JSON evidence"
    $normalized = (& $pythonExe $normalizer $Path) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to normalize JSON evidence: $Path" }
    return ($normalized | ConvertFrom-Json)
}

function Open-ReadLease([string]$Path, [string]$Description) {
    Require-File $Path $Description
    # FileShare.Read denies write/delete/rename for the complete source window.
    return [IO.File]::Open(
        [IO.Path]::GetFullPath($Path),
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read)
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $normalizer "JSON normalizer"
Require-Directory $PilotRoot "fixed full-crop pilot root"
Require-File $FullRecords "full source manifest"
Require-Directory $DatasetRoot "recipient crop dataset root"
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
Require-FreshNonReparseOutput $OutputRoot

$sourceTests = @(
    (Join-Path $repoRoot "tests\test_recipient_full_crop_continuation.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_pilot.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_seed_sanitizer.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v13.py")
)
foreach ($test in $sourceTests) { Require-File $test "continuation source-contract test" }

$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0 -or [string]$gpuRows[0] -notmatch "4090") {
    throw "CUDA device 0 must be an RTX 4090. Observed: $($gpuRows -join '; ')"
}

Write-Host "receipt_recipient_full_crop_legacy_continuation_4090 preflight"
Write-Host "  source: fixed r031004-06/full-crop-pilot-8e-r2 epoch 6 = 5468/6789"
Write-Host "  closure: pilot decision+summary, blind/full manifests, sanitizer lineage, code and hashes"
Write-Host "  init: v13 legacy trim0, exact config/maps/all-state copy; fresh optimizer/epoch/sampler/history"
Write-Host "  fixed: cuda:0 RTX4090, epochs=$epochs, val_every=1, lr=$learningRate, batch=$batchSize, seed=$seed, aug=$augmentation"
Write-Host "  gates: best>=${minimumBestMatches}/${recipientDenominator}; e8-e4>=${minimumGainMatches}; best-e8<=${maximumTailGapMatches}"
Write-Host "  analysis only: no test, no ONNX, no production, no 24/80 epoch route"

Invoke-Python ((@("-m", "pytest", "-q")) + $sourceTests) "continuation source-contract tests"
Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_full_crop_continuation", "inspect",
    "--pilot-root", $PilotRoot,
    "--full-records", $FullRecords
) "fixed continuation source inspection"
if ($CheckOnly) {
    Write-Host "receipt_recipient_full_crop_legacy_continuation_4090 preflight=passed"
    exit 0
}

$pilotTraining = Join-Path $PilotRoot "training-full-crop-pilot"
$pilotBlind = Join-Path $PilotRoot "blind-train-val"
$blindRecords = Join-Path $pilotBlind "unified_fields.train-val.jsonl"
$blindContract = Join-Path $pilotBlind "blind.contract.json"
$sourceDirectory = Join-Path $OutputRoot "sealed-source"
$authorizedCheckpoint = Join-Path $sourceDirectory "authorized-pilot-best.pt"
$sourceContract = Join-Path $sourceDirectory "continuation-source.contract.json"
$trainingRoot = Join-Path $OutputRoot "training-full-crop-continuation-8e"
$decisionPath = Join-Path $trainingRoot "continuation_decision.json"

$leases = [Collections.Generic.List[IDisposable]]::new()
try {
    # Hold the complete local Python/script dependency closure before the seal
    # process starts.  The sealed contract repeats these bindings and its
    # source_artifacts loop below independently checks/leases them again.
    $packageRoot = Join-Path $repoRoot "src\transfer_receipt_ai"
    foreach ($path in @(
        (Join-Path $packageRoot "__init__.py"),
        (Join-Path $packageRoot "labels.py"),
        (Join-Path $packageRoot "model.py"),
        (Join-Path $packageRoot "ocr.py"),
        (Join-Path $packageRoot "onnx_runtime.py"),
        (Join-Path $packageRoot "recipient_beam.py"),
        (Join-Path $packageRoot "recipient_audit.py"),
        (Join-Path $packageRoot "ocr_unified_dataset.py"),
        (Join-Path $packageRoot "ocr_unified_targets.py"),
        (Join-Path $packageRoot "ocr_unified.py"),
        (Join-Path $packageRoot "recipient_blind_manifest.py"),
        (Join-Path $packageRoot "recipient_full_crop_seed_sanitizer.py"),
        (Join-Path $packageRoot "recipient_full_crop_pilot.py"),
        (Join-Path $packageRoot "recipient_full_crop_continuation.py"),
        (Join-Path $PSScriptRoot "receipt-ocr-recipient-full-crop-pilot-4090.ps1"),
        (Join-Path $PSScriptRoot "receipt-ocr-recipient-full-crop-continuation-4090.ps1"),
        $normalizer
    )) {
        $leases.Add((Open-ReadLease $path "continuation code-closure artifact"))
    }

    foreach ($path in @(
        (Join-Path $pilotTraining "best.pt"),
        (Join-Path $pilotTraining "training_summary.json"),
        (Join-Path $pilotTraining "pilot_decision.json"),
        $blindRecords,
        $blindContract,
        $FullRecords
    )) {
        $leases.Add((Open-ReadLease $path "fixed pilot closure artifact"))
    }

    New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
    $created = Get-Item -LiteralPath $OutputRoot -Force -ErrorAction Stop
    if (($created.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Fresh continuation output unexpectedly became a reparse point: $OutputRoot"
    }
    New-Item -ItemType Directory -Path $sourceDirectory -ErrorAction Stop | Out-Null

    Invoke-Python @(
        "-m", "transfer_receipt_ai.recipient_full_crop_continuation", "seal",
        "--pilot-root", $PilotRoot,
        "--output-checkpoint", $authorizedCheckpoint,
        "--output-contract", $sourceContract
    ) "continuation source sealing"

    $sealed = Read-Json $sourceContract
    foreach ($property in $sealed.source_artifacts.PSObject.Properties) {
        $artifactPath = [string]$property.Value.path
        $leases.Add((Open-ReadLease $artifactPath ("sealed source artifact " + $property.Name)))
    }
    $leases.Add((Open-ReadLease $authorizedCheckpoint "authorized continuation checkpoint"))
    $leases.Add((Open-ReadLease $sourceContract "continuation source contract"))

    Invoke-Python @(
        "-m", "transfer_receipt_ai.recipient_full_crop_continuation", "verify",
        "--pilot-root", $PilotRoot,
        "--contract", $sourceContract,
        "--authorized-checkpoint", $authorizedCheckpoint,
        "--full-records", $FullRecords
    ) "leased continuation source verification"

    Invoke-Python @(
        "-m", "transfer_receipt_ai.recipient_full_crop_continuation", "run",
        "--pilot-root", $PilotRoot,
        "--source-contract", $sourceContract,
        "--authorized-checkpoint", $authorizedCheckpoint,
        "--records", $blindRecords,
        "--blind-contract", $blindContract,
        "--dataset-root", $DatasetRoot,
        "--output", $trainingRoot,
        "--device", "cuda:0",
        "--num-workers", "$NumWorkers",
        "--prefetch-factor", "$PrefetchFactor",
        "--train-progress-every", "$TrainProgressEvery"
    ) "fixed eight-epoch legacy continuation"

    $decision = Read-Json $decisionPath
    if ([string]$decision.kind -ne "receipt_recipient_full_crop_legacy_continuation_decision_v1" `
        -or [string]$decision.source_kind -ne "receipt_recipient_full_crop_legacy_continuation_source_v1" `
        -or [string]$decision.source_authorization -ne $authorization `
        -or $decision.analysis_only -ne $true `
        -or $decision.production_route_authorized -ne $false `
        -or $decision.test_opened -ne $false `
        -or $decision.onnx_exported -ne $false `
        -or [int]$decision.epochs -ne $epochs `
        -or [int]$decision.fixed_gates.recipient_denominator -ne $recipientDenominator `
        -or [int]$decision.fixed_gates.source_recipient_matches -ne $sourceMatches `
        -or [int]$decision.fixed_gates.minimum_best_matches -ne $minimumBestMatches `
        -or [int]$decision.fixed_gates.minimum_epoch4_to_8_gain_matches -ne $minimumGainMatches `
        -or [int]$decision.fixed_gates.maximum_best_to_epoch8_gap_matches -ne $maximumTailGapMatches `
        -or [int]$decision.fixed_gates.final_target_matches -ne $finalTargetMatches `
        -or $decision.passed -ne $true `
        -or [string]$decision.pass_authorization.authorization -ne "fresh_exactly_16_from_original_pilot_best_only" `
        -or [int]$decision.pass_authorization.epochs -ne 16 `
        -or [string]$decision.pass_authorization.source -ne "original_pilot_best_not_b8_best" `
        -or $decision.pass_authorization.no_24_epoch_route -ne $true `
        -or $decision.pass_authorization.no_80_epoch_route -ne $true) {
        throw "Continuation decision does not satisfy the fixed B8 analysis contract."
    }

    Write-Host "CONTINUATION PASS: analysis only; test unopened; no ONNX or production authorization."
    Write-Host ("  best={0}/{1}; e8-e4={2}; best-e8={3}" -f `
        [int]$decision.observed.best_matches, $recipientDenominator, `
        [int]$decision.observed.epoch4_to_8_gain_matches, `
        [int]$decision.observed.best_to_epoch8_gap_matches)
    Write-Host "  only next authorization: fresh exactly 16 epochs from original pilot best"
}
finally {
    for ($index = $leases.Count - 1; $index -ge 0; $index--) {
        $leases[$index].Dispose()
    }
}
