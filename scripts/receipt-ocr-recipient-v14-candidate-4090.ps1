[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FullRecords,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$SeedCheckpoint,
    [string]$FullCropPilotRoot,
    [string]$FullCropSourceContract,
    [string]$CandidatePilotEvidence,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [ValidateRange(1, 100)]
    [int]$Epochs = 60,
    [ValidateRange(1, 32)]
    [int]$BatchSize = 10,
    [ValidateRange(1, 20)]
    [int]$ValidationEvery = 2,
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.0003,
    [ValidateRange(0, 16)]
    [int]$NumWorkers = 4,
    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2,
    [ValidateRange(0, 1000000)]
    [int]$TrainProgressEvery = 250,
    [switch]$Pilot,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fixed delivery floors. They are constants, not caller parameters.
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusTextFloor = 0.90
$requiredBackbone = "residual_positional_transformer_v2"
$requiredInit = "recipient_visual_context_reinit"
$requiredStatusPolicy = "decode_and_normalize_review_only"
$pilotMinimumBestRecipient = 0.75
$pilotMinimumEpoch4To8Gain = 0.02

if ($Pilot -and $Epochs -ne 8) {
    throw "Pilot mode is fixed to exactly 8 epochs so its stop rule remains comparable."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$normalizer = Join-Path $PSScriptRoot "normalize_json_summary.py"
$sourceModule = "transfer_receipt_ai.recipient_full_crop_candidate_source"

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

function Assert-NoReparseChain([string]$Path, [string]$Description) {
    $current = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "${Description} must not traverse a symlink/junction/reparse path: $current"
        }
        $next = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $current) {
            break
        }
        $current = $next
    }
}

function Require-FreshNonReparseOutput([string]$Path) {
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing -or (Test-Path -LiteralPath $Path)) {
        throw "Refusing to reuse existing, symlink, or reparse recipient v14 output: $Path"
    }
    Assert-NoReparseChain $Path "Recipient v14 output"
}

function Open-ReadLease([string]$Path, [string]$Description) {
    Require-File $Path $Description
    return [IO.File]::Open(
        [IO.Path]::GetFullPath($Path),
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read)
}

function Write-CreateNewUtf8([string]$Path, [string]$Text, [string]$Description) {
    $stream = $null
    try {
        $stream = [IO.File]::Open(
            $Path,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::Read)
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    catch [IO.IOException] {
        throw "${Description} already exists or could not be created atomically: $Path"
    }
    finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

function Get-TextSha256([string]$Text) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Protect-AuditRoot([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Full-crop training audit root must not be a reparse point: $Path"
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $identity) {
        throw "Unable to resolve the current Windows identity for the training audit registry."
    }
    $acl = Get-Acl -LiteralPath $Path
    $rule = [Security.AccessControl.FileSystemAccessRule]::new(
        $identity,
        [Security.AccessControl.FileSystemRights]::Delete -bor `
            [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles,
        [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor `
            [Security.AccessControl.InheritanceFlags]::ObjectInherit,
        [Security.AccessControl.PropagationFlags]::None,
        [Security.AccessControl.AccessControlType]::Deny)
    $acl.SetAccessRule($rule)
    Set-Acl -LiteralPath $Path -AclObject $acl
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
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to normalize JSON evidence: $Path"
    }
    return ($normalized | ConvertFrom-Json)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RawExact([object]$Summary, [string]$Field) {
    $fieldProperty = $Summary.by_field.PSObject.Properties[$Field]
    if ($null -eq $fieldProperty -or $null -eq $fieldProperty.Value) {
        throw "Validation summary has no $Field field."
    }
    $metric = $fieldProperty.Value.PSObject.Properties["raw_exact_match"]
    if ($null -eq $metric -or $null -eq $metric.Value) {
        throw "Validation summary has no $Field raw_exact_match."
    }
    return [double]$metric.Value
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $normalizer "JSON normalizer"
Require-File $FullRecords "full v13 unified manifest"
Require-Directory $DatasetRoot "recipient crop root"
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$FullRecords = [IO.Path]::GetFullPath($FullRecords)
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
Assert-NoReparseChain $FullRecords "Candidate full manifest"
Assert-NoReparseChain $DatasetRoot "Candidate dataset root"
Require-FreshNonReparseOutput $OutputRoot

$hasLegacySeed = -not [string]::IsNullOrWhiteSpace($SeedCheckpoint)
$hasFullCropRoot = -not [string]::IsNullOrWhiteSpace($FullCropPilotRoot)
$hasFullCropContract = -not [string]::IsNullOrWhiteSpace($FullCropSourceContract)
$hasCandidatePilotEvidence = -not [string]::IsNullOrWhiteSpace($CandidatePilotEvidence)
if ($hasLegacySeed -and ($hasFullCropRoot -or $hasFullCropContract -or $hasCandidatePilotEvidence)) {
    throw "SeedCheckpoint cannot be mixed with the full-crop source-contract route."
}
if ($hasFullCropRoot -ne $hasFullCropContract) {
    throw "FullCropPilotRoot and FullCropSourceContract are required together."
}
if (-not $hasLegacySeed -and -not ($hasFullCropRoot -and $hasFullCropContract)) {
    throw "Choose exactly one source route: SeedCheckpoint, or FullCropPilotRoot plus FullCropSourceContract."
}
$fullCropSourceMode = $hasFullCropRoot -and $hasFullCropContract
if (-not $fullCropSourceMode -and $hasCandidatePilotEvidence) {
    throw "CandidatePilotEvidence is valid only for the full-crop source-contract route."
}
if ($fullCropSourceMode) {
    if ($BatchSize -ne 10 `
        -or [Math]::Abs($LearningRate - 0.0003) -gt 0.000000000001 `
        -or $NumWorkers -ne 4 `
        -or $PrefetchFactor -ne 2) {
        throw "The full-crop residual route locks batch=10, lr=0.0003, workers=4, prefetch=2."
    }
    if ($Pilot -and $hasCandidatePilotEvidence) {
        throw "The fresh eight-epoch residual pilot cannot consume prior candidate-pilot evidence."
    }
    if (-not $Pilot -and -not $hasCandidatePilotEvidence) {
        throw "A fresh 60-epoch full-crop candidate requires passed CandidatePilotEvidence."
    }
    if (-not $Pilot -and $Epochs -ne 60) {
        throw "The post-pilot full-crop candidate is fixed to exactly 60 fresh epochs."
    }
    if ($Pilot) {
        if ($PSBoundParameters.ContainsKey("ValidationEvery") -and $ValidationEvery -ne 1) {
            throw "The full-crop residual pilot is fixed to validation at every epoch."
        }
        $ValidationEvery = 1
    }
    elseif ($ValidationEvery -ne 2) {
        throw "The full-crop 60-epoch candidate is fixed to validation every 2 epochs."
    }
    $FullCropPilotRoot = [IO.Path]::GetFullPath($FullCropPilotRoot)
    $FullCropSourceContract = [IO.Path]::GetFullPath($FullCropSourceContract)
    Require-Directory $FullCropPilotRoot "passed full-crop pilot root"
    Require-File $FullCropSourceContract "full-crop source contract"
}
else {
    $SeedCheckpoint = [IO.Path]::GetFullPath($SeedCheckpoint)
    Require-File $SeedCheckpoint "accepted v13 seed checkpoint"
    if ([IO.Path]::GetExtension($SeedCheckpoint) -ne ".pt") {
        throw "SeedCheckpoint must be a PyTorch .pt file."
    }
}

$sourceTests = @(
    (Join-Path $repoRoot "tests\test_recipient_v14_candidate.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_candidate_source.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_pilot.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v12.py"),
    (Join-Path $repoRoot "tests\test_ocr_unified_v13.py")
)
foreach ($sourceTest in $sourceTests) {
    Require-File $sourceTest "recipient v14 contract test"
}

$sourceLeases = [Collections.Generic.List[IDisposable]]::new()
$sourceContract = $null
$candidatePilotContract = $null
$sourceRouteMode = "legacy_v13_visual_context_reinit"
$recipientValueLeftTrim = 0.30
$resolvedSeedCheckpoint = $SeedCheckpoint
if ($fullCropSourceMode) {
    $sourceRouteMode = "attested_full_crop_pilot_visual_context_reinit"
    $recipientValueLeftTrim = 0.0
    $sourceLeases.Add((Open-ReadLease $FullCropSourceContract "full-crop source contract"))
    Invoke-Python @(
        "-m", $sourceModule, "verify-source",
        "--pilot-root", $FullCropPilotRoot,
        "--contract", $FullCropSourceContract,
        "--full-records", $FullRecords
    ) "full-crop candidate-source verification"
    $sourceContract = Read-Json $FullCropSourceContract
    if ([string]$sourceContract.kind -ne "receipt_recipient_full_crop_candidate_source_v1" `
        -or $sourceContract.analysis_only -ne $true `
        -or $sourceContract.production_route_authorized -ne $false `
        -or $sourceContract.test_opened -ne $false `
        -or $sourceContract.onnx_exported -ne $false `
        -or [double]$sourceContract.recomputed_pilot_decision.observed.best_recipient_exact -lt $pilotMinimumBestRecipient `
        -or [double]$sourceContract.recomputed_pilot_decision.observed.best_recipient_exact -gt $recipientFloor `
        -or [string]$sourceContract.recomputed_pilot_decision.decision -ne "analysis_only_continue_to_separate_guarded_candidate") {
        throw "Full-crop source contract does not authorize the separate guarded candidate."
    }
    foreach ($artifactProperty in $sourceContract.artifacts.PSObject.Properties) {
        $artifactPath = [string]$artifactProperty.Value.path
        $sourceLeases.Add((Open-ReadLease $artifactPath ("source artifact " + $artifactProperty.Name)))
    }
    # Reopen the complete contract while every bound source/code artifact is
    # immutable. This closes the inspection-to-training mutation window.
    Invoke-Python @(
        "-m", $sourceModule, "verify-source",
        "--pilot-root", $FullCropPilotRoot,
        "--contract", $FullCropSourceContract,
        "--full-records", $FullRecords
    ) "leased full-crop candidate-source reinspection"
    $resolvedSeedCheckpoint = [string]$sourceContract.artifacts.best_checkpoint.path
    Require-File $resolvedSeedCheckpoint "source-contract pilot best checkpoint"
    if ([IO.Path]::GetExtension($resolvedSeedCheckpoint) -ne ".pt" `
        -or (Get-Sha256 $resolvedSeedCheckpoint) -ne [string]$sourceContract.artifacts.best_checkpoint.sha256) {
        throw "Full-crop source checkpoint is not the same content-bound pilot best.pt."
    }

    if (-not $Pilot) {
        $CandidatePilotEvidence = [IO.Path]::GetFullPath($CandidatePilotEvidence)
        Require-File $CandidatePilotEvidence "passed residual candidate-pilot evidence"
        $sourceLeases.Add((Open-ReadLease $CandidatePilotEvidence "candidate-pilot evidence"))
        Invoke-Python @(
            "-m", $sourceModule, "verify-candidate-pilot",
            "--evidence", $CandidatePilotEvidence,
            "--source-contract", $FullCropSourceContract,
            "--full-records", $FullRecords
        ) "residual candidate-pilot verification"
        $candidatePilotContract = Read-Json $CandidatePilotEvidence
        if ([string]$candidatePilotContract.kind -ne "receipt_recipient_v14_full_crop_residual_pilot_v1" `
            -or $candidatePilotContract.analysis_only -ne $true `
            -or $candidatePilotContract.production_route_authorized -ne $false `
            -or $candidatePilotContract.test_opened -ne $false `
            -or $candidatePilotContract.onnx_exported -ne $false `
            -or $candidatePilotContract.passed -ne $true `
            -or [string]$candidatePilotContract.decision -ne "analysis_only_continue_to_fresh_60_epoch_candidate") {
            throw "Candidate-pilot evidence does not authorize one fresh 60-epoch run."
        }
        foreach ($artifactProperty in $candidatePilotContract.artifacts.PSObject.Properties) {
            $artifactPath = [string]$artifactProperty.Value.path
            $sourceLeases.Add((Open-ReadLease $artifactPath ("candidate-pilot artifact " + $artifactProperty.Name)))
        }
        Invoke-Python @(
            "-m", $sourceModule, "verify-candidate-pilot",
            "--evidence", $CandidatePilotEvidence,
            "--source-contract", $FullCropSourceContract,
            "--full-records", $FullRecords
        ) "leased residual candidate-pilot reinspection"
    }
}
else {
    # Preserve the historical v14 route, but keep the selected legacy seed
    # immutable through parameter loading and evidence sealing.
    $sourceLeases.Add((Open-ReadLease $resolvedSeedCheckpoint "legacy v13 seed checkpoint"))
}
$sourceLeases.Add((Open-ReadLease $FullRecords "candidate full manifest"))

$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi did not report a CUDA GPU."
}
if ([string]$gpuRows[0] -notmatch "4090") {
    throw "CUDA device 0 must be an RTX 4090. Observed: $($gpuRows[0])"
}

Write-Host "receipt_recipient_v14_candidate preflight"
Write-Host "  architecture=v13 ABI + residual positional Transformer recipient branch"
Write-Host ("  source_route={0}; recipient_value_left_trim={1}" -f $sourceRouteMode, $recipientValueLeftTrim)
Write-Host "  optimizer=train only; checkpoint selection=val only; test=physically excluded"
Write-Host ("  fixed floors: amount={0:P2}, time={1:P2}, payment={2:P2}, recipient={3:P2}, status={4:P2}" -f `
    $amountFloor, $timeFloor, $paymentFloor, $recipientFloor, $statusTextFloor)
Write-Host ("  GPU: {0}" -f ($gpuRows -join "; "))

Invoke-Python ((@("-m", "pytest", "-q")) + $sourceTests) "recipient v14 source-contract tests"
if ($CheckOnly) {
    Write-Host "receipt_recipient_v14_candidate preflight=passed"
    exit 0
}

if ($fullCropSourceMode) {
    # A validation trend authorizes one attempt, not an unlimited sweep over
    # fresh output paths. The content-derived key survives path copies; an
    # interrupted/failed process still consumes the fixed experiment.
    $sourceSubjectId = [string]$sourceContract.source_subject_id
    if ($sourceSubjectId -notmatch "^[0-9a-f]{64}$") {
        throw "Full-crop source contract has no canonical path-independent subject identity."
    }
    $attemptStage = if ($Pilot) { "residual-8e" } else { "candidate-60e" }
    $candidatePilotSubjectId = $null
    $attemptCandidatePilotSubjectId = $null
    $attemptSubject = if ($Pilot) {
        "receipt-v14-full-crop-residual-8e-v1|$sourceSubjectId"
    }
    else {
        $candidatePilotSubjectId = [string]$candidatePilotContract.candidate_pilot_subject_id
        if ($candidatePilotSubjectId -notmatch "^[0-9a-f]{64}$") {
            throw "Candidate-pilot evidence has no canonical path-independent subject identity."
        }
        if ([string]$candidatePilotContract.source_subject_id -ne $sourceSubjectId) {
            throw "Candidate-pilot evidence is not bound to the source subject identity."
        }
        $attemptCandidatePilotSubjectId = $candidatePilotSubjectId
        "receipt-v14-full-crop-candidate-60e-v1|$sourceSubjectId|$candidatePilotSubjectId"
    }
    $attemptId = Get-TextSha256 $attemptSubject
    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    if ([string]::IsNullOrWhiteSpace($commonData)) {
        throw "Windows CommonApplicationData is unavailable; no persistent training-attempt registry can be used."
    }
    $trainingAuditRoot = Join-Path $commonData "ReceiptAI\recipient-v14-full-crop-training-v1"
    New-Item -ItemType Directory -Path $trainingAuditRoot -Force | Out-Null
    Protect-AuditRoot $trainingAuditRoot
    $attemptPath = Join-Path $trainingAuditRoot ("$attemptId.attempt.json")
    $attemptPayload = [ordered]@{
        schema_version = 1
        kind = "receipt_recipient_v14_full_crop_training_attempt_v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        attempt_id = $attemptId
        stage = $attemptStage
        source_subject_id = $sourceSubjectId
        candidate_pilot_subject_id = $attemptCandidatePilotSubjectId
        output_root = $OutputRoot
        full_manifest_sha256 = Get-Sha256 $FullRecords
        threat_model = "persistent local no-rerun guard; crash and failed training consume the fixed attempt"
    }
    Write-CreateNewUtf8 `
        $attemptPath (($attemptPayload | ConvertTo-Json -Depth 8) + "`n") `
        ("one-shot " + $attemptStage + " training attempt")
}

$blindRoot = Join-Path $OutputRoot "blind-train-val"
$blindRecords = Join-Path $blindRoot "unified_fields.train-val.jsonl"
$blindContractPath = Join-Path $blindRoot "blind.contract.json"
$trainingRoot = Join-Path $OutputRoot "training-v14-candidate"
$checkpoint = Join-Path $trainingRoot "best.pt"
$artifactRoot = Join-Path $OutputRoot "artifacts"
$model = Join-Path $artifactRoot "recipient-v14-candidate.onnx"
$validationRoot = Join-Path $OutputRoot "onnx-val-gpu"
$validationSummaryPath = Join-Path $validationRoot "summary.json"
$evidencePath = Join-Path $OutputRoot "recipient_v14_candidate.json"
$candidatePilotEvidencePath = Join-Path $OutputRoot "recipient_v14_candidate_pilot.json"
$trainingRecipePath = Join-Path $OutputRoot "recipient_v14_training_recipe.json"

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$createdOutput = Get-Item -LiteralPath $OutputRoot -Force -ErrorAction Stop
if (($createdOutput.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Fresh recipient v14 output unexpectedly became a reparse point: $OutputRoot"
}
Assert-NoReparseChain $OutputRoot "Created recipient v14 output"
if ($fullCropSourceMode) {
    $trainingRecipe = [ordered]@{
        schema_version = 1
        kind = "receipt_recipient_v14_full_crop_training_recipe_v1"
        analysis_only = $true
        production_route_authorized = $false
        test_opened = $false
        stage = $attemptStage
        source_subject_id = $sourceSubjectId
        candidate_pilot_subject_id = $candidatePilotSubjectId
        source_checkpoint_sha256 = Get-Sha256 $resolvedSeedCheckpoint
        full_manifest_sha256 = Get-Sha256 $FullRecords
        training_args = [ordered]@{
            device = "cuda:0"
            epochs = $Epochs
            batch_size = $BatchSize
            learning_rate = $LearningRate
            validation_every = $ValidationEvery
            seed = 42
            num_workers = $NumWorkers
            prefetch_factor = $PrefetchFactor
            persistent_workers = ($NumWorkers -gt 0)
            cuda_tf32 = $true
            cudnn_benchmark = $true
        }
    }
    Write-CreateNewUtf8 `
        $trainingRecipePath (($trainingRecipe | ConvertTo-Json -Depth 8) + "`n") `
        "full-crop training recipe"
    $sourceLeases.Add((Open-ReadLease $trainingRecipePath "full-crop training recipe"))
}
Invoke-Python @(
    "-m", "transfer_receipt_ai.recipient_blind_manifest",
    "--source", $FullRecords,
    "--output", $blindRecords,
    "--contract", $blindContractPath
) "blind train/val manifest build"

$blindContract = Read-Json $blindContractPath
if ([string]$blindContract.kind -ne "receipt_recipient_blind_train_val_manifest_v1" `
    -or $blindContract.test_labels_used -ne $false `
    -or $blindContract.test_metrics_computed -ne $false `
    -or (($blindContract.optimizer_supervision_splits -join ",") -ne "train") `
    -or (($blindContract.checkpoint_selection_splits -join ",") -ne "val") `
    -or (($blindContract.final_gate_only_splits -join ",") -ne "test") `
    -or [int]$blindContract.split_counts.train -le 0 `
    -or [int]$blindContract.split_counts.val -le 0 `
    -or [int]$blindContract.split_counts.test_excluded -le 0) {
    throw "Blind manifest does not prove train/val/test isolation."
}
$sourceLeases.Add((Open-ReadLease $blindRecords "candidate blind manifest"))
$sourceLeases.Add((Open-ReadLease $blindContractPath "candidate blind contract"))

$trainArgs = @(
    "-m", "transfer_receipt_ai.ocr_unified", "train",
    "--records", $blindRecords,
    "--dataset-root", $DatasetRoot,
    "--output", $trainingRoot,
    "--device", "cuda:0",
    "--architecture", "v13",
    "--image-height", "80",
    "--image-width", "512",
    "--base-channels", "32",
    "--numeric-hidden-size", "96",
    "--payment-hidden-size", "128",
    "--recipient-input-height", "128",
    "--recipient-input-width", "1536",
    "--recipient-value-left-trim", "$recipientValueLeftTrim",
    "--recipient-branch-channels", "16",
    "--recipient-hidden-size", "192",
    "--recipient-open-text-layers", "4",
    "--recipient-open-text-heads", "8",
    "--recipient-open-text-feedforward", "1536",
    "--recipient-open-text-dropout", "0.10",
    "--recipient-backbone", $requiredBackbone,
    "--recipient-train-augmentation", "robust_v2",
    "--recipient-train-splits", "train",
    "--recipient-low-confidence-threshold", "0.95",
    "--recipient-low-confidence-loss-weight", "0.50",
    "--recipient-confidence-curriculum-epochs", "10",
    "--recipient-tail-rare-character-max-support", "3",
    "--recipient-tail-rare-character-loss-weight", "1.5",
    "--recipient-tail-long-text-min-length", "9",
    "--recipient-tail-long-text-loss-weight", "1.5",
    "--recipient-only-fine-tune",
    "--init-checkpoint", $resolvedSeedCheckpoint,
    "--init-checkpoint-mode", $requiredInit,
    "--checkpoint-selection", "recipient_priority",
    "--checkpoint-min-amount-candidate-exact", "$amountFloor",
    "--checkpoint-min-time-candidate-exact", "$timeFloor",
    "--checkpoint-min-payment-candidate-exact", "$paymentFloor",
    "--amount-format-min-confidence", "0.80",
    "--payment-bank-prefix-min-support", "3",
    "--ctc-loss-weight", "1.0",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--validation-every", "$ValidationEvery",
    "--seed", "42",
    "--num-workers", "$NumWorkers",
    "--prefetch-factor", "$PrefetchFactor",
    "--train-progress-every", "$TrainProgressEvery",
    "--cuda-tf32",
    "--cudnn-benchmark"
)
if ($NumWorkers -gt 0) {
    $trainArgs += "--persistent-workers"
}
Invoke-Python $trainArgs "blind recipient v14 CUDA training"

$trainingSummaryPath = Join-Path $trainingRoot "training_summary.json"
$training = Read-Json $trainingSummaryPath
$fineTune = $training.fine_tune_policy
$runtime = $training.training_runtime
$initialization = $training.initialization
$requiredRecipientMap = "fresh_train_only_reinitialized_recipient_v1"
if ([string]$training.kind -ne "receipt_unified_field_reader_v13" `
    -or [int]$training.config.architecture_version -ne 13 `
    -or [string]$training.config.recipient_backbone -ne $requiredBackbone `
    -or [int]$training.config.recipient_open_text_layers -ne 4 `
    -or [Math]::Abs([double]$training.config.recipient_open_text_dropout - 0.10) -gt 0.000000001 `
    -or [Math]::Abs([double]$training.config.recipient_value_left_trim - $recipientValueLeftTrim) -gt 0.000000001 `
    -or [string]$initialization.mode -ne "parameter_only_recipient_visual_context_reinit" `
    -or [string]$initialization.source_kind -ne "receipt_unified_field_reader_v13" `
    -or [Math]::Abs([double]$initialization.source_config.recipient_value_left_trim - $recipientValueLeftTrim) -gt 0.000000001 `
    -or [string]$initialization.checkpoint_sha256 -ne (Get-Sha256 $resolvedSeedCheckpoint) `
    -or [string]$initialization.financial_label_policy.recipient_character_map.mode -ne $requiredRecipientMap `
    -or [string]$fineTune.mode -ne "recipient_only_v13" `
    -or [string]$fineTune.trainable_parameter_prefix -ne "recipient_" `
    -or [string]$fineTune.training_forward -ne "private_recipient_branch_only_v13" `
    -or [string]$training.recipient_train_split_policy.mode -ne "standard_train_only" `
    -or (($training.recipient_train_split_policy.splits -join ",") -ne "train") `
    -or [string]$training.recipient_train_augmentation_policy.mode -ne "robust_v2" `
    -or [string]$training.status_text_runtime_policy -ne $requiredStatusPolicy `
    -or $runtime.uses_cuda -ne $true `
    -or -not ([string]$runtime.device).StartsWith("cuda", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Training summary does not prove the blind v14 recipient recipe."
}
if ($fullCropSourceMode -and (
        [int]$runtime.num_workers -ne 4 `
        -or [int]$runtime.prefetch_factor -ne 2 `
        -or $runtime.persistent_workers -ne $true `
        -or [int]$runtime.validation_every -ne $ValidationEvery `
        -or $runtime.cuda_tf32_requested -ne $true `
        -or $runtime.cudnn_benchmark_requested -ne $true `
        -or [string]$runtime.device -ne "cuda:0" `
        -or $runtime.uses_cuda -ne $true `
        -or ([string]$runtime.cuda_device_name) -notmatch "4090")) {
    throw "Training summary does not prove the fixed full-crop residual runtime recipe."
}
if ([int]$training.field_counts.recipient_field.test -ne 0 `
    -or [int]$training.recipient_oov_by_split.test.records -ne 0) {
    throw "Training process observed test recipient labels; candidate is invalid."
}
$bestEpoch = [int]$training.best_checkpoint_epoch
$bestRows = @($training.records | Where-Object { [int]$_.epoch -eq $bestEpoch })
if ($bestRows.Count -ne 1 -or $bestRows[0].checkpoint_selection_eligible -ne $true) {
    throw "Training did not select exactly one val-eligible checkpoint."
}
$bestRecipient = [double]$bestRows[0].val_candidate_text_by_field.recipient_field.exact_match
$sourceLeases.Add((Open-ReadLease $trainingSummaryPath "candidate training summary"))
$sourceLeases.Add((Open-ReadLease $checkpoint "candidate best checkpoint"))
if ($fullCropSourceMode -and -not $Pilot) {
    Invoke-Python @(
        "-m", $sourceModule, "verify-candidate-training",
        "--summary", $trainingSummaryPath
    ) "full-crop candidate 60e recipient coverage verification"
}
if ($Pilot) {
    $epoch4Rows = @($training.records | Where-Object { [int]$_.epoch -eq 4 })
    $epoch8Rows = @($training.records | Where-Object { [int]$_.epoch -eq 8 })
    if ($epoch4Rows.Count -ne 1 -or $epoch8Rows.Count -ne 1 `
        -or $null -eq $epoch4Rows[0].val_candidate_text_by_field.recipient_field `
        -or $null -eq $epoch8Rows[0].val_candidate_text_by_field.recipient_field) {
        throw "Pilot stop audit requires recipient validation at epochs 4 and 8."
    }
    $epoch4Recipient = [double]$epoch4Rows[0].val_candidate_text_by_field.recipient_field.exact_match
    $epoch8Recipient = [double]$epoch8Rows[0].val_candidate_text_by_field.recipient_field.exact_match
    $pilotGain = $epoch8Recipient - $epoch4Recipient
    if ($bestRecipient -lt $pilotMinimumBestRecipient) {
        throw ("PILOT STOP: best val recipient {0:P2} is below {1:P2}; do not run 60 epochs." -f `
            $bestRecipient, $pilotMinimumBestRecipient)
    }
    if ($pilotGain -lt $pilotMinimumEpoch4To8Gain) {
        throw ("PILOT STOP: epoch4-to-8 gain {0:P2} is below {1:P2}; do not run 60 epochs." -f `
            $pilotGain, $pilotMinimumEpoch4To8Gain)
    }
    if ($fullCropSourceMode) {
        Invoke-Python @(
            "-m", $sourceModule, "seal-candidate-pilot",
            "--candidate-root", $OutputRoot,
            "--source-contract", $FullCropSourceContract,
            "--full-records", $FullRecords,
            "--output-evidence", $candidatePilotEvidencePath
        ) "full-crop residual candidate-pilot sealing"
        $sealedPilot = Read-Json $candidatePilotEvidencePath
        if ([string]$sealedPilot.kind -ne "receipt_recipient_v14_full_crop_residual_pilot_v1" `
            -or $sealedPilot.analysis_only -ne $true `
            -or $sealedPilot.production_route_authorized -ne $false `
            -or $sealedPilot.test_opened -ne $false `
            -or $sealedPilot.onnx_exported -ne $false `
            -or $sealedPilot.passed -ne $true) {
            throw "Fresh residual candidate pilot could not be sealed as analysis-only evidence."
        }
    }
    Write-Host "PILOT PASS: val trend justifies one fresh 60-epoch train/val-only run."
    Write-Host ("  best={0:P2}; epoch4={1:P2}; epoch8={2:P2}; gain={3:P2}" -f `
        $bestRecipient, $epoch4Recipient, $epoch8Recipient, $pilotGain)
    Write-Host "  test remains unopened; use a new OutputRoot for the full candidate."
    if ($fullCropSourceMode) {
        Write-Host "  candidate_pilot_evidence=$candidatePilotEvidencePath"
    }
    exit 0
}
if ($bestRecipient -le $recipientFloor) {
    throw ("Val recipient exact {0:P2} is not strictly above {1:P2}; test remains unopened." -f $bestRecipient, $recipientFloor)
}
Require-File $checkpoint "best recipient v14 checkpoint"

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "export",
    "--checkpoint", $checkpoint,
    "--output", $model
) "recipient v14 ONNX export"

Invoke-Python @(
    "-m", "transfer_receipt_ai.ocr_unified", "evaluate",
    "--model", $model,
    "--records", $blindRecords,
    "--dataset-root", $DatasetRoot,
    "--split", "val",
    "--output", $validationRoot,
    "--device", "cuda:0",
    "--min-amount-exact-match", "$amountFloor",
    "--min-time-exact-match", "$timeFloor",
    "--min-payment-exact-match", "$paymentFloor",
    "--min-recipient-exact-match", "$recipientFloor",
    "--min-status-exact-match", "$statusTextFloor",
    "--max-non-success-to-success", "0",
    "--progress-every", "250"
) "recipient v14 val-only ONNX gate"

$validation = Read-Json $validationSummaryPath
$sourceLeases.Add((Open-ReadLease $validationSummaryPath "candidate validation summary"))
if ($fullCropSourceMode) {
    Invoke-Python @(
        "-m", $sourceModule, "verify-candidate-val",
        "--summary", $validationSummaryPath
    ) "full-crop candidate ONNX val recipient coverage verification"
}
$statusRaw = [double]$validation.by_field.transfer_status.ctc_raw_exact_match
if ($validation.providers -notcontains "CUDAExecutionProvider" `
    -or $validation.acceptance.passed -ne $true `
    -or [string]$validation.status_text_policy.runtime_policy -ne $requiredStatusPolicy `
    -or (Get-RawExact $validation "amount") -lt $amountFloor `
    -or (Get-RawExact $validation "time") -lt $timeFloor `
    -or (Get-RawExact $validation "payment_method_field") -lt $paymentFloor `
    -or (Get-RawExact $validation "recipient_field") -le $recipientFloor `
    -or $statusRaw -lt $statusTextFloor `
    -or [int]$validation.by_field.transfer_status.non_success_to_success -ne 0) {
    throw "Val-only ONNX evidence did not pass fixed delivery floors."
}

$modelContractPath = [IO.Path]::ChangeExtension($model, ".contract.json")
$modelLabelsPath = [IO.Path]::ChangeExtension($model, ".labels.json")
Require-File $modelContractPath "candidate ONNX contract"
Require-File $modelLabelsPath "candidate ONNX labels"
$sourceLeases.Add((Open-ReadLease $model "candidate ONNX model"))
$sourceLeases.Add((Open-ReadLease $modelContractPath "candidate ONNX contract"))
$sourceLeases.Add((Open-ReadLease $modelLabelsPath "candidate ONNX labels"))
$sourceRouteEvidence = if ($fullCropSourceMode) {
    [ordered]@{
        mode = $sourceRouteMode
        recipient_value_left_trim = $recipientValueLeftTrim
        source_contract = $FullCropSourceContract
        source_contract_sha256 = Get-Sha256 $FullCropSourceContract
        source_subject_id = [string]$sourceContract.source_subject_id
        full_crop_pilot_root = $FullCropPilotRoot
        source_checkpoint = $resolvedSeedCheckpoint
        source_checkpoint_sha256 = Get-Sha256 $resolvedSeedCheckpoint
        candidate_pilot_evidence = $CandidatePilotEvidence
        candidate_pilot_evidence_sha256 = Get-Sha256 $CandidatePilotEvidence
        candidate_pilot_subject_id = [string]$candidatePilotContract.candidate_pilot_subject_id
    }
}
else {
    [ordered]@{
        mode = $sourceRouteMode
        recipient_value_left_trim = $recipientValueLeftTrim
        source_checkpoint = $resolvedSeedCheckpoint
        source_checkpoint_sha256 = Get-Sha256 $resolvedSeedCheckpoint
    }
}
$trainingEvidence = [ordered]@{
    summary = $trainingSummaryPath
    summary_sha256 = Get-Sha256 $trainingSummaryPath
    best_epoch = $bestEpoch
}
if ($fullCropSourceMode) {
    $trainingEvidence["recipe"] = $trainingRecipePath
    $trainingEvidence["recipe_sha256"] = Get-Sha256 $trainingRecipePath
}
$evidence = [ordered]@{
    schema_version = 1
    kind = "receipt_recipient_v14_blind_candidate_v1"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    analysis_only = $true
    production_route_authorized = $false
    split_policy = [ordered]@{
        optimizer_supervision = @("train")
        checkpoint_selection = @("val")
        final_gate_only = @("test")
        test_evaluated = $false
        blind_contract = $blindContractPath
        blind_contract_sha256 = Get-Sha256 $blindContractPath
    }
    full_manifest = [IO.Path]::GetFullPath($FullRecords)
    full_manifest_sha256 = Get-Sha256 $FullRecords
    blind_manifest = $blindRecords
    blind_manifest_sha256 = Get-Sha256 $blindRecords
    source_route = $sourceRouteEvidence
    candidate = [ordered]@{
        checkpoint = $checkpoint
        checkpoint_sha256 = Get-Sha256 $checkpoint
        model = $model
        model_sha256 = Get-Sha256 $model
        contract = $modelContractPath
        contract_sha256 = Get-Sha256 $modelContractPath
        labels = $modelLabelsPath
        labels_sha256 = Get-Sha256 $modelLabelsPath
        architecture_version = 13
        recipe_name = "recipient_v14_residual_positional_transformer"
        backbone = $requiredBackbone
    }
    training = $trainingEvidence
    val_evaluation = [ordered]@{
        summary = $validationSummaryPath
        summary_sha256 = Get-Sha256 $validationSummaryPath
        amount = Get-RawExact $validation "amount"
        time = Get-RawExact $validation "time"
        payment_method_field = Get-RawExact $validation "payment_method_field"
        recipient_field = Get-RawExact $validation "recipient_field"
        recipient_records = [int]$validation.by_field.recipient_field.records
        recipient_exact_matches = [int]$validation.by_field.recipient_field.raw_exact_matches
        recipient_candidate_coverage = 1.0
        visible_transfer_status_cjk_text = $statusRaw
        status_non_success_to_success = [int]$validation.by_field.transfer_status.non_success_to_success
    }
    fixed_floors = [ordered]@{
        amount = $amountFloor
        time = $timeFloor
        payment_method_field = $paymentFloor
        recipient_field = $recipientFloor
        visible_transfer_status_cjk_text = $statusTextFloor
    }
}
$evidenceJson = ($evidence | ConvertTo-Json -Depth 12) + "`n"
Assert-NoReparseChain $OutputRoot "Recipient v14 evidence output"
Write-CreateNewUtf8 $evidencePath $evidenceJson "candidate evidence"

Write-Host ""
Write-Host "PASS: val-selected recipient v14 candidate is sealed; test remains unopened."
Write-Host "  candidate_evidence=$evidencePath"
Write-Host "  model=$model"
Write-Host "  trusted_full_manifest_sha256=$($evidence.full_manifest_sha256)"
Write-Host "  next=scripts\receipt-ocr-recipient-v14-final-gate-4090.ps1"
