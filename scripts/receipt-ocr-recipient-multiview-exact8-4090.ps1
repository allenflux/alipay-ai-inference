[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FullRecords,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$FullCropPilotRoot,
    [Parameter(Mandatory = $true)]
    [string]$FullCropSourceContract,
    [Parameter(Mandatory = $true)]
    [string]$CandidatePilotEvidence,
    [Parameter(Mandatory = $true)]
    [string]$FailureEvidence,
    [Parameter(Mandatory = $true)]
    [string]$FailureAttemptRegistry,
    [Parameter(Mandatory = $true)]
    [string]$Fixed2OverlayContract,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$exact8Module = "transfer_receipt_ai.recipient_multiview_exact8"

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

function Require-FreshOutput([string]$Path) {
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing -or (Test-Path -LiteralPath $Path)) {
        throw "Refusing to reuse fixed2 exact8 output: $Path"
    }
    $parent = Split-Path -Parent $Path
    Require-Directory $parent "fixed2 exact8 output parent"
    Assert-NoReparseChain $Path "Fixed2 exact8 output"
}

function Open-ReadLease([string]$Path, [string]$Description) {
    Require-File $Path $Description
    return [IO.File]::Open(
        [IO.Path]::GetFullPath($Path),
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read)
}

function Protect-AuditRoot([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Exact8 audit root must not be a reparse point: $Path"
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $identity) {
        throw "Unable to resolve the current Windows identity for the exact8 registry."
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

function Invoke-Inspection([string[]]$Arguments, [string]$Description) {
    $text = (& $pythonExe @Arguments) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
    try {
        $payload = $text | ConvertFrom-Json
    }
    catch {
        throw "$Description did not return one JSON object: $text"
    }
    return [pscustomobject]@{ Text = $text; Payload = $payload }
}

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $FullRecords "full v13 manifest"
Require-Directory $DatasetRoot "original recipient dataset root"
Require-Directory $FullCropPilotRoot "original full-crop pilot root"
Require-File $FullCropSourceContract "full-crop source contract"
Require-File $CandidatePilotEvidence "A8 candidate-pilot evidence"
Require-File $FailureEvidence "fresh60 failure evidence"
Require-Directory $FailureAttemptRegistry "fresh60 attempt registry"
Require-File $Fixed2OverlayContract "fixed2 overlay contract"

$FullRecords = [IO.Path]::GetFullPath($FullRecords)
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$FullCropPilotRoot = [IO.Path]::GetFullPath($FullCropPilotRoot)
$FullCropSourceContract = [IO.Path]::GetFullPath($FullCropSourceContract)
$CandidatePilotEvidence = [IO.Path]::GetFullPath($CandidatePilotEvidence)
$FailureEvidence = [IO.Path]::GetFullPath($FailureEvidence)
$FailureAttemptRegistry = [IO.Path]::GetFullPath($FailureAttemptRegistry)
$Fixed2OverlayContract = [IO.Path]::GetFullPath($Fixed2OverlayContract)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

foreach ($inputPath in @(
        $FullRecords,
        $DatasetRoot,
        $FullCropPilotRoot,
        $FullCropSourceContract,
        $CandidatePilotEvidence,
        $FailureEvidence,
        $FailureAttemptRegistry,
        $Fixed2OverlayContract)) {
    Assert-NoReparseChain $inputPath "Exact8 authority input"
}
Require-FreshOutput $OutputRoot

$focusedTests = @(
    (Join-Path $repoRoot "tests\test_recipient_fixed2_teacher_export.py"),
    (Join-Path $repoRoot "tests\test_recipient_multiview_exact8.py"),
    (Join-Path $repoRoot "tests\test_recipient_multiview_overlay.py"),
    (Join-Path $repoRoot "tests\test_recipient_v14_failure_attestor.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_candidate_source.py"),
    (Join-Path $repoRoot "tests\test_recipient_full_crop_continuation.py")
)
foreach ($testPath in $focusedTests) {
    Require-File $testPath "fixed2 exact8 contract test"
}

$gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0 -or [string]$gpuRows[0] -notmatch "4090") {
    throw "CUDA device 0 must be an RTX 4090. Observed: $($gpuRows -join '; ')"
}

Write-Host "receipt_recipient_multiview_fixed2_exact8 preflight"
Write-Host "  views=standard,fixed_value; selector=context-distinct fixed-value pair anti-repeat v2; train multiplier=1"
Write-Host "  initialization=original full-crop pilot best via fresh visual-context reinit"
Write-Host "  epochs=8; validation=every epoch; held-out final gate remains unopened"
Write-Host ("  GPU: {0}" -f ($gpuRows -join "; "))

& $pythonExe -m pytest -q @focusedTests
if ($LASTEXITCODE -ne 0) {
    throw "Fixed2 exact8 focused tests failed with exit code $LASTEXITCODE"
}

$inspectArguments = @(
    "-m", $exact8Module, "inspect",
    "--full-records", $FullRecords,
    "--dataset-root", $DatasetRoot,
    "--full-crop-pilot-root", $FullCropPilotRoot,
    "--source-contract", $FullCropSourceContract,
    "--candidate-pilot-evidence", $CandidatePilotEvidence,
    "--failure-evidence", $FailureEvidence,
    "--failure-attempt-registry", $FailureAttemptRegistry,
    "--overlay-contract", $Fixed2OverlayContract,
    "--require-unconsumed"
)
$opening = Invoke-Inspection $inspectArguments "opening exact8 authority inspection"
$inspection = $opening.Payload
if ([string]$inspection.kind -ne "receipt_recipient_multiview_fixed2_exact8_subject_v2" `
    -or $inspection.analysis_only -ne $true `
    -or $inspection.production_route_authorized -ne $false `
    -or $inspection.test_opened -ne $false `
    -or $inspection.onnx_exported -ne $false `
    -or [string]$inspection.route_subject_id -notmatch "^[0-9a-f]{64}$" `
    -or [string]$inspection.attempt_id -notmatch "^[0-9a-f]{64}$") {
    throw "Exact8 inspection did not return the fixed analysis-only subject."
}

$sourceLeases = [Collections.Generic.List[IDisposable]]::new()
try {
    foreach ($guardPath in @($inspection.guard_paths)) {
        $sourceLeases.Add((Open-ReadLease ([string]$guardPath) "exact8 guarded authority artifact"))
    }
    foreach ($guardDirectory in @($inspection.guard_directories)) {
        Require-Directory ([string]$guardDirectory) "exact8 guarded authority directory"
        # Directory identity is reverified after training. The sealed
        # decision explicitly records that selected train images and all
        # unchanged validation images do not have continuous read leases
        # (the accepted swap/restore risk).
        Assert-NoReparseChain ([string]$guardDirectory) "exact8 guarded authority directory"
    }
    $leased = Invoke-Inspection $inspectArguments "leased exact8 authority reinspection"
    if ([string]$leased.Text -cne [string]$opening.Text) {
        throw "Exact8 authority changed between opening and leased reinspection."
    }

    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    if ([string]::IsNullOrWhiteSpace($commonData)) {
        throw "Windows CommonApplicationData is unavailable; exact8 cannot be guarded one-shot."
    }
    $receiptRoot = Join-Path $commonData "ReceiptAI"
    $auditRoot = Join-Path $receiptRoot "recipient-v14-multiview-fixed2-training-v1"
    Assert-NoReparseChain $auditRoot "Exact8 attempt registry"
    $auditEntry = Get-Item -LiteralPath $auditRoot -Force -ErrorAction SilentlyContinue
    if ($null -ne $auditEntry) {
        if (-not $auditEntry.PSIsContainer) {
            throw "Exact8 attempt registry is not a directory: $auditRoot"
        }
        $registryEntries = @(Get-ChildItem -LiteralPath $auditRoot -Force)
        if ($registryEntries.Count -ne 0) {
            throw "Fixed2 exact8 lineage is already consumed or its dedicated registry is not empty: $auditRoot"
        }
    }
    $attemptPath = Join-Path $auditRoot (([string]$inspection.attempt_id) + ".attempt.json")

    if ($CheckOnly) {
        Write-Host "receipt_recipient_multiview_fixed2_exact8 preflight=passed"
        Write-Host ("  route_subject_id={0}" -f [string]$inspection.route_subject_id)
        Write-Host "  fixed ProgramData one-shot registry is absent or empty"
        Write-Host "  no attempt lock or output was created"
        exit 0
    }

    # The wrapper protects both parent levels but never creates the marker.
    # Python anchors ProgramData and revalidates its ACL descriptor, but the
    # elevated-admin ProgramData DELETE_CHILD capability is explicitly outside
    # the threat model. ReceiptAI/registry effective delete denials remain gates.
    # Python run owns the atomic CreateNew, verifies the marker's inherited deny, and holds all
    # leases through training and decision publication.
    New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
    Assert-NoReparseChain $receiptRoot "Exact8 ReceiptAI root"
    Assert-NoReparseChain $auditRoot "Exact8 attempt registry"
    Protect-AuditRoot $receiptRoot
    Protect-AuditRoot $auditRoot
    $attemptEntry = Get-Item -LiteralPath $attemptPath -Force -ErrorAction SilentlyContinue
    if ($null -ne $attemptEntry -or (Test-Path -LiteralPath $attemptPath)) {
        throw "Fixed2 exact8 one-shot subject is already consumed: $attemptPath"
    }

    $runArguments = @(
        "-m", $exact8Module, "run",
        "--full-records", $FullRecords,
        "--dataset-root", $DatasetRoot,
        "--full-crop-pilot-root", $FullCropPilotRoot,
        "--source-contract", $FullCropSourceContract,
        "--candidate-pilot-evidence", $CandidatePilotEvidence,
        "--failure-evidence", $FailureEvidence,
        "--failure-attempt-registry", $FailureAttemptRegistry,
        "--overlay-contract", $Fixed2OverlayContract,
        "--output-root", $OutputRoot,
        "--attempt-lock", $attemptPath
    )
    & $pythonExe @runArguments
    $runExit = $LASTEXITCODE
    if (Test-Path -LiteralPath $attemptPath -PathType Leaf) {
        $sourceLeases.Add((Open-ReadLease $attemptPath "Python-created exact8 one-shot attempt lock"))
    }

    $decisionPath = Join-Path $OutputRoot "recipient_multiview_exact8_decision.json"
    if (Test-Path -LiteralPath $decisionPath -PathType Leaf) {
        $sourceLeases.Add((Open-ReadLease $decisionPath "exact8 decision evidence"))
        $decision = (Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8) | ConvertFrom-Json
        foreach ($artifactProperty in $decision.training_artifacts.PSObject.Properties) {
            $artifactPath = [string]$artifactProperty.Value.path
            if ([IO.Path]::GetFullPath($artifactPath) -ne [IO.Path]::GetFullPath($attemptPath)) {
                $sourceLeases.Add((Open-ReadLease $artifactPath ("exact8 output artifact " + $artifactProperty.Name)))
            }
        }
        $verifyArguments = @(
            "-m", $exact8Module, "verify-decision",
            "--full-records", $FullRecords,
            "--dataset-root", $DatasetRoot,
            "--full-crop-pilot-root", $FullCropPilotRoot,
            "--source-contract", $FullCropSourceContract,
            "--candidate-pilot-evidence", $CandidatePilotEvidence,
            "--failure-evidence", $FailureEvidence,
            "--failure-attempt-registry", $FailureAttemptRegistry,
            "--overlay-contract", $Fixed2OverlayContract,
            "--output-root", $OutputRoot,
            "--attempt-lock", $attemptPath,
            "--decision", $decisionPath
        )
        & $pythonExe @verifyArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Independent exact8 decision verification failed with exit code $LASTEXITCODE"
        }
        Write-Host ("fixed2 exact8 passed={0}; decision={1}" -f $decision.passed, $decision.decision)
        Write-Host ("  best={0}/{1}; epoch8={2}/{1}" -f `
            $decision.observed.best_matches,
            $decision.observed.recipient_denominator,
            $decision.observed.epoch8_matches)
        Write-Host ("  evidence={0}" -f $decisionPath)
    }
    elseif ($runExit -eq 0) {
        throw "Exact8 returned success without sealed decision evidence."
    }
    exit $runExit
}
finally {
    foreach ($lease in $sourceLeases) {
        if ($null -ne $lease) { $lease.Dispose() }
    }
}
