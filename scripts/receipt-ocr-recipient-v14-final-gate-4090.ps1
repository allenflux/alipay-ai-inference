[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateEvidence,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9A-Fa-f]{64}$")]
    [string]$TrustedFullManifestSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fixed delivery floors. They are constants, not caller parameters.
$amountFloor = 0.7885
$timeFloor = 0.9840
$paymentFloor = 0.9325
$recipientFloor = 0.90
$statusTextFloor = 0.90
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-cu126\Scripts\python.exe"
$inspectionModule = "transfer_receipt_ai.recipient_final_gate"

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

function Invoke-Inspection([string[]]$Arguments, [string]$Description) {
    $text = (& $pythonExe -m $inspectionModule @Arguments) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
    try {
        return ($text | ConvertFrom-Json)
    }
    catch {
        throw "$Description did not return strict JSON. $($_.Exception.Message)"
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
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

function Open-ReadLease([string]$Path, [string]$Description) {
    Require-File $Path $Description
    # FileShare.Read lets the evaluator read the same immutable bytes while
    # denying write/delete/rename until post-test rehashing has completed.
    return [IO.File]::Open(
        [IO.Path]::GetFullPath($Path),
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read)
}

function Get-VerifiedRate([object]$Value, [string]$Description) {
    if ($null -eq $Value) { throw "Missing ${Description}." }
    $number = [double]$Value
    if ([double]::IsNaN($number) `
        -or [double]::IsInfinity($number) `
        -or $number -lt 0.0 `
        -or $number -gt 1.0) {
        throw "Invalid ${Description}: $Value"
    }
    return $number
}

function Protect-AuditRoot([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Final-gate audit root must not be a reparse point: $Path"
    }
    # Deny routine deletion to the account that runs the gate. This is a local
    # accident/tuning guard, not a claim against a malicious administrator,
    # who can take ownership and rewrite local ACLs.
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $identity) { throw "Unable to resolve the current Windows identity for audit ACLs." }
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

Require-File $pythonExe "CUDA virtual-environment Python"
Require-File $CandidateEvidence "sealed val candidate evidence"
Require-Directory $DatasetRoot "recipient crop root"
$CandidateEvidence = [IO.Path]::GetFullPath($CandidateEvidence)
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$TrustedFullManifestSha256 = $TrustedFullManifestSha256.ToLowerInvariant()
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse final test gate output: $OutputRoot"
}

$inspectArguments = @(
    "inspect",
    "--candidate-evidence", $CandidateEvidence,
    "--trusted-full-manifest-sha256", $TrustedFullManifestSha256
)
$inspection = Invoke-Inspection $inspectArguments "recipient v14 candidate cryptographic inspection"
if ([string]$inspection.kind -ne "receipt_recipient_v14_final_gate_subject_v1" `
    -or [string]$inspection.full_manifest_sha256 -ne $TrustedFullManifestSha256 `
    -or [string]$inspection.gate_subject_id -notmatch "^[0-9a-f]{64}$" `
    -or [string]$inspection.evidence_identity -notmatch "^[0-9a-f]{64}$") {
    throw "Candidate inspection did not produce a trusted path-independent gate identity."
}

$leaseNames = @(
    "candidate_evidence",
    "full_manifest",
    "blind_manifest",
    "blind_contract",
    "model",
    "contract",
    "labels",
    "checkpoint",
    "training_summary",
    "val_summary"
)
$leases = [Collections.Generic.List[IDisposable]]::new()
try {
    foreach ($name in $leaseNames) {
        $path = if ($name -eq "candidate_evidence") {
            $CandidateEvidence
        }
        else {
            [string]$inspection.paths.PSObject.Properties[$name].Value
        }
        $leases.Add((Open-ReadLease $path $name))
    }
    foreach ($sourceGuard in @($inspection.source_guard_artifacts)) {
        $leases.Add((Open-ReadLease ([string]$sourceGuard.path) ("source guard " + [string]$sourceGuard.name)))
    }

    # Recompute every identity while immutable read leases are held. Any
    # between-inspection mutation fails before the test-attempt lock is made.
    $secondInspection = Invoke-Inspection $inspectArguments "leased candidate reinspection"
    foreach ($name in @(
            "gate_subject_id",
            "evidence_identity",
            "candidate_evidence_sha256",
            "model_sha256",
            "full_manifest_sha256",
            "contract_sha256",
            "labels_sha256",
            "source_guard_digest"
        )) {
        if ([string]$secondInspection.PSObject.Properties[$name].Value -ne `
            [string]$inspection.PSObject.Properties[$name].Value) {
            throw "Candidate bytes changed during final-gate inspection ($name)."
        }
    }
    $inspection = $secondInspection

    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    if ([string]::IsNullOrWhiteSpace($commonData)) {
        throw "Windows CommonApplicationData is unavailable; no persistent final-gate registry can be used."
    }
    $auditRoot = Join-Path $commonData "ReceiptAI\recipient-v14-final-gate-v1"
    New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
    Protect-AuditRoot $auditRoot

    $gateSubjectId = [string]$inspection.gate_subject_id
    $evidenceIdentity = [string]$inspection.evidence_identity
    $attemptLock = Join-Path $auditRoot ("$gateSubjectId.attempt.json")
    $evidenceLock = Join-Path $auditRoot ("$gateSubjectId.$evidenceIdentity.evidence.json")
    $resultRegistry = Join-Path $auditRoot ("$gateSubjectId.result.json")
    $attemptPayload = [ordered]@{
        schema_version = 1
        kind = "receipt_recipient_v14_persistent_test_attempt_v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        gate_subject_id = $gateSubjectId
        evidence_identity = $evidenceIdentity
        candidate_evidence_sha256 = [string]$inspection.candidate_evidence_sha256
        model_sha256 = [string]$inspection.model_sha256
        full_manifest_sha256 = [string]$inspection.full_manifest_sha256
        contract_sha256 = [string]$inspection.contract_sha256
        labels_sha256 = [string]$inspection.labels_sha256
        threat_model = "persistent local guard against accidental/path-copy reruns; not malicious administrator deletion"
    }
    $attemptJson = ($attemptPayload | ConvertTo-Json -Depth 8) + "`n"

    # The subject lock excludes evidence paths and evidence hashes by design:
    # copying or editing metadata cannot create a new attempt for the same
    # model/sidecars/full-test-manifest bytes. Evidence identity is separately
    # registered for audit. Both files use atomic FileMode.CreateNew.
    Write-CreateNewUtf8 $attemptLock $attemptJson "one-shot subject lock"
    Write-CreateNewUtf8 $evidenceLock $attemptJson "one-shot evidence registry"

    # The lock is now permanent. A crash, failed inference, missing summary or
    # rejected metric is still an observed test attempt and can never become a
    # new tuning iteration for this gate_subject_id.
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
    $inspectionPath = Join-Path $OutputRoot "pretest_cryptographic_inspection.json"
    Write-CreateNewUtf8 `
        $inspectionPath (($inspection | ConvertTo-Json -Depth 12) + "`n") `
        "pretest cryptographic inspection"
    # Keep the exact intermediate identity document immutable until its
    # post-test verification and result registry write have both completed.
    $leases.Add((Open-ReadLease $inspectionPath "pretest cryptographic inspection"))

    $model = [string]$inspection.paths.model
    $fullRecords = [string]$inspection.paths.full_manifest
    $evaluationRoot = Join-Path $OutputRoot "onnx-test-gpu-once"
    & $pythonExe -m transfer_receipt_ai.ocr_unified evaluate `
        --model $model `
        --records $fullRecords `
        --dataset-root $DatasetRoot `
        --split test `
        --output $evaluationRoot `
        --device cuda:0 `
        --min-amount-exact-match $amountFloor `
        --min-time-exact-match $timeFloor `
        --min-payment-exact-match $paymentFloor `
        --min-recipient-exact-match $recipientFloor `
        --min-status-exact-match $statusTextFloor `
        --max-non-success-to-success 0 `
        --progress-every 250
    $evaluateExit = $LASTEXITCODE
    $summaryPath = Join-Path $evaluationRoot "summary.json"

    $failures = @()
    $verified = $null
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        $failures += "missing_test_summary"
    }
    else {
        try {
            $verified = Invoke-Inspection @(
                "verify-test",
                "--inspection", $inspectionPath,
                "--summary", $summaryPath
            ) "one-shot test summary verification"
            if ([string]$verified.kind -ne "receipt_recipient_v14_verified_test_summary_v1" `
                -or [string]$verified.gate_subject_id -ne $gateSubjectId `
                -or [string]$verified.evidence_identity -ne $evidenceIdentity) {
                throw "Verified test summary is bound to a different gate subject or evidence identity."
            }
            foreach ($failure in @($verified.failures)) {
                if (-not [string]::IsNullOrWhiteSpace([string]$failure)) {
                    $failures += [string]$failure
                }
            }
        }
        catch {
            $failures += "test_summary_verification_failed"
            $failures += $_.Exception.Message
        }
    }
    if ($evaluateExit -ne 0) { $failures += "evaluation_exit=$evaluateExit" }

    $metrics = [ordered]@{
        amount = $null
        time = $null
        payment_method_field = $null
        recipient_field = $null
        recipient_records = $null
        recipient_exact_matches = $null
        recipient_candidate_coverage = $null
        visible_transfer_status_cjk_text = $null
        status_non_success_to_success = $null
    }
    if ($null -ne $verified) {
        $metrics.amount = Get-VerifiedRate $verified.metrics.amount "verified amount exact"
        $metrics.time = Get-VerifiedRate $verified.metrics.time "verified time exact"
        $metrics.payment_method_field = Get-VerifiedRate `
            $verified.metrics.payment_method_field "verified payment exact"
        $metrics.recipient_field = Get-VerifiedRate `
            $verified.metrics.recipient_field "verified recipient exact"
        $metrics.recipient_records = [int]$verified.recipient_records
        $metrics.recipient_exact_matches = [int]$verified.recipient_exact_matches
        $metrics.recipient_candidate_coverage = Get-VerifiedRate `
            $verified.recipient_candidate_coverage "verified recipient candidate coverage"
        $metrics.visible_transfer_status_cjk_text = Get-VerifiedRate `
            $verified.metrics.visible_transfer_status_cjk_text "verified status OCR exact"
        $metrics.status_non_success_to_success = [int]$verified.status_non_success_to_success
        if ($metrics.recipient_field -le $recipientFloor `
            -and $failures -notcontains "recipient_field_not_strictly_above_floor") {
            $failures += "recipient_field_not_strictly_above_floor"
        }
    }
    $passed = $failures.Count -eq 0 -and $null -ne $verified -and $verified.passed -eq $true
    $gatePath = Join-Path $OutputRoot "recipient_v14_final_test.json"
    $gate = [ordered]@{
        schema_version = 1
        kind = "receipt_recipient_v14_one_shot_test_gate_v2"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        gate_subject_id = $gateSubjectId
        evidence_identity = $evidenceIdentity
        candidate_evidence = $CandidateEvidence
        candidate_evidence_sha256 = [string]$inspection.candidate_evidence_sha256
        trusted_full_manifest_sha256 = $TrustedFullManifestSha256
        attempt_lock = $attemptLock
        evidence_registry = $evidenceLock
        test_attempt_number = 1
        checkpoint_selection_used_test = $false
        passed = $passed
        failures = $failures
        summary = if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { $summaryPath } else { $null }
        summary_sha256 = if ($null -ne $verified) { [string]$verified.summary_sha256 } else { $null }
        metrics = $metrics
        fixed_floors = [ordered]@{
            amount = $amountFloor
            time = $timeFloor
            payment_method_field = $paymentFloor
            recipient_field = $recipientFloor
            visible_transfer_status_cjk_text = $statusTextFloor
        }
        threat_model_boundary = (
            "The machine-wide ACL registry prevents accidental/path-copy reruns. " +
            "It does not claim resistance to a malicious administrator who can take ownership or delete local state."
        )
    }
    $gateJson = ($gate | ConvertTo-Json -Depth 12) + "`n"
    Write-CreateNewUtf8 $gatePath $gateJson "final test result"
    Write-CreateNewUtf8 $resultRegistry $gateJson "persistent final test result registry"
    if (-not $passed) {
        throw ("ONE-SHOT TEST FAILED; do not tune on this result. failures=" + ($failures -join ","))
    }
    Write-Host "PASS: one-shot held-out test gate accepted recipient v14."
    Write-Host ("  recipient={0:P2}; amount={1:P2}; time={2:P2}; payment={3:P2}; status={4:P2}" -f `
        $metrics.recipient_field, $metrics.amount, $metrics.time, `
        $metrics.payment_method_field, $metrics.visible_transfer_status_cjk_text)
    Write-Host "  gate_subject_id=$gateSubjectId"
    Write-Host "  persistent_lock=$attemptLock"
    Write-Host "  evidence=$gatePath"
}
finally {
    foreach ($lease in $leases) {
        try { $lease.Dispose() } catch { }
    }
}
