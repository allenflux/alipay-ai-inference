from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_ROOT = ROOT / "dotnet" / "ReceiptMlNet.Cli" / "DeliveryScripts"
SINGLE = DELIVERY_ROOT / "run-receipt-single-cpu.ps1"
BATCH = DELIVERY_ROOT / "run-receipt-batch-cpu.ps1"
README = DELIVERY_ROOT / "README-CPU.md"
ENTRYPOINTS = (SINGLE, BATCH)


def _source(path: Path) -> str:
    payload = path.read_bytes()
    assert all(byte < 128 for byte in payload), f"{path.name} must remain ASCII-only"
    return payload.decode("ascii")


def _function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}(")
    end = source.index(f"function {next_name}(", start)
    return source[start:end]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_entrypoint_requires_the_formal_hybrid_cpu_package(entrypoint: Path) -> None:
    source = _source(entrypoint)

    assert "receipt_mlnet_hybrid_recipient_delivery_package_v1" in source
    assert "receipt_mlnet_hybrid_recipient_package_validation_v1" in source
    assert 'validation_scope -ne "full_val_end_to_end_scored_cpu"' in source
    assert 'runtime_device -ne "cpu"' in source
    assert 'include_device_model -ne $true' in source
    assert 'validation.inference_summary.unified_provider -ne "cpu"' in source
    assert 'validation.inference_summary.paddle_ocr_provider -ne "cpu"' in source
    assert 'end_to_end_evaluation.performed -ne $true' in source
    assert 'end_to_end_evaluation.status -ne "accepted"' in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_entrypoint_verifies_contained_recipient_ppocr_artifacts(entrypoint: Path) -> None:
    source = _source(entrypoint)

    for token in (
        "function Resolve-ContainedPackageDirectory",
        "function Read-PaddleDeliveryFileEvidence",
        "function Read-PaddleRecipientEvidence",
        "function Assert-DeclaredPaddleRecipientArtifact",
        'bundle_path -ne "models/recipient-ppocr"',
        'contract_path -ne "models/recipient-ppocr/paddle_ocr_delivery.contract.json"',
        'kind -ne "paddle_ocr_v2_delivery"',
        'foreach ($role in @("det", "cls", "rec"))',
        "Paddle OCR ${Description} escapes its contained delivery bundle",
        "(Get-Sha256 $target) -ne $expectedSha256",
        "(Get-Item -LiteralPath $target).Length -ne $expectedBytes",
        "package_size_bytes is inconsistent",
        "delivery contract changed during verification",
        'foreach ($dependency in @("Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files"))',
        "bundle is not closed over its contract, det/cls/rec ONNX files, and dictionary",
        'Properties["recipient_ppocr"]',
    ):
        assert token in source

    assert source.count("Assert-DeclaredPaddleRecipientArtifact") >= 2
    assert "$configDeclaration.Value" in source
    assert "$validationDeclaration.Value" in source
    assert "ocr_bundle = [string]$PaddleEvidence.ContractFileName" in source
    assert "ocr_bundle_contract_sha256 = [string]$PaddleEvidence.ContractSha256" in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_entrypoint_runs_the_complete_hybrid_route_on_cpu(entrypoint: Path) -> None:
    source = _source(entrypoint)

    assert "--detector" in source
    assert "--device-model" in source
    assert '"hybrid-recipient"' in source or "--ocr hybrid-recipient" in source
    assert "--ocr-model" in source
    assert "--ocr-bundle" in source
    assert "$paddleEvidence.BundlePath" in source
    assert 'summary.unified_provider -ne "cpu"' in source
    assert 'summary.paddle_ocr_provider -ne "cpu"' in source
    assert "detector + device classifier + v13 OCR + recipient PP-OCR (pure ONNX)" in source
    assert "complete pure-ONNX detector/device/v13/PP-OCR CPU pipeline" in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_entrypoint_keeps_closed_package_and_fresh_output_guards(entrypoint: Path) -> None:
    source = _source(entrypoint)

    integrity = source.index("Assert-PackageIntegrity $packageRoot")
    config_read = source.index("$config = Get-Content", integrity)
    paddle_verify = source.index("$paddleEvidence = Read-PaddleRecipientEvidence", config_read)
    inference = source.index("& $executable", paddle_verify)
    assert integrity < config_read < paddle_verify < inference
    assert "Delivery package hash manifest is not closed" in source
    assert "Delivery package contains a reparse point" in source
    assert "Refusing to mix" in source
    assert "outside the immutable delivery package" in source


def test_single_entrypoint_prints_visible_status_ocr_and_review_state() -> None:
    source = _source(SINGLE)

    assert 'Write-Host "TRANSFER STATUS OCR"' in source
    assert '"Raw OCR" = [string]$result.fields.transfer_status.raw' in source
    assert '"Normalized" = [string]$result.fields.transfer_status.normalized' in source
    assert '"Review state" = [string]$result.fields.transfer_status.state' in source
    assert "Get-NormalizedTransferStatus $rawStatus" in source


def test_batch_entrypoint_reports_hybrid_stage_latency() -> None:
    source = _source(BATCH)

    assert 'stage_latency_ms.paddle_ocr.mean "recipient PP-OCR mean latency"' in source
    assert 'Write-Host ("PP-OCR mean   : {0:N2} ms" -f $paddleMeanMs)' in source


def test_readme_describes_the_same_pure_onnx_single_and_batch_contract() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "architecture-v13 unified receipt OCR" in readme
    assert "recipient-only PP-OCR detection, angle classification, and recognition" in readme
    assert "pure ONNX through the .NET CPU runtime" in readme
    assert "No Python, Paddle framework, CUDA, or network service" in readme
    assert "run-receipt-single-cpu.ps1" in readme
    assert "run-receipt-batch-cpu.ps1" in readme
    assert "Raw OCR" in readme
    assert "Normalized" in readme
    assert "Review state" in readme
    assert "both ONNX providers report `cpu`" in readme
    assert "10,016-image full-validation CPU A/B" in readme
    assert "amount 78.85%, time 98.40%, payment method 93.25%" in readme
    assert "recipient 90%" in readme
    assert "visible transfer status 90%" in readme
    assert "p95 overhead ceiling may not exceed 250 ms" in readme
    assert "fresh" in readme
    assert "checked independently against the same five" in readme
    assert "cannot mask a regression during final package validation" in readme
    assert "recalculated from the two contained, hash-bound CPU" in readme
    assert "complete `app/` publish closure" in readme
    assert "canonically sorted path/hash/size manifest" in readme


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_legacy_v13_pre_gate_only_waives_recipient_failures(entrypoint: Path) -> None:
    source = _source(entrypoint)

    assert 'StartsWith("recipient_field:", [StringComparison]::Ordinal)' in source
    assert "$nonRecipientOnnxFailures.Count -ne 0" in source
    assert "($onnxValidation.acceptance.passed -eq $true -and $onnxFailures.Count -ne 0)" in source
    assert "($onnxValidation.acceptance.passed -ne $true -and $onnxFailures.Count -eq 0)" in source
    assert "$onnxFailures.Count -ne 0 `" not in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_formal_ab_evidence_is_closed_hash_bound_and_versioned(entrypoint: Path) -> None:
    source = _source(entrypoint)

    assert "function Assert-HybridFormalEvidence" in source
    assert 'Properties["hybrid_ab_evidence"]' in source
    assert 'Properties["hybrid_formal_ab"]' in source
    for relative_path in (
        "evidence/hybrid-formal-ab-summary.json",
        "evidence/hybrid-formal-ab-comparisons.jsonl",
        "evidence/hybrid-formal-accuracy-summary.json",
        "evidence/hybrid-formal-accuracy-comparisons.jsonl",
        "evidence/hybrid-formal-fixed-inputs.txt",
        "evidence/hybrid-formal-baseline-inference-manifest.json",
        "evidence/hybrid-formal-hybrid-inference-manifest.json",
        "evidence/hybrid-formal-baseline-inference-summary.json",
        "evidence/hybrid-formal-hybrid-inference-summary.json",
        "evidence/hybrid-formal-cli-app-closure.json",
    ):
        assert relative_path in source
    assert "Resolve-ContainedPackageFile" in source
    assert "(Get-Sha256 $resolvedPath) -cne [string]$configHashProperty.Value" in source
    assert "[string]$validationHashProperty.Value -cne [string]$configHashProperty.Value" in source
    assert "[int]$config.schema_version -ne 1" in source
    assert "[int]$validation.schema_version -ne 1" in source
    assert "[int]$onnxValidation.schema_version -ne 1" in source
    assert "[int]$endToEndSummary.schema_version -ne 1" in source
    assert "[int]$comparisonSummary.schema_version -ne 2" in source
    assert "[int]$accuracySummary.schema_version -ne 1" in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_formal_ab_requires_clean_full_10016_cpu_run(entrypoint: Path) -> None:
    source = _source(entrypoint)

    for token in (
        "$requiredRecords = 10016",
        '$comparisonSummary.evaluation_mode -ne "formal"',
        "$comparisonSummary.accepted -ne $true",
        "$comparisonFailures.Count -ne 0",
        "$comparisonSummary.input_set_identical -ne $true",
        "$comparisonSummary.cli_summary_counts_verified -ne $true",
        "$comparisonSummary.records -ne $requiredRecords",
        "$comparisonSummary.invariant_records -ne $requiredRecords",
        "$comparisonSummary.input_set.records -ne $requiredRecords",
        "$comparisonSummary.input_set.input_manifest.records -ne $requiredRecords",
        "$comparisonSummary.run_manifests.baseline.records -ne $requiredRecords",
        "$comparisonSummary.run_manifests.hybrid.records -ne $requiredRecords",
        "input_set.normalized_source_set_sha256 -notmatch '^[0-9a-f]{64}$'",
        "run_manifests.baseline.normalized_source_set_sha256",
        "run_manifests.hybrid.normalized_source_set_sha256",
        "cli_build.assembly.size_bytes -le 0",
        "cli_build.assembly.sha256 -cne (Get-Sha256 $deliveredAssembly)",
        'evidencePaths["InputManifest"]',
        'evidencePaths["BaselineManifest"]',
        'evidencePaths["HybridManifest"]',
        "source and run manifests are not the contained 10016-record evidence",
        "$accuracySummary.manifest_sha256 -ne",
        "$comparisonSummary.run_manifests.hybrid.sha256",
        "$comparisonSummary.recipient_candidate_coverage -ne 1.0",
        "$p95Ceiling -gt 250.0",
        "$p95Overhead -gt $p95Ceiling",
        'evaluation_scope.kind -ne "full_split"',
        "$null -eq $requestedLimitProperty",
        "$null -ne $requestedLimitProperty.Value",
        "evaluation_scope.evaluated_expected_receipts -ne $requiredRecords",
        "evaluation_scope.full_split_expected_receipts -ne $requiredRecords",
        "coverage.expected_receipts -ne $requiredRecords",
        "coverage.matched_result_receipts -ne $requiredRecords",
        "coverage.fully_scored_receipts -ne $requiredRecords",
        "coverage.result_coverage -ne 1.0",
        "coverage.fully_scored_coverage -ne 1.0",
        "coverage_contract_version -ne 2",
        'candidate_coverage_domain -ne "all_expected_receipts"',
        "coverage.fully_candidate_covered_receipts -ne $requiredRecords",
        "coverage.all_field_candidate_coverage -ne 1.0",
    ):
        assert token in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_formal_ab_enforces_all_fixed_field_guards(entrypoint: Path) -> None:
    source = _source(entrypoint)

    for token in (
        '@{ Field = "amount"; Floor = 0.7885 }',
        '@{ Field = "time"; Floor = 0.9840 }',
        '@{ Field = "payment_method_field"; Floor = 0.9325 }',
        '@{ Field = "recipient_field"; Floor = 0.90 }',
        '@{ Field = "transfer_status"; Floor = 0.90 }',
        "$metricProperty.Value.records -ne [int]$referenceCountProperty.Value",
        "$denominatorProperty.Value -ne [int]$referenceCountProperty.Value",
        "$candidateProperty.Value.candidate_records -ne $requiredRecords",
        "$candidateProperty.Value.candidate_coverage -ne 1.0",
        "$metricProperty.Value.raw_exact_match -lt $requiredFloor",
        "$accuracySummary.by_field.transfer_status.non_success_to_success -ne 0",
        'Properties["max_non_success_to_success"]',
        "$null -eq $maxStatusSafetyProperty.Value",
        "$maxStatusSafetyProperty.Value -ne 0",
    ):
        assert token in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_formal_ab_binds_models_cpu_providers_and_config_to_validation(entrypoint: Path) -> None:
    source = _source(entrypoint)

    for token in (
        'config.ocr_mode -ne "hybrid-recipient"',
        'validation.ocr_mode -ne "hybrid-recipient"',
        'validation.inference_summary.requested_device -ne "cpu"',
        'validation.inference_summary.unified_provider -ne "cpu"',
        'validation.inference_summary.paddle_ocr_provider -ne "cpu"',
        "$validation.include_device_model -ne $true",
        "artifact_hashes.detector_sha256",
        "artifact_hashes.device_sha256",
        "artifact_hashes.unified_ocr_model_sha256",
        "paddle_delivery.contract_sha256",
        "$bindingProperties",
        "formal A/B scalar bindings disagree",
        "$configBinding.records_sha256 -ne [string]$Config.records_sha256",
        "$configBinding.records_sha256 -ne [string]$Validation.end_to_end_evaluation.records_sha256",
        "$configBinding.expected_receipts -ne $requiredRecords",
        '$configBinding.cli_assembly -ne "app/ReceiptMlNet.Cli.dll"',
        "$configBinding.paddle_package_size_bytes -ne [long]$PaddleEvidence.PackageSizeBytes",
        "$configBinding.baseline_runtime_summary_sha256 -ne",
        "$configBinding.hybrid_runtime_summary_sha256 -ne",
        "$configBinding.cli_app_closure_manifest_sha256 -cne",
        "$configBinding.cli_app_closure_sha256 -cne",
        "$configBinding.cli_app_closure_file_count -ne",
    ):
        assert token in source


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_ppocr_contract_rejects_zero_byte_payloads(entrypoint: Path) -> None:
    source = _source(entrypoint)
    paddle_file_reader = source[
        source.index("function Read-PaddleDeliveryFileEvidence") :
        source.index("function Read-PaddleRecipientEvidence")
    ]

    assert "$expectedBytes -le 0" in paddle_file_reader
    assert "$declaredPackageSize -le 0" in source
    assert "$expectedBytes -lt 0" not in paddle_file_reader


def test_single_and_batch_share_the_identical_formal_release_gate() -> None:
    single_source = _source(SINGLE)
    batch_source = _source(BATCH)

    assert _function(
        single_source,
        "Assert-HybridFormalEvidence",
        "Assert-AcceptedPackageBinding",
    ) == _function(
        batch_source,
        "Assert-HybridFormalEvidence",
        "Assert-AcceptedPackageBinding",
    )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_fresh_package_score_independently_repeats_the_full_formal_gate(
    entrypoint: Path,
) -> None:
    source = _source(entrypoint)
    gate = _function(
        source,
        "Assert-FreshFormalAccuracyEvidence",
        "Assert-AcceptedPackageBinding",
    )

    for token in (
        "$requiredRecords = 10016",
        "$Summary.formal_delivery_gate -ne $true",
        "$Summary.acceptance.formal_delivery_gate -ne $true",
        "$Summary.accepted -ne $true",
        "$Summary.acceptance.passed -ne $true",
        "$failures.Count -ne 0",
        "$acceptanceFailures.Count -ne 0",
        '$Summary.evaluation_scope.kind -ne "full_split"',
        "$null -eq $requestedLimitProperty",
        "$null -ne $requestedLimitProperty.Value",
        "$Summary.evaluation_scope.evaluated_expected_receipts -ne $requiredRecords",
        "$Summary.evaluation_scope.full_split_expected_receipts -ne $requiredRecords",
        "$Summary.coverage.expected_receipts -ne $requiredRecords",
        "$Summary.coverage.matched_result_receipts -ne $requiredRecords",
        "$Summary.coverage.fully_scored_receipts -ne $requiredRecords",
        "$Summary.coverage.result_coverage -ne 1.0",
        "$Summary.coverage.fully_scored_coverage -ne 1.0",
        "$Summary.coverage_contract_version -ne 2",
        "$Summary.coverage.coverage_contract_version -ne 2",
        '$Summary.coverage.candidate_coverage_domain -ne "all_expected_receipts"',
        "$Summary.coverage.fully_candidate_covered_receipts -ne $requiredRecords",
        "$Summary.coverage.all_field_candidate_coverage -ne 1.0",
        "$Summary.input_selection.hash_bound -ne $true",
        "$Summary.input_selection.sha256 -ne $ExpectedInputManifestSha256",
        '$Summary.accuracy_denominators.source -ne "input_selection.field_reference_counts"',
        '$Summary.all_receipt_candidate_coverage.scope -ne "all_selected_receipts"',
        "$Summary.all_receipt_candidate_coverage.complete_receipts -ne $requiredRecords",
        '@{ Field = "amount"; Floor = 0.7885 }',
        '@{ Field = "time"; Floor = 0.9840 }',
        '@{ Field = "payment_method_field"; Floor = 0.9325 }',
        '@{ Field = "recipient_field"; Floor = 0.90 }',
        '@{ Field = "transfer_status"; Floor = 0.90 }',
        "$records -ne [int]$referenceCountProperty.Value",
        "$denominatorProperty.Value -ne [int]$referenceCountProperty.Value",
        "$candidateProperty.Value.candidate_records -ne $requiredRecords",
        "$coverage -ne 1.0",
        "$exactMatch -lt $requiredFloor",
        'Properties["non_success_to_success"]',
        "$null -eq $statusSafetyProperty.Value",
        "$statusSafetyProperty.Value -ne 0",
        "$maxStatusSafetyProperty.Value -ne 0",
    ):
        assert token in gate

    accepted_binding = source.index("function Assert-AcceptedPackageBinding")
    fresh_call = source.index("Assert-FreshFormalAccuracyEvidence `", accepted_binding)
    aggregate_check = source.index("if ([int]$onnxValidation.schema_version", fresh_call)
    assert accepted_binding < fresh_call < aggregate_check
    assert "$endToEndSummary ([string]$UnifiedEvidence.ModelSha256)" in source[fresh_call:aggregate_check]
    assert "([string]$Config.records_sha256) $manifestSha256 (Get-Sha256 $validationInputListPath)" in source[fresh_call:aggregate_check]


def test_single_and_batch_share_the_identical_fresh_package_gate() -> None:
    single_source = _source(SINGLE)
    batch_source = _source(BATCH)

    assert _function(
        single_source,
        "Assert-FreshFormalAccuracyEvidence",
        "Assert-AcceptedPackageBinding",
    ) == _function(
        batch_source,
        "Assert-FreshFormalAccuracyEvidence",
        "Assert-AcceptedPackageBinding",
    )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_runtime_entrypoints_require_packager_five_field_candidate_counters(
    entrypoint: Path,
) -> None:
    source = _source(entrypoint)
    gate = _function(source, "Assert-AcceptedPackageBinding", "Assert-ProductionGeometry")

    assert "$Validation.candidate_complete -ne $requiredRecords" in gate
    assert '$Validation.PSObject.Properties["candidates_by_field"]' in gate
    assert '@("amount", "time", "recipient", "payment_method", "transfer_status")' in gate
    assert "$candidateCountProperty.Value -ne $requiredRecords" in gate


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_formal_ab_recomputes_latency_from_contained_runtime_summaries(
    entrypoint: Path,
) -> None:
    source = _source(entrypoint)
    gate = _function(
        source,
        "Assert-HybridFormalEvidence",
        "Assert-FreshFormalAccuracyEvidence",
    )

    for token in (
        'evidencePaths["BaselineRuntimeSummary"]',
        'evidencePaths["HybridRuntimeSummary"]',
        "$comparisonSummary.run_summaries.baseline",
        "$comparisonSummary.run_summaries.hybrid",
        "$baselineRuntimeRecord.path",
        "$hybridRuntimeRecord.path",
        "runtime summaries do not match their hash/size bindings",
        '$runtime.Summary.requested_device -ne "cpu"',
        '$runtime.Summary.unified_provider -ne "cpu"',
        "$runtime.Summary.input -ne $requiredRecords",
        "$runtime.Summary.written -ne $requiredRecords",
        "$runtime.Summary.skipped -ne 0",
        "$runtime.Summary.errors -ne 0",
        "$runtime.Summary.inference_latency_ms.count -ne $requiredRecords",
        "$null -eq $runtime.Paddle -and $null -ne $paddleProviderProperty.Value",
        '$null -ne $runtime.Paddle -and [string]$paddleProviderProperty.Value -ne "cpu"',
        "$rawP95Overhead = $rawHybridP95 - $rawBaselineP95",
        "$baselineP95 -ne $rawBaselineP95",
        "$hybridP95 -ne $rawHybridP95",
        "$p95Overhead -ne $rawP95Overhead",
        "$rawP95Overhead -gt $p95Ceiling",
    ):
        assert token in gate


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_wrapper_closes_the_entire_delivered_cli_app(entrypoint: Path) -> None:
    source = _source(entrypoint)
    closure = _function(
        source,
        "Assert-ContainedCliAppClosure",
        "Assert-HybridFormalEvidence",
    )
    formal_gate = _function(
        source,
        "Assert-HybridFormalEvidence",
        "Assert-FreshFormalAccuracyEvidence",
    )

    for token in (
        'Resolve-ContainedPackageDirectory $PackageRoot "app"',
        "$ExpectedManifestSha256 -cnotmatch '^[0-9a-f]{64}$'",
        "$rows.Count -ne $ExpectedFileCount",
        "$propertyNames.Count -ne 3",
        '$propertyNames -notcontains "path"',
        '$propertyNames -notcontains "sha256"',
        '$row.PSObject.Properties["size_bytes"]',
        "$sizeProperty.Value -isnot [int]",
        "$sizeProperty.Value -isnot [long]",
        "Resolve-ContainedPackageFile $appRoot",
        "$relative -cne [string]$row.path",
        "$listed.ContainsKey($key)",
        "manifest paths are not canonically sorted",
        "Get-PackagePayloadFiles $appRoot",
        "$actual.ContainsKey($key)",
        "$missing.Count -ne 0 -or $extra.Count -ne 0",
        "Delivered CLI app closure is not exact",
        '"receiptmlnet.cli.exe" = $false',
        '"onnxruntime.dll" = $false',
        '"opencvsharpextern.dll" = $false',
        "$missingRequired.Count -ne 0",
        "lacks required managed/native payload",
        "closure manifest changed during verification",
    ):
        assert token in closure

    for token in (
        "$comparisonSummary.cli_build.app_closure",
        "$appClosure.root",
        "$closureManifestRecord.path",
        'evidencePaths["CliAppClosure"]',
        "$appClosure.closure_sha256 -ne [string]$closureManifestRecord.sha256",
        "$appClosure.closure_sha256 -ne (Get-Sha256 $closureManifestPath)",
        "$closureManifestSizeProperty.Value -ne [long]$closureManifestItem.Length",
        "$closureFileCountProperty.Value -le 0",
        "Assert-ContainedCliAppClosure `",
    ):
        assert token in formal_gate


def test_single_and_batch_share_the_identical_cli_app_closure_gate() -> None:
    single_source = _source(SINGLE)
    batch_source = _source(BATCH)

    assert _function(
        single_source,
        "Assert-ContainedCliAppClosure",
        "Assert-HybridFormalEvidence",
    ) == _function(
        batch_source,
        "Assert-ContainedCliAppClosure",
        "Assert-HybridFormalEvidence",
    )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_hybrid_delivery_powershell_parses_when_available(entrypoint: Path) -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(entrypoint).replace("'", "''")
    parser_command = (
        "$errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_path}',[ref]$null,[ref]$errors); "
        "if($errors.Count -gt 0){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
