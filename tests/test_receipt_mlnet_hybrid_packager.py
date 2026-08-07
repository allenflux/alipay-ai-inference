from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "receipt-mlnet-unified-package-validate-4090.ps1"


def _source() -> str:
    return PACKAGER.read_text(encoding="utf-8")


def test_hybrid_packager_requires_verified_recipient_bundle_and_supports_portable_dotnet() -> None:
    source = _source()

    parameter_prefix = source.split("Set-StrictMode", 1)[0]
    assert "[Parameter(Mandatory = $true)]\n    [string]$PaddleDeliveryBundle" in parameter_prefix
    assert "[string]$HybridAbEvidence" in parameter_prefix
    assert "[string]$DotnetExe" in parameter_prefix
    assert 'Join-Path $repoRoot "artifacts\\dotnet8\\dotnet.exe"' in source
    assert "& $DotnetExe restore" in source
    assert "& $DotnetExe publish" in source
    assert source.count("& $DotnetExe run") >= 2
    assert "& dotnet " not in source


def test_paddle_delivery_is_a_closed_hash_verified_pure_onnx_contract() -> None:
    source = _source()

    assert "function Assert-PaddleDeliveryBundle" in source
    assert "function Assert-PaddleDeliveryFileRecord" in source
    assert 'kind -ne "paddle_ocr_v2_delivery"' in source
    assert '$modelNames -notcontains "det"' in source
    assert '$modelNames -notcontains "cls"' in source
    assert '$modelNames -notcontains "rec"' in source
    assert "Get-Sha256 $target" in source
    assert "(Get-Item -LiteralPath $target).Length -ne $expectedBytes" in source
    assert 'foreach ($dependency in @("Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files"))' in source
    assert "payload is not closed over pure ONNX det/cls/rec + dictionary" in source
    assert 'PSObject.Properties["package_size_bytes"]' in source
    assert "$packageSizeBytes -le 0" in source
    assert "$packageSizeBytes -ne $verifiedPackageSizeBytes" in source
    assert "PackageSizeBytes = $packageSizeBytes" in source
    assert '$recipientPaddleDirectory = Join-Path $modelDirectory "recipient-ppocr"' in source
    assert "Assert-PaddleDeliveryBundle $recipientPaddleDirectory" in source


def test_validation_executes_hybrid_cpu_pipeline_and_proves_both_ocr_providers() -> None:
    source = _source()

    invocation = source.split("function Invoke-MlNetValidation", 1)[1].split(
        'Write-Host "mlnet_hybrid_recipient_${RuntimeFlavor}_validate"', 1
    )[0]
    assert '"--ocr", "hybrid-recipient"' in invocation
    assert '"--ocr-model", $deliveryUnifiedModel' in invocation
    assert '"--ocr-bundle", $recipientPaddleDirectory' in invocation
    assert '"--device-model", $deliveryDevice' in invocation
    assert "Hybrid recipient delivery packaging is CPU-only" in source
    assert "Hybrid recipient delivery validation must not skip the device model" in source
    assert "Published ML.NET unified OCR did not prove strict" in source
    assert "Published ML.NET PP-OCR did not prove strict cpu det/cls/rec execution" in source
    assert '[string]$runtimeSummary.unified_provider -ne $requiredRuntimeProvider' in source
    assert '[string]$runtimeSummary.paddle_ocr_provider -ne "cpu"' in source
    assert 'model_contracts.ocr_bundle_contract_sha256 -ne $paddleDeliveryContractSha256' in source


def test_v13_pre_gate_excludes_legacy_recipient_but_formal_score_guards_all_five_fields() -> None:
    source = _source()

    pre_gate = source.split("$preFieldGates = @(", 1)[1].split("\n)", 1)[0]
    assert 'Field = "amount"' in pre_gate
    assert 'Field = "time"' in pre_gate
    assert 'Field = "payment_method_field"' in pre_gate
    assert "recipient_field" not in pre_gate
    assert 'StartsWith("recipient_field:"' in source

    final_gate = source.split("$finalFieldGates = @(", 1)[1].split("\n        )", 1)[0]
    for field in ("amount", "time", "payment_method_field", "recipient_field"):
        assert f'Field = "{field}"' in final_gate
    assert '"--status-floor"' in source
    assert '$statusScoreExactMatch -lt $requiredStatusTextFloor' in source
    assert "$scoreCandidateCoverage -ne 1.0" in source
    assert "$statusScoreCandidateCoverage -ne 1.0" in source
    assert "$endToEndSummary.formal_delivery_gate -ne $true" in source
    assert '$endToEndSummary.evaluation_scope.kind -ne "full_split"' in source
    assert '$endToEndSummary.coverage.result_coverage -ne 1.0' in source
    assert '$endToEndSummary.coverage.fully_scored_coverage -ne 1.0' in source


def test_formal_package_requires_and_hash_binds_cpu_ab_evidence() -> None:
    source = _source()

    assert "Formal hybrid delivery requires -HybridAbEvidence" in source
    assert 'Join-Path $HybridAbEvidence "comparison\\summary.json"' in source
    assert 'Join-Path $HybridAbEvidence "hybrid-val-score\\summary.json"' in source
    assert 'kind -ne "receipt_mlnet_hybrid_recipient_cpu_ab_v1"' in source
    assert "Assert-HybridAbComparisonSchema $hybridAbSummary" in source
    assert "Assert-HybridAbScoreSchema $hybridAbScore" in source
    assert "$requiredFormalReceiptCount = 10016" in source
    assert "$hybridAbRecords -ne $requiredFormalReceiptCount" in source
    assert "$hybridAbScoreExpectedRecords -ne $requiredFormalReceiptCount" in source
    assert "$expectedRecords -ne $requiredFormalReceiptCount" in source
    assert 'evaluation_mode -ne "formal"' in source
    assert "$hybridAbP95Ceiling -gt 250.0" in source
    assert "$hybridAbP95Overhead -gt $hybridAbP95Ceiling" in source
    assert "artifact_hashes.unified_ocr_model_sha256" in source
    assert "paddle_delivery.contract_sha256 -ne $paddleDeliveryContractSha256" in source
    assert '$hybridAbScore.formal_delivery_gate -ne $true' in source
    assert '$hybridAbScore.by_field.transfer_status.non_success_to_success -ne 0' in source
    fixed_gates = source.split("$hybridFixedGates = @(", 1)[1].split("\n    )", 1)[0]
    for field in ("amount", "time", "payment_method_field", "recipient_field", "transfer_status"):
        assert f'Field = "{field}"' in fixed_gates
    for evidence_name in (
        "hybrid-formal-ab-summary.json",
        "hybrid-formal-ab-comparisons.jsonl",
        "hybrid-formal-accuracy-summary.json",
        "hybrid-formal-accuracy-comparisons.jsonl",
    ):
        assert evidence_name in source
    assert "Formal hybrid CPU A/B evidence changed during package validation" in source


def test_formal_ab_binds_source_manifests_score_source_and_cli_assembly() -> None:
    source = _source()

    for required_fragment in (
        "input_set.normalized_source_set_sha256",
        "input_set.input_manifest.normalized_source_set_sha256",
        "run_manifests.baseline.normalized_source_set_sha256",
        "run_manifests.hybrid.normalized_source_set_sha256",
        "$hybridAbInputManifestPath",
        "$hybridAbBaselineManifestPath",
        "$hybridAbHybridManifestPath",
        "$hybridAbBaselineRuntimeSummaryPath",
        "$hybridAbHybridRuntimeSummaryPath",
        "$hybridAbCliAssemblySha256",
        "$hybridAbCliClosureManifestPath",
        "$hybridAbCliClosureSha256",
        "$hybridAbScore.manifest_sha256 -ne [string]$hybridAbSummary.run_manifests.hybrid.sha256",
        "$hybridAbScoreResultsRoot.Equals(",
        "Published ReceiptMlNet.Cli assembly does not match the hash-bound formal A/B build",
        "Hybrid A/B p95 evidence does not exactly equal the hash-bound raw runtime summaries",
        '$runtimeLatency.count -ne $requiredFormalReceiptCount',
        "Fresh canonical full-val input manifest does not match the hash-bound formal A/B input manifest",
    ):
        assert required_fragment in source
    for evidence_name in (
        "hybrid-formal-fixed-inputs.txt",
        "hybrid-formal-baseline-inference-manifest.json",
        "hybrid-formal-hybrid-inference-manifest.json",
        "hybrid-formal-baseline-inference-summary.json",
        "hybrid-formal-hybrid-inference-summary.json",
        "hybrid-formal-cli-app-closure.json",
    ):
        assert evidence_name in source


def test_packager_reverifies_exact_cli_publish_closure() -> None:
    source = _source()

    assert "function Assert-CliAppClosure" in source
    assert "CLI app closure is not exact" in source
    for basename in (
        "receiptmlnet.cli.exe",
        "receiptmlnet.cli.dll",
        "receiptmlnet.cli.deps.json",
        "receiptmlnet.cli.runtimeconfig.json",
        "microsoft.ml.onnxruntime.dll",
        "onnxruntime.dll",
        "opencvsharp.dll",
        "opencvsharpextern.dll",
    ):
        assert basename in source
    assert "$verifiedPublishedCliClosure = Assert-CliAppClosure" in source
    assert "$finalPublishedCliClosure = Assert-CliAppClosure" in source
    assert "cli_app_closure_manifest_sha256 = $hybridAbCliClosureSha256" in source


def test_hybrid_package_schema_binds_bundle_models_and_formal_evidence() -> None:
    source = _source()

    assert 'kind = "receipt_mlnet_hybrid_recipient_package_validation_v1"' in source
    assert '"receipt_mlnet_hybrid_recipient_delivery_package_v1"' in source
    assert '"receipt_mlnet_hybrid_recipient_candidate_smoke_package_v1"' in source
    assert 'ocr_mode = "hybrid-recipient"' in source
    assert 'recipient_ocr_bundle = "models/recipient-ppocr"' in source
    assert 'recipient_ppocr = $recipientPaddleArtifactEvidence' in source
    assert 'bundle_path = "models/recipient-ppocr"' in source
    assert 'contract_path = "models/recipient-ppocr/paddle_ocr_delivery.contract.json"' in source
    assert 'path = "models/recipient-ppocr/$([string]$record.RelativePath)"' in source
    assert "hybrid_formal_ab = $hybridAbEvidenceBinding" in source
    assert "hybrid_ab_evidence = $hybridAbEvidenceBinding" in source
    assert "paddle_delivery_contract_sha256 = $paddleDeliveryContractSha256" in source


def test_hybrid_publication_remains_hidden_staged_and_atomic() -> None:
    source = _source()

    stage_assignment = source.index(
        '$stagingRoot = Join-Path $deliveryParent (".receipt-mlnet-unified-staging-"'
    )
    integrity = source.index("Assert-PackageIntegrity $stagingRoot", stage_assignment)
    atomic_move = source.index("[IO.Directory]::Move($stagingRoot, $DeliveryDir)", integrity)
    assert stage_assignment < integrity < atomic_move
    assert "if (-not $published -and (Test-Path -LiteralPath $stagingRoot))" in source
    assert "Remove-Item -LiteralPath $stagingRoot -Recurse -Force" in source


def test_hybrid_packager_powershell_parses_when_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(PACKAGER).replace("'", "''")
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
