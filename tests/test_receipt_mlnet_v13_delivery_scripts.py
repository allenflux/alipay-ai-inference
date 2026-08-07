from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_VALIDATOR = ROOT / "scripts" / "receipt-mlnet-unified-package-validate-4090.ps1"
JSON_NORMALIZER = ROOT / "scripts" / "normalize_json_summary.py"
END_TO_END_SCORER = ROOT / "scripts" / "receipt_mlnet_unified_evaluate.py"
DELIVERY_SCRIPTS = (
    ROOT / "scripts" / "receipt-mlnet-add-production-entrypoints.ps1",
    ROOT / "dotnet" / "ReceiptMlNet.Cli" / "DeliveryScripts" / "run-receipt-single-cpu.ps1",
    ROOT / "dotnet" / "ReceiptMlNet.Cli" / "DeliveryScripts" / "run-receipt-batch-cpu.ps1",
)
ALL_SCRIPTS = (PACKAGE_VALIDATOR, *DELIVERY_SCRIPTS)


def test_json_normalizer_accepts_windows_powershell_utf8_bom(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b"\xef\xbb\xbf" + b'{"metric":NaN,"passed":true}\n')

    completed = subprocess.run(
        [sys.executable, str(JSON_NORMALIZER), str(evidence)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"metric": None, "passed": True}


@pytest.mark.parametrize("script_path", ALL_SCRIPTS, ids=lambda path: path.name)
def test_delivery_contract_selects_v12_or_v13_strictly_from_sidecar(
    script_path: Path,
) -> None:
    script = script_path.read_text(encoding="utf-8")

    assert '12 { "receipt_unified_field_reader_v12" }' in script
    assert '13 { "receipt_unified_field_reader_v13" }' in script
    assert "$artifactKind -ne $expectedKind" in script
    assert "unsupported architecture_version" in script
    assert 'Properties["status_text_logits"]' in script
    assert '"decode_and_normalize_review_only"' in script
    assert "v12 contract must not declare the v13 status_text_logits output" in script


@pytest.mark.parametrize("script_path", DELIVERY_SCRIPTS, ids=lambda path: path.name)
def test_delivery_entrypoints_bind_v13_policy_and_keep_v12_legacy_declarations(
    script_path: Path,
) -> None:
    script = script_path.read_text(encoding="utf-8")

    assert "Kind = $artifactKind" in script
    assert "ArchitectureVersion = $architectureVersion" in script
    assert 'Properties["architecture_version"]' in script
    assert 'Properties["status_text_delivery_policy"]' in script
    assert 'Properties["status_text_review_value"]' in script
    assert "Legacy v12 declarations did not record architecture_version" in script
    assert 'Properties["transfer_status"]' in script
    assert '[int]$UnifiedEvidence.ArchitectureVersion -eq 13' in script
    assert '[string]$stateProperty.Value -eq "absent"' in script
    assert '[string]$stateProperty.Value -ne "review"' in script
    assert 'Properties["raw"]' in script
    assert 'Properties["candidate"]' in script
    assert 'Properties["ctc_candidate"]' in script
    assert 'Properties["normalized"]' in script
    assert "incomplete or inconsistent OCR text evidence" in script
    assert "complete delivery path requires visible OCR text" in script
    assert "Get-NormalizedTransferStatus $rawStatus" in script


def test_single_image_entrypoint_labels_status_ocr_semantic_and_decision_separately() -> None:
    script = DELIVERY_SCRIPTS[1].read_text(encoding="utf-8")

    assert 'Write-Host "TRANSFER STATUS OCR"' in script
    assert '"Raw OCR" = [string]$result.fields.transfer_status.raw' in script
    assert '"Normalized" = [string]$result.fields.transfer_status.normalized' in script
    assert '"Decision" = [string]$result.fields.transfer_status.state' in script
    assert "Get-NormalizedTransferStatus $rawStatus" in script


def test_package_evidence_records_the_actual_unified_contract_version() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "kind = $unifiedKind" in script
    assert "architecture_version = $unifiedArchitectureVersion" in script
    assert "unified_ocr_kind = $unifiedKind" in script
    assert "unified_ocr_architecture_version = $unifiedArchitectureVersion" in script
    assert '$packageValidation["status_text_delivery_policy"]' in script
    assert '$packageConfig["status_text_delivery_policy"]' in script
    assert 'Properties["transfer_status"]' in script


def test_v13_artifacts_and_bound_validation_enter_packager_without_renaming() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "[string]$UnifiedModelPath" in script
    assert "[string]$OnnxValidationSummaryPath" in script
    assert "Supply -UnifiedModelPath and -OnnxValidationSummaryPath together" in script
    assert "Test-PathWithin $UnifiedModelPath $RunDirectory" in script
    assert "Test-PathWithin $OnnxValidationSummaryPath $RunDirectory" in script
    assert "$unifiedModel = if ($usesExplicitUnifiedArtifactBinding)" in script
    assert "$onnxValidationSummary = if ($usesExplicitUnifiedArtifactBinding)" in script
    assert 'Join-Path $RunDirectory "best.onnx"' in script
    assert 'Join-Path $RunDirectory "onnx-val\\summary.json"' in script
    assert "Assert-UnifiedBundle $unifiedModel" in script
    assert "summary.model_sha256 -ne $unifiedModelSha256" in script
    assert 'binding = if ($usesExplicitUnifiedArtifactBinding)' in script
    assert "onnx_validation_summary_sha256 = Get-Sha256 $onnxValidationSummary" in script
    assert 'Join-Path $RunDirectory "v13_status_ocr_validation.json"' in script
    assert "guardedValidationEvidence.candidate.model_sha256 -ne $unifiedModelSha256" in script
    assert "guardedValidationEvidence.candidate.contract_sha256 -ne $unifiedContractSha256" in script
    assert "guardedValidationEvidence.candidate.labels_sha256 -ne $unifiedLabelsSha256" in script
    assert "packagingBinding.onnx_validation_summary_sha256" in script
    assert "$valEvidence[0].summary_sha256" in script
    assert "$testEvidence[0].summary_sha256" in script
    assert 'Join-Path $evidenceDirectory "v13-guarded-validation.json"' in script


def test_v13_direct_packaging_keeps_fixed_visible_status_ocr_gate() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "$requiredStatusTextFloor = 0.90" in script
    assert 'Properties["min_status_exact_match"]' in script
    assert "$statusMetric.ctc_records" in script
    assert "$statusMetric.ctc_raw_exact_match" in script
    assert "$requestedStatusFloor -lt $requiredStatusTextFloor" in script
    assert "$statusCtcExactMatch -lt $requiredStatusTextFloor" in script
    assert 'runtime_policy -ne "decode_and_normalize_review_only"' in script
    assert 'review_value -ne "review"' in script
    assert '$validatedMetrics["transfer_status"]' in script
    assert 'metric = "ctc_raw_exact_match"' in script
    assert "Formal end-to-end delivery validation requires -IncludeDeviceModel" in script
    assert '"--status-floor"' in script
    assert '$endToEndSummary.by_field.PSObject.Properties["transfer_status"]' in script
    assert '$statusScoreExactMatch -lt $requiredStatusTextFloor' in script
    assert "$statusScoreCandidateCoverage -ne 1.0" in script
    assert '$validatedEndToEndMetrics["transfer_status"]' in script
    assert "$statusScoreMetric.non_success_to_success -ne 0" in script
    assert '$endToEndSummary.acceptance.PSObject.Properties["max_non_success_to_success"]' in script
    assert "formal v13 delivery requires visible OCR text" in script
    assert "Get-NormalizedTransferStatus $statusRaw" in script

    scorer = END_TO_END_SCORER.read_text(encoding="utf-8")
    assert 'field_result_keys["transfer_status"] = STATUS_RESULT_KEY' in scorer
    assert 'floors["transfer_status"] = float(status_floor)' in scorer
    assert '"min_status_exact_match": float(status_floor)' in scorer
    assert '"non_success_to_success": non_success_to_success' in scorer
    assert 'f"non_success_to_success={int(status_metrics[\'non_success_to_success\'])} > 0"' in scorer


def test_packager_floors_can_only_be_raised_not_weakened() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "$minimumAmountFloor = 0.7885" in script
    assert "$minimumTimeFloor = 0.9840" in script
    assert "$minimumPaymentFloor = 0.9325" in script
    assert "$minimumRecipientFloor = 0.90" in script
    assert "$AmountFloor -lt $minimumAmountFloor" in script
    assert "$TimeFloor -lt $minimumTimeFloor" in script
    assert "$PaymentFloor -lt $minimumPaymentFloor" in script
    assert "$RecipientFloor -lt $minimumRecipientFloor" in script
    assert "Delivery floors may be raised but must not be lower" in script
    assert "guardedFloors.visible_transfer_status_cjk_text -lt $requiredStatusTextFloor" in script
    assert "summary.records_sha256 -ne (Get-Sha256 $v13SummaryRecordsPath)" in script
    assert "guardedValidationEvidence.manifest.records_sha256" in script
    assert 'Join-Path $evidenceDirectory "v13-onnx-test-summary.json"' in script
    assert 'Join-Path $evidenceDirectory "bound-unified-fields.jsonl"' in script
    assert '"--records", $scoringRecords' in script
    assert "Get-Sha256 $recordsSnapshot" in script
    assert "Get-Sha256 $Records) -ne $requestedRecordsSha256" in script
    assert "refusing atomic publication" in script


def test_v13_status_review_validation_does_not_change_four_field_scoring() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "[double]$AmountFloor = 0.7885" in script
    assert "[double]$TimeFloor = 0.9840" in script
    assert "[double]$PaymentFloor = 0.9325" in script
    assert "[double]$RecipientFloor = 0.90" in script
    assert script.count('foreach ($fieldName in @("amount", "time", "recipient", "payment_method"))') == 1

    four_field_loop = script.index(
        'foreach ($fieldName in @("amount", "time", "recipient", "payment_method"))'
    )
    status_guard = script.index("if ($unifiedArchitectureVersion -eq 13)", four_field_loop)
    assert status_guard > four_field_loop


@pytest.mark.parametrize("script_path", ALL_SCRIPTS, ids=lambda path: path.name)
def test_delivery_powershell_parses_when_powershell_is_available(
    script_path: Path,
) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(script_path).replace("'", "''")
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
