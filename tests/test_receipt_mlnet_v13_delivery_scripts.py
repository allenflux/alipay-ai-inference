from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_VALIDATOR = ROOT / "scripts" / "receipt-mlnet-unified-package-validate-4090.ps1"
DELIVERY_SCRIPTS = (
    ROOT / "scripts" / "receipt-mlnet-add-production-entrypoints.ps1",
    ROOT / "dotnet" / "ReceiptMlNet.Cli" / "DeliveryScripts" / "run-receipt-single-cpu.ps1",
    ROOT / "dotnet" / "ReceiptMlNet.Cli" / "DeliveryScripts" / "run-receipt-batch-cpu.ps1",
)
ALL_SCRIPTS = (PACKAGE_VALIDATOR, *DELIVERY_SCRIPTS)


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


def test_package_evidence_records_the_actual_unified_contract_version() -> None:
    script = PACKAGE_VALIDATOR.read_text(encoding="utf-8")

    assert "kind = $unifiedKind" in script
    assert "architecture_version = $unifiedArchitectureVersion" in script
    assert "unified_ocr_kind = $unifiedKind" in script
    assert "unified_ocr_architecture_version = $unifiedArchitectureVersion" in script
    assert '$packageValidation["status_text_delivery_policy"]' in script
    assert '$packageConfig["status_text_delivery_policy"]' in script
    assert 'Properties["transfer_status"]' in script


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
    )
    assert completed.returncode == 0, completed.stderr
