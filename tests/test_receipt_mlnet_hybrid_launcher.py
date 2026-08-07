from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "v13-cpu.ps1"


def _source() -> str:
    payload = LAUNCHER.read_bytes()
    assert all(byte < 128 for byte in payload), "hybrid CPU orchestrator must remain Windows PowerShell ASCII"
    return payload.decode("ascii")


def test_orchestrator_requires_and_passes_verified_ppocr_bundle_to_both_runs() -> None:
    source = _source()

    assert "[Parameter(Mandatory = $true)]\n    [string]$PaddleDeliveryBundle" in source
    assert "[Parameter(Mandatory = $true)]\n    [string]$HybridAbEvidence" in source
    assert '$PaddleDeliveryBundle = [IO.Path]::GetFullPath($PaddleDeliveryBundle)' in source
    assert 'Require-Directory $PaddleDeliveryBundle "Paddle OCR recipient delivery bundle"' in source
    assert 'Join-Path $PaddleDeliveryBundle "paddle_ocr_delivery.contract.json"' in source
    assert '[string]$paddleContract.kind -ne "paddle_ocr_v2_delivery"' in source
    assert 'Join-Path $PaddleDeliveryBundle "paddle"' in source
    assert "PaddleDeliveryBundle = $PaddleDeliveryBundle" in source
    assert 'Require-Directory $HybridAbEvidence "complete formal hybrid CPU A/B evidence"' in source
    assert '$formalArguments["HybridAbEvidence"] = $HybridAbEvidence' in source
    assert source.index("PaddleDeliveryBundle = $PaddleDeliveryBundle") < source.index(
        '$pilotArguments["Output"]'
    )
    assert source.index("PaddleDeliveryBundle = $PaddleDeliveryBundle") < source.index(
        '$formalArguments["Records"]'
    )


def test_orchestrator_supports_system_or_portable_dotnet_and_passes_exact_host() -> None:
    source = _source()

    assert "[string]$DotnetExe" in source
    assert "Get-Command dotnet -ErrorAction SilentlyContinue" in source
    assert 'Join-Path $repoRoot "artifacts\\dotnet8\\dotnet.exe"' in source
    assert '$DotnetExe = [IO.Path]::GetFullPath($DotnetExe)' in source
    assert '@{ Path = $DotnetExe; Description = ".NET 8 host" }' in source
    assert "DotnetExe = $DotnetExe" in source


def test_orchestrator_gates_hybrid_kinds_artifacts_and_both_cpu_providers() -> None:
    source = _source()

    assert "receipt_mlnet_hybrid_recipient_candidate_smoke_package_v1" in source
    assert "receipt_mlnet_hybrid_recipient_delivery_package_v1" in source
    assert source.count("receipt_mlnet_hybrid_recipient_package_validation_v1") == 2
    assert source.count('paddle_ocr_provider -ne "cpu"') == 2
    assert source.count('unified_provider -ne "cpu"') == 2
    assert 'model_artifacts.PSObject.Properties["recipient_ppocr"]' in source
    assert '[string]$artifact.kind -ne "paddle_ocr_v2_delivery"' in source
    assert '[string]$artifact.bundle_path -ne "models/recipient-ppocr"' in source
    assert 'models/recipient-ppocr/paddle_ocr_delivery.contract.json' in source
    assert source.count("Assert-RecipientPpocrArtifact") == 5  # declaration + pilot/config + formal/config
    assert 'IncludeDeviceModel = $true' in source
    assert '$packageValidation.include_device_model -eq $true' in source
    assert '$null -ne $packageValidation.model_artifacts.device' in source
    assert '$null -ne $packageConfig.model_artifacts.device' in source


def test_orchestrator_keeps_full_10016_atomic_formal_and_all_fixed_field_gates() -> None:
    source = _source()

    assert "$formalCount = 10016" in source
    assert 'inputLines.Count -ne $formalCount' in source
    assert '$formalArguments["Limit"]' not in source
    assert '$formalArguments["Records"] = $selected.Records' in source
    assert '$formalArguments["EndToEndEvaluationDir"] = $formalEvaluation' in source
    assert '$scoreSummary.formal_delivery_gate -ne $true' in source
    assert "Assert-FiveFieldCoverage $scoreSummary $formalCount" in source
    for field in ("amount", "time", "payment_method_field", "recipient_field", "transfer_status"):
        assert f'"{field}"' in source
    for floor in (
        "$amountFloor = 0.7885",
        "$timeFloor = 0.9840",
        "$paymentFloor = 0.9325",
        "$recipientFloor = 0.90",
        "$statusFloor = 0.90",
    ):
        assert floor in source
    assert 'candidate_coverage -ne 1.0' in source
    assert 'non_success_to_success -ne 0' in source
    assert '$atomicPublished = (' in source
    assert 'SHA256SUMS.json' in source
    assert '-or -not $atomicPublished' in source
    assert "V13_HYBRID_RECIPIENT_CPU_DELIVERY_PASS" in source


def test_orchestrator_allows_only_legacy_recipient_failure_before_hybrid_gate() -> None:
    source = _source()

    assert 'StartsWith("recipient_field:"' in source
    assert '$nonRecipientFailures.Count -ne 0' in source
    assert '$Summary.acceptance.passed -eq $false' in source
    assert 'Field = "recipient_field"; Metric = "raw_exact_match"' not in source
    assert '$valEvidence[0].accepted -ne $true' not in source
    assert '$testEvidence[0].accepted -ne $true' not in source


def test_hybrid_orchestrator_powershell_parses_when_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(LAUNCHER).replace("'", "''")
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
