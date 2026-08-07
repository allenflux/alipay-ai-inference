from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "v13-cpu.ps1"


def _source() -> str:
    payload = LAUNCHER.read_bytes()
    assert all(byte < 128 for byte in payload), "v13 CPU launcher must remain ASCII-only"
    return payload.decode("ascii")


def test_launcher_has_one_short_command_and_auto_selects_latest_passed_v13_run() -> None:
    source = _source()

    assert LAUNCHER.name == "v13-cpu.ps1"
    assert "powershell -ExecutionPolicy Bypass -File .\\scripts\\v13-cpu.ps1" in source
    assert '[string]$TeacherRoot = "D:\\alipay-ai-data\\receipt-lite-teacher-120k-v1"' in source
    assert 'Filter "unified-run-v13-*"' in source
    assert 'Join-Path $_.FullName "v13_status_ocr_validation.json"' in source
    assert "Sort-Object LastWriteTimeUtc -Descending" in source
    assert 'receipt_unified_status_text_v13_guarded_validation_v1' in source
    assert 'receipt_unified_field_reader_v13' in source
    assert '$valEvidence[0].accepted -ne $true' not in source
    assert '$testEvidence[0].accepted -ne $true' not in source
    assert 'status_text_exact_match -lt $statusFloor' in source
    assert '$testSummaryPath = [IO.Path]::GetFullPath([string]$testEvidence[0].summary_path)' in source
    assert '[string]$testEvidence[0].summary_sha256 -ne $testSummarySha256' in source
    assert 'Assert-PassedGpuSummary $testSummary "test"' in source
    assert 'StartsWith("recipient_field:"' in source
    assert '$nonRecipientFailures.Count -ne 0' in source
    assert 'Field = "recipient_field"; Metric = "raw_exact_match"' not in source
    assert '$Summary.acceptance.passed -eq $false' in source
    assert 'Rejected candidates: ' in source


def test_launcher_keeps_all_fixed_floors_and_exact_sample_counts() -> None:
    source = _source()

    assert "$pilotCount = 100" in source
    assert "$formalCount = 10016" in source
    assert "$amountFloor = 0.7885" in source
    assert "$timeFloor = 0.9840" in source
    assert "$paymentFloor = 0.9325" in source
    assert "$recipientFloor = 0.90" in source
    assert "$statusFloor = 0.90" in source
    assert 'inputLines.Count -ne $formalCount' in source
    assert '$pilotArguments["Limit"] = $pilotCount' in source
    assert '$formalArguments["Records"] = $selected.Records' in source
    assert '$formalArguments["EndToEndEvaluationDir"] = $formalEvaluation' in source
    assert '$formalArguments["Limit"]' not in source


def test_launcher_explicitly_runs_detector_device_and_v13_unified_on_cpu() -> None:
    source = _source()

    assert 'UnifiedModelPath = $selected.UnifiedModel' in source
    assert 'OnnxValidationSummaryPath = $selected.ValidationSummary' in source
    assert 'DetectorModel = $detectorModel' in source
    assert 'DeviceModel = $deviceModel' in source
    assert 'IncludeDeviceModel = $true' in source
    assert 'RuntimeFlavor = "cpu"' in source
    assert 'Rectification = "max-side-1600"' in source
    assert 'receipt_lrcnn_v1.onnx' in source
    assert 'statusbar_device_v1.onnx' in source
    assert 'UnifiedModelPath = $selected.UnifiedModel' in source
    assert source.count('unified_provider -ne "cpu"') == 2
    assert 'unified_artifact_source.binding -ne "explicit_run_contained"' in source


def test_launcher_runs_pilot_before_formal_and_never_reuses_result_paths() -> None:
    source = _source()

    pilot_call = source.index('& $packager @pilotArguments')
    pilot_gate = source.index('The 100-image complete three-model CPU pilot did not pass')
    formal_call = source.index('& $packager @formalArguments')
    assert pilot_call < pilot_gate < formal_call
    assert 'refusing result reuse' in source
    assert 'yyyyMMdd-HHmmssfff' in source
    assert '$formalDelivery = Join-Path $deliveryRoot' in source


def test_launcher_reports_formal_metrics_latency_paths_and_atomic_publication() -> None:
    source = _source()

    for token in (
        "amount_raw_exact",
        "time_raw_exact",
        "payment_raw_exact",
        "recipient_raw_exact",
        "status_raw_exact",
        "cpu_p50_ms",
        "cpu_p95_ms",
        "errors=",
        "output=",
        "evaluation=",
        "delivery=",
        "atomic_published=",
    ):
        assert token in source
    assert 'full_val_end_to_end_scored_cpu' in source
    assert 'unified_artifact_source.binding -eq "explicit_run_contained"' in source
    assert 'receipt_mlnet_unified_delivery_package_v1' in source
    assert 'scoreSummary.by_field.transfer_status.candidate_coverage -ne 1.0' in source
    assert 'scoreSummary.by_field.transfer_status.non_success_to_success -ne 0' in source


def test_launcher_powershell_parses_when_powershell_is_available() -> None:
    # Prefer Windows PowerShell 5.1 when both hosts are installed. The launcher
    # targets the inbox Windows shell; pwsh remains a useful fallback on CI.
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
