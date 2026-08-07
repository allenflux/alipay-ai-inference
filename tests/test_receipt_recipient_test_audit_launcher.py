from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "receipt-recipient-test-audit.ps1"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_reads_latest_v13_test_evidence_without_training_on_test() -> None:
    source = _source()

    assert 'Filter "unified-run-v13-status-cjk-ocr-wide1536-*"' in source
    assert '"onnx-test-gpu\\comparisons.jsonl"' in source
    assert '"manifest-v13"' in source
    assert '"unified_fields.jsonl"' in source
    assert '"recipient-data-audit.py"' in source
    assert '"recipient-slice-report.py"' in source
    assert '"--target", "0.90"' in source
    assert "--skip-geometry" in source
    assert "test split is not used for training" in source
    assert "ocr_unified train" not in source


def test_launcher_refuses_to_overwrite_diagnostic_outputs() -> None:
    source = _source()

    assert '"recipient-test-data-audit.json"' in source
    assert '"recipient-test-slice-report.json"' in source
    assert "Refusing to overwrite recipient diagnostic" in source


def test_launcher_powershell_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
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
