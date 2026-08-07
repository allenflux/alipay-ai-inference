from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "receipt-recipient-test-geometry.ps1"


def test_launcher_reads_test_crops_without_changing_models_or_truth() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '"onnx-test-gpu\\comparisons.jsonl"' in source
    assert '"manifest-v13\\unified_fields.jsonl"' in source
    assert '"recipient-slice-report.py"' in source
    assert "--dataset-root" in source
    assert "--left-trim 0.30" in source
    assert "--skip-geometry" not in source
    assert "mode=read-only image evidence" in source
    assert "ocr_unified train" not in source


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
