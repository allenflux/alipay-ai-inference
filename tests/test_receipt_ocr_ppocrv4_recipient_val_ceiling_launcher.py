from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "receipt-ocr-ppocrv4-recipient-val-ceiling-4090.ps1"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_hard_locked_to_val_and_cuda() -> None:
    source = _source()

    assert "[string]$Split" not in source
    assert '[ValidateSet("val", "test")]' not in source
    assert '"--split", "val"' in source
    assert '"--device", "cuda"' in source
    assert ".venv-cu126\\Scripts\\python.exe" in source
    assert "& $pythonPath @arguments" in source


def test_launcher_uses_only_existing_teacher_assets_and_never_trains_or_exports() -> None:
    source = _source()

    assert '"unified-manifest-v12-r3-4090-r1"' in source
    assert '"unified_fields.jsonl"' in source
    assert '"paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1"' in source
    assert '"paddle-recipient-evaluate.py"' in source
    assert '"--bundle"' in source
    assert '"--skip-detection"' in source
    assert "ocr_unified train" not in source
    assert "export-onnx" not in source
    assert "package-delivery" not in source


def test_launcher_refuses_output_reuse() -> None:
    source = _source()

    assert "Test-Path -LiteralPath $OutputDirectory" in source
    assert "Refusing to reuse PP-OCRv4 recipient val output" in source


def test_launcher_powershell_parses_when_available() -> None:
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
