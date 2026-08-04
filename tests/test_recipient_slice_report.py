from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.recipient_slice_report import (
    _atomic_write_json,
    build_recipient_slice_report,
    format_recipient_slice_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _recipient_record(
    receipt_id: str,
    *,
    split: str,
    text: str,
    confidence: float | None,
    image: str,
) -> dict[str, object]:
    slot: dict[str, object] = {"text": text, "image": image}
    if confidence is not None:
        slot["paddle_confidence"] = confidence
    return {"id": receipt_id, "split": split, "slots": {"recipient_field": slot}}


def _write_geometry_images(root: Path) -> tuple[Path, Path]:
    crossing = np.full((8, 20), 255, dtype=np.uint8)
    crossing[:, 7:13] = 0
    crossing_path = root / "crossing.png"
    Image.fromarray(crossing, mode="L").save(crossing_path)

    clear = np.full((8, 20), 255, dtype=np.uint8)
    clear[:, 1:4] = 0
    clear[:, 14:17] = 0
    clear_path = root / "clear.png"
    Image.fromarray(clear, mode="L").save(clear_path)
    return crossing_path, clear_path


def test_recipient_slice_report_joins_support_confidence_and_geometry(tmp_path: Path) -> None:
    crossing, clear = _write_geometry_images(tmp_path)
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [
            _recipient_record("train-one", split="train", text="甲常", confidence=0.99, image="crossing.png"),
            _recipient_record("train-two", split="train", text="常乙", confidence=0.97, image="clear.png"),
            _recipient_record("val-exact", split="val", text="甲常", confidence=0.99, image="crossing.png"),
            _recipient_record("val-oov", split="val", text="乙丙", confidence=0.96, image="clear.png"),
            _recipient_record("val-empty", split="val", text="常常常常常", confidence=None, image="missing.png"),
        ],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        comparisons,
        [
            {"id": "val-exact", "field": "recipient_field", "image": crossing.as_posix(), "reference_text": "甲常", "candidate_text": "甲常", "raw_exact": True, "cer_edits": 0, "reference_has_oov_character": False},
            {"id": "val-oov", "field": "recipient_field", "image": clear.as_posix(), "reference_text": "乙丙", "candidate_text": "乙", "raw_exact": False, "cer_edits": 1, "reference_has_oov_character": True},
            {"id": "val-empty", "field": "recipient_field", "reference_text": "常常常常常", "candidate_text": "", "raw_exact": False, "cer_edits": 5, "reference_has_oov_character": False},
            {"id": "not-recipient", "field": "amount", "reference_text": "1.00", "candidate_text": "1.00"},
        ],
    )

    report = build_recipient_slice_report(
        comparisons_path=comparisons,
        manifest_path=manifest,
        dataset_root=tmp_path,
        left_trim_fraction=0.5,
    )

    assert report["kind"] == "receipt_recipient_slice_report_v1"
    overall = report["overall"]
    assert overall["records"] == 3
    assert overall["exact_matches"] == 1
    assert overall["raw_exact_match"] == pytest.approx(1 / 3)
    assert overall["cer_edits"] == 6
    assert overall["reference_characters"] == 9
    assert overall["micro_cer"] == pytest.approx(6 / 9)
    assert overall["macro_cer"] == pytest.approx((0 / 2 + 1 / 2 + 5 / 5) / 3)
    assert overall["empty_candidate_records"] == 1
    assert overall["empty_candidate_rate"] == pytest.approx(1 / 3)
    assert overall["confidence_records"] == 2
    assert overall["mean_paddle_confidence"] == pytest.approx(0.975)
    assert overall["geometry_records"] == 2
    assert overall["cut_window_ink_records"] == 1
    assert overall["cut_window_ink_rate"] == pytest.approx(1 / 2)
    assert overall["nearest_blank_gap_touch_records"] == 1
    assert overall["nearest_blank_gap_touch_rate"] == pytest.approx(1 / 2)
    slices = report["slices"]
    assert slices["oov"]["oov"]["records"] == 1
    assert slices["oov"]["oov"]["raw_exact_match"] == 0.0
    assert slices["reference_length"]["1-4"]["records"] == 2
    assert slices["reference_length"]["5-8"]["records"] == 1
    assert slices["min_train_character_support"]["0"]["records"] == 1
    assert slices["min_train_character_support"]["1"]["records"] == 1
    assert slices["min_train_character_support"]["2-3"]["records"] == 1
    assert slices["paddle_confidence"][">=0.98"]["records"] == 1
    assert slices["paddle_confidence"]["0.95-<0.98"]["records"] == 1
    assert slices["paddle_confidence"]["missing"]["records"] == 1
    assert slices["cer_edits"]["0"]["records"] == 1
    assert slices["cer_edits"]["1"]["records"] == 1
    assert slices["cer_edits"]["2+"]["records"] == 1
    assert slices["candidate_empty"]["empty"]["records"] == 1
    assert slices["cut_window_ink"]["ink"]["records"] == 1
    assert slices["cut_window_ink"]["no_ink"]["records"] == 1
    assert slices["cut_window_ink"]["unavailable"]["records"] == 1
    assert report["comparison_summary"]["geometry_error_counts"] == {"FileNotFoundError": 1}
    text = format_recipient_slice_report(report)
    assert "strict=33.33%" in text
    assert "[min_train_character_support]" in text
    assert "[cut_window_ink]" in text


def test_recipient_slice_report_rejects_comparison_without_manifest_slot(tmp_path: Path) -> None:
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [_recipient_record("train", split="train", text="甲", confidence=0.99, image="unused.png")],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        comparisons,
        [{"id": "absent", "field": "recipient_field", "reference_text": "甲", "candidate_text": "甲"}],
    )

    with pytest.raises(ValueError, match="absent from manifest"):
        build_recipient_slice_report(
            comparisons_path=comparisons,
            manifest_path=manifest,
            include_geometry=False,
        )


def test_recipient_slice_report_writes_new_json_without_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    payload = {"schema_version": 1, "message": "只读"}

    _atomic_write_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _atomic_write_json(destination, payload)


def test_rdp_powershell_wrapper_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed in this local test environment")
    wrapper = Path(__file__).parents[1] / "scripts" / "receipt-ocr-recipient-slice-4090.ps1"
    escaped_wrapper = wrapper.as_posix().replace("'", "''")
    parser_command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped_wrapper}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )

    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
