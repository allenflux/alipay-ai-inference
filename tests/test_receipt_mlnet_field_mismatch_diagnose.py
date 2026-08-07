from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-field-mismatch-diagnose.py"
)
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_field_mismatch_diagnose", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "field": "payment_method_field",
        "source": r"D:\receipts\one.jpg",
        "reference_text": "余额",
        "candidate_text": "银行卡",
        "candidate_present": True,
        "missing_reason": None,
        "raw_exact": False,
        "detection_score": 0.94,
        "detection_bbox_image": [10.0, 20.0, 100.0, 40.0],
        "reference_detector_score": 0.97,
        "reference_bbox_rectified": [9.0, 19.0, 101.0, 41.0],
        "reference_crop_sha256": "a" * 64,
        "ctc_candidate_text": "银行卡",
        "structured_candidate_text": "银行卡",
        "result_geometry": {"rotation_degrees": 0},
        "manifest_status": "written",
        "result_json": r"D:\results\one.json",
        "teacher_result_json": r"D:\teacher\one.json",
    }
    row.update(overrides)
    return row


def test_payment_diagnosis_prints_only_strict_mismatches_and_summary(tmp_path: Path) -> None:
    comparisons = tmp_path / "score" / "comparisons.jsonl"
    _write_rows(
        comparisons,
        [
            _row(),
            _row(
                source=r"D:\receipts\two.jpg",
                reference_text="储蓄卡",
                candidate_text="储蓄卡",
                raw_exact=True,
            ),
            _row(
                source=r"D:\receipts\missing.jpg",
                candidate_text=None,
                candidate_present=False,
                missing_reason="candidate_missing",
            ),
            _row(
                field="amount",
                source=r"D:\receipts\amount.jpg",
                reference_text="1.00",
                candidate_text="2.00",
            ),
        ],
    )
    original = comparisons.read_bytes()

    lines = MODULE.diagnose(score_dir=comparisons.parent, field="payment")

    mismatches = [json.loads(line.removeprefix("mismatch=")) for line in lines if line.startswith("mismatch=")]
    assert [row["source"] for row in mismatches] == [
        r"D:\receipts\one.jpg",
        r"D:\receipts\missing.jpg",
    ]
    assert mismatches[0]["detector_diagnostics"] == {
        "detection_bbox_image": [10.0, 20.0, 100.0, 40.0],
        "detection_score": 0.94,
        "reference_bbox_rectified": [9.0, 19.0, 101.0, 41.0],
        "reference_crop_sha256": "a" * 64,
        "reference_detector_score": 0.97,
    }
    assert mismatches[0]["field_diagnostics"]["ctc_candidate_text"] == "银行卡"
    assert mismatches[1]["candidate_present"] is False
    assert mismatches[1]["missing_reason"] == "candidate_missing"
    summary = json.loads(next(line.removeprefix("summary=") for line in lines if line.startswith("summary=")))
    assert summary == {
        "candidate_missing_mismatches": 1,
        "comparison_field": "payment_method_field",
        "field": "payment",
        "raw_exact_matches": 1,
        "raw_exact_mismatches": 2,
        "records": 3,
    }
    assert comparisons.read_bytes() == original


def test_brief_diagnosis_omits_large_diagnostics(tmp_path: Path) -> None:
    comparisons = tmp_path / "score" / "comparisons.jsonl"
    _write_rows(comparisons, [_row()])

    lines = MODULE.diagnose(
        score_dir=comparisons.parent,
        field="payment",
        brief=True,
    )

    mismatch = json.loads(
        next(line.removeprefix("mismatch=") for line in lines if line.startswith("mismatch="))
    )
    assert mismatch == {
        "candidate_present": True,
        "candidate_text": "银行卡",
        "difference": {
            "candidate_codepoints": ["U+94F6", "U+884C", "U+5361"],
            "candidate_length": 3,
            "candidate_segment": "银行卡",
            "first_difference_index": 0,
            "reference_codepoints": ["U+4F59", "U+989D"],
            "reference_length": 2,
            "reference_segment": "余额",
        },
        "missing_reason": None,
        "reference_text": "余额",
        "source": r"D:\receipts\one.jpg",
    }


def test_text_difference_isolates_mixed_closing_parenthesis() -> None:
    assert MODULE._text_difference("储蓄卡（1234）", "储蓄卡（1234)") == {
        "candidate_codepoints": ["U+0029"],
        "candidate_length": 9,
        "candidate_segment": ")",
        "first_difference_index": 8,
        "reference_codepoints": ["U+FF09"],
        "reference_length": 9,
        "reference_segment": "）",
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"field":"payment_method_field","field":"amount"}\n', "duplicate JSON key"),
        ('{"field":"payment_method_field","score":NaN}\n', "non-standard JSON constant"),
        ("[]\n", "expected one JSON object"),
        ("\n", "blank JSONL line"),
    ],
)
def test_diagnosis_rejects_non_strict_jsonl(
    tmp_path: Path, contents: str, message: str
) -> None:
    comparisons = tmp_path / "score" / "comparisons.jsonl"
    comparisons.parent.mkdir(parents=True)
    comparisons.write_text(contents, encoding="utf-8")

    with pytest.raises(MODULE.DiagnosisError, match=message):
        MODULE.diagnose(score_dir=comparisons.parent, field="payment")


def test_diagnosis_rejects_inconsistent_raw_exact(tmp_path: Path) -> None:
    comparisons = tmp_path / "score" / "comparisons.jsonl"
    _write_rows(comparisons, [_row(candidate_text="余额", raw_exact=False)])

    with pytest.raises(MODULE.DiagnosisError, match="raw_exact disagrees"):
        MODULE.diagnose(score_dir=comparisons.parent, field="payment")


def test_main_returns_two_for_missing_field_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    comparisons = tmp_path / "score" / "comparisons.jsonl"
    _write_rows(comparisons, [_row(field="amount")])

    exit_code = MODULE.main(["--score-dir", str(comparisons.parent), "--field", "payment"])

    assert exit_code == 2
    assert "diagnosis_error=comparisons contain no rows for field 'payment'" in capsys.readouterr().out
