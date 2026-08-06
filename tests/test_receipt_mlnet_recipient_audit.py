from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "receipt-mlnet-recipient-audit.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_recipient_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _teacher(
    path: Path,
    *,
    rotation: int,
    screen_detected: bool,
    source_size: tuple[int, int],
    rectified_size: tuple[int, int],
    homography: list[list[float]],
    bbox: list[float],
) -> None:
    _write_json(
        path,
        {
            "geometry": {
                "rotation_degrees": rotation,
                "screen_detected": screen_detected,
                "source_size": {"width": source_size[0], "height": source_size[1]},
                "rectified_size": {"width": rectified_size[0], "height": rectified_size[1]},
                "H_original_to_rectified": homography,
            },
            "detections": [
                {"label": "recipient_field", "score": 0.99, "bbox_rectified": bbox}
            ],
        },
    )


def test_project_bbox_uses_all_four_corners_and_levenshtein_is_unicode_safe() -> None:
    projected = AUDIT._project_bbox(
        [1.0, 2.0, 5.0, 6.0],
        [[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]],
    )
    assert projected == pytest.approx([3.0, 5.0, 11.0, 17.0])
    assert AUDIT._levenshtein("商户甲", "商户乙") == 1
    assert AUDIT._levenshtein("ＡＢＣ", "ABC") == 3


def test_recipient_audit_aggregates_text_geometry_and_exact_mismatch_alignment(
    tmp_path: Path,
) -> None:
    exact_teacher = tmp_path / "exact-teacher.json"
    recovered_teacher = tmp_path / "recovered-teacher.json"
    _teacher(
        exact_teacher,
        rotation=0,
        screen_detected=False,
        source_size=(100, 200),
        rectified_size=(100, 200),
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        bbox=[10.0, 20.0, 30.0, 40.0],
    )
    _teacher(
        recovered_teacher,
        rotation=90,
        screen_detected=True,
        source_size=(100, 200),
        rectified_size=(200, 100),
        homography=[[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]],
        bbox=[4.0, 5.0, 12.0, 17.0],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        comparisons,
        [
            {
                "id": "exact",
                "field": "recipient_field",
                "source": "exact.png",
                "reference_text": "商户甲",
                "candidate_text": "商户甲",
                "raw_exact": True,
                "teacher_result_json": exact_teacher.as_posix(),
                "detection_bbox_image": [10.0, 20.0, 30.0, 40.0],
            },
            {
                "id": "normalised-recovery",
                "field": "recipient_field",
                "source": "recovered.png",
                "reference_text": "ABC商户",
                "candidate_text": "ＡＢＣ商户 ",
                "raw_exact": False,
                "teacher_result_json": recovered_teacher.as_posix(),
                "detection_bbox_image": [1.0, 2.0, 5.0, 6.0],
            },
            {
                "id": "missing-geometry",
                "field": "recipient_field",
                "source": "missing.png",
                "reference_text": "商户乙",
                "candidate_text": "商户丙",
                "raw_exact": False,
                "teacher_result_json": (tmp_path / "absent.json").as_posix(),
                "detection_bbox_image": [1.0, 2.0, 5.0, 6.0],
            },
            {
                "id": "ignored",
                "field": "amount",
                "reference_text": "1.00",
                "candidate_text": "1.00",
            },
        ],
    )
    summary = tmp_path / "summary.json"
    _write_json(
        summary,
        {
            "kind": "partial",
            "formal_delivery_gate": False,
            "pilot_thresholds_passed": True,
            "by_field": {
                "amount": {
                    "raw_exact_matches": 8,
                    "records": 10,
                    "raw_exact_match": 0.8,
                    "candidate_records": 10,
                    "candidate_coverage": 1.0,
                },
                "recipient_field": {
                    "raw_exact_matches": 1,
                    "records": 3,
                    "raw_exact_match": 1 / 3,
                    "candidate_records": 3,
                    "candidate_coverage": 1.0,
                },
            },
            "amount_semantic": {
                "records": 10,
                "exact_matches": 9,
                "exact_match": 0.9,
                "diagnostic_only": True,
                "affects_acceptance": False,
            },
        },
    )

    report = AUDIT.build_recipient_audit(
        comparisons_path=comparisons, summary_path=summary, worst_limit=5
    )

    assert report["kind"] == "receipt_mlnet_recipient_audit_v1"
    text = report["text"]
    assert text["records"] == 3
    assert text["strict_exact_matches"] == 1
    assert text["nfkc_trim_exact_matches"] == 2
    assert text["nfkc_trim_recovered_matches"] == 1
    assert text["raw_edit_distance"]["distribution"] == {"0": 1, "1": 1, "4": 1}
    assert text["nfkc_trim_edit_distance"]["distribution"] == {"0": 2, "1": 1}

    geometry = report["teacher_geometry"]
    assert geometry["all"]["geometry_records"] == 2
    assert geometry["all"]["rotation_degrees"] == {"0": 1, "90": 1, "missing": 1}
    assert geometry["all"]["screen_detected"] == {"false": 1, "missing": 1, "true": 1}
    assert geometry["all"]["source_size"] == {"100x200": 2, "missing": 1}
    assert geometry["strict_exact"]["rotation_degrees"] == {"0": 1}
    assert geometry["strict_mismatch"]["rotation_degrees"] == {"90": 1, "missing": 1}

    alignment = report["bbox_alignment"]
    assert alignment["all"]["available_records"] == 2
    assert alignment["strict_exact"]["available_records"] == 1
    assert alignment["strict_exact"]["iou"]["mean"] == 1.0
    assert alignment["strict_mismatch"]["records"] == 2
    assert alignment["strict_mismatch"]["available_records"] == 1
    assert alignment["strict_mismatch"]["iou"]["mean"] == pytest.approx(7 / 9)
    assert alignment["strict_mismatch"]["mean_absolute_edge_deviation_per_record_px"][
        "mean"
    ] == pytest.approx(0.5)
    assert alignment["strict_mismatch"]["missing_by_reason"] == {
        "teacher_result_not_found": 1
    }
    assert report["teacher_result_errors"] == {"teacher_result_not_found": 1}
    assert report["evaluation_snapshot"]["amount_semantic"]["exact_match"] == 0.9

    rendered = AUDIT.format_recipient_audit(report)
    assert "amount=8/10=80.00%" in rendered
    assert "amount_semantic=9/10=90.00%" in rendered
    assert "strict=1/3=33.33%" in rendered
    assert "strict_mismatch: available=1/2=50.00%" in rendered


def test_main_writes_default_report_and_replaces_stale_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    teacher = tmp_path / "teacher.json"
    _teacher(
        teacher,
        rotation=0,
        screen_detected=False,
        source_size=(10, 10),
        rectified_size=(10, 10),
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        bbox=[1.0, 1.0, 9.0, 9.0],
    )
    _write_jsonl(
        tmp_path / "comparisons.jsonl",
        [
            {
                "id": "one",
                "field": "recipient_field",
                "reference_text": "甲",
                "candidate_text": "甲",
                "teacher_result_json": teacher.as_posix(),
                "detection_bbox_image": [1.0, 1.0, 9.0, 9.0],
            }
        ],
    )
    destination = tmp_path / "recipient-audit.json"
    destination.write_text("stale", encoding="utf-8")

    assert AUDIT.main(["--evaluation-dir", str(tmp_path)]) == 0

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["text"]["strict_exact_match"] == 1.0
    output = capsys.readouterr().out
    assert "summary.json not found" in output
    assert f"recipient_audit_json={destination}" in output
