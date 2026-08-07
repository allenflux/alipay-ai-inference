from __future__ import annotations

import json
from pathlib import Path

import pytest

from transfer_receipt_ai.recipient_error_forensics import (
    _atomic_write_json,
    align_recipient_text,
    build_recipient_error_forensics,
    format_recipient_error_forensics,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(receipt_id: str, *, split: str, text: str, image: str) -> dict[str, object]:
    return {
        "id": receipt_id,
        "split": split,
        "slots": {"recipient_field": {"text": text, "image": image}},
    }


def _comparison(
    receipt_id: str,
    reference: str,
    candidate: str,
    *,
    image: str,
) -> dict[str, object]:
    edits = sum(item["operation"] != "equal" for item in align_recipient_text(reference, candidate))
    return {
        "id": receipt_id,
        "field": "recipient_field",
        "split": "test",
        "image": image,
        "reference_text": reference,
        "candidate_text": candidate,
        "confidence": 0.91,
        "raw_exact": reference == candidate,
        "cer_edits": edits,
        "reference_characters": len(reference),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [
            _record("train-a", split="train", text="甲乙乙常", image="train-a.png"),
            _record("train-b", split="train", text="乙丙常", image="train-b.png"),
            _record("test-sub", split="test", text="甲乙", image="fallback-sub.png"),
            _record("test-del", split="test", text="乙乙", image="fallback-del.png"),
            _record("test-ins", split="test", text="丁常", image="fallback-ins.png"),
            _record("test-exact", split="test", text="常常常常常", image="fallback-exact.png"),
        ],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        comparisons,
        [
            _comparison("test-sub", "甲乙", "甲丙", image="D:/crops/sub.png"),
            _comparison("test-del", "乙乙", "乙", image="D:/crops/del.png"),
            _comparison("test-ins", "丁常", "火丁常", image="D:/crops/ins.png"),
            _comparison("test-exact", "常常常常常", "常常常常常", image="D:/crops/exact.png"),
            {
                "id": "other-field",
                "field": "amount",
                "reference_text": "1.00",
                "candidate_text": "1.00",
            },
        ],
    )
    return manifest, comparisons


def test_align_recipient_text_reports_deterministic_edit_types() -> None:
    substitution = align_recipient_text("甲乙", "甲丙")
    deletion = align_recipient_text("甲乙", "甲")
    insertion = align_recipient_text("甲", "乙甲")

    assert [item["operation"] for item in substitution] == ["equal", "substitution"]
    assert substitution[1]["reference_character"] == "乙"
    assert substitution[1]["candidate_character"] == "丙"
    assert [item["operation"] for item in deletion] == ["equal", "deletion"]
    assert deletion[1]["reference_character"] == "乙"
    assert [item["operation"] for item in insertion] == ["insertion", "equal"]
    assert insertion[0]["candidate_character"] == "乙"


def test_recipient_error_forensics_ranks_edits_and_keeps_representative_paths(tmp_path: Path) -> None:
    manifest, comparisons = _fixture(tmp_path)

    report = build_recipient_error_forensics(
        comparisons_path=comparisons,
        manifest_path=manifest,
        split="test",
        top=10,
        examples_per_operation=2,
        representative_limit=10,
    )

    assert report["kind"] == "receipt_recipient_error_forensics_v1"
    overall = report["overall"]
    assert overall["records"] == 4
    assert overall["exact_matches"] == 1
    assert overall["raw_exact_match"] == pytest.approx(0.25)
    assert overall["cer_edits"] == 3
    assert overall["reference_characters"] == 11
    assert overall["micro_cer"] == pytest.approx(3 / 11)
    assert overall["substitutions"] == 1
    assert overall["deletions"] == 1
    assert overall["insertions"] == 1
    assert overall["empty_candidate_records"] == 0
    assert overall["empty_candidate_rate"] == 0.0
    assert overall["operation_count_consistent"] is True

    operations = report["top_edit_operations"]
    assert operations["substitutions"][0]["reference_character"] == "乙"
    assert operations["substitutions"][0]["candidate_character"] == "丙"
    assert operations["substitutions"][0]["reference_train_support"] == 3
    assert operations["deletions"][0]["reference_character"] == "乙"
    assert operations["insertions"][0]["candidate_character"] == "火"
    assert operations["insertions"][0]["examples"][0]["image"] == "D:/crops/ins.png"

    slices = report["record_slices"]
    assert slices["reference_length"]["1-4"]["records"] == 3
    assert slices["reference_length"]["5-8"]["records"] == 1
    assert slices["minimum_train_character_support"]["0"]["records"] == 1
    assert slices["minimum_train_character_support"]["1"]["records"] == 1
    assert slices["minimum_train_character_support"]["2-3"]["records"] == 2

    character_slices = report["reference_character_support_slices"]
    assert character_slices["0"]["reference_characters"] == 1
    assert character_slices["0"]["correct_characters"] == 1
    assert character_slices["2-3"]["substitutions"] == 1
    assert character_slices["2-3"]["deletions"] == 1

    representative = report["representative_misses"]
    assert {item["id"] for item in representative} == {"test-sub", "test-del", "test-ins"}
    assert all(item["image"].startswith("D:/crops/") for item in representative)
    rendered = format_recipient_error_forensics(report)
    assert "exact=1/4=25.00%" in rendered
    assert "[top_substitutions]" in rendered
    assert '"甲乙" -> "甲丙"' in rendered
    assert "image=D:/crops/sub.png" in rendered


def test_recipient_error_forensics_rejects_untrustworthy_join_and_metrics(tmp_path: Path) -> None:
    manifest, comparisons = _fixture(tmp_path)
    rows = [json.loads(line) for line in comparisons.read_text(encoding="utf-8").splitlines()]
    rows[0]["reference_text"] = "错误标签"
    _write_jsonl(comparisons, rows)

    with pytest.raises(ValueError, match="reference_text disagrees"):
        build_recipient_error_forensics(comparisons_path=comparisons, manifest_path=manifest)

    _, comparisons = _fixture(tmp_path)
    rows = [json.loads(line) for line in comparisons.read_text(encoding="utf-8").splitlines()]
    rows[0]["cer_edits"] = 99
    _write_jsonl(comparisons, rows)
    with pytest.raises(ValueError, match="cer_edits disagrees"):
        build_recipient_error_forensics(comparisons_path=comparisons, manifest_path=manifest)


def test_recipient_error_forensics_rejects_partial_held_out_denominator(tmp_path: Path) -> None:
    manifest, comparisons = _fixture(tmp_path)
    rows = [json.loads(line) for line in comparisons.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(
        comparisons,
        [row for row in rows if row.get("id") not in {"test-exact", "other-field"}],
    )

    with pytest.raises(ValueError, match="denominator 3 disagrees with manifest test recipient records 4"):
        build_recipient_error_forensics(comparisons_path=comparisons, manifest_path=manifest)


def test_recipient_error_forensics_output_is_new_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "recipient-test-error-forensics.json"
    payload = {"schema_version": 1, "kind": "只读取证"}

    _atomic_write_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _atomic_write_json(destination, payload)
