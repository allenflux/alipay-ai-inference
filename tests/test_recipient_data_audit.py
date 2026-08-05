from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from transfer_receipt_ai.recipient_data_audit import (
    _atomic_write_json,
    build_recipient_data_audit,
    format_recipient_data_audit,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _record(
    receipt_id: str,
    *,
    split: str,
    text: str,
    group_id: str | None = None,
    source: str | None = None,
    result_json: str | None = None,
    receipt_key: str | None = None,
    crop_sha256: str | None = None,
    source_record_id: str | None = None,
) -> dict[str, object]:
    slot: dict[str, object] = {"text": text, "image": f"images/{receipt_id}.png"}
    if crop_sha256 is not None:
        slot["crop_sha256"] = crop_sha256
    if source_record_id is not None:
        slot["source_record_id"] = source_record_id
    record: dict[str, object] = {
        "id": receipt_id,
        "split": split,
        "group_id": group_id or f"group:{receipt_id}",
        "slots": {"recipient_field": slot},
    }
    if source is not None:
        record["source"] = source
    if result_json is not None:
        record["result_json"] = result_json
    if receipt_key is not None:
        record["receipt_key"] = receipt_key
    return record


def _comparison(receipt_id: str, reference_text: str, candidate_text: str, *, split: str = "val") -> dict[str, object]:
    return {
        "id": receipt_id,
        "field": "recipient_field",
        "split": split,
        "reference_text": reference_text,
        "candidate_text": candidate_text,
        "raw_exact": reference_text == candidate_text,
    }


def test_recipient_data_audit_reports_train_name_ceiling_and_integrity(tmp_path: Path) -> None:
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [
            _record("train-a", split="train", text="商户甲", crop_sha256="crop-a", source_record_id="source-a"),
            _record("train-b", split="train", text="商户甲", crop_sha256="crop-b", source_record_id="source-b"),
            _record("train-c", split="train", text="商户乙", crop_sha256="crop-c", source_record_id="source-c"),
            _record("val-known-miss", split="val", text="商户甲", crop_sha256="crop-d", source_record_id="source-d"),
            _record("val-unseen-exact", split="val", text="商户丙", crop_sha256="crop-e", source_record_id="source-e"),
            _record("val-unseen-miss", split="val", text="商户丁", crop_sha256="crop-f", source_record_id="source-f"),
        ],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        comparisons,
        [
            _comparison("val-known-miss", "商户甲", "商户乙"),
            _comparison("val-unseen-exact", "商户丙", "商户丙"),
            _comparison("val-unseen-miss", "商户丁", "商户甲"),
            {"id": "amount", "field": "amount", "reference_text": "1.00", "candidate_text": "1.00"},
        ],
    )
    quality = tmp_path / "recipient_quality_audit.jsonl"
    _write_jsonl(
        quality,
        [
            {"split": "train", "quality_decision": "accepted", "retained_in_unified_manifest": True},
            {
                "split": "val",
                "quality_decision": "rejected",
                "quality_reason": "polluted",
                "retained_in_unified_manifest": False,
            },
        ],
    )

    report = build_recipient_data_audit(
        comparisons_path=comparisons,
        manifest_path=manifest,
        quality_audit_path=quality,
        target_raw_exact_match=0.90,
    )

    assert report["kind"] == "receipt_recipient_data_audit_v1"
    evaluation = report["evaluation"]
    assert evaluation["records"] == 3
    assert evaluation["current_exact_matches"] == 1
    assert evaluation["current_raw_exact_match"] == pytest.approx(1 / 3)
    assert evaluation["incorrect_candidates_equal_some_train_name"] == 2
    coverage = report["train_reference_coverage"]
    assert coverage["held_out_references_seen_in_train"] == 1
    assert coverage["held_out_references_seen_in_train_rate"] == pytest.approx(1 / 3)
    assert coverage["thresholds"][">=1"]["records"] == 1
    assert coverage["thresholds"][">=2"]["records"] == 1
    assert coverage["thresholds"][">=3"]["records"] == 0
    assert coverage["support_bins"]["2-3"]["records"] == 1
    assert coverage["support_bins"]["0"]["records"] == 2
    ceiling = report["omniscient_train_only_closed_set_ceiling"]
    assert ceiling["max_exact_matches"] == 2
    assert ceiling["max_raw_exact_match"] == pytest.approx(2 / 3)
    assert not ceiling["target_reachable_under_oracle"]
    assert report["manifest_summary"]["integrity"]["clean"]
    assert report["quality_audit"]["quality_accepted"] == 1
    assert report["quality_audit"]["quality_rejected"] == 1
    assert report["quality_audit"]["rejected_by_reason"] == {"polluted": 1}
    rendered = format_recipient_data_audit(report)
    assert "oracle closed-set ceiling=2/3=66.67%" in rendered
    assert "potential_gain=1" in rendered


def test_recipient_data_audit_reports_cross_split_and_label_conflicts(tmp_path: Path) -> None:
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(
                "train", split="train", text="商户甲", group_id="same-group", source="same-source",
                result_json="same-result", receipt_key="same-receipt", crop_sha256="same-crop", source_record_id="same-row",
            ),
            _record(
                "val", split="val", text="商户乙", group_id="same-group", source="same-source",
                result_json="same-result", receipt_key="same-receipt", crop_sha256="same-crop", source_record_id="same-row",
            ),
        ],
    )
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(comparisons, [_comparison("val", "商户乙", "商户乙")])

    report = build_recipient_data_audit(comparisons_path=comparisons, manifest_path=manifest)

    integrity = report["manifest_summary"]["integrity"]
    assert not integrity["clean"]
    assert integrity["cross_split_collision_keys"] == 6
    assert integrity["recipient_label_conflict_keys"] == 2
    assert integrity["cross_split_collisions"]["crop_sha256"]["cross_split_keys"] == 1
    assert integrity["recipient_label_conflicts"]["crop_sha256"]["conflicting_keys"] == 1
    assert not report["decision"]["train_only_closed_set_route_eligible_for_target"]


def test_recipient_data_audit_rejects_wrong_manifest_or_inconsistent_comparison(tmp_path: Path) -> None:
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(manifest, [_record("train", split="train", text="商户甲"), _record("val", split="val", text="商户乙")])
    comparisons = tmp_path / "comparisons.jsonl"
    _write_jsonl(comparisons, [_comparison("val", "错误参考", "错误参考")])

    with pytest.raises(ValueError, match="disagrees with manifest"):
        build_recipient_data_audit(comparisons_path=comparisons, manifest_path=manifest)

    _write_jsonl(
        comparisons,
        [{"id": "val", "field": "recipient_field", "reference_text": "商户乙", "candidate_text": "商户甲", "raw_exact": True}],
    )
    with pytest.raises(ValueError, match="raw_exact disagrees"):
        build_recipient_data_audit(comparisons_path=comparisons, manifest_path=manifest)


def test_recipient_data_audit_writes_only_new_json(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    _atomic_write_json(destination, {"schema_version": 1, "message": "只读"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"message": "只读", "schema_version": 1}
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _atomic_write_json(destination, {"schema_version": 1})


def test_rdp_data_audit_wrapper_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed in this local test environment")
    wrapper = Path(__file__).parents[1] / "scripts" / "receipt-ocr-recipient-data-audit-4090.ps1"
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
