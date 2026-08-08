from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-formal-missing-fields-audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_formal_missing_fields_audit", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = {
    "amount": ("amount", "amount"),
    "time": ("time", "time"),
    "payment_method_field": ("payment_method", "payment_method_field"),
    "recipient_field": ("recipient", "recipient_field"),
    "transfer_status": ("transfer_status", "transfer_status"),
}
VALUES = {
    "amount": "88.00",
    "time": "12:34",
    "payment_method_field": "余额",
    "recipient_field": "商户甲",
    "transfer_status": "转账成功",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _result(
    source: Path, *, missing: set[str], hybrid: bool
) -> dict[str, object]:
    fields: dict[str, object] = {}
    detections: list[dict[str, object]] = []
    for index, (field, (result_key, detection_label)) in enumerate(FIELDS.items()):
        candidate = None if field in missing else VALUES[field]
        value: dict[str, object] = {
            "state": "unreadable" if candidate is None else "review",
            "candidate": candidate,
            "ctc_candidate": candidate,
            "structured_candidate": None,
            "raw": candidate,
        }
        if hybrid and field == "recipient_field":
            value.update(
                {
                    "hybrid_ocr_route": "none" if candidate is None else "primary",
                    "hybrid_ocr_first_raw": "商户甲",
                    "hybrid_ocr_retry_raw": "商户甲",
                }
            )
        fields[result_key] = value
        detections.append(
            {
                "label": detection_label,
                "score": 0.91 + index / 100,
                "bbox_image": [index, 2, 30, 40],
            }
        )
    return {
        "source": str(source),
        "fields": fields,
        "detections": detections,
        "device": {"platform": "android", "confidence": 0.96},
        "geometry": {"rotation_degrees": 0, "rectification": "max-side-1600"},
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "ab"
    score = root / "hybrid-val-score-diagnostic"
    sources = [tmp_path / "inputs" / f"r{index}.jpg" for index in range(4)]
    missing = [
        {"time", "recipient_field"},
        {"time"},
        {"payment_method_field", "transfer_status"},
        set(),
    ]
    manifests: dict[str, Path] = {}
    for run_name, hybrid in (("baseline-v13", False), ("hybrid-recipient", True)):
        run = root / run_name
        manifest: list[dict[str, object]] = []
        for index, source in enumerate(sources):
            result_path = run / "results" / f"r{index}.json"
            _write_json(result_path, _result(source, missing=missing[index], hybrid=hybrid))
            manifest.append(
                {"source": str(source), "result": str(result_path), "status": "written"}
            )
        manifest_path = run / "inference_manifest.json"
        _write_json(manifest_path, manifest)
        manifests[run_name] = manifest_path

    ab_rows = []
    for index, source in enumerate(sources):
        recipient_missing = "recipient_field" in missing[index]
        ab_rows.append(
            {
                "source": str(source),
                "recipient_candidate": None if recipient_missing else "商户甲",
                "invariant": not recipient_missing,
                "failures": (
                    ["hybrid recipient candidate missing"] if recipient_missing else []
                ),
            }
        )
    _write_jsonl(root / "comparison" / "comparisons.jsonl", ab_rows)
    input_list = root / "fixed-selected-inputs.txt"
    input_list.write_text("".join(f"{source}\n" for source in sources), encoding="utf-8")
    _write_json(
        root / "comparison" / "summary.json",
        {
            "schema_version": 2,
            "kind": "receipt_mlnet_hybrid_recipient_cpu_ab_v1",
            "evaluation_mode": "pilot",
            "records": len(sources),
            "invariant_records": 3,
            "recipient_candidate_coverage": 0.75,
            "input_set": {"input_manifest": _identity(input_list)},
            "run_manifests": {
                "baseline": _identity(manifests["baseline-v13"]),
                "hybrid": _identity(manifests["hybrid-recipient"]),
            },
        },
    )

    records = tmp_path / "unified_fields.jsonl"
    absent_references = {
        (0, "recipient_field"),
        (1, "time"),
        (2, "transfer_status"),
    }
    record_rows: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        # Real v13 records use an empty text placeholder for some unlabeled
        # slots.  The scorer excludes those from the accuracy denominator.
        slots = {
            field: {
                "text": "" if (index, field) in absent_references else value
            }
            for field, value in VALUES.items()
        }
        record_rows.append(
            {"id": f"r{index}", "split": "val", "source": str(source), "slots": slots}
        )
    _write_jsonl(records, record_rows)
    model = tmp_path / "unified.onnx"
    model.write_bytes(b"model")

    score_rows: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        for field, value in VALUES.items():
            if (index, field) in absent_references:
                continue
            candidate = None if field in missing[index] else value
            score_rows.append(
                {
                    "schema_version": 1,
                    "kind": "receipt_mlnet_unified_comparison_v1",
                    "source": str(source),
                    "field": field,
                    "reference_text": value,
                    "candidate_text": candidate,
                    "candidate_present": candidate is not None,
                    "missing_reason": None if candidate is not None else "candidate_missing",
                    "raw_exact": candidate == value if candidate is not None else False,
                    "result_json": str(root / "hybrid-recipient" / "results" / f"r{index}.json"),
                }
            )
    _write_jsonl(score / "comparisons.jsonl", score_rows)
    missing_by_field = {
        field: [str(sources[index]) for index in range(len(sources)) if field in missing[index]]
        for field in FIELDS
    }
    reference_counts = {
        field: sum((index, field) not in absent_references for index in range(len(sources)))
        for field in FIELDS
    }
    _write_json(
        score / "summary.json",
        {
            "schema_version": 1,
            "kind": "receipt_mlnet_unified_candidate_evaluation_v1",
            "coverage_contract_version": 2,
            "records": str(records),
            "records_sha256": _sha256(records),
            "results_root": str(root / "hybrid-recipient"),
            "manifest": str(manifests["hybrid-recipient"]),
            "manifest_sha256": _sha256(manifests["hybrid-recipient"]),
            "model": str(model),
            "model_sha256": _sha256(model),
            "evaluation_split": "val",
            "input_selection": {
                "path": str(input_list),
                "sha256": _sha256(input_list),
                "hash_bound": True,
                "records": len(sources),
                "selection_order": "first_unique_source_in_records_manifest_order",
                "field_reference_counts": reference_counts,
            },
            "evaluation_scope": {
                "kind": "full_split",
                "requested_limit": None,
                "evaluated_expected_receipts": len(sources),
                "full_split_expected_receipts": len(sources),
                "input_list_path": str(input_list),
                "input_list_sha256": _sha256(input_list),
                "selection_order": "first_unique_source_in_records_manifest_order",
            },
            "by_field": {
                field: {"records": reference_counts[field]} for field in FIELDS
            },
            "accuracy_denominators": {
                "hash_bound": True,
                "by_field": reference_counts,
            },
            "floors": {
                "amount": 0.7885,
                "time": 0.984,
                "payment_method_field": 0.9325,
                "recipient_field": 0.9,
                "transfer_status": 0.9,
            },
            "all_receipt_candidate_coverage": {
                "by_field": {
                    field: {
                        "expected_receipts": len(sources),
                        "candidate_records": len(sources) - len(values),
                        "missing_candidate_records": len(values),
                    }
                    for field, values in missing_by_field.items()
                }
            },
            "missing": {
                "all_receipt_field_candidates": {
                    field: {"records": len(values), "sources": values}
                    for field, values in missing_by_field.items()
                }
            },
        },
    )
    return root, score, sources


def test_audit_reconstructs_all_field_missing_sets_and_overlap(tmp_path: Path) -> None:
    root, score, sources = _fixture(tmp_path)

    summary, findings, _ = MODULE.audit(
        root=root,
        score=score,
        expected_missing={
            "amount": 0,
            "time": 2,
            "payment_method_field": 1,
            "recipient_field": 1,
            "transfer_status": 1,
        },
    )

    assert summary["records"] == 4
    assert summary["missing_by_field"]["amount"]["records"] == 0
    assert summary["missing_by_field"]["time"] == {
        "records": 2,
        "reference_present_records": 1,
        "reference_missing_records": 1,
        "sources": [str(sources[0]), str(sources[1])],
    }
    assert summary["overlap"]["union_missing_records"] == 3
    assert summary["overlap"]["missing_field_count_distribution"] == {"1": 1, "2": 2}
    exact = {
        tuple(row["fields"]): row["records"]
        for row in summary["overlap"]["exact_missing_field_sets"]
    }
    assert exact == {
        ("time",): 1,
        ("time", "recipient_field"): 1,
        ("payment_method_field", "transfer_status"): 1,
    }
    pairs = {
        tuple(row["fields"]): row["records"] for row in summary["overlap"]["pairwise"]
    }
    assert pairs == {
        ("time", "recipient_field"): 1,
        ("payment_method_field", "transfer_status"): 1,
    }
    first = next(row for row in findings if row["source"] == str(sources[0]))
    assert first["reference_present_by_field"] == {
        "time": True,
        "recipient_field": False,
    }
    assert first["hybrid_device"]["platform"] == "android"
    assert first["hybrid_geometry"]["rotation_degrees"] == 0
    assert first["by_missing_field"]["time"]["hybrid_detection"]["label"] == "time"
    assert first["hybrid_candidate_channels"]["recipient_field"]["hybrid_ocr_route"] == "none"


@pytest.mark.parametrize(
    "field",
    [
        "amount",
        "time",
        "payment_method_field",
        "recipient_field",
        "transfer_status",
    ],
)
def test_empty_slot_text_is_not_a_reference(field: str) -> None:
    assert MODULE._reference_text(field, {"text": ""}) is None


def test_reference_visible_text_selection_matches_scorer() -> None:
    assert MODULE._reference_text(
        "amount", {"text": "12.00", "visible_text": "¥ 12.00"}
    ) == "¥ 12.00"
    assert MODULE._reference_text(
        "amount", {"text": "12.00", "visible_text": "malformed"}
    ) == "12.00"
    assert MODULE._reference_text(
        "amount", {"text": "12.00", "visible_text": ""}
    ) == "12.00"
    assert MODULE._reference_text(
        "amount", {"text": "", "visible_text": "¥ 12.00"}
    ) == "¥ 12.00"
    assert MODULE._reference_text(
        "time", {"text": "12:34", "visible_text": "2026-08-08 12:34"}
    ) == "2026-08-08 12:34"
    assert MODULE._reference_text(
        "time", {"text": "12:34", "visible_text": ""}
    ) == "12:34"
    assert MODULE._reference_text(
        "time", {"text": "", "visible_text": "2026-08-08 12:34"}
    ) == "2026-08-08 12:34"


def test_atomic_output_refuses_overwrite_and_preserves_inputs(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    summary, findings, bindings = MODULE.audit(root=root, score=score)
    output = tmp_path / "audit-output"

    MODULE.write_atomic(output, summary=summary, findings=findings, bindings=bindings)

    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["kind"] == (
        "receipt_mlnet_formal_missing_fields_audit_summary_v1"
    )
    assert len((output / "findings.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert all(path.read_bytes() == contents for path, contents in before.items())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_atomic(output, summary=summary, findings=findings, bindings=bindings)


def test_atomic_output_refuses_to_replace_broken_symlink(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    summary, findings, bindings = MODULE.audit(root=root, score=score)
    output = tmp_path / "audit-output"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_atomic(output, summary=summary, findings=findings, bindings=bindings)

    assert output.is_symlink()


def test_audit_rejects_reordered_input_even_when_hashes_are_rebound(
    tmp_path: Path,
) -> None:
    root, score, sources = _fixture(tmp_path)
    input_list = root / "fixed-selected-inputs.txt"
    input_list.write_text(
        "".join(f"{source}\n" for source in reversed(sources)), encoding="utf-8"
    )
    score_summary = json.loads((score / "summary.json").read_text(encoding="utf-8"))
    score_summary["input_selection"]["sha256"] = _sha256(input_list)
    score_summary["evaluation_scope"]["input_list_sha256"] = _sha256(input_list)
    _write_json(score / "summary.json", score_summary)
    ab_summary = json.loads(
        (root / "comparison" / "summary.json").read_text(encoding="utf-8")
    )
    ab_summary["input_set"]["input_manifest"] = _identity(input_list)
    _write_json(root / "comparison" / "summary.json", ab_summary)

    with pytest.raises(MODULE.AuditError, match="complete canonical"):
        MODULE.audit(root=root, score=score)


def test_audit_rejects_score_result_path_drift(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    comparisons = [
        json.loads(line)
        for line in (score / "comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    comparisons[0]["result_json"] = str(
        root / "hybrid-recipient" / "results" / "r1.json"
    )
    _write_jsonl(score / "comparisons.jsonl", comparisons)

    with pytest.raises(MODULE.AuditError, match="result_json disagrees"):
        MODULE.audit(root=root, score=score)


def test_audit_rejects_ab_candidate_drift(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    comparisons = [
        json.loads(line)
        for line in (root / "comparison" / "comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    comparisons[0]["recipient_candidate"] = "错误候选"
    _write_jsonl(root / "comparison" / "comparisons.jsonl", comparisons)

    with pytest.raises(MODULE.AuditError, match="recipient candidate disagrees"):
        MODULE.audit(root=root, score=score)


def test_audit_rejects_scorer_missing_source_drift(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    summary_path = score / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["missing"]["all_receipt_field_candidates"]["time"] = {
        "records": 0,
        "sources": [],
    }
    payload["all_receipt_candidate_coverage"]["by_field"]["time"] = {
        "expected_receipts": 4,
        "candidate_records": 4,
        "missing_candidate_records": 0,
    }
    _write_json(summary_path, payload)

    with pytest.raises(MODULE.AuditError, match="disagrees with hybrid results"):
        MODULE.audit(root=root, score=score)


def test_atomic_output_rejects_result_mutation_after_audit(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)
    summary, findings, bindings = MODULE.audit(root=root, score=score)
    result = root / "hybrid-recipient" / "results" / "r0.json"
    result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")
    output = tmp_path / "must-not-publish"

    with pytest.raises(MODULE.AuditError, match="hybrid result changed"):
        MODULE.write_atomic(output, summary=summary, findings=findings, bindings=bindings)

    assert not output.exists()


def test_require_formal_rejects_non_10016_input(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)

    with pytest.raises(MODULE.AuditError, match="requires exactly 10016"):
        MODULE.audit(root=root, score=score, require_formal=True)


def test_expected_missing_is_fail_closed(tmp_path: Path) -> None:
    root, score, _ = _fixture(tmp_path)

    with pytest.raises(MODULE.AuditError, match="expected time missing=339, observed 2"):
        MODULE.audit(root=root, score=score, expected_missing={"time": 339})
