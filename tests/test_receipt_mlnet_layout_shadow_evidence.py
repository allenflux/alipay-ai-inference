from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-layout-shadow-evidence.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_layout_shadow_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> bytes:
    data = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    path.write_bytes(data)
    return data


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    data = b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.write_bytes(data)
    return data


def _identity(path: Path, payload: bytes | None = None) -> dict[str, object]:
    data = path.read_bytes() if payload is None else payload
    return {"path": str(path.resolve()), "sha256": _sha(data), "size_bytes": len(data)}


def _selection_closure(identities: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in identities:
        path = Path(str(item["path"])).resolve()
        key = os.path.normcase(os.path.normpath(str(path)))
        digest.update(
            f"{key}\0{item['path']}\0{item['sha256']}\0{item['size_bytes']}\n".encode()
        )
    return digest.hexdigest()


def _quad(x1: float, y1: float, x2: float, y2: float) -> tuple[list[list[float]], list[list[float]]]:
    raw = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    normalized = [[x / 999.0, y / 1599.0] for x, y in raw]
    return raw, normalized


def _line(index: int, text: str, confidence: float, box: tuple[float, float, float, float]) -> dict[str, object]:
    raw, normalized = _quad(*box)
    return {
        "index": index,
        "text": text,
        "confidence": confidence,
        "passes_drop_score": confidence >= 0.5,
        "quad_rectified": raw,
        "quad_rectified_normalized": normalized,
    }


def _record(index: int, source: Path, lines: list[dict[str, object]], *, rotation: int = 0) -> dict[str, object]:
    accepted = [line for line in lines if line["passes_drop_score"]]
    accepted_text = " ".join(str(line["text"]).strip() for line in accepted if str(line["text"]).strip())
    return {
        "schema_version": 1,
        "kind": MODULE.LAYOUT_RECORD_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "index": index,
        "source": str(source.resolve()),
        "source_image_sha256": _sha(source.read_bytes()),
        "source_image_size_bytes": source.stat().st_size,
        "execution_provider": "cpu",
        "geometry": {
            "source_size": {"width": 1000, "height": 1600},
            "rectified_size": {"width": 1000, "height": 1600},
            "rectification": "max-side-1600",
            "rotation_degrees": rotation,
            "screen_detected": False,
            "screen_quad_original": [[0, 0], [999, 0], [999, 1599], [0, 1599]],
            "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "H_rectified_to_original": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "quad_coordinate_space": MODULE.QUAD_COORDINATE_SPACE,
        "quad_normalization": MODULE.QUAD_NORMALIZATION,
        "confidence_semantics": MODULE.CONFIDENCE_SEMANTICS,
        "accepted_text": accepted_text,
        "accepted_confidence": (
            sum(float(line["confidence"]) for line in accepted) / len(accepted)
            if accepted else None
        ),
        "accepted_line_count": len(accepted),
        "raw_line_count": len(lines),
        "lines": lines,
        "timing_ms": {"image_load": 1.0, "rectification": 2.0, "layout_ocr": 3.0, "total": 6.5},
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    audit = tmp_path / "audit"
    selection = tmp_path / "selection"
    layout = tmp_path / "layout"
    images = tmp_path / "images"
    for directory in (audit, selection, layout, images):
        directory.mkdir(parents=True)
    sources: list[Path] = []
    for index in range(MODULE.EXPECTED_RECORDS):
        source = images / f"receipt-{index:03d}.jpg"
        source.write_bytes(f"source-{index}\n".encode())
        sources.append(source.resolve())

    field_sources = {
        "time": sources,
        "payment_method_field": [sources[0]],
        "transfer_status": [sources[0]],
    }
    audit_summary = {
        "schema_version": 1,
        "kind": MODULE.AUDIT_SUMMARY_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "formal_required": True,
        "records": MODULE.FORMAL_RECORDS,
        "missing_by_field": {
            field: {
                "records": len(selected),
                "reference_present_records": 0,
                "reference_missing_records": len(selected),
                "sources": [str(source) for source in selected],
            }
            for field, selected in field_sources.items()
        },
        "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
    }
    audit_summary_bytes = _write_json(audit / "summary.json", audit_summary)
    findings = []
    for index, source in enumerate(sources):
        fields = ["time"] if index else ["time", "payment_method_field", "transfer_status"]
        findings.append({
            "schema_version": 1,
            "kind": MODULE.AUDIT_FINDING_KIND,
            "source": str(source),
            "missing_fields": fields,
            "reference_present_by_field": {field: False for field in fields},
            "by_missing_field": {
                field: {
                    "reference_present": False,
                    "reference_text": None,
                    "score_comparison": None,
                }
                for field in fields
            },
        })
    audit_findings_bytes = _write_jsonl(audit / "findings.jsonl", findings)

    inputs_bytes = ("\n".join(str(source) for source in sources) + "\n").encode()
    (selection / "inputs.txt").write_bytes(inputs_bytes)
    source_ids = [_identity(source) for source in sources]
    selection_payload = {
        "schema_version": 1,
        "kind": MODULE.SELECTION_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "selection_field": "time",
        "selection_order": "formal_audit_missing_by_field_time_sources_order",
        "records": MODULE.EXPECTED_RECORDS,
        "external_reference_present_records": 0,
        "external_reference_missing_records": MODULE.EXPECTED_RECORDS,
        "formal_audit": {
            "directory": str(audit.resolve()),
            "summary": _identity(audit / "summary.json", audit_summary_bytes),
            "findings": _identity(audit / "findings.jsonl", audit_findings_bytes),
            "records": MODULE.FORMAL_RECORDS,
        },
        "input_list": {
            "relative_path": "inputs.txt",
            "sha256": _sha(inputs_bytes),
            "size_bytes": len(inputs_bytes),
            "records": MODULE.EXPECTED_RECORDS,
            "encoding": "utf-8-no-bom",
            "terminal_newline": True,
        },
        "source_files": source_ids,
        "source_closure_sha256": _selection_closure(source_ids),
        "source_total_bytes": sum(int(item["size_bytes"]) for item in source_ids),
    }
    _write_json(selection / "selection.json", selection_payload)

    records: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        if index == 0:
            lines = [
                _line(0, "05:49", 0.96, (30, 20, 140, 60)),
                _line(1, "付款方式 余额", 0.93, (100, 500, 400, 550)),
                _line(2, "转账成功", 0.94, (350, 300, 650, 360)),
            ]
        elif index == 1:
            lines = [
                _line(0, "2026-08-08 12:34", 0.9, (100, 500, 500, 550)),
                _line(1, "12:34:56", 0.9, (100, 600, 300, 650)),
            ]
        else:
            lines = [_line(0, "回单", 0.9, (100, 400, 300, 450))]
        records.append(_record(index, source, lines))
    records_bytes = _write_jsonl(layout / "records.jsonl", records)
    layout_summary = {
        "schema_version": 1,
        "kind": MODULE.LAYOUT_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "expected_records": MODULE.EXPECTED_RECORDS,
        "records": MODULE.EXPECTED_RECORDS,
        "errors": 0,
        "execution_provider": "cpu",
        "rectification": MODULE.RECTIFICATION,
        "quad_coordinate_space": MODULE.QUAD_COORDINATE_SPACE,
        "quad_normalization": MODULE.QUAD_NORMALIZATION,
        "confidence_semantics": MODULE.CONFIDENCE_SEMANTICS,
        "paddle_drop_score": 0.5,
        "input_list": {
            "path": str((selection / "inputs.txt").resolve()),
            "sha256": _sha(inputs_bytes),
            "size_bytes": len(inputs_bytes),
            "records": MODULE.EXPECTED_RECORDS,
        },
        "paddle_bundle": {"directory": str((tmp_path / "bundle").resolve())},
        "latency_ms": {"total": {"count": MODULE.EXPECTED_RECORDS, "p95": 6.5}},
        "artifacts": {
            "records_jsonl": {
                "relative_path": "records.jsonl",
                "sha256": _sha(records_bytes),
                "size_bytes": len(records_bytes),
            }
        },
    }
    _write_json(layout / "summary.json", layout_summary)
    return {
        "audit": audit,
        "selection": selection,
        "layout": layout,
        "sources": sources,
        "records": records,
        "layout_summary": layout_summary,
    }


def _prepare(fixture: dict[str, object]):
    return MODULE.prepare_analysis(
        selection_directory=fixture["selection"],
        audit_directory=fixture["audit"],
        layout_directory=fixture["layout"],
    )


def test_prepare_and_atomic_publish_read_only_field_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary, evidence_bytes, bindings = _prepare(fixture)
    assert summary["diagnostic_only"] is True
    assert summary["formal_delivery_gate"] is False
    assert summary["candidate_write_enabled"] is False
    assert summary["records"] == 339
    assert summary["errors"] == 0
    assert summary["execution_provider"] == "cpu"
    assert summary["analysis_contract"]["time"]["semantic_scope"] == (
        "visible_screen_status_bar_clock_only"
    )
    assert summary["analysis_contract"]["time"]["source_and_rectified_full_image_top_fraction"] == 0.08
    assert summary["status_safety"] == {
        "field_candidate_writes": 0,
        "success_acceptances": 0,
        "non_success_to_success": 0,
    }
    assert summary["coverage_by_field"]["time"]["audit_missing_target"][
        "records_with_unique_diagnostic_coverage"
    ] == 1
    assert summary["coverage_by_field"]["time"]["audit_missing_target"][
        "time_evidence_counts"
    ]["records_with_exact_clock_anywhere"] == 1
    assert summary["coverage_by_field"]["time"]["audit_missing_target"][
        "time_evidence_counts"
    ]["records_with_excluded_time_like_text"] == 1
    assert summary["coverage_by_field"]["payment_method_field"]["audit_missing_target"][
        "records_with_unique_diagnostic_coverage"
    ] == 1
    assert summary["coverage_by_field"]["transfer_status"]["audit_missing_target"][
        "records_with_unique_diagnostic_coverage"
    ] == 1

    rows = [json.loads(line) for line in evidence_bytes.decode().splitlines()]
    assert len(rows) == 339
    first_time = rows[0]["evidence_by_field"]["time"]
    assert first_time["ambiguity"] == "unique_top8_status_bar_anchor_with_body_support"
    assert first_time["anchors"][0]["source_top8_membership"] is True
    assert first_time["anchors"][0]["rectified_top8_membership"] is True
    assert len(first_time["anchors"][0]["accepted_body_anchors_below"]) == 2
    assert rows[1]["evidence_by_field"]["time"]["anchor_count"] == 0
    assert rows[1]["evidence_by_field"]["time"]["excluded_time_like_count"] == 2

    output = tmp_path / "evidence-output"
    MODULE.write_atomic(output, summary=summary, evidence_bytes=evidence_bytes, bindings=bindings)
    assert sorted(path.name for path in output.iterdir()) == ["evidence.jsonl", "summary.json"]
    published = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert published == summary
    assert published["artifacts"]["evidence_jsonl"]["sha256"] == _sha(
        (output / "evidence.jsonl").read_bytes()
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("records", 338, "records must be 339"),
        ("errors", 1, "errors must be 0"),
        ("execution_provider", "cuda:0", "must be cpu"),
        ("candidate_write_enabled", True, "must be false"),
    ],
)
def test_rejects_nonformal_layout_summary(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    fixture = _fixture(tmp_path)
    path = Path(fixture["layout"]) / "summary.json"
    summary = fixture["layout_summary"]
    summary[key] = value
    _write_json(path, summary)
    with pytest.raises(MODULE.EvidenceError, match=message):
        _prepare(fixture)


def test_rejects_layout_records_hash_order_and_source_identity_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    records_path = Path(fixture["layout"]) / "records.jsonl"
    records = fixture["records"]
    records[0]["source"] = records[1]["source"]
    _write_jsonl(records_path, records)
    with pytest.raises(MODULE.EvidenceError, match="records SHA-256 differs"):
        _prepare(fixture)

    # Bind the mutated artifact in summary; order validation must still reject it.
    summary_path = Path(fixture["layout"]) / "summary.json"
    summary = fixture["layout_summary"]
    data = records_path.read_bytes()
    summary["artifacts"]["records_jsonl"]["sha256"] = _sha(data)
    summary["artifacts"]["records_jsonl"]["size_bytes"] = len(data)
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.EvidenceError, match="source/order differs"):
        _prepare(fixture)


def test_rejects_selection_audit_and_source_closure_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection_path = Path(fixture["selection"]) / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["external_reference_present_records"] = 1
    _write_json(selection_path, selection)
    with pytest.raises(MODULE.EvidenceError, match="external-reference-present"):
        _prepare(fixture)

    fixture = _fixture(tmp_path / "second")
    source = fixture["sources"][0]
    source.write_bytes(b"changed")
    with pytest.raises(MODULE.EvidenceError, match=r"selection source\[0\].*differs"):
        _prepare(fixture)


def test_time_parser_never_promotes_seconds_datetime_embedded_or_reverse(tmp_path: Path) -> None:
    def prepared(text: str, index: int) -> dict[str, object]:
        line = _line(index, text, 0.9, (30, 20 + index * 20, 300, 35 + index * 20))
        line["_rotation_degrees"] = 0
        line["_geometry"] = {
            "rectified_normalized": {"x_min": 0.03, "x_max": 0.3, "y_min": 0.01, "y_max": 0.02,
                                       "x_center": 0.16, "y_center": 0.015, "width": 0.27, "height": 0.01},
            "source_normalized": {"x_min": 0.03, "x_max": 0.3, "y_min": 0.01, "y_max": 0.02,
                                  "x_center": 0.16, "y_center": 0.015, "width": 0.27, "height": 0.01},
            "rectified_top8_membership": True,
            "source_top8_membership": True,
            "top8_membership_disagrees": False,
        }
        return line

    lines = [
        prepared("2026-08-08 12:34", 0),
        prepared("12:34:56", 1),
        prepared("交易时间 12:34", 2),
        prepared("80:00", 3),
    ]
    evidence = MODULE._time_evidence(lines)
    assert evidence["anchor_count"] == 0
    assert evidence["excluded_time_like_count"] == 4
    assert evidence["reverse_clock_repair_applied"] is False
    assert evidence["unique_diagnostic_coverage"] is False


def test_status_and_payment_rules_are_fail_closed() -> None:
    assert MODULE._payment_value_grammar("余额") == "balance"
    assert MODULE._payment_value_grammar("邮储银行储蓄卡(8885)") == "bank_card_tail4"
    assert MODULE._payment_value_grammar("邮储银行储蓄卡（8885)") is None
    assert MODULE._payment_value_grammar("任意开放文本") is None

    def line(text: str, index: int) -> dict[str, object]:
        return {
            "index": index,
            "text": text,
            "confidence": 0.9,
            "passes_drop_score": True,
            "quad_rectified_normalized": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "_rotation_degrees": 0,
            "_geometry": {
                "rectified_normalized": {"x_min": 0.1, "x_max": 0.5, "y_min": 0.3, "y_max": 0.4,
                                           "x_center": 0.3, "y_center": 0.35, "width": 0.4, "height": 0.1},
                "source_normalized": {"x_min": 0.1, "x_max": 0.5, "y_min": 0.3, "y_max": 0.4,
                                      "x_center": 0.3, "y_center": 0.35, "width": 0.4, "height": 0.1},
                "rectified_top8_membership": False,
                "source_top8_membership": False,
                "top8_membership_disagrees": False,
            },
        }

    blocked = MODULE._status_evidence([line("并未确认转账成功", 0)])
    assert blocked["blocked_success_evidence"] is True
    assert blocked["success_acceptance_enabled"] is False
    assert blocked["unique_diagnostic_coverage"] is False
    ambiguous = MODULE._status_evidence([line("转账成功", 0), line("转账失败", 1)])
    assert ambiguous["ambiguity"] == "multiple_distinct_status_phrases"
    assert ambiguous["unique_diagnostic_coverage"] is False


def test_homography_validation_accepts_realistic_cancellation_residual() -> None:
    source_width, source_height = 1080, 2400
    rectified_width, rectified_height = 720, 1600
    scale_x = (rectified_width - 1) / (source_width - 1)
    scale_y = (rectified_height - 1) / (source_height - 1)
    forward = [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]]
    # A five-micro-pixel translation residual makes a literal A^-1*A
    # comparison fail at abs_tol=1e-6, while remaining far below any observable
    # pixel displacement. This models cancellation around scaled coordinates.
    inverse = [[1.0 / scale_x, 0.0, 5e-6], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]]
    MODULE._require_homography_pair(
        forward,
        inverse,
        source_width=source_width,
        source_height=source_height,
        rectified_width=rectified_width,
        rectified_height=rectified_height,
        rotation_degrees=0,
        description="realistic homography",
    )


def test_intrinsically_degenerate_db_quad_is_diagnosed_only_in_explicit_exclusion_mode() -> None:
    raw = [[10.0, 20.0], [30.0, 20.0], [30.0, 20.0], [10.0, 20.0]]
    normalized = [[x / 99.0, y / 99.0] for x, y in raw]
    line = {
        "quad_rectified": raw,
        "quad_rectified_normalized": normalized,
    }
    with pytest.raises(MODULE.EvidenceError, match="quad is degenerate"):
        MODULE._quad_geometry(
            line,
            record_index=146,
            line_index=20,
            rectified_width=100,
            rectified_height=100,
            source_width=100,
            source_height=100,
            rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

    geometry = MODULE._quad_geometry(
        line,
        record_index=146,
        line_index=20,
        rectified_width=100,
        rectified_height=100,
        source_width=100,
        source_height=100,
        rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        allow_intrinsically_degenerate=True,
    )
    assert geometry["degenerate_quad"] == {
        "classification": "repeated_points",
        "polygon_area_pixels2": 0.0,
        "convex_hull_area_pixels2": 0.0,
        "unique_points": 2,
        "bounding_width_pixels": 20.0,
        "bounding_height_pixels": 0.0,
        "candidate_eligible": False,
    }


def test_layout_closure_can_report_one_degenerate_raw_line_without_using_it_as_field_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    records = fixture["records"]
    raw, normalized = _quad(100, 700, 300, 700)
    records[0]["lines"].append(
        {
            "index": len(records[0]["lines"]),
            "text": "收款方：不得使用",
            "confidence": 0.4,
            "passes_drop_score": False,
            "quad_rectified": raw,
            "quad_rectified_normalized": normalized,
        }
    )
    records[0]["raw_line_count"] = len(records[0]["lines"])
    records_path = Path(fixture["layout"]) / "records.jsonl"
    records_bytes = _write_jsonl(records_path, records)
    layout_summary = fixture["layout_summary"]
    layout_summary["artifacts"]["records_jsonl"]["sha256"] = _sha(records_bytes)
    layout_summary["artifacts"]["records_jsonl"]["size_bytes"] = len(records_bytes)
    _write_json(Path(fixture["layout"]) / "summary.json", layout_summary)

    selection, sources, source_ids, _ = MODULE._validate_selection(
        fixture["selection"], fixture["audit"]
    )
    missing_sets, _ = MODULE._validate_audit(
        fixture["audit"], selection, sources
    )
    with pytest.raises(MODULE.EvidenceError, match="quad is degenerate"):
        MODULE._validate_layout(
            fixture["layout"],
            fixture["selection"],
            selection,
            sources,
            source_ids,
            missing_sets,
        )
    evidence, bindings = MODULE._validate_layout(
        fixture["layout"],
        fixture["selection"],
        selection,
        sources,
        source_ids,
        missing_sets,
        allow_intrinsically_degenerate_lines=True,
    )
    excluded = bindings["excluded_intrinsically_degenerate_lines"]
    assert len(excluded) == 1
    assert excluded[0]["record_index"] == 0
    assert excluded[0]["line_index"] == 3
    assert excluded[0]["classification"] == "repeated_points"
    assert excluded[0]["candidate_eligible"] is False
    assert all(
        anchor.get("text") != "收款方：不得使用"
        for field in evidence[0]["evidence_by_field"].values()
        for anchor in field["anchors"]
    )


def test_layout_closure_quarantines_entire_record_for_order_cancellation_contract_violation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    records = fixture["records"]
    source = fixture["sources"][0]
    raw = [[10.0, 10.0], [30.0, 30.0], [10.0, 30.0], [30.0, 10.0]]
    cancelled = {
        "index": len(records[0]["lines"]),
        "text": "收款方：不得使用",
        "confidence": 0.95,
        "passes_drop_score": True,
        "quad_rectified": raw,
        "quad_rectified_normalized": [[x / 999.0, y / 1599.0] for x, y in raw],
    }
    records[0] = _record(0, source, [*records[0]["lines"], cancelled])
    records_path = Path(fixture["layout"]) / "records.jsonl"
    records_bytes = _write_jsonl(records_path, records)
    layout_summary = fixture["layout_summary"]
    layout_summary["artifacts"]["records_jsonl"]["sha256"] = _sha(records_bytes)
    layout_summary["artifacts"]["records_jsonl"]["size_bytes"] = len(records_bytes)
    _write_json(Path(fixture["layout"]) / "summary.json", layout_summary)

    selection, sources, source_ids, _ = MODULE._validate_selection(
        fixture["selection"], fixture["audit"]
    )
    missing_sets, _ = MODULE._validate_audit(
        fixture["audit"], selection, sources
    )
    evidence, bindings = MODULE._validate_layout(
        fixture["layout"],
        fixture["selection"],
        selection,
        sources,
        source_ids,
        missing_sets,
        allow_intrinsically_degenerate_lines=True,
        allow_order_cancellation_contract_violation_lines=True,
    )
    excluded = bindings["excluded_quad_contract_lines"]
    assert len(excluded) == 1
    assert excluded[0]["record_index"] == 0
    assert excluded[0]["line_index"] == 3
    assert excluded[0]["classification"] == "order_cancels_nondegenerate_hull"
    assert excluded[0]["record_candidate_eligible"] is False
    # The valid clock/payment/status lines in the same record are also kept
    # out of diagnostic field evidence; no ambiguity can be removed by
    # deleting only the malformed line.
    assert all(
        field["anchors"] == []
        for field in evidence[0]["evidence_by_field"].values()
    )


def test_order_cancellation_requires_explicit_contract_violation_mode_and_preserves_raw_evidence() -> None:
    bow_tie = [[10.0, 10.0], [30.0, 30.0], [10.0, 30.0], [30.0, 10.0]]
    line = {
        "quad_rectified": bow_tie,
        "quad_rectified_normalized": [[x / 99.0, y / 99.0] for x, y in bow_tie],
    }
    with pytest.raises(MODULE.EvidenceError, match="cancels a non-degenerate hull") as raised:
        MODULE._quad_geometry(
            line,
            record_index=0,
            line_index=0,
            rectified_width=100,
            rectified_height=100,
            source_width=100,
            source_height=100,
            rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            allow_intrinsically_degenerate=True,
        )
    assert '"quad_rectified":[[10.0,10.0],[30.0,30.0],[10.0,30.0],[30.0,10.0]]' in str(
        raised.value
    )
    assert '"polygon_area_pixels2":0.0' in str(raised.value)
    assert '"convex_hull_area_pixels2":400.0' in str(raised.value)

    geometry = MODULE._quad_geometry(
        line,
        record_index=146,
        line_index=20,
        rectified_width=100,
        rectified_height=100,
        source_width=100,
        source_height=100,
        rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        allow_intrinsically_degenerate=True,
        allow_order_cancellation_contract_violation=True,
    )
    assert geometry["degenerate_quad"] == {
        "classification": "order_cancels_nondegenerate_hull",
        "polygon_area_pixels2": 0.0,
        "convex_hull_area_pixels2": 400.0,
        "unique_points": 4,
        "bounding_width_pixels": 20.0,
        "bounding_height_pixels": 20.0,
        "candidate_eligible": False,
        "producer_contract_violation": True,
        "record_candidate_eligible": False,
        "canonicalized": False,
    }


def test_explicit_order_cancellation_mode_does_not_accept_arbitrary_bow_ties() -> None:
    # This is a producer-reachable boundary-clipped ordering: Paddle's
    # sum/difference ordering yields a crossing with non-zero ordered area,
    # while its width/height checks still exceed three pixels.  The one-record
    # quarantine is deliberately limited to exact cancellation; it is not a
    # generic point-set or bow-tie canonicalizer.
    bow_tie = [[3.0, 0.0], [0.0, 3.0], [7.0, 0.0], [0.0, 7.0]]
    line = {
        "quad_rectified": bow_tie,
        "quad_rectified_normalized": [[x / 99.0, y / 99.0] for x, y in bow_tie],
    }
    with pytest.raises(MODULE.EvidenceError, match="self-intersecting with nonzero ordered area"):
        MODULE._quad_geometry(
            line,
            record_index=0,
            line_index=0,
            rectified_width=100,
            rectified_height=100,
            source_width=100,
            source_height=100,
            rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            allow_intrinsically_degenerate=True,
            allow_order_cancellation_contract_violation=True,
        )

    # A crossing arbitrarily close to an endpoint is still proper.  This
    # locks strict sign semantics rather than an epsilon that could turn a
    # small-but-real crossing into accepted geometry.
    near_endpoint = [
        [0.0, 1.0],
        [1000.0, 1.0],
        [500.0, 100.0],
        [0.000001, 0.9999999999],
    ]
    with pytest.raises(MODULE.EvidenceError, match="self-intersecting with nonzero ordered area"):
        MODULE._quad_geometry(
            {
                "quad_rectified": near_endpoint,
                "quad_rectified_normalized": [
                    [x / 1000.0, y / 100.0] for x, y in near_endpoint
                ],
            },
            record_index=0,
            line_index=0,
            rectified_width=1001,
            rectified_height=101,
            source_width=1001,
            source_height=101,
            rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            allow_intrinsically_degenerate=True,
            allow_order_cancellation_contract_violation=True,
        )


def test_quad_exclusion_mode_still_rejects_normalization_damage() -> None:
    bow_tie = [[10.0, 10.0], [30.0, 30.0], [10.0, 30.0], [30.0, 10.0]]
    line = {
        "quad_rectified": bow_tie,
        "quad_rectified_normalized": [[x / 99.0, y / 99.0] for x, y in bow_tie],
    }

    damaged = dict(line)
    damaged["quad_rectified"] = [[10.0, 20.0]] * 4
    damaged["quad_rectified_normalized"] = [[0.9, 0.9]] * 4
    with pytest.raises(MODULE.EvidenceError, match="normalized quad disagrees"):
        MODULE._quad_geometry(
            damaged,
            record_index=0,
            line_index=0,
            rectified_width=100,
            rectified_height=100,
            source_width=100,
            source_height=100,
            rectified_to_source=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            allow_intrinsically_degenerate=True,
        )


@pytest.mark.parametrize("failure", ["wrong_inverse", "wrong_forward", "singular"])
def test_homography_validation_rejects_observably_wrong_or_singular_pairs(failure: str) -> None:
    source_width, source_height = 1080, 2400
    rectified_width, rectified_height = 720, 1600
    scale_x = (rectified_width - 1) / (source_width - 1)
    scale_y = (rectified_height - 1) / (source_height - 1)
    forward = [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]]
    inverse = [[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]]
    if failure == "wrong_inverse":
        inverse[0][2] = 1.0
    elif failure == "wrong_forward":
        forward[0][0] *= 1.01
        inverse[0][0] = 1.0 / forward[0][0]
    else:
        forward[1] = [0.0, 0.0, 0.0]
    with pytest.raises(MODULE.EvidenceError, match="homography|matrix|projection|round-trip"):
        MODULE._require_homography_pair(
            forward,
            inverse,
            source_width=source_width,
            source_height=source_height,
            rectified_width=rectified_width,
            rectified_height=rectified_height,
            rotation_degrees=0,
            description="bad homography",
        )


def test_atomic_publish_refuses_overwrite_and_detects_toctou(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary, evidence_bytes, bindings = _prepare(fixture)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_atomic(output, summary=summary, evidence_bytes=evidence_bytes, bindings=bindings)
    assert marker.read_text(encoding="utf-8") == "mine"
    marker.unlink()
    output.rmdir()

    (Path(fixture["layout"]) / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="bound input changed"):
        MODULE.write_atomic(output, summary=summary, evidence_bytes=evidence_bytes, bindings=bindings)
    assert not output.exists()


def test_rejects_strict_json_failures(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary_path = Path(fixture["layout"]) / "summary.json"
    summary_path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="duplicate JSON key"):
        _prepare(fixture)

    fixture = _fixture(tmp_path / "nan")
    summary_path = Path(fixture["layout"]) / "summary.json"
    summary_path.write_text('{"schema_version":NaN}\n', encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="non-standard JSON constant"):
        _prepare(fixture)


def test_main_prints_coverage_and_refuses_reuse(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"
    args = [
        "--selection-directory", str(fixture["selection"]),
        "--audit-directory", str(fixture["audit"]),
        "--layout-directory", str(fixture["layout"]),
        "--output-directory", str(output),
    ]
    assert MODULE.main(args) == 0
    printed = capsys.readouterr().out
    assert "records=339" in printed
    assert "time_target_unique=1" in printed
    assert MODULE.main(args) == 2
    assert "refusing to overwrite" in capsys.readouterr().out
