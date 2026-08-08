from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load(
    "receipt_mlnet_hybrid_global_gate_audit_tested",
    ROOT / "scripts" / "receipt-mlnet-hybrid-global-gate-audit.py",
)
PROBE = _load(
    "receipt_mlnet_hybrid_failure_truth_probe_for_gate_audit_test",
    ROOT / "scripts" / "receipt-mlnet-hybrid-failure-truth-probe.py",
)
TARGET_FIXTURE = _load(
    "receipt_mlnet_hybrid_targeted_replay_fixture_for_gate_audit",
    ROOT / "tests" / "test_receipt_mlnet_hybrid_targeted_replay.py",
)
PROBE_FIXTURE = _load(
    "receipt_mlnet_hybrid_failure_truth_probe_fixture_for_gate_audit",
    ROOT / "tests" / "test_receipt_mlnet_hybrid_failure_truth_probe.py",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _patch_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (AUDIT,):
        monkeypatch.setattr(module, "FORMAL_RECORDS", 6)
        monkeypatch.setattr(module, "FAILURE_RECORDS", 2)
        monkeypatch.setattr(module, "EXPECTED_CANDIDATES", 0)
        monkeypatch.setattr(module, "EXPECTED_REMAINING", 2)
        monkeypatch.setattr(module, "EXPECTED_GATE_FAILURES", 2)
        monkeypatch.setattr(module, "EXPECTED_EXACT", 0)
        monkeypatch.setattr(module, "EXPECTED_DOMINANT", 0)
        monkeypatch.setattr(module, "EXPECTED_AMBIGUOUS", 0)
        monkeypatch.setattr(module, "EXPECTED_REJECTED_BY_GATE", 2)
        monkeypatch.setattr(module, "EXPECTED_UNRESOLVED", 0)
        monkeypatch.setattr(
            module,
            "EXPECTED_GATE_STATE_COUNTS",
            module.Counter({"rejected_by_global_gate": 2}),
        )
    monkeypatch.setattr(AUDIT.REPLAY, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(AUDIT.REPLAY, "MISSING_RECORDS", 2)
    monkeypatch.setattr(PROBE, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(PROBE, "FORMAL_FAILURES", 2)


def _result_by_source(formal: Path) -> dict[str, Path]:
    rows = json.loads(
        (formal / "hybrid-recipient" / "inference_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {AUDIT._source_key(row["source"]): Path(row["result"]) for row in rows}


def _verified_result_geometry(*, rotated: bool, score: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if rotated:
        scale_x = 799.0 / 1999.0
        scale_y = 1599.0 / 3999.0
        geometry = {
            "source_size": {"width": 4000, "height": 2000},
            "rectified_size": {"width": 800, "height": 1600},
            "rectification": "max-side-1600",
            "rotation_degrees": 90,
            "screen_detected": False,
            "H_original_to_rectified": [
                [0.0, -scale_x, 799.0],
                [scale_y, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "H_rectified_to_original": [
                [0.0, 1.0 / scale_y, 0.0],
                [-1.0 / scale_x, 0.0, 1999.0],
                [0.0, 0.0, 1.0],
            ],
        }
        detections = [
            {"label": "amount", "score": 0.95, "bbox_image": [400, 0, 600, 1999]},
            {
                "label": "recipient_field",
                "score": score,
                "bbox_image": [800, 0, 1000, 1999],
            },
            {
                "label": "payment_method_field",
                "score": 0.96,
                "bbox_image": [1200, 0, 1400, 1999],
            },
        ]
        return geometry, detections
    geometry = {
        "source_size": {"width": 1000, "height": 1600},
        "rectified_size": {"width": 1000, "height": 1600},
        "rectification": "max-side-1600",
        "rotation_degrees": 0,
        "screen_detected": False,
        "H_original_to_rectified": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "H_rectified_to_original": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    detections = [
        {"label": "amount", "score": 0.95, "bbox_image": [0, 200, 1000, 300]},
        {
            "label": "recipient_field",
            "score": score,
            "bbox_image": [0, 400, 1000, 500],
        },
        {
            "label": "payment_method_field",
            "score": 0.96,
            "bbox_image": [0, 600, 1000, 700],
        },
    ]
    return geometry, detections


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    frozen = TARGET_FIXTURE._formal_fixture(tmp_path, monkeypatch)
    _patch_counts(monkeypatch)
    formal = frozen["formal"]
    diagnostic = frozen["diagnostic"]
    result_paths = _result_by_source(formal)
    rows: list[dict[str, Any]] = []
    for index, source_path in enumerate(frozen["sources"][:2]):
        result_path = result_paths[AUDIT._source_key(source_path.resolve().as_posix())]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        geometry, detections = _verified_result_geometry(
            rotated=index == 0,
            score=0.90 if index == 0 else 0.67,
        )
        result["geometry"] = geometry
        result["detections"] = detections

        detection_map = {row["label"]: row for row in detections}
        geometry_reasons = AUDIT.DIAGNOSE._geometry_reasons(result, detection_map)
        row = PROBE_FIXTURE._finding(
            index,
            reference=f"商户{index}",
            first=[(0.96, f"商户{index}")],
            retry=[(0.95, f"商户{index}")],
            recipient_score=0.90 if index == 0 else 0.67,
            geometry_reasons=geometry_reasons,
            envelope=index != 0,
        )
        row["source"] = source_path.resolve().as_posix()
        row["amount_candidate"] = result["fields"]["amount"]["candidate"]
        row["amount_score"] = 0.95
        row["payment_score"] = 0.96
        row["ppocr_route"] = "none"
        row["geometry_evidence"] = AUDIT.DIAGNOSE._geometry_evidence(
            result, detection_map
        )
        recipient = result["fields"]["recipient"]
        recipient.update(
            {
                "hybrid_ocr_route": row["ppocr_route"],
                "hybrid_ocr_failure_reason": row["ppocr_failure_reason"],
                "hybrid_ocr_first_raw": row["first_raw"],
                "hybrid_ocr_first_line_count": row["first_line_count"],
                "hybrid_ocr_retry_raw": row["retry_raw"],
                "hybrid_ocr_retry_line_count": row["retry_line_count"],
                "hybrid_ocr_third_route": row.get("third_route"),
                "hybrid_ocr_right_value_raw": row["right_value_raw"],
                "hybrid_ocr_right_value_line_count": row[
                    "right_value_line_count"
                ],
                "hybrid_ocr_right_value_line_confidences": row[
                    "right_value_line_confidences"
                ],
            }
        )
        _write_json(result_path, result)
        rows.append(row)
    _write_jsonl(diagnostic / "findings.jsonl", rows)
    diagnostic_summary = json.loads(
        (diagnostic / "summary.json").read_text(encoding="utf-8")
    )
    diagnostic_summary["by_comparator_failure"] = [
        {"name": AUDIT.REPLAY.RECIPIENT_MISSING_FAILURE, "records": 2}
    ]
    _write_json(diagnostic / "summary.json", diagnostic_summary)

    probe = tmp_path / "consensus-probe-v4"
    analyzed, evidence = PROBE._load_input(diagnostic)
    probe_summary = PROBE.summarize(analyzed, evidence=evidence)
    PROBE.write_atomic(
        probe,
        input_directory=diagnostic,
        summary=probe_summary,
        findings=analyzed,
    )
    return {**frozen, "probe": probe}


def test_atomic_audit_classifies_rotation_score_and_safe_upper_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    inputs_before = {
        path: path.read_bytes()
        for root in (fixture["formal"], fixture["diagnostic"], fixture["probe"])
        for path in root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "gate-audit"

    result = AUDIT.audit(
        formal_directory=fixture["formal"],
        diagnostic_directory=fixture["diagnostic"],
        probe_directory=fixture["probe"],
        output_directory=output,
    )

    assert result["formal_delivery_gate"] is False
    assert result["accepted"] is False
    assert result["counts"] == {
        "formal_records": 6,
        "formal_recipient_omissions": 2,
        "v4_candidate_records": 0,
        "v4_remaining_records": 2,
        "v4_remaining_global_gate_failure_records": 2,
        "v4_remaining_global_gate_clear_records": 0,
        "v4_exact_consensus_candidates_preserved": 0,
        "v4_dominant_consensus_candidates_preserved": 0,
    }
    upper = result["investigation_upper_bound"]
    assert upper["alternative_envelope_evidence_records"] == 1
    assert upper["selected_consensus_candidate_records"] == 1
    assert upper["retrospective_external_reference_exact_records"] == 1
    assert upper["formal_accuracy_claimed"] is False
    assert upper["safe_repair_claimed"] is False
    assert result["safe_repair_upper_bound"]["records"] == 0
    assert result["safe_repair_upper_bound"]["proved_by_frozen_evidence"] is False
    assert {
        "probe_contract",
        "pilot_diagnose_contract",
        "targeted_replay_contract",
        "audit_implementation",
    }.issubset(result["source_evidence"])
    assert all(
        len(group["examples"]) <= 3
        for groups in result["groups"].values()
        for group in groups
    )
    findings = [
        json.loads(line)
        for line in (output / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(findings) == 2
    assert findings[0]["classification"]["rotation_projection"] == (
        "rotation_90|projection_verified"
    )
    assert findings[0]["classification"]["alternative_envelope"] == (
        "stored_false|recomputed_default_verified"
    )
    assert findings[0]["upper_bound_evidence"][
        "investigation_candidate_without_floor_change"
    ] is True
    assert findings[0]["upper_bound_evidence"][
        "safe_repair_proved_by_frozen_evidence"
    ] is False
    assert findings[1]["classification"]["detector_score"] == "below_0.68"
    assert "true_detector_score_failure" in findings[1]["classification"][
        "failure_nature_combination"
    ]
    assert findings[1]["upper_bound_evidence"][
        "investigation_candidate_without_floor_change"
    ] is False
    assert inputs_before == {
        path: path.read_bytes() for path in inputs_before
    }


def test_probe_semantic_mutation_not_self_bound_by_summary_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    probe_findings = fixture["probe"] / "findings.jsonl"
    rows = [
        json.loads(line)
        for line in probe_findings.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["attempts"]["first"]["lines"][0]["normalized_text"] = "语义篡改"
    _write_jsonl(probe_findings, rows)
    output = tmp_path / "must-not-publish"

    with pytest.raises(AUDIT.AuditError, match="fresh truth-free analysis"):
        AUDIT.audit(
            formal_directory=fixture["formal"],
            diagnostic_directory=fixture["diagnostic"],
            probe_directory=fixture["probe"],
            output_directory=output,
        )

    assert not output.exists()


def test_formal_detector_score_must_equal_diagnostic_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result_paths = _result_by_source(fixture["formal"])
    first = result_paths[
        AUDIT._source_key(fixture["sources"][0].resolve().as_posix())
    ]
    result = json.loads(first.read_text(encoding="utf-8"))
    recipient = next(
        row for row in result["detections"] if row["label"] == "recipient_field"
    )
    recipient["score"] = 0.91
    _write_json(first, result)
    output = tmp_path / "must-not-publish-score"

    with pytest.raises(AUDIT.AuditError, match="recipient_score binding mismatch"):
        AUDIT.audit(
            formal_directory=fixture["formal"],
            diagnostic_directory=fixture["diagnostic"],
            probe_directory=fixture["probe"],
            output_directory=output,
        )

    assert not output.exists()


def test_refuses_overwrite_and_input_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(AUDIT.AuditError, match="refusing to overwrite"):
        AUDIT.audit(
            formal_directory=fixture["formal"],
            diagnostic_directory=fixture["diagnostic"],
            probe_directory=fixture["probe"],
            output_directory=existing,
        )
    with pytest.raises(AUDIT.AuditError, match="overlaps diagnostic root"):
        AUDIT.audit(
            formal_directory=fixture["formal"],
            diagnostic_directory=fixture["diagnostic"],
            probe_directory=fixture["probe"],
            output_directory=fixture["diagnostic"] / "nested-output",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rotation_degrees", 180, "invalid rotation_degrees"),
        ("rectification", "none", "not rectification=max-side-1600"),
        (
            "H_rectified_to_original",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "homography/inverse are inconsistent",
        ),
    ],
)
def test_rectification_mode_and_rotation_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result_paths = _result_by_source(fixture["formal"])
    first = result_paths[
        AUDIT._source_key(fixture["sources"][0].resolve().as_posix())
    ]
    result = json.loads(first.read_text(encoding="utf-8"))
    result["geometry"][field] = value
    _write_json(first, result)
    output = tmp_path / "must-not-publish-geometry"

    with pytest.raises(AUDIT.AuditError, match=message):
        AUDIT.audit(
            formal_directory=fixture["formal"],
            diagnostic_directory=fixture["diagnostic"],
            probe_directory=fixture["probe"],
            output_directory=output,
        )

    assert not output.exists()


def test_float32_boundary_is_a_label_not_a_floor_change() -> None:
    tolerance = AUDIT._boundary_tolerance(0.68)
    assert tolerance > 0.0
    assert AUDIT._ordinary_geometry_state(
        ["payment_edge_overlap"], ["payment_edge_slack_25pct"]
    ).startswith("float32_threshold_boundary+")
    assert AUDIT.MINIMUM_RECIPIENT_SCORE == 0.68
