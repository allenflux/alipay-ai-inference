from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-recipient-derived-crop-shadow.py"
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_recipient_derived_crop_shadow", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _attempt(lines: list[tuple[str, float]] | None = None) -> dict[str, object]:
    values = lines or []
    return {
        "name": "fixture",
        "present": True,
        "line_count": len(values),
        "alternative_route": "none",
        "geometry": "not_evaluated",
        "lines": [
            {
                "index": index,
                "text": text,
                "normalized_text": text,
                "confidence": confidence,
            }
            for index, (text, confidence) in enumerate(values)
        ],
    }


def _build_frozen_inputs(root: Path) -> tuple[Path, Path]:
    diagnostic = root / "diagnostic"
    truth = root / "truth"
    sources: list[Path] = []
    for index in range(204):
        source = root / "images" / f"receipt-{index:03d}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"fixture-{index}".encode())
        sources.append(source)

    _write_json(
        diagnostic / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.DIAGNOSTIC_SUMMARY_KIND,
            "comparison_evaluation_mode": "formal",
            "failed_records": 204,
            "recipient_missing_only_records": 204,
        },
    )
    diagnostic_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        geometry = {
            "rectified_width": 1000,
            "rectified_height": 1600,
            "amount_box": [200.0, 300.0, 800.0, 500.0],
            "recipient_box": [100.0, 600.0, 900.0, 700.0],
            "payment_box": [200.0, 800.0, 800.0, 900.0],
        }
        diagnostic_rows.append(
            {
                "schema_version": 1,
                "kind": MODULE.DIAGNOSTIC_FINDING_KIND,
                "source": str(source),
                "geometry_evidence": geometry,
            }
        )
        if index < 75:
            state = "candidate"
            remaining_cluster: dict[str, object] | None = None
            gate_failures: list[str] = []
        elif index < 90:
            state = "ambiguous"
            remaining_cluster = {"strict_state": state}
            gate_failures = []
        elif index < 120:
            state = "rejected_by_global_gate"
            remaining_cluster = {"strict_state": state}
            gate_failures = ["alternative_envelope_not_verified"]
        else:
            state = "unresolved"
            remaining_cluster = {"strict_state": state}
            gate_failures = (
                [] if index < 168 else ["ordinary_25pct_geometry_not_verified"]
            )
        first: list[tuple[str, float]] = []
        retry: list[tuple[str, float]] = []
        if index == 75:
            first = [("商户甲", 0.91)]
            retry = [("另一个值", 0.92)]
        truth_rows.append(
            {
                "schema_version": 1,
                "kind": MODULE.TRUTH_FINDING_KIND,
                "source": str(source),
                "recipient_detector_score": 0.95,
                "geometry_reasons": [] if not gate_failures else ["fixture_gate"],
                "alternative_envelope": not gate_failures,
                "attempts": {
                    "first": _attempt(first),
                    "retry": _attempt(retry),
                    "right_value": _attempt(),
                },
                "shadow_candidate_truth_free": {
                    "state": state,
                    "global_gate_failures": gate_failures,
                },
                "remaining_failure_cluster": remaining_cluster,
            }
        )
    _write_jsonl(diagnostic / "findings.jsonl", diagnostic_rows)
    _write_json(
        truth / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.TRUTH_SUMMARY_KIND,
            "paddle_teacher_consensus": {
                "records": 75,
                "by_state": [
                    {"name": "ambiguous", "records": 15},
                    {"name": "candidate", "records": 75},
                    {"name": "rejected_by_global_gate", "records": 30},
                    {"name": "unresolved", "records": 84},
                ],
            },
            "remaining_failure_analysis": {
                "records": 129,
                "strict_candidate_records": 75,
            },
            "remaining_global_gate_overlay_analysis": {
                "records": 129,
                "any_global_gate_failure_records": 66,
                "clear_global_gate_records": 63,
            },
        },
    )
    _write_jsonl(truth / "findings.jsonl", truth_rows)
    return diagnostic, truth


def _line(text: str, confidence: float, box: list[int]) -> dict[str, object]:
    crop_quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
    return {
        "index": 0,
        "text": text,
        "confidence": confidence,
        "passes_drop_score": confidence >= 0.5,
        "quad_crop": crop_quad,
        "quad_rectified": [
            [point[0] + box[0], point[1] + box[1]] for point in crop_quad
        ],
    }


def _build_layout(plan: Path, output: Path, *, provider: str = "cpu") -> None:
    plan_summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    plans = [
        json.loads(line)
        for line in (plan / "plans.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows: list[dict[str, object]] = []
    for index, record in enumerate(plans):
        crops: list[dict[str, object]] = []
        for crop in record["crops"]:
            lines: list[dict[str, object]] = []
            if index == 0 and crop["name"] == MODULE.CROP4:
                lines = [_line("商户甲", 0.90, crop["rectified_box"])]
            elif index == 1:
                lines = [_line("双新裁剪商户", 0.88, crop["rectified_box"])]
            elif index == 2:
                lines = [_line("Payment Method", 0.99, crop["rectified_box"])]
            crops.append(
                {
                    "name": crop["name"],
                    "rectified_box": crop["rectified_box"],
                    "width": crop["width"],
                    "height": crop["height"],
                    "lines": lines,
                }
            )
        rows.append(
            {
                "schema_version": 1,
                "kind": MODULE.LAYOUT_RECORD_KIND,
                "diagnostic_only": True,
                "formal_delivery_gate": False,
                "candidate_write_enabled": False,
                "execution_provider": "cpu",
                "source": record["source"],
                "source_image_sha256": record["source_image"]["sha256"],
                "rectified_size": record["rectified_size"],
                "plan_id": record["plan_id"],
                "crops": crops,
            }
        )
    _write_jsonl(output / "records.jsonl", rows)
    records_bytes = (output / "records.jsonl").read_bytes()
    paddle_bundle: dict[str, dict[str, object]] = {}
    for role in ("detector", "classifier", "recognizer", "dictionary"):
        path = output / "bundle" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"verified-{role}".encode())
        paddle_bundle[role] = MODULE._identity(path, description=role)
    _write_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.LAYOUT_SUMMARY_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "execution_provider": provider,
            "records": 63,
            "errors": 0,
            "rectification": "max_side_1600",
            "paddle_drop_score": 0.5,
            "input_plan": {
                "sha256": plan_summary["artifacts"]["plans"]["sha256"],
                "size_bytes": plan_summary["artifacts"]["plans"]["size_bytes"],
                "records": 63,
            },
            "paddle_bundle": paddle_bundle,
            "artifacts": {
                "records": {
                    "path": "records.jsonl",
                    "sha256": MODULE._sha256_bytes(records_bytes),
                    "size_bytes": len(records_bytes),
                    "records": 63,
                }
            },
        },
    )


def test_prepare_and_evaluate_are_atomic_truth_free_diagnostics(tmp_path: Path) -> None:
    diagnostic, truth = _build_frozen_inputs(tmp_path)
    plan = tmp_path / "plan"
    MODULE.prepare(diagnostic, truth, plan)

    summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    plans = [
        json.loads(line)
        for line in (plan / "plans.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["records"] == 63
    assert summary["frozen_v4"] == {
        "formal_failures": 204,
        "candidate_records": 75,
        "remaining_records": 129,
        "remaining_with_global_gate_failures": 66,
        "remaining_with_clear_global_gates": 63,
    }
    assert summary["required_layout_producer"] == {
        "api": "PaddleOcrEngine.RecognizeLayoutDiagnostic",
        "execution_provider": "cpu",
        "rectification": "max_side_1600",
        "requires_raw_quad_crop_and_rectified_coordinates": True,
        "requires_verified_paddle_bundle_identity": True,
        "required_summary_kind": MODULE.LAYOUT_SUMMARY_KIND,
        "required_record_kind": MODULE.LAYOUT_RECORD_KIND,
    }
    assert all(record["global_gate_evidence"]["global_gate_failures"] == [] for record in plans)
    assert all(record["candidate_write_enabled"] is False for record in plans)
    assert all([crop["name"] for crop in record["crops"]] == [MODULE.CROP4, MODULE.CROP5] for record in plans)
    assert all(record["crops"][0]["rectified_box"] != record["crops"][1]["rectified_box"] for record in plans)

    layout = tmp_path / "layout"
    _build_layout(plan, layout)
    evaluation = tmp_path / "evaluation"
    MODULE.evaluate(plan, layout, evaluation)
    evaluated_summary = json.loads(
        (evaluation / "summary.json").read_text(encoding="utf-8")
    )
    findings = [
        json.loads(line)
        for line in (evaluation / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert evaluated_summary["shadow_candidate_records"] == 2
    assert evaluated_summary["unresolved_records"] == 61
    assert evaluated_summary["accuracy_claimed"] is False
    assert evaluated_summary["truth_used_for_candidate_selection"] is False
    assert findings[0]["shadow_candidate"] == "商户甲"
    assert findings[0]["shadow_route"] == "derived_crop4_existing_exact_shadow"
    assert findings[1]["shadow_candidate"] == "双新裁剪商户"
    assert findings[1]["shadow_route"] == "derived_crop4_crop5_exact_shadow"
    assert findings[2]["shadow_candidate"] is None
    assert findings[2]["strict_crop4_lines"] == {}
    assert findings[2]["strict_crop5_lines"] == {}
    assert all(finding["candidate_write_enabled"] is False for finding in findings)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.prepare(diagnostic, truth, plan)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.evaluate(plan, layout, evaluation)


def test_prepare_refuses_non_v4_gate_overlay(tmp_path: Path) -> None:
    diagnostic, truth = _build_frozen_inputs(tmp_path)
    summary_path = truth / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["remaining_global_gate_overlay_analysis"]["clear_global_gate_records"] = 64
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.ShadowError, match="does not match frozen v4"):
        MODULE.prepare(diagnostic, truth, tmp_path / "plan")


def test_evaluate_refuses_non_cpu_layout_evidence(tmp_path: Path) -> None:
    diagnostic, truth = _build_frozen_inputs(tmp_path)
    plan = tmp_path / "plan"
    MODULE.prepare(diagnostic, truth, plan)
    layout = tmp_path / "layout"
    _build_layout(plan, layout, provider="cuda:0")
    with pytest.raises(MODULE.ShadowError, match="CPU diagnostic-only"):
        MODULE.evaluate(plan, layout, tmp_path / "evaluation")


def test_evaluate_refuses_source_image_mutation_after_planning(tmp_path: Path) -> None:
    diagnostic, truth = _build_frozen_inputs(tmp_path)
    plan = tmp_path / "plan"
    MODULE.prepare(diagnostic, truth, plan)
    layout = tmp_path / "layout"
    _build_layout(plan, layout)
    first_plan = json.loads(
        (plan / "plans.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    Path(first_plan["source"]).write_bytes(b"mutated-after-plan")
    with pytest.raises(MODULE.ShadowError, match="source image changed"):
        MODULE.evaluate(plan, layout, tmp_path / "evaluation")


def test_source_contract_never_connects_shadow_to_production_fields() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "V4_GLOBAL_GATE_CLEAR_REMAINING = 63" in source
    assert "candidate_write_enabled" in source
    assert '"production_output_changed": False' in source
    assert '"truth_used_for_candidate_selection": False' in source
    assert "derived_crop4_existing_exact_shadow" in source
    assert "derived_crop4_crop5_exact_shadow" in source
    assert "PaddleOcrEngine.RecognizeLayoutDiagnostic" in source
    assert "confidence < 0.80" in source
    assert "_shadow_line_allowed" in source
    assert "global_gate_failures" in source
    assert "formal_delivery_gate" in source
    assert "PaddleRecipientHybrid" not in source
    assert "UnifiedOcrCandidate" not in source
