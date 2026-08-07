from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "receipt-mlnet-formal-diagnose.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_formal_diagnose", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_diagnose_reports_gate_failure_and_missing_candidate(tmp_path: Path) -> None:
    tag = "20260806-165128"
    output = tmp_path / "delivery-validation" / f"mlnet-wide1536-cpu-full-{tag}"
    evaluation = tmp_path / "delivery-validation" / f"mlnet-wide1536-cpu-full-e2e-{tag}"
    source = r"D:\images\blank-amount.jpg"
    result_path = output / "input-list" / "blank-amount.json"
    _write_json(
        output / "inference_summary.json",
        {
            "requested_device": "cpu",
            "unified_provider": "cpu",
            "input": 1,
            "written": 1,
            "skipped": 0,
            "errors": 0,
            "total_seconds": 1.5,
            "inference_latency_ms": {"mean": 1000.0, "p50": 900.0, "p95": 1200.0},
        },
    )
    _write_json(
        result_path,
        {
            "fields": {
                "amount": {
                    "state": "unreadable",
                    "candidate": None,
                    "ctc_candidate": None,
                    "structured_candidate": None,
                }
            },
            "detections": [{"label": "amount", "score": 0.91, "bbox_image": [1, 2, 3, 4]}],
        },
    )
    fields = {
        "amount": {
            "records": 1,
            "raw_exact_matches": 0,
            "raw_exact_match": 0.0,
            "candidate_records": 0,
            "candidate_coverage": 0.0,
        },
        "time": {
            "records": 1,
            "raw_exact_matches": 1,
            "raw_exact_match": 1.0,
            "candidate_records": 1,
            "candidate_coverage": 1.0,
        },
        "payment_method_field": {
            "records": 1,
            "raw_exact_matches": 1,
            "raw_exact_match": 1.0,
            "candidate_records": 1,
            "candidate_coverage": 1.0,
        },
        "recipient_field": {
            "records": 1,
            "raw_exact_matches": 1,
            "raw_exact_match": 1.0,
            "candidate_records": 1,
            "candidate_coverage": 1.0,
        },
    }
    missing = {
        field: {"records": int(field == "amount"), "sources": [source] if field == "amount" else []}
        for field in fields
    }
    _write_json(
        evaluation / "summary.json",
        {
            "kind": "receipt_mlnet_unified_candidate_evaluation_v1",
            "accepted": False,
            "by_field": fields,
            "floors": {
                "amount": 0.7885,
                "time": 0.984,
                "payment_method_field": 0.9325,
                "recipient_field": 0.9,
            },
            "missing": {"field_candidates": missing},
            "failures": ["amount: candidate_coverage=0.0000 < 1.0000"],
        },
    )
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "comparisons.jsonl").write_text(
        json.dumps(
            {
                "field": "amount",
                "source": source.replace("\\", "/"),
                "reference_text": "9.95",
                "missing_reason": "candidate_missing",
                "result_json": str(result_path),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = MODULE.diagnose(data_root=tmp_path, tag=tag)

    assert "delivery_exists=False" in lines
    assert any("runtime requested_device=cpu unified_provider=cpu" in line for line in lines)
    assert any("field=amount matches=0/1" in line and "coverage_pass=False" in line for line in lines)
    assert "failure=amount: candidate_coverage=0.0000 < 1.0000" in lines
    assert any(
        "missing_detail field=amount" in line
        and "field_state=unreadable" in line
        and "detector_score=0.91" in line
        for line in lines
    )
