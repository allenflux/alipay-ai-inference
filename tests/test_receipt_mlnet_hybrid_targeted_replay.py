from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-hybrid-targeted-replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_hybrid_targeted_replay", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODEL_BYTES = b"fixture-v13-unified-model"
MODEL_SHA256 = hashlib.sha256(MODEL_BYTES).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(path: Path, *, records: int | None = None, sources: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    if sources is not None:
        result["normalized_source_set_sha256"] = MODULE._normalized_source_set_sha256(
            tuple(MODULE._source_key(source) for source in sources)
        )
    return result


def _result(source: Path, *, recipient: str | None, route: str, status: str = "转账成功") -> dict[str, Any]:
    recipient_field: dict[str, Any] = {
        "state": "unreadable" if recipient is None else "review",
        "candidate": recipient,
        "ctc_candidate": recipient,
        "detector_score": 0.99,
        "delivery_policy": "review_only_pending_independent_human_truth_calibration",
        "delivery_value": "review",
        "value": "review",
        "hybrid_ocr_route": route,
    }
    return {
        "result_schema_version": 1,
        "result_semantics_version": "fixture-v13",
        "source": source.resolve().as_posix(),
        "inference_engine": "mlnet",
        "geometry": {
            "rectification": "max-side-1600",
            "source_size": {"width": 100, "height": 200},
            "rectified_size": {"width": 100, "height": 200},
            "screen_detected": False,
            "rotation_degrees": 0,
        },
        "device": {"platform": "android", "confidence": 0.99},
        "model_contracts": {
            "detector": "detector.contract.json",
            "detector_sha256": "a" * 64,
            "detector_contract_sha256": "b" * 64,
            "device": "device.contract.json",
            "device_sha256": "c" * 64,
            "device_contract_sha256": "d" * 64,
            "unified_ocr_model": "v13.onnx",
            "unified_ocr_contract": "v13.contract.json",
            "unified_ocr_model_sha256": MODEL_SHA256,
            "unified_ocr_labels_sha256": "f" * 64,
            "unified_ocr_contract_sha256": "0" * 64,
        },
        "fields": {
            "amount": {"candidate": "10.00", "delivery_policy": "review"},
            "time": {"candidate": "12:30", "delivery_policy": "review"},
            "payment_method": {"candidate": "余额", "delivery_policy": "review"},
            "recipient": recipient_field,
            "transfer_status": {
                "candidate": status,
                "normalized": MODULE.normalize_status(status),
                "delivery_policy": "review",
            },
        },
        "detections": [
            {"label": "amount", "score": 0.99, "bbox_image": [0, 0, 10, 10]},
            {"label": "time", "score": 0.99, "bbox_image": [0, 10, 10, 20]},
            {"label": "recipient_field", "score": 0.99, "bbox_image": [0, 20, 10, 30]},
            {"label": "payment_method_field", "score": 0.99, "bbox_image": [0, 30, 10, 40]},
            {"label": "transfer_status", "score": 0.99, "bbox_image": [0, 40, 10, 50]},
        ],
        "limitations": ["review-only"],
    }


def _write_run(
    root: Path,
    sources: list[Path],
    *,
    hybrid: bool,
    candidates: list[str | None],
    routes: list[str],
    statuses: list[str] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    inference_values: list[float] = []
    stage_values: dict[str, list[float]] = {
        stage: [] for stage in MODULE.ALL_STAGES
    }
    statuses = statuses or ["转账成功"] * len(sources)
    for index, (source, candidate, route, status) in enumerate(
        zip(sources, candidates, routes, statuses, strict=True)
    ):
        result_path = root / "results" / f"{index}.json"
        result = _result(source, recipient=candidate, route=route, status=status)
        if hybrid:
            result["model_contracts"].update(
                {
                    "ocr_bundle": "paddle_ocr_delivery.contract.json",
                    "ocr_bundle_contract_sha256": "1" * 64,
                }
            )
        _write_json(result_path, result)
        inference_ms = 100.0 + index + (20.0 if hybrid else 0.0)
        inference_values.append(inference_ms)
        stages = {
            "image_load": 1.0,
            "device": 2.0,
            "detector_preprocess": 3.0,
            "detector_inference": 4.0,
            "detector_postprocess": 5.0,
            "paddle_ocr": 20.0 if hybrid else None,
            "unified_ocr_preprocess": 6.0,
            "unified_ocr_inference": 7.0,
            "unified_ocr_postprocess": 8.0,
            "result_assembly": 9.0,
        }
        for stage in MODULE.ALL_STAGES:
            value = stages[stage]
            if value is not None:
                stage_values[stage].append(value)
        manifest.append(
            {
                "source": source.resolve().as_posix(),
                "result": result_path.resolve().as_posix(),
                "status": "written",
                "inference_ms": inference_ms,
                "stage_latency_ms": stages,
            }
        )
    _write_json(root / "inference_manifest.json", manifest)
    (root / "inference_errors.jsonl").write_text("", encoding="utf-8")
    count = len(sources)
    _write_json(
        root / "inference_summary.json",
        {
            "requested_device": "cpu",
            "paddle_ocr_provider": "cpu" if hybrid else None,
            "unified_provider": "cpu",
            "input": count,
            "written": count,
            "skipped": 0,
            "errors": 0,
            "total_seconds": float(count),
            "inference_latency_ms": MODULE._summarize(inference_values),
            "stage_latency_ms": {
                stage: MODULE._summarize(values)
                for stage, values in stage_values.items()
            },
        },
    )
    return {
        "manifest": _identity(
            root / "inference_manifest.json",
            records=count,
            sources=[source.resolve().as_posix() for source in sources],
        ),
        "summary": _identity(root / "inference_summary.json"),
    }


def _formal_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(MODULE, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(MODULE, "MISSING_RECORDS", 2)
    monkeypatch.setattr(MODULE, "CONTROL_RECORDS", 2)
    monkeypatch.setattr(MODULE, "TARGET_RECORDS", 4)
    formal = tmp_path / "formal"
    sources = [tmp_path / "images" / f"receipt-{index}.jpg" for index in range(6)]
    for index, source in enumerate(sources):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"image-{index}".encode())
    rendered_sources = [source.resolve().as_posix() for source in sources]
    input_list = formal / "fixed-selected-inputs.txt"
    input_list.parent.mkdir(parents=True)
    input_list.write_text(
        "".join(f"{source}\n" for source in rendered_sources), encoding="utf-8"
    )
    routes = ["none", "none", "primary", "left_context", "right_value", "primary"]
    statuses = ["转账成功", "转账成功", "转账成功", "转账成功", "处理中", "转账成功"]
    baseline_candidates = [f"基线{index}" for index in range(6)]
    old_hybrid_candidates = [None, None, "商户2", "商户3", "商户4", "商户5"]
    baseline_ids = _write_run(
        formal / "baseline-v13",
        sources,
        hybrid=False,
        candidates=baseline_candidates,
        routes=["none"] * 6,
        statuses=statuses,
    )
    hybrid_ids = _write_run(
        formal / "hybrid-recipient",
        sources,
        hybrid=True,
        candidates=old_hybrid_candidates,
        routes=routes,
        statuses=statuses,
    )
    comparison = formal / "comparison"
    comparison.mkdir()
    comparison_rows = []
    for source, candidate in zip(rendered_sources, old_hybrid_candidates, strict=True):
        missing = candidate is None
        comparison_rows.append(
            {
                "source": source,
                "recipient_candidate": candidate,
                "invariant": not missing,
                "failures": [MODULE.RECIPIENT_MISSING_FAILURE] if missing else [],
            }
        )
    _write_jsonl(comparison / "comparisons.jsonl", comparison_rows)
    input_identity = _identity(input_list, records=6, sources=rendered_sources)
    _write_json(
        comparison / "summary.json",
        {
            "schema_version": 2,
            "kind": MODULE.AB_KIND,
            "evaluation_mode": "formal",
            "records": 6,
            "input_set_identical": True,
            "cli_summary_counts_verified": True,
            "input_set": {
                "records": 6,
                "normalized_source_set_sha256": input_identity[
                    "normalized_source_set_sha256"
                ],
                "input_manifest": input_identity,
            },
            "run_manifests": {
                "baseline": baseline_ids["manifest"],
                "hybrid": hybrid_ids["manifest"],
            },
            "run_summaries": {
                "baseline": baseline_ids["summary"],
                "hybrid": hybrid_ids["summary"],
            },
            "invariant_records": 4,
            "recipient_candidate_coverage": 4 / 6,
            "accepted": False,
            "failures": ["missing"] * 2,
        },
    )
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    findings = [
        {
            "schema_version": 1,
            "kind": MODULE.DIAGNOSTIC_FINDING_KIND,
            "source": rendered_sources[index],
            "failures": [MODULE.RECIPIENT_MISSING_FAILURE],
        }
        for index in range(2)
    ]
    _write_jsonl(diagnostic / "findings.jsonl", findings)
    _write_json(
        diagnostic / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.DIAGNOSTIC_KIND,
            "read_only_existing_results": True,
            "ocr_rerun": False,
            "comparison_evaluation_mode": "formal",
            "comparison_records": 6,
            "invariant_failure_records": 2,
            "recipient_missing_records": 2,
            "non_missing_invariant_failure_records": 0,
            "recipient_missing_only_records": 2,
            "recipient_missing_with_additional_failures_records": 0,
            "failed_records": 2,
            "source_evidence": {
                "comparison_summary": _identity(comparison / "summary.json"),
                "comparisons": _identity(comparison / "comparisons.jsonl"),
                "hybrid_manifest": hybrid_ids["manifest"],
            },
        },
    )
    records = tmp_path / "unified-fields.jsonl"
    _write_jsonl(
        records,
        [
            {
                "id": f"id-{index}",
                "split": "val",
                "source": source,
                "slots": {
                    "amount": {"text": "10.00"},
                    "time": {"text": "12:30"},
                    "payment_method_field": {"text": "余额"},
                    "recipient_field": {"text": f"商户{index}"},
                    "transfer_status": {
                        "text": statuses[index],
                        "class_name": MODULE.normalize_status(statuses[index]),
                    },
                },
            }
            for index, source in enumerate(rendered_sources)
        ],
    )
    return {
        "formal": formal,
        "diagnostic": diagnostic,
        "records": records,
        "sources": sources,
        "old_hybrid_candidates": old_hybrid_candidates,
    }


def _score(
    directory: Path,
    *,
    prepared: Path,
    results_root: Path,
    manifest: Path,
) -> None:
    model = prepared.parent / "v13.onnx"
    model.write_bytes(MODEL_BYTES)
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    records = selection["records"]
    result_by_key: dict[str, dict[str, Any]] = {}
    manifest_by_key: dict[str, dict[str, Any]] = {}
    manifest_rows = json.loads(manifest.read_text(encoding="utf-8"))
    for row in manifest_rows:
        key = MODULE._source_key(row["source"])
        manifest_by_key[key] = row
        result_by_key[key] = json.loads(
            Path(row["result"]).read_text(encoding="utf-8")
        )
    references_by_key: dict[str, dict[str, str]] = {}
    status_classes_by_key: dict[str, str] = {}
    for row in MODULE._load_jsonl(
        prepared / "subset-records.jsonl", description="fixture subset records"
    ):
        key = MODULE._source_key(row["source"])
        references_by_key[key] = {
            "amount": row["slots"]["amount"]["text"],
            "time": row["slots"]["time"]["text"],
            "payment_method_field": row["slots"]["payment_method_field"]["text"],
            "recipient_field": row["slots"]["recipient_field"]["text"],
            "transfer_status": row["slots"]["transfer_status"]["text"],
        }
        status_classes_by_key[key] = row["slots"]["transfer_status"]["class_name"]
    field_map = MODULE.SCORE_RESULT_FIELDS
    rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, int]] = {}
    for selected in records:
        key = MODULE._source_key(selected["source"])
        result = result_by_key[key]
        for field, result_field in field_map.items():
            candidate = MODULE._candidate(result, result_field)
            result_field_payload = result["fields"][result_field]
            reference = references_by_key[key][field]
            score_row = {
                "schema_version": 1,
                "kind": "receipt_mlnet_unified_comparison_v1",
                "id": f"id-{selected['canonical_index']}",
                "group_id": None,
                "split": "val",
                "source": selected["source"],
                "teacher_result_json": None,
                "field": field,
                "reference_text": reference,
                "reference_crop_sha256": None,
                "reference_detector_score": None,
                "reference_bbox_rectified": None,
                "candidate_text": candidate,
                "candidate_present": candidate is not None,
                "raw_exact": candidate == reference if candidate is not None else False,
                "result_json": Path(manifest_by_key[key]["result"])
                .resolve()
                .as_posix(),
                "manifest_status": manifest_by_key[key]["status"],
                "unified_model_sha256": MODEL_SHA256,
                "ctc_candidate_text": result_field_payload.get("ctc_candidate"),
                "structured_candidate_text": result_field_payload.get(
                    "structured_candidate"
                ),
            }
            if field == "transfer_status":
                candidate_class = MODULE.normalize_status(candidate)
                reference_class = status_classes_by_key[key]
                score_row.update(
                    {
                        "reference_status_class": reference_class,
                        "candidate_status_class": candidate_class,
                        "non_success_to_success": (
                            reference_class in {"pending", "failed"}
                            and candidate_class == "success"
                        ),
                    }
                )
            rows.append(score_row)
            field_metrics = metrics.setdefault(
                field, {"records": 0, "candidate_records": 0, "raw_exact_matches": 0}
            )
            field_metrics["records"] += 1
            field_metrics["candidate_records"] += int(candidate is not None)
            field_metrics["raw_exact_matches"] += int(candidate == reference)
    for field, field_metrics in metrics.items():
        records_count = field_metrics["records"]
        field_metrics["raw_exact_match"] = (
            field_metrics["raw_exact_matches"] / records_count
        )
        field_metrics["candidate_coverage"] = (
            field_metrics["candidate_records"] / records_count
        )
        if field == "transfer_status":
            status_rows = [row for row in rows if row["field"] == "transfer_status"]
            field_metrics["non_success_truth_records"] = sum(
                row["reference_status_class"] in {"pending", "failed"}
                for row in status_rows
            )
            field_metrics["non_success_to_success"] = sum(
                row["non_success_to_success"] for row in status_rows
            )
    _write_jsonl(directory / "comparisons.jsonl", rows)
    _write_json(
        directory / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.SCORE_KIND,
            "records": (prepared / "subset-records.jsonl").resolve().as_posix(),
            "records_sha256": hashlib.sha256(
                (prepared / "subset-records.jsonl").read_bytes()
            ).hexdigest(),
            "results_root": results_root.resolve().as_posix(),
            "manifest": manifest.resolve().as_posix(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "model": model.resolve().as_posix(),
            "model_sha256": MODEL_SHA256,
            "input_selection": None,
            "evaluation_split": "val",
            "floors": MODULE.FIXED_FLOORS,
            "by_field": metrics,
            "accuracy_denominators": {
                "scope": "selected_reference_records",
                "hash_bound": False,
                "source": "records_manifest_selected_field_reference_counts",
                "by_field": {
                    field: field_metrics["records"]
                    for field, field_metrics in metrics.items()
                },
            },
            "coverage": {
                "expected_receipts": 4,
                "matched_result_receipts": 4,
                "result_coverage": 1.0,
                "extra_manifest_sources": [],
            },
            "missing": {"result_receipts": 0},
            "artifact_audit": {
                "manifest_records": 4,
                "usable_manifest_sources": 4,
                "all_results_match_model": True,
            },
            "evaluation_scope": {
                "kind": "full_split",
                "requested_limit": None,
                "evaluated_expected_receipts": 4,
                "full_split_expected_receipts": 4,
                "input_list_path": None,
                "input_list_sha256": None,
                "formal_delivery_gate": False,
            },
            "formal_delivery_gate": False,
            "accepted": False,
            "acceptance": {"formal_delivery_gate": False},
        },
    )


def _target_fixture(tmp_path: Path, prepared: Path) -> dict[str, Path]:
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    sources = [Path(row["source"]) for row in selection["records"]]
    baseline_candidates = []
    statuses = []
    new_candidates = []
    for row in selection["records"]:
        old_baseline = json.loads(Path(row["old_baseline_result"]["path"]).read_text())
        baseline_candidates.append(old_baseline["fields"]["recipient"]["candidate"])
        statuses.append(old_baseline["fields"]["transfer_status"]["candidate"])
        new_candidates.append(f"商户{row['canonical_index']}")
    baseline = tmp_path / "target-baseline"
    hybrid = tmp_path / "target-hybrid"
    baseline_ids = _write_run(
        baseline,
        sources,
        hybrid=False,
        candidates=baseline_candidates,
        routes=["none"] * 4,
        statuses=statuses,
    )
    hybrid_ids = _write_run(
        hybrid,
        sources,
        hybrid=True,
        candidates=new_candidates,
        routes=["independent_crop_exact_consensus"] * 4,
        statuses=statuses,
    )
    comparison = tmp_path / "target-comparison"
    comparison.mkdir()
    cli_app = tmp_path / "target-cli-app"
    cli_app.mkdir()
    (cli_app / "ReceiptMlNet.Cli.dll").write_bytes(b"new-target-cli")
    (cli_app / "onnxruntime.dll").write_bytes(b"cpu-runtime")
    closure_rows = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(cli_app.iterdir(), key=lambda path: path.name.casefold())
    ]
    closure_manifest = tmp_path / "target-cli-app-closure.json"
    _write_json(closure_manifest, closure_rows)
    _write_jsonl(
        comparison / "comparisons.jsonl",
        [
            {
                "source": source.resolve().as_posix(),
                "recipient_candidate": candidate,
                "invariant": True,
                "failures": [],
            }
            for source, candidate in zip(sources, new_candidates, strict=True)
        ],
    )
    inputs = prepared / "inputs.txt"
    input_identity = _identity(
        inputs,
        records=4,
        sources=[source.resolve().as_posix() for source in sources],
    )
    _write_json(
        comparison / "summary.json",
        {
            "schema_version": 2,
            "kind": MODULE.AB_KIND,
            "evaluation_mode": "pilot",
            "records": 4,
            "input_set_identical": True,
            "cli_summary_counts_verified": True,
            "input_set": {
                "records": 4,
                "normalized_source_set_sha256": input_identity[
                    "normalized_source_set_sha256"
                ],
                "input_manifest": input_identity,
            },
            "run_manifests": {
                "baseline": baseline_ids["manifest"],
                "hybrid": hybrid_ids["manifest"],
            },
            "run_summaries": {
                "baseline": baseline_ids["summary"],
                "hybrid": hybrid_ids["summary"],
            },
            "cli_build": {
                "assembly": _identity(cli_app / "ReceiptMlNet.Cli.dll"),
                "app_closure": {
                    "root": cli_app.resolve().as_posix(),
                    "manifest": _identity(closure_manifest),
                    "closure_sha256": hashlib.sha256(
                        closure_manifest.read_bytes()
                    ).hexdigest(),
                    "file_count": len(closure_rows),
                },
            },
            "invariant_records": 4,
            "recipient_candidate_coverage": 1.0,
            "cpu": {"p95_overhead_ms": 20.0, "max_p95_overhead_ms": 250.0},
            "accepted": True,
            "failures": [],
        },
    )
    old_score = tmp_path / "old-score"
    new_score = tmp_path / "new-score"
    _score(
        old_score,
        prepared=prepared,
        results_root=prepared / "old-hybrid-subset",
        manifest=prepared / "old-hybrid-subset" / "inference_manifest.json",
    )
    _score(
        new_score,
        prepared=prepared,
        results_root=hybrid,
        manifest=hybrid / "inference_manifest.json",
    )
    return {
        "baseline": baseline,
        "hybrid": hybrid,
        "comparison": comparison,
        "old_score": old_score,
        "new_score": new_score,
    }


def _drop_score_row(directory: Path, *, field: str) -> None:
    rows_path = directory / "comparisons.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    removed_index = next(index for index, row in enumerate(rows) if row["field"] == field)
    removed = rows.pop(removed_index)
    _write_jsonl(rows_path, rows)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    metric = summary["by_field"][field]
    metric["records"] -= 1
    metric["candidate_records"] -= int(removed["candidate_present"])
    metric["raw_exact_matches"] -= int(removed["raw_exact"])
    metric["candidate_coverage"] = metric["candidate_records"] / metric["records"]
    metric["raw_exact_match"] = metric["raw_exact_matches"] / metric["records"]
    summary["accuracy_denominators"]["by_field"][field] -= 1
    _write_json(summary_path, summary)


def _rewrite_new_recipient(
    target: dict[str, Path], *, source: str, candidate: str
) -> None:
    key = MODULE._source_key(source)
    manifest = json.loads((target["hybrid"] / "inference_manifest.json").read_text())
    manifest_row = next(row for row in manifest if MODULE._source_key(row["source"]) == key)
    result_path = Path(manifest_row["result"])
    result = json.loads(result_path.read_text())
    result["fields"]["recipient"]["candidate"] = candidate
    result["fields"]["recipient"]["ctc_candidate"] = candidate
    _write_json(result_path, result)

    comparison_path = target["comparison"] / "comparisons.jsonl"
    comparison_rows = [
        json.loads(line) for line in comparison_path.read_text().splitlines()
    ]
    comparison_row = next(
        row for row in comparison_rows if MODULE._source_key(row["source"]) == key
    )
    comparison_row["recipient_candidate"] = candidate
    _write_jsonl(comparison_path, comparison_rows)

    score_path = target["new_score"] / "comparisons.jsonl"
    score_rows = [json.loads(line) for line in score_path.read_text().splitlines()]
    score_row = next(
        row
        for row in score_rows
        if MODULE._source_key(row["source"]) == key
        and row["field"] == "recipient_field"
    )
    was_exact = score_row["raw_exact"]
    score_row["candidate_text"] = candidate
    score_row["ctc_candidate_text"] = candidate
    score_row["candidate_present"] = True
    score_row["raw_exact"] = candidate == score_row["reference_text"]
    _write_jsonl(score_path, score_rows)
    summary_path = target["new_score"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    metric = summary["by_field"]["recipient_field"]
    metric["raw_exact_matches"] += int(score_row["raw_exact"]) - int(was_exact)
    metric["raw_exact_match"] = metric["raw_exact_matches"] / metric["records"]
    _write_json(summary_path, summary)


def test_prepare_and_gate_strict_targeted_replay_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"

    report = MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=prepared,
    )

    assert report["formal_delivery_gate"] is False
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    assert [row["role"] for row in selection["records"]].count("recipient_missing") == 2
    assert [row["role"] for row in selection["records"]].count("control") == 2
    assert [row["canonical_index"] for row in selection["records"]] == sorted(
        row["canonical_index"] for row in selection["records"]
    )
    target = _target_fixture(tmp_path, prepared)
    gate_output = tmp_path / "gate"

    gated = MODULE.gate(
        prepared=prepared,
        baseline=target["baseline"],
        hybrid=target["hybrid"],
        comparison=target["comparison"],
        old_score=target["old_score"],
        new_score=target["new_score"],
        output=gate_output,
    )

    assert gated["accepted"] is True
    assert gated["formal_delivery_gate"] is False
    assert gated["counts"] == {
        "selected": 4,
        "recipient_missing_recovered": 2,
        "controls": 2,
        "baseline_prediction_differences": 0,
        "all_field_correct_to_wrong": 0,
        "control_recipient_correct_to_wrong": 0,
        "status_non_success_to_success": 0,
    }
    assert gated["cpu"]["p95_overhead_ms"] == 20.0


def test_prepare_rejects_non_target_formal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    comparisons = fixture["formal"] / "comparison" / "comparisons.jsonl"
    rows = [json.loads(line) for line in comparisons.read_text().splitlines()]
    rows[0]["failures"].append("fields.amount changed")
    _write_jsonl(comparisons, rows)

    with pytest.raises(MODULE.ReplayError, match="all/only"):
        MODULE.prepare(
            formal_root=fixture["formal"],
            diagnostic=fixture["diagnostic"],
            records=fixture["records"],
            output=tmp_path / "prepared",
        )


@pytest.mark.parametrize("mutation", ["baseline", "p95", "status"])
def test_gate_rejects_prediction_drift_latency_overhead_or_status_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=prepared,
    )
    target = _target_fixture(tmp_path, prepared)
    if mutation == "baseline":
        manifest = json.loads(
            (target["baseline"] / "inference_manifest.json").read_text()
        )
        result_path = Path(manifest[0]["result"])
        result = json.loads(result_path.read_text())
        result["device"]["platform"] = "ios"
        _write_json(result_path, result)
        message = "prediction differences"
    elif mutation == "p95":
        summary_path = target["comparison"] / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["cpu"]["p95_overhead_ms"] = 250.01
        _write_json(summary_path, summary)
        message = "p95 overhead"
    else:
        rows_path = target["new_score"] / "comparisons.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        status = next(row for row in rows if row["field"] == "transfer_status")
        status["reference_status_class"] = "failed"
        status["non_success_to_success"] = True
        _write_jsonl(rows_path, rows)
        message = "truth class differs from subset-records"

    with pytest.raises(MODULE.ReplayError, match=message):
        MODULE.gate(
            prepared=prepared,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate",
        )


def test_prepare_selection_is_deterministic_and_gate_rejects_swapped_old_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    first = tmp_path / "prepared-first"
    second = tmp_path / "prepared-second"
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=first,
    )
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=second,
    )
    first_selection = json.loads((first / "selection.json").read_text())
    second_selection = json.loads((second / "selection.json").read_text())
    assert first_selection == second_selection

    manifest_path = first / "old-hybrid-subset" / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[0]["result"], manifest[1]["result"] = (
        manifest[1]["result"],
        manifest[0]["result"],
    )
    _write_json(manifest_path, manifest)
    summary_path = first / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["artifacts"]["old_hybrid_subset_manifest"] = _identity(manifest_path)
    _write_json(summary_path, summary)
    target = _target_fixture(tmp_path, first)

    with pytest.raises(MODULE.ReplayError, match="frozen formal manifest"):
        MODULE.gate(
            prepared=first,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate",
        )


def test_gate_rejects_coordinated_score_domain_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=prepared,
    )
    target = _target_fixture(tmp_path, prepared)
    _drop_score_row(target["old_score"], field="recipient_field")
    _drop_score_row(target["new_score"], field="recipient_field")

    with pytest.raises(MODULE.ReplayError, match="scorer domain is incomplete"):
        MODULE.gate(
            prepared=prepared,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate",
        )


def test_gate_rejects_wrong_missing_recoveries_below_frozen_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=prepared,
    )
    target = _target_fixture(tmp_path, prepared)
    selection = json.loads((prepared / "selection.json").read_text())
    missing_sources = [
        row["source"] for row in selection["records"] if row["role"] == "recipient_missing"
    ]
    for index, source in enumerate(missing_sources):
        _rewrite_new_recipient(target, source=source, candidate=f"错误候选{index}")

    with pytest.raises(MODULE.ReplayError, match="recipient_field raw_exact_match"):
        MODULE.gate(
            prepared=prepared,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate",
        )


def test_gate_rejects_type_only_invariant_drift_and_status_class_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _formal_fixture(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    MODULE.prepare(
        formal_root=fixture["formal"],
        diagnostic=fixture["diagnostic"],
        records=fixture["records"],
        output=prepared,
    )
    target = _target_fixture(tmp_path, prepared)
    manifest = json.loads((target["hybrid"] / "inference_manifest.json").read_text())
    result_path = Path(manifest[0]["result"])
    result = json.loads(result_path.read_text())
    result["geometry"]["source_size"]["width"] = 100.0
    _write_json(result_path, result)

    with pytest.raises(MODULE.ReplayError, match="changed geometry"):
        MODULE.gate(
            prepared=prepared,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate-type",
        )

    result["geometry"]["source_size"]["width"] = 100
    _write_json(result_path, result)
    rows_path = target["new_score"] / "comparisons.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    status = next(row for row in rows if row["field"] == "transfer_status")
    status["candidate_status_class"] = "unknown"
    _write_jsonl(rows_path, rows)

    with pytest.raises(MODULE.ReplayError, match="candidate class is inconsistent"):
        MODULE.gate(
            prepared=prepared,
            baseline=target["baseline"],
            hybrid=target["hybrid"],
            comparison=target["comparison"],
            old_score=target["old_score"],
            new_score=target["new_score"],
            output=tmp_path / "gate-status",
        )


def test_cli_never_exposes_a_formal_or_relaxed_count_switch() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "formal_delivery_gate\": False" in source
    assert "204 missing + 128 control" in source
    assert "--max-p95" not in source
    assert "--missing-records" not in source
    assert "--control-records" not in source
    assert "subprocess" not in source
    assert "stage.replace" not in source
    assert "stage.rename(output)" in source
    assert "os.O_CREAT | os.O_EXCL | os.O_WRONLY" in source
