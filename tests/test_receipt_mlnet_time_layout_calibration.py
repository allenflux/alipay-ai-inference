from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "receipt-mlnet-time-layout-calibration.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_time_layout_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(path: Path, *, records: int | None = None, sources: list[str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    if records is not None:
        value["records"] = records
    if sources is not None:
        value["normalized_source_set_sha256"] = MODULE.TARGETED._normalized_source_set_sha256(
            tuple(MODULE.TARGETED._source_key(source) for source in sources)
        )
    return value


def _result(source: Path, time: str, *, device: str, status: str) -> dict[str, Any]:
    return {
        "source": source.resolve().as_posix(),
        "geometry": {
            "rectification": "max-side-1600",
            "source_size": {"width": 1000, "height": 1600},
            "rectified_size": {"width": 1000, "height": 1600},
            "screen_detected": False,
            "rotation_degrees": 0,
        },
        "device": {"platform": device, "confidence": 0.99},
        "fields": {
            "amount": {"candidate": "10.00", "delivery_policy": "review"},
            "time": {"candidate": time, "delivery_policy": "review"},
            "payment_method": {"candidate": "余额", "delivery_policy": "review"},
            "recipient": {
                "candidate": "商户甲", "ctc_candidate": "商户甲",
                "hybrid_ocr_route": "primary", "delivery_policy": "review",
            },
            "transfer_status": {"candidate": status, "delivery_policy": "review"},
        },
        "detections": [],
    }


def _write_run(
    root: Path,
    sources: list[Path],
    candidates: list[str],
    devices: list[str],
    statuses: list[str],
    *,
    hybrid: bool,
) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    inference: list[float] = []
    stage_values: dict[str, list[float]] = {stage: [] for stage in MODULE.TARGETED.ALL_STAGES}
    for index, (source, candidate, device, status) in enumerate(
        zip(sources, candidates, devices, statuses, strict=True)
    ):
        result_path = root / "results" / f"{index}.json"
        _write_json(result_path, _result(source, candidate, device=device, status=status))
        inference_ms = 100.0 + index
        inference.append(inference_ms)
        stages: dict[str, float | None] = {
            "image_load": 1.0,
            "device": 2.0,
            "detector_preprocess": 3.0,
            "detector_inference": 4.0,
            "detector_postprocess": 5.0,
            "paddle_ocr": 6.0 if hybrid else None,
            "unified_ocr_preprocess": 7.0,
            "unified_ocr_inference": 8.0,
            "unified_ocr_postprocess": 9.0,
            "result_assembly": 10.0,
        }
        for stage, value in stages.items():
            if value is not None:
                stage_values[stage].append(value)
        manifest.append({
            "source": source.resolve().as_posix(),
            "result": result_path.resolve().as_posix(),
            "status": "written",
            "inference_ms": inference_ms,
            "stage_latency_ms": stages,
        })
    manifest_path = root / "inference_manifest.json"
    summary_path = root / "inference_summary.json"
    _write_json(manifest_path, manifest)
    (root / "inference_errors.jsonl").write_text("", encoding="utf-8")
    _write_json(summary_path, {
        "requested_device": "cpu",
        "paddle_ocr_provider": "cpu" if hybrid else None,
        "unified_provider": "cpu",
        "input": len(sources), "written": len(sources), "skipped": 0, "errors": 0,
        "total_seconds": float(len(sources)),
        "inference_latency_ms": MODULE.TARGETED._summarize(inference),
        "stage_latency_ms": {
            stage: MODULE.TARGETED._summarize(values) for stage, values in stage_values.items()
        },
    })
    rendered = [source.resolve().as_posix() for source in sources]
    return {
        "manifest": _identity(manifest_path, records=len(sources), sources=rendered),
        "summary": _identity(summary_path),
    }


@pytest.fixture()
def formal_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(MODULE, "FORMAL_RECORDS", 8)
    monkeypatch.setattr(MODULE, "SHARD_RECORDS", 3)
    monkeypatch.setattr(MODULE, "TARGET_RECORDS", 6)
    monkeypatch.setattr(MODULE.TARGETED, "FORMAL_RECORDS", 8)
    formal = tmp_path / "formal"
    # Formal inputs/manifests preserve records-manifest order, while the real
    # A/B comparator writes comparisons in sorted normalized-source-key order.
    source_order = [4, 1, 7, 0, 6, 2, 5, 3]
    sources = [tmp_path / "images" / f"receipt-{index}.jpg" for index in source_order]
    for index, source in enumerate(sources):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"image-{index}".encode())
    rendered = [source.resolve().as_posix() for source in sources]
    input_path = formal / "fixed-selected-inputs.txt"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("".join(f"{source}\n" for source in rendered), encoding="utf-8")
    references = ["05:40", "05:41", "05:42", "05:43", "05:44", "05:45", "05:46", "2026-08-08 05:47"]
    candidates = ["05:40", "11:11", "05:42", "11:13", "05:44", "11:15", "11:16", "05:47"]
    devices = ["android", "ios", "android", "ios", "android", "ios", "android", "ios"]
    statuses = ["success", "failed", "pending", "success", "failed", "pending", "success", "failed"]
    baseline_ids = _write_run(formal / "baseline-v13", sources, candidates, devices, statuses, hybrid=False)
    hybrid_ids = _write_run(formal / "hybrid-recipient", sources, candidates, devices, statuses, hybrid=True)
    comparison = formal / "comparison"
    rows = [
        {"source": source, "invariant": True, "failures": []}
        for source in sorted(rendered, key=MODULE.TARGETED._source_key)
    ]
    _write_jsonl(comparison / "comparisons.jsonl", rows)
    input_identity = _identity(input_path, records=8, sources=rendered)
    _write_json(comparison / "summary.json", {
        "schema_version": 2, "kind": MODULE.AB_KIND, "evaluation_mode": "formal",
        "records": 8, "input_set_identical": True, "cli_summary_counts_verified": True,
        "input_set": {"records": 8, "input_manifest": input_identity},
        "run_manifests": {"baseline": baseline_ids["manifest"], "hybrid": hybrid_ids["manifest"]},
        "run_summaries": {"baseline": baseline_ids["summary"], "hybrid": hybrid_ids["summary"]},
    })
    records_path = tmp_path / "unified-fields.jsonl"
    record_rows = []
    for index, (source, reference, status) in enumerate(zip(rendered, references, statuses, strict=True)):
        record_rows.append({
            "id": f"id-{index}", "group_id": f"group-{index // 2}", "split": "val", "source": source,
            "slots": {
                "amount": {"text": "10.00", "visible_text": "10.00"},
                "time": {"text": reference, "visible_text": reference},
                "payment_method_field": {"text": "余额"},
                "recipient_field": {"text": "商户甲"},
                "transfer_status": {"text": status, "class_name": status},
            },
        })
    _write_jsonl(records_path, record_rows)
    score = formal / "hybrid-val-score"
    score_rows = []
    field_values = {
        "amount": ("10.00", lambda index: "10.00"),
        "time": (None, lambda index: candidates[index]),
        "payment_method_field": ("余额", lambda index: "余额"),
        "recipient_field": ("商户甲", lambda index: "商户甲"),
        "transfer_status": (None, lambda index: statuses[index]),
    }
    for index, source in enumerate(rendered):
        for field, (fixed_reference, candidate_fn) in field_values.items():
            reference = (
                references[index] if field == "time"
                else statuses[index] if field == "transfer_status"
                else fixed_reference
            )
            candidate = candidate_fn(index)
            score_rows.append({
                "schema_version": 1, "kind": "receipt_mlnet_unified_comparison_v1",
                "field": field, "source": source, "reference_text": reference,
                "candidate_text": candidate, "candidate_present": candidate is not None,
                "raw_exact": candidate is not None and candidate == reference,
                "result_json": (
                    formal / "hybrid-recipient" / "results" / f"{index}.json"
                ).resolve().as_posix(),
            })
    _write_jsonl(score / "comparisons.jsonl", score_rows)
    model = tmp_path / "v13.onnx"
    model.write_bytes(b"fixture-v13")
    by_field: dict[str, dict[str, Any]] = {}
    for field in field_values:
        field_rows = [row for row in score_rows if row["field"] == field]
        exact = sum(row["raw_exact"] for row in field_rows)
        by_field[field] = {
            "records": 8, "raw_exact_matches": exact, "raw_exact_match": exact / 8,
        }
    field_reference_counts = {field: 8 for field in field_values}
    _write_json(score / "summary.json", {
        "schema_version": 1, "kind": MODULE.SCORE_KIND,
        "coverage_contract_version": 2, "evaluation_split": "val",
        "formal_delivery_gate": False, "accepted": False,
        "diagnostic_thresholds_passed": False,
        "failures": ["recipient exact/coverage floor not met"],
        "acceptance": {
            "passed": False, "formal_delivery_gate": False,
            "diagnostic_thresholds_passed": False,
            "failures": ["recipient exact/coverage floor not met"],
        },
        "evaluation_scope": {
            "kind": "full_split", "formal_delivery_gate": False,
            "requested_limit": None,
            "evaluated_expected_receipts": 8, "full_split_expected_receipts": 8,
            "input_list_path": input_path.resolve().as_posix(),
            "input_list_sha256": input_identity["sha256"],
            "selection_order": MODULE.FORMAL_AUDIT.FULL_SELECTION_ORDER,
        },
        "records": records_path.resolve().as_posix(),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "manifest": Path(hybrid_ids["manifest"]["path"]).resolve().as_posix(),
        "manifest_sha256": hybrid_ids["manifest"]["sha256"],
        "results_root": (formal / "hybrid-recipient").resolve().as_posix(),
        "model": model.resolve().as_posix(),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "floors": dict(MODULE.FORMAL_AUDIT.FIXED_FLOORS),
        "input_selection": {
            "path": input_path.resolve().as_posix(), "sha256": input_identity["sha256"],
            "records": 8, "hash_bound": True,
            "selection_order": MODULE.FORMAL_AUDIT.FULL_SELECTION_ORDER,
            "field_reference_counts": field_reference_counts,
        },
        "accuracy_denominators": {"hash_bound": True, "by_field": field_reference_counts},
        "by_field": by_field,
    })
    return {
        "root": tmp_path, "formal": formal, "sources": sources, "references": references,
        "candidates": candidates, "records": records_path, "score": score,
    }


def _prepare(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return MODULE.prepare(
        formal_root=fixture["formal"], records_path=fixture["records"],
        score_directory=fixture["score"], output_directory=output,
    )


def test_prepare_cli_records_from_score_is_mutually_exclusive_and_uses_bound_path(
    formal_fixture: dict[str, Any]
) -> None:
    output = formal_fixture["root"] / "prepared-from-score"
    exit_code = MODULE.main([
        "prepare",
        "--formal-root", str(formal_fixture["formal"]),
        "--records-from-score",
        "--score-directory", str(formal_fixture["score"]),
        "--output-directory", str(output),
    ])
    assert exit_code == 0
    assert output.is_dir()
    with pytest.raises(SystemExit):
        MODULE._parser().parse_args([
            "prepare", "--formal-root", str(formal_fixture["formal"]),
            "--records", str(formal_fixture["records"]), "--records-from-score",
            "--score-directory", str(formal_fixture["score"]),
            "--output-directory", str(formal_fixture["root"] / "both"),
        ])
    with pytest.raises(SystemExit):
        MODULE._parser().parse_args([
            "prepare", "--formal-root", str(formal_fixture["formal"]),
            "--score-directory", str(formal_fixture["score"]),
            "--output-directory", str(formal_fixture["root"] / "neither"),
        ])


def test_records_from_score_rejects_relative_missing_and_symlink_paths(
    formal_fixture: dict[str, Any]
) -> None:
    summary_path = formal_fixture["score"] / "summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    relative = dict(original)
    relative["records"] = "unified-fields.jsonl"
    _write_json(summary_path, relative)
    with pytest.raises(MODULE.CalibrationError, match="absolute"):
        MODULE._records_path_from_score(formal_fixture["score"])

    missing = dict(original)
    missing["records"] = (formal_fixture["root"] / "missing-records.jsonl").resolve().as_posix()
    _write_json(summary_path, missing)
    with pytest.raises(MODULE.CalibrationError, match="missing"):
        MODULE._records_path_from_score(formal_fixture["score"])

    link = formal_fixture["root"] / "records-link.jsonl"
    try:
        link.symlink_to(formal_fixture["records"])
    except OSError:
        pytest.skip("symlink creation is unavailable")
    linked = dict(original)
    linked["records"] = link.resolve(strict=False).as_posix()
    # Keep the symlink path rather than Path.resolve()'s target.
    linked["records"] = link.absolute().as_posix()
    _write_json(summary_path, linked)
    with pytest.raises(MODULE.CalibrationError, match="symlink|reparse|junction"):
        MODULE._records_path_from_score(formal_fixture["score"])


def test_records_from_score_preserves_path_and_hash_binding(formal_fixture: dict[str, Any]) -> None:
    copy = formal_fixture["root"] / "records-copy.jsonl"
    shutil.copyfile(formal_fixture["records"], copy)
    with pytest.raises(MODULE.CalibrationError, match="records binding"):
        MODULE.prepare(
            formal_root=formal_fixture["formal"], records_path=copy,
            score_directory=formal_fixture["score"],
            output_directory=formal_fixture["root"] / "copy-output",
        )

    rows = [json.loads(line) for line in formal_fixture["records"].read_text(encoding="utf-8").splitlines()]
    rows[0]["id"] = "tampered-but-valid-json"
    _write_jsonl(formal_fixture["records"], rows)
    exit_code = MODULE.main([
        "prepare", "--formal-root", str(formal_fixture["formal"]),
        "--records-from-score", "--score-directory", str(formal_fixture["score"]),
        "--output-directory", str(formal_fixture["root"] / "tampered-output"),
    ])
    assert exit_code == 2
    assert not (formal_fixture["root"] / "tampered-output").exists()


def test_records_from_score_rejects_windows_reparse_attribute(
    formal_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = formal_fixture["records"].resolve()
    original_lstat = Path.lstat
    original_metadata = original_lstat(records)

    def fake_lstat(path: Path):
        if path == records:
            return SimpleNamespace(
                st_file_attributes=0x400,
                st_mode=original_metadata.st_mode,
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(MODULE.CalibrationError, match="reparse"):
        MODULE._require_regular_non_reparse_file(
            records, description="Windows records fixture"
        )


def _line(index: int, text: str, confidence: float, box: tuple[float, float, float, float]) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    raw = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return {
        "index": index, "text": text, "confidence": confidence,
        "passes_drop_score": confidence >= 0.5,
        "quad_rectified": raw,
        "quad_rectified_normalized": [[x / 999.0, y / 1599.0] for x, y in raw],
    }


def _bundle(root: Path) -> dict[str, Any]:
    root.mkdir()
    contract = root / "paddle_ocr_delivery.contract.json"
    contract.write_bytes(b"contract")
    result: dict[str, Any] = {
        "directory": root.resolve().as_posix(),
        "contract_path": contract.resolve().as_posix(),
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "source_audit_contract_sha256": "a" * 64,
    }
    total = 0
    for role in ("detector", "classifier", "recognizer", "dictionary"):
        path = root / f"{role}.bin"
        path.write_bytes(role.encode())
        total += path.stat().st_size
        result[role] = {
            "relative_path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    result["package_size_bytes"] = total
    return result


def _write_layout(
    output: Path,
    prepared: Path,
    shard_index: int,
    bundle: dict[str, Any],
    candidate_overrides: dict[str, str | None] | None = None,
) -> None:
    output.mkdir()
    input_path = prepared / f"shard-{shard_index}-inputs.txt"
    sources = input_path.read_text(encoding="utf-8").splitlines()
    truths = [json.loads(line) for line in (prepared / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    truth_by_source = {row["source"]: row for row in truths}
    records = []
    totals: list[float] = []
    stage_values = {"image_load": [], "rectification": [], "layout_ocr": [], "total": []}
    for index, source in enumerate(sources):
        truth = truth_by_source[source]
        candidate = truth["reference_text"]
        if candidate_overrides and source in candidate_overrides:
            candidate = candidate_overrides[source]
        lines = [_line(0, candidate, 0.95, (30, 20, 140, 60))] if candidate is not None else []
        lines.append(_line(len(lines), "付款方式 余额", 0.93, (100, 500, 400, 550)))
        accepted = [line for line in lines if line["passes_drop_score"]]
        timing = {
            "image_load": 1.0 + index / 10,
            "rectification": 2.0,
            "layout_ocr": 3.0,
            "total": 6.0 + index,
        }
        for stage, value in timing.items():
            stage_values[stage].append(value)
        totals.append(timing["total"])
        source_path = Path(source)
        records.append({
            "schema_version": 1, "kind": MODULE.LAYOUT_RECORD_KIND,
            "diagnostic_only": True, "formal_delivery_gate": False,
            "candidate_write_enabled": False, "index": index, "source": source,
            "source_image_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_image_size_bytes": source_path.stat().st_size,
            "execution_provider": "cpu",
            "geometry": {
                "source_size": {"width": 1000, "height": 1600},
                "rectified_size": {"width": 1000, "height": 1600},
                "rectification": MODULE.LAYOUT_EVIDENCE.RECTIFICATION,
                "rotation_degrees": 0, "screen_detected": False,
                "screen_quad_original": [[0, 0], [999, 0], [999, 1599], [0, 1599]],
                "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "H_rectified_to_original": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "quad_coordinate_space": MODULE.LAYOUT_EVIDENCE.QUAD_COORDINATE_SPACE,
            "quad_normalization": MODULE.LAYOUT_EVIDENCE.QUAD_NORMALIZATION,
            "confidence_semantics": MODULE.LAYOUT_EVIDENCE.CONFIDENCE_SEMANTICS,
            "accepted_text": " ".join(line["text"] for line in accepted),
            "accepted_confidence": sum(line["confidence"] for line in accepted) / len(accepted),
            "accepted_line_count": len(accepted), "raw_line_count": len(lines),
            "lines": lines, "timing_ms": timing,
        })
    _write_jsonl(output / "records.jsonl", records)
    records_identity = _identity(output / "records.jsonl")
    input_identity = _identity(input_path)
    _write_json(output / "summary.json", {
        "schema_version": 1, "kind": MODULE.LAYOUT_SUMMARY_KIND,
        "diagnostic_only": True, "formal_delivery_gate": False,
        "candidate_write_enabled": False, "expected_records": 3, "records": 3,
        "errors": 0, "execution_provider": "cpu",
        "rectification": MODULE.LAYOUT_EVIDENCE.RECTIFICATION,
        "quad_coordinate_space": MODULE.LAYOUT_EVIDENCE.QUAD_COORDINATE_SPACE,
        "quad_normalization": MODULE.LAYOUT_EVIDENCE.QUAD_NORMALIZATION,
        "confidence_semantics": MODULE.LAYOUT_EVIDENCE.CONFIDENCE_SEMANTICS,
        "paddle_drop_score": 0.5,
        "input_list": {**input_identity, "records": 3},
        "paddle_bundle": bundle,
        "latency_ms": {stage: MODULE._latency(values) for stage, values in stage_values.items()},
        "artifacts": {"records_jsonl": {
            "relative_path": "records.jsonl", "sha256": records_identity["sha256"],
            "size_bytes": records_identity["size_bytes"],
        }},
    })


def _replace_layout_line_quad(
    output: Path,
    *,
    record_index: int,
    line_index: int,
    quad: list[list[float]],
) -> None:
    records_path = output / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    line = rows[record_index]["lines"][line_index]
    line["quad_rectified"] = quad
    line["quad_rectified_normalized"] = [
        [point[0] / 999.0, point[1] / 1599.0]
        for point in quad
    ]
    _write_jsonl(records_path, rows)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = _identity(records_path)
    summary["artifacts"]["records_jsonl"].update({
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    })
    _write_json(summary_path, summary)


def test_prepare_and_evaluate_two_frozen_shards(formal_fixture: dict[str, Any]) -> None:
    prepared = formal_fixture["root"] / "prepared"
    result = _prepare(formal_fixture, prepared)
    assert result["records"] == 6
    shard_0 = (prepared / "shard-0-inputs.txt").read_text(encoding="utf-8").splitlines()
    shard_1 = (prepared / "shard-1-inputs.txt").read_text(encoding="utf-8").splitlines()
    truth = [json.loads(line) for line in (prepared / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    selection_payload = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    assert selection_payload["source_score_disposition"] == {
        "source_score_accepted": False,
        "source_score_formal_delivery_gate": False,
        "source_score_diagnostic_thresholds_passed": False,
        "source_score_failures": ["recipient exact/coverage floor not met"],
        "source_score_scope": "hash_bound_full_split_val",
        "source_score_five_field_floors": MODULE.FORMAL_AUDIT.FIXED_FLOORS,
        "inherited_delivery_authority": False,
    }
    assert len(shard_0) == len(shard_1) == 3
    assert shard_0 + shard_1 == [row["source"] for row in truth]
    assert len(set(shard_0 + shard_1)) == 6
    assert {row["old_v13_raw_exact"] for row in truth} == {True, False}
    assert all("2026-" not in row["reference_text"] for row in truth)
    prepared_repeat = formal_fixture["root"] / "prepared-repeat"
    _prepare(formal_fixture, prepared_repeat)
    for name in ("selection.json", "truth.jsonl", "pool-closure.jsonl",
                 "shard-0-inputs.txt", "shard-1-inputs.txt"):
        assert (prepared / name).read_bytes() == (prepared_repeat / name).read_bytes()

    bundle = _bundle(formal_fixture["root"] / "bundle")
    layout_0 = formal_fixture["root"] / "layout-0"
    layout_1 = formal_fixture["root"] / "layout-1"
    correct_source = next(row["source"] for row in truth if row["old_v13_raw_exact"])
    overrides = {correct_source: "01:11"}
    _write_layout(layout_0, prepared, 0, bundle, overrides)
    _write_layout(layout_1, prepared, 1, bundle, overrides)
    evaluated = MODULE.evaluate(
        prepared_directory=prepared, layout_shard_0=layout_0, layout_shard_1=layout_1,
        output_directory=formal_fixture["root"] / "evaluated",
    )
    assert evaluated["records"] == 6
    assert evaluated["overall"]["candidate_coverage"] == 1.0
    assert evaluated["overall"]["correct_to_wrong"] == 1
    assert evaluated["overall"]["wrong_to_correct"] >= 1
    assert evaluated["overall"]["raw_exact_accuracy_delta"] > 0
    assert evaluated["cpu_latency_ms"]["count"] == 6
    assert len(evaluated["cpu_latency_ms_by_shard"]) == 2
    assert evaluated["candidate_write_enabled"] is False
    assert evaluated["formal_delivery_gate"] is False
    assert evaluated["source_score_disposition"]["source_score_formal_delivery_gate"] is False


def test_evaluate_quarantines_complete_record_with_invalid_quad_contract(
    formal_fixture: dict[str, Any],
) -> None:
    prepared = formal_fixture["root"] / "prepared-invalid-quad"
    _prepare(formal_fixture, prepared)
    bundle = _bundle(formal_fixture["root"] / "bundle-invalid-quad")
    layout_0 = formal_fixture["root"] / "layout-invalid-quad-0"
    layout_1 = formal_fixture["root"] / "layout-invalid-quad-1"
    _write_layout(layout_0, prepared, 0, bundle)
    _write_layout(layout_1, prepared, 1, bundle)
    # Keep the valid clock line intact and corrupt an unrelated accepted body
    # line.  The complete record must still be candidate-ineligible.
    _replace_layout_line_quad(
        layout_0,
        record_index=0,
        line_index=1,
        quad=[[306.0, 25.0], [292.0, 140.0], [321.0, 140.0], [306.0, 155.0]],
    )

    output = formal_fixture["root"] / "evaluated-invalid-quad"
    evaluated = MODULE.evaluate(
        prepared_directory=prepared,
        layout_shard_0=layout_0,
        layout_shard_1=layout_1,
        output_directory=output,
    )
    comparisons = [
        json.loads(line)
        for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    quarantined = [row for row in comparisons if row["layout_record_quarantined"]]
    assert len(quarantined) == 1
    assert quarantined[0]["candidate_text"] is None
    assert quarantined[0]["candidate_present"] is False
    assert quarantined[0]["route_ambiguity"] == "invalid_quad_contract_record_quarantined"
    assert quarantined[0]["invalid_quad_contract_lines"][0]["classification"] \
        == "self_intersects_nondegenerate_hull"
    assert evaluated["overall"]["candidate_records"] == 5
    assert evaluated["layout_geometry_safety"] == {
        "invalid_quad_contract_violation_lines": 1,
        "records_forced_candidate_ineligible": 1,
        "by_classification": {"self_intersects_nondegenerate_hull": 1},
        "invalid_quad_geometry_used": False,
        "invalid_quad_canonicalized": 0,
        "contract_violation_policy": "fail_closed_whole_record_unresolved",
        "quarantined_records_remain_in_accuracy_denominator": True,
    }


def test_evaluate_still_rejects_out_of_bounds_quad(
    formal_fixture: dict[str, Any],
) -> None:
    prepared = formal_fixture["root"] / "prepared-out-of-bounds"
    _prepare(formal_fixture, prepared)
    bundle = _bundle(formal_fixture["root"] / "bundle-out-of-bounds")
    layout_0 = formal_fixture["root"] / "layout-out-of-bounds-0"
    layout_1 = formal_fixture["root"] / "layout-out-of-bounds-1"
    _write_layout(layout_0, prepared, 0, bundle)
    _write_layout(layout_1, prepared, 1, bundle)
    _replace_layout_line_quad(
        layout_0,
        record_index=0,
        line_index=1,
        quad=[[-1.0, 25.0], [30.0, 25.0], [30.0, 40.0], [-1.0, 40.0]],
    )
    with pytest.raises(ValueError, match="quad x"):
        MODULE.evaluate(
            prepared_directory=prepared,
            layout_shard_0=layout_0,
            layout_shard_1=layout_1,
            output_directory=formal_fixture["root"] / "out-of-bounds-output",
        )


def test_prepare_rejects_duplicate_nonstrict_score_row(formal_fixture: dict[str, Any]) -> None:
    comparisons = formal_fixture["score"] / "comparisons.jsonl"
    rows = [json.loads(line) for line in comparisons.read_text(encoding="utf-8").splitlines()]
    rows.append(dict(rows[-1]))
    _write_jsonl(comparisons, rows)
    with pytest.raises(MODULE.CalibrationError, match="duplicate"):
        _prepare(formal_fixture, formal_fixture["root"] / "bad-prepare")


def test_prepare_requires_real_comparator_canonical_order(formal_fixture: dict[str, Any]) -> None:
    comparison = formal_fixture["formal"] / "comparison" / "comparisons.jsonl"
    fixed_sources = (formal_fixture["formal"] / "fixed-selected-inputs.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    _write_jsonl(
        comparison,
        [{"source": source, "invariant": True, "failures": []} for source in fixed_sources],
    )
    with pytest.raises(MODULE.CalibrationError, match="comparator canonical"):
        _prepare(formal_fixture, formal_fixture["root"] / "wrong-comparison-order")


def test_prepare_rejects_floor_or_five_field_closure_drift(formal_fixture: dict[str, Any]) -> None:
    summary_path = formal_fixture["score"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["floors"]["time"] = 0.1
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.CalibrationError, match="five-field floors"):
        _prepare(formal_fixture, formal_fixture["root"] / "bad-floor")

    summary["floors"] = dict(MODULE.FORMAL_AUDIT.FIXED_FLOORS)
    _write_json(summary_path, summary)
    comparisons_path = formal_fixture["score"] / "comparisons.jsonl"
    rows = [json.loads(line) for line in comparisons_path.read_text(encoding="utf-8").splitlines()]
    payment = next(row for row in rows if row["field"] == "payment_method_field")
    payment["candidate_text"] = "花呗"
    _write_jsonl(comparisons_path, rows)
    with pytest.raises(MODULE.CalibrationError, match="five-field comparison closure"):
        _prepare(formal_fixture, formal_fixture["root"] / "bad-five-field")


def test_prepare_rejects_inconsistent_acceptance_failures(
    formal_fixture: dict[str, Any],
) -> None:
    summary_path = formal_fixture["score"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["acceptance"]["failures"] = []
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.CalibrationError, match="disposition flags/failures"):
        _prepare(formal_fixture, formal_fixture["root"] / "bad-acceptance-failures")


def test_load_prepared_rejects_rehashed_non_time_score_tamper(
    formal_fixture: dict[str, Any],
) -> None:
    prepared = formal_fixture["root"] / "prepared-score-tamper"
    _prepare(formal_fixture, prepared)

    comparisons_path = formal_fixture["score"] / "comparisons.jsonl"
    rows = [
        json.loads(line)
        for line in comparisons_path.read_text(encoding="utf-8").splitlines()
    ]
    payment = next(row for row in rows if row["field"] == "payment_method_field")
    payment["candidate_text"] = "花呗"
    payment["raw_exact"] = False
    _write_jsonl(comparisons_path, rows)

    selection_path = prepared / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["source_evidence"]["score_comparisons"] = _identity(comparisons_path)
    _write_json(selection_path, selection)
    summary_path = prepared / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["selection"].update(_identity(selection_path))
    _write_json(summary_path, summary)

    with pytest.raises(MODULE.CalibrationError, match="five-field comparison closure"):
        MODULE._load_prepared(prepared)


def test_evaluate_rejects_rehashed_truth_semantic_tamper(formal_fixture: dict[str, Any]) -> None:
    prepared = formal_fixture["root"] / "prepared-tamper"
    _prepare(formal_fixture, prepared)
    truth_path = prepared / "truth.jsonl"
    rows = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["reference_text"] = "01:23"
    _write_jsonl(truth_path, rows)
    selection_path = prepared / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["artifacts"]["truth"].update(_identity(truth_path))
    selection["artifacts"]["truth"]["relative_path"] = "truth.jsonl"
    selection["artifacts"]["truth"]["records"] = 6
    _write_json(selection_path, selection)
    summary_path = prepared / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["truth"] = dict(selection["artifacts"]["truth"])
    summary["artifacts"]["selection"].update(_identity(selection_path))
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.CalibrationError, match="external reference|source semantics"):
        MODULE._load_prepared(prepared)


def test_evaluate_rejects_swapped_shards_and_non_cpu(formal_fixture: dict[str, Any]) -> None:
    prepared = formal_fixture["root"] / "prepared-contract"
    _prepare(formal_fixture, prepared)
    bundle = _bundle(formal_fixture["root"] / "bundle-contract")
    layout_0 = formal_fixture["root"] / "layout-contract-0"
    layout_1 = formal_fixture["root"] / "layout-contract-1"
    _write_layout(layout_0, prepared, 0, bundle)
    _write_layout(layout_1, prepared, 1, bundle)
    with pytest.raises(MODULE.CalibrationError, match="input binding"):
        MODULE.evaluate(
            prepared_directory=prepared, layout_shard_0=layout_1, layout_shard_1=layout_0,
            output_directory=formal_fixture["root"] / "swapped-output",
        )
    summary_path = layout_0 / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["execution_provider"] = "cuda"
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.CalibrationError, match="execution_provider"):
        MODULE.evaluate(
            prepared_directory=prepared, layout_shard_0=layout_0, layout_shard_1=layout_1,
            output_directory=formal_fixture["root"] / "cuda-output",
        )


def test_evaluate_rechecks_prepared_artifacts_before_publish(
    formal_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = formal_fixture["root"] / "prepared-toctou"
    _prepare(formal_fixture, prepared)
    bundle = _bundle(formal_fixture["root"] / "bundle-toctou")
    layout_0 = formal_fixture["root"] / "layout-toctou-0"
    layout_1 = formal_fixture["root"] / "layout-toctou-1"
    _write_layout(layout_0, prepared, 0, bundle)
    _write_layout(layout_1, prepared, 1, bundle)
    original = MODULE._write_json
    changed = False

    def mutate_after_stage_summary(path: Path, value: Any) -> None:
        nonlocal changed
        original(path, value)
        if not changed and path.name == "summary.json" and path.parent.name.endswith(".tmp"):
            changed = True
            with (prepared / "truth.jsonl").open("ab") as stream:
                stream.write(b" ")

    monkeypatch.setattr(MODULE, "_write_json", mutate_after_stage_summary)
    output = formal_fixture["root"] / "toctou-output"
    with pytest.raises(MODULE.CalibrationError, match="identity changed|mismatch"):
        MODULE.evaluate(
            prepared_directory=prepared, layout_shard_0=layout_0, layout_shard_1=layout_1,
            output_directory=output,
        )
    assert changed is True
    assert not output.exists()
