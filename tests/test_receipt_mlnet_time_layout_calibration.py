from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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
    sources = [tmp_path / "images" / f"receipt-{index}.jpg" for index in range(8)]
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
    rows = [{"source": source, "invariant": True, "failures": []} for source in rendered]
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
                "time": {"text": reference, "visible_text": reference},
                "transfer_status": {"class_name": status},
            },
        })
    _write_jsonl(records_path, record_rows)
    score = formal / "hybrid-val-score"
    score_rows = []
    for source, reference, candidate in zip(rendered, references, candidates, strict=True):
        score_rows.append({
            "schema_version": 1, "kind": "receipt_mlnet_unified_comparison_v1",
            "field": "time", "source": source, "reference_text": reference,
            "candidate_text": candidate, "candidate_present": True,
            "raw_exact": candidate == reference,
        })
    _write_jsonl(score / "comparisons.jsonl", score_rows)
    exact = sum(row["raw_exact"] for row in score_rows)
    _write_json(score / "summary.json", {
        "schema_version": 1, "kind": MODULE.SCORE_KIND,
        "formal_delivery_gate": True, "accepted": True,
        "evaluation_scope": {
            "kind": "full_split", "formal_delivery_gate": True,
            "evaluated_expected_receipts": 8, "full_split_expected_receipts": 8,
        },
        "records": records_path.resolve().as_posix(),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "manifest": Path(hybrid_ids["manifest"]["path"]).resolve().as_posix(),
        "manifest_sha256": hybrid_ids["manifest"]["sha256"],
        "input_selection": {
            "path": input_path.resolve().as_posix(), "sha256": input_identity["sha256"],
            "records": 8, "hash_bound": True,
        },
        "by_field": {"time": {
            "records": 8, "raw_exact_matches": exact, "raw_exact_match": exact / 8,
        }},
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


def test_prepare_and_evaluate_two_frozen_shards(formal_fixture: dict[str, Any]) -> None:
    prepared = formal_fixture["root"] / "prepared"
    result = _prepare(formal_fixture, prepared)
    assert result["records"] == 6
    shard_0 = (prepared / "shard-0-inputs.txt").read_text(encoding="utf-8").splitlines()
    shard_1 = (prepared / "shard-1-inputs.txt").read_text(encoding="utf-8").splitlines()
    truth = [json.loads(line) for line in (prepared / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
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


def test_prepare_rejects_duplicate_nonstrict_score_row(formal_fixture: dict[str, Any]) -> None:
    comparisons = formal_fixture["score"] / "comparisons.jsonl"
    rows = [json.loads(line) for line in comparisons.read_text(encoding="utf-8").splitlines()]
    rows.append(dict(rows[-1]))
    _write_jsonl(comparisons, rows)
    with pytest.raises(MODULE.CalibrationError, match="duplicate"):
        _prepare(formal_fixture, formal_fixture["root"] / "bad-prepare")


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
