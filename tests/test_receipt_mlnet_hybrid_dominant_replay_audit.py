from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-hybrid-dominant-replay-audit.py"
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_hybrid_dominant_replay_audit", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TARGET_TEST = ROOT / "tests" / "test_receipt_mlnet_hybrid_targeted_replay.py"
TARGET_SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_targeted_replay_test_fixture", TARGET_TEST
)
assert TARGET_SPEC is not None and TARGET_SPEC.loader is not None
TARGET = importlib.util.module_from_spec(TARGET_SPEC)
TARGET_SPEC.loader.exec_module(TARGET)


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


def _patch_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(MODULE, "FAILURE_RECORDS", 2)
    monkeypatch.setattr(MODULE, "EXPECTED_CANDIDATES", 2)
    monkeypatch.setattr(MODULE, "EXPECTED_MISSING", 0)
    monkeypatch.setattr(MODULE, "EXPECTED_DOMINANT", 1)
    monkeypatch.setattr(MODULE, "EXPECTED_PREEXISTING", 1)
    monkeypatch.setattr(MODULE.REPLAY, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(MODULE.REPLAY, "MISSING_RECORDS", 2)


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    _patch_counts(monkeypatch)
    formal = TARGET._formal_fixture(tmp_path, monkeypatch)
    missing_sources = formal["sources"][:2]
    rendered_sources = [source.resolve().as_posix() for source in missing_sources]

    probe = tmp_path / "probe-v4"
    probe.mkdir()
    probe_findings = []
    candidates = ["旧恢复商户", "三路商户"]
    routes = [MODULE.EXACT_ROUTE, MODULE.DOMINANT_ROUTE]
    for source, candidate, route in zip(
        rendered_sources, candidates, routes, strict=True
    ):
        shadow = {
            "state": "candidate",
            "candidate": candidate,
            "runtime_route": route,
        }
        probe_findings.append(
            {
                "schema_version": 1,
                "kind": MODULE.PROBE_FINDING_KIND,
                "source": source,
                "truth_used_for_analysis_only": True,
                "runtime_truth_lookup": False,
                "formal_delivery_gate": False,
                "shadow_candidate_truth_free": shadow,
                "paddle_teacher_consensus": dict(shadow),
            }
        )
    _write_jsonl(probe / "findings.jsonl", probe_findings)
    _write_json(
        probe / "summary.json",
        {
            "schema_version": 1,
            "kind": MODULE.PROBE_SUMMARY_KIND,
            "read_only_existing_diagnostic": True,
            "ocr_rerun": False,
            "truth_used_for_analysis_only": True,
            "runtime_truth_lookup": False,
            "formal_delivery_gate": False,
            "findings_records": 2,
            "unique_sources": 2,
            "formal_contract": {
                "comparison_evaluation_mode": "formal",
                "comparison_records": 6,
                "failed_records": 2,
                "recipient_missing_only_records": 2,
                "recipient_missing_with_additional_failures_records": 0,
                "non_missing_invariant_failure_records": 0,
            },
            "paddle_teacher_consensus": {
                "external_truth": False,
                "truth_used_for_analysis_only": True,
                "formal_delivery_gate": False,
                "records": 2,
                "interpretation": "self_consistency_coverage_not_human_accuracy",
                "by_runtime_route": [
                    {"name": MODULE.DOMINANT_ROUTE, "records": 1},
                    {"name": MODULE.EXACT_ROUTE, "records": 1},
                ],
                "contract": {
                    "dominant_fallback_requires_multiple_eligible_candidates": True,
                    "dominant_fallback_requires_same_exact_line_in_all_crops": 3,
                    "dominant_fallback_requires_unique_all_crop_candidate": True,
                    "requires_same_exact_line_in_independent_crops": 2,
                },
            },
            "remaining_failure_analysis": {
                "records": 0,
                "strict_candidate_records": 2,
            },
            "source_evidence": {
                "input_summary": TARGET._identity(
                    formal["diagnostic"] / "summary.json"
                ),
                "input_findings": TARGET._identity(
                    formal["diagnostic"] / "findings.jsonl"
                ),
            },
        },
    )

    replay = tmp_path / "fresh-dominant-replay"
    TARGET._write_run(
        replay,
        missing_sources,
        hybrid=True,
        candidates=candidates,
        routes=routes,
    )
    manifest = json.loads((replay / "inference_manifest.json").read_text())
    for row, candidate in zip(manifest, candidates, strict=True):
        result_path = Path(row["result"])
        result = json.loads(result_path.read_text())
        recipient = result["fields"]["recipient"]
        recipient.update(
            {
                "state": "review",
                "raw": candidate,
                "ocr_confidence": 0.95,
                "ctc_confidence": 0.95,
                "structured_candidate": None,
                "structured_confidence": None,
            }
        )
        detection = next(
            item
            for item in result["detections"]
            if item["label"] == "recipient_field"
        )
        detection["ocr"] = {"text": candidate, "confidence": 0.95}
        _write_json(result_path, result)

    Path(str(replay.resolve()) + ".inputs.txt").write_text(
        "".join(f"{source}\n" for source in rendered_sources), encoding="utf-8"
    )
    cli = Path(str(replay.resolve()) + ".cli-app")
    cli.mkdir()
    (cli / "ReceiptMlNet.Cli.dll").write_bytes(b"dominant-cli")
    (cli / "onnxruntime.dll").write_bytes(b"cpu-onnxruntime")
    (cli / "ReceiptMlNet.Cli.deps.json").write_bytes(b"{}")
    return {
        **formal,
        "probe": probe,
        "replay": replay,
        "cli": cli,
        "manifest": manifest,
    }


def _run(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return MODULE.audit(
        formal_root=fixture["formal"],
        diagnostic_directory=fixture["diagnostic"],
        probe_directory=fixture["probe"],
        replay_directory=fixture["replay"],
        output_directory=output,
    )


def test_audit_binds_dominant_replay_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "audit"

    summary = _run(fixture, output)

    assert summary["accepted"] is True
    assert summary["formal_delivery_gate"] is False
    assert summary["counts"] == {
        "formal_records": 6,
        "replay_records": 2,
        "candidate_records": 2,
        "missing_records": 0,
        "preexisting_exact_consensus_preserved": 1,
        "dominant_three_crop_records": 1,
        "ctc_candidate_equals_candidate": 2,
        "nonrecipient_invariant_records": 2,
        "transfer_status_non_success_to_success": 0,
    }
    assert summary["cpu_contract"] == {
        "requested_device": "cpu",
        "unified_provider": "cpu",
        "paddle_ocr_provider": "cpu",
        "replay_errors": 0,
    }
    findings = [
        json.loads(line)
        for line in (output / "findings.jsonl").read_text().splitlines()
    ]
    assert len(findings) == 2
    assert all(row["ctc_candidate_equals_candidate"] for row in findings)
    assert all(
        row["detector_device_geometry_and_other_fields_identical"]
        for row in findings
    )
    assert json.loads((output / "cli-closure.json").read_text())
    assert not list(tmp_path.glob(".audit.*.tmp"))

    with pytest.raises(MODULE.AuditError, match="refusing to overwrite"):
        _run(fixture, output)


def test_audit_rejects_nonrecipient_or_ctc_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    first = Path(fixture["manifest"][0]["result"])
    result = json.loads(first.read_text())
    result["fields"]["amount"]["candidate"] = "999.00"
    _write_json(first, result)

    with pytest.raises(MODULE.AuditError, match=r"changed fields\.amount"):
        _run(fixture, tmp_path / "amount-audit")

    fixture = _fixture(tmp_path / "ctc", monkeypatch)
    first = Path(fixture["manifest"][0]["result"])
    result = json.loads(first.read_text())
    result["fields"]["recipient"]["ctc_candidate"] = "不一致"
    _write_json(first, result)
    with pytest.raises(MODULE.AuditError, match="candidate/ctc_candidate"):
        _run(fixture, tmp_path / "ctc-audit")


def test_audit_rejects_preexisting_recovery_regression_and_gpu_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    first = Path(fixture["manifest"][0]["result"])
    result = json.loads(first.read_text())
    result["fields"]["recipient"]["candidate"] = None
    result["fields"]["recipient"]["ctc_candidate"] = None
    _write_json(first, result)
    with pytest.raises(MODULE.AuditError, match="differs from the probe"):
        _run(fixture, tmp_path / "regression-audit")

    fixture = _fixture(tmp_path / "gpu", monkeypatch)
    (fixture["cli"] / "onnxruntime_providers_cuda.dll").write_bytes(b"gpu")
    with pytest.raises(MODULE.AuditError, match="GPU provider"):
        _run(fixture, tmp_path / "gpu-audit")


def test_audit_rejects_probe_hash_or_cpu_provider_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    (fixture["diagnostic"] / "findings.jsonl").write_text(
        (fixture["diagnostic"] / "findings.jsonl").read_text() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MODULE.AuditError, match="invalid failure diagnostic|binding mismatch"):
        _run(fixture, tmp_path / "hash-audit")

    fixture = _fixture(tmp_path / "provider", monkeypatch)
    summary_path = fixture["replay"] / "inference_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["paddle_ocr_provider"] = "cuda"
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.AuditError, match="must use CPU PP-OCR"):
        _run(fixture, tmp_path / "provider-audit")
