from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-v13-recipient-fallback-audit.py"
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_v13_recipient_fallback_audit", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _patch_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "FORMAL_RECORDS", 6)
    monkeypatch.setattr(MODULE, "MISSING_RECORDS", 4)
    monkeypatch.setattr(MODULE, "STRICT_PSEUDO_TRUTH_RECORDS", 2)
    monkeypatch.setattr(MODULE, "REMAINING_RECORDS", 2)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    baselines: list[str | None] | None = None,
    references: list[str | None] | None = None,
) -> dict[str, Any]:
    _patch_counts(monkeypatch)
    sources = [rf"D:\receipts\formal\{index:05d}.jpg" for index in range(4)]
    pseudo = ["商户甲", "商户乙", None, None]
    states = ["candidate", "candidate", "unresolved", "rejected_by_global_gate"]
    baselines = baselines or ["商户甲", "错误商户", "旧商户丙", None]
    references = references or [None, None, None, None]

    missing_findings: list[dict[str, Any]] = []
    probe_findings: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        reference = references[index]
        missing_findings.append(
            {
                "source": source,
                "invariant": False,
                "recipient_candidate": None,
                "failures": [MODULE.RECIPIENT_MISSING_FAILURE],
                "baseline_recipient_field": {
                    "state": "review" if baselines[index] else "unreadable",
                    "candidate": baselines[index],
                },
                "hybrid_recipient_field": {
                    "state": "unreadable",
                    "candidate": None,
                },
                "hybrid_recipient_reference_evidence": {
                    "field": "recipient_field",
                    "reference_text": reference,
                    "candidate_text": None,
                    "raw_exact": False,
                    "reference_present": reference is not None,
                    "missing_reason": None if reference is not None else "absent",
                    "provenance": "records_fallback",
                },
            }
        )
        shadow = {
            "state": states[index],
            "candidate": pseudo[index],
            "runtime_route": "preexisting_exact_consensus" if pseudo[index] else None,
            "eligible_candidates": [],
            "global_gate_failures": [],
        }
        probe_findings.append(
            {
                "schema_version": 1,
                "kind": MODULE.PROBE_FINDING_KIND,
                "source": source.replace("\\", "/"),
                "reference_recipient": reference,
                "external_reference_present": reference is not None,
                "truth_used_for_analysis_only": True,
                "runtime_truth_lookup": False,
                "formal_delivery_gate": False,
                "shadow_candidate_truth_free": shadow,
                "paddle_teacher_consensus": dict(shadow),
            }
        )

    missing_path = tmp_path / "missing-audit.json"
    _write_json(
        missing_path,
        {
            "schema_version": 1,
            "kind": MODULE.MISSING_AUDIT_KIND,
            "ab_root": "D:/formal",
            "records": 6,
            "invariant_failure_records": 4,
            "recipient_missing_records": 4,
            "flagged_records": 4,
            "findings": missing_findings,
        },
    )
    probe = tmp_path / "consensus-probe-v4"
    _write_jsonl(probe / "findings.jsonl", probe_findings)
    diagnostic = tmp_path / "formal-diagnostic"
    _write_json(diagnostic / "summary.json", {"kind": "frozen-diagnostic"})
    _write_jsonl(
        diagnostic / "findings.jsonl",
        [{"source": source} for source in sources],
    )
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
            "findings_records": 4,
            "unique_sources": 4,
            "formal_contract": {
                "comparison_evaluation_mode": "formal",
                "comparison_records": 6,
                "failed_records": 4,
                "recipient_missing_only_records": 4,
                "recipient_missing_with_additional_failures_records": 0,
                "non_missing_invariant_failure_records": 0,
            },
            "external_reference": {
                "present_records": sum(reference is not None for reference in references),
                "missing_records": sum(reference is None for reference in references),
            },
            "paddle_teacher_consensus": {
                "external_truth": False,
                "truth_used_for_analysis_only": True,
                "formal_delivery_gate": False,
                "interpretation": "self_consistency_coverage_not_human_accuracy",
                "records": 2,
                "contract": {
                    "minimum_line_confidence": 0.80,
                    "minimum_recipient_detector_score": 0.68,
                    "requires_empty_geometry_reasons": True,
                    "requires_verified_alternative_envelope": True,
                    "requires_same_exact_line_in_independent_crops": 2,
                    "dominant_fallback_requires_multiple_eligible_candidates": True,
                    "dominant_fallback_requires_same_exact_line_in_all_crops": 3,
                    "dominant_fallback_requires_unique_all_crop_candidate": True,
                },
            },
            "remaining_failure_analysis": {
                "records": 2,
                "strict_candidate_records": 2,
            },
            "source_evidence": {
                "input_summary": MODULE._identity(
                    diagnostic / "summary.json", description="diagnostic summary"
                ),
                "input_findings": MODULE._identity(
                    diagnostic / "findings.jsonl", description="diagnostic findings"
                ),
            },
        },
    )
    return {
        "sources": sources,
        "missing": missing_path,
        "probe": probe,
        "diagnostic": diagnostic,
        "missing_findings": missing_findings,
        "probe_findings": probe_findings,
    }


def _run(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return MODULE.audit(
        missing_audit_json=fixture["missing"],
        probe_directory=fixture["probe"],
        output_directory=output,
    )


def test_atomic_analysis_separates_coverage_consistency_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    inputs_before = {
        path: path.read_bytes()
        for path in (
            fixture["missing"],
            *(fixture["probe"].iterdir()),
            *(fixture["diagnostic"].iterdir()),
        )
    }
    output = tmp_path / "fallback-audit"

    summary = _run(fixture, output)

    assert summary["analysis_only"] is True
    assert summary["formal_delivery_gate"] is False
    assert summary["production_fallback_authorized"] is False
    assert summary["decision"]["failed_conditions"] == [
        "strict_pseudo_truth_consistency_not_100_percent",
        "remaining_baseline_coverage_not_100_percent",
        "remaining_pollution_safety_not_proved",
    ]
    assert summary["source_set_closure"] == {
        "closed": True,
        "hybrid_missing_sources": 4,
        "v4_probe_sources": 4,
        "intersection_sources": 4,
    }
    coverage = summary["baseline_candidate_coverage"]
    assert coverage["all_missing_records"]["candidate_records"] == 3
    assert coverage["strict_pseudo_truth_records"]["candidate_records"] == 2
    assert coverage["remaining_records"]["candidate_records"] == 1
    consistency = summary["strict_pseudo_truth_consistency"]
    assert consistency["normalized_exact_records"] == 1
    assert consistency["raw_exact_records"] == 1
    assert consistency["normalized_mismatch_records"] == 1
    assert consistency["missing_records"] == 0
    pollution = summary["remaining_pollution_safety_evidence"]
    assert pollution["reference_present_records"] == 0
    assert pollution["reference_missing_records"] == 2
    assert pollution["satisfied"] is False
    assert summary["external_reference_presence"] == {
        "all_missing_records": 0,
        "strict_pseudo_truth_records": 0,
        "remaining_records": 0,
        "external_truth_is_runtime_input": False,
        "external_reference_text_copied_to_output": False,
    }
    assert all(
        len(examples) <= 3
        for block in (consistency["examples"], pollution["examples"])
        for examples in block.values()
    )
    findings = [
        json.loads(line)
        for line in (output / "findings.jsonl").read_text().splitlines()
    ]
    assert len(findings) == 4
    assert all(row["analysis_only"] for row in findings)
    assert all(row["runtime_truth_lookup"] is False for row in findings)
    assert inputs_before == {path: path.read_bytes() for path in inputs_before}
    assert not list(tmp_path.glob(".fallback-audit.*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(fixture, output)


def test_authorizes_only_when_every_strict_condition_is_proved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        baselines=["商户甲", "商户乙", "商户丙", "商户丁"],
        references=[None, None, "商户丙", "商户丁"],
    )

    summary = _run(fixture, tmp_path / "authorized")

    assert summary["strict_pseudo_truth_consistency"]["satisfied"] is True
    assert summary["baseline_candidate_coverage"]["remaining_records"][
        "satisfied"
    ] is True
    assert summary["remaining_pollution_safety_evidence"]["satisfied"] is True
    assert summary["production_fallback_authorized"] is True
    assert summary["decision"]["failed_conditions"] == []
    assert summary["formal_delivery_gate"] is False


def test_normalized_match_does_not_hide_raw_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        baselines=["商户甲", "Ａ 商户", "商户丙", "商户丁"],
        references=[None, None, "商户丙", "商户丁"],
    )
    fixture["probe_findings"][1]["shadow_candidate_truth_free"]["candidate"] = (
        "A 商户"
    )
    fixture["probe_findings"][1]["paddle_teacher_consensus"]["candidate"] = "A 商户"
    _write_jsonl(fixture["probe"] / "findings.jsonl", fixture["probe_findings"])

    summary = _run(fixture, tmp_path / "normalized-only")

    consistency = summary["strict_pseudo_truth_consistency"]
    assert consistency["normalized_exact_records"] == 2
    assert consistency["raw_exact_records"] == 1
    assert consistency["raw_mismatch_records"] == 1
    assert consistency["satisfied"] is False
    assert summary["production_fallback_authorized"] is False


def test_rejects_source_set_drift_and_duplicate_json_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["probe_findings"][0]["source"] = "D:/receipts/formal/other.jpg"
    _write_jsonl(fixture["probe"] / "findings.jsonl", fixture["probe_findings"])

    with pytest.raises(MODULE.AuditError, match="source set closure failed"):
        _run(fixture, tmp_path / "closure-failure")

    fixture = _fixture(tmp_path / "duplicates", monkeypatch)
    fixture["missing"].write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(MODULE.AuditError, match="duplicate JSON key"):
        _run(fixture, tmp_path / "duplicate-failure")


def test_input_hash_is_rechecked_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "must-not-publish"
    original_assert = MODULE._assert_identities_current
    calls = 0

    def mutate_after_second_check(identities: object) -> None:
        nonlocal calls
        calls += 1
        original_assert(identities)
        if calls == 2:
            with (fixture["probe"] / "findings.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")

    monkeypatch.setattr(MODULE, "_assert_identities_current", mutate_after_second_check)

    with pytest.raises(MODULE.AuditError, match="changed while the audit was reading"):
        _run(fixture, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-publish.*.tmp"))


def test_rejects_output_inside_probe_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    with pytest.raises(MODULE.AuditError, match="must not be inside"):
        _run(fixture, fixture["probe"] / "nested-output")


def test_source_declares_analysis_only_not_a_runtime_or_floor_change() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"formal_delivery_gate": False' in source
    assert '"runtime_truth_lookup": False' in source
    assert "production_fallback_authorized" in source
    assert "subprocess" not in source
    assert "onnxruntime" not in source.casefold()
    assert "paddleocr" not in source.casefold()
