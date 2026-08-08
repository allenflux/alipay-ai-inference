from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-hybrid-pilot-diagnose.py"
)
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_hybrid_pilot_diagnose", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _result(
    source: Path,
    *,
    candidate: str | None,
    route: str,
    failure_reason: str | None,
) -> dict[str, object]:
    return {
        "source": str(source),
        "geometry": {"rectified_size": {"width": 1000, "height": 2000}},
        "fields": {
            "amount": {"candidate": "100.00"},
            "recipient": {
                "candidate": candidate,
                "hybrid_ocr_route": route,
                "hybrid_ocr_failure_reason": failure_reason,
                "hybrid_ocr_first_raw": "商户",
                "hybrid_ocr_first_line_count": 1,
                "hybrid_ocr_retry_raw": "商户",
                "hybrid_ocr_retry_line_count": 1,
            },
        },
        "detections": [
            {
                "label": "amount",
                "score": 0.95,
                "bbox_image": [0, 200, 1000, 300],
            },
            {
                "label": "recipient_field",
                "score": 0.93,
                "bbox_image": [0, 400, 1000, 500],
            },
            {
                "label": "payment_method_field",
                "score": 0.96,
                "bbox_image": [0, 600, 1000, 700],
            },
        ],
    }


def _write_comparison_summary(
    comparison: Path,
    hybrid: Path,
    rows: list[dict[str, object]],
    *,
    mode: str,
) -> None:
    manifest_path = (hybrid / "inference_manifest.json").resolve()
    source_keys = [MODULE._source_key(row["source"]) for row in rows]
    manifest_identity = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "size_bytes": manifest_path.stat().st_size,
        "records": len(rows),
        "normalized_source_set_sha256": MODULE._normalized_source_set_sha256(
            source_keys
        ),
    }
    _write_json(
        comparison / "summary.json",
        {
            "schema_version": MODULE.COMPARISON_SUMMARY_SCHEMA_VERSION,
            "kind": MODULE.COMPARISON_SUMMARY_KIND,
            "evaluation_mode": mode,
            "records": len(rows),
            "invariant_records": sum(row["invariant"] is True for row in rows),
            "recipient_candidate_coverage": sum(
                isinstance(row["recipient_candidate"], str)
                and bool(row["recipient_candidate"])
                for row in rows
            )
            / len(rows),
            "run_manifests": {"hybrid": manifest_identity},
        },
    )


def _fixture(root: Path, *, mode: str = "pilot") -> tuple[Path, Path]:
    comparison = root / "comparison"
    hybrid = root / "hybrid-recipient"
    sources = [root / "inputs" / f"{name}.jpg" for name in ("good", "missing", "changed")]
    rows = [
        {
            "source": str(sources[0]),
            "recipient_candidate": "商户甲",
            "invariant": True,
            "failures": [],
        },
        {
            "source": str(sources[1]),
            "recipient_candidate": None,
            "invariant": False,
            "failures": ["hybrid recipient candidate missing"],
        },
        {
            "source": str(sources[2]),
            "recipient_candidate": "商户乙",
            "invariant": False,
            "failures": ["fields.amount changed"],
        },
    ]
    comparison.mkdir(parents=True)
    (comparison / "comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    results = [
        _result(sources[0], candidate="商户甲", route="primary", failure_reason=None),
        _result(
            sources[1],
            candidate=None,
            route="none",
            failure_reason=(
                "anchored_or_alternative_parse_failed;"
                "alternative_envelope=False"
            ),
        ),
        _result(sources[2], candidate="商户乙", route="primary", failure_reason=None),
    ]
    manifest = []
    for index, (source, result) in enumerate(zip(sources, results, strict=True)):
        result_path = hybrid / "results" / f"{index}.json"
        _write_json(result_path, result)
        manifest.append(
            {"source": str(source), "result": str(result_path), "status": "written"}
        )
    _write_json(hybrid / "inference_manifest.json", manifest)
    _write_comparison_summary(comparison, hybrid, rows, mode=mode)
    return comparison, hybrid


def test_atomic_failure_summary_covers_failure_route_and_blocker_without_rerun(
    tmp_path: Path,
) -> None:
    comparison, hybrid = _fixture(tmp_path / "ab")
    before = {
        path: path.read_bytes()
        for root in (comparison, hybrid)
        for path in root.rglob("*")
        if path.is_file()
    }
    diagnostics = MODULE.diagnose(comparison, hybrid)
    summary = MODULE.summarize(
        diagnostics,
        comparison=comparison,
        hybrid=hybrid,
    )
    output = tmp_path / "diagnostic"

    MODULE.write_diagnostic_atomic(output, summary=summary, diagnostics=diagnostics)

    written_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["read_only_existing_results"] is True
    assert written_summary["ocr_rerun"] is False
    assert written_summary["comparison_records"] == 3
    assert written_summary["invariant_failure_records"] == 2
    assert written_summary["recipient_missing_records"] == 1
    assert written_summary["non_missing_invariant_failure_records"] == 1
    assert written_summary["recipient_missing_only_records"] == 1
    assert written_summary["recipient_missing_with_additional_failures_records"] == 0
    assert written_summary["failed_records"] == 2
    assert written_summary["comparison_evaluation_mode"] == "pilot"
    for name in ("comparison_summary", "comparisons", "hybrid_manifest"):
        identity = written_summary["source_evidence"][name]
        assert Path(identity["path"]).is_file()
        assert len(identity["sha256"]) == 64
        assert identity["size_bytes"] > 0
    assert written_summary["by_comparator_failure"] == [
        {"name": "fields.amount changed", "records": 1},
        {"name": "hybrid recipient candidate missing", "records": 1},
    ]
    assert written_summary["by_ppocr_route"] == [
        {"name": "none", "records": 1},
        {"name": "primary", "records": 1},
    ]
    assert written_summary["by_primary_blocker"] == [
        {"name": "anchored_or_alternative_parse_failed", "records": 1},
        {"name": "non_recipient_invariant_change", "records": 1},
    ]
    findings = [
        json.loads(line)
        for line in (output / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(findings) == 2
    assert {finding["kind"] for finding in findings} == {
        MODULE.DIAGNOSTIC_FINDING_KIND
    }
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not list(tmp_path.glob(".diagnostic.*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_diagnostic_atomic(
            output, summary=summary, diagnostics=diagnostics
        )


def test_main_with_output_directory_prints_only_compact_publication_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    comparison, hybrid = _fixture(tmp_path / "ab")
    output = tmp_path / "diagnostic"

    assert MODULE.main(
        [
            "--comparison",
            str(comparison),
            "--hybrid",
            str(hybrid),
            "--output-directory",
            str(output),
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert stdout.count("\n") == 1
    publication = json.loads(stdout)
    assert publication == {
        "kind": MODULE.DIAGNOSTIC_SUMMARY_KIND,
        "failed_records": 2,
        "output_directory": output.resolve().as_posix(),
    }


def test_summary_separates_missing_only_from_missing_with_additional_failures(
    tmp_path: Path,
) -> None:
    comparison, hybrid = _fixture(tmp_path / "ab")
    comparison_path = comparison / "comparisons.jsonl"
    rows = [
        json.loads(line)
        for line in comparison_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["failures"].append("fields.amount changed")
    comparison_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    diagnostics = MODULE.diagnose(comparison, hybrid)
    summary = MODULE.summarize(
        diagnostics,
        comparison=comparison,
        hybrid=hybrid,
    )

    assert summary["comparison_records"] == 3
    assert summary["invariant_failure_records"] == 2
    assert summary["recipient_missing_records"] == 1
    assert summary["non_missing_invariant_failure_records"] == 1
    assert summary["recipient_missing_only_records"] == 0
    assert summary["recipient_missing_with_additional_failures_records"] == 1


def test_summary_can_prove_all_204_failures_are_missing_only(tmp_path: Path) -> None:
    comparison = tmp_path / "ab" / "comparison"
    hybrid = tmp_path / "ab" / "hybrid-recipient"
    comparison.mkdir(parents=True)
    rows = []
    manifest = []
    for index in range(204):
        source = tmp_path / "inputs" / f"missing-{index:03d}.jpg"
        rows.append(
            {
                "source": str(source),
                "recipient_candidate": None,
                "invariant": False,
                "failures": [MODULE.RECIPIENT_MISSING_FAILURE],
            }
        )
        result_path = hybrid / "results" / f"{index:03d}.json"
        _write_json(
            result_path,
            _result(
                source,
                candidate=None,
                route="none",
                failure_reason=(
                    "anchored_or_alternative_parse_failed;"
                    "alternative_envelope=False"
                ),
            ),
        )
        manifest.append(
            {"source": str(source), "result": str(result_path), "status": "written"}
        )
    comparison.joinpath("comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(hybrid / "inference_manifest.json", manifest)
    _write_comparison_summary(comparison, hybrid, rows, mode="pilot")

    diagnostics = MODULE.diagnose(comparison, hybrid)
    summary = MODULE.summarize(
        diagnostics,
        comparison=comparison,
        hybrid=hybrid,
    )

    assert summary["comparison_records"] == 204
    assert summary["invariant_failure_records"] == 204
    assert summary["recipient_missing_records"] == 204
    assert summary["non_missing_invariant_failure_records"] == 0
    assert summary["recipient_missing_only_records"] == 204
    assert summary["recipient_missing_with_additional_failures_records"] == 0
    assert summary["failed_records"] == 204


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("manifest_not_array", "non-empty array"),
        ("incomplete_status", "disallowed status"),
        ("duplicate_source", "duplicate hybrid manifest source"),
        ("escaped_result", "escapes hybrid root"),
        ("result_source_mismatch", "manifest/result source mismatch"),
    ],
)
def test_diagnose_rejects_unbound_hybrid_manifest_or_result(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    comparison, hybrid = _fixture(tmp_path / case / "ab")
    manifest_path = hybrid / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "manifest_not_array":
        _write_json(manifest_path, {"records": manifest})
    elif case == "incomplete_status":
        manifest[0]["status"] = "error"
        _write_json(manifest_path, manifest)
    elif case == "duplicate_source":
        manifest[1]["source"] = manifest[0]["source"]
        _write_json(manifest_path, manifest)
    elif case == "escaped_result":
        outside = hybrid.parent / "outside.json"
        _write_json(
            outside,
            _result(
                Path(manifest[0]["source"]),
                candidate="商户甲",
                route="primary",
                failure_reason=None,
            ),
        )
        manifest[0]["result"] = str(outside)
        _write_json(manifest_path, manifest)
    elif case == "result_source_mismatch":
        result_path = Path(manifest[0]["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["source"] = str(tmp_path / "different.jpg")
        _write_json(result_path, result)
    else:  # pragma: no cover - keeps the mutation table exhaustive.
        raise AssertionError(case)

    with pytest.raises(MODULE.DiagnosticError, match=message):
        MODULE.diagnose(comparison, hybrid)


def test_diagnose_rejects_comparison_candidate_that_differs_from_bound_result(
    tmp_path: Path,
) -> None:
    comparison, hybrid = _fixture(tmp_path / "candidate-mismatch" / "ab")
    result_path = hybrid / "results" / "1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["fields"]["recipient"]["candidate"] = "篡改候选"
    _write_json(result_path, result)

    with pytest.raises(
        MODULE.DiagnosticError,
        match=r"comparison\[1\] recipient_candidate differs from hybrid result",
    ):
        MODULE.diagnose(comparison, hybrid)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "schema_version"),
        ("kind", "kind"),
        ("mode", "evaluation_mode"),
        ("records", "records differs"),
        ("invariant", "invariant_records differs"),
        ("coverage", "recipient_candidate_coverage differs"),
        ("manifest_path", "path does not point"),
        ("manifest_sha256", "sha256 differs"),
        ("manifest_size", "size_bytes differs"),
        ("manifest_records", "records differs from --hybrid"),
        ("manifest_sources", "normalized_source_set_sha256 differs"),
    ],
)
def test_diagnose_rejects_mismatched_comparison_summary_or_manifest_identity(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    comparison, hybrid = _fixture(tmp_path / case / "ab")
    summary_path = comparison / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if case == "schema":
        summary["schema_version"] = 1
    elif case == "kind":
        summary["kind"] = "wrong"
    elif case == "mode":
        summary["evaluation_mode"] = "wrong"
    elif case == "records":
        summary["records"] += 1
    elif case == "invariant":
        summary["invariant_records"] += 1
    elif case == "coverage":
        summary["recipient_candidate_coverage"] = 0.5
    elif case == "manifest_path":
        summary["run_manifests"]["hybrid"]["path"] = str(
            (hybrid / "results" / "0.json").resolve()
        )
    elif case == "manifest_sha256":
        summary["run_manifests"]["hybrid"]["sha256"] = "0" * 64
    elif case == "manifest_size":
        summary["run_manifests"]["hybrid"]["size_bytes"] += 1
    elif case == "manifest_records":
        summary["run_manifests"]["hybrid"]["records"] += 1
    elif case == "manifest_sources":
        summary["run_manifests"]["hybrid"][
            "normalized_source_set_sha256"
        ] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(case)
    _write_json(summary_path, summary)

    with pytest.raises(MODULE.DiagnosticError, match=message):
        MODULE.diagnose(comparison, hybrid)


def test_require_formal_preserves_pilot_default_and_enforces_formal_10016(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_comparison, pilot_hybrid = _fixture(tmp_path / "pilot" / "ab")
    assert len(MODULE.diagnose(pilot_comparison, pilot_hybrid)) == 2
    with pytest.raises(MODULE.DiagnosticError, match="requires a formal"):
        MODULE.diagnose(
            pilot_comparison,
            pilot_hybrid,
            require_formal=True,
        )

    formal_comparison, formal_hybrid = _fixture(
        tmp_path / "formal" / "ab", mode="formal"
    )
    assert len(MODULE.diagnose(formal_comparison, formal_hybrid)) == 2
    with pytest.raises(MODULE.DiagnosticError, match="exactly 10016"):
        MODULE.diagnose(
            formal_comparison,
            formal_hybrid,
            require_formal=True,
        )
    monkeypatch.setattr(MODULE, "FORMAL_EXPECTED_RECORDS", 3)
    assert len(
        MODULE.diagnose(
            formal_comparison,
            formal_hybrid,
            require_formal=True,
        )
    ) == 2


def test_formal_rejects_skipped_existing_while_pilot_default_remains_compatible(
    tmp_path: Path,
) -> None:
    pilot_comparison, pilot_hybrid = _fixture(tmp_path / "pilot-skip" / "ab")
    pilot_manifest_path = pilot_hybrid / "inference_manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    pilot_manifest[0]["status"] = "skipped_existing"
    _write_json(pilot_manifest_path, pilot_manifest)
    pilot_rows = [
        json.loads(line)
        for line in (pilot_comparison / "comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    _write_comparison_summary(
        pilot_comparison, pilot_hybrid, pilot_rows, mode="pilot"
    )
    assert len(MODULE.diagnose(pilot_comparison, pilot_hybrid)) == 2

    formal_comparison, formal_hybrid = _fixture(
        tmp_path / "formal-skip" / "ab", mode="formal"
    )
    formal_manifest_path = formal_hybrid / "inference_manifest.json"
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    formal_manifest[0]["status"] = "skipped_existing"
    _write_json(formal_manifest_path, formal_manifest)
    formal_rows = [
        json.loads(line)
        for line in (formal_comparison / "comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    _write_comparison_summary(
        formal_comparison, formal_hybrid, formal_rows, mode="formal"
    )
    with pytest.raises(MODULE.DiagnosticError, match="disallowed status"):
        MODULE.diagnose(formal_comparison, formal_hybrid)
