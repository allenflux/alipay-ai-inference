from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-hybrid-missing-audit.py"
)
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_hybrid_missing_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _result(source: Path, *, hybrid: bool, missing: bool = False) -> dict[str, object]:
    recipient: dict[str, object] = {
        "state": "unreadable" if missing else "review",
        "candidate": None if missing else "商户甲",
        "ctc_candidate": None if missing else "商户甲",
        "detector_score": 0.93,
    }
    if hybrid:
        recipient.update(
            {
                "hybrid_ocr_route": "none" if missing else "primary",
                "hybrid_ocr_failure_reason": (
                    "anchored_or_alternative_parse_failed;first=lines=[0:0.61:商户]"
                    if missing
                    else None
                ),
                "hybrid_ocr_first_raw": "商户",
                "hybrid_ocr_first_line_count": 1,
                "hybrid_ocr_first_crop_width": 600,
                "hybrid_ocr_first_crop_height": 90,
                "hybrid_ocr_retry_raw": "收款方 商户",
                "hybrid_ocr_retry_line_count": 1,
                "hybrid_ocr_retry_crop_width": 720,
                "hybrid_ocr_retry_crop_height": 90,
            }
        )
    return {
        "source": str(source),
        "fields": {"recipient": recipient},
        "detections": [
            {"label": "recipient_field", "score": 0.93, "bbox_image": [1, 2, 3, 4]}
        ],
        "model_contracts": {
            "unified_ocr_model_sha256": "a" * 64,
            **({"ocr_bundle_contract_sha256": "b" * 64} if hybrid else {}),
        },
    }


def _write_fixture(root: Path) -> tuple[list[Path], dict[str, Path], dict[str, Path]]:
    sources = [root / "inputs" / f"{name}.jpg" for name in ("good", "bad", "missing")]
    comparisons = [
        {"source": str(sources[0]), "recipient_candidate": "商户甲", "invariant": True, "failures": []},
        {
            "source": str(sources[1]),
            "recipient_candidate": None,
            "invariant": False,
            "failures": ["hybrid recipient candidate missing"],
        },
        {"source": str(sources[2]), "recipient_candidate": None, "invariant": True, "failures": []},
    ]
    comparison_path = root / "comparison" / "comparisons.jsonl"
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparisons),
        encoding="utf-8",
    )
    baseline_paths: dict[str, Path] = {}
    hybrid_paths: dict[str, Path] = {}
    for run_name, hybrid in (("baseline-v13", False), ("hybrid-recipient", True)):
        run = root / run_name
        manifest = []
        for index, source in enumerate(sources):
            result_path = run / "results" / f"{index}.json"
            _write_json(result_path, _result(source, hybrid=hybrid, missing=index > 0))
            manifest.append(
                {"source": str(source), "result": str(result_path), "status": "written"}
            )
            (hybrid_paths if hybrid else baseline_paths)[str(source)] = result_path
        _write_json(run / "inference_manifest.json", manifest)
    return sources, baseline_paths, hybrid_paths


def test_audit_outputs_only_failed_or_missing_rows_with_ppocr_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ab"
    sources, _, _ = _write_fixture(root)
    before = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    payload = MODULE.audit(root)

    assert payload["records"] == 3
    assert payload["invariant_failure_records"] == 1
    assert payload["recipient_missing_records"] == 2
    assert payload["flagged_records"] == 2
    assert [finding["source"] for finding in payload["findings"]] == [
        str(sources[1]),
        str(sources[2]),
    ]
    failure = payload["findings"][0]
    assert failure["baseline_recipient_field"]["candidate"] is None
    assert failure["hybrid_recipient_detection"]["score"] == 0.93
    assert failure["hybrid_model_contracts"]["ocr_bundle_contract_sha256"] == "b" * 64
    assert failure["hybrid_ppocr_evidence"] == {
        "failure_reason": "anchored_or_alternative_parse_failed;first=lines=[0:0.61:商户]",
        "first_crop_height": 90,
        "first_crop_width": 600,
        "first_line_count": 1,
        "first_raw": "商户",
        "retry_crop_height": 90,
        "retry_crop_width": 720,
        "retry_line_count": 1,
        "retry_raw": "收款方 商户",
        "route": "none",
    }
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_comparison", "duplicate comparison source"),
        ("duplicate_manifest", "duplicate hybrid manifest source"),
        ("source_mismatch", "manifest/result source mismatch"),
        ("path_escape", "result path escapes run root"),
        ("missing_result", "result file is missing"),
    ],
)
def test_audit_rejects_unbound_or_escaping_results(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / "ab"
    sources, _, hybrid_paths = _write_fixture(root)
    comparison_path = root / "comparison" / "comparisons.jsonl"
    manifest_path = root / "hybrid-recipient" / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "duplicate_comparison":
        first = comparison_path.read_text(encoding="utf-8").splitlines()[0]
        comparison_path.write_text(first + "\n" + first + "\n", encoding="utf-8")
    elif mutation == "duplicate_manifest":
        manifest.append(dict(manifest[0]))
        _write_json(manifest_path, manifest)
    elif mutation == "source_mismatch":
        payload = json.loads(hybrid_paths[str(sources[0])].read_text(encoding="utf-8"))
        payload["source"] = str(tmp_path / "different.jpg")
        _write_json(hybrid_paths[str(sources[0])], payload)
    elif mutation == "path_escape":
        outside = root / "outside.json"
        _write_json(outside, _result(sources[0], hybrid=True))
        manifest[0]["result"] = str(outside)
        _write_json(manifest_path, manifest)
    else:
        manifest[0]["result"] = str(root / "hybrid-recipient" / "missing.json")
        _write_json(manifest_path, manifest)

    with pytest.raises(MODULE.AuditError, match=message):
        MODULE.audit(root)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"source":"a","source":"b"}\n', "duplicate JSON key"),
        ('{"source":"a","score":NaN}\n', "non-standard JSON constant"),
        ("\n", "blank line"),
    ],
)
def test_audit_rejects_non_strict_comparison_jsonl(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    root = tmp_path / "ab"
    _write_fixture(root)
    (root / "comparison" / "comparisons.jsonl").write_text(contents, encoding="utf-8")

    with pytest.raises(MODULE.AuditError, match=message):
        MODULE.audit(root)


def test_main_emits_exactly_one_json_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "ab"
    _write_fixture(root)

    assert MODULE.main(["--root", str(root)]) == 0

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output)["kind"] == "receipt_mlnet_hybrid_missing_audit_v1"


def test_main_text_format_is_complete_and_terminal_width_bounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ab"
    sources, _, _ = _write_fixture(root)

    assert MODULE.main(["--root", str(root), "--format", "text"]) == 0

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines[0] == "Receipt ML.NET hybrid missing audit"
    assert "Counts:" in lines
    assert "  records: 3" in lines
    assert "  invariant failures: 1" in lines
    assert "  recipient missing: 2" in lines
    assert "  flagged: 2" in lines
    assert "Finding 1/2" in lines
    assert '  basename: "bad.jpg"' in lines
    collapsed = "".join(line.strip() for line in lines)
    assert str(sources[1]) in collapsed
    assert "hybrid recipient candidate missing" in output
    assert "  baseline recipient:" in lines
    assert "  hybrid recipient:" in lines
    assert "    candidate: null" in lines
    assert "    raw: <missing>" in lines
    assert "    value: <missing>" in lines
    assert '    state: "unreadable"' in lines
    assert "  baseline recipient detection:" in lines
    assert "    bbox: [1,2,3,4]" in lines
    assert "    score: 0.93" in lines
    assert "  hybrid PP-OCR evidence:" in lines
    assert '    route: "none"' in lines
    assert "anchored_or_alternative_parse_failed" in output
    assert '      raw: "商户"' in lines
    assert "      line_count: 1" in lines
    assert '      crop_wxh: "600x90"' in lines
    assert '      crop_wxh: "720x90"' in lines
    assert max(map(len, lines)) <= 96
