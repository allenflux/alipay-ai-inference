from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-layout-shadow-selection.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_layout_shadow_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audit = tmp_path / "formal-missing-audit"
    images = tmp_path / "images"
    audit.mkdir()
    images.mkdir()
    sources: list[Path] = []
    for index in range(MODULE.TIME_MISSING_RECORDS):
        source = images / f"receipt-{index:03d}.jpg"
        source.write_bytes(f"image-{index}\n".encode())
        sources.append(source.resolve())

    summary = {
        "schema_version": 1,
        "kind": MODULE.AUDIT_SUMMARY_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "formal_required": True,
        "records": MODULE.FORMAL_RECORDS,
        "missing_by_field": {
            "time": {
                "records": MODULE.TIME_MISSING_RECORDS,
                "reference_present_records": 0,
                "reference_missing_records": MODULE.TIME_MISSING_RECORDS,
                "sources": [str(source) for source in sources],
            }
        },
        "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
    }
    _write_json(audit / "summary.json", summary)
    findings = [
        {
            "schema_version": 1,
            "kind": MODULE.AUDIT_FINDING_KIND,
            "source": str(source),
            "missing_fields": ["time"],
            "reference_present_by_field": {"time": False},
            "by_missing_field": {
                "time": {
                    "reference_present": False,
                    "reference_text": None,
                    "score_comparison": None,
                }
            },
        }
        for source in sources
    ]
    (audit / "findings.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in findings
        ),
        encoding="utf-8",
    )
    return audit, sources


def test_prepare_and_atomic_write_freeze_exact_time_339(tmp_path: Path) -> None:
    audit, sources = _fixture(tmp_path)
    selection, inputs_bytes, bindings = MODULE.prepare_selection(audit)

    assert selection["schema_version"] == 1
    assert selection["kind"] == MODULE.SELECTION_KIND
    assert selection["diagnostic_only"] is True
    assert selection["formal_delivery_gate"] is False
    assert selection["selection_field"] == "time"
    assert selection["records"] == 339
    assert selection["external_reference_present_records"] == 0
    assert selection["external_reference_missing_records"] == 339
    assert selection["formal_audit"]["summary"]["sha256"] == _sha256(
        audit / "summary.json"
    )
    assert selection["formal_audit"]["findings"]["sha256"] == _sha256(
        audit / "findings.jsonl"
    )
    assert len(selection["source_files"]) == 339
    assert selection["source_total_bytes"] == sum(path.stat().st_size for path in sources)
    assert not inputs_bytes.startswith(b"\xef\xbb\xbf")
    assert inputs_bytes.endswith(b"\n")
    assert inputs_bytes.decode().splitlines() == [str(path) for path in sources]

    output = tmp_path / "selection"
    MODULE.write_atomic(
        output,
        selection=selection,
        inputs_bytes=inputs_bytes,
        bindings=bindings,
    )
    assert sorted(path.name for path in output.iterdir()) == ["inputs.txt", "selection.json"]
    assert (output / "inputs.txt").read_bytes() == inputs_bytes
    published = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert published == selection
    assert published["input_list"] == {
        "relative_path": "inputs.txt",
        "sha256": _sha256(output / "inputs.txt"),
        "size_bytes": len(inputs_bytes),
        "records": 339,
        "encoding": "utf-8-no-bom",
        "terminal_newline": True,
    }
    assert "fields" not in published
    assert "candidate" not in published


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("kind",), "wrong", "schema/kind"),
        (("formal_required",), False, "--require-formal"),
        (("records",), 10015, "10016"),
        (("missing_by_field", "time", "records"), 338, "time missing count"),
        (
            ("missing_by_field", "time", "reference_missing_records"),
            338,
            "reference-missing count",
        ),
        (
            ("missing_by_field", "time", "reference_present_records"),
            1,
            "zero external references",
        ),
    ],
)
def test_prepare_rejects_nonformal_or_drifted_summary(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    audit, _ = _fixture(tmp_path)
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.SelectionError, match=message):
        MODULE.prepare_selection(audit)


def test_prepare_requires_exact_summary_findings_source_set(tmp_path: Path) -> None:
    audit, sources = _fixture(tmp_path)
    replacement = tmp_path / "replacement.jpg"
    replacement.write_bytes(b"replacement")
    findings_path = audit / "findings.jsonl"
    rows = [json.loads(line) for line in findings_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["source"] = str(replacement.resolve())
    findings_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.SelectionError, match="source sets differ"):
        MODULE.prepare_selection(audit)

    # Restore the finding, then duplicate a summary source. This must fail as
    # a duplicate rather than being silently reduced to a 338-set.
    rows[0]["source"] = str(sources[0])
    findings_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["missing_by_field"]["time"]["sources"][-1] = str(sources[0])
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.SelectionError, match="duplicate time source"):
        MODULE.prepare_selection(audit)


def test_prepare_rejects_relative_missing_and_reference_bearing_sources(tmp_path: Path) -> None:
    audit, sources = _fixture(tmp_path)
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["missing_by_field"]["time"]["sources"][0] = "relative.jpg"
    _write_json(summary_path, summary)
    with pytest.raises(MODULE.SelectionError, match="absolute path"):
        MODULE.prepare_selection(audit)

    summary["missing_by_field"]["time"]["sources"][0] = str(sources[0])
    _write_json(summary_path, summary)
    findings_path = audit / "findings.jsonl"
    rows = [json.loads(line) for line in findings_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["reference_present_by_field"]["time"] = True
    findings_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.SelectionError, match="external reference"):
        MODULE.prepare_selection(audit)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"schema_version":1,"schema_version":1}\n', "duplicate JSON key"),
        ('{"schema_version":NaN}\n', "non-standard JSON constant"),
        ('[]\n', "must be one JSON object"),
    ],
)
def test_prepare_rejects_non_strict_summary_json(
    tmp_path: Path, payload: str, message: str
) -> None:
    audit, _ = _fixture(tmp_path)
    (audit / "summary.json").write_text(payload, encoding="utf-8")
    with pytest.raises(MODULE.SelectionError, match=message):
        MODULE.prepare_selection(audit)


def test_prepare_rejects_blank_or_duplicate_finding_rows(tmp_path: Path) -> None:
    audit, _ = _fixture(tmp_path)
    findings = audit / "findings.jsonl"
    lines = findings.read_text(encoding="utf-8").splitlines()
    findings.write_text(lines[0] + "\n\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.SelectionError, match="blank line"):
        MODULE.prepare_selection(audit)

    findings.write_text("\n".join([lines[0], lines[0], *lines[2:]]) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.SelectionError, match="duplicate finding source"):
        MODULE.prepare_selection(audit)


def test_atomic_write_refuses_overwrite_and_detects_audit_toctou(tmp_path: Path) -> None:
    audit, _ = _fixture(tmp_path)
    selection, inputs_bytes, bindings = MODULE.prepare_selection(audit)
    output = tmp_path / "selection"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_atomic(
            output,
            selection=selection,
            inputs_bytes=inputs_bytes,
            bindings=bindings,
        )
    assert marker.read_text(encoding="utf-8") == "mine"

    marker.unlink()
    output.rmdir()
    (audit / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MODULE.SelectionError, match="summary changed"):
        MODULE.write_atomic(
            output,
            selection=selection,
            inputs_bytes=inputs_bytes,
            bindings=bindings,
        )
    assert not output.exists()


def test_atomic_write_detects_source_toctou_and_input_payload_drift(tmp_path: Path) -> None:
    audit, sources = _fixture(tmp_path)
    selection, inputs_bytes, bindings = MODULE.prepare_selection(audit)
    output = tmp_path / "selection"
    sources[0].write_bytes(b"mutated")
    with pytest.raises(MODULE.SelectionError, match="source image changed"):
        MODULE.write_atomic(
            output,
            selection=selection,
            inputs_bytes=inputs_bytes,
            bindings=bindings,
        )
    assert not output.exists()

    # Rebuild valid bindings, then corrupt the caller-supplied list bytes.
    audit, _ = _fixture(tmp_path / "second")
    selection, inputs_bytes, bindings = MODULE.prepare_selection(audit)
    with pytest.raises(MODULE.SelectionError, match="input-list identity"):
        MODULE.write_atomic(
            output,
            selection=selection,
            inputs_bytes=inputs_bytes + b"extra\n",
            bindings=bindings,
        )
    assert not output.exists()


def test_atomic_write_refuses_broken_symlink_when_supported(tmp_path: Path) -> None:
    audit, _ = _fixture(tmp_path)
    selection, inputs_bytes, bindings = MODULE.prepare_selection(audit)
    output = tmp_path / "selection-link"
    try:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_atomic(
            output,
            selection=selection,
            inputs_bytes=inputs_bytes,
            bindings=bindings,
        )
    assert output.is_symlink()


def test_main_returns_two_on_invalid_audit_and_prints_bound_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit, _ = _fixture(tmp_path)
    output = tmp_path / "selection"
    assert MODULE.main(
        ["--audit-directory", str(audit), "--output-directory", str(output)]
    ) == 0
    printed = capsys.readouterr().out
    assert "records=339" in printed
    assert _sha256(output / "inputs.txt") in printed

    assert MODULE.main(
        ["--audit-directory", str(audit), "--output-directory", str(output)]
    ) == 2
    assert "refusing to overwrite" in capsys.readouterr().out
