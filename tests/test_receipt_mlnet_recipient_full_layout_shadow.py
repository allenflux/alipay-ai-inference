from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-recipient-full-layout-shadow.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_recipient_full_layout_shadow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FILTER_SCRIPT = ROOT / "scripts" / "receipt-mlnet-hybrid-failure-truth-probe.py"
FILTER_SPEC = importlib.util.spec_from_file_location("recipient_truth_filter_for_test", FILTER_SCRIPT)
assert FILTER_SPEC is not None and FILTER_SPEC.loader is not None
FILTER = importlib.util.module_from_spec(FILTER_SPEC)
FILTER_SPEC.loader.exec_module(FILTER)


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _quad(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _line(
    index: int,
    text: str,
    *,
    confidence: float = 0.95,
    box: tuple[float, float, float, float] = (0.1, 0.4, 0.3, 0.45),
    drop_score: float = 0.5,
) -> dict[str, object]:
    return {
        "index": index,
        "text": text,
        "confidence": confidence,
        "passes_drop_score": confidence >= drop_score,
        "quad_rectified_normalized": _quad(*box),
    }


def test_fixed_scope_and_source_contract_do_not_touch_production() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert MODULE.TARGET_RECORDS == 61
    assert MODULE.CONTROL_RECORDS == 278
    assert MODULE.EXPECTED_RECORDS == 339
    assert '"candidate_write_enabled": False' in source
    assert '"formal_delivery_gate": False' in source
    assert '"production_output_changed": False' in source
    assert "receipt-mlnet-recipient-derived-crop-shadow.py" in source
    assert "receipt-mlnet-formal-missing-fields-audit.py" in source
    assert "receipt-mlnet-layout-shadow-evidence.py" in source
    assert "dotnet/" not in source


def test_stratified_allocation_is_proportional_deterministic_and_spread() -> None:
    allocation = MODULE._allocate_strata(
        {"existing_exact": 900, "existing_wrong": 100}, 278
    )
    assert allocation == {"existing_exact": 250, "existing_wrong": 28}
    assert MODULE._allocate_strata(
        {"existing_exact": 999, "existing_wrong": 1}, 278
    ) == {"existing_exact": 277, "existing_wrong": 1}
    rows = [{"index": index} for index in range(100)]
    selected = MODULE._evenly_spread(rows, 4)
    assert [row["index"] for row in selected] == [12, 37, 62, 87]


def test_exact_label_rhs_and_unique_same_row_right_neighbor() -> None:
    rhs = MODULE._recipient_shadow(
        [_line(0, "收款方：司源(**源)")], drop_score=0.5, filter_module=FILTER
    )
    assert rhs["state"] == "shadow_candidate"
    assert rhs["shadow_candidate"] == "司源(**源)"
    assert rhs["shadow_route"] == "full_layout_label_rhs_shadow"

    row = MODULE._recipient_shadow(
        [
            _line(0, "收款账户", box=(0.08, 0.40, 0.24, 0.45)),
            _line(1, "张三", box=(0.58, 0.405, 0.72, 0.452)),
            _line(2, "付款方式", box=(0.08, 0.52, 0.24, 0.57)),
        ],
        drop_score=0.5,
        filter_module=FILTER,
    )
    assert row["state"] == "shadow_candidate"
    assert row["shadow_candidate"] == "张三"
    assert row["shadow_route"] == "full_layout_label_right_neighbor_shadow"
    assert row["evidence"][0]["candidate_line_index"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "收款方司源",  # no exact label/value boundary
        "付款方：司源",  # unsupported label
        "收款方：200.00",  # amount
        "收款人：05:49",  # time
        "收款账户：邮储银行卡",  # negative/payment token
    ],
)
def test_label_and_value_filters_fail_closed(text: str) -> None:
    result = MODULE._recipient_shadow(
        [_line(0, text)], drop_score=0.5, filter_module=FILTER
    )
    assert result["state"] == "unresolved"
    assert result["shadow_candidate"] is None


def test_low_confidence_and_multiple_right_values_do_not_emit() -> None:
    low = MODULE._recipient_shadow(
        [_line(0, "收款方：张三", confidence=0.79)],
        drop_score=0.5,
        filter_module=FILTER,
    )
    assert low["state"] == "unresolved"

    ambiguous = MODULE._recipient_shadow(
        [
            _line(0, "收款人", box=(0.08, 0.40, 0.24, 0.45)),
            _line(1, "张三", box=(0.50, 0.40, 0.62, 0.45)),
            _line(2, "李四", box=(0.70, 0.40, 0.82, 0.45)),
        ],
        drop_score=0.5,
        filter_module=FILTER,
    )
    assert ambiguous["state"] == "ambiguous"
    assert ambiguous["shadow_candidate"] is None
    assert ambiguous["ambiguous_anchor_indices"] == [0]


def test_prepare_atomically_freezes_targets_without_truth_and_stratified_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "TARGET_RECORDS", 2)
    monkeypatch.setattr(MODULE, "CONTROL_RECORDS", 4)
    monkeypatch.setattr(MODULE, "EXPECTED_RECORDS", 6)
    images: list[Path] = []
    for index in range(8):
        image = tmp_path / f"receipt-{index}.jpg"
        image.write_bytes(f"image-{index}".encode())
        images.append(image.resolve())

    class FormalStub:
        @staticmethod
        def _source_key(value: object) -> str:
            return str(Path(str(value)).resolve()).casefold()

        @staticmethod
        def _candidate(result: dict[str, object], key: str) -> str | None:
            return result.get("candidate")  # type: ignore[return-value]

    formal_stub = FormalStub()
    monkeypatch.setattr(MODULE, "_load_module", lambda *_: formal_stub)
    target_rows = [
        {"source": str(images[index]), "source_image": _identity(images[index]), "plan_id": f"plan-{index}"}
        for index in range(2)
    ]
    monkeypatch.setattr(
        MODULE,
        "_load_derived_closure",
        lambda *_, **__: {"targets": target_rows, "identities": {}},
    )
    keys = [formal_stub._source_key(image) for image in images]
    references = {key: {} for key in keys}
    hybrid: dict[str, dict[str, object]] = {key: {"candidate": None} for key in keys}
    values = [
        ("甲", "甲"),
        ("乙", "错乙"),
        ("丙", "丙"),
        ("丁", "错丁"),
        ("戊", "戊"),
        ("己", "错己"),
    ]
    for offset, (reference, candidate) in enumerate(values, start=2):
        references[keys[offset]]["recipient_field"] = reference
        hybrid[keys[offset]]["candidate"] = candidate
    formal = {
        "input_sources": [str(image) for image in images],
        "input_by_key": dict(zip(keys, map(str, images), strict=True)),
        "references": references,
        "hybrid": hybrid,
        "missing_recipient_keys": set(keys[:2]),
        "identities": {},
    }
    monkeypatch.setattr(MODULE, "_load_formal_closure", lambda *_, **__: formal)
    monkeypatch.setattr(MODULE, "_assert_formal_bindings_current", lambda *_: None)

    output = tmp_path / "selection"
    MODULE.prepare(
        plan_directory=tmp_path,
        derived_evaluation_directory=tmp_path,
        formal_audit_directory=tmp_path,
        truth_probe_directory=tmp_path,
        output_directory=output,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "inputs.txt",
        "selection.jsonl",
        "summary.json",
    ]
    rows = [json.loads(line) for line in (output / "selection.jsonl").read_text().splitlines()]
    # Each exactness stratum is spread over its complete canonical range, then
    # the combined selection is restored to canonical formal order.
    assert [row["source"] for row in rows] == [
        str(images[index]) for index in (0, 1, 2, 3, 6, 7)
    ]
    targets = [row for row in rows if row["cohort"] == "target_unresolved"]
    controls = [row for row in rows if row["cohort"] == "reference_control"]
    assert len(targets) == 2 and len(controls) == 4
    assert all("control_evidence" not in row for row in targets)
    assert all("external_reference" not in json.dumps(row) for row in targets)
    assert {row["control_evidence"]["stratum"] for row in controls} == {
        "existing_exact",
        "existing_wrong",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.prepare(
            plan_directory=tmp_path,
            derived_evaluation_directory=tmp_path,
            formal_audit_directory=tmp_path,
            truth_probe_directory=tmp_path,
            output_directory=output,
        )


def test_evaluate_reports_target_shadow_only_and_control_regression_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "TARGET_RECORDS", 2)
    monkeypatch.setattr(MODULE, "CONTROL_RECORDS", 4)
    monkeypatch.setattr(MODULE, "EXPECTED_RECORDS", 6)
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image")
    source_identity = _identity(source)
    frozen = {}
    for name in ("selection-summary.json", "selection.jsonl", "inputs.txt", "layout-summary.json", "layout.jsonl"):
        path = tmp_path / name
        path.write_text(name)
        frozen[name] = _identity(path)
    selection_rows: list[dict[str, object]] = []
    for index in range(6):
        row: dict[str, object] = {
            "source": f"source-{index}.jpg",
            "cohort": "target_unresolved" if index < 2 else "reference_control",
        }
        if index < 2:
            row["target_evidence"] = {"derived_plan_id": f"plan-{index}", "derived_state": "unresolved"}
        else:
            references = ["张三", "李四", "王五", "赵六"]
            existing = ["张三", "李四", "错误", "错误"]
            row["control_evidence"] = {
                "stratum": "existing_exact" if existing[index - 2] == references[index - 2] else "existing_wrong",
                "existing_recipient": existing[index - 2],
                "external_reference": references[index - 2],
            }
        selection_rows.append(row)
    fake_selection = {
        "directory": tmp_path,
        "summary": {"recipient_shadow_contract": {"labels": list(MODULE.LABELS)}},
        "summary_identity": frozen["selection-summary.json"],
        "selection_identity": frozen["selection.jsonl"],
        "input_identity": frozen["inputs.txt"],
        "rows": selection_rows,
        "sources": [source] * 6,
        "source_identities": [source_identity] * 6,
        "contracts": {},
    }
    texts = ["张三", "", "张三", "错误值", "王五", ""]
    records = [
        {
            "lines": (
                [_line(0, f"收款方：{text}")] if text else [_line(0, "转账成功")]
            )
        }
        for text in texts
    ]
    layout_bindings = {
        "layout_summary": frozen["layout-summary.json"],
        "layout_records": frozen["layout.jsonl"],
        "paddle_bundle": {"directory": "bundle"},
        "paddle_drop_score": 0.5,
        "latency_ms": {"total": {"count": 6}},
    }
    monkeypatch.setattr(MODULE, "_load_selection", lambda *_, **__: fake_selection)
    monkeypatch.setattr(MODULE, "_validate_layout", lambda *_, **__: (records, layout_bindings))

    def fake_module(path: Path, name: str):
        if path.name == FILTER_SCRIPT.name:
            return FILTER
        return SimpleNamespace()

    monkeypatch.setattr(MODULE, "_load_module", fake_module)
    output = tmp_path / "evaluation"
    MODULE.evaluate(selection_directory=tmp_path, layout_directory=tmp_path, output_directory=output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["targets"] == {
        "records": 2,
        "truth_reported": False,
        "shadow_candidate_records": 1,
        "ambiguous_records": 0,
        "unresolved_records": 1,
        "by_state": {"shadow_candidate": 1, "unresolved": 1},
        "by_shadow_route": {"full_layout_label_rhs_shadow": 1, "none": 1},
    }
    assert summary["controls"]["shadow_candidate_records"] == 3
    assert summary["controls"]["shadow_exact_records"] == 2
    assert summary["controls"]["false_positive_records"] == 1
    assert summary["controls"]["correct_to_wrong_records"] == 1
    assert summary["controls"]["wrong_to_correct_records"] == 1
    findings = [json.loads(line) for line in (output / "findings.jsonl").read_text().splitlines()]
    assert all("control_evaluation" not in row for row in findings[:2])
    assert all("external_reference" not in json.dumps(row) for row in findings[:2])
