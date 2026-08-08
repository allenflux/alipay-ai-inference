from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "receipt-mlnet-hybrid-failure-truth-probe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_hybrid_failure_truth_probe", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _section(
    lines: list[tuple[float, str]] | None,
    *,
    route: str = "none",
    geometry: str = "not_evaluated",
) -> str:
    if lines is None:
        return "none"
    rendered = ",".join(
        f"{index}:{confidence}:{text}"
        for index, (confidence, text) in enumerate(lines)
    )
    return (
        f"line_count={len(lines)},alternative_route={route},"
        f"geometry={geometry},lines=[{rendered}]"
    )


def _finding(
    index: int,
    *,
    reference: str | None = None,
    first: list[tuple[float, str]] | None = None,
    retry: list[tuple[float, str]] | None = None,
    right: list[tuple[float, str]] | None = None,
    recipient_score: float = 0.9,
    geometry_reasons: list[str] | None = None,
    envelope: bool = True,
) -> dict[str, object]:
    first = [(0.96, f"商户{index}")] if first is None else first
    retry = [(0.95, f"商户{index}")] if retry is None else retry

    def raw(lines: list[tuple[float, str]] | None) -> str | None:
        if lines is None:
            return None
        return " ".join(" ".join(text.split()) for _, text in lines if text.strip())

    return {
        "schema_version": 1,
        "kind": MODULE.INPUT_FINDING_KIND,
        "source": rf"C:\Receipt Inputs\formal\{index:05d}.jpg",
        "reference": {"recipient": reference, "amount": "1,234.00"},
        "failures": [MODULE.RECIPIENT_MISSING_FAILURE],
        "recipient_candidate": None,
        "recipient_score": recipient_score,
        "geometry_reasons": [] if geometry_reasons is None else geometry_reasons,
        "ppocr_failure_reason": (
            "anchored_or_alternative_parse_failed;"
            f"alternative_envelope={envelope};"
            f"first={_section(first)};"
            f"retry={_section(retry)};"
            f"right_value={_section(right)}"
        ),
        "first_raw": raw(first),
        "first_line_count": len(first),
        "retry_raw": raw(retry),
        "retry_line_count": len(retry),
        "right_value_raw": raw(right),
        "right_value_line_count": None if right is None else len(right),
        "right_value_line_confidences": (
            None if right is None else [confidence for confidence, _ in right]
        ),
    }


def _write_input(
    root: Path, *, replacements: dict[int, dict[str, object]] | None = None
) -> Path:
    root.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "kind": MODULE.INPUT_SUMMARY_KIND,
        "comparison_evaluation_mode": "formal",
        "comparison_records": 10016,
        "invariant_failure_records": 204,
        "recipient_missing_records": 204,
        "recipient_missing_only_records": 204,
        "failed_records": 204,
        "non_missing_invariant_failure_records": 0,
        "recipient_missing_with_additional_failures_records": 0,
        "by_comparator_failure": [
            {"name": MODULE.RECIPIENT_MISSING_FAILURE, "records": 204}
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = [_finding(index) for index in range(204)]
    for index, row in (replacements or {}).items():
        rows[index] = row
    (root / "findings.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return root


def test_atomic_probe_separates_raw_consensus_strict_shadow_and_external_truth(
    tmp_path: Path,
) -> None:
    exact = _finding(
        0,
        reference="商户甲",
        first=[
            (0.96, "商户甲"),
            (0.91, "¥1,234.00"),
            (0.90, "说明:冒号文本"),
        ],
        retry=[(0.93, "商户甲")],
        right=[(0.92, "商户甲")],
    )
    wrong = _finding(
        1,
        reference="正确商户",
        first=[(0.94, "错误商户")],
        retry=[(0.91, "错误商户")],
    )
    low_confidence = _finding(
        2,
        first=[(0.79, "低置信商户")],
        retry=[(0.99, "低置信商户")],
    )
    geometry_rejected = _finding(
        3,
        first=[(0.99, "几何商户")],
        retry=[(0.98, "几何商户")],
        geometry_reasons=["payment_edge_overlap"],
    )
    ambiguous = _finding(
        4,
        first=[(0.96, "商户甲"), (0.95, "商户乙")],
        retry=[(0.94, "商户甲"), (0.93, "商户乙")],
    )
    wide_envelope_only = _finding(
        5,
        first=[(0.96, "宽包络商户")],
        retry=[(0.95, "宽包络商户")],
        envelope=False,
    )
    low_detector = _finding(
        6,
        first=[(0.96, "低检测商户")],
        retry=[(0.95, "低检测商户")],
        recipient_score=0.67,
    )
    unreported = _finding(7)
    unreported.update(
        {
            "ppocr_route": None,
            "ppocr_failure_reason": None,
            "third_route": None,
            "first_raw": None,
            "first_line_count": None,
            "retry_raw": None,
            "retry_line_count": None,
            "right_value_raw": None,
            "right_value_line_count": None,
            "right_value_line_confidences": None,
            "recipient_score": None,
            "geometry_reasons": ["recipient_score_missing", "recipient_box_invalid"],
        }
    )
    source = _write_input(
        tmp_path / "formal-diagnostic-truth",
        replacements={
            0: exact,
            1: wrong,
            2: low_confidence,
            3: geometry_rejected,
            4: ambiguous,
            5: wide_envelope_only,
            6: low_detector,
            7: unreported,
        },
    )
    before = {path: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "truth-probe"

    assert MODULE.main(
        [
            "--input-directory",
            str(source),
            "--output-directory",
            str(output),
        ]
    ) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    findings = [
        json.loads(line)
        for line in (output / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["formal_contract"] == {
        "comparison_evaluation_mode": "formal",
        "comparison_records": 10016,
        "failed_records": 204,
        "recipient_missing_only_records": 204,
        "recipient_missing_with_additional_failures_records": 0,
        "non_missing_invariant_failure_records": 0,
    }
    assert summary["external_reference"]["present_records"] == 2
    assert summary["external_reference"]["missing_records"] == 202
    assert summary["external_reference"]["teacher_consensus_truth_outcome"] == [
        {"name": "exact", "records": 1},
        {"name": "not_available", "records": 202},
        {"name": "wrong", "records": 1},
    ]
    assert summary["paddle_teacher_consensus"]["external_truth"] is False
    assert (
        summary["paddle_teacher_consensus"]["pseudo_truth_source"]
        == "ppocr_independent_crop_exact_consensus"
    )
    assert summary["paddle_teacher_consensus"]["interpretation"] == (
        "self_consistency_coverage_not_human_accuracy"
    )
    assert summary["first_alternative_route_by_geometry"] == [
        {"name": "alternative_route=none|geometry=not_evaluated", "records": 203},
        {"name": "alternative_route=unreported|geometry=unreported", "records": 1},
    ]
    assert summary["retry_alternative_route_by_geometry"] == [
        {"name": "alternative_route=none|geometry=not_evaluated", "records": 203},
        {"name": "alternative_route=unreported|geometry=unreported", "records": 1},
    ]
    assert summary["groups"]["geometry"]
    assert all(len(group["examples"]) <= 3 for group in summary["groups"]["geometry"])

    first = findings[0]
    assert first["attempts"]["first"]["lines"][0]["text"] == "商户甲"
    assert first["attempts"]["first"]["lines"][1]["text"] == "¥1,234.00"
    assert first["attempts"]["first"]["lines"][2]["text"] == "说明:冒号文本"
    assert first["reference_exact_positions"] == ["first:0", "retry:0", "right_value:0"]
    assert first["strict_runtime_shadow"]["candidate"] == "商户甲"
    assert first["strict_runtime_shadow"]["truth_outcome"] == "exact"
    assert "truth_outcome" not in first["shadow_candidate_truth_free"]
    assert "truth_outcome" not in first["paddle_teacher_consensus"]
    assert first["formal_delivery_gate"] is False
    assert first["runtime_truth_lookup"] is False
    assert findings[2]["raw_consensus"]["state"] == "one"
    assert findings[2]["strict_runtime_shadow"]["state"] == "unresolved"
    assert findings[3]["strict_runtime_shadow"]["state"] == "rejected_by_global_gate"
    assert findings[4]["strict_runtime_shadow"]["state"] == "ambiguous"
    assert findings[5]["strict_runtime_shadow"]["global_gate_failures"] == [
        "alternative_envelope_not_verified"
    ]
    assert findings[6]["strict_runtime_shadow"]["global_gate_failures"] == [
        "recipient_score_below_0.68"
    ]
    assert findings[7]["failure_reason_type"] == "unreported"
    assert findings[7]["raw_consensus"] == {"candidates": [], "state": "none"}
    assert findings[7]["strict_runtime_shadow"]["state"] == "unresolved"
    assert findings[7]["strict_runtime_shadow"]["candidate"] is None
    assert findings[7]["strict_runtime_shadow"]["global_gate_failures"] == [
        "recipient_score_not_available",
        "ordinary_25pct_geometry_not_verified",
        "alternative_envelope_not_verified",
    ]
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not list(tmp_path.glob(".truth-probe.*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.main(
            [
                "--input-directory",
                str(source),
                "--output-directory",
                str(output),
            ]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"first_raw": "不匹配"}, "first_raw disagrees"),
        ({"first_line_count": 9}, "line_count disagrees"),
        ({"recipient_score": 2.0}, "recipient_score must be within"),
        ({"geometry_reasons": "none"}, "geometry_reasons must be"),
    ],
)
def test_probe_rejects_cross_evidence_mismatches(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    row = _finding(0)
    row.update(mutation)
    source = _write_input(tmp_path / "input", replacements={0: row})

    with pytest.raises(MODULE.ProbeError, match=message):
        MODULE._load_input(source)


def test_probe_rejects_duplicate_normalized_windows_source(tmp_path: Path) -> None:
    first = _finding(0)
    second = _finding(1)
    second["source"] = "c:/receipt inputs/formal/00000.JPG"
    source = _write_input(
        tmp_path / "input", replacements={0: first, 1: second}
    )

    with pytest.raises(MODULE.ProbeError, match="duplicate finding source"):
        MODULE._load_input(source)


def test_probe_rejects_missing_failure_reason_with_partial_ocr_evidence(
    tmp_path: Path,
) -> None:
    row = _finding(0)
    row["ppocr_failure_reason"] = None
    row["ppocr_route"] = None
    source = _write_input(tmp_path / "input", replacements={0: row})

    with pytest.raises(MODULE.ProbeError, match="reports first_raw"):
        MODULE._load_input(source)


def test_real_shape_all_external_references_missing_and_one_route_unreported(
    tmp_path: Path,
) -> None:
    unreported = _finding(203)
    unreported.update(
        {
            "ppocr_route": None,
            "ppocr_failure_reason": None,
            "third_route": None,
            "first_raw": None,
            "first_line_count": None,
            "retry_raw": None,
            "retry_line_count": None,
            "right_value_raw": None,
            "right_value_line_count": None,
            "right_value_line_confidences": None,
            "recipient_score": None,
            "geometry_reasons": ["recipient_score_missing", "recipient_box_invalid"],
        }
    )
    source = _write_input(
        tmp_path / "diagnostic-truth", replacements={203: unreported}
    )
    output = tmp_path / "consensus-probe"

    assert MODULE.main(
        ["--input-directory", str(source), "--output-directory", str(output)]
    ) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["external_reference"]["present_records"] == 0
    assert summary["external_reference"]["missing_records"] == 204
    assert summary["external_reference"]["exact_line_2_of_3_crop_consensus"] == {
        "records": 0,
        "denominator": 0,
        "coverage": None,
        "by_crop_combination": [],
    }
    assert summary["external_reference"]["teacher_consensus_truth_outcome"] == [
        {"name": "not_available", "records": 204}
    ]
    assert summary["paddle_teacher_consensus"]["records"] == 203
    teacher_states = {
        row["name"]: row["records"]
        for row in summary["paddle_teacher_consensus"]["by_state"]
    }
    assert teacher_states == {
        "candidate": 203,
        "unresolved": 1,
    }


@pytest.mark.parametrize(
    "value",
    ["CNY 200.00", "200.00 RMB", "招商银行储蓄卡(8885)", "合计200元"],
)
def test_strict_shadow_rejects_currency_and_payment_lines(value: str) -> None:
    allowed, reason = MODULE._shadow_line_allowed(value)
    assert allowed is False
    assert reason in {"amount", "negative_token"}


def test_probe_rejects_nonformal_summary_and_output_inside_input(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input")
    findings, evidence = MODULE._load_input(source)
    summary = MODULE.summarize(findings, evidence=evidence)

    with pytest.raises(MODULE.ProbeError, match="must not be inside"):
        MODULE.write_atomic(
            source / "derived",
            input_directory=source,
            summary=summary,
            findings=findings,
        )

    payload = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    payload["comparison_records"] = 10015
    (source / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.ProbeError, match="comparison_records must equal 10016"):
        MODULE._load_input(source)
