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
    dominant = _finding(
        8,
        first=[(0.96, "三路商户"), (0.95, "两路商户")],
        retry=[(0.94, "三路商户"), (0.93, "两路商户")],
        right=[(0.92, "三路商户")],
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
            8: dominant,
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
    assert summary["paddle_teacher_consensus"]["contract"][
        "recipient_label_pinyin_keys"
    ] == ["shoukuanfang", "shoukuanting", "shoukudnfang"]
    assert summary["paddle_teacher_consensus"]["contract"][
        "ascii_ui_line_keys"
    ] == sorted(MODULE.ASCII_UI_LINE_KEYS)
    assert summary["paddle_teacher_consensus"]["contract"][
        "dominant_fallback_requires_multiple_eligible_candidates"
    ] is True
    assert summary["paddle_teacher_consensus"]["contract"][
        "dominant_fallback_requires_same_exact_line_in_all_crops"
    ] == 3
    assert summary["paddle_teacher_consensus"]["contract"][
        "dominant_fallback_requires_unique_all_crop_candidate"
    ] is True
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
    remaining = summary["remaining_failure_analysis"]
    assert remaining["records"] == 6
    assert remaining["strict_candidate_records"] == 198
    assert remaining["unreported_failure_reason_records"] == 1
    assert all(
        len(group["examples"]) <= 3
        for groups in remaining["groups"].values()
        for group in groups
    )
    global_gate = summary["global_gate_failure_analysis"]
    assert global_gate["records"] == 3
    assert global_gate["selected_consensus_records"] == 3
    assert global_gate["single_eligible_candidate_records"] == 3
    assert global_gate["candidate_derivation_intact"] is True
    assert global_gate["parser_bypass_allowed"] is False
    assert global_gate["protection_floor_changes_allowed"] is False
    assert global_gate["repair_surface_definitions"][
        "rectification_or_projection"
    ] == "repair direction, homography, or coordinate projection"
    assert {
        group["name"]: group["records"]
        for group in global_gate["groups"]["repair_surface_record_incidence"]
    } == {
        "alternative_envelope_generation_or_verification": 1,
        "detector_layout_geometry": 1,
        "detector_score": 1,
    }
    overlay = summary["remaining_global_gate_overlay_analysis"]
    assert overlay["records"] == 6
    assert overlay["any_global_gate_failure_records"] == 4
    assert overlay["clear_global_gate_records"] == 2
    assert {
        group["name"]: group["records"]
        for group in overlay["groups"]["strict_state_by_gate_presence"]
    } == {
        "strict_state=ambiguous|global_gates=clear": 1,
        "strict_state=rejected_by_global_gate|global_gates=failed": 3,
        "strict_state=unresolved|global_gates=clear": 1,
        "strict_state=unresolved|global_gates=failed": 1,
    }
    unresolved = summary["unresolved_filter_analysis"]
    assert unresolved["records"] == 2
    assert unresolved["line_filters_remain_protective"] is True
    assert unresolved["protection_floor_changes_allowed"] is False
    assert unresolved["rejected_line_occurrences"] == [
        {"name": "low_confidence", "occurrences": 1}
    ]
    assert {
        group["name"]: group["records"]
        for group in unresolved["groups"]["primary_filter_blocker"]
    } == {
        "failure_evidence_unreported": 1,
        "raw_consensus_filtered:insufficient_high_confidence_crop_agreement": 1,
    }

    first = findings[0]
    assert first["attempts"]["first"]["lines"][0]["text"] == "商户甲"
    assert first["attempts"]["first"]["lines"][1]["text"] == "¥1,234.00"
    assert first["attempts"]["first"]["lines"][2]["text"] == "说明:冒号文本"
    assert first["reference_exact_positions"] == ["first:0", "retry:0", "right_value:0"]
    assert first["strict_runtime_shadow"]["candidate"] == "商户甲"
    assert first["strict_runtime_shadow"]["truth_outcome"] == "exact"
    assert first["remaining_failure_cluster"] is None
    assert "truth_outcome" not in first["shadow_candidate_truth_free"]
    assert "truth_outcome" not in first["paddle_teacher_consensus"]
    assert first["formal_delivery_gate"] is False
    assert first["runtime_truth_lookup"] is False
    assert findings[2]["raw_consensus"]["state"] == "one"
    assert findings[2]["strict_runtime_shadow"]["state"] == "unresolved"
    assert findings[2]["remaining_failure_cluster"]["unresolved_primary_blocker"] == (
        "raw_consensus_filtered:insufficient_high_confidence_crop_agreement"
    )
    assert findings[3]["strict_runtime_shadow"]["state"] == "rejected_by_global_gate"
    assert findings[3]["remaining_failure_cluster"][
        "global_gate_failures_combination"
    ] == "ordinary_25pct_geometry_not_verified"
    assert findings[4]["strict_runtime_shadow"]["state"] == "ambiguous"
    assert findings[4]["remaining_failure_cluster"]["ambiguous_candidate_count"] == 2
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
    assert findings[7]["remaining_failure_cluster"]["unresolved_primary_blocker"] == (
        "failure_evidence_unreported"
    )
    assert findings[8]["strict_runtime_shadow"]["candidate"] == "三路商户"
    assert findings[8]["strict_runtime_shadow"]["runtime_route"] == (
        "independent_crop_dominant_three_crop_consensus"
    )
    assert findings[8]["strict_runtime_shadow"]["selected_consensus_route"] == (
        findings[8]["strict_runtime_shadow"]["runtime_route"]
    )
    assert findings[8]["remaining_failure_cluster"] is None
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


@pytest.mark.parametrize(
    "gate_overrides",
    [
        {"recipient_score": 0.67},
        {"geometry_reasons": ["payment_edge_overlap"]},
        {"envelope": False},
    ],
)
def test_dominant_three_crop_shadow_never_bypasses_global_gates(
    gate_overrides: dict[str, object],
) -> None:
    row = _finding(
        0,
        first=[(0.96, "三路商户"), (0.95, "两路商户")],
        retry=[(0.94, "三路商户"), (0.93, "两路商户")],
        right=[(0.92, "三路商户")],
        **gate_overrides,
    )
    analyzed = MODULE._analyze_finding(row, index=0)
    shadow = analyzed["strict_runtime_shadow"]
    assert shadow["candidate"] is None
    assert shadow["state"] == "rejected_by_global_gate"
    assert shadow["runtime_route"] is None
    assert shadow["selected_consensus_route"] == (
        "independent_crop_dominant_three_crop_consensus"
    )
    assert shadow["global_gate_failures"]


def test_dominant_three_crop_shadow_keeps_two_dominant_values_ambiguous() -> None:
    row = _finding(
        0,
        first=[(0.96, "商户甲"), (0.95, "商户乙")],
        retry=[(0.94, "商户甲"), (0.93, "商户乙")],
        right=[(0.92, "商户甲"), (0.91, "商户乙")],
    )
    shadow = MODULE._analyze_finding(row, index=0)["strict_runtime_shadow"]
    assert shadow["candidate"] is None
    assert shadow["state"] == "ambiguous"
    assert shadow["runtime_route"] is None


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
    remaining = summary["remaining_failure_analysis"]
    assert remaining["records"] == 1
    assert remaining["strict_candidate_records"] == 203
    assert remaining["unreported_failure_reason_records"] == 1


def test_real_204_shape_matches_frozen_v3_state_transitions(
    tmp_path: Path,
) -> None:
    replacements: dict[int, dict[str, object]] = {}

    # Six old candidate rows contain only the pinyin recipient-row label and
    # therefore move candidate -> unresolved. Keep all three exact observed
    # spellings represented without changing the raw-consensus cardinality.
    for index, label in enumerate(
        ["shoukuanfang"] * 4 + ["shou kuan ting", "shoukudnfang"]
    ):
        replacements[index] = _finding(
            index,
            first=[(0.96, label)],
            retry=[(0.95, label)],
        )

    # The other 25 raw-unique rows remain strict candidates. Add
    # 44 raw-multiple rows whose amount line is removed by the strict contract,
    # leaving exactly one eligible merchant candidate.
    for index in range(31, 75):
        replacements[index] = _finding(
            index,
            first=[(0.96, f"商户{index}"), (0.99, "¥1,234.00")],
            retry=[(0.95, f"商户{index}"), (0.98, "¥1,234.00")],
        )

    # Twenty old records have two independently repeated strings. Two pair a
    # real payee with the UI label and move ambiguous -> candidate. Frozen v3
    # has three more records where one candidate alone spans all three crops;
    # one still fails the global envelope gate, so only two become candidates.
    for index in range(75, 95):
        replacements[index] = _finding(
            index,
            first=[(0.96, f"商户甲{index}"), (0.95, f"商户乙{index}")],
            retry=[(0.94, f"商户甲{index}"), (0.93, f"商户乙{index}")],
            right=(
                [(0.92, f"商户甲{index}")]
                if index in (77, 78, 79)
                else None
            ),
            # Real v3 overlay: one dominant row and five still-ambiguous rows
            # also carry a global-envelope failure. The parser must never use
            # dominant evidence to clear that independent gate.
            envelope=index not in {79, 80, 81, 82, 83, 84},
        )
    for index, label in zip(
        (75, 76), ("shoukuanfang", "shou kuan ting"), strict=True
    ):
        replacements[index] = _finding(
            index,
            first=[(0.96, f"商户{index}"), (0.95, label)],
            retry=[(0.94, f"商户{index}"), (0.93, label)],
        )

    # Thirty-two records were in the old global-gate rejection bucket. Thirty
    # have a real envelope failure; two (96 and 97) only had the diagnostic's
    # source-vs-rectified geometry-space bug and are now verified candidates.
    # The extra raw amount candidate proves raw-vs-strict grouping without
    # changing candidate derivation.
    for index in range(95, 127):
        replacements[index] = _finding(
            index,
            first=[(0.96, f"商户{index}"), (0.99, "¥1,234.00")],
            retry=[(0.95, f"商户{index}"), (0.98, "¥1,234.00")],
            envelope=index in (96, 97),
        )
    # One old rejected-by-envelope row has only the label as its eligible
    # string (the amount remains descriptive raw consensus), so it moves
    # rejected_by_global_gate -> unresolved.
    replacements[95] = _finding(
        95,
        first=[(0.96, "shoukudnfang"), (0.99, "¥1,234.00")],
        retry=[(0.95, "shoukudnfang"), (0.98, "¥1,234.00")],
        envelope=False,
    )
    # Seventy-six records have multiple raw consensus strings, but every one
    # is rejected by an explicit strict line contract.
    for index in range(127, 203):
        replacements[index] = _finding(
            index,
            first=[(0.99, "¥1,234.00"), (0.98, "付款方式")],
            retry=[(0.97, "¥1,234.00"), (0.96, "付款方式")],
            # Together with row 95 and the unreported row, these 29 overlays
            # reproduce the real 31 unresolved records with failed gates.
            envelope=index > 155,
        )

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
    replacements[203] = unreported

    source = _write_input(tmp_path / "diagnostic", replacements=replacements)
    findings, evidence = MODULE._load_input(source)
    summary = MODULE.summarize(findings, evidence=evidence)

    assert summary["paddle_teacher_consensus"]["records"] == 75
    assert summary["paddle_teacher_consensus"]["by_state"] == [
        {"name": "ambiguous", "records": 15},
        {"name": "candidate", "records": 75},
        {"name": "rejected_by_global_gate", "records": 30},
        {"name": "unresolved", "records": 84},
    ]
    assert summary["paddle_teacher_consensus"]["by_runtime_route"] == [
        {"name": "independent_crop_dominant_three_crop_consensus", "records": 2},
        {"name": "independent_crop_exact_consensus", "records": 73},
    ]
    assert summary["raw_consensus"]["by_state"] == [
        {"name": "multiple", "records": 172},
        {"name": "none", "records": 1},
        {"name": "one", "records": 31},
    ]

    remaining = summary["remaining_failure_analysis"]
    assert remaining["records"] == 129
    assert remaining["strict_candidate_records"] == 75
    assert remaining["unreported_failure_reason_records"] == 1
    assert remaining["by_failure_reason_type_all_records"] == [
        {"name": "anchored_or_alternative_parse_failed", "records": 203},
        {"name": "unreported", "records": 1},
    ]

    groups = remaining["groups"]
    assert {
        group["name"]: group["records"]
        for group in groups["eligible_candidate_count"]
    } == {"0": 84, "1": 29, "2": 16}
    assert groups["ambiguous_candidate_count"][0]["name"] == "2"
    assert groups["ambiguous_candidate_count"][0]["records"] == 15
    blocker_counts = {
        group["name"]: group["records"]
        for group in groups["unresolved_primary_blocker"]
    }
    assert blocker_counts == {
        "failure_evidence_unreported": 1,
        "raw_consensus_filtered:line_contract:negative_token": 6,
        "raw_consensus_filtered:line_contract:amount+line_contract:negative_token": 77,
    }
    assert all(
        len(group["examples"]) <= 3
        for grouped_rows in groups.values()
        for group in grouped_rows
    )

    assert findings[0]["remaining_failure_cluster"][
        "unresolved_primary_blocker"
    ] == "raw_consensus_filtered:line_contract:negative_token"
    assert all(
        findings[index]["strict_runtime_shadow"]["state"] == "unresolved"
        for index in range(6)
    )
    assert findings[7]["remaining_failure_cluster"] is None
    assert all(
        findings[index]["strict_runtime_shadow"]["state"] == "candidate"
        for index in (75, 76)
    )
    assert findings[75]["strict_runtime_shadow"]["candidate"] == "商户75"
    assert findings[77]["strict_runtime_shadow"]["candidate"] == "商户甲77"
    assert findings[77]["strict_runtime_shadow"]["runtime_route"] == (
        "independent_crop_dominant_three_crop_consensus"
    )
    assert findings[77]["strict_runtime_shadow"]["selected_consensus_route"] == (
        findings[77]["strict_runtime_shadow"]["runtime_route"]
    )
    assert findings[77]["remaining_failure_cluster"] is None
    assert findings[79]["strict_runtime_shadow"]["state"] == "rejected_by_global_gate"
    assert findings[79]["strict_runtime_shadow"]["runtime_route"] is None
    assert findings[79]["strict_runtime_shadow"]["selected_consensus_route"] == (
        "independent_crop_dominant_three_crop_consensus"
    )
    assert findings[79]["strict_runtime_shadow"]["global_gate_failures"] == [
        "alternative_envelope_not_verified"
    ]
    assert findings[80]["remaining_failure_cluster"]["ambiguous_candidate_count"] == 2
    assert findings[95]["strict_runtime_shadow"]["state"] == "unresolved"
    assert findings[95]["remaining_failure_cluster"][
        "global_gate_failures_combination"
    ] == "alternative_envelope_not_verified"
    assert all(
        findings[index]["strict_runtime_shadow"]["state"] == "candidate"
        for index in (96, 97)
    )
    assert all(findings[index]["remaining_failure_cluster"] is None for index in (96, 97))
    assert findings[127]["remaining_failure_cluster"]["raw_vs_strict"] == (
        "raw=multiple|strict=unresolved|raw_candidates=2|eligible=0"
    )
    assert findings[203]["remaining_failure_cluster"][
        "alternative_envelope_geometry_score"
    ] == (
        "envelope=unreported|"
        "geometry=failed:recipient_box_invalid+recipient_score_missing|"
        "score=unreported"
    )

    global_gate = summary["global_gate_failure_analysis"]
    assert global_gate["records"] == 30
    assert global_gate["selected_consensus_records"] == 30
    assert global_gate["single_eligible_candidate_records"] == 29
    assert global_gate["candidate_derivation_intact"] is True
    assert global_gate["parser_bypass_allowed"] is False
    assert global_gate["groups"]["geometry_reason_combination"] == [
        {
            "name": "verified",
            "records": 30,
            "examples": [
                rf"C:\Receipt Inputs\formal\{index:05d}.jpg"
                for index in (79, 98, 99)
            ],
        }
    ]
    assert {
        group["name"]: group["records"]
        for group in global_gate["groups"]["repair_surface_combination"]
    } == {"alternative_envelope_generation_or_verification": 30}

    overlay = summary["remaining_global_gate_overlay_analysis"]
    assert overlay["records"] == 129
    assert overlay["any_global_gate_failure_records"] == 66
    assert overlay["clear_global_gate_records"] == 63
    assert overlay["gate_failure_is_decisive_only_for_state"] == (
        "rejected_by_global_gate"
    )
    assert {
        group["name"]: group["records"]
        for group in overlay["groups"]["strict_state_by_gate_presence"]
    } == {
        "strict_state=ambiguous|global_gates=clear": 10,
        "strict_state=ambiguous|global_gates=failed": 5,
        "strict_state=rejected_by_global_gate|global_gates=failed": 30,
        "strict_state=unresolved|global_gates=clear": 53,
        "strict_state=unresolved|global_gates=failed": 31,
    }

    unresolved = summary["unresolved_filter_analysis"]
    assert unresolved["records"] == 84
    assert unresolved["parser_bypass_allowed"] is False
    assert unresolved["rejected_line_occurrences"] == [
        {"name": "amount", "occurrences": 154},
        {"name": "negative_token", "occurrences": 166},
    ]
    unresolved_groups = unresolved["groups"]
    assert {
        group["name"]: group["records"]
        for group in unresolved_groups["raw_consensus_state"]
    } == {"multiple": 77, "none": 1, "one": 6}
    assert {
        group["name"]: group["records"]
        for group in unresolved_groups["raw_candidate_filter_reason_combination"]
    } == {
        "line_contract:amount+line_contract:negative_token": 77,
        "line_contract:negative_token": 6,
        "none": 1,
    }
    assert {
        group["name"]: group["records"]
        for group in unresolved_groups["rejected_line_reason_record_incidence"]
    } == {"amount": 77, "negative_token": 83}
    assert {
        group["name"]: group["records"]
        for group in unresolved_groups["rejected_line_occurrence_signature"]
    } == {
        "amount=2|negative_token=2": 77,
        "negative_token=2": 6,
        "none": 1,
    }


@pytest.mark.parametrize(
    "value",
    [
        "CNY 200.00",
        "200.00 RMB",
        "招商银行储蓄卡(8885)",
        "合计200元",
        "shoukuanfang",
        "shou kuan ting",
        "shoukudnfang",
    ],
)
def test_strict_shadow_rejects_currency_and_payment_lines(value: str) -> None:
    allowed, reason = MODULE._shadow_line_allowed(value)
    assert allowed is False
    assert reason in {"amount", "negative_token"}


@pytest.mark.parametrize(
    "value",
    [
        "jia",
        "you",
        "V1SC",
        "D2-SAM ROSS",
        "Success Store",
        "Payment Labs",
        "TransferWise",
    ],
)
def test_strict_shadow_preserves_opaque_ascii_payee_candidates(value: str) -> None:
    assert MODULE._shadow_line_allowed(value) == (True, "accepted")


@pytest.mark.parametrize(
    "value",
    [
        "Payment Method",
        "Transfer Success",
        "Recipient",
        "Payee",
        "Amount",
        "Time",
        "Status",
        "Transfer Failed",
        "Processing",
        "Bank Card",
    ],
)
def test_strict_shadow_rejects_exact_ascii_ui_lines(value: str) -> None:
    assert MODULE._shadow_line_allowed(value) == (False, "negative_token")


def test_every_canonical_ascii_ui_key_is_rejected() -> None:
    assert MODULE.ASCII_UI_LINE_KEYS
    assert all(
        MODULE._shadow_line_allowed(value) == (False, "negative_token")
        for value in MODULE.ASCII_UI_LINE_KEYS
    )


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("recipient_score_below_0.68", "detector_score"),
        ("payment_box_invalid", "detector_box"),
        ("recipient_left_edge", "layout_relation"),
        (
            "H_original_to_rectified_missing_or_invalid",
            "rectification_or_projection",
        ),
        ("recipient_box_projection_invalid", "rectification_or_projection"),
        ("future_reason", "unclassified"),
    ],
)
def test_geometry_reason_categories_keep_repair_surfaces_explicit(
    reason: str, category: str
) -> None:
    assert MODULE._geometry_reason_category(reason) == category


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
