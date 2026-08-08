#!/usr/bin/env python3
"""Audit the 66 remaining v4 recipient global-gate failures without OCR.

The audit binds the rejected 10,016-image formal A/B run, its 204-record
failure diagnostic, and the 75/129 frozen v4 consensus probe.  It replays no
model and changes no protection floor.  Its only purpose is to separate
detector-score, ordinary 25% geometry, alternative-envelope, and
rotation/projection evidence, then publish a conservative repair upper bound.

A successful report is analysis-only and always has
``formal_delivery_gate=false``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import importlib.util
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_repository_script(module_name: str, filename: str) -> Any:
    path = REPOSITORY_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - corrupt checkout
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = _load_repository_script(
    "receipt_mlnet_hybrid_targeted_replay_for_global_gate_audit",
    "receipt-mlnet-hybrid-targeted-replay.py",
)
DIAGNOSE = _load_repository_script(
    "receipt_mlnet_hybrid_pilot_diagnose_for_global_gate_audit",
    "receipt-mlnet-hybrid-pilot-diagnose.py",
)
PROBE_CONTRACT = _load_repository_script(
    "receipt_mlnet_hybrid_failure_truth_probe_for_global_gate_audit",
    "receipt-mlnet-hybrid-failure-truth-probe.py",
)


FORMAL_RECORDS = 10016
FAILURE_RECORDS = 204
EXPECTED_CANDIDATES = 75
EXPECTED_REMAINING = 129
EXPECTED_GATE_FAILURES = 66
EXPECTED_EXACT = 73
EXPECTED_DOMINANT = 2
EXPECTED_AMBIGUOUS = 15
EXPECTED_REJECTED_BY_GATE = 30
EXPECTED_UNRESOLVED = 84
EXPECTED_GATE_STATE_COUNTS = Counter(
    {
        "ambiguous": 5,
        "rejected_by_global_gate": 30,
        "unresolved": 31,
    }
)
MINIMUM_RECIPIENT_SCORE = 0.68
EXACT_ROUTE = "independent_crop_exact_consensus"
DOMINANT_ROUTE = "independent_crop_dominant_three_crop_consensus"
PROBE_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
PROBE_FINDING_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
AUDIT_SUMMARY_KIND = "receipt_mlnet_hybrid_global_gate_audit_v1"
AUDIT_FINDING_KIND = "receipt_mlnet_hybrid_global_gate_audit_finding_v1"
ALLOWED_GATE_FAILURES = (
    "recipient_score_not_available",
    "recipient_score_below_0.68",
    "ordinary_25pct_geometry_not_verified",
    "alternative_envelope_not_verified",
)
LAYOUT_REASONS = frozenset(
    {
        "recipient_left_edge",
        "recipient_right_edge",
        "recipient_width",
        "recipient_height",
        "amount_before_recipient",
        "recipient_before_payment",
        "amount_edge_overlap",
        "payment_edge_overlap",
    }
)
RECTIFICATION_REASONS = frozenset(
    {
        "geometry_missing",
        "source_size_missing_or_invalid",
        "rectified_size_missing_or_invalid",
        "H_original_to_rectified_missing_or_invalid",
    }
)


class AuditError(ValueError):
    """Frozen evidence is incomplete, inconsistent, or unsafe to classify."""


def _fail(message: str) -> None:
    raise AuditError(message)


def _identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return REPLAY._file_identity(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _assert_identity(identity: object, *, description: str) -> dict[str, Any]:
    try:
        return REPLAY._assert_identity(identity, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _load_json(path: Path, *, description: str) -> Any:
    try:
        return REPLAY._load_json(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    try:
        return REPLAY._load_jsonl(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _source_key(value: object) -> str:
    return REPLAY._source_key(value)


def _same(left: Any, right: Any) -> bool:
    return REPLAY._type_sensitive_equal(left, right)


def _nonempty(value: object, *, description: str) -> str:
    try:
        return REPLAY._nonempty_string(value, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _resolve_directory(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        _fail(f"{description} must be a regular directory: {resolved}")
    return resolved


def _assert_bound_identity(
    bound: object,
    actual_path: Path,
    *,
    description: str,
) -> dict[str, Any]:
    if not isinstance(bound, Mapping):
        _fail(f"{description} binding must be an object")
    actual = _identity(actual_path, description=description)
    try:
        REPLAY._assert_bound_identity(bound, actual, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error
    return actual


def _exact(payload: Mapping[str, Any], key: str, expected: Any, *, prefix: str) -> None:
    observed = payload.get(key)
    if type(observed) is not type(expected) or observed != expected:
        _fail(f"{prefix}.{key} must be {expected!r}, got {observed!r}")


def _count_rows(rows: object, *, description: str) -> Counter[str]:
    if not isinstance(rows, list):
        _fail(f"{description} must be an array")
    result: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"{description}[{index}] must be an object")
        name = row.get("name")
        records = row.get("records")
        if not isinstance(name, str) or not name:
            _fail(f"{description}[{index}].name must be non-empty")
        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            _fail(f"{description}[{index}].records must be a nonnegative integer")
        if name in result:
            _fail(f"{description} has duplicate group {name!r}")
        result[name] = records
    return result


def _expected_gate_failures(
    *, score: float | None, geometry_reasons: Sequence[str], envelope: bool | None
) -> list[str]:
    result: list[str] = []
    if score is None:
        result.append("recipient_score_not_available")
    elif score < MINIMUM_RECIPIENT_SCORE:
        result.append("recipient_score_below_0.68")
    if geometry_reasons:
        result.append("ordinary_25pct_geometry_not_verified")
    if envelope is not True:
        result.append("alternative_envelope_not_verified")
    return result


def _finite_score(value: object, *, source: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        _fail(f"probe {source!r} recipient_detector_score must be within [0, 1]")
    return float(value)


def _selected_candidate(shadow: Mapping[str, Any], *, source: str) -> str | None:
    route = shadow.get("selected_consensus_route")
    eligible = shadow.get("eligible_candidates")
    if not isinstance(eligible, list):
        _fail(f"probe {source!r} eligible_candidates must be an array")
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for index, row in enumerate(eligible):
        if not isinstance(row, Mapping):
            _fail(f"probe {source!r} eligible candidate {index} must be an object")
        candidate = row.get("candidate")
        crops = row.get("crops")
        if not isinstance(candidate, str) or not candidate.strip():
            _fail(f"probe {source!r} eligible candidate {index} has no candidate")
        if (
            not isinstance(crops, list)
            or any(not isinstance(crop, str) or not crop for crop in crops)
            or len(crops) != len(set(crops))
        ):
            _fail(f"probe {source!r} eligible candidate {index} crops are invalid")
        parsed.append((candidate, tuple(crops)))
    if route is None:
        return None
    if route == EXACT_ROUTE and len(parsed) == 1:
        return parsed[0][0]
    if route == DOMINANT_ROUTE and len(parsed) > 1:
        dominant = [candidate for candidate, crops in parsed if len(crops) == 3]
        if len(dominant) == 1:
            return dominant[0]
    _fail(f"probe {source!r} selected route has no unique matching candidate")


def _load_probe(
    root: Path,
    *,
    diagnostic_root: Path,
    diagnostic_by_source: Mapping[str, Mapping[str, Any]],
    missing_keys: set[str],
) -> dict[str, Any]:
    probe_root = _resolve_directory(root, description="v4 consensus probe")
    summary_path = probe_root / "summary.json"
    findings_path = probe_root / "findings.jsonl"
    summary_identity = _identity(summary_path, description="v4 probe summary")
    findings_identity = _identity(findings_path, description="v4 probe findings")
    summary = _load_json(summary_path, description="v4 probe summary")
    findings = _load_jsonl(findings_path, description="v4 probe findings")
    if not isinstance(summary, Mapping):
        _fail("v4 probe summary must be an object")
    for key, expected in (
        ("schema_version", 1),
        ("kind", PROBE_SUMMARY_KIND),
        ("read_only_existing_diagnostic", True),
        ("ocr_rerun", False),
        ("truth_used_for_analysis_only", True),
        ("runtime_truth_lookup", False),
        ("formal_delivery_gate", False),
        ("findings_records", FAILURE_RECORDS),
        ("unique_sources", FAILURE_RECORDS),
    ):
        _exact(summary, key, expected, prefix="v4 probe summary")

    formal_contract = summary.get("formal_contract")
    if not isinstance(formal_contract, Mapping):
        _fail("v4 probe has no formal_contract")
    for key, expected in (
        ("comparison_evaluation_mode", "formal"),
        ("comparison_records", FORMAL_RECORDS),
        ("failed_records", FAILURE_RECORDS),
        ("recipient_missing_only_records", FAILURE_RECORDS),
        ("recipient_missing_with_additional_failures_records", 0),
        ("non_missing_invariant_failure_records", 0),
    ):
        _exact(formal_contract, key, expected, prefix="v4 probe formal_contract")

    source_evidence = summary.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        _fail("v4 probe has no source_evidence")
    diagnostic_summary_identity = _assert_bound_identity(
        source_evidence.get("input_summary"),
        diagnostic_root / "summary.json",
        description="probe-bound diagnostic summary",
    )
    diagnostic_findings_identity = _assert_bound_identity(
        source_evidence.get("input_findings"),
        diagnostic_root / "findings.jsonl",
        description="probe-bound diagnostic findings",
    )

    teacher = summary.get("paddle_teacher_consensus")
    remaining = summary.get("remaining_failure_analysis")
    overlay = summary.get("remaining_global_gate_overlay_analysis")
    if not all(isinstance(value, Mapping) for value in (teacher, remaining, overlay)):
        _fail("v4 probe lacks teacher, remaining, or gate-overlay analysis")
    assert isinstance(teacher, Mapping)
    assert isinstance(remaining, Mapping)
    assert isinstance(overlay, Mapping)
    for key, expected in (
        ("external_truth", False),
        ("truth_used_for_analysis_only", True),
        ("formal_delivery_gate", False),
        ("interpretation", "self_consistency_coverage_not_human_accuracy"),
        ("records", EXPECTED_CANDIDATES),
    ):
        _exact(teacher, key, expected, prefix="v4 probe teacher")
    _exact(remaining, "records", EXPECTED_REMAINING, prefix="v4 probe remaining")
    _exact(
        remaining,
        "strict_candidate_records",
        EXPECTED_CANDIDATES,
        prefix="v4 probe remaining",
    )
    _exact(overlay, "records", EXPECTED_REMAINING, prefix="v4 probe overlay")
    _exact(
        overlay,
        "any_global_gate_failure_records",
        EXPECTED_GATE_FAILURES,
        prefix="v4 probe overlay",
    )
    _exact(
        overlay,
        "clear_global_gate_records",
        EXPECTED_REMAINING - EXPECTED_GATE_FAILURES,
        prefix="v4 probe overlay",
    )
    expected_routes = Counter({EXACT_ROUTE: EXPECTED_EXACT, DOMINANT_ROUTE: EXPECTED_DOMINANT})
    if _count_rows(
        teacher.get("by_runtime_route"), description="v4 probe teacher routes"
    ) != expected_routes:
        _fail("v4 probe runtime-route counts changed")

    contract = teacher.get("contract")
    if not isinstance(contract, Mapping):
        _fail("v4 probe teacher contract is missing")
    for key, expected in (
        ("minimum_recipient_detector_score", MINIMUM_RECIPIENT_SCORE),
        ("requires_empty_geometry_reasons", True),
        ("requires_verified_alternative_envelope", True),
        ("requires_same_exact_line_in_independent_crops", 2),
        ("dominant_fallback_requires_same_exact_line_in_all_crops", 3),
        ("dominant_fallback_requires_unique_all_crop_candidate", True),
    ):
        _exact(contract, key, expected, prefix="v4 probe teacher contract")

    if len(findings) != FAILURE_RECORDS:
        _fail(f"v4 probe must contain exactly {FAILURE_RECORDS} findings")
    by_source: dict[str, dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    gate_count = 0
    gate_state_counts: Counter[str] = Counter()
    candidate_count = 0
    remaining_count = 0
    for index, finding in enumerate(findings):
        for key, expected in (
            ("schema_version", 1),
            ("kind", PROBE_FINDING_KIND),
            ("truth_used_for_analysis_only", True),
            ("runtime_truth_lookup", False),
            ("formal_delivery_gate", False),
        ):
            _exact(finding, key, expected, prefix=f"v4 probe finding[{index}]")
        source = _nonempty(
            finding.get("source"), description=f"v4 probe finding[{index}] source"
        )
        key = _source_key(source)
        if key in by_source:
            _fail(f"duplicate v4 probe source: {source!r}")
        diagnostic_finding = diagnostic_by_source.get(key)
        if diagnostic_finding is None:
            _fail(f"v4 probe {source!r} has no bound diagnostic finding")
        try:
            recomputed_finding = PROBE_CONTRACT._analyze_finding(
                diagnostic_finding, index=index
            )
        except PROBE_CONTRACT.ProbeError as error:
            raise AuditError(
                f"could not rederive v4 probe finding {source!r}: {error}"
            ) from error
        if not _same(recomputed_finding, finding):
            _fail(
                f"v4 probe {source!r} differs from a fresh truth-free analysis "
                "of its bound diagnostic finding"
            )
        shadow = finding.get("shadow_candidate_truth_free")
        teacher_shadow = finding.get("paddle_teacher_consensus")
        strict_shadow = finding.get("strict_runtime_shadow")
        if not isinstance(shadow, Mapping) or not _same(shadow, teacher_shadow):
            _fail(f"v4 probe {source!r} teacher/shadow mismatch")
        if not isinstance(strict_shadow, Mapping):
            _fail(f"v4 probe {source!r} has no strict_runtime_shadow")
        truth_free_strict = {
            name: value for name, value in strict_shadow.items() if name != "truth_outcome"
        }
        if not _same(dict(shadow), truth_free_strict):
            _fail(f"v4 probe {source!r} strict/truth-free shadow mismatch")
        state = shadow.get("state")
        if state not in {"candidate", "unresolved", "ambiguous", "rejected_by_global_gate"}:
            _fail(f"v4 probe {source!r} has unsupported state {state!r}")
        state_counts[str(state)] += 1
        score = _finite_score(finding.get("recipient_detector_score"), source=source)
        geometry_reasons = finding.get("geometry_reasons")
        if (
            not isinstance(geometry_reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in geometry_reasons)
            or len(geometry_reasons) != len(set(geometry_reasons))
        ):
            _fail(f"v4 probe {source!r} geometry_reasons must be unique strings")
        envelope = finding.get("alternative_envelope")
        if envelope is not None and type(envelope) is not bool:
            _fail(f"v4 probe {source!r} alternative_envelope must be bool or null")
        gate_failures = shadow.get("global_gate_failures")
        if not isinstance(gate_failures, list) or any(
            failure not in ALLOWED_GATE_FAILURES for failure in gate_failures
        ):
            _fail(f"v4 probe {source!r} has unsupported global gate failures")
        expected_failures = _expected_gate_failures(
            score=score,
            geometry_reasons=geometry_reasons,
            envelope=envelope,
        )
        if gate_failures != expected_failures:
            _fail(f"v4 probe {source!r} global gate failures were not exactly recomputed")
        candidate = shadow.get("candidate")
        route = shadow.get("runtime_route")
        remaining_cluster = finding.get("remaining_failure_cluster")
        if state == "candidate":
            if (
                not isinstance(candidate, str)
                or not candidate.strip()
                or route not in {EXACT_ROUTE, DOMINANT_ROUTE}
                or gate_failures
                or remaining_cluster is not None
            ):
                _fail(f"v4 probe candidate {source!r} is internally inconsistent")
            candidate_count += 1
            route_counts[str(route)] += 1
        else:
            if candidate is not None or route is not None or not isinstance(
                remaining_cluster, Mapping
            ):
                _fail(f"v4 probe remaining {source!r} is internally inconsistent")
            remaining_count += 1
        selected_candidate = _selected_candidate(shadow, source=source)
        if state == "rejected_by_global_gate" and selected_candidate is None:
            _fail(f"v4 probe rejected record {source!r} has no selected consensus")
        if state != "candidate" and gate_failures:
            gate_count += 1
            gate_state_counts[str(state)] += 1
        by_source[key] = {
            "source": source,
            "finding": finding,
            "shadow": shadow,
            "score": score,
            "geometry_reasons": list(geometry_reasons),
            "alternative_envelope": envelope,
            "gate_failures": list(gate_failures),
            "selected_candidate": selected_candidate,
        }
    if set(by_source) != missing_keys:
        _fail("v4 probe source set differs from the 204 formal omissions")
    if candidate_count != EXPECTED_CANDIDATES or remaining_count != EXPECTED_REMAINING:
        _fail("v4 probe per-finding candidate/remaining counts changed")
    if gate_count != EXPECTED_GATE_FAILURES:
        _fail("v4 probe per-finding global-gate count changed")
    if route_counts != expected_routes:
        _fail("v4 probe per-finding runtime-route counts changed")
    expected_states = Counter(
        {
            "ambiguous": EXPECTED_AMBIGUOUS,
            "candidate": EXPECTED_CANDIDATES,
            "rejected_by_global_gate": EXPECTED_REJECTED_BY_GATE,
            "unresolved": EXPECTED_UNRESOLVED,
        }
    )
    if state_counts != expected_states:
        _fail("v4 probe frozen strict-state counts changed")
    if gate_state_counts != EXPECTED_GATE_STATE_COUNTS:
        _fail("v4 probe frozen per-state global-gate overlay counts changed")
    if _count_rows(
        teacher.get("by_state"), description="v4 probe teacher states"
    ) != state_counts:
        _fail("v4 probe summary state counts differ from findings")
    return {
        "root": probe_root,
        "summary": dict(summary),
        "by_source": by_source,
        "identities": {
            "probe_summary": summary_identity,
            "probe_findings": findings_identity,
            "diagnostic_summary": diagnostic_summary_identity,
            "diagnostic_findings": diagnostic_findings_identity,
            "probe_contract": _identity(
                REPOSITORY_ROOT
                / "scripts"
                / "receipt-mlnet-hybrid-failure-truth-probe.py",
                description="v4 probe truth-free contract",
            ),
        },
    }


def _load_diagnostic(
    root: Path,
    *,
    missing_keys: set[str],
) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(root / "findings.jsonl", description="formal diagnostic findings")
    if len(rows) != FAILURE_RECORDS:
        _fail(f"formal diagnostic findings must contain {FAILURE_RECORDS} records")
    by_source: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != 1 or row.get("kind") != REPLAY.DIAGNOSTIC_FINDING_KIND:
            _fail(f"formal diagnostic finding[{index}] has unsupported schema/kind")
        source = _nonempty(
            row.get("source"), description=f"formal diagnostic finding[{index}] source"
        )
        key = _source_key(source)
        if key in by_source:
            _fail(f"duplicate formal diagnostic source: {source!r}")
        by_source[key] = row
    if set(by_source) != missing_keys:
        _fail("formal diagnostic source set differs from the formal omission set")
    return by_source


def _reason_category(reason: str) -> str:
    if "_score_missing" in reason or "_score_below_" in reason:
        return "detector_score"
    if reason.endswith("_box_invalid") or reason.endswith("_box_outside_source"):
        return "detector_box"
    if (
        reason in RECTIFICATION_REASONS
        or reason.endswith("_box_projection_invalid")
        or reason.endswith("_box_outside_rectified")
    ):
        return "rectification_or_projection"
    if reason in LAYOUT_REASONS:
        return "layout_relation"
    _fail(f"unclassified frozen geometry reason: {reason!r}")


def _float32_ulp(value: float) -> float:
    value = abs(float(value))
    if value == 0.0:
        return math.ldexp(1.0, -149)
    _mantissa, exponent = math.frexp(value)
    return math.ldexp(1.0, exponent - 24)


def _boundary_tolerance(scale: float) -> float:
    # Eight float32 ULPs cover the bounded chain of products, sums, and one
    # division used by the public homography/geometry contract.  This is only
    # an audit label; it never changes a comparison or protection floor.
    return max(1e-6, 8.0 * _float32_ulp(max(1.0, abs(float(scale)))))


def _matrix_product(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][inner]) * float(right[inner][column]) for inner in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _validate_rectification_contract(
    geometry: Mapping[str, Any],
    context: object,
    *,
    rotation: int,
    source: str,
) -> None:
    if geometry.get("rectification") != "max-side-1600":
        _fail(f"formal result {source!r} is not rectification=max-side-1600")
    if geometry.get("screen_detected") is not False:
        _fail(f"formal result {source!r} screen_detected must be false")
    if isinstance(context, str):
        _fail(
            f"formal result {source!r} has an invalid production rectification "
            f"contract: {context}"
        )
    source_size, rectified_size, homography = context
    source_width, source_height = source_size
    expected_rotation = 90 if source_width > source_height else 0
    if rotation != expected_rotation:
        _fail(
            f"formal result {source!r} rotation {rotation} disagrees with "
            f"source dimensions {source_width}x{source_height}"
        )
    rotated_width = source_height if rotation == 90 else source_width
    rotated_height = source_width if rotation == 90 else source_height
    expected_width = rotated_width
    expected_height = rotated_height
    longest = max(rotated_width, rotated_height)
    if longest > 1600:
        scale = 1600.0 / longest
        expected_width = max(2, round(rotated_width * scale))
        expected_height = max(2, round(rotated_height * scale))
    if rectified_size != (expected_width, expected_height):
        _fail(
            f"formal result {source!r} rectified size {rectified_size!r} "
            f"does not match max-side-1600 {(expected_width, expected_height)!r}"
        )
    inverse = DIAGNOSE._homography(geometry.get("H_rectified_to_original"))
    if inverse is None:
        _fail(f"formal result {source!r} has no valid inverse homography")
    identity_tolerance = 1e-5
    for product in (
        _matrix_product(homography, inverse),
        _matrix_product(inverse, homography),
    ):
        for row in range(3):
            for column in range(3):
                expected = 1.0 if row == column else 0.0
                if abs(product[row][column] - expected) > identity_tolerance:
                    _fail(
                        f"formal result {source!r} homography/inverse are inconsistent"
                    )
    projected_source = DIAGNOSE._project_box_to_rectified(
        [0.0, 0.0, float(source_width - 1), float(source_height - 1)],
        homography,
    )
    if projected_source is None:
        _fail(f"formal result {source!r} source boundary does not project")
    expected_boundary = [
        0.0,
        0.0,
        float(expected_width - 1),
        float(expected_height - 1),
    ]
    if any(
        abs(observed - expected) > 0.01
        for observed, expected in zip(
            projected_source, expected_boundary, strict=True
        )
    ):
        _fail(
            f"formal result {source!r} homography does not map the full source "
            "onto the frozen rectified dimensions"
        )
    affine_tolerance = 1e-8
    if any(abs(homography[2][index]) > affine_tolerance for index in (0, 1)) or abs(
        homography[2][2] - 1.0
    ) > affine_tolerance:
        _fail(f"formal result {source!r} homography is not full-image affine")
    if rotation == 0:
        if (
            abs(homography[0][1]) > affine_tolerance
            or abs(homography[1][0]) > affine_tolerance
            or homography[0][0] <= 0.0
            or homography[1][1] <= 0.0
        ):
            _fail(f"formal result {source!r} homography disagrees with rotation 0")
    elif (
        abs(homography[0][0]) > affine_tolerance
        or abs(homography[1][1]) > affine_tolerance
        or abs(homography[0][1]) <= affine_tolerance
        or abs(homography[1][0]) <= affine_tolerance
    ):
        _fail(f"formal result {source!r} homography disagrees with rotation 90")


def _geometry_analysis(result: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    geometry = result.get("geometry")
    if not isinstance(geometry, Mapping):
        geometry = {}
    rotation = geometry.get("rotation_degrees")
    if isinstance(rotation, bool) or not isinstance(rotation, int) or rotation not in {
        0,
        90,
    }:
        _fail(f"formal result {source!r} has invalid rotation_degrees")
    detection_rows = result.get("detections")
    if not isinstance(detection_rows, list):
        _fail(f"formal result {source!r} has no detections array")
    detections: dict[str, Mapping[str, Any]] = {}
    for row in detection_rows:
        if not isinstance(row, Mapping):
            continue
        label = row.get("label")
        if isinstance(label, str) and label:
            if label in detections:
                _fail(f"formal result {source!r} has duplicate detection {label!r}")
            detections[label] = row

    recomputed_reasons = DIAGNOSE._geometry_reasons(result, detections)
    recomputed_evidence = DIAGNOSE._geometry_evidence(result, detections)
    context = DIAGNOSE._geometry_context(result)
    _validate_rectification_contract(
        geometry, context, rotation=rotation, source=source
    )
    projection_state = "verified"
    rectified_boxes: dict[str, list[float]] = {}
    rectified_size: tuple[int, int] | None = None
    if isinstance(context, str):
        projection_state = context
    else:
        source_size, rectified_size, homography = context
        _source_boxes, rectified_boxes, projection_reasons = (
            DIAGNOSE._project_detection_boxes(
                detections,
                source_size=source_size,
                rectified_size=rectified_size,
                homography=homography,
            )
        )
        if projection_reasons:
            projection_state = "+".join(sorted(set(projection_reasons)))

    margins: dict[str, float | None] = {
        "recipient_score_minus_0.68": None,
        "amount_score_minus_0.80": None,
        "payment_score_minus_0.80": None,
        "recipient_left_edge_slack": None,
        "recipient_right_edge_slack": None,
        "recipient_width_slack": None,
        "recipient_height_slack": None,
        "amount_before_recipient_slack": None,
        "recipient_before_payment_slack": None,
        "amount_edge_slack_25pct": None,
        "payment_edge_slack_25pct": None,
        "payment_edge_slack_45pct": None,
    }
    score_floors = {
        "recipient_field": ("recipient_score_minus_0.68", 0.68),
        "amount": ("amount_score_minus_0.80", 0.80),
        "payment_method_field": ("payment_score_minus_0.80", 0.80),
    }
    for label, (name, floor) in score_floors.items():
        score = DIAGNOSE._score(detections.get(label))
        if score is not None:
            margins[name] = float(score) - floor

    boundary_checks: list[str] = []
    for name in (
        "recipient_score_minus_0.68",
        "amount_score_minus_0.80",
        "payment_score_minus_0.80",
    ):
        value = margins[name]
        if value is not None:
            floor = 0.68 if name.startswith("recipient") else 0.80
            if abs(value) <= _boundary_tolerance(floor):
                boundary_checks.append(name)

    alternative_45_verified = False
    if projection_state == "verified" and rectified_size is not None:
        recipient = rectified_boxes["recipient"]
        amount = rectified_boxes["amount"]
        payment = rectified_boxes["payment"]
        width, height = rectified_size
        recipient_width = recipient[2] - recipient[0]
        recipient_height = recipient[3] - recipient[1]
        recipient_center = (recipient[1] + recipient[3]) * 0.5
        amount_center = (amount[1] + amount[3]) * 0.5
        payment_center = (payment[1] + payment[3]) * 0.5
        amount_tolerance = max(4.0, recipient_height * 0.25)
        payment_tolerance_25 = max(4.0, recipient_height * 0.25)
        payment_tolerance_45 = max(4.0, recipient_height * 0.45)
        margins.update(
            {
                "recipient_left_edge_slack": width * 0.20 - recipient[0],
                "recipient_right_edge_slack": recipient[2] - width * 0.80,
                "recipient_width_slack": recipient_width - width * 0.60,
                "recipient_height_slack": height * 0.15 - recipient_height,
                "amount_before_recipient_slack": recipient_center - amount_center,
                "recipient_before_payment_slack": payment_center - recipient_center,
                "amount_edge_slack_25pct": recipient[1]
                - (amount[3] - amount_tolerance),
                "payment_edge_slack_25pct": payment[1]
                + payment_tolerance_25
                - recipient[3],
                "payment_edge_slack_45pct": payment[1]
                + payment_tolerance_45
                - recipient[3],
            }
        )
        strict_positive = {
            "amount_before_recipient_slack",
            "recipient_before_payment_slack",
        }
        scale = max(float(width), float(height))
        tolerance = _boundary_tolerance(scale)
        for name, value in margins.items():
            if name.endswith(("0.68", "0.80")) or value is None:
                continue
            if abs(value) <= tolerance:
                boundary_checks.append(name)
        score_ok = all(
            margins[name] is not None and margins[name] >= 0.0
            for name in (
                "recipient_score_minus_0.68",
                "amount_score_minus_0.80",
                "payment_score_minus_0.80",
            )
        )
        common_layout_ok = all(
            margins[name] is not None
            and (margins[name] > 0.0 if name in strict_positive else margins[name] >= 0.0)
            for name in (
                "recipient_left_edge_slack",
                "recipient_right_edge_slack",
                "recipient_width_slack",
                "recipient_height_slack",
                "amount_before_recipient_slack",
                "recipient_before_payment_slack",
                "amount_edge_slack_25pct",
            )
        )
        alternative_45_verified = bool(
            score_ok
            and common_layout_ok
            and margins["payment_edge_slack_45pct"] is not None
            and margins["payment_edge_slack_45pct"] >= 0.0
        )

    ordinary_verified = not recomputed_reasons
    return {
        "rotation_degrees": rotation,
        "projection_state": projection_state,
        "recomputed_geometry_reasons": list(recomputed_reasons),
        "recomputed_geometry_evidence": recomputed_evidence,
        "ordinary_25pct_verified": ordinary_verified,
        "alternative_45pct_verified": alternative_45_verified,
        "margins": {
            name: None if value is None else round(float(value), 8)
            for name, value in margins.items()
        },
        "detection_scores": {
            label: DIAGNOSE._score(detections.get(label))
            for label in ("recipient_field", "amount", "payment_method_field")
        },
        "float32_boundary_checks": sorted(set(boundary_checks)),
    }


BOUNDARY_MARGIN_BY_REASON = {
    "recipient_score_below_0.68": "recipient_score_minus_0.68",
    "amount_score_below_0.80": "amount_score_minus_0.80",
    "payment_score_below_0.80": "payment_score_minus_0.80",
    "recipient_left_edge": "recipient_left_edge_slack",
    "recipient_right_edge": "recipient_right_edge_slack",
    "recipient_width": "recipient_width_slack",
    "recipient_height": "recipient_height_slack",
    "amount_before_recipient": "amount_before_recipient_slack",
    "recipient_before_payment": "recipient_before_payment_slack",
    "amount_edge_overlap": "amount_edge_slack_25pct",
    "payment_edge_overlap": "payment_edge_slack_25pct",
}


def _boundary_reasons(
    reasons: Sequence[str], boundary_checks: Sequence[str]
) -> list[str]:
    observed = set(boundary_checks)
    return sorted(
        reason
        for reason in reasons
        if BOUNDARY_MARGIN_BY_REASON.get(reason) in observed
    )


def _ordinary_geometry_state(reasons: Sequence[str], boundary_checks: Sequence[str]) -> str:
    if not reasons:
        return "verified"
    categories = sorted({_reason_category(reason) for reason in reasons})
    if set(_boundary_reasons(reasons, boundary_checks)) == set(reasons):
        return "float32_threshold_boundary+" + "+".join(categories)
    return "+".join(categories)


def _alternative_state(
    stored: bool | None,
    *,
    ordinary_verified: bool,
    alternative_45_verified: bool,
) -> str:
    recomputed = (
        "default_verified"
        if ordinary_verified
        else "45pct_only_verified"
        if alternative_45_verified
        else "not_verified"
    )
    stored_state = "true" if stored is True else "false" if stored is False else "unreported"
    return f"stored_{stored_state}|recomputed_{recomputed}"


def _group_rows(groups: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "records": len(sources),
            "examples": sorted(sources, key=_source_key)[:3],
        }
        for name, sources in sorted(groups.items(), key=lambda item: item[0])
    ]


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def audit(
    *,
    formal_directory: Path,
    diagnostic_directory: Path,
    probe_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    diagnostic_root = _resolve_directory(
        diagnostic_directory, description="formal failure diagnostic"
    )
    try:
        formal = REPLAY._load_formal_ab(
            formal_directory, diagnostic=diagnostic_root
        )
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error
    if len(formal["input_sources"]) != FORMAL_RECORDS:
        _fail(f"formal input must contain {FORMAL_RECORDS} records")
    if len(formal["missing_keys"]) != FAILURE_RECORDS:
        _fail(f"formal omission set must contain {FAILURE_RECORDS} records")
    diagnostic = _load_diagnostic(
        diagnostic_root, missing_keys=set(formal["missing_keys"])
    )
    probe = _load_probe(
        probe_directory,
        diagnostic_root=diagnostic_root,
        diagnostic_by_source=diagnostic,
        missing_keys=set(formal["missing_keys"]),
    )
    analysis_contracts = {
        "pilot_diagnose_contract": _identity(
            REPOSITORY_ROOT
            / "scripts"
            / "receipt-mlnet-hybrid-pilot-diagnose.py",
            description="pilot diagnostic geometry contract",
        ),
        "targeted_replay_contract": _identity(
            REPOSITORY_ROOT
            / "scripts"
            / "receipt-mlnet-hybrid-targeted-replay.py",
            description="formal binding and atomic publish contract",
        ),
    }

    if output_directory.is_symlink():
        _fail(f"refusing a symbolic-link global-gate audit output: {output_directory}")
    output = output_directory.resolve()
    if output.exists():
        _fail(f"refusing to overwrite global-gate audit: {output}")
    for root, description in (
        (formal["root"], "formal root"),
        (diagnostic_root, "diagnostic root"),
        (probe["root"], "probe root"),
    ):
        if _paths_overlap(output, Path(root).resolve()):
            _fail(f"global-gate audit output overlaps {description}")

    gate_groups: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list)
        for name in (
            "global_gate_failures_combination",
            "detector_score",
            "ordinary_25pct_geometry",
            "alternative_envelope",
            "rotation_projection",
            "failure_nature_combination",
            "strict_state_effect",
            "repair_surface",
        )
    }
    findings: list[dict[str, Any]] = []
    all_input_identities: list[tuple[str, Mapping[str, Any]]] = [
        (f"formal {name}", identity)
        for name, identity in formal["source_evidence"].items()
    ]
    all_input_identities.extend(
        (name, identity) for name, identity in probe["identities"].items()
    )
    all_input_identities.extend(
        (name, identity) for name, identity in analysis_contracts.items()
    )
    gate_evidence_investigation_upper_bound = 0
    candidate_investigation_upper_bound = 0
    reference_exact_investigation_upper_bound = 0
    threshold_boundary_records = 0
    state_counts: Counter[str] = Counter()

    missing_sources = [
        source
        for source in formal["input_sources"]
        if _source_key(source) in formal["missing_keys"]
    ]
    if len(missing_sources) != FAILURE_RECORDS:
        _fail("formal fixed-input order did not reproduce the omission set")
    for source in missing_sources:
        key = _source_key(source)
        probe_row = probe["by_source"][key]
        if not probe_row["gate_failures"]:
            continue
        diagnostic_row = diagnostic[key]
        finding = probe_row["finding"]
        shadow = probe_row["shadow"]
        old_entry = formal["hybrid"]["results"].get(key)
        if old_entry is None:
            _fail(f"formal hybrid result missing for {source!r}")
        old_result = old_entry["payload"]
        all_input_identities.append(
            (f"formal hybrid result {source}", old_entry["identity"])
        )

        try:
            old_recipient = REPLAY._field(old_result, "recipient")
            old_amount = REPLAY._field(old_result, "amount")
        except REPLAY.ReplayError as error:
            raise AuditError(str(error)) from error
        for diagnostic_name, observed in (
            ("amount_candidate", old_amount.get("candidate")),
            ("ppocr_route", old_recipient.get("hybrid_ocr_route")),
            (
                "ppocr_failure_reason",
                old_recipient.get("hybrid_ocr_failure_reason"),
            ),
            ("first_raw", old_recipient.get("hybrid_ocr_first_raw")),
            (
                "first_line_count",
                old_recipient.get("hybrid_ocr_first_line_count"),
            ),
            ("retry_raw", old_recipient.get("hybrid_ocr_retry_raw")),
            (
                "retry_line_count",
                old_recipient.get("hybrid_ocr_retry_line_count"),
            ),
            ("third_route", old_recipient.get("hybrid_ocr_third_route")),
            (
                "right_value_raw",
                old_recipient.get("hybrid_ocr_right_value_raw"),
            ),
            (
                "right_value_line_count",
                old_recipient.get("hybrid_ocr_right_value_line_count"),
            ),
            (
                "right_value_line_confidences",
                old_recipient.get("hybrid_ocr_right_value_line_confidences"),
            ),
        ):
            if not _same(diagnostic_row.get(diagnostic_name), observed):
                _fail(
                    f"diagnostic/formal result {source!r} {diagnostic_name} "
                    "binding mismatch"
                )

        for name, observed, expected in (
            ("recipient_score", diagnostic_row.get("recipient_score"), probe_row["score"]),
            (
                "geometry_reasons",
                diagnostic_row.get("geometry_reasons"),
                probe_row["geometry_reasons"],
            ),
            ("failures", diagnostic_row.get("failures"), [REPLAY.RECIPIENT_MISSING_FAILURE]),
            ("recipient_candidate", diagnostic_row.get("recipient_candidate"), None),
        ):
            if not _same(observed, expected):
                _fail(f"diagnostic/probe {source!r} {name} binding mismatch")

        geometry = _geometry_analysis(old_result, source=source)
        for diagnostic_name, label in (
            ("recipient_score", "recipient_field"),
            ("amount_score", "amount"),
            ("payment_score", "payment_method_field"),
        ):
            if not _same(
                diagnostic_row.get(diagnostic_name),
                geometry["detection_scores"][label],
            ):
                _fail(
                    f"diagnostic/formal detector {source!r} {diagnostic_name} "
                    "binding mismatch"
                )
        recomputed_reasons = geometry["recomputed_geometry_reasons"]
        diagnostic_geometry_matches_recompute = _same(
            diagnostic_row.get("geometry_reasons"), recomputed_reasons
        )
        diagnostic_evidence_matches_recompute = _same(
            diagnostic_row.get("geometry_evidence"),
            geometry["recomputed_geometry_evidence"],
        )
        score = probe_row["score"]
        score_state = (
            "not_available"
            if score is None
            else "verified_0.68_float32_boundary"
            if score >= MINIMUM_RECIPIENT_SCORE
            and score - MINIMUM_RECIPIENT_SCORE
            <= _boundary_tolerance(MINIMUM_RECIPIENT_SCORE)
            else "verified_0.68_plus"
            if score >= MINIMUM_RECIPIENT_SCORE
            else "below_0.68_float32_boundary"
            if MINIMUM_RECIPIENT_SCORE - score
            <= _boundary_tolerance(MINIMUM_RECIPIENT_SCORE)
            else "below_0.68"
        )
        ordinary_state = _ordinary_geometry_state(
            probe_row["geometry_reasons"], geometry["float32_boundary_checks"]
        )
        alternative_state = _alternative_state(
            probe_row["alternative_envelope"],
            ordinary_verified=geometry["ordinary_25pct_verified"],
            alternative_45_verified=geometry["alternative_45pct_verified"],
        )
        rotation_projection = (
            f"rotation_{geometry['rotation_degrees']}|projection_"
            f"{geometry['projection_state']}"
        )
        state = str(shadow["state"])
        state_counts[state] += 1
        selected_candidate = probe_row["selected_candidate"]
        state_effect = (
            "decisive_selected_consensus_block"
            if state == "rejected_by_global_gate"
            else "overlay_on_ambiguous"
            if state == "ambiguous"
            else "overlay_on_unresolved"
        )

        natures: set[str] = set()
        categories = {_reason_category(reason) for reason in probe_row["geometry_reasons"]}
        boundary_reasons = set(
            _boundary_reasons(
                probe_row["geometry_reasons"], geometry["float32_boundary_checks"]
            )
        )
        if score_state == "not_available" or score_state == "below_0.68":
            natures.add("true_detector_score_failure")
        elif score_state == "below_0.68_float32_boundary":
            natures.add("threshold_boundary_observation")
        if "detector_score" in categories:
            detector_score_reasons = {
                reason
                for reason in probe_row["geometry_reasons"]
                if _reason_category(reason) == "detector_score"
            }
            if detector_score_reasons and detector_score_reasons.issubset(
                boundary_reasons
            ):
                natures.add("threshold_boundary_observation")
            else:
                natures.add("true_detector_score_failure")
        if "detector_box" in categories:
            natures.add("true_detector_localization_failure")
        if "layout_relation" in categories:
            layout_reasons = {
                reason
                for reason in probe_row["geometry_reasons"]
                if _reason_category(reason) == "layout_relation"
            }
            if layout_reasons and layout_reasons.issubset(boundary_reasons):
                natures.add("threshold_boundary_observation")
            else:
                natures.add("true_layout_failure")
        if "rectification_or_projection" in categories:
            natures.add("rectification_or_projection_failure")
        alternative_mismatch = bool(
            probe_row["alternative_envelope"] is not True
            and (
                geometry["ordinary_25pct_verified"]
                or geometry["alternative_45pct_verified"]
            )
        )
        if alternative_mismatch:
            natures.add("coordinate_or_envelope_evidence_mismatch")
        elif probe_row["alternative_envelope"] is not True:
            natures.add("alternative_envelope_true_geometry_failure")
        if not diagnostic_geometry_matches_recompute or not diagnostic_evidence_matches_recompute:
            natures.add("diagnostic_recompute_mismatch")
        if geometry["float32_boundary_checks"]:
            natures.add("threshold_boundary_observation")
            threshold_boundary_records += 1
        nature_key = "+".join(sorted(natures)) or "none"

        failures_key = "+".join(probe_row["gate_failures"])
        repair_surface = (
            "restore_alternative_envelope_evidence"
            if alternative_mismatch
            else "detector_or_layout_evidence"
        )
        for group_name, group_key in (
            ("global_gate_failures_combination", failures_key),
            ("detector_score", score_state),
            ("ordinary_25pct_geometry", ordinary_state),
            ("alternative_envelope", alternative_state),
            ("rotation_projection", rotation_projection),
            ("failure_nature_combination", nature_key),
            ("strict_state_effect", state_effect),
            ("repair_surface", repair_surface),
        ):
            gate_groups[group_name][group_key].append(source)

        only_alternative_gate = probe_row["gate_failures"] == [
            "alternative_envelope_not_verified"
        ]
        investigation_candidate_without_floor_change = bool(
            only_alternative_gate
            and score is not None
            and score >= MINIMUM_RECIPIENT_SCORE
            and geometry["ordinary_25pct_verified"]
            and diagnostic_geometry_matches_recompute
            and diagnostic_evidence_matches_recompute
            and alternative_mismatch
            and not geometry["float32_boundary_checks"]
        )
        selected_consensus_investigation_candidate = bool(
            investigation_candidate_without_floor_change
            and state == "rejected_by_global_gate"
            and selected_candidate is not None
        )
        reference = finding.get("reference_recipient")
        reference_exact = bool(
            selected_consensus_investigation_candidate
            and isinstance(reference, str)
            and reference
            and selected_candidate == reference
        )
        gate_evidence_investigation_upper_bound += int(
            investigation_candidate_without_floor_change
        )
        candidate_investigation_upper_bound += int(
            selected_consensus_investigation_candidate
        )
        reference_exact_investigation_upper_bound += int(reference_exact)

        findings.append(
            {
                "schema_version": 1,
                "kind": AUDIT_FINDING_KIND,
                "source": source,
                "read_only_existing_results": True,
                "ocr_rerun": False,
                "formal_delivery_gate": False,
                "old_hybrid_result": old_entry["identity"],
                "strict_state": state,
                "selected_consensus_route": shadow.get("selected_consensus_route"),
                "selected_candidate_truth_free": selected_candidate,
                "global_gate_failures": probe_row["gate_failures"],
                "recipient_detector_score": score,
                "geometry_reasons": probe_row["geometry_reasons"],
                "alternative_envelope": probe_row["alternative_envelope"],
                "classification": {
                    "detector_score": score_state,
                    "ordinary_25pct_geometry": ordinary_state,
                    "alternative_envelope": alternative_state,
                    "rotation_projection": rotation_projection,
                    "failure_nature_combination": nature_key,
                    "strict_state_effect": state_effect,
                },
                "geometry_recalculation": geometry,
                "diagnostic_geometry_matches_recalculation": (
                    diagnostic_geometry_matches_recompute
                ),
                "diagnostic_geometry_evidence_matches_recalculation": (
                    diagnostic_evidence_matches_recompute
                ),
                "upper_bound_evidence": {
                    "investigation_candidate_without_floor_change": (
                        investigation_candidate_without_floor_change
                    ),
                    "selected_consensus_investigation_candidate": (
                        selected_consensus_investigation_candidate
                    ),
                    "safe_repair_proved_by_frozen_evidence": False,
                    "external_reference_present": isinstance(reference, str)
                    and bool(reference),
                    "retrospective_reference_exact": reference_exact,
                    "runtime_truth_lookup": False,
                    "formal_accuracy_claimed": False,
                },
            }
        )

    if len(findings) != EXPECTED_GATE_FAILURES:
        _fail(
            f"audit must classify exactly {EXPECTED_GATE_FAILURES} gate failures; "
            f"got {len(findings)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        findings_path = stage / "findings.jsonl"
        summary_path = stage / "summary.json"
        findings_path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in findings
            ),
            encoding="utf-8",
            newline="\n",
        )
        implementation_identity = _identity(
            Path(__file__), description="global-gate audit implementation"
        )
        summary = {
            "schema_version": 1,
            "kind": AUDIT_SUMMARY_KIND,
            "read_only_existing_results": True,
            "ocr_rerun": False,
            "protection_floor_changes_allowed": False,
            "parser_bypass_allowed": False,
            "formal_delivery_gate": False,
            "accepted": False,
            "audit_completed": True,
            "counts": {
                "formal_records": FORMAL_RECORDS,
                "formal_recipient_omissions": FAILURE_RECORDS,
                "v4_candidate_records": EXPECTED_CANDIDATES,
                "v4_remaining_records": EXPECTED_REMAINING,
                "v4_remaining_global_gate_failure_records": len(findings),
                "v4_remaining_global_gate_clear_records": (
                    EXPECTED_REMAINING - len(findings)
                ),
                "v4_exact_consensus_candidates_preserved": EXPECTED_EXACT,
                "v4_dominant_consensus_candidates_preserved": EXPECTED_DOMINANT,
            },
            "fixed_contract": {
                "minimum_recipient_detector_score": MINIMUM_RECIPIENT_SCORE,
                "ordinary_payment_overlap_fraction": 0.25,
                "exact_cjk_alternative_payment_overlap_fraction": 0.45,
                "float32_boundary_observation_ulps": 8,
                "boundary_observations_do_not_change_comparisons": True,
                "formal_gate": False,
            },
            "groups": {
                name: _group_rows(values) for name, values in gate_groups.items()
            },
            "investigation_upper_bound": {
                "alternative_envelope_evidence_records": (
                    gate_evidence_investigation_upper_bound
                ),
                "selected_consensus_candidate_records": (
                    candidate_investigation_upper_bound
                ),
                "retrospective_external_reference_exact_records": (
                    reference_exact_investigation_upper_bound
                ),
                "float32_threshold_boundary_records": threshold_boundary_records,
                "definition": (
                    "Requires the sole frozen failure to be alternative-envelope "
                    "verification, score>=0.68, recomputed ordinary 25% geometry "
                    "verified, and diagnostic/recalculation equality. Candidate "
                    "recovery additionally requires a uniquely selected frozen "
                    "consensus. No detector/layout floor is relaxed."
                ),
                "upper_bound_not_expected_yield": True,
                "safe_repair_claimed": False,
                "retrospective_truth_not_available_at_runtime": True,
                "requires_fresh_targeted_cpu_replay_before_any_production_change": True,
                "formal_accuracy_claimed": False,
                "formal_delivery_gate": False,
            },
            "safe_repair_upper_bound": {
                "records": 0,
                "proved_by_frozen_evidence": False,
                "reason": (
                    "The frozen public bbox_image values are source-space, "
                    "axis-aligned envelopes produced after projection and "
                    "clamping. Reprojecting them cannot reconstruct the exact "
                    "pre-projection rectified boxes used by the runtime gate."
                ),
                "evidence_required_to_raise_above_zero": (
                    "raw runtime rectified boxes or a fresh identity-bound CPU "
                    "replay that proves the existing gates without changing floors"
                ),
                "formal_delivery_gate": False,
            },
            "state_counts_with_gate_failures": dict(sorted(state_counts.items())),
            "source_evidence": {
                **formal["source_evidence"],
                "probe_summary": probe["identities"]["probe_summary"],
                "probe_findings": probe["identities"]["probe_findings"],
                "probe_contract": probe["identities"]["probe_contract"],
                **analysis_contracts,
                "audit_implementation": implementation_identity,
            },
            "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
            "warning": (
                "This is a read-only 66-record forensic audit. It cannot change "
                "the 0.68 detector floor, geometry gates, parser contract, or "
                "formal delivery status."
            ),
        }
        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for description, identity in all_input_identities:
            _assert_identity(identity, description=description)
        _assert_identity(implementation_identity, description="global-gate audit implementation")
        if output.exists() or output.is_symlink():
            _fail(f"refusing to overwrite global-gate audit: {output}")
        try:
            REPLAY._publish_directory(stage, output, description="global-gate audit")
        except REPLAY.ReplayError as error:
            raise AuditError(str(error)) from error
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {**summary, "output_directory": output.as_posix()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-directory", type=Path, required=True)
    parser.add_argument("--diagnostic-directory", type=Path, required=True)
    parser.add_argument("--probe-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit(
        formal_directory=args.formal_directory,
        diagnostic_directory=args.diagnostic_directory,
        probe_directory=args.probe_directory,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "kind": AUDIT_SUMMARY_KIND,
                "gate_failure_records": summary["counts"][
                    "v4_remaining_global_gate_failure_records"
                ],
                "candidate_investigation_upper_bound": summary[
                    "investigation_upper_bound"
                ][
                    "selected_consensus_candidate_records"
                ],
                "output_directory": summary["output_directory"],
                "formal_delivery_gate": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
