#!/usr/bin/env python3
"""Audit whether frozen v13 recipient candidates can be a safe fallback.

This program is deliberately inference-free and analysis-only.  It binds the
JSON emitted by ``receipt-mlnet-hybrid-missing-audit.py`` to the atomic v4
consensus-probe directory, proves a closed 204-source set, and compares the
old v13 candidate with the truth-free strict PP-OCR shadow.  External
references, when present, are used only to rule out pollution in this report;
they are never emitted as, or authorized for, a runtime lookup table.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import unicodedata
from uuid import uuid4


MISSING_AUDIT_KIND = "receipt_mlnet_hybrid_missing_audit_v1"
PROBE_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
PROBE_FINDING_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
OUTPUT_SUMMARY_KIND = "receipt_mlnet_v13_recipient_fallback_audit_summary_v1"
OUTPUT_FINDING_KIND = "receipt_mlnet_v13_recipient_fallback_audit_finding_v1"
RECIPIENT_MISSING_FAILURE = "hybrid recipient candidate missing"
FORMAL_RECORDS = 10016
MISSING_RECORDS = 204
STRICT_PSEUDO_TRUTH_RECORDS = 75
REMAINING_RECORDS = 129


class AuditError(ValueError):
    """The frozen inputs do not prove the requested audit contract."""


def _fail(message: str) -> None:
    raise AuditError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise AuditError(f"invalid JSON at {location}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"{description} must be a regular file: {resolved}")
    return resolved


def _regular_directory(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        _fail(f"{description} must be a regular directory: {resolved}")
    return resolved


def _identity(path: Path, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    size = resolved.stat().st_size
    if size <= 0:
        _fail(f"{description} must be non-empty: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": size,
    }


def _assert_identities_current(
    identities: Mapping[str, Mapping[str, Any]],
) -> None:
    for description, expected in identities.items():
        path = Path(str(expected.get("path") or ""))
        actual = _identity(path, description=description)
        if actual != dict(expected):
            _fail(f"{description} changed while the audit was reading it")


def _assert_bound_identity(value: object, *, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{description} binding must be an object")
    path = value.get("path")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    if not isinstance(path, str) or not path:
        _fail(f"{description} binding has no path")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        _fail(f"{description} binding has no lowercase SHA-256")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        _fail(f"{description} binding has an invalid size_bytes")
    actual = _identity(Path(path), description=description)
    if actual != dict(value):
        _fail(f"{description} binding differs from the current file")
    return actual


def _load_json(path: Path, *, description: str) -> Any:
    resolved = _regular_file(path, description=description)
    return _loads(resolved.read_text(encoding="utf-8-sig"), location=str(resolved))


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, description=description)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                _fail(f"invalid {description} {resolved}:{line_number}: blank line")
            value = _loads(line, location=f"{resolved}:{line_number}")
            if not isinstance(value, dict):
                _fail(
                    f"invalid {description} {resolved}:{line_number}: expected object"
                )
            rows.append(value)
    if not rows:
        _fail(f"{description} is empty: {resolved}")
    return rows


def _exact(
    payload: Mapping[str, Any], key: str, expected: Any, *, description: str
) -> None:
    observed = payload.get(key)
    if type(observed) is not type(expected) or observed != expected:
        _fail(f"{description}.{key} must be {expected!r}, got {observed!r}")


def _source(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{description} must be a non-empty string")
    return value


def _source_key(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    return os.path.normpath(text).replace("\\", "/").casefold()


def _candidate(value: object, *, description: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(f"{description} must be a string or null")
    return value if value.strip() else None


def _normalize_recipient(value: str) -> str:
    """Conservative display normalization; deliberately preserves case."""

    return " ".join(unicodedata.normalize("NFKC", value).split())


def _type_sensitive_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _type_sensitive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_sensitive_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _load_missing_audit(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _load_json(path, description="hybrid missing audit JSON")
    if not isinstance(payload, Mapping):
        _fail("hybrid missing audit must be an object")
    for key, expected in (
        ("schema_version", 1),
        ("kind", MISSING_AUDIT_KIND),
        ("records", FORMAL_RECORDS),
        ("invariant_failure_records", MISSING_RECORDS),
        ("recipient_missing_records", MISSING_RECORDS),
        ("flagged_records", MISSING_RECORDS),
    ):
        _exact(payload, key, expected, description="hybrid missing audit")
    findings = payload.get("findings")
    if not isinstance(findings, list) or len(findings) != MISSING_RECORDS:
        _fail(
            "hybrid missing audit findings must contain "
            f"{MISSING_RECORDS} records"
        )
    by_source: dict[str, dict[str, Any]] = {}
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping):
            _fail(f"hybrid missing finding[{index}] must be an object")
        source = _source(
            finding.get("source"),
            description=f"hybrid missing finding[{index}].source",
        )
        key = _source_key(source)
        if key in by_source:
            _fail(f"duplicate hybrid missing source: {source!r}")
        if finding.get("invariant") is not False:
            _fail(f"hybrid missing finding {source!r} must be an invariant failure")
        if finding.get("failures") != [RECIPIENT_MISSING_FAILURE]:
            _fail(f"hybrid missing finding {source!r} is not missing-only")
        if _candidate(
            finding.get("recipient_candidate"),
            description=f"hybrid missing finding {source!r}.recipient_candidate",
        ) is not None:
            _fail(f"hybrid missing finding {source!r} has a recipient candidate")

        hybrid_field = finding.get("hybrid_recipient_field")
        if not isinstance(hybrid_field, Mapping):
            _fail(f"hybrid missing finding {source!r} has no hybrid recipient field")
        if _candidate(
            hybrid_field.get("candidate"),
            description=f"hybrid missing finding {source!r} hybrid candidate",
        ) is not None:
            _fail(f"hybrid missing finding {source!r} hybrid field is not missing")

        baseline_field = finding.get("baseline_recipient_field")
        if baseline_field is not None and not isinstance(baseline_field, Mapping):
            _fail(f"hybrid missing finding {source!r} baseline field is invalid")
        baseline_candidate = _candidate(
            baseline_field.get("candidate")
            if isinstance(baseline_field, Mapping)
            else None,
            description=f"hybrid missing finding {source!r} baseline candidate",
        )

        reference = finding.get("hybrid_recipient_reference_evidence")
        if not isinstance(reference, Mapping):
            _fail(f"hybrid missing finding {source!r} has no reference evidence")
        reference_present = reference.get("reference_present")
        if type(reference_present) is not bool:
            _fail(
                f"hybrid missing finding {source!r} reference_present must be boolean"
            )
        reference_text_raw = reference.get("reference_text")
        if reference_present:
            reference_text = _candidate(
                reference_text_raw,
                description=f"hybrid missing finding {source!r} reference text",
            )
            if reference_text is None:
                _fail(f"hybrid missing finding {source!r} has an empty reference")
        else:
            if reference_text_raw is not None and not isinstance(reference_text_raw, str):
                _fail(
                    f"hybrid missing finding {source!r} reference text must be string/null"
                )
            if isinstance(reference_text_raw, str) and reference_text_raw.strip():
                _fail(
                    f"hybrid missing finding {source!r} hides a present reference"
                )
            reference_text = None
        by_source[key] = {
            "source": source,
            "baseline_candidate": baseline_candidate,
            "reference_present": reference_present,
            "reference_text": reference_text,
        }
    return by_source, dict(payload)


def _load_probe(
    root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    summary = _load_json(root / "summary.json", description="v4 probe summary")
    findings = _load_jsonl(root / "findings.jsonl", description="v4 probe findings")
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
        ("findings_records", MISSING_RECORDS),
        ("unique_sources", MISSING_RECORDS),
    ):
        _exact(summary, key, expected, description="v4 probe summary")
    formal = summary.get("formal_contract")
    if not isinstance(formal, Mapping):
        _fail("v4 probe summary has no formal_contract")
    for key, expected in (
        ("comparison_evaluation_mode", "formal"),
        ("comparison_records", FORMAL_RECORDS),
        ("failed_records", MISSING_RECORDS),
        ("recipient_missing_only_records", MISSING_RECORDS),
        ("recipient_missing_with_additional_failures_records", 0),
        ("non_missing_invariant_failure_records", 0),
    ):
        _exact(formal, key, expected, description="v4 probe formal_contract")
    teacher = summary.get("paddle_teacher_consensus")
    remaining = summary.get("remaining_failure_analysis")
    if not isinstance(teacher, Mapping) or not isinstance(remaining, Mapping):
        _fail("v4 probe summary lacks teacher or remaining analysis")
    for key, expected in (
        ("external_truth", False),
        ("truth_used_for_analysis_only", True),
        ("formal_delivery_gate", False),
        ("interpretation", "self_consistency_coverage_not_human_accuracy"),
        ("records", STRICT_PSEUDO_TRUTH_RECORDS),
    ):
        _exact(teacher, key, expected, description="v4 probe teacher")
    contract = teacher.get("contract")
    if not isinstance(contract, Mapping):
        _fail("v4 probe teacher has no strict consensus contract")
    for key, expected in (
        ("minimum_line_confidence", 0.80),
        ("minimum_recipient_detector_score", 0.68),
        ("requires_empty_geometry_reasons", True),
        ("requires_verified_alternative_envelope", True),
        ("requires_same_exact_line_in_independent_crops", 2),
        ("dominant_fallback_requires_multiple_eligible_candidates", True),
        ("dominant_fallback_requires_same_exact_line_in_all_crops", 3),
        ("dominant_fallback_requires_unique_all_crop_candidate", True),
    ):
        _exact(contract, key, expected, description="v4 probe teacher contract")
    _exact(
        remaining,
        "records",
        REMAINING_RECORDS,
        description="v4 probe remaining analysis",
    )
    external = summary.get("external_reference")
    if not isinstance(external, Mapping):
        _fail("v4 probe summary has no external_reference analysis")
    present_records = external.get("present_records")
    missing_reference_records = external.get("missing_records")
    if (
        isinstance(present_records, bool)
        or not isinstance(present_records, int)
        or present_records < 0
        or isinstance(missing_reference_records, bool)
        or not isinstance(missing_reference_records, int)
        or missing_reference_records < 0
        or present_records + missing_reference_records != MISSING_RECORDS
    ):
        _fail("v4 probe external-reference counts are invalid")
    source_evidence = summary.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        _fail("v4 probe summary has no source_evidence")
    bound_inputs = {
        "probe_bound_diagnostic_summary": _assert_bound_identity(
            source_evidence.get("input_summary"),
            description="probe-bound diagnostic summary",
        ),
        "probe_bound_diagnostic_findings": _assert_bound_identity(
            source_evidence.get("input_findings"),
            description="probe-bound diagnostic findings",
        ),
    }
    _exact(
        remaining,
        "strict_candidate_records",
        STRICT_PSEUDO_TRUTH_RECORDS,
        description="v4 probe remaining analysis",
    )
    if len(findings) != MISSING_RECORDS:
        _fail(f"v4 probe findings must contain {MISSING_RECORDS} records")

    by_source: dict[str, dict[str, Any]] = {}
    candidate_records = 0
    observed_reference_records = 0
    for index, finding in enumerate(findings):
        for key, expected in (
            ("schema_version", 1),
            ("kind", PROBE_FINDING_KIND),
            ("truth_used_for_analysis_only", True),
            ("runtime_truth_lookup", False),
            ("formal_delivery_gate", False),
        ):
            _exact(finding, key, expected, description=f"v4 probe finding[{index}]")
        source = _source(
            finding.get("source"), description=f"v4 probe finding[{index}].source"
        )
        source_key = _source_key(source)
        if source_key in by_source:
            _fail(f"duplicate v4 probe source: {source!r}")
        shadow = finding.get("shadow_candidate_truth_free")
        teacher_shadow = finding.get("paddle_teacher_consensus")
        if not isinstance(shadow, Mapping) or not isinstance(teacher_shadow, Mapping):
            _fail(f"v4 probe finding {source!r} has no truth-free shadow")
        if not _type_sensitive_equal(dict(shadow), dict(teacher_shadow)):
            _fail(f"v4 probe finding {source!r} shadow/teacher evidence differs")
        state = shadow.get("state")
        if not isinstance(state, str) or not state:
            _fail(f"v4 probe finding {source!r} has no strict state")
        pseudo_candidate = _candidate(
            shadow.get("candidate"),
            description=f"v4 probe finding {source!r} shadow candidate",
        )
        if state == "candidate":
            if pseudo_candidate is None:
                _fail(f"v4 probe finding {source!r} candidate state has no candidate")
            if pseudo_candidate != " ".join(pseudo_candidate.split()):
                _fail(f"v4 probe finding {source!r} candidate is not probe-normalized")
            candidate_records += 1
            subset = "strict_pseudo_truth"
        else:
            if pseudo_candidate is not None:
                _fail(f"v4 probe finding {source!r} non-candidate state has a candidate")
            subset = "remaining"

        reference_present = finding.get("external_reference_present")
        if type(reference_present) is not bool:
            _fail(f"v4 probe finding {source!r} reference flag must be boolean")
        reference_raw = finding.get("reference_recipient")
        if reference_present:
            observed_reference_records += 1
            reference = _candidate(
                reference_raw,
                description=f"v4 probe finding {source!r} external reference",
            )
            if reference is None:
                _fail(f"v4 probe finding {source!r} external reference is empty")
        else:
            if reference_raw is not None:
                _fail(f"v4 probe finding {source!r} has unflagged external reference")
            reference = None
        by_source[source_key] = {
            "source": source,
            "subset": subset,
            "strict_state": state,
            "pseudo_candidate": pseudo_candidate,
            "reference_present": reference_present,
            "reference_text": reference,
        }
    if candidate_records != STRICT_PSEUDO_TRUTH_RECORDS:
        _fail(
            "v4 probe strict candidate count differs from summary: "
            f"{candidate_records} != {STRICT_PSEUDO_TRUTH_RECORDS}"
        )
    if observed_reference_records != present_records:
        _fail(
            "v4 probe external-reference finding count differs from summary: "
            f"{observed_reference_records} != {present_records}"
        )
    return by_source, dict(summary), bound_inputs


def _examples(sources: Sequence[str]) -> list[str]:
    return sorted(set(sources), key=_source_key)[:3]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _analyze(
    missing: Mapping[str, Mapping[str, Any]],
    probe: Mapping[str, Mapping[str, Any]],
    *,
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_keys = set(missing)
    probe_keys = set(probe)
    if missing_keys != probe_keys:
        _fail(
            "source set closure failed between hybrid missing audit and v4 probe: "
            f"missing_from_probe={len(missing_keys - probe_keys)} "
            f"extra_in_probe={len(probe_keys - missing_keys)}"
        )
    if len(missing_keys) != MISSING_RECORDS:
        _fail(f"closed source set must contain {MISSING_RECORDS} records")

    findings: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {
        "pseudo_missing": [],
        "pseudo_normalized_mismatch": [],
        "pseudo_raw_mismatch": [],
        "remaining_missing": [],
        "pollution_reference_missing": [],
        "pollution_normalized_mismatch": [],
        "pollution_raw_mismatch": [],
    }
    counts = {
        "overall_baseline_present": 0,
        "pseudo_baseline_present": 0,
        "pseudo_normalized_exact": 0,
        "pseudo_raw_exact": 0,
        "remaining_baseline_present": 0,
        "remaining_reference_present": 0,
        "remaining_reference_normalized_exact": 0,
        "remaining_reference_raw_exact": 0,
        "all_reference_present": 0,
        "pseudo_reference_present": 0,
    }
    for key in sorted(missing_keys):
        old = missing[key]
        shadow = probe[key]
        source = str(old["source"])
        baseline = old["baseline_candidate"]
        pseudo = shadow["pseudo_candidate"]
        subset = str(shadow["subset"])
        old_reference_present = bool(old["reference_present"])
        probe_reference_present = bool(shadow["reference_present"])
        if old_reference_present != probe_reference_present:
            _fail(f"external reference presence differs for source {source!r}")
        old_reference = old["reference_text"]
        probe_reference = shadow["reference_text"]
        if old_reference_present and _normalize_recipient(str(old_reference)) != (
            _normalize_recipient(str(probe_reference))
        ):
            _fail(f"external reference text differs for source {source!r}")
        reference = str(old_reference) if old_reference_present else None

        baseline_present = baseline is not None
        counts["overall_baseline_present"] += int(baseline_present)
        counts["all_reference_present"] += int(old_reference_present)
        normalized_pseudo_exact: bool | None = None
        raw_pseudo_exact: bool | None = None
        normalized_reference_exact: bool | None = None
        raw_reference_exact: bool | None = None

        if subset == "strict_pseudo_truth":
            counts["pseudo_reference_present"] += int(old_reference_present)
            if not baseline_present:
                groups["pseudo_missing"].append(source)
            else:
                counts["pseudo_baseline_present"] += 1
                assert pseudo is not None
                normalized_pseudo_exact = _normalize_recipient(str(baseline)) == (
                    _normalize_recipient(str(pseudo))
                )
                raw_pseudo_exact = baseline == pseudo
                counts["pseudo_normalized_exact"] += int(normalized_pseudo_exact)
                counts["pseudo_raw_exact"] += int(raw_pseudo_exact)
                if not normalized_pseudo_exact:
                    groups["pseudo_normalized_mismatch"].append(source)
                if not raw_pseudo_exact:
                    groups["pseudo_raw_mismatch"].append(source)
        else:
            if not baseline_present:
                groups["remaining_missing"].append(source)
            else:
                counts["remaining_baseline_present"] += 1
            if not old_reference_present:
                groups["pollution_reference_missing"].append(source)
            else:
                counts["remaining_reference_present"] += 1
                if baseline_present:
                    assert reference is not None
                    normalized_reference_exact = _normalize_recipient(
                        str(baseline)
                    ) == _normalize_recipient(reference)
                    raw_reference_exact = baseline == reference
                    counts["remaining_reference_normalized_exact"] += int(
                        normalized_reference_exact
                    )
                    counts["remaining_reference_raw_exact"] += int(
                        raw_reference_exact
                    )
                    if not normalized_reference_exact:
                        groups["pollution_normalized_mismatch"].append(source)
                    if not raw_reference_exact:
                        groups["pollution_raw_mismatch"].append(source)

        findings.append(
            {
                "schema_version": 1,
                "kind": OUTPUT_FINDING_KIND,
                "source": source,
                "analysis_only": True,
                "runtime_truth_lookup": False,
                "formal_delivery_gate": False,
                "subset": subset,
                "strict_probe_state": shadow["strict_state"],
                "baseline_candidate": baseline,
                "baseline_candidate_present": baseline_present,
                "baseline_candidate_normalized": (
                    _normalize_recipient(str(baseline)) if baseline_present else None
                ),
                "strict_pseudo_truth_candidate": pseudo,
                "strict_pseudo_truth_candidate_normalized": (
                    _normalize_recipient(str(pseudo)) if pseudo is not None else None
                ),
                "strict_pseudo_truth_normalized_exact": normalized_pseudo_exact,
                "strict_pseudo_truth_raw_exact": raw_pseudo_exact,
                "external_reference": {
                    "present": old_reference_present,
                    "truth_used_for_analysis_only": True,
                    "runtime_lookup_allowed": False,
                    "reference_text_copied_to_output": False,
                    "baseline_normalized_exact": normalized_reference_exact,
                    "baseline_raw_exact": raw_reference_exact,
                },
            }
        )

    pseudo_consistency_satisfied = (
        counts["pseudo_baseline_present"] == STRICT_PSEUDO_TRUTH_RECORDS
        and counts["pseudo_normalized_exact"] == STRICT_PSEUDO_TRUTH_RECORDS
        and counts["pseudo_raw_exact"] == STRICT_PSEUDO_TRUTH_RECORDS
    )
    remaining_coverage_satisfied = (
        counts["remaining_baseline_present"] == REMAINING_RECORDS
    )
    pollution_satisfied = (
        counts["remaining_reference_present"] == REMAINING_RECORDS
        and counts["remaining_reference_normalized_exact"] == REMAINING_RECORDS
        and counts["remaining_reference_raw_exact"] == REMAINING_RECORDS
        and counts["remaining_baseline_present"] == REMAINING_RECORDS
    )
    authorized = (
        pseudo_consistency_satisfied
        and remaining_coverage_satisfied
        and pollution_satisfied
    )
    failed_conditions: list[str] = []
    if not pseudo_consistency_satisfied:
        failed_conditions.append("strict_pseudo_truth_consistency_not_100_percent")
    if not remaining_coverage_satisfied:
        failed_conditions.append("remaining_baseline_coverage_not_100_percent")
    if not pollution_satisfied:
        failed_conditions.append("remaining_pollution_safety_not_proved")

    summary = {
        "schema_version": 1,
        "kind": OUTPUT_SUMMARY_KIND,
        "read_only_existing_evidence": True,
        "ocr_rerun": False,
        "analysis_only": True,
        "truth_used_for_analysis_only": True,
        "runtime_truth_lookup": False,
        "formal_delivery_gate": False,
        "production_fallback_authorized": authorized,
        "decision": {
            "authorized": authorized,
            "failed_conditions": failed_conditions,
            "authorization_requires_all_conditions": True,
        },
        "source_set_closure": {
            "closed": True,
            "hybrid_missing_sources": MISSING_RECORDS,
            "v4_probe_sources": MISSING_RECORDS,
            "intersection_sources": MISSING_RECORDS,
        },
        "counts": {
            "formal_records": FORMAL_RECORDS,
            "hybrid_missing_records": MISSING_RECORDS,
            "strict_pseudo_truth_records": STRICT_PSEUDO_TRUTH_RECORDS,
            "remaining_records": REMAINING_RECORDS,
        },
        "baseline_candidate_coverage": {
            "all_missing_records": {
                "candidate_records": counts["overall_baseline_present"],
                "records": MISSING_RECORDS,
                "coverage": _ratio(
                    counts["overall_baseline_present"], MISSING_RECORDS
                ),
            },
            "strict_pseudo_truth_records": {
                "candidate_records": counts["pseudo_baseline_present"],
                "records": STRICT_PSEUDO_TRUTH_RECORDS,
                "coverage": _ratio(
                    counts["pseudo_baseline_present"], STRICT_PSEUDO_TRUTH_RECORDS
                ),
            },
            "remaining_records": {
                "candidate_records": counts["remaining_baseline_present"],
                "records": REMAINING_RECORDS,
                "coverage": _ratio(
                    counts["remaining_baseline_present"], REMAINING_RECORDS
                ),
                "required_coverage": 1.0,
                "satisfied": remaining_coverage_satisfied,
                "missing_examples": _examples(groups["remaining_missing"]),
            },
        },
        "strict_pseudo_truth_consistency": {
            "truth_source": "truth_free_strict_ppocr_cross_crop_consensus",
            "external_accuracy_claimed": False,
            "records": STRICT_PSEUDO_TRUTH_RECORDS,
            "baseline_candidate_records": counts["pseudo_baseline_present"],
            "normalized_exact_records": counts["pseudo_normalized_exact"],
            "normalized_exact_rate": _ratio(
                counts["pseudo_normalized_exact"], STRICT_PSEUDO_TRUTH_RECORDS
            ),
            "raw_exact_records": counts["pseudo_raw_exact"],
            "raw_exact_rate": _ratio(
                counts["pseudo_raw_exact"], STRICT_PSEUDO_TRUTH_RECORDS
            ),
            "missing_records": len(groups["pseudo_missing"]),
            "normalized_mismatch_records": len(
                groups["pseudo_normalized_mismatch"]
            ),
            "raw_mismatch_records": len(groups["pseudo_raw_mismatch"]),
            "satisfied": pseudo_consistency_satisfied,
            "examples": {
                "missing": _examples(groups["pseudo_missing"]),
                "normalized_mismatch": _examples(
                    groups["pseudo_normalized_mismatch"]
                ),
                "raw_mismatch": _examples(groups["pseudo_raw_mismatch"]),
            },
        },
        "external_reference_presence": {
            "all_missing_records": counts["all_reference_present"],
            "strict_pseudo_truth_records": counts["pseudo_reference_present"],
            "remaining_records": counts["remaining_reference_present"],
            "external_truth_is_runtime_input": False,
            "external_reference_text_copied_to_output": False,
        },
        "remaining_pollution_safety_evidence": {
            "strict_condition": (
                "every remaining source has an external recipient reference and "
                "a non-empty v13 candidate that is both NFKC-whitespace-normalized "
                "exact and raw exact to that reference"
            ),
            "records": REMAINING_RECORDS,
            "reference_present_records": counts["remaining_reference_present"],
            "baseline_candidate_records": counts["remaining_baseline_present"],
            "normalized_exact_records": counts[
                "remaining_reference_normalized_exact"
            ],
            "raw_exact_records": counts["remaining_reference_raw_exact"],
            "reference_missing_records": len(
                groups["pollution_reference_missing"]
            ),
            "normalized_mismatch_records": len(
                groups["pollution_normalized_mismatch"]
            ),
            "raw_mismatch_records": len(groups["pollution_raw_mismatch"]),
            "satisfied": pollution_satisfied,
            "examples": {
                "reference_missing": _examples(
                    groups["pollution_reference_missing"]
                ),
                "normalized_mismatch": _examples(
                    groups["pollution_normalized_mismatch"]
                ),
                "raw_mismatch": _examples(groups["pollution_raw_mismatch"]),
            },
        },
        "normalization_contract": {
            "unicode": "NFKC",
            "whitespace": "collapse_and_trim",
            "case_folding": False,
            "raw_equality_reported_separately": True,
        },
        "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
        "source_evidence": {name: dict(value) for name, value in identities.items()},
    }
    return summary, findings


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_atomic(
    output_directory: Path,
    *,
    probe_directory: Path,
    identities: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> None:
    output = output_directory.resolve()
    if _is_within(output, probe_directory):
        _fail("output directory must not be inside the v4 probe input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fallback audit output: {output}")
    _assert_identities_current(identities)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        (stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "findings.jsonl").write_text(
            "".join(
                json.dumps(
                    finding,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for finding in findings
            ),
            encoding="utf-8",
        )
        _assert_identities_current(identities)
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite fallback audit output: {output}"
            )
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def audit(
    *,
    missing_audit_json: Path,
    probe_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    missing_path = _regular_file(
        missing_audit_json, description="hybrid missing audit JSON"
    )
    probe_root = _regular_directory(
        probe_directory, description="v4 consensus probe"
    )
    identities = {
        "hybrid_missing_audit": _identity(
            missing_path, description="hybrid missing audit JSON"
        ),
        "v4_probe_summary": _identity(
            probe_root / "summary.json", description="v4 probe summary"
        ),
        "v4_probe_findings": _identity(
            probe_root / "findings.jsonl", description="v4 probe findings"
        ),
    }
    missing, _ = _load_missing_audit(missing_path)
    probe, _, bound_probe_inputs = _load_probe(probe_root)
    identities.update(bound_probe_inputs)
    _assert_identities_current(identities)
    summary, findings = _analyze(missing, probe, identities=identities)
    _write_atomic(
        output_directory,
        probe_directory=probe_root,
        identities=identities,
        summary=summary,
        findings=findings,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-audit-json", type=Path, required=True)
    parser.add_argument("--probe-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = audit(
        missing_audit_json=args.missing_audit_json,
        probe_directory=args.probe_directory,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "kind": OUTPUT_SUMMARY_KIND,
                "output_directory": args.output_directory.resolve().as_posix(),
                "analysis_only": True,
                "formal_delivery_gate": False,
                "production_fallback_authorized": summary[
                    "production_fallback_authorized"
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
