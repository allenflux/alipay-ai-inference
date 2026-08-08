#!/usr/bin/env python3
"""Freeze and evaluate a reference-bearing CPU status-bar-time calibration.

``prepare`` validates the existing 10,016-record formal A/B, scorer, and truth
closures, then freezes 678 deterministic controls as two canonical 339-image
LayoutShadow shards. ``evaluate`` validates both fresh CPU shards and compares
the exact same strict visible status-bar clock route used by the diagnostic
evidence tool against the frozen external references.

Neither command writes production candidates or acts as a delivery gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import statistics
import sys
from typing import Any
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load required validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TARGETED = _load_script(
    "receipt_mlnet_hybrid_targeted_replay_validator",
    SCRIPTS_ROOT / "receipt-mlnet-hybrid-targeted-replay.py",
)
LAYOUT_EVIDENCE = _load_script(
    "receipt_mlnet_layout_shadow_evidence_validator",
    SCRIPTS_ROOT / "receipt-mlnet-layout-shadow-evidence.py",
)
FORMAL_AUDIT = _load_script(
    "receipt_mlnet_formal_missing_fields_audit_validator",
    SCRIPTS_ROOT / "receipt-mlnet-formal-missing-fields-audit.py",
)


FORMAL_RECORDS = 10016
SHARD_RECORDS = 339
TARGET_RECORDS = SHARD_RECORDS * 2
PREPARE_KIND = "receipt_mlnet_time_layout_calibration_selection_v1"
PREPARE_SUMMARY_KIND = "receipt_mlnet_time_layout_calibration_prepare_v1"
TRUTH_KIND = "receipt_mlnet_time_layout_calibration_truth_v1"
POOL_KIND = "receipt_mlnet_time_layout_calibration_pool_closure_v1"
EVALUATE_KIND = "receipt_mlnet_time_layout_calibration_evaluation_v1"
COMPARISON_KIND = "receipt_mlnet_time_layout_calibration_comparison_v1"
SCORE_KIND = "receipt_mlnet_unified_candidate_evaluation_v1"
AB_KIND = "receipt_mlnet_hybrid_recipient_cpu_ab_v1"
LAYOUT_SUMMARY_KIND = LAYOUT_EVIDENCE.LAYOUT_SUMMARY_KIND
LAYOUT_RECORD_KIND = LAYOUT_EVIDENCE.LAYOUT_RECORD_KIND


class CalibrationError(ValueError):
    """Raised when a frozen calibration closure or result is inconsistent."""


def _loads(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=TARGETED._reject_json_constant,
            object_pairs_hook=TARGETED._object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise CalibrationError(f"invalid JSON at {location}: {error}") from error


def _read_bytes(path: Path, *, description: str) -> bytes:
    if path.is_symlink():
        raise CalibrationError(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise CalibrationError(f"missing {description}: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise CalibrationError(f"{description} must be a regular non-symlink file: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise CalibrationError(f"cannot read {description}: {resolved}: {error}") from error


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path, payload: bytes | None = None) -> dict[str, Any]:
    data = _read_bytes(path, description="bound file") if payload is None else payload
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha(data),
        "size_bytes": len(data),
    }


def _require_regular_non_reparse_file(path: Path, *, description: str) -> Path:
    try:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise CalibrationError(f"{description} must not be a symlink/junction: {path}")
        metadata = path.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise CalibrationError(f"{description} must not be a reparse point: {path}")
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise CalibrationError(f"missing {description}: {path}") from error
    except OSError as error:
        raise CalibrationError(f"cannot inspect {description}: {path}: {error}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise CalibrationError(f"{description} must be a regular file: {resolved}")
    return resolved


def _records_path_from_score(score_directory: Path) -> Path:
    summary, _identity_unused = _load_json(
        score_directory / "summary.json", description="records-from-score summary"
    )
    if summary.get("schema_version") != 1 or summary.get("kind") != SCORE_KIND:
        raise CalibrationError("records-from-score summary schema/kind is unsupported")
    raw = summary.get("records")
    if not isinstance(raw, str) or not raw:
        raise CalibrationError("records-from-score summary has no records path")
    path = Path(raw)
    if not path.is_absolute():
        raise CalibrationError("records-from-score requires an absolute scorer records path")
    return _require_regular_non_reparse_file(
        path, description="records-from-score bound records file"
    )


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CalibrationError(f"{description} is not UTF-8: {path}") from error
    value = _loads(text, location=str(path))
    if not isinstance(value, dict):
        raise CalibrationError(f"{description} must be one JSON object")
    return value, _identity(path, payload)


def _load_jsonl(path: Path, *, description: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CalibrationError(f"{description} is not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CalibrationError(f"{description} has a blank line at {path}:{line_number}")
        value = _loads(line, location=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise CalibrationError(f"{description} row is not an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise CalibrationError(f"{description} is empty: {path}")
    return rows, _identity(path, payload)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _path_key(value: object) -> str:
    return TARGETED._source_key(value)


def _require_int(value: object, expected: int | None = None, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(f"{description} must be an integer")
    if expected is not None and value != expected:
        raise CalibrationError(f"{description} must be {expected}, found {value}")
    return value


def _require_number(value: object, *, description: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise CalibrationError(f"{description} is non-finite or below {minimum}")
    return result


def _assert_identity(expected: object, *, description: str) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise CalibrationError(f"{description} identity is missing")
    path = expected.get("path")
    if not isinstance(path, str):
        raise CalibrationError(f"{description} path is missing")
    # Some reused validators enrich identities with records/source-set fields.
    # Those fields remain part of the enclosing contract, but file identity is
    # exactly path + SHA-256 + byte size.  Comparing the whole mapping would
    # incorrectly reject those valid enriched identities.
    try:
        return TARGETED._assert_identity(
            expected,
            description=description,
            expected_path=Path(path),
        )
    except TARGETED.ReplayError as error:
        raise CalibrationError(str(error)) from error


def _same_path(left: object, right: Path) -> bool:
    return TARGETED._same_path(left, right)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return left_resolved == right_resolved \
        or left_resolved in right_resolved.parents \
        or right_resolved in left_resolved.parents


def _parse_input_payload(payload: bytes, *, records: int, description: str) -> list[str]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CalibrationError(f"{description} must be UTF-8 without BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CalibrationError(f"{description} is not strict UTF-8") from error
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise CalibrationError(f"{description} must end with exactly one newline")
    sources = text.splitlines()
    if len(sources) != records:
        raise CalibrationError(f"{description} must contain {records} records")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not source or source != source.strip() or not Path(source).is_absolute():
            raise CalibrationError(f"{description}[{index}] is blank, padded, or non-absolute")
        try:
            source_path = Path(source)
            if source_path.is_symlink():
                raise CalibrationError(f"{description}[{index}] must not be a symbolic link")
            resolved = source_path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise CalibrationError(f"{description}[{index}] does not exist: {source}") from error
        if not resolved.is_file() or resolved.is_symlink():
            raise CalibrationError(f"{description}[{index}] is not a regular file: {source}")
        key = _path_key(source)
        if key in seen:
            raise CalibrationError(f"{description} contains duplicate source: {source}")
        seen.add(key)
    return sources


def _reference_time_raw(record: Mapping[str, Any]) -> str | None:
    slots = record.get("slots")
    slot = slots.get("time") if isinstance(slots, Mapping) else None
    if not isinstance(slot, Mapping):
        return None
    text = slot.get("text")
    visible = slot.get("visible_text")
    reference = visible if isinstance(visible, str) and visible else text
    if not isinstance(reference, str) or not reference:
        return None
    return reference


def _reference_time(record: Mapping[str, Any]) -> str | None:
    reference = _reference_time_raw(record)
    if reference is None:
        return None
    # Calibration is intentionally narrower than the historical scorer: only
    # references that belong to the visible status-bar H:MM/HH:MM contract are
    # eligible. Datetimes, seconds, embedded and reversed strings are excluded.
    return reference if LAYOUT_EVIDENCE._clock_value(reference) == reference.replace("：", ":").strip() else None


def _status_class(record: Mapping[str, Any]) -> str:
    slots = record.get("slots")
    slot = slots.get("transfer_status") if isinstance(slots, Mapping) else None
    value = slot.get("class_name") if isinstance(slot, Mapping) else None
    return value if value in {"success", "failed", "pending"} else "unknown"


def _device_token(result: Mapping[str, Any]) -> str:
    device = result.get("device")
    if isinstance(device, Mapping):
        value = device.get("platform")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    raise CalibrationError("formal hybrid result device.platform is missing")


def _geometry_tokens(result: Mapping[str, Any]) -> tuple[int, str, int, int]:
    geometry = result.get("geometry")
    if not isinstance(geometry, Mapping):
        raise CalibrationError("formal hybrid result has no geometry")
    rotation = _require_int(geometry.get("rotation_degrees"), description="formal rotation")
    if rotation not in {0, 90}:
        raise CalibrationError(f"formal rotation is unsupported: {rotation}")
    size = geometry.get("source_size")
    if not isinstance(size, Mapping):
        raise CalibrationError("formal hybrid result has no source_size")
    width = _require_int(size.get("width"), description="formal source width")
    height = _require_int(size.get("height"), description="formal source height")
    if width < 2 or height < 2:
        raise CalibrationError("formal source dimensions are invalid")
    longest = max(width, height)
    size_bin = (
        "lt1200" if longest < 1200 else
        "1200_1999" if longest < 2000 else
        "2000_2999" if longest < 3000 else
        "ge3000"
    )
    return rotation, size_bin, width, height


def _load_records(records_path: Path, input_keys: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows, identity = _load_jsonl(records_path, description="unified records")
    by_source: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("split") != "val":
            continue
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise CalibrationError(f"records row {index} has no source")
        key = _path_key(source)
        if key not in input_keys:
            continue
        if key in by_source:
            raise CalibrationError(f"records contain duplicate selected source: {source}")
        by_source[key] = dict(row)
    if set(by_source) != input_keys:
        raise CalibrationError(
            f"records do not cover the formal input set: missing={len(input_keys-set(by_source))}"
        )
    identity["records"] = len(rows)
    return by_source, identity


def _load_formal(formal_root: Path) -> dict[str, Any]:
    try:
        root = formal_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise CalibrationError(f"formal root does not exist: {formal_root}") from error
    input_sources, input_identity = TARGETED._read_input_list(root / "fixed-selected-inputs.txt")
    if len(input_sources) != FORMAL_RECORDS:
        raise CalibrationError(f"formal input count must be {FORMAL_RECORDS}")
    baseline = TARGETED._load_run(
        root / "baseline-v13", expected_sources=input_sources, hybrid=False
    )
    hybrid = TARGETED._load_run(
        root / "hybrid-recipient", expected_sources=input_sources, hybrid=True
    )
    comparison = root / "comparison"
    summary, summary_identity = _load_json(comparison / "summary.json", description="formal A/B summary")
    rows, rows_identity = _load_jsonl(comparison / "comparisons.jsonl", description="formal A/B comparisons")
    for key, expected in {
        "schema_version": 2,
        "kind": AB_KIND,
        "evaluation_mode": "formal",
        "records": FORMAL_RECORDS,
        "input_set_identical": True,
        "cli_summary_counts_verified": True,
    }.items():
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            raise CalibrationError(f"formal A/B summary {key} must be {expected!r}")
    if len(rows) != FORMAL_RECORDS:
        raise CalibrationError("formal A/B comparisons count differs")
    input_set = summary.get("input_set")
    manifests = summary.get("run_manifests")
    summaries = summary.get("run_summaries")
    if not isinstance(input_set, Mapping) or not isinstance(manifests, Mapping) or not isinstance(summaries, Mapping):
        raise CalibrationError("formal A/B summary closure is incomplete")
    TARGETED._assert_bound_identity(input_set.get("input_manifest"), input_identity,
                                    description="formal input list")
    TARGETED._assert_bound_identity(manifests.get("baseline"), baseline["manifest_identity"],
                                    description="formal baseline manifest")
    TARGETED._assert_bound_identity(manifests.get("hybrid"), hybrid["manifest_identity"],
                                    description="formal hybrid manifest")
    TARGETED._assert_bound_identity(summaries.get("baseline"), baseline["summary_identity"],
                                    description="formal baseline summary")
    TARGETED._assert_bound_identity(summaries.get("hybrid"), hybrid["summary_identity"],
                                    description="formal hybrid summary")
    row_keys: list[str] = []
    seen_row_keys: set[str] = set()
    for index, row in enumerate(rows):
        source = row.get("source")
        if not isinstance(source, str):
            raise CalibrationError(f"formal A/B row {index} has no source")
        key = _path_key(source)
        if key in seen_row_keys:
            raise CalibrationError(f"formal A/B contains duplicate source: {source}")
        row_keys.append(key)
        seen_row_keys.add(key)
        if key not in hybrid["results"] or not isinstance(row.get("failures"), list) \
                or type(row.get("invariant")) is not bool:
            raise CalibrationError(f"formal A/B row {index} is not bound to the hybrid run")
    # The formal input list (and both inference manifests) is the authority for
    # calibration order.  The A/B comparator deliberately writes its rows in
    # sorted normalized-source-key order (`for source in sorted(results)`).
    # Validate that exact producer order rather than incorrectly requiring the
    # fixed input order; the source set remains hash-bound and exact.
    expected_comparison_order = sorted(_path_key(source) for source in input_sources)
    if row_keys != expected_comparison_order:
        raise CalibrationError("formal A/B source order differs from comparator canonical source-key order")
    return {
        "root": root,
        "input_sources": input_sources,
        "input_identity": input_identity,
        "baseline": baseline,
        "hybrid": hybrid,
        "source_evidence": {
            "formal_input_list": input_identity,
            "baseline_summary": baseline["summary_identity"],
            "baseline_manifest": baseline["manifest_identity"],
            "hybrid_summary": hybrid["summary_identity"],
            "hybrid_manifest": hybrid["manifest_identity"],
            "ab_summary": summary_identity,
            "ab_comparisons": rows_identity,
        },
    }


def _load_score(
    score_directory: Path,
    *,
    formal: Mapping[str, Any],
    records_identity: Mapping[str, Any],
    truth_by_source: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    summary, summary_identity = _load_json(
        score_directory / "summary.json", description="formal scorer summary"
    )
    rows, rows_identity = _load_jsonl(
        score_directory / "comparisons.jsonl", description="formal scorer comparisons"
    )
    if summary.get("schema_version") != 1 or summary.get("kind") != SCORE_KIND:
        raise CalibrationError("formal scorer summary schema/kind is unsupported")
    scope = summary.get("evaluation_scope")
    formal_gate = summary.get("formal_delivery_gate")
    accepted = summary.get("accepted")
    diagnostic_passed = summary.get("diagnostic_thresholds_passed")
    acceptance = summary.get("acceptance")
    failures = summary.get("failures")
    if type(formal_gate) is not bool or type(accepted) is not bool \
            or type(diagnostic_passed) is not bool or not isinstance(acceptance, Mapping) \
            or not isinstance(failures, list) \
            or not all(isinstance(value, str) and value for value in failures):
        raise CalibrationError("formal scorer disposition is incomplete")
    if accepted is not formal_gate or diagnostic_passed is not formal_gate \
            or acceptance.get("passed") is not formal_gate \
            or acceptance.get("formal_delivery_gate") is not formal_gate \
            or acceptance.get("diagnostic_thresholds_passed") is not diagnostic_passed \
            or acceptance.get("failures") != failures \
            or bool(failures) is diagnostic_passed:
        raise CalibrationError("formal scorer disposition flags/failures are inconsistent")
    if not isinstance(scope, Mapping) or scope.get("kind") != "full_split" \
            or scope.get("formal_delivery_gate") is not formal_gate \
            or scope.get("requested_limit") is not None \
            or scope.get("evaluated_expected_receipts") != FORMAL_RECORDS \
            or scope.get("full_split_expected_receipts") != FORMAL_RECORDS:
        raise CalibrationError("formal scorer evaluation scope is not the full frozen input set")
    if summary.get("coverage_contract_version") != 2 or summary.get("evaluation_split") != "val":
        raise CalibrationError("formal scorer coverage/split contract differs")
    if summary.get("floors") != FORMAL_AUDIT.FIXED_FLOORS:
        raise CalibrationError("formal scorer five-field floors differ from fixed delivery floors")
    if summary.get("records_sha256") != records_identity["sha256"] or not _same_path(
        summary.get("records"), Path(str(records_identity["path"]))
    ):
        raise CalibrationError("formal scorer records binding differs")
    hybrid = formal["hybrid"]
    if summary.get("manifest_sha256") != hybrid["manifest_identity"]["sha256"] or not _same_path(
        summary.get("manifest"), hybrid["manifest_path"]
    ):
        raise CalibrationError("formal scorer hybrid manifest binding differs")
    results_root = summary.get("results_root")
    results_path = Path(str(results_root))
    if not results_path.is_absolute():
        results_path = score_directory / results_path
    if not _same_path(str(results_path), hybrid["root"]):
        raise CalibrationError("formal scorer results_root differs from the frozen hybrid run")
    selection = summary.get("input_selection")
    if not isinstance(selection, Mapping) or selection.get("hash_bound") is not True \
            or selection.get("records") != FORMAL_RECORDS \
            or selection.get("selection_order") != FORMAL_AUDIT.FULL_SELECTION_ORDER \
            or selection.get("sha256") != formal["input_identity"]["sha256"] \
            or not _same_path(selection.get("path"), Path(str(formal["input_identity"]["path"]))):
        raise CalibrationError("formal scorer input selection binding differs")
    if scope.get("selection_order") != FORMAL_AUDIT.FULL_SELECTION_ORDER \
            or scope.get("input_list_sha256") != formal["input_identity"]["sha256"] \
            or not _same_path(scope.get("input_list_path"), Path(str(formal["input_identity"]["path"]))):
        raise CalibrationError("formal scorer evaluation-scope input binding differs")

    model_raw = summary.get("model")
    if not isinstance(model_raw, str) or not model_raw:
        raise CalibrationError("formal scorer model path is missing")
    model_path = Path(model_raw)
    if not model_path.is_absolute():
        model_path = score_directory / model_path
    try:
        model_identity = TARGETED._file_identity(model_path, description="formal scorer model")
    except TARGETED.ReplayError as error:
        raise CalibrationError(str(error)) from error
    if summary.get("model_sha256") != model_identity["sha256"]:
        raise CalibrationError("formal scorer model SHA-256 differs")

    try:
        audit_references = FORMAL_AUDIT._record_references(
            Path(str(records_identity["path"])),
            selected_order=[FORMAL_AUDIT._source_key(source) for source in formal["input_sources"]],
            split="val",
        )
        audit_hybrid_results = {
            FORMAL_AUDIT._source_key(value["source"]): value["payload"]
            for value in hybrid["results"].values()
        }
        audit_hybrid_ids = {
            FORMAL_AUDIT._source_key(value["source"]): value["identity"]
            for value in hybrid["results"].values()
        }
        validated_rows, reference_counts = FORMAL_AUDIT._score_comparisons(
            score_directory / "comparisons.jsonl",
            selected_keys=set(audit_references),
            references=audit_references,
            hybrid_results=audit_hybrid_results,
            hybrid_result_ids=audit_hybrid_ids,
        )
    except FORMAL_AUDIT.AuditError as error:
        raise CalibrationError(f"formal scorer five-field comparison closure differs: {error}") from error

    by_field = summary.get("by_field")
    denominators = summary.get("accuracy_denominators")
    denominator_fields = denominators.get("by_field") if isinstance(denominators, Mapping) else None
    input_reference_counts = selection.get("field_reference_counts")
    if not isinstance(by_field, Mapping) or not isinstance(denominators, Mapping) \
            or denominators.get("hash_bound") is not True \
            or not isinstance(denominator_fields, Mapping) \
            or not isinstance(input_reference_counts, Mapping):
        raise CalibrationError("formal scorer five-field denominator closure is incomplete")
    for field in FORMAL_AUDIT.FIELD_SPECS:
        field_rows = [row for (source_key, row_field), row in validated_rows.items() if row_field == field]
        metric = by_field.get(field)
        exact_matches = sum(row["raw_exact"] is True for row in field_rows)
        observed_exact_rate = _require_number(
            metric.get("raw_exact_match") if isinstance(metric, Mapping) else None,
            description=f"formal scorer {field} raw_exact_match",
            minimum=0,
        )
        expected_exact_rate = exact_matches / len(field_rows) if field_rows else 0.0
        if not isinstance(metric, Mapping) or metric.get("records") != reference_counts.get(field, 0) \
                or metric.get("raw_exact_matches") != exact_matches \
                or observed_exact_rate > 1 \
                or not math.isclose(observed_exact_rate, expected_exact_rate, rel_tol=0, abs_tol=1e-12) \
                or denominator_fields.get(field) != reference_counts.get(field, 0) \
                or input_reference_counts.get(field) != reference_counts.get(field, 0):
            raise CalibrationError(f"formal scorer {field} metric/denominator differs")
    all_time_by_source: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != 1 or row.get("kind") != "receipt_mlnet_unified_comparison_v1":
            raise CalibrationError(f"formal score row {index} schema/kind is unsupported")
        if row.get("field") != "time":
            continue
        source = row.get("source")
        reference = row.get("reference_text")
        if not isinstance(source, str) or not isinstance(reference, str) or not reference:
            raise CalibrationError(f"formal time score row {index} is incomplete")
        key = _path_key(source)
        if key in all_time_by_source or key not in truth_by_source:
            raise CalibrationError(f"formal time score row {index} source is duplicate/outside truth")
        expected_reference = _reference_time_raw(truth_by_source[key])
        if expected_reference is None:
            raise CalibrationError(f"formal time score row has no matching records reference: {source}")
        if reference != expected_reference:
            raise CalibrationError(f"formal time score reference differs from records: {source}")
        candidate = row.get("candidate_text")
        present = isinstance(candidate, str) and bool(candidate.strip())
        if row.get("candidate_present") is not present or row.get("raw_exact") is not (
            present and candidate == reference
        ):
            raise CalibrationError(f"formal time score comparison is internally inconsistent: {source}")
        hybrid_candidate = TARGETED._candidate(hybrid["results"][key]["payload"], "time")
        if candidate != hybrid_candidate:
            raise CalibrationError(f"formal score candidate differs from hybrid result: {source}")
        all_time_by_source[key] = dict(row)
        if _reference_time(truth_by_source[key]) is not None:
            by_source[key] = dict(row)
    expected_time_domain = {
        key for key, record in truth_by_source.items() if _reference_time_raw(record) is not None
    }
    if set(all_time_by_source) != expected_time_domain:
        raise CalibrationError(
            "formal time score domain differs from records references: "
            f"missing={len(expected_time_domain-set(all_time_by_source))}"
        )
    eligible = {key for key in expected_time_domain if _reference_time(truth_by_source[key]) is not None}
    if set(by_source) != eligible:
        raise CalibrationError(
            f"formal time score domain differs from strict reference domain: missing={len(eligible-set(by_source))}"
        )
    metrics = summary.get("by_field", {}).get("time") \
        if isinstance(summary.get("by_field"), Mapping) else None
    exact = sum(row["raw_exact"] is True for row in all_time_by_source.values())
    observed_rate = _require_number(
        metrics.get("raw_exact_match") if isinstance(metrics, Mapping) else None,
        description="formal scorer time raw_exact_match",
        minimum=0,
    )
    if not isinstance(metrics, Mapping) or metrics.get("records") != len(all_time_by_source) \
            or metrics.get("raw_exact_matches") != exact \
            or observed_rate > 1 \
            or not math.isclose(observed_rate, exact / len(all_time_by_source), rel_tol=0, abs_tol=1e-12):
        raise CalibrationError("formal scorer time metrics differ from comparisons")
    return by_source, {
        "score_summary": summary_identity,
        "score_comparisons": rows_identity,
        "score_model": model_identity,
    }, {
        "source_score_accepted": accepted,
        "source_score_formal_delivery_gate": formal_gate,
        "source_score_diagnostic_thresholds_passed": diagnostic_passed,
        "source_score_failures": list(failures),
        "source_score_scope": "hash_bound_full_split_val",
        "source_score_five_field_floors": dict(FORMAL_AUDIT.FIXED_FLOORS),
        "inherited_delivery_authority": False,
    }


def _stratum(record: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        f"correct:{str(record['old_v13_raw_exact']).lower()}",
        f"device:{record['device_platform']}",
        f"status:{record['status_class']}",
        f"rotation:{record['rotation_degrees']}",
        f"size:{record['size_bin']}",
    )


def _select_stratified(candidates: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise CalibrationError(f"only {len(candidates)} candidates available for quota {count}")
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    uncovered = {token for row in remaining for token in _stratum(row)[1:]}
    while remaining and uncovered and len(selected) < count:
        best = max(
            range(len(remaining)),
            key=lambda position: (
                len(set(_stratum(remaining[position])[1:]) & uncovered),
                -int(remaining[position]["canonical_index"]),
            ),
        )
        chosen = remaining.pop(best)
        selected.append(chosen)
        uncovered.difference_update(_stratum(chosen)[1:])
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    selected_sources = {_path_key(row["source"]) for row in selected}
    for row in remaining:
        if _path_key(row["source"]) not in selected_sources:
            groups[_stratum(row)[1:]].append(row)
    group_keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in group_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            raise CalibrationError("deterministic stratum fill exhausted unexpectedly")
    return selected


def _closure_row(
    index: int,
    source: str,
    baseline: Mapping[str, Any],
    hybrid: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": POOL_KIND,
        "canonical_index": index,
        "source": source,
        "baseline_result": baseline["identity"],
        "hybrid_result": hybrid["identity"],
    }


def _bindings_from_prepare(
    source_evidence: Mapping[str, Any],
    pool_rows: Sequence[Mapping[str, Any]],
    source_files: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bindings = [dict(value) for value in source_evidence.values() if isinstance(value, Mapping)]
    for row in pool_rows:
        bindings.extend([dict(row["baseline_result"]), dict(row["hybrid_result"])])
    bindings.extend(dict(value) for value in source_files)
    return bindings


def _source_closure(identities: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        path = str(identity["path"])
        digest.update(
            f"{_path_key(path)}\0{path}\0{identity['sha256']}\0{identity['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _assert_bindings(bindings: Sequence[Mapping[str, Any]]) -> None:
    for expected in bindings:
        _assert_identity(expected, description="calibration bound input")


def prepare(
    *,
    formal_root: Path,
    records_path: Path,
    score_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.absolute()
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite time calibration selection: {output}")
    for input_path in (formal_root, records_path, score_directory):
        if _paths_overlap(output, input_path):
            raise CalibrationError("prepare output must not overlap a frozen input path")
    formal = _load_formal(formal_root)
    input_keys = {_path_key(source) for source in formal["input_sources"]}
    records_by_source, records_identity = _load_records(records_path, input_keys)
    score_by_source, score_evidence, score_disposition = _load_score(
        score_directory,
        formal=formal,
        records_identity=records_identity,
        truth_by_source=records_by_source,
    )
    candidates: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for index, source in enumerate(formal["input_sources"]):
        key = _path_key(source)
        baseline = formal["baseline"]["results"][key]
        hybrid = formal["hybrid"]["results"][key]
        pool_rows.append(_closure_row(index, source, baseline, hybrid))
        comparison = score_by_source.get(key)
        reference = _reference_time(records_by_source[key])
        if comparison is None or reference is None:
            continue
        baseline_candidate = TARGETED._candidate(baseline["payload"], "time")
        hybrid_candidate = TARGETED._candidate(hybrid["payload"], "time")
        if baseline_candidate != hybrid_candidate or hybrid_candidate != comparison.get("candidate_text"):
            raise CalibrationError(f"formal A/B changed the time candidate: {source}")
        rotation, size_bin, width, height = _geometry_tokens(hybrid["payload"])
        candidates.append({
            "canonical_index": index,
            "source": source,
            "reference_text": reference,
            "old_v13_candidate": comparison.get("candidate_text"),
            "old_v13_raw_exact": comparison.get("raw_exact") is True,
            "device_platform": _device_token(hybrid["payload"]),
            "status_class": _status_class(records_by_source[key]),
            "rotation_degrees": rotation,
            "size_bin": size_bin,
            "source_width": width,
            "source_height": height,
            "record_id": records_by_source[key].get("id"),
            "group_id": records_by_source[key].get("group_id"),
        })
    if len(candidates) < TARGET_RECORDS:
        raise CalibrationError(f"strict reference-bearing pool has only {len(candidates)} records")
    incorrect = [row for row in candidates if not row["old_v13_raw_exact"]]
    correct = [row for row in candidates if row["old_v13_raw_exact"]]
    if not incorrect or not correct:
        raise CalibrationError("strict calibration pool must contain both old-v13-correct and incorrect records")
    incorrect_count = min(len(incorrect), TARGET_RECORDS // 2)
    correct_count = TARGET_RECORDS - incorrect_count
    if len(correct) < correct_count:
        correct_count = len(correct)
        incorrect_count = TARGET_RECORDS - correct_count
    selected = _select_stratified(incorrect, incorrect_count) + _select_stratified(correct, correct_count)
    selected.sort(key=lambda row: int(row["canonical_index"]))
    if len(selected) != TARGET_RECORDS or len({_path_key(row["source"]) for row in selected}) != TARGET_RECORDS:
        raise CalibrationError("internal calibration selection count/uniqueness differs")
    for name in ("device_platform", "status_class", "rotation_degrees", "size_bin"):
        available = {str(row[name]) for row in candidates}
        covered = {str(row[name]) for row in selected}
        if covered != available:
            raise CalibrationError(
                f"deterministic calibration selection failed hard {name} marginal coverage: "
                f"missing={sorted(available-covered)}"
            )

    source_files: list[dict[str, Any]] = []
    for row in selected:
        source_files.append(_identity(Path(str(row["source"]))))
    source_evidence = {
        **formal["source_evidence"],
        "records": records_identity,
        **score_evidence,
    }
    bindings = _bindings_from_prepare(source_evidence, pool_rows, source_files)
    _assert_bindings(bindings)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        shard_contracts: list[dict[str, Any]] = []
        for shard_index in range(2):
            shard = selected[shard_index * SHARD_RECORDS:(shard_index + 1) * SHARD_RECORDS]
            payload = ("\n".join(str(row["source"]) for row in shard) + "\n").encode("utf-8")
            path = stage / f"shard-{shard_index}-inputs.txt"
            path.write_bytes(payload)
            shard_contracts.append({
                "index": shard_index,
                "relative_path": path.name,
                "records": SHARD_RECORDS,
                "sha256": _sha(payload),
                "size_bytes": len(payload),
                "canonical_index_min": shard[0]["canonical_index"],
                "canonical_index_max": shard[-1]["canonical_index"],
            })
        truth_rows = [
            {
                "schema_version": 1,
                "kind": TRUTH_KIND,
                "diagnostic_only": True,
                "formal_delivery_gate": False,
                "candidate_write_enabled": False,
                **row,
                "source_file": source_files[index],
            }
            for index, row in enumerate(selected)
        ]
        truth_bytes = _jsonl_bytes(truth_rows)
        pool_bytes = _jsonl_bytes(pool_rows)
        (stage / "truth.jsonl").write_bytes(truth_bytes)
        (stage / "pool-closure.jsonl").write_bytes(pool_bytes)
        selection = {
            "schema_version": 1,
            "kind": PREPARE_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "population_accuracy_claim": False,
            "calibration_scope": "fixed_stratified_reference_bearing_controls",
            "records": TARGET_RECORDS,
            "shard_records": SHARD_RECORDS,
            "shards": shard_contracts,
            "selection_order": "old_formal_canonical_input_order",
            "selection_strategy": (
                "all_or_up_to_half_old_time_incorrect_then_stratified_device_status_rotation_size_fill"
            ),
            "counts": {
                "old_v13_correct": sum(row["old_v13_raw_exact"] for row in selected),
                "old_v13_incorrect": sum(not row["old_v13_raw_exact"] for row in selected),
            },
            "available_strata": {
                name: sorted({str(row[name]) for row in candidates})
                for name in ("device_platform", "status_class", "rotation_degrees", "size_bin")
            },
            "selected_strata": {
                name: dict(sorted(Counter(str(row[name]) for row in selected).items()))
                for name in ("device_platform", "status_class", "rotation_degrees", "size_bin")
            },
            "source_evidence": source_evidence,
            "source_score_disposition": score_disposition,
            "source_closure_sha256": _source_closure(source_files),
            "source_total_bytes": sum(int(value["size_bytes"]) for value in source_files),
            "artifacts": {
                "truth": {
                    "relative_path": "truth.jsonl", "sha256": _sha(truth_bytes),
                    "size_bytes": len(truth_bytes), "records": TARGET_RECORDS,
                },
                "pool_closure": {
                    "relative_path": "pool-closure.jsonl", "sha256": _sha(pool_bytes),
                    "size_bytes": len(pool_bytes), "records": FORMAL_RECORDS,
                },
            },
        }
        _write_json(stage / "selection.json", selection)
        selection_bytes = (stage / "selection.json").read_bytes()
        summary = {
            "schema_version": 1,
            "kind": PREPARE_SUMMARY_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "population_accuracy_claim": False,
            "calibration_scope": "fixed_stratified_reference_bearing_controls",
            "records": TARGET_RECORDS,
            "shards": 2,
            "counts": selection["counts"],
            "source_score_disposition": score_disposition,
            "artifacts": {
                "selection": {
                    "relative_path": "selection.json", "sha256": _sha(selection_bytes),
                    "size_bytes": len(selection_bytes),
                },
                **selection["artifacts"],
                "shards": shard_contracts,
            },
        }
        _write_json(stage / "summary.json", summary)
        _assert_bindings(bindings)
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite time calibration selection: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {**summary, "output_directory": str(output)}


def _validate_truth_semantics(
    *,
    truth_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    formal_sources: Sequence[str],
    source_evidence: Mapping[str, Any],
) -> None:
    records_contract = source_evidence.get("records")
    score_contract = source_evidence.get("score_comparisons")
    if not isinstance(records_contract, Mapping) or not isinstance(records_contract.get("path"), str) \
            or not isinstance(score_contract, Mapping) or not isinstance(score_contract.get("path"), str):
        raise CalibrationError("prepared records/scorer semantic bindings are missing")
    input_keys = {_path_key(source) for source in formal_sources}
    records_by_source, observed_records = _load_records(Path(records_contract["path"]), input_keys)
    if not _same_path(records_contract.get("path"), Path(str(observed_records["path"]))):
        raise CalibrationError("prepared records semantic binding path differs")
    for key in ("sha256", "size_bytes"):
        if type(observed_records.get(key)) is not type(records_contract.get(key)) \
                or observed_records.get(key) != records_contract.get(key):
            raise CalibrationError(f"prepared records semantic binding {key} differs")
    score_rows, observed_score = _load_jsonl(
        Path(str(score_contract["path"])), description="prepared scorer comparisons"
    )
    if not _same_path(score_contract.get("path"), Path(str(observed_score["path"]))):
        raise CalibrationError("prepared scorer comparison semantic binding path differs")
    for key in ("sha256", "size_bytes"):
        if type(observed_score.get(key)) is not type(score_contract.get(key)) \
                or observed_score.get(key) != score_contract.get(key):
            raise CalibrationError(f"prepared scorer comparison semantic binding {key} differs")
    score_by_source: dict[str, Mapping[str, Any]] = {}
    for score_index, row in enumerate(score_rows):
        if row.get("field") != "time":
            continue
        source = row.get("source")
        if not isinstance(source, str):
            raise CalibrationError(f"prepared scorer time row[{score_index}] has no source")
        key = _path_key(source)
        if key in score_by_source:
            raise CalibrationError(f"prepared scorer has duplicate time source: {source}")
        score_by_source[key] = row

    semantic_keys = (
        "canonical_index", "source", "reference_text", "old_v13_candidate",
        "old_v13_raw_exact", "device_platform", "status_class", "rotation_degrees",
        "size_bin", "source_width", "source_height", "record_id", "group_id",
    )
    for truth_index, truth in enumerate(truth_rows):
        canonical_index = truth["canonical_index"]
        pool = pool_rows[canonical_index]
        source = truth["source"]
        key = _path_key(source)
        if _path_key(pool.get("source")) != key or _path_key(formal_sources[canonical_index]) != key:
            raise CalibrationError(f"prepared truth row[{truth_index}] canonical source differs")
        baseline_contract = pool.get("baseline_result")
        hybrid_contract = pool.get("hybrid_result")
        if not isinstance(baseline_contract, Mapping) or not isinstance(hybrid_contract, Mapping):
            raise CalibrationError("prepared pool result identities are missing")
        baseline_payload, baseline_identity = _load_json(
            Path(str(baseline_contract.get("path"))), description="prepared baseline result"
        )
        hybrid_payload, hybrid_identity = _load_json(
            Path(str(hybrid_contract.get("path"))), description="prepared hybrid result"
        )
        for description, contract, observed in (
            ("baseline", baseline_contract, baseline_identity),
            ("hybrid", hybrid_contract, hybrid_identity),
        ):
            if not _same_path(contract.get("path"), Path(str(observed["path"]))) \
                    or any(
                        type(contract.get(identity_key)) is not type(observed[identity_key])
                        or contract.get(identity_key) != observed[identity_key]
                        for identity_key in ("sha256", "size_bytes")
                    ):
                raise CalibrationError(f"prepared {description} result identity differs")
        if _path_key(baseline_payload.get("source")) != key \
                or _path_key(hybrid_payload.get("source")) != key:
            raise CalibrationError(f"prepared truth row[{truth_index}] result source differs")
        record = records_by_source[key]
        reference = _reference_time(record)
        score = score_by_source.get(key)
        if reference is None or score is None or score.get("reference_text") != reference:
            raise CalibrationError(f"prepared truth row[{truth_index}] strict external reference differs")
        candidate = score.get("candidate_text")
        present = isinstance(candidate, str) and bool(candidate.strip())
        exact = present and candidate == reference
        if score.get("candidate_present") is not present or score.get("raw_exact") is not exact:
            raise CalibrationError(f"prepared truth row[{truth_index}] scorer semantics differ")
        baseline_candidate = TARGETED._candidate(baseline_payload, "time")
        hybrid_candidate = TARGETED._candidate(hybrid_payload, "time")
        if baseline_candidate != hybrid_candidate or candidate != hybrid_candidate:
            raise CalibrationError(f"prepared truth row[{truth_index}] formal time candidate differs")
        rotation, size_bin, width, height = _geometry_tokens(hybrid_payload)
        rebuilt = {
            "canonical_index": canonical_index,
            "source": source,
            "reference_text": reference,
            "old_v13_candidate": candidate,
            "old_v13_raw_exact": exact,
            "device_platform": _device_token(hybrid_payload),
            "status_class": _status_class(record),
            "rotation_degrees": rotation,
            "size_bin": size_bin,
            "source_width": width,
            "source_height": height,
            "record_id": record.get("id"),
            "group_id": record.get("group_id"),
        }
        if any(type(truth.get(name)) is not type(rebuilt[name]) or truth.get(name) != rebuilt[name]
               for name in semantic_keys):
            raise CalibrationError(f"prepared truth row[{truth_index}] differs from frozen source semantics")


def _validate_pool_manifest_closure(
    *,
    pool_rows: Sequence[Mapping[str, Any]],
    formal_sources: Sequence[str],
    source_evidence: Mapping[str, Any],
) -> None:
    for run_name in ("baseline", "hybrid"):
        manifest_contract = source_evidence.get(f"{run_name}_manifest")
        if not isinstance(manifest_contract, Mapping) or not isinstance(manifest_contract.get("path"), str):
            raise CalibrationError(f"prepared {run_name} manifest binding is missing")
        path = Path(str(manifest_contract["path"]))
        payload = _read_bytes(path, description=f"prepared {run_name} manifest")
        observed = _identity(path, payload)
        if not _same_path(manifest_contract.get("path"), Path(str(observed["path"]))) \
                or any(manifest_contract.get(key) != observed[key] for key in ("sha256", "size_bytes")):
            raise CalibrationError(f"prepared {run_name} manifest identity differs")
        manifest = _loads(payload.decode("utf-8-sig"), location=str(path))
        if not isinstance(manifest, list) or len(manifest) != FORMAL_RECORDS:
            raise CalibrationError(f"prepared {run_name} manifest count differs")
        result_key = f"{run_name}_result"
        for index, (row, pool, source) in enumerate(zip(manifest, pool_rows, formal_sources, strict=True)):
            if not isinstance(row, Mapping) or row.get("status") != "written" \
                    or _path_key(row.get("source")) != _path_key(source) \
                    or _path_key(pool.get("source")) != _path_key(source):
                raise CalibrationError(f"prepared {run_name} manifest row[{index}] source/status differs")
            result_contract = pool.get(result_key)
            if not isinstance(result_contract, Mapping) \
                    or not _same_path(result_contract.get("path"), Path(str(row.get("result")))):
                raise CalibrationError(f"prepared {run_name} manifest row[{index}] result binding differs")


def _revalidate_source_score_contract(
    *,
    selection: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    pool_rows: Sequence[Mapping[str, Any]],
    formal_sources: Sequence[str],
) -> None:
    """Re-run the original full five-field scorer closure from bound inputs.

    Prepared artifacts are self-describing rather than signed.  Rechecking only
    their stored hashes would let a caller rewrite a non-time comparison and
    update the prepared hashes together.  Reuse ``_load_score`` so evaluate has
    the same records/result/model/floor/disposition closure as prepare.
    """
    formal_input = source_evidence.get("formal_input_list")
    records_contract = source_evidence.get("records")
    hybrid_manifest = source_evidence.get("hybrid_manifest")
    score_summary = source_evidence.get("score_summary")
    if not all(
        isinstance(value, Mapping) and isinstance(value.get("path"), str)
        for value in (formal_input, records_contract, hybrid_manifest, score_summary)
    ):
        raise CalibrationError("prepared source scorer closure bindings are missing")

    input_keys = {_path_key(source) for source in formal_sources}
    records_by_source, observed_records = _load_records(
        Path(str(records_contract["path"])), input_keys
    )
    if not _same_path(records_contract.get("path"), Path(str(observed_records["path"]))) \
            or any(
                records_contract.get(key) != observed_records[key]
                for key in ("sha256", "size_bytes")
            ):
        raise CalibrationError("prepared source scorer records identity differs")

    hybrid_results: dict[str, dict[str, Any]] = {}
    for index, (pool, source) in enumerate(zip(pool_rows, formal_sources, strict=True)):
        result_contract = pool.get("hybrid_result")
        if not isinstance(result_contract, Mapping) or not isinstance(
            result_contract.get("path"), str
        ):
            raise CalibrationError(
                f"prepared source scorer hybrid result[{index}] binding is missing"
            )
        payload, observed = _load_json(
            Path(str(result_contract["path"])),
            description=f"prepared source scorer hybrid result[{index}]",
        )
        if not _same_path(result_contract.get("path"), Path(str(observed["path"]))) \
                or any(
                    result_contract.get(key) != observed[key]
                    for key in ("sha256", "size_bytes")
                ):
            raise CalibrationError(
                f"prepared source scorer hybrid result[{index}] identity differs"
            )
        key = _path_key(source)
        if _path_key(payload.get("source")) != key or key in hybrid_results:
            raise CalibrationError(
                f"prepared source scorer hybrid result[{index}] source differs"
            )
        hybrid_results[key] = {
            "source": source,
            "payload": payload,
            "identity": observed,
        }

    manifest_path = Path(str(hybrid_manifest["path"]))
    score_summary_path = Path(str(score_summary["path"]))
    if score_summary_path.name != "summary.json":
        raise CalibrationError("prepared source scorer summary path differs")
    formal = {
        "input_sources": list(formal_sources),
        "input_identity": dict(formal_input),
        "hybrid": {
            "root": manifest_path.parent,
            "manifest_path": manifest_path,
            "manifest_identity": dict(hybrid_manifest),
            "results": hybrid_results,
        },
    }
    _, observed_evidence, observed_disposition = _load_score(
        score_summary_path.parent,
        formal=formal,
        records_identity=observed_records,
        truth_by_source=records_by_source,
    )
    for name, observed in observed_evidence.items():
        expected = source_evidence.get(name)
        if not isinstance(expected, Mapping) \
                or not _same_path(expected.get("path"), Path(str(observed["path"]))) \
                or any(expected.get(key) != observed[key] for key in ("sha256", "size_bytes")):
            raise CalibrationError(f"prepared source scorer {name} identity differs")
    if selection.get("source_score_disposition") != observed_disposition:
        raise CalibrationError("prepared source scorer disposition differs from full closure")


def _load_prepared(prepared_directory: Path) -> dict[str, Any]:
    if prepared_directory.is_symlink():
        raise CalibrationError(f"prepared directory must not be a symbolic link: {prepared_directory}")
    root = prepared_directory.resolve(strict=True)
    selection, selection_identity = _load_json(root / "selection.json", description="calibration selection")
    summary, summary_identity = _load_json(root / "summary.json", description="calibration prepare summary")
    if selection.get("schema_version") != 1 or selection.get("kind") != PREPARE_KIND \
            or summary.get("schema_version") != 1 or summary.get("kind") != PREPARE_SUMMARY_KIND:
        raise CalibrationError("prepared calibration schema/kind is unsupported")
    for payload in (selection, summary):
        if payload.get("diagnostic_only") is not True or payload.get("formal_delivery_gate") is not False \
                or payload.get("candidate_write_enabled") is not False \
                or payload.get("population_accuracy_claim") is not False \
                or payload.get("calibration_scope") != "fixed_stratified_reference_bearing_controls":
            raise CalibrationError("prepared calibration safety flags changed")
    _require_int(selection.get("records"), TARGET_RECORDS, description="prepared records")
    _require_int(selection.get("shard_records"), SHARD_RECORDS, description="prepared shard records")
    _require_int(summary.get("records"), TARGET_RECORDS, description="prepared summary records")
    _require_int(summary.get("shards"), 2, description="prepared summary shards")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CalibrationError("prepared summary artifacts are missing")
    expected_selection = artifacts.get("selection")
    if not isinstance(expected_selection, Mapping) or expected_selection.get("sha256") != selection_identity["sha256"] \
            or expected_selection.get("size_bytes") != selection_identity["size_bytes"] \
            or expected_selection.get("relative_path") != "selection.json":
        raise CalibrationError("prepared selection identity differs from summary")
    truth_rows, truth_identity = _load_jsonl(root / "truth.jsonl", description="calibration truth")
    pool_rows, pool_identity = _load_jsonl(root / "pool-closure.jsonl", description="calibration pool closure")
    for name, identity, count in (
        ("truth", truth_identity, TARGET_RECORDS),
        ("pool_closure", pool_identity, FORMAL_RECORDS),
    ):
        contract = selection.get("artifacts", {}).get(name)
        summary_contract = artifacts.get(name)
        if not isinstance(contract, Mapping) or contract.get("sha256") != identity["sha256"] \
                or contract.get("size_bytes") != identity["size_bytes"] or contract.get("records") != count:
            raise CalibrationError(f"prepared {name} identity differs")
        if summary_contract != contract:
            raise CalibrationError(f"prepared summary {name} contract differs from selection")
    if len(truth_rows) != TARGET_RECORDS or len(pool_rows) != FORMAL_RECORDS:
        raise CalibrationError("prepared truth/pool row count differs")
    canonical = [row.get("canonical_index") for row in truth_rows]
    if any(type(value) is not int or value < 0 or value >= FORMAL_RECORDS for value in canonical) \
            or canonical != sorted(canonical) or len(set(canonical)) != TARGET_RECORDS:
        raise CalibrationError("prepared truth is not in unique canonical order")
    shards = selection.get("shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise CalibrationError("prepared selection must contain two shards")
    if artifacts.get("shards") != shards:
        raise CalibrationError("prepared summary shard contracts differ from selection")
    shard_sources: list[list[str]] = []
    for index, contract in enumerate(shards):
        path = root / f"shard-{index}-inputs.txt"
        payload = _read_bytes(path, description=f"calibration shard {index} input")
        if not isinstance(contract, Mapping) or contract.get("index") != index \
                or contract.get("relative_path") != path.name \
                or contract.get("records") != SHARD_RECORDS \
                or contract.get("sha256") != _sha(payload) \
                or contract.get("size_bytes") != len(payload):
            raise CalibrationError(f"prepared shard {index} contract differs")
        sources = _parse_input_payload(
            payload, records=SHARD_RECORDS, description=f"calibration shard {index} input set"
        )
        shard_truth = truth_rows[index * SHARD_RECORDS:(index + 1) * SHARD_RECORDS]
        if contract.get("canonical_index_min") != shard_truth[0].get("canonical_index") \
                or contract.get("canonical_index_max") != shard_truth[-1].get("canonical_index"):
            raise CalibrationError(f"prepared shard {index} canonical bounds differ")
        shard_sources.append(sources)
    combined = [source for shard in shard_sources for source in shard]
    if combined != [row.get("source") for row in truth_rows] or len({_path_key(source) for source in combined}) != TARGET_RECORDS:
        raise CalibrationError("prepared shards omit/duplicate/reorder truth sources")
    source_evidence = selection.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise CalibrationError("prepared source evidence is missing")
    if summary.get("source_score_disposition") != selection.get("source_score_disposition"):
        raise CalibrationError("prepared summary source scorer disposition differs")
    bindings = [dict(value) for value in source_evidence.values() if isinstance(value, Mapping)]
    pool_sources: list[str] = []
    for pool_index, row in enumerate(pool_rows):
        if row.get("schema_version") != 1 or row.get("kind") != POOL_KIND \
                or row.get("canonical_index") != pool_index or not isinstance(row.get("source"), str):
            raise CalibrationError("prepared pool row schema/order changed")
        pool_sources.append(str(row["source"]))
        if not isinstance(row.get("baseline_result"), Mapping) \
                or not isinstance(row.get("hybrid_result"), Mapping):
            raise CalibrationError("prepared pool result identity is missing")
        bindings.extend([dict(row["baseline_result"]), dict(row["hybrid_result"])])
    formal_input = source_evidence.get("formal_input_list")
    if not isinstance(formal_input, Mapping) or not isinstance(formal_input.get("path"), str):
        raise CalibrationError("prepared formal input-list evidence is missing")
    formal_payload = _read_bytes(Path(str(formal_input["path"])), description="prepared formal input list")
    formal_sources = _parse_input_payload(
        formal_payload, records=FORMAL_RECORDS, description="prepared formal input set"
    )
    if [_path_key(value) for value in pool_sources] != [_path_key(value) for value in formal_sources]:
        raise CalibrationError("prepared pool sources differ from the frozen formal input order")
    _validate_pool_manifest_closure(
        pool_rows=pool_rows,
        formal_sources=formal_sources,
        source_evidence=source_evidence,
    )
    _revalidate_source_score_contract(
        selection=selection,
        source_evidence=source_evidence,
        pool_rows=pool_rows,
        formal_sources=formal_sources,
    )
    truth_source_identities: list[dict[str, Any]] = []
    for row_index, row in enumerate(truth_rows):
        if row.get("schema_version") != 1 or row.get("kind") != TRUTH_KIND \
                or row.get("diagnostic_only") is not True \
                or row.get("formal_delivery_gate") is not False \
                or row.get("candidate_write_enabled") is not False \
                or not isinstance(row.get("source_file"), Mapping):
            raise CalibrationError("prepared truth row schema changed")
        source = row.get("source")
        reference = row.get("reference_text")
        if not isinstance(source, str) or not isinstance(reference, str) \
                or LAYOUT_EVIDENCE._clock_value(reference) != reference.replace("：", ":").strip():
            raise CalibrationError(f"prepared truth row[{row_index}] source/reference is invalid")
        source_file = row["source_file"]
        if not _same_path(source_file.get("path"), Path(source)):
            raise CalibrationError(f"prepared truth row[{row_index}] source identity path differs")
        if type(row.get("old_v13_raw_exact")) is not bool \
                or row.get("status_class") not in {"success", "failed", "pending", "unknown"} \
                or not isinstance(row.get("device_platform"), str) or not row.get("device_platform") \
                or row.get("rotation_degrees") not in {0, 90} \
                or row.get("size_bin") not in {"lt1200", "1200_1999", "2000_2999", "ge3000"}:
            raise CalibrationError(f"prepared truth row[{row_index}] strata contract differs")
        if _require_int(row.get("source_width"), description="prepared truth source width") < 2 \
                or _require_int(row.get("source_height"), description="prepared truth source height") < 2:
            raise CalibrationError(f"prepared truth row[{row_index}] source dimensions are invalid")
        source_identity = dict(row["source_file"])
        truth_source_identities.append(source_identity)
        bindings.append(source_identity)
    if selection.get("source_closure_sha256") != _source_closure(truth_source_identities) \
            or selection.get("source_total_bytes") != sum(
                int(value["size_bytes"]) for value in truth_source_identities
            ):
        raise CalibrationError("prepared selected source closure differs")
    _validate_truth_semantics(
        truth_rows=truth_rows,
        pool_rows=pool_rows,
        formal_sources=formal_sources,
        source_evidence=source_evidence,
    )
    expected_counts = {
        "old_v13_correct": sum(row["old_v13_raw_exact"] is True for row in truth_rows),
        "old_v13_incorrect": sum(row["old_v13_raw_exact"] is False for row in truth_rows),
    }
    if selection.get("counts") != expected_counts or summary.get("counts") != expected_counts \
            or min(expected_counts.values()) <= 0:
        raise CalibrationError("prepared old-v13 correctness counts differ")
    expected_selected_strata = {
        name: dict(sorted(Counter(str(row[name]) for row in truth_rows).items()))
        for name in ("device_platform", "status_class", "rotation_degrees", "size_bin")
    }
    if selection.get("selected_strata") != expected_selected_strata:
        raise CalibrationError("prepared selected strata counts differ")
    expected_available_strata = {
        name: sorted(expected_selected_strata[name]) for name in expected_selected_strata
    }
    if selection.get("available_strata") != expected_available_strata:
        raise CalibrationError("prepared available strata differ from the hard-covered selection")
    prepared_artifacts = {
        "selection": selection_identity,
        "summary": summary_identity,
        "truth": truth_identity,
        "pool_closure": pool_identity,
        "shard_inputs": [_identity(root / f"shard-{index}-inputs.txt") for index in range(2)],
    }
    bindings.extend([
        prepared_artifacts["selection"], prepared_artifacts["summary"],
        prepared_artifacts["truth"], prepared_artifacts["pool_closure"],
        *prepared_artifacts["shard_inputs"],
    ])
    _assert_bindings(bindings)
    return {
        "root": root,
        "selection": selection,
        "truth": truth_rows,
        "shard_sources": shard_sources,
        "bindings": bindings,
        "prepared_artifacts": prepared_artifacts,
    }


def _require_sha256(value: object, *, description: str) -> str:
    try:
        return LAYOUT_EVIDENCE._require_sha(value, description=description)
    except LAYOUT_EVIDENCE.EvidenceError as error:
        raise CalibrationError(str(error)) from error


def _validate_paddle_bundle(bundle: object, *, shard_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(bundle, Mapping):
        raise CalibrationError(f"layout shard {shard_index} Paddle bundle evidence is missing")
    directory_raw = bundle.get("directory")
    contract_raw = bundle.get("contract_path")
    if not isinstance(directory_raw, str) or not isinstance(contract_raw, str):
        raise CalibrationError(f"layout shard {shard_index} Paddle bundle paths are missing")
    directory_path = Path(directory_raw)
    if directory_path.is_symlink():
        raise CalibrationError(f"layout shard {shard_index} Paddle bundle must not be a symlink")
    directory = directory_path.resolve(strict=True)
    if not directory.is_dir():
        raise CalibrationError(f"layout shard {shard_index} Paddle bundle is not a directory")
    contract_path = Path(contract_raw)
    if contract_path.is_symlink():
        raise CalibrationError(f"layout shard {shard_index} Paddle contract must not be a symlink")
    contract = _identity(contract_path)
    try:
        contract_path.resolve(strict=True).relative_to(directory)
    except ValueError as error:
        raise CalibrationError(f"layout shard {shard_index} Paddle contract escapes the bundle") from error
    if contract_path.name != "paddle_ocr_delivery.contract.json" \
            or bundle.get("contract_sha256") != contract["sha256"]:
        raise CalibrationError(f"layout shard {shard_index} Paddle contract identity differs")
    _require_sha256(
        bundle.get("source_audit_contract_sha256"),
        description=f"layout shard {shard_index} source audit contract SHA-256",
    )
    identities: list[dict[str, Any]] = [contract]
    components: dict[str, Any] = {}
    component_paths: set[Path] = set()
    package_size = 0
    for role in ("detector", "classifier", "recognizer", "dictionary"):
        value = bundle.get(role)
        if not isinstance(value, Mapping):
            raise CalibrationError(f"layout shard {shard_index} Paddle {role} evidence is missing")
        try:
            relative = TARGETED._safe_relative_path(
                value.get("relative_path"), description=f"layout shard {shard_index} Paddle {role} path"
            )
        except TARGETED.ReplayError as error:
            raise CalibrationError(str(error)) from error
        path = directory / Path(relative)
        identity = _identity(path)
        resolved_component = Path(str(identity["path"])).resolve(strict=True)
        try:
            resolved_component.relative_to(directory)
        except ValueError as error:
            raise CalibrationError(f"layout shard {shard_index} Paddle {role} escapes the bundle") from error
        if resolved_component in component_paths:
            raise CalibrationError(f"layout shard {shard_index} Paddle components are not unique")
        component_paths.add(resolved_component)
        if value.get("sha256") != identity["sha256"] \
                or type(value.get("size_bytes")) is not int \
                or value.get("size_bytes") != identity["size_bytes"]:
            raise CalibrationError(f"layout shard {shard_index} Paddle {role} identity differs")
        identities.append(identity)
        package_size += int(identity["size_bytes"])
        components[role] = {"relative_path": relative, **identity}
    if type(bundle.get("package_size_bytes")) is not int \
            or bundle.get("package_size_bytes") != package_size:
        raise CalibrationError(f"layout shard {shard_index} Paddle package size differs")
    return {
        "directory": str(directory),
        "contract": contract,
        "source_audit_contract_sha256": bundle["source_audit_contract_sha256"],
        "package_size_bytes": package_size,
        "components": components,
    }, identities


def _latency_distribution(values: Sequence[float]) -> dict[str, Any]:
    return _latency(values)


def _validate_latency_summary(
    contract: object,
    values: Mapping[str, Sequence[float]],
    *,
    shard_index: int,
) -> None:
    if not isinstance(contract, Mapping) or set(contract) != set(values):
        raise CalibrationError(f"layout shard {shard_index} latency summary stages differ")
    for stage, stage_values in values.items():
        observed = contract.get(stage)
        expected = _latency_distribution(stage_values)
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise CalibrationError(f"layout shard {shard_index} latency {stage} shape differs")
        for key, expected_value in expected.items():
            actual = observed.get(key)
            if key == "count":
                if type(actual) is not int or actual != expected_value:
                    raise CalibrationError(f"layout shard {shard_index} latency {stage}.{key} differs")
            elif isinstance(actual, bool) or not isinstance(actual, (int, float)) \
                    or not math.isfinite(float(actual)) \
                    or not math.isclose(float(actual), float(expected_value), rel_tol=0, abs_tol=0.00011):
                raise CalibrationError(f"layout shard {shard_index} latency {stage}.{key} differs")


def _load_layout_shard(
    directory: Path,
    *,
    shard_index: int,
    expected_sources: Sequence[str],
    expected_input_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[float]]:
    if directory.is_symlink():
        raise CalibrationError(f"layout shard {shard_index} directory must not be a symbolic link")
    root = directory.resolve(strict=True)
    summary, summary_identity = _load_json(root / "summary.json", description=f"layout shard {shard_index} summary")
    records, records_identity = _load_jsonl(root / "records.jsonl", description=f"layout shard {shard_index} records")
    if summary.get("schema_version") != 1 or summary.get("kind") != LAYOUT_SUMMARY_KIND:
        raise CalibrationError(f"layout shard {shard_index} summary schema/kind differs")
    for key, expected in {
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "expected_records": SHARD_RECORDS,
        "records": SHARD_RECORDS,
        "errors": 0,
        "execution_provider": "cpu",
        "rectification": LAYOUT_EVIDENCE.RECTIFICATION,
        "quad_coordinate_space": LAYOUT_EVIDENCE.QUAD_COORDINATE_SPACE,
        "quad_normalization": LAYOUT_EVIDENCE.QUAD_NORMALIZATION,
        "confidence_semantics": LAYOUT_EVIDENCE.CONFIDENCE_SEMANTICS,
    }.items():
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            raise CalibrationError(f"layout shard {shard_index} {key} must be {expected!r}")
    input_contract = summary.get("input_list")
    if not isinstance(input_contract, Mapping) or input_contract.get("records") != SHARD_RECORDS \
            or input_contract.get("sha256") != expected_input_identity["sha256"] \
            or input_contract.get("size_bytes") != expected_input_identity["size_bytes"] \
            or not _same_path(input_contract.get("path"), Path(str(expected_input_identity["path"]))):
        raise CalibrationError(f"layout shard {shard_index} input binding differs")
    artifact = summary.get("artifacts", {}).get("records_jsonl") \
        if isinstance(summary.get("artifacts"), Mapping) else None
    if not isinstance(artifact, Mapping) or artifact.get("relative_path") != "records.jsonl" \
            or artifact.get("sha256") != records_identity["sha256"] \
            or artifact.get("size_bytes") != records_identity["size_bytes"]:
        raise CalibrationError(f"layout shard {shard_index} records identity differs")
    if len(records) != SHARD_RECORDS:
        raise CalibrationError(f"layout shard {shard_index} record count differs")
    drop_score = _require_number(summary.get("paddle_drop_score"), description="Paddle drop score", minimum=0)
    if drop_score > 1:
        raise CalibrationError("Paddle drop score must be at most 1")
    bundle_evidence, bundle_bindings = _validate_paddle_bundle(
        summary.get("paddle_bundle"), shard_index=shard_index
    )
    analyzed: list[dict[str, Any]] = []
    invalid_quad_contract_lines: list[dict[str, Any]] = []
    timing_by_stage: dict[str, list[float]] = {
        "image_load": [], "rectification": [], "layout_ocr": [], "total": [],
    }
    for index, (record, source) in enumerate(zip(records, expected_sources, strict=True)):
        if record.get("schema_version") != 1 or record.get("kind") != LAYOUT_RECORD_KIND \
                or record.get("index") != index or record.get("execution_provider") != "cpu" \
                or record.get("diagnostic_only") is not True \
                or record.get("formal_delivery_gate") is not False \
                or record.get("candidate_write_enabled") is not False \
                or record.get("quad_coordinate_space") != LAYOUT_EVIDENCE.QUAD_COORDINATE_SPACE \
                or record.get("quad_normalization") != LAYOUT_EVIDENCE.QUAD_NORMALIZATION \
                or record.get("confidence_semantics") != LAYOUT_EVIDENCE.CONFIDENCE_SEMANTICS \
                or _path_key(record.get("source")) != _path_key(source):
            raise CalibrationError(f"layout shard {shard_index} record[{index}] contract/order differs")
        source_identity = _identity(Path(source))
        if record.get("source_image_sha256") != source_identity["sha256"] \
                or record.get("source_image_size_bytes") != source_identity["size_bytes"]:
            raise CalibrationError(f"layout shard {shard_index} record[{index}] source identity differs")
        geometry = record.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("rectification") != LAYOUT_EVIDENCE.RECTIFICATION:
            raise CalibrationError(f"layout shard {shard_index} record[{index}] geometry differs")
        source_size = geometry.get("source_size")
        rectified_size = geometry.get("rectified_size")
        if not isinstance(source_size, Mapping) or not isinstance(rectified_size, Mapping):
            raise CalibrationError("layout geometry sizes are missing")
        sw = _require_int(source_size.get("width"), description="layout source width")
        sh = _require_int(source_size.get("height"), description="layout source height")
        rw = _require_int(rectified_size.get("width"), description="layout rectified width")
        rh = _require_int(rectified_size.get("height"), description="layout rectified height")
        rotation = _require_int(geometry.get("rotation_degrees"), description="layout rotation")
        if sw < 2 or sh < 2 or rw < 2 or rh < 2 or max(rw, rh) > 1600 \
                or rotation not in {0, 90} or geometry.get("screen_detected") is not False:
            raise CalibrationError(f"layout shard {shard_index} record[{index}] geometry contract differs")
        forward = LAYOUT_EVIDENCE._matrix3(
            geometry.get("H_original_to_rectified"), description="H_original_to_rectified"
        )
        inverse = LAYOUT_EVIDENCE._matrix3(
            geometry.get("H_rectified_to_original"), description="H_rectified_to_original"
        )
        LAYOUT_EVIDENCE._require_homography_pair(
            forward, inverse,
            source_width=sw, source_height=sh, rectified_width=rw, rectified_height=rh,
            rotation_degrees=rotation, description=f"layout shard {shard_index} record[{index}]",
        )
        screen_quad = geometry.get("screen_quad_original")
        expected_screen_quad = [
            [0.0, 0.0], [float(sw - 1), 0.0],
            [float(sw - 1), float(sh - 1)], [0.0, float(sh - 1)],
        ]
        if not isinstance(screen_quad, list) or len(screen_quad) != 4:
            raise CalibrationError(f"layout shard {shard_index} record[{index}] screen quad differs")
        for point, expected_point in zip(screen_quad, expected_screen_quad, strict=True):
            if not isinstance(point, list) or len(point) != 2 or any(
                not math.isclose(
                    _require_number(value, description="screen quad coordinate"),
                    expected_value, rel_tol=0, abs_tol=1e-5,
                )
                for value, expected_value in zip(point, expected_point, strict=True)
            ):
                raise CalibrationError(f"layout shard {shard_index} record[{index}] screen quad differs")
        lines = record.get("lines")
        if not isinstance(lines, list) or record.get("raw_line_count") != len(lines):
            raise CalibrationError(f"layout shard {shard_index} record[{index}] lines differ")
        prepared_lines: list[dict[str, Any]] = []
        record_invalid_quad_contract_lines: list[dict[str, Any]] = []
        for line_index, line in enumerate(lines):
            if not isinstance(line, dict) or line.get("index") != line_index \
                    or not isinstance(line.get("text"), str):
                raise CalibrationError("layout line schema differs")
            confidence = _require_number(line.get("confidence"), description="line confidence", minimum=0)
            if confidence > 1 or line.get("passes_drop_score") is not (confidence >= drop_score):
                raise CalibrationError("layout line confidence/drop-score differs")
            item = dict(line)
            item["_geometry"] = LAYOUT_EVIDENCE._quad_geometry(
                line, record_index=index, line_index=line_index,
                rectified_width=rw, rectified_height=rh,
                source_width=sw, source_height=sh,
                rectified_to_source=inverse,
                allow_invalid_quad_contract_violation=True,
            )
            item["_rotation_degrees"] = rotation
            prepared_lines.append(item)
            invalid_quad = item["_geometry"].get("degenerate_quad")
            if isinstance(invalid_quad, Mapping):
                diagnostic = {
                    "shard_index": shard_index,
                    "record_index": index,
                    "line_index": line_index,
                    "source": source,
                    "text": line["text"],
                    "confidence": confidence,
                    "passes_drop_score": line["passes_drop_score"],
                    "quad_rectified": line.get("quad_rectified"),
                    "quad_rectified_normalized": line.get("quad_rectified_normalized"),
                    **dict(invalid_quad),
                }
                record_invalid_quad_contract_lines.append(diagnostic)
                invalid_quad_contract_lines.append(diagnostic)
        accepted = [line for line in prepared_lines if line["passes_drop_score"]]
        if record.get("accepted_line_count") != len(accepted):
            raise CalibrationError("layout accepted_line_count differs")
        expected_text = " ".join(
            value for value in (LAYOUT_EVIDENCE._clean_text(line["text"]) for line in accepted) if value
        )
        if record.get("accepted_text") != expected_text:
            raise CalibrationError("layout accepted_text projection differs")
        accepted_confidence = record.get("accepted_confidence")
        if not accepted:
            if accepted_confidence is not None:
                raise CalibrationError("layout accepted_confidence must be null")
        else:
            observed_confidence = _require_number(
                accepted_confidence, description="layout accepted confidence", minimum=0
            )
            expected_confidence = statistics.fmean(line["confidence"] for line in accepted)
            if observed_confidence > 1 or not math.isclose(
                observed_confidence, expected_confidence, rel_tol=0, abs_tol=2e-6
            ):
                raise CalibrationError("layout accepted_confidence projection differs")
        if record_invalid_quad_contract_lines:
            # A malformed DB quadrilateral makes the producer's line geometry
            # contract unreliable for the complete receipt.  Keep the receipt
            # in the calibration denominator, but do not derive a candidate
            # from any other line in that record.
            time_evidence = LAYOUT_EVIDENCE._time_evidence(())
            time_evidence["ambiguity"] = "invalid_quad_contract_record_quarantined"
            time_evidence["record_candidate_eligible"] = False
            time_evidence["invalid_quad_contract_lines"] = record_invalid_quad_contract_lines
            candidate = None
        else:
            time_evidence = LAYOUT_EVIDENCE._time_evidence(prepared_lines)
            time_evidence["record_candidate_eligible"] = True
            time_evidence["invalid_quad_contract_lines"] = []
            strict_anchors = [
                anchor for anchor in time_evidence["anchors"]
                if anchor["status_bar_geometry_evidence"] and anchor["passes_drop_score"]
            ]
            candidate = strict_anchors[0]["visible_clock"] \
                if time_evidence["unique_diagnostic_coverage"] and len(strict_anchors) == 1 else None
        analyzed.append({
            "source": source,
            "source_image_sha256": source_identity["sha256"],
            "candidate": candidate,
            "time_evidence": time_evidence,
            "rotation_degrees": rotation,
            "source_width": sw,
            "source_height": sh,
            "invalid_quad_contract_lines": record_invalid_quad_contract_lines,
        })
        timing = record.get("timing_ms")
        if not isinstance(timing, Mapping):
            raise CalibrationError("layout timing is missing")
        for stage in timing_by_stage:
            timing_by_stage[stage].append(
                _require_number(timing.get(stage), description=f"layout {stage} timing", minimum=0)
            )
        analyzed[-1]["cpu_latency_ms"] = timing_by_stage["total"][-1]
    _validate_latency_summary(
        summary.get("latency_ms"), timing_by_stage, shard_index=shard_index
    )
    invalid_classifications = Counter(
        str(item["classification"]) for item in invalid_quad_contract_lines
    )
    return analyzed, {
        "summary": summary_identity,
        "records": records_identity,
        "input_list": dict(expected_input_identity),
        "paddle_bundle": bundle_evidence,
        "paddle_drop_score": drop_score,
        "bound_files": bundle_bindings,
        "cpu_latency_ms": _latency(timing_by_stage["total"]),
        "layout_geometry_safety": {
            "invalid_quad_contract_violation_lines": len(invalid_quad_contract_lines),
            "records_forced_candidate_ineligible": len({
                (int(item["shard_index"]), int(item["record_index"]))
                for item in invalid_quad_contract_lines
            }),
            "by_classification": dict(sorted(invalid_classifications.items())),
            "invalid_quad_geometry_used": False,
            "invalid_quad_canonicalized": 0,
            "contract_violation_policy": "fail_closed_whole_record_unresolved",
        },
    }, timing_by_stage["total"]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = len(rows)
    covered = sum(row["candidate_present"] for row in rows)
    exact = sum(row["raw_exact"] for row in rows)
    old_exact = sum(row["old_v13_raw_exact"] for row in rows)
    metrics = {
        "records": records,
        "candidate_records": covered,
        "candidate_coverage": covered / records if records else None,
        "raw_exact_matches": exact,
        "raw_exact_accuracy": exact / records if records else None,
        "old_v13_raw_exact_matches": old_exact,
        "old_v13_raw_exact_accuracy": old_exact / records if records else None,
        "raw_exact_accuracy_delta": (exact - old_exact) / records if records else None,
        "correct_to_wrong": sum(row["correct_to_wrong"] for row in rows),
        "wrong_to_correct": sum(row["wrong_to_correct"] for row in rows),
    }
    latency_values = [float(row["cpu_latency_ms"]) for row in rows if "cpu_latency_ms" in row]
    if latency_values:
        metrics["cpu_latency_ms"] = _latency(latency_values)
    return metrics


def _latency(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "p50": round(LAYOUT_EVIDENCE._percentile(ordered, 0.50), 4),
        "p95": round(LAYOUT_EVIDENCE._percentile(ordered, 0.95), 4),
        "p99": round(LAYOUT_EVIDENCE._percentile(ordered, 0.99), 4),
        "max": round(ordered[-1], 4),
    }


def evaluate(
    *,
    prepared_directory: Path,
    layout_shard_0: Path,
    layout_shard_1: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.absolute()
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite time calibration evaluation: {output}")
    try:
        resolved_layouts = [layout_shard_0.resolve(strict=True), layout_shard_1.resolve(strict=True)]
    except FileNotFoundError as error:
        raise CalibrationError("one or both layout shard directories do not exist") from error
    if resolved_layouts[0] == resolved_layouts[1]:
        raise CalibrationError("layout shard directories must be distinct fresh outputs")
    for input_path in (prepared_directory, layout_shard_0, layout_shard_1):
        if _paths_overlap(output, input_path):
            raise CalibrationError("evaluation output must not overlap a frozen input directory")
    prepared = _load_prepared(prepared_directory)
    layouts: list[dict[str, Any]] = []
    layout_evidence: list[dict[str, Any]] = []
    timings: list[float] = []
    for index, directory in enumerate((layout_shard_0, layout_shard_1)):
        rows, evidence, shard_timings = _load_layout_shard(
            directory,
            shard_index=index,
            expected_sources=prepared["shard_sources"][index],
            expected_input_identity=prepared["prepared_artifacts"]["shard_inputs"][index],
        )
        layouts.extend(rows)
        layout_evidence.append(evidence)
        timings.extend(shard_timings)
    if len(layouts) != TARGET_RECORDS or len({_path_key(row["source"]) for row in layouts}) != TARGET_RECORDS:
        raise CalibrationError("combined layout shards omit or duplicate records")
    if layout_evidence[0]["paddle_drop_score"] != layout_evidence[1]["paddle_drop_score"] \
            or layout_evidence[0]["paddle_bundle"] != layout_evidence[1]["paddle_bundle"]:
        raise CalibrationError("layout shards used different Paddle OCR bundle/drop-score contracts")
    if [row["source"] for row in layouts] != [row["source"] for row in prepared["truth"]]:
        raise CalibrationError("combined layout shard order differs from frozen truth")
    comparisons: list[dict[str, Any]] = []
    for index, (truth, observed) in enumerate(zip(prepared["truth"], layouts, strict=True)):
        if observed["source_width"] != truth["source_width"] \
                or observed["source_height"] != truth["source_height"]:
            raise CalibrationError(f"layout source dimensions differ from frozen formal truth: {truth['source']}")
        candidate = observed["candidate"]
        reference = truth["reference_text"]
        exact = candidate == reference if candidate is not None else False
        old_exact = truth["old_v13_raw_exact"] is True
        invalid_quad_lines = observed["invalid_quad_contract_lines"]
        comparisons.append({
            "schema_version": 1,
            "kind": COMPARISON_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "index": index,
            "canonical_index": truth["canonical_index"],
            "source": truth["source"],
            "reference_text": reference,
            "candidate_text": candidate,
            "candidate_present": candidate is not None,
            "raw_exact": exact,
            "old_v13_candidate": truth["old_v13_candidate"],
            "old_v13_raw_exact": old_exact,
            "correct_to_wrong": old_exact and not exact,
            "wrong_to_correct": not old_exact and exact,
            "device_platform": truth["device_platform"],
            "status_class": truth["status_class"],
            "formal_rotation_degrees": truth["rotation_degrees"],
            "rotation_degrees": observed["rotation_degrees"],
            "rotation_changed": observed["rotation_degrees"] != truth["rotation_degrees"],
            "size_bin": truth["size_bin"],
            "route_ambiguity": observed["time_evidence"]["ambiguity"],
            "layout_record_quarantined": bool(invalid_quad_lines),
            "invalid_quad_contract_lines": invalid_quad_lines,
            "cpu_latency_ms": observed["cpu_latency_ms"],
        })
    grouped: dict[str, Any] = {}
    for field in (
        "device_platform", "status_class", "formal_rotation_degrees", "rotation_degrees",
        "size_bin", "old_v13_raw_exact",
    ):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in comparisons:
            values[str(row[field]).lower()].append(row)
        grouped[field] = {key: _metrics(rows) for key, rows in sorted(values.items())}
    comparison_bytes = _jsonl_bytes(comparisons)
    errors = [row for row in comparisons if not row["raw_exact"]]
    quarantined_records = sum(
        row["layout_record_quarantined"] for row in comparisons
    )
    invalid_classifications = Counter(
        str(item["classification"])
        for row in comparisons
        for item in row["invalid_quad_contract_lines"]
    )
    summary = {
        "schema_version": 1,
        "kind": EVALUATE_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "population_accuracy_claim": False,
        "calibration_scope": "fixed_stratified_reference_bearing_controls",
        "records": TARGET_RECORDS,
        "errors": 0,
        "execution_provider": "cpu",
        "truth_semantics": "external_records_visible_status_bar_h_mm_or_hh_mm_raw_exact",
        "truth_provenance": "frozen_unified_records_external_to_layout_route_not_independent_human_truth",
        "source_score_disposition": prepared["selection"]["source_score_disposition"],
        "route_semantics": "shared_receipt_mlnet_layout_shadow_evidence_strict_time_route",
        "overall": _metrics(comparisons),
        "grouped": grouped,
        "route_ambiguity_counts": dict(sorted(Counter(row["route_ambiguity"] for row in comparisons).items())),
        "cpu_latency_ms": _latency(timings),
        "cpu_latency_ms_by_shard": [evidence["cpu_latency_ms"] for evidence in layout_evidence],
        "rotation_changed_records": sum(row["rotation_changed"] for row in comparisons),
        "layout_geometry_safety": {
            "invalid_quad_contract_violation_lines": sum(
                len(row["invalid_quad_contract_lines"]) for row in comparisons
            ),
            "records_forced_candidate_ineligible": quarantined_records,
            "by_classification": dict(sorted(invalid_classifications.items())),
            "invalid_quad_geometry_used": False,
            "invalid_quad_canonicalized": 0,
            "contract_violation_policy": "fail_closed_whole_record_unresolved",
            "quarantined_records_remain_in_accuracy_denominator": True,
        },
        "error_examples": [
            {
                key: row[key]
                for key in (
                    "source", "reference_text", "candidate_text", "old_v13_candidate",
                    "old_v13_raw_exact", "device_platform", "status_class",
                    "formal_rotation_degrees", "rotation_degrees", "rotation_changed",
                    "size_bin", "route_ambiguity",
                )
            }
            for row in errors[:100]
        ],
        "status_safety": {"field_candidate_writes": 0, "non_success_to_success": 0},
        "input_evidence": {
            "prepared": prepared["prepared_artifacts"],
            "layout_shards": layout_evidence,
        },
        "artifacts": {
            "comparisons": {
                "relative_path": "comparisons.jsonl",
                "sha256": _sha(comparison_bytes),
                "size_bytes": len(comparison_bytes),
                "records": TARGET_RECORDS,
            }
        },
    }
    prepared_bindings = prepared["bindings"]
    _assert_bindings(prepared_bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        (stage / "comparisons.jsonl").write_bytes(comparison_bytes)
        _write_json(stage / "summary.json", summary)
        _assert_bindings(prepared_bindings)
        for evidence in layout_evidence:
            _assert_identity(evidence["summary"], description="layout shard summary")
            _assert_identity(evidence["records"], description="layout shard records")
            for identity in evidence["bound_files"]:
                _assert_identity(identity, description="layout shard Paddle bundle file")
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite time calibration evaluation: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {**summary, "output_directory": str(output)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--formal-root", required=True, type=Path)
    records_group = prepare_parser.add_mutually_exclusive_group(required=True)
    records_group.add_argument("--records", type=Path)
    records_group.add_argument(
        "--records-from-score",
        action="store_true",
        help="use the absolute records path hash-bound by --score-directory/summary.json",
    )
    prepare_parser.add_argument("--score-directory", required=True, type=Path)
    prepare_parser.add_argument("--output-directory", required=True, type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--prepared-directory", required=True, type=Path)
    evaluate_parser.add_argument("--layout-shard-0", required=True, type=Path)
    evaluate_parser.add_argument("--layout-shard-1", required=True, type=Path)
    evaluate_parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            records_path = (
                _records_path_from_score(args.score_directory)
                if args.records_from_score else args.records
            )
            assert records_path is not None
            result = prepare(
                formal_root=args.formal_root,
                records_path=records_path,
                score_directory=args.score_directory,
                output_directory=args.output_directory,
            )
            print(
                f"time_calibration_prepare records={result['records']} shards={result['shards']} "
                f"output={result['output_directory']}"
            )
        else:
            result = evaluate(
                prepared_directory=args.prepared_directory,
                layout_shard_0=args.layout_shard_0,
                layout_shard_1=args.layout_shard_1,
                output_directory=args.output_directory,
            )
            print(
                f"time_calibration_evaluate records={result['records']} "
                f"coverage={result['overall']['candidate_coverage']:.6f} "
                f"accuracy={result['overall']['raw_exact_accuracy']:.6f} "
                f"output={result['output_directory']}"
            )
    except (CalibrationError, TARGETED.ReplayError, LAYOUT_EVIDENCE.EvidenceError,
            FileExistsError, OSError, UnicodeError) as error:
        print(f"Time layout calibration failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
