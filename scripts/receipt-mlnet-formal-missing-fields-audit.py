#!/usr/bin/env python3
"""Audit all missing formal CPU candidates from frozen scorer/A/B evidence.

This helper never runs OCR and never mutates inference or scorer artifacts.  It
binds the scorer summary/comparisons to the formal input list, records manifest,
A/B manifests, and every baseline/hybrid result JSON; reconstructs all-receipt
candidate coverage; and atomically publishes a diagnostic-only report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transfer_receipt_ai.ocr_unified_targets import parse_amount_visible_format_target


FORMAL_RECORDS = 10016
SCORE_KIND = "receipt_mlnet_unified_candidate_evaluation_v1"
AB_KIND = "receipt_mlnet_hybrid_recipient_cpu_ab_v1"
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
FIELD_SPECS = {
    "amount": ("amount", "amount"),
    "time": ("time", "time"),
    "payment_method_field": ("payment_method", "payment_method_field"),
    "recipient_field": ("recipient", "recipient_field"),
    "transfer_status": ("transfer_status", "transfer_status"),
}
FIELD_ALIASES = {
    "amount": "amount",
    "time": "time",
    "payment": "payment_method_field",
    "payment_method_field": "payment_method_field",
    "recipient": "recipient_field",
    "recipient_field": "recipient_field",
    "status": "transfer_status",
    "transfer_status": "transfer_status",
}
FIXED_FLOORS = {
    "amount": 0.7885,
    "time": 0.9840,
    "payment_method_field": 0.9325,
    "recipient_field": 0.90,
    "transfer_status": 0.90,
}
FULL_SELECTION_ORDER = "first_unique_source_in_records_manifest_order"
SUMMARY_KIND = "receipt_mlnet_formal_missing_fields_audit_summary_v1"
FINDING_KIND = "receipt_mlnet_formal_missing_fields_audit_finding_v1"


class AuditError(ValueError):
    """Raised when frozen formal evidence is incomplete or inconsistent."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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


def _load_json(path: Path, *, description: str) -> Any:
    if not path.is_file():
        raise AuditError(f"missing {description}: {path}")
    return _loads(path.read_text(encoding="utf-8-sig"), location=str(path))


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"missing {description}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise AuditError(f"invalid {description} {path}:{line_number}: blank line")
            value = _loads(line, location=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise AuditError(
                    f"invalid {description} {path}:{line_number}: expected an object"
                )
            rows.append(value)
    if not rows:
        raise AuditError(f"{description} is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise AuditError(f"{description} is not a file: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _assert_identity(
    identity: Mapping[str, Any], *, description: str, expected_path: Path | None = None
) -> None:
    path = Path(str(identity.get("path", "")))
    if expected_path is not None and path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise AuditError(f"{description} path disagrees with the selected artifact")
    observed = _file_identity(path, description=description)
    for key in ("sha256", "size_bytes"):
        if identity.get(key) != observed[key]:
            raise AuditError(f"{description} {key} mismatch")


def _source_key(value: object) -> str:
    raw = str(value or "")
    if WINDOWS_ABSOLUTE_PATH.match(raw) or "\\" in raw:
        return "windows:" + ntpath.normcase(ntpath.normpath(raw)).replace("\\", "/")
    return "posix:" + os.path.normcase(os.path.normpath(os.path.abspath(raw)))


def _nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{description} must be a non-empty string")
    return value


def _resolve_bound_path(raw: object, *, base: Path, description: str) -> Path:
    value = _nonempty_string(raw, description=description)
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error


def _candidate(result: Mapping[str, Any], result_key: str) -> str | None:
    fields = result.get("fields")
    field = fields.get(result_key) if isinstance(fields, Mapping) else None
    candidate = field.get("candidate") if isinstance(field, Mapping) else None
    return candidate if isinstance(candidate, str) and candidate.strip() else None


def _field(result: Mapping[str, Any], result_key: str) -> dict[str, Any] | None:
    fields = result.get("fields")
    field = fields.get(result_key) if isinstance(fields, Mapping) else None
    return dict(field) if isinstance(field, Mapping) else None


def _detection(result: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    detections = result.get("detections")
    if not isinstance(detections, list):
        return None
    matched = [
        dict(item)
        for item in detections
        if isinstance(item, Mapping) and item.get("label") == label
    ]
    if len(matched) > 1:
        raise AuditError(f"result has duplicate {label!r} detections")
    return matched[0] if matched else None


def _candidate_channels(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fields = result.get("fields")
    if not isinstance(fields, Mapping):
        return {}
    keys = {
        "candidate",
        "ctc_candidate",
        "structured_candidate",
        "raw",
        "normalized",
        "state",
    }
    output: dict[str, dict[str, Any]] = {}
    for public_field, (result_key, _) in FIELD_SPECS.items():
        value = fields.get(result_key)
        if not isinstance(value, Mapping):
            continue
        selected = {
            key: item
            for key, item in value.items()
            if key in keys or key.startswith("hybrid_ocr_")
        }
        output[public_field] = selected
    return output


def _contained_result_path(raw: object, *, run_root: Path, manifest: Path) -> Path:
    value = _nonempty_string(raw, description="manifest result")
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"manifest result is missing: {path}") from error
    if not resolved.is_file():
        raise AuditError(f"manifest result is not a file: {resolved}")
    try:
        resolved.relative_to(run_root)
    except ValueError as error:
        raise AuditError(f"manifest result escapes run root: {resolved}") from error
    return resolved


def _manifest_results(
    run_directory: Path, *, label: str, require_written: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        run_root = run_directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {label} run directory: {run_directory}") from error
    if not run_root.is_dir():
        raise AuditError(f"{label} run directory is not a directory: {run_root}")
    manifest = run_root / "inference_manifest.json"
    manifest_identity = _file_identity(manifest, description=f"{label} manifest")
    payload = _load_json(manifest, description=f"{label} manifest")
    if not isinstance(payload, list) or not payload:
        raise AuditError(f"{label} manifest must be a non-empty array")
    results: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    result_paths: set[Path] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise AuditError(f"{label} manifest[{index}] must be an object")
        source = _nonempty_string(
            item.get("source"), description=f"{label} manifest[{index}].source"
        )
        status = item.get("status")
        allowed = {"written"} if require_written else {"written", "skipped_existing"}
        if status not in allowed:
            raise AuditError(f"{label} manifest[{index}] has disallowed status {status!r}")
        key = _source_key(source)
        if key in results:
            raise AuditError(f"duplicate {label} manifest source {source!r}")
        result_path = _contained_result_path(
            item.get("result"), run_root=run_root, manifest=manifest
        )
        if result_path in result_paths:
            raise AuditError(f"duplicate {label} result path: {result_path}")
        result_paths.add(result_path)
        result = _load_json(result_path, description=f"{label} result")
        if not isinstance(result, Mapping):
            raise AuditError(f"{label} result must be an object: {result_path}")
        if _source_key(result.get("source")) != key:
            raise AuditError(f"{label} manifest/result source mismatch for {source!r}")
        results[key] = dict(result)
        identities[key] = _file_identity(result_path, description=f"{label} result")
    closure = hashlib.sha256()
    total_bytes = 0
    for key in sorted(identities):
        identity = identities[key]
        total_bytes += int(identity["size_bytes"])
        closure.update(
            f"{key}\0{identity['path']}\0{identity['sha256']}\0{identity['size_bytes']}\n".encode(
                "utf-8"
            )
        )
    return results, identities, {
        **manifest_identity,
        "records": len(results),
        "result_closure_sha256": closure.hexdigest(),
        "result_total_bytes": total_bytes,
    }


def _input_sources(path: Path) -> tuple[list[str], dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sources: list[str] = []
    indexed: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        source = line.strip()
        if not source:
            raise AuditError(f"input list {path}:{line_number} is blank")
        key = _source_key(source)
        if key in indexed:
            raise AuditError(f"input list has duplicate source {source!r}")
        sources.append(source)
        indexed[key] = source
    if not sources:
        raise AuditError(f"input list is empty: {path}")
    return sources, indexed


def _reference_text(field: str, slot: Mapping[str, Any]) -> str | None:
    text = slot.get("text")
    if not isinstance(text, str):
        return None
    if field == "amount":
        visible = slot.get("visible_text")
        if (
            isinstance(visible, str)
            and parse_amount_visible_format_target(visible) is not None
        ):
            return visible
    if field == "time":
        visible = slot.get("visible_text")
        if isinstance(visible, str) and visible:
            return visible
    # Match the scorer's effective reference-present contract: after
    # ``_reference_text`` it calls ``_has_reference``, which requires the
    # chosen string's boolean value to be true.  Thus ``text: ""`` is only a
    # placeholder unless amount/time has a valid non-empty visible target.
    return text if text else None


def _record_references(
    records_path: Path, *, selected_order: Sequence[str], split: str
) -> dict[str, dict[str, str]]:
    selected_keys = set(selected_order)
    references: dict[str, dict[str, str]] = {key: {} for key in selected_keys}
    seen_selected: set[str] = set()
    canonical_order: list[str] = []
    seen_split: set[str] = set()
    for line_number, record in enumerate(
        _load_jsonl(records_path, description="unified records"), start=1
    ):
        if record.get("split") != split:
            continue
        source = _nonempty_string(
            record.get("source"), description=f"records[{line_number}].source"
        )
        key = _source_key(source)
        if key not in seen_split:
            seen_split.add(key)
            canonical_order.append(key)
        slots = record.get("slots")
        if not isinstance(slots, Mapping):
            raise AuditError(f"records[{line_number}] has no slots object")
        if key not in selected_keys:
            continue
        seen_selected.add(key)
        for field in FIELD_SPECS:
            slot = slots.get(field)
            if not isinstance(slot, Mapping):
                continue
            reference = _reference_text(field, slot)
            if reference is None:
                continue
            previous = references[key].get(field)
            if previous is not None and previous != reference:
                raise AuditError(f"conflicting {field} references for {source!r}")
            references[key][field] = reference
    if canonical_order != list(selected_order):
        raise AuditError(
            "formal input list is not the complete canonical first-unique source order "
            f"for split={split!r}: records={len(canonical_order)} input={len(selected_order)}"
        )
    if seen_selected != selected_keys:
        raise AuditError(
            f"records manifest does not cover {len(selected_keys - seen_selected)} selected source(s)"
        )
    return references


def _score_comparisons(
    path: Path,
    *,
    selected_keys: set[str],
    references: Mapping[str, Mapping[str, str]],
    hybrid_results: Mapping[str, Mapping[str, Any]],
    hybrid_result_ids: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for index, row in enumerate(
        _load_jsonl(path, description="score comparisons")
    ):
        if row.get("schema_version") != 1 or row.get("kind") != (
            "receipt_mlnet_unified_comparison_v1"
        ):
            raise AuditError(f"score comparison[{index}] has unsupported schema")
        field = _nonempty_string(row.get("field"), description=f"score[{index}].field")
        if field not in FIELD_SPECS:
            raise AuditError(f"score comparison[{index}] has unsupported field {field!r}")
        source = _nonempty_string(
            row.get("source"), description=f"score[{index}].source"
        )
        key = _source_key(source)
        if key not in selected_keys:
            raise AuditError(f"score comparison[{index}] source is outside input list")
        pair = (key, field)
        if pair in indexed:
            raise AuditError(f"duplicate score comparison for {source!r} {field!r}")
        reference = row.get("reference_text")
        if not isinstance(reference, str) or references[key].get(field) != reference:
            raise AuditError(f"score comparison[{index}] reference disagrees with records")
        present = row.get("candidate_present")
        exact = row.get("raw_exact")
        if not isinstance(present, bool) or not isinstance(exact, bool):
            raise AuditError(f"score comparison[{index}] candidate booleans are invalid")
        candidate = row.get("candidate_text")
        result_key = FIELD_SPECS[field][0]
        expected_candidate = _candidate(hybrid_results[key], result_key)
        if candidate != expected_candidate or present != (expected_candidate is not None):
            raise AuditError(f"score comparison[{index}] candidate disagrees with hybrid result")
        if exact != (candidate is not None and candidate == reference):
            raise AuditError(f"score comparison[{index}] raw_exact disagrees with text")
        result_path = _resolve_bound_path(
            row.get("result_json"),
            base=path.parent,
            description=f"score comparison[{index}] result_json",
        )
        expected_result_path = Path(str(hybrid_result_ids[key]["path"])).resolve(
            strict=True
        )
        if result_path != expected_result_path:
            raise AuditError(
                f"score comparison[{index}] result_json disagrees with hybrid manifest"
            )
        indexed[pair] = row
        counts[field] += 1
    expected_pairs = {
        (key, field) for key, values in references.items() for field in values
    }
    if set(indexed) != expected_pairs:
        raise AuditError(
            "score comparisons do not exactly cover records references: "
            f"missing={len(expected_pairs - set(indexed))} extra={len(set(indexed) - expected_pairs)}"
        )
    return indexed, dict(counts)


def _ab_comparisons(
    path: Path,
    *,
    selected_keys: set[str],
    hybrid_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_load_jsonl(path, description="A/B comparisons")):
        source = _nonempty_string(row.get("source"), description=f"A/B[{index}].source")
        key = _source_key(source)
        if key not in selected_keys:
            raise AuditError(f"A/B[{index}] source is outside input list")
        if key in indexed:
            raise AuditError(f"duplicate A/B comparison source {source!r}")
        if not isinstance(row.get("invariant"), bool):
            raise AuditError(f"A/B[{index}].invariant must be boolean")
        failures = row.get("failures")
        if not isinstance(failures, list) or not all(
            isinstance(value, str) and value for value in failures
        ):
            raise AuditError(f"A/B[{index}].failures must contain strings")
        if row["invariant"] != (len(failures) == 0):
            raise AuditError(f"A/B[{index}] invariant/failures disagree")
        hybrid_recipient = _field(hybrid_results[key], "recipient")
        expected_candidate = (
            hybrid_recipient.get("candidate")
            if isinstance(hybrid_recipient, Mapping)
            else None
        )
        if row.get("recipient_candidate") != expected_candidate:
            raise AuditError(
                f"A/B[{index}] recipient candidate disagrees with hybrid result"
            )
        indexed[key] = row
    if set(indexed) != selected_keys:
        raise AuditError(
            f"A/B comparison source set differs: missing={len(selected_keys-set(indexed))} "
            f"extra={len(set(indexed)-selected_keys)}"
        )
    return indexed


def _summary_missing_sets(
    score_summary: Mapping[str, Any], *, selected: Mapping[str, str]
) -> dict[str, set[str]]:
    missing_root = score_summary.get("missing")
    values = (
        missing_root.get("all_receipt_field_candidates")
        if isinstance(missing_root, Mapping)
        else None
    )
    if not isinstance(values, Mapping):
        raise AuditError("score summary has no all-receipt missing source lists")
    result: dict[str, set[str]] = {}
    coverage = score_summary.get("all_receipt_candidate_coverage")
    coverage_fields = coverage.get("by_field") if isinstance(coverage, Mapping) else None
    if not isinstance(coverage_fields, Mapping):
        raise AuditError("score summary has no all-receipt candidate coverage")
    for field in FIELD_SPECS:
        item = values.get(field)
        sources = item.get("sources") if isinstance(item, Mapping) else None
        records = item.get("records") if isinstance(item, Mapping) else None
        if type(records) is not int or not isinstance(sources, list):
            raise AuditError(f"score summary missing list for {field} is invalid")
        keys = [_source_key(source) for source in sources]
        if len(keys) != len(set(keys)) or len(keys) != records:
            raise AuditError(f"score summary missing list for {field} has duplicates/count drift")
        if not set(keys).issubset(selected):
            raise AuditError(f"score summary missing list for {field} contains unknown sources")
        metric = coverage_fields.get(field)
        if not isinstance(metric, Mapping):
            raise AuditError(f"score summary all-receipt metric for {field} is invalid")
        if metric.get("expected_receipts") != len(selected):
            raise AuditError(f"score summary expected receipt count for {field} differs")
        if metric.get("missing_candidate_records") != records:
            raise AuditError(f"score summary missing count for {field} differs")
        if metric.get("candidate_records") != len(selected) - records:
            raise AuditError(f"score summary candidate count for {field} differs")
        result[field] = set(keys)
    return result


def _expectations(values: Sequence[str]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise AuditError(f"invalid --expect-missing {value!r}; use FIELD=COUNT")
        raw_field, raw_count = value.split("=", maxsplit=1)
        field = FIELD_ALIASES.get(raw_field.strip())
        if field is None:
            raise AuditError(f"unsupported expected field {raw_field!r}")
        if field in expected:
            raise AuditError(f"duplicate expected field {field!r}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise AuditError(f"invalid expected count {raw_count!r}") from error
        if count < 0:
            raise AuditError("expected missing counts must be non-negative")
        expected[field] = count
    return expected


def _assert_source_evidence_current(
    evidence: Mapping[str, Mapping[str, Any]],
    baseline_result_ids: Mapping[str, Mapping[str, Any]],
    hybrid_result_ids: Mapping[str, Mapping[str, Any]],
) -> None:
    for description, identity in evidence.items():
        observed = _file_identity(Path(str(identity["path"])), description=description)
        if dict(identity) != observed:
            raise AuditError(f"{description} changed while the audit was running")
    for label, identities in (
        ("baseline result", baseline_result_ids),
        ("hybrid result", hybrid_result_ids),
    ):
        for identity in identities.values():
            observed = _file_identity(Path(str(identity["path"])), description=label)
            if dict(identity) != observed:
                raise AuditError(f"{label} changed while the audit was running")


def audit(
    *,
    root: Path,
    score: Path,
    require_formal: bool = False,
    expected_missing: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve(strict=True)
    score = score.resolve(strict=True)
    if not root.is_dir() or not score.is_dir():
        raise AuditError("root and score must be directories")

    score_summary_path = score / "summary.json"
    score_comparisons_path = score / "comparisons.jsonl"
    ab_summary_path = root / "comparison" / "summary.json"
    ab_comparisons_path = root / "comparison" / "comparisons.jsonl"
    source_evidence: dict[str, dict[str, Any]] = {
        "score summary": _file_identity(score_summary_path, description="score summary"),
        "score comparisons": _file_identity(
            score_comparisons_path, description="score comparisons"
        ),
        "A/B summary": _file_identity(ab_summary_path, description="A/B summary"),
        "A/B comparisons": _file_identity(
            ab_comparisons_path, description="A/B comparisons"
        ),
    }
    score_summary = _load_json(score_summary_path, description="score summary")
    ab_summary = _load_json(ab_summary_path, description="A/B summary")
    if not isinstance(score_summary, Mapping) or score_summary.get("schema_version") != 1:
        raise AuditError("score summary schema is unsupported")
    if score_summary.get("kind") != SCORE_KIND:
        raise AuditError("score summary kind is not a full scorer evaluation")
    if not isinstance(ab_summary, Mapping) or ab_summary.get("schema_version") != 2:
        raise AuditError("A/B summary schema is unsupported")
    if ab_summary.get("kind") != AB_KIND:
        raise AuditError("A/B summary kind is unsupported")

    scope = score_summary.get("evaluation_scope")
    input_selection = score_summary.get("input_selection")
    if not isinstance(scope, Mapping) or scope.get("kind") != "full_split":
        raise AuditError("score summary is not a full-split evaluation")
    if not isinstance(input_selection, Mapping) or input_selection.get("hash_bound") is not True:
        raise AuditError("score summary input selection is not SHA-256 bound")
    if score_summary.get("coverage_contract_version") != 2:
        raise AuditError("score summary does not use all-receipt coverage contract v2")
    if score_summary.get("evaluation_split") != "val":
        raise AuditError("score summary evaluation split must be val")
    if input_selection.get("selection_order") != FULL_SELECTION_ORDER or scope.get(
        "selection_order"
    ) != FULL_SELECTION_ORDER:
        raise AuditError("score summary does not use canonical full-split selection order")
    input_path = _resolve_bound_path(
        input_selection.get("path"), base=score, description="formal input list"
    )
    input_identity = _file_identity(input_path, description="formal input list")
    if input_identity["sha256"] != input_selection.get("sha256"):
        raise AuditError("formal input list SHA-256 differs from score summary")
    scope_input_path = _resolve_bound_path(
        scope.get("input_list_path"),
        base=score,
        description="score evaluation-scope input list",
    )
    if scope_input_path != input_path:
        raise AuditError("score evaluation scope points to a different input list")
    if scope.get("input_list_sha256") != input_identity["sha256"]:
        raise AuditError("score evaluation-scope input SHA-256 differs")
    if scope.get("requested_limit") is not None:
        raise AuditError("full-split score evaluation unexpectedly has a requested limit")
    source_evidence["formal input list"] = input_identity
    input_sources, selected = _input_sources(input_path)
    selected_keys = set(selected)
    if input_selection.get("records") != len(selected):
        raise AuditError("score summary input record count differs from input list")
    if scope.get("evaluated_expected_receipts") != len(selected) or scope.get(
        "full_split_expected_receipts"
    ) != len(selected):
        raise AuditError("score summary full-split counts differ from input list")
    if require_formal and len(selected) != FORMAL_RECORDS:
        raise AuditError(
            f"--require-formal requires exactly {FORMAL_RECORDS} records, found {len(selected)}"
        )
    if require_formal and ab_summary.get("evaluation_mode") != "formal":
        raise AuditError("--require-formal requires a formal A/B summary")
    if require_formal:
        floors = score_summary.get("floors")
        if not isinstance(floors, Mapping) or any(
            floors.get(field) != floor for field, floor in FIXED_FLOORS.items()
        ):
            raise AuditError("formal score floors differ from fixed delivery floors")

    records_path = _resolve_bound_path(
        score_summary.get("records"), base=score, description="records manifest"
    )
    records_identity = _file_identity(records_path, description="records manifest")
    if records_identity["sha256"] != score_summary.get("records_sha256"):
        raise AuditError("records manifest SHA-256 differs from score summary")
    source_evidence["records manifest"] = records_identity
    model_path = _resolve_bound_path(
        score_summary.get("model"), base=score, description="unified model"
    )
    model_identity = _file_identity(model_path, description="unified model")
    if model_identity["sha256"] != score_summary.get("model_sha256"):
        raise AuditError("unified model SHA-256 differs from score summary")
    source_evidence["unified model"] = model_identity

    baseline, baseline_ids, baseline_closure = _manifest_results(
        root / "baseline-v13", label="baseline", require_written=require_formal
    )
    hybrid, hybrid_ids, hybrid_closure = _manifest_results(
        root / "hybrid-recipient", label="hybrid", require_written=require_formal
    )
    source_evidence["baseline manifest"] = {
        key: baseline_closure[key] for key in ("path", "sha256", "size_bytes")
    }
    source_evidence["hybrid manifest"] = {
        key: hybrid_closure[key] for key in ("path", "sha256", "size_bytes")
    }
    if set(baseline) != selected_keys or set(hybrid) != selected_keys:
        raise AuditError("baseline/hybrid manifest source sets differ from formal input list")
    if ab_summary.get("records") != len(selected):
        raise AuditError("A/B summary record count differs from input list")
    ab_input_set = ab_summary.get("input_set")
    ab_input_manifest = (
        ab_input_set.get("input_manifest")
        if isinstance(ab_input_set, Mapping)
        else None
    )
    if not isinstance(ab_input_manifest, Mapping):
        raise AuditError("A/B summary has no bound input manifest")
    _assert_identity(
        ab_input_manifest,
        description="A/B input manifest",
        expected_path=input_path,
    )
    if ab_input_manifest.get("sha256") != input_identity["sha256"]:
        raise AuditError("A/B input manifest SHA-256 differs from scorer input")
    run_manifests = ab_summary.get("run_manifests")
    if not isinstance(run_manifests, Mapping):
        raise AuditError("A/B summary has no bound run manifests")
    for label, actual in (("baseline", baseline_closure), ("hybrid", hybrid_closure)):
        identity = run_manifests.get(label)
        if not isinstance(identity, Mapping):
            raise AuditError(f"A/B summary has no {label} manifest identity")
        _assert_identity(
            identity,
            description=f"A/B {label} manifest",
            expected_path=Path(str(actual["path"])),
        )
        if identity.get("sha256") != actual["sha256"]:
            raise AuditError(f"A/B {label} manifest SHA-256 differs")
    score_manifest = _resolve_bound_path(
        score_summary.get("manifest"), base=score, description="score hybrid manifest"
    )
    if score_manifest != Path(str(hybrid_closure["path"])):
        raise AuditError("score summary does not point to selected hybrid manifest")
    if score_summary.get("manifest_sha256") != hybrid_closure["sha256"]:
        raise AuditError("score summary hybrid manifest SHA-256 differs")
    results_root = _resolve_bound_path(
        score_summary.get("results_root"), base=score, description="score results root"
    )
    if results_root != (root / "hybrid-recipient").resolve(strict=True):
        raise AuditError("score summary results root differs from selected hybrid run")

    references = _record_references(
        records_path,
        selected_order=[_source_key(source) for source in input_sources],
        split=str(score_summary.get("evaluation_split")),
    )
    score_rows, reference_counts = _score_comparisons(
        score_comparisons_path,
        selected_keys=selected_keys,
        references=references,
        hybrid_results=hybrid,
        hybrid_result_ids=hybrid_ids,
    )
    by_field = score_summary.get("by_field")
    if not isinstance(by_field, Mapping):
        raise AuditError("score summary by_field is invalid")
    for field in FIELD_SPECS:
        metric = by_field.get(field)
        if not isinstance(metric, Mapping) or metric.get("records") != reference_counts.get(
            field, 0
        ):
            raise AuditError(f"score summary reference denominator for {field} differs")
    accuracy_denominators = score_summary.get("accuracy_denominators")
    denominator_fields = (
        accuracy_denominators.get("by_field")
        if isinstance(accuracy_denominators, Mapping)
        else None
    )
    input_reference_counts = input_selection.get("field_reference_counts")
    if (
        not isinstance(accuracy_denominators, Mapping)
        or accuracy_denominators.get("hash_bound") is not True
        or not isinstance(denominator_fields, Mapping)
        or not isinstance(input_reference_counts, Mapping)
    ):
        raise AuditError("score summary reference denominators are not hash-bound")
    for field in FIELD_SPECS:
        observed = reference_counts.get(field, 0)
        if denominator_fields.get(field) != observed or input_reference_counts.get(field) != observed:
            raise AuditError(f"score summary bound reference count for {field} differs")

    ab_rows = _ab_comparisons(
        ab_comparisons_path,
        selected_keys=selected_keys,
        hybrid_results=hybrid,
    )
    invariant_records = sum(row["invariant"] for row in ab_rows.values())
    recipient_records = sum(
        isinstance(row.get("recipient_candidate"), str)
        and bool(row.get("recipient_candidate"))
        for row in ab_rows.values()
    )
    if ab_summary.get("invariant_records") != invariant_records or ab_summary.get(
        "recipient_candidate_coverage"
    ) != recipient_records / len(selected):
        raise AuditError("A/B summary counts differ from A/B comparisons")
    summary_missing = _summary_missing_sets(score_summary, selected=selected)
    observed_missing: dict[str, set[str]] = {}
    for field, (result_key, _) in FIELD_SPECS.items():
        observed_missing[field] = {
            key for key, result in hybrid.items() if _candidate(result, result_key) is None
        }
        if observed_missing[field] != summary_missing[field]:
            raise AuditError(
                f"scorer missing source set for {field} disagrees with hybrid results: "
                f"summary_only={len(summary_missing[field]-observed_missing[field])} "
                f"result_only={len(observed_missing[field]-summary_missing[field])}"
            )
    for field, expected in (expected_missing or {}).items():
        observed = len(observed_missing[field])
        if observed != expected:
            raise AuditError(
                f"expected {field} missing={expected}, observed {observed}"
            )

    missing_by_source: dict[str, list[str]] = {}
    for key in selected_keys:
        fields = [field for field in FIELD_SPECS if key in observed_missing[field]]
        if fields:
            missing_by_source[key] = fields
    findings: list[dict[str, Any]] = []
    for key in sorted(missing_by_source, key=lambda item: selected[item].casefold()):
        baseline_result = baseline[key]
        hybrid_result = hybrid[key]
        per_field: dict[str, dict[str, Any]] = {}
        for field in missing_by_source[key]:
            result_key, detection_label = FIELD_SPECS[field]
            comparison = score_rows.get((key, field))
            per_field[field] = {
                "reference_present": comparison is not None,
                "reference_text": references[key].get(field),
                "score_comparison": comparison,
                "baseline_field": _field(baseline_result, result_key),
                "hybrid_field": _field(hybrid_result, result_key),
                "baseline_detection": _detection(baseline_result, detection_label),
                "hybrid_detection": _detection(hybrid_result, detection_label),
            }
        findings.append(
            {
                "schema_version": 1,
                "kind": FINDING_KIND,
                "source": selected[key],
                "missing_fields": missing_by_source[key],
                "reference_present_by_field": {
                    field: field in references[key] for field in missing_by_source[key]
                },
                "ab_comparison": ab_rows[key],
                "baseline_result": baseline_ids[key],
                "hybrid_result": hybrid_ids[key],
                "baseline_device": baseline_result.get("device"),
                "hybrid_device": hybrid_result.get("device"),
                "baseline_geometry": baseline_result.get("geometry"),
                "hybrid_geometry": hybrid_result.get("geometry"),
                "baseline_candidate_channels": _candidate_channels(baseline_result),
                "hybrid_candidate_channels": _candidate_channels(hybrid_result),
                "by_missing_field": per_field,
            }
        )

    exact_sets: Counter[tuple[str, ...]] = Counter(
        tuple(fields) for fields in missing_by_source.values()
    )
    pairwise: list[dict[str, Any]] = []
    fields = list(FIELD_SPECS)
    for left_index, left in enumerate(fields):
        for right in fields[left_index + 1 :]:
            overlap = observed_missing[left] & observed_missing[right]
            if overlap:
                pairwise.append(
                    {
                        "fields": [left, right],
                        "records": len(overlap),
                        "sources": sorted(selected[key] for key in overlap),
                    }
                )
    summary = {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "formal_required": require_formal,
        "ab_root": root.as_posix(),
        "score_directory": score.as_posix(),
        "records": len(selected),
        "missing_by_field": {
            field: {
                "records": len(keys),
                "reference_present_records": sum(
                    field in references[key] for key in keys
                ),
                "reference_missing_records": sum(
                    field not in references[key] for key in keys
                ),
                "sources": sorted(selected[key] for key in keys),
            }
            for field, keys in observed_missing.items()
        },
        "overlap": {
            "union_missing_records": len(missing_by_source),
            "missing_field_count_distribution": {
                str(count): records
                for count, records in sorted(
                    Counter(map(len, missing_by_source.values())).items()
                )
            },
            "exact_missing_field_sets": [
                {
                    "fields": list(field_set),
                    "records": count,
                    "sources": sorted(
                        selected[key]
                        for key, current in missing_by_source.items()
                        if tuple(current) == field_set
                    ),
                }
                for field_set, count in sorted(exact_sets.items())
            ],
            "pairwise": pairwise,
        },
        "source_evidence": source_evidence,
        "result_closures": {
            "baseline": baseline_closure,
            "hybrid": hybrid_closure,
        },
        "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
    }
    bindings = {
        "source_evidence": source_evidence,
        "baseline_result_ids": baseline_ids,
        "hybrid_result_ids": hybrid_ids,
    }
    _assert_source_evidence_current(source_evidence, baseline_ids, hybrid_ids)
    return summary, findings, bindings


def write_atomic(
    output: Path,
    *,
    summary: Mapping[str, Any],
    findings: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> None:
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    _assert_source_evidence_current(
        bindings["source_evidence"],
        bindings["baseline_result_ids"],
        bindings["hybrid_result_ids"],
    )
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
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
                for row in findings
            ),
            encoding="utf-8",
        )
        _assert_source_evidence_current(
            bindings["source_evidence"],
            bindings["baseline_result_ids"],
            bindings["hybrid_result_ids"],
        )
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite audit output: {output}")
        stage.replace(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="hybrid CPU A/B root")
    parser.add_argument("--score", required=True, type=Path, help="existing scorer output")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument(
        "--expect-missing",
        action="append",
        default=[],
        metavar="FIELD=COUNT",
        help="fail when a reconstructed missing count differs; repeat per field",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected = _expectations(args.expect_missing)
        summary, findings, bindings = audit(
            root=args.root,
            score=args.score,
            require_formal=args.require_formal,
            expected_missing=expected,
        )
        write_atomic(
            args.output_directory,
            summary=summary,
            findings=findings,
            bindings=bindings,
        )
    except (AuditError, FileExistsError, OSError, UnicodeError) as error:
        print(f"Formal missing-fields audit failed: {error}")
        return 2
    counts = " ".join(
        f"{field}={summary['missing_by_field'][field]['records']}"
        for field in FIELD_SPECS
    )
    print(
        f"formal_missing_fields_audit records={summary['records']} "
        f"union={summary['overlap']['union_missing_records']} {counts} "
        f"output={args.output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
