#!/usr/bin/env python3
"""Prepare and score ML.NET v12/v13 unified-reader delivery runs.

``prepare`` writes either the complete unique source set or an exact,
deterministic five-field-covered pilot selection. ``score`` hash-binds an
explicit pilot list, joins the C# inference manifest to its result JSON files,
verifies that every result came from the requested unified ONNX artifact, and
applies the same strict candidate/reference comparison used by the Python
unified evaluator.

This evaluates the diagnostic ``candidate`` channel.  It does not promote the
artifact's review-only business ``value`` or turn Paddle-derived labels into
independent human truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import ntpath
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transfer_receipt_ai.ocr_unified_targets import parse_amount_visible_format_target
from transfer_receipt_ai.ocr import normalize_status


SCHEMA_VERSION = 1
DEFAULT_AMOUNT_FLOOR = 0.7885
DEFAULT_TIME_FLOOR = 0.9840
DEFAULT_PAYMENT_FLOOR = 0.9325
DEFAULT_RECIPIENT_FLOOR = 0.90
DEFAULT_STATUS_TEXT_FLOOR = 0.90
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

# Python evaluator field name -> C# ``fields`` property name.
FIELD_RESULT_KEYS = {
    "amount": "amount",
    "time": "time",
    "payment_method_field": "payment_method",
    "recipient_field": "recipient",
}
STATUS_RESULT_KEY = "transfer_status"
PILOT_REQUIRED_FIELDS = (
    "amount",
    "time",
    "payment_method_field",
    "recipient_field",
    "transfer_status",
)
PILOT_FIELD_QUOTA = 16
PILOT_SELECTION_ORDER = "deterministic_min16_field_quota_then_records_manifest_order"
FULL_SELECTION_ORDER = "first_unique_source_in_records_manifest_order"


class EvaluationInputError(ValueError):
    """Raised when delivery evidence is malformed or ambiguous."""


def _has_reference(entry: Mapping[str, Any], field: str) -> bool:
    references = entry.get("references")
    return (
        isinstance(references, Mapping)
        and isinstance(references.get(field), str)
        and bool(references[field])
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exception:
                raise EvaluationInputError(f"{path}:{line_number}: invalid JSON: {exception.msg}") from exception
            if not isinstance(value, dict):
                raise EvaluationInputError(f"{path}:{line_number}: expected one JSON object")
            rows.append(value)
    return rows


def _load_json(path: Path, *, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exception:
        raise EvaluationInputError(f"{description} {path} is invalid JSON: {exception.msg}") from exception


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _source_key(value: str) -> str:
    """Normalise a source for joins without rewriting the displayed value."""
    if WINDOWS_ABSOLUTE_PATH.match(value) or "\\" in value:
        return "windows:" + ntpath.normcase(ntpath.normpath(value)).replace("\\", "/")
    return "posix:" + os.path.normpath(os.path.abspath(value))


def _source_from_record(record: Mapping[str, Any], *, source: Path) -> str:
    value = record.get("source")
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"{source}: record {record.get('id')!r} has no non-empty source")
    return value


def _pilot_selection(
    expected: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise EvaluationInputError("pilot input selection requires a positive limit")
    if limit > len(expected):
        raise EvaluationInputError(
            f"pilot input limit {limit} exceeds {len(expected)} unique source(s)"
        )

    available = {
        field: sum(_has_reference(entry, field) for entry in expected.values())
        for field in PILOT_REQUIRED_FIELDS
    }
    missing_fields = [field for field, count in available.items() if count <= 0]
    if missing_fields:
        raise EvaluationInputError(
            "pilot input selection has no validation reference for required field(s): "
            + ", ".join(missing_fields)
        )
    quotas = {
        field: min(PILOT_FIELD_QUOTA, available[field], limit)
        for field in PILOT_REQUIRED_FIELDS
    }
    selected: list[str] = []
    selected_set: set[str] = set()
    coverage = {field: 0 for field in PILOT_REQUIRED_FIELDS}
    ordered_keys = list(expected)

    while any(coverage[field] < quotas[field] for field in PILOT_REQUIRED_FIELDS):
        if len(selected) >= limit:
            deficits = {
                field: quotas[field] - coverage[field]
                for field in PILOT_REQUIRED_FIELDS
                if coverage[field] < quotas[field]
            }
            raise EvaluationInputError(
                f"pilot limit {limit} cannot satisfy deterministic field quotas {deficits}"
            )
        best_key: str | None = None
        best_gain = 0
        for key in ordered_keys:
            if key in selected_set:
                continue
            gain = sum(
                _has_reference(expected[key], field) and coverage[field] < quotas[field]
                for field in PILOT_REQUIRED_FIELDS
            )
            if gain > best_gain:
                best_key = key
                best_gain = gain
        if best_key is None or best_gain <= 0:
            raise EvaluationInputError("pilot field quotas cannot be satisfied by unique sources")
        selected.append(best_key)
        selected_set.add(best_key)
        for field in PILOT_REQUIRED_FIELDS:
            if _has_reference(expected[best_key], field):
                coverage[field] += 1

    for key in ordered_keys:
        if len(selected) >= limit:
            break
        if key not in selected_set:
            selected.append(key)
            selected_set.add(key)
            for field in PILOT_REQUIRED_FIELDS:
                if _has_reference(expected[key], field):
                    coverage[field] += 1
    if len(selected) != limit or len(selected_set) != limit:
        raise EvaluationInputError(
            f"pilot selection produced {len(selected)} unique source(s), expected exactly {limit}"
        )
    if any(coverage[field] <= 0 for field in PILOT_REQUIRED_FIELDS):
        raise EvaluationInputError("pilot selection did not cover every required field")
    return selected, quotas, coverage


def prepare_input_list(
    *,
    records_path: Path,
    output_path: Path,
    split: str = "val",
    limit: int = 0,
) -> dict[str, Any]:
    records_path = records_path.resolve()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise EvaluationInputError("limit must be a non-negative integer")
    rows = _load_jsonl(records_path)
    split_records = sum(record.get("split") == split for record in rows)
    expected = _expected_receipts(
        records_path,
        split=split,
        include_status_text=True,
    )
    quotas: dict[str, int] | None = None
    if limit > 0:
        selected_keys, quotas, selected_coverage = _pilot_selection(expected, limit=limit)
        selection_order = PILOT_SELECTION_ORDER
    else:
        selected_keys = list(expected)
        selected_coverage = {
            field: sum(_has_reference(entry, field) for entry in expected.values())
            for field in PILOT_REQUIRED_FIELDS
        }
        selection_order = FULL_SELECTION_ORDER
    sources = [str(expected[key]["source"]) for key in selected_keys]
    _atomic_write_text(output_path, "".join(f"{source}\n" for source in sources))
    return {
        "records": split_records,
        "unique_sources": len(sources),
        "full_unique_sources": len(expected),
        "split": split,
        "limit": limit if limit > 0 else None,
        "selection_order": selection_order,
        "field_quotas": quotas,
        "selected_field_reference_counts": selected_coverage,
        "output": output_path.resolve().as_posix(),
        "output_sha256": _sha256(output_path),
    }


def _reference_text(field: str, slot: Mapping[str, Any]) -> str | None:
    """Mirror the architecture-v12/v13 reference selection in ``ocr_unified``."""
    text = slot.get("text")
    if not isinstance(text, str):
        return None
    if field == "amount":
        visible = slot.get("visible_text")
        if isinstance(visible, str) and parse_amount_visible_format_target(visible) is not None:
            return visible
        return text
    if field == "time":
        visible = slot.get("visible_text")
        return visible if isinstance(visible, str) and visible else text
    if field in {"payment_method_field", "recipient_field", "transfer_status"}:
        return text
    raise AssertionError(field)


def _amount_semantic_decimal(value: str | None) -> Decimal | None:
    """Return a strict numeric CNY diagnostic, never an OCR repair.

    Raw display exact-match remains the delivery guard.  This parallel view
    only ignores surrounding whitespace plus the two supported yen glyphs,
    their optional display-space and valid thousands separators.  The shared
    v8 parser rejects malformed grouping, OCR substitutions and ambiguous
    signs before ``Decimal`` is allowed to compare the digits.
    """
    if not isinstance(value, str):
        return None
    parsed = parse_amount_visible_format_target(value.strip())
    if parsed is None:
        return None
    canonical = parsed.get("canonical_decimal")
    if not isinstance(canonical, str):
        return None
    try:
        decimal = Decimal(canonical)
    except InvalidOperation:
        return None
    return decimal if decimal.is_finite() else None


def _amount_semantic_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _expected_receipts(
    records_path: Path,
    *,
    split: str,
    include_status_text: bool = False,
) -> dict[str, dict[str, Any]]:
    field_result_keys = dict(FIELD_RESULT_KEYS)
    if include_status_text:
        field_result_keys["transfer_status"] = STATUS_RESULT_KEY
    expected: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(records_path):
        if record.get("split") != split:
            continue
        source = _source_from_record(record, source=records_path)
        key = _source_key(source)
        raw_slots = record.get("slots")
        if not isinstance(raw_slots, Mapping):
            raise EvaluationInputError(f"{records_path}: record {record.get('id')!r} has no slots object")
        references: dict[str, str] = {}
        reference_diagnostics: dict[str, dict[str, Any]] = {}
        for field in field_result_keys:
            slot = raw_slots.get(field)
            if isinstance(slot, Mapping):
                reference = _reference_text(field, slot)
                if reference is not None:
                    references[field] = reference
                    reference_diagnostics[field] = {
                        "reference_crop_sha256": slot.get("crop_sha256"),
                        "reference_detector_score": slot.get("detector_score"),
                        "reference_bbox_rectified": slot.get("bbox_rectified"),
                    }
                    if field == "transfer_status":
                        reference_diagnostics[field]["reference_status_class"] = slot.get("class_name")
        if key not in expected:
            expected[key] = {
                "source": source,
                "id": str(record.get("id", source)),
                "group_id": record.get("group_id"),
                "teacher_result_json": record.get("result_json"),
                "references": references,
                "reference_diagnostics": reference_diagnostics,
            }
            continue
        existing = expected[key]
        existing_references = existing["references"]
        existing_diagnostics = existing["reference_diagnostics"]
        for field, reference in references.items():
            previous = existing_references.get(field)
            if previous is not None and previous != reference:
                raise EvaluationInputError(
                    f"{records_path}: source {source!r} has conflicting {field} references "
                    f"{previous!r} and {reference!r}"
                )
            existing_references[field] = reference
            existing_diagnostics.setdefault(field, reference_diagnostics[field])
    if not expected:
        raise EvaluationInputError(f"{records_path}: no records with split={split!r}")
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_input_list(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[str], list[str], str]:
    if not path.is_file():
        raise EvaluationInputError(f"Explicit input list not found: {path}")
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ) is None:
        raise EvaluationInputError("input_list_sha256 must be a 64-character SHA-256")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise EvaluationInputError(
            f"explicit input list SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"observed {actual_sha256}"
        )

    sources: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    text = payload.decode("utf-8-sig")
    for line_number, line in enumerate(text.splitlines(), start=1):
        source = line.strip()
        if not source:
            raise EvaluationInputError(
                f"explicit input list {path}:{line_number} contains an empty source"
            )
        key = _source_key(source)
        if key in seen:
            raise EvaluationInputError(
                f"explicit input list {path}:{line_number} contains duplicate source {source!r}"
            )
        seen.add(key)
        sources.append(source)
        keys.append(key)
    if not sources:
        raise EvaluationInputError(f"explicit input list {path} is empty")
    return sources, keys, actual_sha256


def _resolve_result_path(raw: str, *, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    beside_manifest = manifest_path.parent / path
    from_working_directory = Path.cwd() / path
    if beside_manifest.exists() or not from_working_directory.exists():
        return beside_manifest
    return from_working_directory


def _candidate(result: Mapping[str, Any], result_key: str) -> tuple[str | None, str | None]:
    fields = result.get("fields")
    if not isinstance(fields, Mapping):
        return None, "fields_missing"
    field = fields.get(result_key)
    if not isinstance(field, Mapping):
        return None, "field_missing"
    candidate = field.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        return None, "candidate_missing"
    return candidate, None


def _result_field_diagnostics(
    result: Mapping[str, Any] | None,
    *,
    result_key: str,
    detector_label: str,
) -> dict[str, Any]:
    if result is None:
        return {
            "ctc_candidate_text": None,
            "structured_candidate_text": None,
            "detection_bbox_image": None,
            "detection_score": None,
            "result_geometry": None,
        }

    fields = result.get("fields")
    result_field = fields.get(result_key) if isinstance(fields, Mapping) else None
    ctc_candidate = result_field.get("ctc_candidate") if isinstance(result_field, Mapping) else None
    structured_candidate = (
        result_field.get("structured_candidate") if isinstance(result_field, Mapping) else None
    )

    detection_bbox: list[float] | None = None
    detection_score: float | None = None
    detections = result.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, Mapping) or detection.get("label") != detector_label:
                continue
            raw_bbox = detection.get("bbox_image")
            if (
                isinstance(raw_bbox, list)
                and len(raw_bbox) == 4
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw_bbox)
                and all(math.isfinite(float(value)) for value in raw_bbox)
            ):
                detection_bbox = [float(value) for value in raw_bbox]
            raw_score = detection.get("score")
            if (
                isinstance(raw_score, (int, float))
                and not isinstance(raw_score, bool)
                and math.isfinite(float(raw_score))
            ):
                detection_score = float(raw_score)
            break

    geometry = result.get("geometry")
    return {
        "ctc_candidate_text": ctc_candidate if isinstance(ctc_candidate, str) else None,
        "structured_candidate_text": structured_candidate if isinstance(structured_candidate, str) else None,
        "detection_bbox_image": detection_bbox,
        "detection_score": detection_score,
        "result_geometry": dict(geometry) if isinstance(geometry, Mapping) else None,
    }


def _floor(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exception:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exception
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return parsed


def _limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exception:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exception
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _load_manifest_results(
    *,
    manifest_path: Path,
    model_sha256: str,
    allowed_source_keys: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    payload = _load_json(manifest_path, description="ML.NET inference manifest")
    if not isinstance(payload, list):
        raise EvaluationInputError(f"ML.NET inference manifest {manifest_path} must contain a JSON array")
    if allowed_source_keys is not None and not payload:
        raise EvaluationInputError(
            f"ML.NET inference manifest {manifest_path} is empty for the bound input list"
        )

    results: dict[str, dict[str, Any]] = {}
    integrity_failures: list[str] = []
    observed_hashes: set[str] = set()
    missing_hash_sources: list[str] = []
    mismatched_hash_sources: list[str] = []
    source_mismatches: list[dict[str, str]] = []
    missing_result_files: list[dict[str, str]] = []
    invalid_result_files: list[dict[str, str]] = []
    duplicate_sources: list[str] = []
    usable_manifest_sources: set[str] = set()
    seen_manifest_sources: set[str] = set()

    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            if allowed_source_keys is not None:
                raise EvaluationInputError(
                    f"ML.NET inference manifest {manifest_path}[{index}] is not an object"
                )
            integrity_failures.append(f"manifest[{index}] is not an object")
            continue
        manifest_source = item.get("source")
        result_value = item.get("result")
        status = item.get("status")
        if not isinstance(manifest_source, str) or not manifest_source:
            if allowed_source_keys is not None:
                raise EvaluationInputError(
                    f"ML.NET inference manifest {manifest_path}[{index}] has no source"
                )
            integrity_failures.append(f"manifest[{index}] has no source")
            continue
        source_key = _source_key(manifest_source)
        if source_key in seen_manifest_sources and allowed_source_keys is not None:
            raise EvaluationInputError(
                f"ML.NET inference manifest {manifest_path} contains duplicate source "
                f"{manifest_source!r}"
            )
        seen_manifest_sources.add(source_key)
        if allowed_source_keys is not None and source_key not in allowed_source_keys:
            raise EvaluationInputError(
                f"ML.NET inference manifest source {manifest_source!r} is outside the "
                "hash-bound explicit input list"
            )
        if status not in {"written", "skipped_existing"}:
            integrity_failures.append(f"manifest source {manifest_source!r} has incomplete status {status!r}")
            continue
        if not isinstance(result_value, str) or not result_value:
            missing_result_files.append({"source": manifest_source, "result": ""})
            continue
        result_path = _resolve_result_path(result_value, manifest_path=manifest_path)
        if not result_path.is_file():
            missing_result_files.append({"source": manifest_source, "result": result_path.as_posix()})
            continue
        try:
            result = _load_json(result_path, description="ML.NET result")
        except (EvaluationInputError, OSError) as exception:
            invalid_result_files.append(
                {"source": manifest_source, "result": result_path.as_posix(), "error": str(exception)}
            )
            continue
        if not isinstance(result, Mapping):
            invalid_result_files.append(
                {"source": manifest_source, "result": result_path.as_posix(), "error": "top level is not an object"}
            )
            continue

        # Artifact provenance is checked for every readable result referenced
        # by the manifest, even when its source join is subsequently rejected.
        # Otherwise a source-mismatched file could hide a mixed-model batch.
        contracts = result.get("model_contracts")
        observed_hash = contracts.get("unified_ocr_model_sha256") if isinstance(contracts, Mapping) else None
        if isinstance(observed_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", observed_hash):
            observed_hash = observed_hash.lower()
            observed_hashes.add(observed_hash)
            if observed_hash != model_sha256:
                mismatched_hash_sources.append(manifest_source)
        else:
            missing_hash_sources.append(manifest_source)

        result_source = result.get("source")
        if not isinstance(result_source, str) or _source_key(result_source) != source_key:
            source_mismatches.append(
                {
                    "manifest_source": manifest_source,
                    "result_source": result_source if isinstance(result_source, str) else "",
                    "result": result_path.as_posix(),
                }
            )
            continue

        if source_key in results:
            duplicate_sources.append(manifest_source)
            continue
        usable_manifest_sources.add(source_key)
        results[source_key] = {
            "source": manifest_source,
            "path": result_path,
            "status": status,
            "payload": result,
            "model_sha256": observed_hash if isinstance(observed_hash, str) else None,
        }

    if missing_result_files:
        integrity_failures.append(f"manifest result files missing: {len(missing_result_files)}")
    if invalid_result_files:
        integrity_failures.append(f"manifest result files invalid: {len(invalid_result_files)}")
    if source_mismatches:
        integrity_failures.append(f"manifest/result source mismatches: {len(source_mismatches)}")
    if duplicate_sources:
        integrity_failures.append(f"duplicate manifest result sources: {len(duplicate_sources)}")
    if missing_hash_sources:
        integrity_failures.append(f"results missing unified model SHA-256: {len(missing_hash_sources)}")
    if mismatched_hash_sources:
        integrity_failures.append(f"results from a different unified model: {len(mismatched_hash_sources)}")
    if len(observed_hashes) > 1:
        integrity_failures.append(f"results contain multiple unified model SHA-256 values: {len(observed_hashes)}")

    audit = {
        "manifest_records": len(payload),
        "usable_manifest_sources": len(usable_manifest_sources),
        "observed_unified_model_sha256": sorted(observed_hashes),
        "missing_unified_model_sha256_sources": sorted(set(missing_hash_sources)),
        "mismatched_unified_model_sha256_sources": sorted(set(mismatched_hash_sources)),
        "source_mismatches": source_mismatches,
        "missing_result_files": missing_result_files,
        "invalid_result_files": invalid_result_files,
        "duplicate_sources": sorted(set(duplicate_sources)),
        "all_results_match_model": not (
            missing_hash_sources or mismatched_hash_sources or len(observed_hashes) != 1
        )
        and observed_hashes == {model_sha256},
    }
    return results, audit, integrity_failures


def _field_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = len(rows)
    exact_matches = sum(bool(row["raw_exact"]) for row in rows)
    candidates = sum(bool(row["candidate_present"]) for row in rows)
    return {
        "records": records,
        "raw_exact_matches": exact_matches,
        "raw_exact_match": exact_matches / records if records else None,
        "candidate_records": candidates,
        "missing_candidate_records": records - candidates,
        "candidate_coverage": candidates / records if records else None,
    }


def _status_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _field_metrics(rows)
    non_success_rows = [
        row for row in rows if row.get("reference_status_class") in {"pending", "failed"}
    ]
    non_success_to_success = sum(bool(row.get("non_success_to_success")) for row in non_success_rows)
    metrics.update(
        {
            "non_success_truth_records": len(non_success_rows),
            "non_success_safety_calibrated": bool(non_success_rows),
            "non_success_to_success": non_success_to_success,
        }
    )
    return metrics


def _amount_semantic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = len(rows)
    reference_parseable = sum(row.get("reference_amount_decimal") is not None for row in rows)
    candidate_parseable = sum(row.get("candidate_amount_decimal") is not None for row in rows)
    comparable = sum(
        row.get("reference_amount_decimal") is not None and row.get("candidate_amount_decimal") is not None
        for row in rows
    )
    exact_matches = sum(bool(row.get("amount_semantic_exact")) for row in rows)
    return {
        "diagnostic_only": True,
        "affects_acceptance": False,
        "normalization": (
            "strip surrounding whitespace; accept strict v8 CNY display grammar; remove ¥/￥, optional "
            "currency space and valid thousands separators; compare canonical digits with Decimal"
        ),
        "records": records,
        "reference_parseable_records": reference_parseable,
        "candidate_parseable_records": candidate_parseable,
        "comparable_records": comparable,
        "exact_matches": exact_matches,
        "exact_match": exact_matches / records if records else None,
    }


def score_results(
    *,
    records_path: Path,
    results_root: Path,
    model_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    input_list_path: Path | None = None,
    input_list_sha256: str | None = None,
    split: str = "val",
    amount_floor: float = DEFAULT_AMOUNT_FLOOR,
    time_floor: float = DEFAULT_TIME_FLOOR,
    payment_floor: float = DEFAULT_PAYMENT_FLOOR,
    recipient_floor: float = DEFAULT_RECIPIENT_FLOOR,
    status_floor: float | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    records_path = records_path.resolve()
    results_root = results_root.resolve()
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    manifest_path = (manifest_path or (results_root / "inference_manifest.json")).resolve()
    input_list_path = input_list_path.resolve() if input_list_path is not None else None
    for name, floor in {
        "amount_floor": amount_floor,
        "time_floor": time_floor,
        "payment_floor": payment_floor,
        "recipient_floor": recipient_floor,
        "status_floor": status_floor,
    }.items():
        if floor is None:
            continue
        if isinstance(floor, bool) or not math.isfinite(float(floor)) or not 0.0 <= float(floor) <= 1.0:
            raise EvaluationInputError(f"{name} must be between 0 and 1")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise EvaluationInputError("limit must be a non-negative integer")
    if (input_list_path is None) != (input_list_sha256 is None):
        raise EvaluationInputError(
            "input_list_path and input_list_sha256 must be provided together"
        )
    if limit > 0 and input_list_path is None:
        raise EvaluationInputError(
            "partial pilot scoring requires a hash-bound explicit input list"
        )
    if limit > 0 and status_floor is None:
        raise EvaluationInputError(
            "partial pilot scoring requires the transfer_status floor and five-field coverage"
        )
    if not model_path.is_file():
        raise EvaluationInputError(f"Unified ONNX model not found: {model_path}")
    if not manifest_path.is_file():
        raise EvaluationInputError(f"ML.NET inference manifest not found: {manifest_path}")

    model_sha256 = _sha256(model_path)
    field_result_keys = dict(FIELD_RESULT_KEYS)
    if status_floor is not None:
        field_result_keys["transfer_status"] = STATUS_RESULT_KEY
    full_expected = _expected_receipts(
        records_path,
        split=split,
        include_status_text=status_floor is not None,
    )
    full_expected_count = len(full_expected)
    input_selection: dict[str, Any] | None = None
    if input_list_path is not None:
        input_sources, input_keys, observed_input_sha256 = _load_bound_input_list(
            input_list_path,
            expected_sha256=str(input_list_sha256),
        )
        unknown_sources = [
            source for source, key in zip(input_sources, input_keys) if key not in full_expected
        ]
        if unknown_sources:
            raise EvaluationInputError(
                f"explicit input list contains {len(unknown_sources)} source(s) outside the "
                f"{split} reference set: {unknown_sources[0]!r}"
            )
        if limit > 0:
            if len(input_keys) != limit:
                raise EvaluationInputError(
                    f"pilot explicit input list has {len(input_keys)} source(s), expected exactly {limit}"
                )
            deterministic_keys, quotas, selected_field_counts = _pilot_selection(
                full_expected,
                limit=limit,
            )
            if input_keys != deterministic_keys:
                raise EvaluationInputError(
                    "pilot explicit input list does not match the deterministic field-quota "
                    "selection order"
                )
            selection_order = PILOT_SELECTION_ORDER
        else:
            deterministic_keys = list(full_expected)
            if input_keys != deterministic_keys:
                raise EvaluationInputError(
                    "full-split explicit input list must contain every source exactly once in "
                    "records manifest order; explicit subsets are forbidden"
                )
            quotas = None
            selected_field_counts = {
                field: sum(
                    _has_reference(full_expected[key], field) for key in deterministic_keys
                )
                for field in field_result_keys
            }
            selection_order = FULL_SELECTION_ORDER
        expected = {key: full_expected[key] for key in input_keys}
        input_selection = {
            "path": input_list_path.as_posix(),
            "sha256": observed_input_sha256,
            "records": len(input_keys),
            "selection_order": selection_order,
            "field_quotas": quotas,
            "field_reference_counts": selected_field_counts,
        }
    else:
        expected = full_expected
        selection_order = FULL_SELECTION_ORDER
    results, artifact_audit, integrity_failures = _load_manifest_results(
        manifest_path=manifest_path,
        model_sha256=model_sha256,
        allowed_source_keys=set(expected) if input_selection is not None else None,
    )

    comparisons: list[dict[str, Any]] = []
    missing_result_sources: list[str] = []
    missing_field_sources: dict[str, list[str]] = {field: [] for field in field_result_keys}
    fully_scored_receipts = 0
    matched_receipts = 0
    for source_key, truth in expected.items():
        result_entry = results.get(source_key)
        result = result_entry["payload"] if result_entry is not None else None
        if result_entry is None:
            missing_result_sources.append(str(truth["source"]))
        else:
            matched_receipts += 1
        receipt_complete = result is not None
        references = truth["references"]
        for field, reference in references.items():
            result_key = field_result_keys[field]
            if result is None:
                candidate, missing_reason = None, "result_missing"
            else:
                candidate, missing_reason = _candidate(result, result_key)
            candidate_present = candidate is not None
            if not candidate_present:
                missing_field_sources[field].append(str(truth["source"]))
                receipt_complete = False
            comparison = {
                "schema_version": SCHEMA_VERSION,
                "kind": "receipt_mlnet_unified_comparison_v1",
                "id": truth["id"],
                "group_id": truth["group_id"],
                "split": split,
                "source": truth["source"],
                "teacher_result_json": truth.get("teacher_result_json"),
                "result_json": result_entry["path"].resolve().as_posix() if result_entry is not None else None,
                "manifest_status": result_entry["status"] if result_entry is not None else None,
                "field": field,
                "reference_text": reference,
                "candidate_text": candidate,
                "candidate_present": candidate_present,
                "missing_reason": missing_reason,
                "raw_exact": candidate == reference if candidate_present else False,
                "unified_model_sha256": result_entry["model_sha256"] if result_entry is not None else None,
            }
            comparison.update(truth["reference_diagnostics"].get(field, {}))
            comparison.update(
                _result_field_diagnostics(result, result_key=result_key, detector_label=field)
            )
            if field == "transfer_status":
                reference_status_class = comparison.get("reference_status_class")
                candidate_status_class = normalize_status(candidate) if candidate_present else None
                comparison.update(
                    {
                        "candidate_status_class": candidate_status_class,
                        "non_success_to_success": (
                            reference_status_class in {"pending", "failed"}
                            and candidate_status_class == "success"
                        ),
                    }
                )
            if field == "amount":
                reference_decimal = _amount_semantic_decimal(reference)
                candidate_decimal = _amount_semantic_decimal(candidate)
                comparison.update(
                    {
                        "reference_amount_decimal": _amount_semantic_text(reference_decimal),
                        "candidate_amount_decimal": _amount_semantic_text(candidate_decimal),
                        "amount_semantic_exact": (
                            reference_decimal is not None
                            and candidate_decimal is not None
                            and reference_decimal == candidate_decimal
                        ),
                    }
                )
            comparisons.append(comparison)
        if receipt_complete:
            fully_scored_receipts += 1

    comparisons.sort(key=lambda row: (str(row["field"]), str(row["id"]), str(row["source"])))
    by_field = {}
    for field in field_result_keys:
        field_rows = [row for row in comparisons if row["field"] == field]
        by_field[field] = (
            _status_metrics(field_rows) if field == "transfer_status" else _field_metrics(field_rows)
        )
    amount_semantic = _amount_semantic_metrics(
        [row for row in comparisons if row["field"] == "amount"]
    )
    expected_keys = set(expected)
    extra_manifest_sources = sorted(
        str(entry["source"]) for key, entry in results.items() if key not in expected_keys
    )
    coverage = {
        "expected_receipts": len(expected),
        "matched_result_receipts": matched_receipts,
        "result_coverage": matched_receipts / len(expected),
        "fully_scored_receipts": fully_scored_receipts,
        "fully_scored_coverage": fully_scored_receipts / len(expected),
        "extra_manifest_sources": extra_manifest_sources,
        "by_field": {
            field: {
                "references": metrics["records"],
                "candidates": metrics["candidate_records"],
                "missing": metrics["missing_candidate_records"],
                "coverage": metrics["candidate_coverage"],
            }
            for field, metrics in by_field.items()
        },
    }
    missing = {
        "result_receipts": len(missing_result_sources),
        "result_sources": sorted(missing_result_sources),
        "field_candidates": {
            field: {"records": len(sources), "sources": sorted(sources)}
            for field, sources in missing_field_sources.items()
        },
        "manifest_result_files": artifact_audit["missing_result_files"],
        "invalid_result_files": artifact_audit["invalid_result_files"],
    }

    floors = {
        "amount": float(amount_floor),
        "time": float(time_floor),
        "payment_method_field": float(payment_floor),
        "recipient_field": float(recipient_floor),
    }
    if status_floor is not None:
        floors["transfer_status"] = float(status_floor)
    failures = list(integrity_failures)
    if extra_manifest_sources:
        failures.append(
            f"manifest contains {len(extra_manifest_sources)} source(s) outside the {split} reference set"
        )
    if missing_result_sources:
        failures.append(
            f"receipt result_coverage={coverage['result_coverage']:.4f} < 1.0000 "
            f"({len(missing_result_sources)} missing)"
        )
    for field, floor in floors.items():
        metrics = by_field[field]
        observed = metrics["raw_exact_match"]
        candidate_coverage = metrics["candidate_coverage"]
        if observed is None:
            failures.append(f"{field}: no {split} reference labels")
            continue
        if candidate_coverage is None or float(candidate_coverage) < 1.0:
            rendered = "n/a" if candidate_coverage is None else f"{float(candidate_coverage):.4f}"
            failures.append(f"{field}: candidate_coverage={rendered} < 1.0000")
        if float(observed) < floor:
            failures.append(f"{field}: raw_exact_match={float(observed):.4f} < {floor:.4f}")
    status_metrics = by_field.get("transfer_status")
    if (
        isinstance(status_metrics, Mapping)
        and int(status_metrics["non_success_truth_records"]) > 0
        and int(status_metrics["non_success_to_success"]) > 0
    ):
        failures.append(
            "transfer_status: "
            f"non_success_to_success={int(status_metrics['non_success_to_success'])} > 0"
        )

    sample_thresholds_passed = not failures
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_mlnet_unified_candidate_evaluation_v1",
        "records": records_path.as_posix(),
        "records_sha256": _sha256(records_path),
        "results_root": results_root.as_posix(),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "input_selection": input_selection,
        "evaluation_split": split,
        "model": model_path.as_posix(),
        "model_sha256": model_sha256,
        "artifact_audit": artifact_audit,
        "by_field": by_field,
        "amount_semantic": amount_semantic,
        "floors": floors,
        "coverage": coverage,
        "missing": missing,
        "accepted": sample_thresholds_passed,
        "failures": failures,
        "acceptance": {
            "passed": sample_thresholds_passed,
            "failures": failures,
            "min_amount_exact_match": float(amount_floor),
            "min_time_exact_match": float(time_floor),
            "min_payment_exact_match": float(payment_floor),
            "min_recipient_exact_match": float(recipient_floor),
            "min_status_exact_match": float(status_floor) if status_floor is not None else None,
            "max_non_success_to_success": (
                0
                if isinstance(status_metrics, Mapping)
                and int(status_metrics["non_success_truth_records"]) > 0
                else None
            ),
        },
        "warning": (
            "This compares ML.NET candidate text with unified manifest labels. Paddle-derived labels are not "
            "independently verified business truth, and the artifact's business values remain review-only."
        ),
    }
    if limit > 0:
        # A subset is useful for fast diagnostics, but it must never be
        # consumable as formal delivery evidence.  Keep threshold failures
        # separate so the CLI can still return 0/1 for the pilot itself while
        # all delivery-acceptance flags remain fail-closed.
        summary["kind"] = "receipt_mlnet_unified_candidate_partial_pilot_evaluation_v1"
        summary["evaluation_scope"] = {
            "kind": "partial_pilot",
            "requested_limit": limit,
            "evaluated_expected_receipts": len(expected),
            "full_split_expected_receipts": full_expected_count,
            "input_list_path": input_selection["path"] if input_selection is not None else None,
            "input_list_sha256": (
                input_selection["sha256"] if input_selection is not None else None
            ),
            "selection_order": selection_order,
            "formal_delivery_gate": False,
        }
        summary["formal_delivery_gate"] = False
        summary["pilot_thresholds_passed"] = sample_thresholds_passed
        summary["accepted"] = False
        summary["acceptance"]["passed"] = False
        summary["acceptance"]["formal_delivery_gate"] = False
        summary["acceptance"]["pilot_thresholds_passed"] = sample_thresholds_passed
        summary["warning"] = (
            "partial_pilot: a deterministic five-field-covered selection of "
            f"{len(expected)} from {full_expected_count} expected {split} receipt source(s) was evaluated; "
            "formal_delivery_gate=false and this report cannot be accepted as delivery evidence. "
            "It compares ML.NET candidate text with unified manifest labels; Paddle-derived labels are not "
            "independently verified business truth, and business values remain review-only."
        )
    else:
        # An unbounded invocation evaluates every expected receipt in the
        # requested split.  Keep this explicit so release automation cannot
        # mistake a passing partial pilot for formal evidence.
        summary["evaluation_scope"] = {
            "kind": "full_split",
            "requested_limit": None,
            "evaluated_expected_receipts": len(expected),
            "full_split_expected_receipts": full_expected_count,
            "input_list_path": input_selection["path"] if input_selection is not None else None,
            "input_list_sha256": (
                input_selection["sha256"] if input_selection is not None else None
            ),
            "selection_order": selection_order,
            "formal_delivery_gate": sample_thresholds_passed,
        }
        summary["formal_delivery_gate"] = sample_thresholds_passed
        summary["acceptance"]["formal_delivery_gate"] = sample_thresholds_passed
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write unique source paths for one unified-manifest split")
    prepare.add_argument("--records", type=Path, required=True, help="unified_fields.jsonl")
    prepare.add_argument("--output", type=Path, required=True, help="UTF-8 newline-delimited input list")
    prepare.add_argument("--split", default="val")
    prepare.add_argument(
        "--limit",
        type=_limit,
        default=0,
        help="write exactly N deterministic five-field-covered pilot sources; 0 writes all",
    )

    score = subparsers.add_parser("score", help="score ML.NET result JSON against v12/v13 manifest references")
    score.add_argument("--records", type=Path, required=True, help="unified_fields.jsonl")
    score.add_argument("--results", type=Path, required=True, help="ML.NET output root")
    score.add_argument("--manifest", type=Path, help="defaults to RESULTS/inference_manifest.json")
    score.add_argument(
        "--input-list",
        type=Path,
        help="hash-bound explicit input list; required for a partial pilot",
    )
    score.add_argument(
        "--input-list-sha256",
        help="expected SHA-256 of --input-list",
    )
    score.add_argument("--model", type=Path, required=True, help="the delivered v12/v13 unified ONNX")
    score.add_argument("--output", type=Path, required=True, help="evaluation output directory")
    score.add_argument("--split", default="val")
    score.add_argument("--amount-floor", type=_floor, default=DEFAULT_AMOUNT_FLOOR)
    score.add_argument("--time-floor", type=_floor, default=DEFAULT_TIME_FLOOR)
    score.add_argument("--payment-floor", type=_floor, default=DEFAULT_PAYMENT_FLOOR)
    score.add_argument("--recipient-floor", type=_floor, default=DEFAULT_RECIPIENT_FLOOR)
    score.add_argument(
        "--status-floor",
        type=_floor,
        help="enable v13 visible transfer-status raw exact scoring at this fixed floor",
    )
    score.add_argument(
        "--limit",
        type=_limit,
        default=0,
        help="evaluate an explicit deterministic N-source pilot; 0 evaluates the full split",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_input_list(
                records_path=args.records,
                output_path=args.output,
                split=args.split,
                limit=args.limit,
            )
            print(
                f"Wrote {report['unique_sources']} unique {report['split']} source(s) "
                f"from {report['records']} record(s) to {args.output}; "
                f"selection_order={report['selection_order']}; sha256={report['output_sha256']}"
            )
            return 0
        summary = score_results(
            records_path=args.records,
            results_root=args.results,
            manifest_path=args.manifest,
            input_list_path=args.input_list,
            input_list_sha256=args.input_list_sha256,
            model_path=args.model,
            output_dir=args.output,
            split=args.split,
            amount_floor=args.amount_floor,
            time_floor=args.time_floor,
            payment_floor=args.payment_floor,
            recipient_floor=args.recipient_floor,
            status_floor=args.status_floor,
            limit=args.limit,
        )
        metrics = summary["by_field"]
        rendered = ", ".join(
            f"{field}={value['raw_exact_matches']}/{value['records']}="
            + (f"{value['raw_exact_match']:.2%}" if value["raw_exact_match"] is not None else "n/a")
            for field, value in metrics.items()
        )
        print(f"Wrote ML.NET unified evaluation to {args.output} ({rendered}); accepted={summary['accepted']}")
        if summary["failures"]:
            print("ML.NET unified delivery gate failed:\n- " + "\n- ".join(summary["failures"]), file=sys.stderr)
            return 1
        return 0
    except (EvaluationInputError, OSError) as exception:
        parser.error(str(exception))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
