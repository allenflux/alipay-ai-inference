#!/usr/bin/env python3
"""Emit a strict, read-only one-line audit of missing hybrid recipients."""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class AuditError(RuntimeError):
    """Raised when A/B evidence is incomplete, ambiguous, or unbound."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _loads(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exception:
        raise AuditError(f"invalid JSON at {location}: {exception}") from exception


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
                    f"invalid {description} {path}:{line_number}: expected one object"
                )
            rows.append(value)
    if not rows:
        raise AuditError(f"{description} is empty: {path}")
    return rows


def _source_key(value: str) -> str:
    if WINDOWS_ABSOLUTE_PATH.match(value) or "\\" in value:
        return "windows:" + ntpath.normcase(ntpath.normpath(value)).replace("\\", "/")
    return "posix:" + os.path.normpath(os.path.abspath(value))


def _nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{description} must be a non-empty string")
    return value


def _comparison_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _load_jsonl(path, description="A/B comparisons")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        source = _nonempty_string(
            row.get("source"), description=f"comparison[{index}].source"
        )
        if not isinstance(row.get("invariant"), bool):
            raise AuditError(f"comparison[{index}].invariant must be a boolean")
        failures = row.get("failures")
        if not isinstance(failures, list) or not all(
            isinstance(failure, str) for failure in failures
        ):
            raise AuditError(f"comparison[{index}].failures must be a string array")
        candidate = row.get("recipient_candidate")
        if candidate is not None and not isinstance(candidate, str):
            raise AuditError(
                f"comparison[{index}].recipient_candidate must be a string or null"
            )
        key = _source_key(source)
        if key in indexed:
            raise AuditError(f"duplicate comparison source: {source!r}")
        indexed[key] = row
    return rows, indexed


def _contained_result_path(raw: str, *, run_root: Path, manifest_path: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exception:
        raise AuditError(f"manifest result file is missing: {candidate}") from exception
    if not resolved.is_file():
        raise AuditError(f"manifest result is not a file: {resolved}")
    try:
        resolved.relative_to(run_root)
    except ValueError as exception:
        raise AuditError(f"manifest result path escapes run root: {resolved}") from exception
    return resolved


def _manifest_results(run_directory: Path, *, label: str) -> dict[str, dict[str, Any]]:
    run_root = run_directory.resolve(strict=True)
    if not run_root.is_dir():
        raise AuditError(f"{label} run root is not a directory: {run_root}")
    manifest_path = run_root / "inference_manifest.json"
    payload = _load_json(manifest_path, description=f"{label} inference manifest")
    if not isinstance(payload, list) or not payload:
        raise AuditError(f"{label} inference manifest must be a non-empty array")
    results: dict[str, dict[str, Any]] = {}
    result_paths: set[Path] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise AuditError(f"{label} manifest[{index}] must be an object")
        source = _nonempty_string(
            item.get("source"), description=f"{label} manifest[{index}].source"
        )
        if item.get("status") not in {"written", "skipped_existing"}:
            raise AuditError(
                f"{label} manifest source {source!r} has incomplete status "
                f"{item.get('status')!r}"
            )
        key = _source_key(source)
        if key in results:
            raise AuditError(f"duplicate {label} manifest source: {source!r}")
        raw_result = _nonempty_string(
            item.get("result"), description=f"{label} manifest[{index}].result"
        )
        result_path = _contained_result_path(
            raw_result,
            run_root=run_root,
            manifest_path=manifest_path,
        )
        if result_path in result_paths:
            raise AuditError(f"duplicate {label} result path: {result_path}")
        result_paths.add(result_path)
        result = _load_json(result_path, description=f"{label} result")
        if not isinstance(result, dict):
            raise AuditError(f"{label} result must be an object: {result_path}")
        result_source = _nonempty_string(
            result.get("source"), description=f"{label} result source"
        )
        if _source_key(result_source) != key:
            raise AuditError(
                f"{label} manifest/result source mismatch: {source!r} != {result_source!r}"
            )
        results[key] = result
    return results


def _recipient_field(result: Mapping[str, Any]) -> dict[str, Any] | None:
    fields = result.get("fields")
    if not isinstance(fields, Mapping):
        return None
    recipient = fields.get("recipient")
    return dict(recipient) if isinstance(recipient, Mapping) else None


def _recipient_detection(result: Mapping[str, Any]) -> dict[str, Any] | None:
    detections = result.get("detections")
    if not isinstance(detections, list):
        return None
    recipients = [
        dict(detection)
        for detection in detections
        if isinstance(detection, Mapping) and detection.get("label") == "recipient_field"
    ]
    if len(recipients) > 1:
        raise AuditError("result contains duplicate recipient_field detections")
    return recipients[0] if recipients else None


def _model_contracts(result: Mapping[str, Any]) -> dict[str, Any] | None:
    contracts = result.get("model_contracts")
    return dict(contracts) if isinstance(contracts, Mapping) else None


def _ppocr_evidence(recipient: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if recipient is None:
        return None
    evidence = {
        key.removeprefix("hybrid_ocr_"): value
        for key, value in recipient.items()
        if key.startswith("hybrid_ocr_")
    }
    return evidence or None


def audit(root: Path) -> dict[str, Any]:
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as exception:
        raise AuditError(f"A/B root does not exist: {root}") from exception
    if not root.is_dir():
        raise AuditError(f"A/B root is not a directory: {root}")
    comparison_rows, comparisons = _comparison_rows(
        root / "comparison" / "comparisons.jsonl"
    )
    baseline = _manifest_results(root / "baseline-v13", label="baseline")
    hybrid = _manifest_results(root / "hybrid-recipient", label="hybrid")
    comparison_keys = set(comparisons)
    for label, results in (("baseline", baseline), ("hybrid", hybrid)):
        if set(results) != comparison_keys:
            missing = comparison_keys - set(results)
            extra = set(results) - comparison_keys
            raise AuditError(
                f"{label} manifest source set differs from comparisons: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    findings: list[dict[str, Any]] = []
    invariant_failures = 0
    recipient_missing = 0
    for row in comparison_rows:
        candidate = row.get("recipient_candidate")
        missing_candidate = not isinstance(candidate, str) or not candidate.strip()
        if not row["invariant"]:
            invariant_failures += 1
        if missing_candidate:
            recipient_missing += 1
        if row["invariant"] and not missing_candidate:
            continue
        key = _source_key(row["source"])
        baseline_result = baseline[key]
        hybrid_result = hybrid[key]
        baseline_recipient = _recipient_field(baseline_result)
        hybrid_recipient = _recipient_field(hybrid_result)
        findings.append(
            {
                "source": row["source"],
                "invariant": row["invariant"],
                "recipient_candidate": candidate,
                "failures": row["failures"],
                "baseline_recipient_field": baseline_recipient,
                "hybrid_recipient_field": hybrid_recipient,
                "baseline_recipient_detection": _recipient_detection(baseline_result),
                "hybrid_recipient_detection": _recipient_detection(hybrid_result),
                "baseline_model_contracts": _model_contracts(baseline_result),
                "hybrid_model_contracts": _model_contracts(hybrid_result),
                "hybrid_ppocr_evidence": _ppocr_evidence(hybrid_recipient),
            }
        )

    return {
        "schema_version": 1,
        "kind": "receipt_mlnet_hybrid_missing_audit_v1",
        "ab_root": root.as_posix(),
        "records": len(comparison_rows),
        "invariant_failure_records": invariant_failures,
        "recipient_missing_records": recipient_missing,
        "flagged_records": len(findings),
        "findings": findings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="hybrid CPU A/B output root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = audit(args.root)
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        return 0
    except (AuditError, OSError, UnicodeError) as exception:
        print(
            json.dumps(
                {"kind": "receipt_mlnet_hybrid_missing_audit_error_v1", "error": str(exception)},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
