#!/usr/bin/env python3
"""Freeze the exact formal time-missing 339 set for CPU layout shadow.

This package-free producer consumes only an already-published formal
missing-fields audit. It never runs OCR and never changes the audit or source
images. The two output files are published together by one directory rename.
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
from uuid import uuid4


FORMAL_RECORDS = 10016
TIME_MISSING_RECORDS = 339
AUDIT_SUMMARY_KIND = "receipt_mlnet_formal_missing_fields_audit_summary_v1"
AUDIT_FINDING_KIND = "receipt_mlnet_formal_missing_fields_audit_finding_v1"
SELECTION_KIND = "receipt_mlnet_layout_shadow_time_selection_v1"
SELECTION_ORDER = "formal_audit_missing_by_field_time_sources_order"


class SelectionError(ValueError):
    """Raised when the formal audit cannot authorize the frozen selection."""


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
        raise SelectionError(f"invalid JSON at {location}: {error}") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path, *, description: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise SelectionError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise SelectionError(f"{description} is not a file: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise SelectionError(f"cannot read {description}: {resolved}: {error}") from error


def _identity_from_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _file_identity(path: Path, *, description: str) -> dict[str, Any]:
    payload = _read_bytes(path, description=description)
    return _identity_from_bytes(path, payload)


def _load_summary(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SelectionError(f"formal audit summary is not UTF-8: {path}") from error
    value = _loads(text, location=str(path))
    if not isinstance(value, dict):
        raise SelectionError("formal audit summary must be one JSON object")
    return value


def _load_findings(path: Path, payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SelectionError(f"formal audit findings are not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise SelectionError(f"formal audit findings contain a blank line at {path}:{line_number}")
        value = _loads(line, location=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise SelectionError(f"formal audit finding must be an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise SelectionError(f"formal audit findings are empty: {path}")
    return rows


def _require_int(value: object, *, description: str) -> int:
    if type(value) is not int:
        raise SelectionError(f"{description} must be an integer")
    return value


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _resolve_source(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SelectionError(f"{description} must be a non-empty string")
    raw = Path(value)
    if not raw.is_absolute():
        raise SelectionError(f"{description} must be an absolute path: {value!r}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise SelectionError(f"{description} does not exist: {value}") from error
    if not resolved.is_file():
        raise SelectionError(f"{description} is not a file: {resolved}")
    return resolved


def _validate_summary(summary: Mapping[str, Any]) -> list[object]:
    if summary.get("schema_version") != 1 or summary.get("kind") != AUDIT_SUMMARY_KIND:
        raise SelectionError("formal audit summary schema/kind is unsupported")
    if summary.get("read_only_existing_results") is not True or summary.get("ocr_rerun") is not False:
        raise SelectionError("formal audit must be read-only with ocr_rerun=false")
    if summary.get("formal_required") is not True:
        raise SelectionError("formal audit was not produced with --require-formal")
    if _require_int(summary.get("records"), description="formal audit records") != FORMAL_RECORDS:
        raise SelectionError(
            f"formal audit must cover {FORMAL_RECORDS} records"
        )
    missing = summary.get("missing_by_field")
    time = missing.get("time") if isinstance(missing, Mapping) else None
    if not isinstance(time, Mapping):
        raise SelectionError("formal audit summary has no time missing-field evidence")
    if _require_int(time.get("records"), description="time missing records") != TIME_MISSING_RECORDS:
        raise SelectionError(f"formal audit time missing count must be {TIME_MISSING_RECORDS}")
    if _require_int(
        time.get("reference_missing_records"),
        description="time reference-missing records",
    ) != TIME_MISSING_RECORDS:
        raise SelectionError(
            f"formal audit time reference-missing count must be {TIME_MISSING_RECORDS}"
        )
    if _require_int(
        time.get("reference_present_records"),
        description="time reference-present records",
    ) != 0:
        raise SelectionError("formal audit time selection must have zero external references")
    sources = time.get("sources")
    if not isinstance(sources, list) or len(sources) != TIME_MISSING_RECORDS:
        raise SelectionError("formal audit time sources must contain exactly 339 entries")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("summary") != "summary.json" or artifacts.get(
        "findings"
    ) != "findings.jsonl":
        raise SelectionError("formal audit artifact names are unsupported")
    return sources


def _time_finding_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    selected: set[str] = set()
    seen_findings: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != 1 or row.get("kind") != AUDIT_FINDING_KIND:
            raise SelectionError(f"formal audit finding[{index}] schema/kind is unsupported")
        source = _resolve_source(row.get("source"), description=f"finding[{index}].source")
        key = _path_key(source)
        if key in seen_findings:
            raise SelectionError(f"formal audit has duplicate finding source: {source}")
        seen_findings.add(key)
        missing_fields = row.get("missing_fields")
        if not isinstance(missing_fields, list) or not missing_fields or not all(
            isinstance(field, str) and field for field in missing_fields
        ) or len(missing_fields) != len(set(missing_fields)):
            raise SelectionError(f"formal audit finding[{index}] missing_fields is invalid")
        if "time" not in missing_fields:
            continue

        reference_map = row.get("reference_present_by_field")
        if not isinstance(reference_map, Mapping) or reference_map.get("time") is not False:
            raise SelectionError(f"formal audit time finding has an external reference: {source}")
        by_missing = row.get("by_missing_field")
        time = by_missing.get("time") if isinstance(by_missing, Mapping) else None
        if not isinstance(time, Mapping):
            raise SelectionError(f"formal audit time finding has no field evidence: {source}")
        if time.get("reference_present") is not False or time.get("reference_text") is not None:
            raise SelectionError(f"formal audit time finding reference evidence disagrees: {source}")
        if time.get("score_comparison") is not None:
            raise SelectionError(f"reference-absent time finding unexpectedly has a score comparison: {source}")
        selected.add(key)
    if len(selected) != TIME_MISSING_RECORDS:
        raise SelectionError(
            f"formal audit findings must contain exactly {TIME_MISSING_RECORDS} time-missing sources, found {len(selected)}"
        )
    return selected


def _source_closure(identities: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        path = Path(str(identity["path"]))
        digest.update(
            (
                f"{_path_key(path)}\0{identity['path']}\0{identity['sha256']}\0"
                f"{identity['size_bytes']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _assert_bindings_current(bindings: Mapping[str, Any]) -> None:
    for description, expected in (
        ("formal audit summary", bindings["summary"]),
        ("formal audit findings", bindings["findings"]),
    ):
        observed = _file_identity(Path(str(expected["path"])), description=description)
        if observed != expected:
            raise SelectionError(f"{description} changed while selection was being published")
    for expected in bindings["sources"]:
        observed = _file_identity(Path(str(expected["path"])), description="layout source image")
        if observed != expected:
            raise SelectionError(
                f"layout source image changed while selection was being published: {expected['path']}"
            )


def prepare_selection(
    audit_directory: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    try:
        audit = audit_directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise SelectionError(f"formal audit directory does not exist: {audit_directory}") from error
    if not audit.is_dir():
        raise SelectionError(f"formal audit path is not a directory: {audit}")

    summary_path = audit / "summary.json"
    findings_path = audit / "findings.jsonl"
    summary_bytes = _read_bytes(summary_path, description="formal audit summary")
    findings_bytes = _read_bytes(findings_path, description="formal audit findings")
    summary_identity = _identity_from_bytes(summary_path, summary_bytes)
    findings_identity = _identity_from_bytes(findings_path, findings_bytes)
    summary = _load_summary(summary_path, summary_bytes)
    findings = _load_findings(findings_path, findings_bytes)
    summary_sources = _validate_summary(summary)
    finding_keys = _time_finding_keys(findings)

    ordered_sources: list[str] = []
    source_identities: list[dict[str, Any]] = []
    summary_keys: set[str] = set()
    for index, value in enumerate(summary_sources):
        source = _resolve_source(value, description=f"summary time source[{index}]")
        key = _path_key(source)
        if key in summary_keys:
            raise SelectionError(f"formal audit summary has duplicate time source: {source}")
        summary_keys.add(key)
        payload = _read_bytes(source, description="layout source image")
        ordered_sources.append(str(source))
        source_identities.append(_identity_from_bytes(source, payload))
    if summary_keys != finding_keys:
        raise SelectionError(
            "formal audit summary/findings time source sets differ: "
            f"summary_only={len(summary_keys-finding_keys)} findings_only={len(finding_keys-summary_keys)}"
        )

    inputs_bytes = ("\n".join(ordered_sources) + "\n").encode("utf-8")
    input_identity = {
        "relative_path": "inputs.txt",
        "sha256": _sha256_bytes(inputs_bytes),
        "size_bytes": len(inputs_bytes),
        "records": len(ordered_sources),
        "encoding": "utf-8-no-bom",
        "terminal_newline": True,
    }
    selection = {
        "schema_version": 1,
        "kind": SELECTION_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "selection_field": "time",
        "selection_order": SELECTION_ORDER,
        "records": len(ordered_sources),
        "external_reference_present_records": 0,
        "external_reference_missing_records": TIME_MISSING_RECORDS,
        "formal_audit": {
            "directory": str(audit),
            "summary": summary_identity,
            "findings": findings_identity,
            "records": FORMAL_RECORDS,
        },
        "input_list": input_identity,
        "source_files": source_identities,
        "source_closure_sha256": _source_closure(source_identities),
        "source_total_bytes": sum(int(identity["size_bytes"]) for identity in source_identities),
    }
    bindings = {
        "summary": summary_identity,
        "findings": findings_identity,
        "sources": source_identities,
    }
    _assert_bindings_current(bindings)
    return selection, inputs_bytes, bindings


def write_atomic(
    output_directory: Path,
    *,
    selection: Mapping[str, Any],
    inputs_bytes: bytes,
    bindings: Mapping[str, Any],
) -> None:
    output = output_directory.absolute()
    if not output.name or os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite layout shadow selection: {output}")
    input_contract = selection.get("input_list")
    if not isinstance(input_contract, Mapping) or input_contract.get("sha256") != _sha256_bytes(
        inputs_bytes
    ) or input_contract.get("size_bytes") != len(inputs_bytes):
        raise SelectionError("selection input-list identity disagrees with inputs bytes")
    if selection.get("diagnostic_only") is not True or selection.get("formal_delivery_gate") is not False:
        raise SelectionError("selection diagnostic/formal gate flags are invalid")

    _assert_bindings_current(bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        (stage / "inputs.txt").write_bytes(inputs_bytes)
        (stage / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _assert_bindings_current(bindings)
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite layout shadow selection: {output}")
        stage.replace(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selection, inputs_bytes, bindings = prepare_selection(args.audit_directory)
        write_atomic(
            args.output_directory,
            selection=selection,
            inputs_bytes=inputs_bytes,
            bindings=bindings,
        )
    except (SelectionError, FileExistsError, OSError, UnicodeError) as error:
        print(f"Layout shadow selection failed: {error}")
        return 2
    print(
        f"layout_shadow_selection records={selection['records']} "
        f"input_sha256={selection['input_list']['sha256']} "
        f"output={args.output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
