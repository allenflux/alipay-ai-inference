#!/usr/bin/env python3
"""Safely launch and verify the fixed 339-record CPU LayoutShadow CLI.

The wrapper computes the input-list SHA-256 from bytes and invokes the .NET
application with an argument array (never a shell command).  A zero child exit
code is not sufficient: the fresh atomic output is independently checked
before this wrapper reports success.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


EXPECTED_RECORDS = 339
SUMMARY_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_summary_v1"
RECORD_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_record_v1"
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LayoutShadowRunError(ValueError):
    """Raised when launch inputs or published output violate the contract."""


class LayoutShadowProcessError(LayoutShadowRunError):
    """Raised when the fixed LayoutShadow child exits unsuccessfully."""

    def __init__(self, returncode: int):
        super().__init__(f"LayoutShadow process exited with code {returncode}")
        self.returncode = returncode


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
        raise LayoutShadowRunError(f"invalid JSON at {location}: {error}") from error


def _read_bytes(path: Path, *, description: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayoutShadowRunError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise LayoutShadowRunError(f"{description} is not a file: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise LayoutShadowRunError(f"cannot read {description}: {resolved}: {error}") from error


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise LayoutShadowRunError(f"{description} is not UTF-8: {path}") from error
    value = _loads(text, location=str(path))
    if not isinstance(value, dict):
        raise LayoutShadowRunError(f"{description} must contain one JSON object")
    return value, payload


def _load_jsonl(path: Path, *, description: str) -> tuple[list[dict[str, Any]], bytes]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise LayoutShadowRunError(f"{description} is not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise LayoutShadowRunError(f"blank line in {description}: {path}:{line_number}")
        value = _loads(line, location=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise LayoutShadowRunError(f"{description} row must be an object")
        rows.append(value)
    return rows, payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_int(value: object, expected: int, *, description: str) -> None:
    if type(value) is not int or value != expected:
        raise LayoutShadowRunError(f"{description} must be {expected}, found {value!r}")


def _require_flag(value: object, expected: bool, *, description: str) -> None:
    if value is not expected:
        raise LayoutShadowRunError(
            f"{description} must be {str(expected).lower()}, found {value!r}"
        )


def _path_key(path: Path | str, *, strict: bool) -> str:
    resolved = Path(path).resolve(strict=strict)
    return os.path.normcase(os.path.normpath(str(resolved)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_disjoint_output(
    output: Path, *, app: Path, bundle: Path, input_list: Path
) -> Path:
    resolved = output.resolve(strict=False)
    if os.path.lexists(os.fspath(resolved)):
        raise LayoutShadowRunError(f"output must be fresh and absent: {resolved}")
    app = app.resolve(strict=True)
    bundle = bundle.resolve(strict=True)
    input_list = input_list.resolve(strict=True)
    protected = {
        "application directory": app.parent,
        "bundle": bundle,
        "input-list directory": input_list.parent,
    }
    for description, path in protected.items():
        if _is_within(resolved, path) or _is_within(path, resolved):
            raise LayoutShadowRunError(f"output overlaps {description}: {resolved} vs {path}")
    return resolved


def _parse_input_sources(input_list: Path, payload: bytes) -> list[str]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise LayoutShadowRunError("input list must be UTF-8 without BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LayoutShadowRunError("input list is not strict UTF-8") from error
    if not text.endswith("\n"):
        raise LayoutShadowRunError("input list must end with a newline")
    sources = text.splitlines()
    if len(sources) != EXPECTED_RECORDS:
        raise LayoutShadowRunError(
            f"input list must contain {EXPECTED_RECORDS} records, found {len(sources)}"
        )
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not source or source != source.strip():
            raise LayoutShadowRunError(
                f"input list source[{index}] is blank or has surrounding whitespace"
            )
        path = Path(source)
        if not path.is_absolute():
            raise LayoutShadowRunError(f"input list source[{index}] is not absolute: {source}")
        key = _path_key(path, strict=False)
        if key in seen:
            raise LayoutShadowRunError(f"input list contains duplicate source: {source}")
        seen.add(key)
    return sources


def _artifact_path(output: Path, contract: Mapping[str, Any]) -> Path:
    relative = contract.get("relative_path")
    if relative != "records.jsonl":
        raise LayoutShadowRunError(
            "LayoutShadow records artifact relative_path must be records.jsonl"
        )
    path = (output / relative).resolve(strict=True)
    try:
        path.relative_to(output.resolve(strict=True))
    except ValueError as error:
        raise LayoutShadowRunError("LayoutShadow records artifact escapes output") from error
    return path


def validate_output(
    *, output: Path, input_list: Path, input_bytes: bytes, input_sources: Sequence[str]
) -> dict[str, Any]:
    try:
        output = output.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayoutShadowRunError(f"LayoutShadow did not publish output: {output}") from error
    if not output.is_dir():
        raise LayoutShadowRunError(f"LayoutShadow output is not a directory: {output}")
    summary, _ = _load_json(output / "summary.json", description="LayoutShadow summary")
    if summary.get("schema_version") != 1 or summary.get("kind") != SUMMARY_KIND:
        raise LayoutShadowRunError("LayoutShadow summary schema/kind is unsupported")
    _require_flag(summary.get("diagnostic_only"), True, description="summary diagnostic_only")
    _require_flag(
        summary.get("formal_delivery_gate"), False, description="summary formal_delivery_gate"
    )
    _require_flag(
        summary.get("candidate_write_enabled"),
        False,
        description="summary candidate_write_enabled",
    )
    _require_int(
        summary.get("expected_records"), EXPECTED_RECORDS, description="summary expected_records"
    )
    _require_int(summary.get("records"), EXPECTED_RECORDS, description="summary records")
    _require_int(summary.get("errors"), 0, description="summary errors")
    if summary.get("execution_provider") != "cpu":
        raise LayoutShadowRunError("summary execution_provider must be cpu")

    input_contract = summary.get("input_list")
    if not isinstance(input_contract, Mapping):
        raise LayoutShadowRunError("summary input_list contract is missing")
    raw_input_path = input_contract.get("path")
    if not isinstance(raw_input_path, str) or _path_key(
        raw_input_path, strict=True
    ) != _path_key(input_list, strict=True):
        raise LayoutShadowRunError("summary input-list path differs from launcher input")
    expected_input_sha = _sha256(input_bytes)
    if input_contract.get("sha256") != expected_input_sha:
        raise LayoutShadowRunError("summary input-list SHA-256 differs from launcher bytes")
    _require_int(
        input_contract.get("size_bytes"), len(input_bytes), description="summary input-list size"
    )
    _require_int(
        input_contract.get("records"), EXPECTED_RECORDS, description="summary input-list records"
    )

    artifacts = summary.get("artifacts")
    records_contract = artifacts.get("records_jsonl") if isinstance(artifacts, Mapping) else None
    if not isinstance(records_contract, Mapping):
        raise LayoutShadowRunError("summary records_jsonl artifact contract is missing")
    records_path = _artifact_path(output, records_contract)
    records, records_bytes = _load_jsonl(records_path, description="LayoutShadow records")
    records_sha = records_contract.get("sha256")
    if not isinstance(records_sha, str) or LOWER_SHA256.fullmatch(records_sha) is None:
        raise LayoutShadowRunError("records artifact SHA-256 must be lowercase hexadecimal")
    if records_sha != _sha256(records_bytes):
        raise LayoutShadowRunError("records artifact SHA-256 differs from published bytes")
    _require_int(
        records_contract.get("size_bytes"),
        len(records_bytes),
        description="records artifact size",
    )
    if len(records) != EXPECTED_RECORDS:
        raise LayoutShadowRunError(
            f"records artifact must contain {EXPECTED_RECORDS} rows, found {len(records)}"
        )
    for index, (record, expected_source) in enumerate(
        zip(records, input_sources, strict=True)
    ):
        if record.get("schema_version") != 1 or record.get("kind") != RECORD_KIND:
            raise LayoutShadowRunError(f"layout record[{index}] schema/kind is unsupported")
        _require_int(record.get("index"), index, description=f"layout record[{index}] index")
        _require_flag(
            record.get("diagnostic_only"),
            True,
            description=f"layout record[{index}] diagnostic_only",
        )
        _require_flag(
            record.get("formal_delivery_gate"),
            False,
            description=f"layout record[{index}] formal_delivery_gate",
        )
        _require_flag(
            record.get("candidate_write_enabled"),
            False,
            description=f"layout record[{index}] candidate_write_enabled",
        )
        if record.get("execution_provider") != "cpu":
            raise LayoutShadowRunError(f"layout record[{index}] execution_provider must be cpu")
        raw_source = record.get("source")
        if not isinstance(raw_source, str) or _path_key(
            raw_source, strict=False
        ) != _path_key(expected_source, strict=False):
            raise LayoutShadowRunError(
                f"layout record[{index}] source/order differs from input list"
            )
    return summary


def run_layout_shadow(*, app: Path, bundle: Path, input_list: Path, output: Path) -> dict[str, Any]:
    try:
        app = app.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayoutShadowRunError(f"missing LayoutShadow application: {app}") from error
    if not app.is_file():
        raise LayoutShadowRunError(f"LayoutShadow application is not a file: {app}")
    try:
        bundle = bundle.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayoutShadowRunError(f"missing Paddle bundle: {bundle}") from error
    if not bundle.is_dir():
        raise LayoutShadowRunError(f"Paddle bundle is not a directory: {bundle}")
    try:
        input_list = input_list.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayoutShadowRunError(f"missing input list: {input_list}") from error
    if not input_list.is_file():
        raise LayoutShadowRunError(f"input list is not a file: {input_list}")
    output = _require_disjoint_output(
        output, app=app, bundle=bundle, input_list=input_list
    )
    input_bytes = _read_bytes(input_list, description="input list")
    input_sources = _parse_input_sources(input_list, input_bytes)
    input_sha256 = _sha256(input_bytes)
    app_bytes = _read_bytes(app, description="LayoutShadow application")
    command = [
        str(app),
        "--bundle",
        str(bundle),
        "--input-list",
        str(input_list),
        "--input-list-sha256",
        input_sha256,
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=False, shell=False)
    if completed.returncode != 0:
        raise LayoutShadowProcessError(completed.returncode)
    if _read_bytes(input_list, description="input list after run") != input_bytes:
        raise LayoutShadowRunError("input list changed while LayoutShadow was running")
    if _read_bytes(app, description="LayoutShadow application after run") != app_bytes:
        raise LayoutShadowRunError("LayoutShadow application changed while it was running")
    return validate_output(
        output=output,
        input_list=input_list,
        input_bytes=input_bytes,
        input_sources=input_sources,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_layout_shadow(
            app=args.app,
            bundle=args.bundle,
            input_list=args.input_list,
            output=args.output,
        )
    except LayoutShadowProcessError as error:
        print(f"LayoutShadow launcher failed: {error}", file=sys.stderr)
        return error.returncode if 0 < error.returncode < 256 else 2
    except (LayoutShadowRunError, OSError, UnicodeError) as error:
        print(f"LayoutShadow launcher failed: {error}", file=sys.stderr)
        return 2
    print(
        "layout_shadow_launcher PASS "
        f"records={summary['records']} provider={summary['execution_provider']} "
        f"input_sha256={summary['input_list']['sha256']} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
