#!/usr/bin/env python3
"""Prepare and gate a frozen 204+128 hybrid-recipient targeted CPU replay.

This utility is deliberately inference-free.  ``prepare`` binds the failed
formal A/B evidence, selects the 204 recipient-candidate omissions plus 128
stable controls, and atomically writes a hash-frozen replay input bundle.
``gate`` verifies separately produced CPU baseline/hybrid runs, their existing
A/B report, and old/new scorer reports.  A targeted replay is diagnostic
evidence only and can never set ``formal_delivery_gate`` to true.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
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

from transfer_receipt_ai.ocr import normalize_status
from transfer_receipt_ai.ocr_unified_targets import parse_amount_visible_format_target


FORMAL_RECORDS = 10016
MISSING_RECORDS = 204
CONTROL_RECORDS = 128
TARGET_RECORDS = MISSING_RECORDS + CONTROL_RECORDS
MAX_P95_OVERHEAD_MS = 250.0
RECIPIENT_MISSING_FAILURE = "hybrid recipient candidate missing"
AB_KIND = "receipt_mlnet_hybrid_recipient_cpu_ab_v1"
AB_SCHEMA_VERSION = 2
DIAGNOSTIC_KIND = "receipt_mlnet_hybrid_failure_diagnostic_summary_v1"
DIAGNOSTIC_FINDING_KIND = "receipt_mlnet_hybrid_failure_diagnostic_finding_v1"
SCORE_KIND = "receipt_mlnet_unified_candidate_evaluation_v1"
PREPARE_KIND = "receipt_mlnet_hybrid_targeted_replay_selection_v1"
PREPARE_SUMMARY_KIND = "receipt_mlnet_hybrid_targeted_replay_prepare_v1"
GATE_KIND = "receipt_mlnet_hybrid_targeted_replay_gate_v1"
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
FIXED_FLOORS = {
    "amount": 0.7885,
    "time": 0.9840,
    "payment_method_field": 0.9325,
    "recipient_field": 0.90,
    "transfer_status": 0.90,
}
SCORE_RESULT_FIELDS = {
    "amount": "amount",
    "time": "time",
    "payment_method_field": "payment_method",
    "recipient_field": "recipient",
    "transfer_status": "transfer_status",
}
INVARIANT_CONTRACT_KEYS = (
    "detector",
    "detector_sha256",
    "detector_contract_sha256",
    "device",
    "device_sha256",
    "device_contract_sha256",
    "unified_ocr_model",
    "unified_ocr_contract",
    "unified_ocr_model_sha256",
    "unified_ocr_labels_sha256",
    "unified_ocr_contract_sha256",
)
INVARIANT_RESULT_FIELDS = (
    "time",
    "amount",
    "transfer_status",
    "payment_method",
)
REQUIRED_STAGES = (
    "image_load",
    "device",
    "detector_preprocess",
    "detector_inference",
    "detector_postprocess",
    "unified_ocr_preprocess",
    "unified_ocr_inference",
    "unified_ocr_postprocess",
    "result_assembly",
)
ALL_STAGES = REQUIRED_STAGES[:-3] + ("paddle_ocr",) + REQUIRED_STAGES[-3:]
RESULT_EXCLUDED_TOP_LEVEL_KEYS = frozenset(
    {
        "source",
        "timing",
        "timing_ms",
        "inference_ms",
        "latency_ms",
        "stage_latency_ms",
    }
)
MANIFEST_EXCLUDED_KEYS = frozenset(
    {
        "source",
        "result",
        "annotated_rectified",
        "annotated_original",
        "inference_ms",
        "stage_latency_ms",
    }
)


class ReplayError(ValueError):
    """Frozen evidence is incomplete, inconsistent, or unsafe to accept."""


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
        raise ReplayError(f"invalid JSON at {location}: {error}") from error


def _load_json(path: Path, *, description: str) -> Any:
    if not path.is_file():
        raise ReplayError(f"missing {description}: {path}")
    return _loads(path.read_text(encoding="utf-8-sig"), location=str(path))


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReplayError(f"missing {description}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ReplayError(
                    f"invalid {description} {path}:{line_number}: blank line"
                )
            value = _loads(line, location=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ReplayError(
                    f"invalid {description} {path}:{line_number}: expected an object"
                )
            rows.append(value)
    if not rows:
        raise ReplayError(f"{description} is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReplayError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _file_identity(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ReplayError(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing {description}: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ReplayError(f"{description} must be a regular file: {resolved}")
    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise ReplayError(f"{description} must be non-empty: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": size_bytes,
    }


def _same_path(left: object, right: Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    try:
        return Path(left).resolve(strict=True) == right.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_is_within(left, right) or _path_is_within(right, left)


def _assert_identity(
    identity: object,
    *,
    description: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ReplayError(f"{description} identity must be an object")
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ReplayError(f"{description} identity has no path")
    path = Path(path_value)
    if expected_path is not None and not _same_path(path_value, expected_path):
        raise ReplayError(f"{description} identity points to the wrong artifact")
    observed = _file_identity(path, description=description)
    for key in ("sha256", "size_bytes"):
        if type(identity.get(key)) is not type(observed[key]) or identity.get(key) != observed[key]:
            raise ReplayError(f"{description} {key} mismatch")
    return observed


def _safe_relative_path(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"{description} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise ReplayError(f"{description} is unsafe: {value!r}")
    return "/".join(parts)


def _validate_cli_build(summary: Mapping[str, Any]) -> dict[str, Any]:
    build = summary.get("cli_build")
    if not isinstance(build, Mapping):
        raise ReplayError("targeted A/B summary has no frozen CLI build binding")
    assembly_binding = build.get("assembly")
    if not isinstance(assembly_binding, Mapping):
        raise ReplayError("targeted A/B summary has no CLI assembly binding")
    assembly_path = Path(
        _nonempty_string(
            assembly_binding.get("path"), description="targeted CLI assembly path"
        )
    )
    assembly_identity = _assert_identity(
        assembly_binding, description="targeted CLI assembly"
    )
    if assembly_path.name.casefold() != "receiptmlnet.cli.dll":
        raise ReplayError("targeted CLI assembly is not ReceiptMlNet.Cli.dll")
    closure = build.get("app_closure")
    if not isinstance(closure, Mapping):
        raise ReplayError("targeted A/B summary has no exact CLI app closure")
    root_value = _nonempty_string(
        closure.get("root"), description="targeted CLI app root"
    )
    app_root = Path(root_value)
    if app_root.is_symlink():
        raise ReplayError("targeted CLI app root must not be a symbolic link")
    try:
        app_root = app_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing targeted CLI app root: {app_root}") from error
    if not app_root.is_dir():
        raise ReplayError("targeted CLI app root is not a directory")
    try:
        assembly_path.resolve(strict=True).relative_to(app_root)
    except ValueError as error:
        raise ReplayError("targeted CLI assembly is outside the frozen app closure") from error
    manifest_binding = closure.get("manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ReplayError("targeted CLI app closure has no manifest binding")
    manifest_path = Path(
        _nonempty_string(
            manifest_binding.get("path"), description="targeted CLI closure manifest path"
        )
    )
    manifest_identity = _assert_identity(
        manifest_binding, description="targeted CLI closure manifest"
    )
    if closure.get("closure_sha256") != manifest_identity["sha256"]:
        raise ReplayError("targeted CLI app closure SHA-256 differs from its manifest")
    rows = _load_json(manifest_path, description="targeted CLI closure manifest")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("targeted CLI closure manifest must be a non-empty array")
    if closure.get("file_count") != len(rows):
        raise ReplayError("targeted CLI closure file_count differs from its manifest")
    observed_paths: list[str] = []
    listed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ReplayError(
                f"targeted CLI closure row {index} must contain exactly path, sha256, size_bytes"
            )
        relative = _safe_relative_path(
            row.get("path"), description=f"targeted CLI closure row {index} path"
        )
        key = relative.casefold()
        if key in listed:
            raise ReplayError(f"duplicate targeted CLI closure path: {relative}")
        sha256 = row.get("sha256")
        size_bytes = row.get("size_bytes")
        if not isinstance(sha256, str) or LOWER_SHA256.fullmatch(sha256) is None:
            raise ReplayError(f"targeted CLI closure {relative} has an invalid SHA-256")
        _integer(
            size_bytes,
            description=f"targeted CLI closure {relative} size_bytes",
            minimum=1,
        )
        listed[key] = {
            "path": relative,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        observed_paths.append(relative)
    if observed_paths != sorted(
        observed_paths, key=lambda value: (value.casefold(), value)
    ):
        raise ReplayError("targeted CLI closure paths are not canonically sorted")
    actual: dict[str, Path] = {}
    for path in app_root.rglob("*"):
        if path.is_symlink():
            raise ReplayError(f"targeted CLI closure contains a symbolic link: {path}")
        if path.is_file():
            relative = path.relative_to(app_root).as_posix()
            key = relative.casefold()
            if key in actual:
                raise ReplayError(f"duplicate case-insensitive CLI app path: {relative}")
            actual[key] = path
    if set(actual) != set(listed):
        raise ReplayError(
            "targeted CLI app closure is not exact: "
            f"missing={len(set(listed) - set(actual))}, extra={len(set(actual) - set(listed))}"
        )
    for key, record in listed.items():
        path = actual[key]
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ReplayError(f"targeted CLI app file differs from closure: {record['path']}")
    assembly_relative = assembly_path.resolve(strict=True).relative_to(app_root).as_posix().casefold()
    assembly_record = listed.get(assembly_relative)
    if (
        assembly_record is None
        or assembly_record["sha256"] != assembly_identity["sha256"]
        or assembly_record["size_bytes"] != assembly_identity["size_bytes"]
    ):
        raise ReplayError("targeted CLI assembly differs from the exact app closure")
    return {
        "assembly": assembly_identity,
        "app_root": app_root.as_posix(),
        "closure_manifest": manifest_identity,
        "file_count": len(listed),
    }


def _source_key(value: object) -> str:
    raw = str(value or "")
    if WINDOWS_ABSOLUTE_PATH.match(raw) or "\\" in raw:
        return "windows:" + ntpath.normcase(ntpath.normpath(raw)).replace("\\", "/")
    return "posix:" + os.path.normcase(os.path.normpath(os.path.abspath(raw)))


def _normalized_source_set_sha256(keys: Sequence[str]) -> str:
    payload = "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError(f"{description} must be a non-empty string")
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayError(f"{description} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{description} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReplayError(f"{description} must be finite")
    return number


def _nonnegative_number(value: object, *, description: str) -> float:
    number = _finite_number(value, description=description)
    if number < 0:
        raise ReplayError(f"{description} must be non-negative")
    return number


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * (position - lower)


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "p50": round(_percentile(ordered, 0.50), 4),
        "p95": round(_percentile(ordered, 0.95), 4),
    }


def _assert_latency_summary(
    observed: object,
    values: Sequence[float],
    *,
    description: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise ReplayError(f"{description} must be an object")
    expected = _summarize(values)
    if observed.get("count") != expected["count"]:
        raise ReplayError(f"{description}.count differs from the manifest")
    for key in ("mean", "p50", "p95"):
        wanted = expected[key]
        actual = observed.get(key)
        if wanted is None:
            if actual is not None:
                raise ReplayError(f"{description}.{key} must be null")
        elif abs(
            _nonnegative_number(actual, description=f"{description}.{key}") - wanted
        ) > 0.00011:
            raise ReplayError(f"{description}.{key} differs from the manifest")


def _candidate(result: Mapping[str, Any], field: str) -> str | None:
    fields = result.get("fields")
    value = fields.get(field) if isinstance(fields, Mapping) else None
    candidate = value.get("candidate") if isinstance(value, Mapping) else None
    return candidate if isinstance(candidate, str) and candidate.strip() else None


def _field(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    fields = result.get("fields")
    value = fields.get(name) if isinstance(fields, Mapping) else None
    if not isinstance(value, Mapping):
        raise ReplayError(f"result has no fields.{name} object")
    return value


def _read_input_list(path: Path) -> tuple[list[str], dict[str, Any]]:
    identity = _file_identity(path, description="formal fixed input list")
    sources: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        source = line.strip()
        if not source:
            raise ReplayError(
                f"formal fixed input list {path}:{line_number} contains a blank source"
            )
        key = _source_key(source)
        if key in seen:
            raise ReplayError(f"duplicate formal input source: {source!r}")
        seen.add(key)
        sources.append(source)
    if len(sources) != FORMAL_RECORDS:
        raise ReplayError(
            f"formal input list must contain exactly {FORMAL_RECORDS} sources; got {len(sources)}"
        )
    identity.update(
        {
            "records": len(sources),
            "normalized_source_set_sha256": _normalized_source_set_sha256(
                tuple(seen)
            ),
        }
    )
    return sources, identity


def _contained_result_path(raw: object, *, run_root: Path, manifest: Path) -> Path:
    value = _nonempty_string(raw, description="manifest result path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing manifest result: {candidate}") from error
    try:
        resolved.relative_to(run_root.resolve(strict=True))
    except ValueError as error:
        raise ReplayError(f"manifest result escapes its run directory: {resolved}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ReplayError(f"manifest result must be a regular file: {resolved}")
    return resolved


def _load_run(
    directory: Path,
    *,
    expected_sources: Sequence[str] | None,
    hybrid: bool,
) -> dict[str, Any]:
    if directory.is_symlink():
        raise ReplayError(f"inference run must not be a symbolic link: {directory}")
    try:
        root = directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing inference run: {directory}") from error
    if not root.is_dir() or root.is_symlink():
        raise ReplayError(f"inference run must be a regular directory: {root}")
    summary_path = root / "inference_summary.json"
    manifest_path = root / "inference_manifest.json"
    summary_identity = _file_identity(summary_path, description="inference summary")
    manifest_identity = _file_identity(manifest_path, description="inference manifest")
    summary = _load_json(summary_path, description="inference summary")
    manifest = _load_json(manifest_path, description="inference manifest")
    if not isinstance(summary, Mapping) or not isinstance(manifest, list):
        raise ReplayError("inference summary/manifest schema is invalid")
    if summary.get("requested_device") != "cpu" or summary.get("unified_provider") != "cpu":
        raise ReplayError("targeted A/B runs must use the CPU unified provider")
    if hybrid:
        if summary.get("paddle_ocr_provider") != "cpu":
            raise ReplayError("targeted hybrid run must use CPU PP-OCR")
    elif summary.get("paddle_ocr_provider") is not None:
        raise ReplayError("targeted baseline unexpectedly loaded PP-OCR")
    expected_count = len(expected_sources) if expected_sources is not None else len(manifest)
    for name, expected in (
        ("input", expected_count),
        ("written", expected_count),
        ("skipped", 0),
        ("errors", 0),
    ):
        if type(summary.get(name)) is not int or summary.get(name) != expected:
            raise ReplayError(f"inference summary {name} must be {expected}")
    latency = summary.get("inference_latency_ms")
    if not isinstance(latency, Mapping) or latency.get("count") != expected_count:
        raise ReplayError("inference latency does not cover every targeted receipt")
    if len(manifest) != expected_count:
        raise ReplayError(
            f"inference manifest must contain {expected_count} rows; got {len(manifest)}"
        )
    results: dict[str, dict[str, Any]] = {}
    rows_by_source: dict[str, dict[str, Any]] = {}
    ordered_sources: list[str] = []
    result_paths: set[Path] = set()
    inference_values: list[float] = []
    stage_values: dict[str, list[float]] = {stage: [] for stage in ALL_STAGES}
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping) or row.get("status") != "written":
            raise ReplayError(f"inference manifest row {index} is not freshly written")
        inference_values.append(
            _nonnegative_number(
                row.get("inference_ms"),
                description=f"inference manifest row {index} inference_ms",
            )
        )
        stages = row.get("stage_latency_ms")
        if not isinstance(stages, Mapping):
            raise ReplayError(f"inference manifest row {index} has no stage latency")
        for stage in REQUIRED_STAGES:
            stage_values[stage].append(
                _nonnegative_number(
                    stages.get(stage),
                    description=f"inference manifest row {index} {stage} latency",
                )
            )
        paddle_latency = stages.get("paddle_ocr")
        if hybrid:
            stage_values["paddle_ocr"].append(
                _nonnegative_number(
                    paddle_latency,
                    description=f"inference manifest row {index} paddle_ocr latency",
                )
            )
        elif paddle_latency is not None:
            raise ReplayError("baseline manifest unexpectedly reports PP-OCR latency")
        source = _nonempty_string(
            row.get("source"), description=f"inference manifest row {index} source"
        )
        key = _source_key(source)
        if key in results:
            raise ReplayError(f"duplicate inference source: {source!r}")
        result_path = _contained_result_path(
            row.get("result"), run_root=root, manifest=manifest_path
        )
        if result_path in result_paths:
            raise ReplayError(f"duplicate inference result path: {result_path}")
        result_paths.add(result_path)
        result = _load_json(result_path, description="inference result")
        if not isinstance(result, dict):
            raise ReplayError(f"inference result must be an object: {result_path}")
        if _source_key(result.get("source")) != key:
            raise ReplayError(f"manifest/result source mismatch for {source!r}")
        results[key] = {
            "source": source,
            "payload": result,
            "path": result_path,
            "identity": _file_identity(result_path, description="inference result"),
        }
        rows_by_source[key] = dict(row)
        ordered_sources.append(source)
    if expected_sources is not None:
        expected_keys = [_source_key(source) for source in expected_sources]
        if [_source_key(source) for source in ordered_sources] != expected_keys:
            raise ReplayError("inference manifest order differs from the frozen input order")
    _assert_latency_summary(
        summary.get("inference_latency_ms"),
        inference_values,
        description="inference summary latency",
    )
    stage_summary = summary.get("stage_latency_ms")
    if not isinstance(stage_summary, Mapping):
        raise ReplayError("inference summary has no stage latency evidence")
    for stage in ALL_STAGES:
        _assert_latency_summary(
            stage_summary.get(stage),
            stage_values[stage],
            description=f"inference summary {stage} latency",
        )
    total_seconds = _nonnegative_number(
        summary.get("total_seconds"), description="inference summary total_seconds"
    )
    if total_seconds <= 0:
        raise ReplayError("inference summary total_seconds must be positive")
    errors_path = root / "inference_errors.jsonl"
    if not errors_path.is_file() or errors_path.read_text(encoding="utf-8-sig") != "":
        raise ReplayError("inference run must contain an empty inference_errors.jsonl")
    manifest_identity.update(
        {
            "records": len(results),
            "normalized_source_set_sha256": _normalized_source_set_sha256(
                tuple(results)
            ),
        }
    )
    return {
        "root": root,
        "summary": dict(summary),
        "summary_identity": summary_identity,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_identity": manifest_identity,
        "results": results,
        "rows": rows_by_source,
        "ordered_sources": ordered_sources,
    }


def _assert_bound_identity(
    bound: object,
    actual: Mapping[str, Any],
    *,
    description: str,
) -> None:
    if not isinstance(bound, Mapping):
        raise ReplayError(f"{description} binding must be an object")
    if not _same_path(bound.get("path"), Path(str(actual["path"]))):
        raise ReplayError(f"{description} path binding mismatch")
    for key in (
        "sha256",
        "size_bytes",
        "records",
        "normalized_source_set_sha256",
    ):
        if key not in actual:
            continue
        if type(bound.get(key)) is not type(actual[key]) or bound.get(key) != actual[key]:
            raise ReplayError(f"{description} {key} binding mismatch")


def _load_formal_ab(
    root: Path,
    *,
    diagnostic: Path,
) -> dict[str, Any]:
    if root.is_symlink():
        raise ReplayError(f"formal root must not be a symbolic link: {root}")
    try:
        formal_root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing formal root: {root}") from error
    input_sources, input_identity = _read_input_list(
        formal_root / "fixed-selected-inputs.txt"
    )
    baseline = _load_run(
        formal_root / "baseline-v13", expected_sources=input_sources, hybrid=False
    )
    hybrid = _load_run(
        formal_root / "hybrid-recipient", expected_sources=input_sources, hybrid=True
    )
    comparison_dir = formal_root / "comparison"
    summary_path = comparison_dir / "summary.json"
    rows_path = comparison_dir / "comparisons.jsonl"
    summary_identity = _file_identity(summary_path, description="formal A/B summary")
    rows_identity = _file_identity(rows_path, description="formal A/B comparisons")
    summary = _load_json(summary_path, description="formal A/B summary")
    rows = _load_jsonl(rows_path, description="formal A/B comparisons")
    if not isinstance(summary, Mapping):
        raise ReplayError("formal A/B summary must be an object")
    exact = {
        "schema_version": AB_SCHEMA_VERSION,
        "kind": AB_KIND,
        "evaluation_mode": "formal",
        "records": FORMAL_RECORDS,
    }
    for key, expected in exact.items():
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            raise ReplayError(f"formal A/B summary {key} must be {expected!r}")
    if summary.get("input_set_identical") is not True or summary.get(
        "cli_summary_counts_verified"
    ) is not True:
        raise ReplayError("formal A/B input/count binding is not verified")
    if len(rows) != FORMAL_RECORDS:
        raise ReplayError(f"formal A/B comparisons must contain {FORMAL_RECORDS} rows")
    by_source: dict[str, dict[str, Any]] = {}
    missing_keys: set[str] = set()
    invariant_count = 0
    present_count = 0
    for index, row in enumerate(rows):
        source = _nonempty_string(
            row.get("source"), description=f"formal comparison row {index} source"
        )
        key = _source_key(source)
        if key in by_source:
            raise ReplayError(f"duplicate formal comparison source: {source!r}")
        invariant = row.get("invariant")
        failures = row.get("failures")
        candidate = row.get("recipient_candidate")
        if type(invariant) is not bool or not isinstance(failures, list) or any(
            not isinstance(item, str) or not item for item in failures
        ):
            raise ReplayError(f"formal comparison row {index} has invalid invariant/failures")
        if candidate is not None and not isinstance(candidate, str):
            raise ReplayError(f"formal comparison row {index} candidate has invalid type")
        missing = not isinstance(candidate, str) or not candidate
        if missing:
            if invariant is not False or failures != [RECIPIENT_MISSING_FAILURE]:
                raise ReplayError(
                    "formal recipient omissions must be all/only hybrid recipient candidate missing"
                )
            missing_keys.add(key)
        else:
            if invariant is not True or failures != []:
                raise ReplayError(
                    "formal non-missing comparisons must be fully invariant controls"
                )
            invariant_count += 1
            present_count += 1
        hybrid_entry = hybrid["results"].get(key)
        if hybrid_entry is None or _candidate(hybrid_entry["payload"], "recipient") != (
            candidate if isinstance(candidate, str) and candidate else None
        ):
            raise ReplayError(f"formal comparison candidate differs from hybrid result: {source!r}")
        by_source[key] = row
    if len(missing_keys) != MISSING_RECORDS:
        raise ReplayError(
            f"formal A/B must contain exactly {MISSING_RECORDS} recipient omissions; got {len(missing_keys)}"
        )
    if set(by_source) != {_source_key(source) for source in input_sources}:
        raise ReplayError("formal comparison source set differs from the fixed input list")
    expected_invariant = FORMAL_RECORDS - MISSING_RECORDS
    expected_coverage = expected_invariant / FORMAL_RECORDS
    if summary.get("invariant_records") != expected_invariant:
        raise ReplayError("formal A/B invariant_records differs from comparisons")
    coverage = _finite_number(
        summary.get("recipient_candidate_coverage"),
        description="formal A/B recipient candidate coverage",
    )
    if coverage != expected_coverage:
        raise ReplayError("formal A/B recipient candidate coverage differs from comparisons")
    input_set = summary.get("input_set")
    if not isinstance(input_set, Mapping) or input_set.get("records") != FORMAL_RECORDS:
        raise ReplayError("formal A/B summary has no complete input-set binding")
    _assert_bound_identity(
        input_set.get("input_manifest"), input_identity, description="formal input list"
    )
    if input_set.get("normalized_source_set_sha256") != input_identity[
        "normalized_source_set_sha256"
    ]:
        raise ReplayError("formal A/B normalized source-set hash mismatch")
    run_manifests = summary.get("run_manifests")
    run_summaries = summary.get("run_summaries")
    if not isinstance(run_manifests, Mapping) or not isinstance(run_summaries, Mapping):
        raise ReplayError("formal A/B summary lacks run bindings")
    _assert_bound_identity(
        run_manifests.get("baseline"),
        baseline["manifest_identity"],
        description="formal baseline manifest",
    )
    _assert_bound_identity(
        run_manifests.get("hybrid"),
        hybrid["manifest_identity"],
        description="formal hybrid manifest",
    )
    _assert_bound_identity(
        run_summaries.get("baseline"),
        baseline["summary_identity"],
        description="formal baseline summary",
    )
    _assert_bound_identity(
        run_summaries.get("hybrid"),
        hybrid["summary_identity"],
        description="formal hybrid summary",
    )

    diagnostic_summary_path = diagnostic / "summary.json"
    diagnostic_findings_path = diagnostic / "findings.jsonl"
    diagnostic_summary_identity = _file_identity(
        diagnostic_summary_path, description="failure diagnostic summary"
    )
    diagnostic_findings_identity = _file_identity(
        diagnostic_findings_path, description="failure diagnostic findings"
    )
    diagnostic_summary = _load_json(
        diagnostic_summary_path, description="failure diagnostic summary"
    )
    findings = _load_jsonl(
        diagnostic_findings_path, description="failure diagnostic findings"
    )
    if not isinstance(diagnostic_summary, Mapping):
        raise ReplayError("failure diagnostic summary must be an object")
    diagnostic_exact = {
        "schema_version": 1,
        "kind": DIAGNOSTIC_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "comparison_evaluation_mode": "formal",
        "comparison_records": FORMAL_RECORDS,
        "invariant_failure_records": MISSING_RECORDS,
        "recipient_missing_records": MISSING_RECORDS,
        "non_missing_invariant_failure_records": 0,
        "recipient_missing_only_records": MISSING_RECORDS,
        "recipient_missing_with_additional_failures_records": 0,
        "failed_records": MISSING_RECORDS,
    }
    for key, expected in diagnostic_exact.items():
        if (
            type(diagnostic_summary.get(key)) is not type(expected)
            or diagnostic_summary.get(key) != expected
        ):
            raise ReplayError(f"failure diagnostic summary {key} must be {expected!r}")
    source_evidence = diagnostic_summary.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ReplayError("failure diagnostic has no source-evidence binding")
    _assert_bound_identity(
        source_evidence.get("comparison_summary"),
        summary_identity,
        description="diagnostic formal A/B summary",
    )
    _assert_bound_identity(
        source_evidence.get("comparisons"),
        rows_identity,
        description="diagnostic formal A/B comparisons",
    )
    _assert_bound_identity(
        source_evidence.get("hybrid_manifest"),
        hybrid["manifest_identity"],
        description="diagnostic formal hybrid manifest",
    )
    finding_keys: set[str] = set()
    if len(findings) != MISSING_RECORDS:
        raise ReplayError(f"failure diagnostics must contain {MISSING_RECORDS} findings")
    for index, finding in enumerate(findings):
        if finding.get("schema_version") != 1 or finding.get("kind") != DIAGNOSTIC_FINDING_KIND:
            raise ReplayError(f"failure diagnostic finding {index} has unsupported schema/kind")
        source = _nonempty_string(
            finding.get("source"), description=f"failure diagnostic finding {index} source"
        )
        key = _source_key(source)
        if key in finding_keys:
            raise ReplayError(f"duplicate failure diagnostic source: {source!r}")
        failures = finding.get("failures")
        if failures != [RECIPIENT_MISSING_FAILURE]:
            raise ReplayError("failure diagnostic includes a non-target comparator failure")
        finding_keys.add(key)
    if finding_keys != missing_keys:
        raise ReplayError("failure diagnostic source set differs from the 204 formal omissions")
    return {
        "root": formal_root,
        "input_sources": input_sources,
        "input_identity": input_identity,
        "baseline": baseline,
        "hybrid": hybrid,
        "comparison_rows": rows,
        "comparison_by_source": by_source,
        "missing_keys": missing_keys,
        "source_evidence": {
            "formal_input_list": input_identity,
            "formal_ab_summary": summary_identity,
            "formal_ab_comparisons": rows_identity,
            "formal_baseline_manifest": baseline["manifest_identity"],
            "formal_baseline_summary": baseline["summary_identity"],
            "formal_hybrid_manifest": hybrid["manifest_identity"],
            "formal_hybrid_summary": hybrid["summary_identity"],
            "diagnostic_summary": diagnostic_summary_identity,
            "diagnostic_findings": diagnostic_findings_identity,
        },
    }


def _control_tokens(result: Mapping[str, Any]) -> tuple[str, str, str]:
    device = result.get("device")
    if isinstance(device, Mapping):
        device_value = next(
            (
                device.get(name)
                for name in ("platform", "label", "candidate", "value", "device")
                if isinstance(device.get(name), str) and device.get(name)
            ),
            "<other>",
        )
    else:
        device_value = "<missing>"
    status = _field(result, "transfer_status")
    status_value = next(
        (
            status.get(name)
            for name in ("normalized", "candidate_status_class", "candidate", "state")
            if isinstance(status.get(name), str) and status.get(name)
        ),
        "<missing>",
    )
    recipient = _field(result, "recipient")
    route = recipient.get("hybrid_ocr_route")
    route_value = route if isinstance(route, str) and route else "<missing>"
    return (
        f"device:{device_value}",
        f"status:{status_value}",
        f"route:{route_value}",
    )


def _select_controls(
    formal: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    candidates: list[tuple[int, str, tuple[str, str, str]]] = []
    for index, source in enumerate(formal["input_sources"]):
        key = _source_key(source)
        if key in formal["missing_keys"]:
            continue
        row = formal["comparison_by_source"][key]
        if row.get("invariant") is not True or not row.get("recipient_candidate"):
            raise ReplayError("control pool contains a non-invariant or missing record")
        result = formal["hybrid"]["results"][key]["payload"]
        candidates.append((index, key, _control_tokens(result)))
    if len(candidates) < CONTROL_RECORDS:
        raise ReplayError(
            f"only {len(candidates)} invariant controls are available; need {CONTROL_RECORDS}"
        )
    selected: list[tuple[int, str, tuple[str, str, str]]] = []
    remaining = list(candidates)
    uncovered = {token for _, _, tokens in candidates for token in tokens}
    while remaining and uncovered and len(selected) < CONTROL_RECORDS:
        best_position = max(
            range(len(remaining)),
            key=lambda position: (
                len(set(remaining[position][2]) & uncovered),
                -remaining[position][0],
            ),
        )
        chosen = remaining.pop(best_position)
        if not (set(chosen[2]) & uncovered):
            break
        selected.append(chosen)
        uncovered.difference_update(chosen[2])
    selected_keys = {key for _, key, _ in selected}
    for candidate in candidates:
        if len(selected) >= CONTROL_RECORDS:
            break
        if candidate[1] not in selected_keys:
            selected.append(candidate)
            selected_keys.add(candidate[1])
    selected.sort(key=lambda item: item[0])
    covered = {token for _, _, tokens in selected for token in tokens}
    available = {token for _, _, tokens in candidates for token in tokens}
    return [key for _, key, _ in selected], {
        "strategy": "greedy_device_status_route_coverage_then_formal_input_order_fill",
        "available_tokens": sorted(available),
        "covered_tokens": sorted(covered),
        "uncovered_tokens": sorted(available - covered),
    }


def _load_selected_records(
    records_path: Path,
    *,
    selected_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = _file_identity(records_path, description="unified records manifest")
    rows = _load_jsonl(records_path, description="unified records manifest")
    selected: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, row in enumerate(rows):
        source = row.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ReplayError(f"unified records row {index} has no source")
        key = _source_key(source)
        if key in selected_keys:
            if row.get("split") != "val":
                raise ReplayError(f"selected unified record is not split=val: {source!r}")
            selected.append(dict(row))
            observed.add(key)
    if observed != selected_keys:
        raise ReplayError(
            "unified records manifest does not cover every selected source: "
            f"missing={len(selected_keys - observed)}"
        )
    return selected, identity


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _publish_directory(stage: Path, output: Path, *, description: str) -> None:
    """Rename a complete staged directory and recheck the no-clobber contract."""

    lock_path = output.parent / f".{output.name}.publish.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise ReplayError(
            f"another {description} publication is active or left a lock: {lock_path}"
        ) from error
    try:
        if output.exists() or output.is_symlink():
            raise ReplayError(f"refusing to overwrite {description}: {output}")
        try:
            # Windows, where the delivery workflow runs, makes rename fail when
            # the destination appears between the check and this operation.
            # O_EXCL serializes cooperating publishers on POSIX as well.  Never
            # use Path.replace: it explicitly authorizes clobbering a target.
            stage.rename(output)
        except FileExistsError as error:
            raise ReplayError(
                f"refusing to overwrite {description}: {output}"
            ) from error
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def prepare(
    *,
    formal_root: Path,
    diagnostic: Path,
    records: Path,
    output: Path,
) -> dict[str, Any]:
    """Atomically freeze the strict targeted replay selection; never run OCR."""

    if output.is_symlink():
        raise ReplayError(f"refusing a symbolic-link targeted replay output: {output}")
    output = output.resolve()
    if output.exists():
        raise ReplayError(f"refusing to overwrite targeted replay selection: {output}")
    formal = _load_formal_ab(formal_root, diagnostic=diagnostic)
    control_keys, control_evidence = _select_controls(formal)
    selected_keys = set(formal["missing_keys"]) | set(control_keys)
    if len(selected_keys) != TARGET_RECORDS:
        raise ReplayError(f"targeted selection must contain exactly {TARGET_RECORDS} sources")
    selected_sources = [
        source
        for source in formal["input_sources"]
        if _source_key(source) in selected_keys
    ]
    selected_records, records_identity = _load_selected_records(
        records, selected_keys=selected_keys
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        inputs_path = stage / "inputs.txt"
        subset_records_path = stage / "subset-records.jsonl"
        old_subset_dir = stage / "old-hybrid-subset"
        old_subset_dir.mkdir()
        old_subset_manifest_path = old_subset_dir / "inference_manifest.json"
        inputs_path.write_text(
            "".join(f"{source}\n" for source in selected_sources),
            encoding="utf-8",
            newline="\n",
        )
        _write_jsonl(subset_records_path, selected_records)
        old_subset_rows = [
            formal["hybrid"]["rows"][_source_key(source)]
            for source in selected_sources
        ]
        old_subset_manifest_path.write_text(
            json.dumps(
                old_subset_rows,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        selection_records: list[dict[str, Any]] = []
        control_key_set = set(control_keys)
        for canonical_index, source in enumerate(formal["input_sources"]):
            key = _source_key(source)
            if key not in selected_keys:
                continue
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = Path(str(formal["input_identity"]["path"])).parent / source_path
            image_identity = _file_identity(source_path, description="selected receipt image")
            old_baseline = formal["baseline"]["results"][key]
            old_hybrid = formal["hybrid"]["results"][key]
            old_recipient = _field(old_hybrid["payload"], "recipient")
            old_status = _field(old_hybrid["payload"], "transfer_status")
            role = "control" if key in control_key_set else "recipient_missing"
            selection_records.append(
                {
                    "canonical_index": canonical_index,
                    "source": source,
                    "role": role,
                    "control_tokens": (
                        list(_control_tokens(old_hybrid["payload"]))
                        if role == "control"
                        else []
                    ),
                    "image": image_identity,
                    "old_baseline_result": old_baseline["identity"],
                    "old_hybrid_result": old_hybrid["identity"],
                    "old_recipient_candidate": old_recipient.get("candidate"),
                    "old_recipient_ctc_candidate": old_recipient.get("ctc_candidate"),
                    "old_status_candidate": old_status.get("candidate"),
                }
            )
        if len(selection_records) != TARGET_RECORDS:
            raise ReplayError("internal targeted selection count mismatch")
        selection = {
            "schema_version": 1,
            "kind": PREPARE_KIND,
            "read_only_existing_results": True,
            "ocr_rerun": False,
            "targeted_replay_only": True,
            "formal_delivery_gate": False,
            "counts": {
                "formal": FORMAL_RECORDS,
                "recipient_missing": MISSING_RECORDS,
                "controls": CONTROL_RECORDS,
                "selected": TARGET_RECORDS,
            },
            "selection_order": "old_formal_fixed_input_order",
            "control_selection": control_evidence,
            "records": selection_records,
            "source_evidence": {
                **formal["source_evidence"],
                "unified_records": records_identity,
            },
        }
        selection_path = stage / "selection.json"
        _write_json(selection_path, selection)
        def published_identity(
            staged_path: Path, published_path: Path, *, description: str
        ) -> dict[str, Any]:
            identity = _file_identity(staged_path, description=description)
            identity["path"] = published_path.as_posix()
            return identity

        artifacts = {
            "inputs": published_identity(
                inputs_path, output / "inputs.txt", description="targeted inputs"
            ),
            "subset_records": published_identity(
                subset_records_path,
                output / "subset-records.jsonl",
                description="targeted subset records",
            ),
            "old_hybrid_subset_manifest": published_identity(
                old_subset_manifest_path,
                output / "old-hybrid-subset" / "inference_manifest.json",
                description="old hybrid subset manifest",
            ),
            "selection": published_identity(
                selection_path,
                output / "selection.json",
                description="targeted selection",
            ),
        }
        summary = {
            "schema_version": 1,
            "kind": PREPARE_SUMMARY_KIND,
            "read_only_existing_results": True,
            "ocr_rerun": False,
            "targeted_replay_only": True,
            "formal_delivery_gate": False,
            "counts": selection["counts"],
            "artifacts": artifacts,
        }
        _write_json(stage / "summary.json", summary)
        # All evidence is checked once more after the complete bundle exists.
        for identity in formal["source_evidence"].values():
            _assert_identity(identity, description="frozen formal source evidence")
        _assert_identity(records_identity, description="frozen unified records manifest")
        # Every formal hybrid result participates either in the omission set or
        # in deterministic control-token coverage.  Recheck the whole pool so
        # a late result mutation cannot bias the frozen control selection.
        for entry in formal["hybrid"]["results"].values():
            _assert_identity(
                entry["identity"], description="frozen formal hybrid result pool"
            )
        for row in selection_records:
            for identity_name in (
                "image",
                "old_baseline_result",
                "old_hybrid_result",
            ):
                _assert_identity(
                    row[identity_name],
                    description=f"frozen selected {identity_name}",
                )
        _publish_directory(stage, output, description="targeted replay selection")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        **summary,
        "output_directory": output.as_posix(),
    }


def _reference_text(field: str, slot: Mapping[str, Any]) -> str | None:
    """Mirror the frozen unified scorer's reference-text selection."""

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
        return text
    if field == "time":
        visible = slot.get("visible_text")
        return visible if isinstance(visible, str) and visible else text
    return text


def _score_references(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Rebuild the scorer domain and immutable truth from subset-records."""

    receipts: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(rows):
        if record.get("split") != "val":
            raise ReplayError(f"prepared subset record {index} is not split=val")
        source = _nonempty_string(
            record.get("source"),
            description=f"prepared subset record {index} source",
        )
        key = _source_key(source)
        slots = record.get("slots")
        if not isinstance(slots, Mapping):
            raise ReplayError(f"prepared subset record {index} has no slots object")
        receipt = receipts.setdefault(
            key,
            {
                "source": source,
                "id": str(record.get("id", source)),
                "group_id": record.get("group_id"),
                "teacher_result_json": record.get("result_json"),
                "references": {},
            },
        )
        references = receipt["references"]
        for field in SCORE_RESULT_FIELDS:
            slot = slots.get(field)
            if not isinstance(slot, Mapping):
                continue
            reference = _reference_text(field, slot)
            if reference is None:
                continue
            diagnostics = {
                "reference_crop_sha256": slot.get("crop_sha256"),
                "reference_detector_score": slot.get("detector_score"),
                "reference_bbox_rectified": slot.get("bbox_rectified"),
            }
            if field == "transfer_status":
                diagnostics["reference_status_class"] = slot.get("class_name")
            previous = references.get(field)
            if previous is not None and previous["reference_text"] != reference:
                raise ReplayError(
                    f"prepared subset source {source!r} has conflicting {field} references"
                )
            references.setdefault(
                field,
                {"reference_text": reference, **diagnostics},
            )

    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for source_key, receipt in receipts.items():
        for field, reference in receipt["references"].items():
            expected[(source_key, field)] = {
                "schema_version": 1,
                "kind": "receipt_mlnet_unified_comparison_v1",
                "id": receipt["id"],
                "group_id": receipt["group_id"],
                "split": "val",
                "source": receipt["source"],
                "teacher_result_json": receipt["teacher_result_json"],
                "field": field,
                **reference,
            }
    return expected


def _load_prepared(prepared: Path) -> dict[str, Any]:
    if prepared.is_symlink():
        raise ReplayError(f"prepared replay must not be a symbolic link: {prepared}")
    try:
        root = prepared.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReplayError(f"missing prepared replay: {prepared}") from error
    summary = _load_json(root / "summary.json", description="targeted prepare summary")
    selection = _load_json(root / "selection.json", description="targeted selection")
    if not isinstance(summary, Mapping) or not isinstance(selection, Mapping):
        raise ReplayError("prepared summary/selection must be objects")
    for payload, kind in (
        (summary, PREPARE_SUMMARY_KIND),
        (selection, PREPARE_KIND),
    ):
        if payload.get("schema_version") != 1 or payload.get("kind") != kind:
            raise ReplayError("unsupported targeted prepare schema/kind")
        if (
            payload.get("read_only_existing_results") is not True
            or payload.get("ocr_rerun") is not False
            or payload.get("targeted_replay_only") is not True
            or payload.get("formal_delivery_gate") is not False
        ):
            raise ReplayError("prepared replay protection flags changed")
    expected_counts = {
        "formal": FORMAL_RECORDS,
        "recipient_missing": MISSING_RECORDS,
        "controls": CONTROL_RECORDS,
        "selected": TARGET_RECORDS,
    }
    if summary.get("counts") != expected_counts or selection.get("counts") != expected_counts:
        raise ReplayError("prepared replay counts changed")
    source_evidence = selection.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ReplayError("prepared selection has no source evidence")
    required_source_evidence = {
        "formal_input_list",
        "formal_ab_summary",
        "formal_ab_comparisons",
        "formal_baseline_manifest",
        "formal_baseline_summary",
        "formal_hybrid_manifest",
        "formal_hybrid_summary",
        "diagnostic_summary",
        "diagnostic_findings",
        "unified_records",
    }
    if not required_source_evidence.issubset(source_evidence):
        raise ReplayError("prepared selection source-evidence closure is incomplete")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReplayError("prepared replay has no artifact bindings")
    paths = {
        "inputs": root / "inputs.txt",
        "subset_records": root / "subset-records.jsonl",
        "old_hybrid_subset_manifest": root
        / "old-hybrid-subset"
        / "inference_manifest.json",
        "selection": root / "selection.json",
    }
    identities: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        identities[name] = _assert_identity(
            artifacts.get(name), description=f"prepared {name}", expected_path=path
        )
    sources = [
        line.strip()
        for line in paths["inputs"].read_text(encoding="utf-8-sig").splitlines()
    ]
    if len(sources) != TARGET_RECORDS or any(not source for source in sources):
        raise ReplayError("prepared input list count/content changed")
    keys = [_source_key(source) for source in sources]
    if len(set(keys)) != TARGET_RECORDS:
        raise ReplayError("prepared input list contains duplicate sources")
    records = selection.get("records")
    if not isinstance(records, list) or len(records) != TARGET_RECORDS:
        raise ReplayError("prepared selection records changed")
    selection_by_source: dict[str, dict[str, Any]] = {}
    roles = {"recipient_missing": 0, "control": 0}
    canonical_indexes: list[int] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ReplayError(f"prepared selection row {index} is not an object")
        source = _nonempty_string(
            row.get("source"), description=f"prepared selection row {index} source"
        )
        if _source_key(source) != keys[index]:
            raise ReplayError("prepared selection order differs from inputs.txt")
        role = row.get("role")
        if role not in roles:
            raise ReplayError(f"prepared selection row {index} has invalid role")
        roles[str(role)] += 1
        canonical_index = _integer(
            row.get("canonical_index"),
            description=f"prepared selection row {index} canonical index",
        )
        canonical_indexes.append(canonical_index)
        key = keys[index]
        if key in selection_by_source:
            raise ReplayError(f"duplicate prepared selection source: {source!r}")
        for identity_name in ("image", "old_baseline_result", "old_hybrid_result"):
            _assert_identity(
                row.get(identity_name),
                description=f"prepared selection {identity_name}",
            )
        old_baseline = _load_json(
            Path(str(row["old_baseline_result"]["path"])),
            description="prepared old baseline result",
        )
        old_hybrid = _load_json(
            Path(str(row["old_hybrid_result"]["path"])),
            description="prepared old hybrid result",
        )
        if (
            not isinstance(old_baseline, Mapping)
            or not isinstance(old_hybrid, Mapping)
            or _source_key(old_baseline.get("source")) != key
            or _source_key(old_hybrid.get("source")) != key
        ):
            raise ReplayError("prepared old result/source binding changed")
        old_recipient = _field(old_hybrid, "recipient")
        old_status = _field(old_hybrid, "transfer_status")
        if not _type_sensitive_equal(
            row.get("old_recipient_candidate"), old_recipient.get("candidate")
        ) or not _type_sensitive_equal(
            row.get("old_recipient_ctc_candidate"),
            old_recipient.get("ctc_candidate"),
        ):
            raise ReplayError("prepared old recipient candidate snapshot changed")
        if not _type_sensitive_equal(
            row.get("old_status_candidate"), old_status.get("candidate")
        ):
            raise ReplayError("prepared old status snapshot changed")
        if role == "recipient_missing":
            if row.get("old_recipient_candidate") not in (None, "") or row.get(
                "old_recipient_ctc_candidate"
            ) not in (None, ""):
                raise ReplayError("prepared missing source had an old recipient candidate")
            if row.get("control_tokens") != []:
                raise ReplayError("prepared missing source unexpectedly has control tokens")
        elif not isinstance(row.get("old_recipient_candidate"), str) or not row.get(
            "old_recipient_candidate"
        ):
            raise ReplayError("prepared control had no old recipient candidate")
        elif (
            row.get("old_recipient_ctc_candidate")
            != row.get("old_recipient_candidate")
            or row.get("control_tokens") != list(_control_tokens(old_hybrid))
        ):
            raise ReplayError("prepared control candidate/CTC/tokens changed")
        selection_by_source[key] = dict(row)
    if roles != {"recipient_missing": MISSING_RECORDS, "control": CONTROL_RECORDS}:
        raise ReplayError("prepared role counts changed")
    control_selection = selection.get("control_selection")
    if (
        not isinstance(control_selection, Mapping)
        or control_selection.get("strategy")
        != "greedy_device_status_route_coverage_then_formal_input_order_fill"
    ):
        raise ReplayError("prepared deterministic control strategy changed")
    control_sets: dict[str, set[str]] = {}
    for name in ("available_tokens", "covered_tokens", "uncovered_tokens"):
        values = control_selection.get(name)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            raise ReplayError(f"prepared control_selection.{name} is not canonical")
        control_sets[name] = set(values)
    selected_control_tokens = {
        token
        for row in selection_by_source.values()
        if row["role"] == "control"
        for token in row["control_tokens"]
    }
    if (
        control_sets["covered_tokens"] != selected_control_tokens
        or not control_sets["covered_tokens"].issubset(
            control_sets["available_tokens"]
        )
        or control_sets["uncovered_tokens"]
        != control_sets["available_tokens"] - control_sets["covered_tokens"]
    ):
        raise ReplayError("prepared deterministic control coverage changed")
    if canonical_indexes != sorted(canonical_indexes) or len(set(canonical_indexes)) != len(
        canonical_indexes
    ) or any(index >= FORMAL_RECORDS for index in canonical_indexes):
        raise ReplayError("prepared selection is not in canonical formal input order")
    for name, identity in source_evidence.items():
        _assert_identity(identity, description=f"prepared source evidence {name}")
    subset_rows = _load_jsonl(
        paths["subset_records"], description="prepared subset records"
    )
    subset_keys = {
        _source_key(
            _nonempty_string(row.get("source"), description="prepared subset record source")
        )
        for row in subset_rows
    }
    if subset_keys != set(keys):
        raise ReplayError("prepared subset records source set changed")
    score_references = _score_references(subset_rows)
    if {source_key for source_key, _ in score_references} - set(keys):
        raise ReplayError("prepared scorer reference domain escaped the selection")
    old_manifest = _load_json(
        paths["old_hybrid_subset_manifest"], description="old hybrid subset manifest"
    )
    if not isinstance(old_manifest, list) or len(old_manifest) != TARGET_RECORDS:
        raise ReplayError("old hybrid subset manifest count changed")
    if [_source_key(row.get("source")) for row in old_manifest if isinstance(row, Mapping)] != keys:
        raise ReplayError("old hybrid subset manifest order changed")
    formal_hybrid_manifest = _load_json(
        Path(str(source_evidence["formal_hybrid_manifest"]["path"])),
        description="prepared formal hybrid manifest",
    )
    if not isinstance(formal_hybrid_manifest, list):
        raise ReplayError("prepared formal hybrid manifest is not an array")
    formal_hybrid_by_source: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(formal_hybrid_manifest):
        if not isinstance(row, Mapping):
            raise ReplayError(f"prepared formal hybrid manifest row {index} is invalid")
        source = _nonempty_string(
            row.get("source"),
            description=f"prepared formal hybrid manifest row {index} source",
        )
        key = _source_key(source)
        if key in formal_hybrid_by_source:
            raise ReplayError("prepared formal hybrid manifest has duplicate sources")
        formal_hybrid_by_source[key] = row
    old_manifest_by_source: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(old_manifest):
        if not isinstance(row, Mapping):
            raise ReplayError(f"old hybrid subset manifest row {index} is invalid")
        key = keys[index]
        if not _type_sensitive_equal(row, formal_hybrid_by_source.get(key)):
            raise ReplayError("old hybrid subset row differs from the frozen formal manifest")
        if not _same_path(
            row.get("result"),
            Path(str(selection_by_source[key]["old_hybrid_result"]["path"])),
        ):
            raise ReplayError("old hybrid subset result binding changed")
        old_manifest_by_source[key] = dict(row)
    return {
        "root": root,
        "summary": dict(summary),
        "selection": dict(selection),
        "sources": sources,
        "keys": keys,
        "selection_by_source": selection_by_source,
        "paths": paths,
        "identities": identities,
        "score_references": score_references,
        "old_manifest_by_source": old_manifest_by_source,
    }


def _prediction_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in RESULT_EXCLUDED_TOP_LEVEL_KEYS
    }


def _manifest_stable_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in MANIFEST_EXCLUDED_KEYS}


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
    return left == right


def _strict_detections(
    result: Mapping[str, Any], *, description: str
) -> dict[str, Mapping[str, Any]]:
    detections = result.get("detections")
    if not isinstance(detections, list):
        raise ReplayError(f"{description} has no detections array")
    by_label: dict[str, Mapping[str, Any]] = {}
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise ReplayError(f"{description} detection {index} is not an object")
        label = detection.get("label")
        if not isinstance(label, str) or not label:
            raise ReplayError(f"{description} detection {index} has no label")
        if label in by_label:
            raise ReplayError(f"{description} has duplicate detection label {label!r}")
        by_label[label] = detection
    return by_label


def _canonical_detector_score(field: Mapping[str, Any], *, description: str) -> Any:
    detector_score = field.get("detector_score")
    compatibility_score = field.get("score")
    if detector_score is not None and compatibility_score is not None:
        raise ReplayError(f"{description} has both detector_score and score")
    value = detector_score if detector_score is not None else compatibility_score
    if value is not None:
        _finite_number(value, description=f"{description} detector score")
    return value


def _assert_hybrid_invariants(
    baseline: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Recheck the A/B semantic invariants with type-sensitive equality."""

    for key in (
        "result_schema_version",
        "result_semantics_version",
        "source",
        "inference_engine",
        "geometry",
        "device",
    ):
        if not _type_sensitive_equal(baseline.get(key), hybrid.get(key)):
            raise ReplayError(f"targeted hybrid {source!r} changed {key}")

    baseline_contracts = baseline.get("model_contracts")
    hybrid_contracts = hybrid.get("model_contracts")
    if not isinstance(baseline_contracts, Mapping) or not isinstance(
        hybrid_contracts, Mapping
    ):
        raise ReplayError(f"targeted hybrid {source!r} has no model contracts")
    for key in INVARIANT_CONTRACT_KEYS:
        if not _type_sensitive_equal(
            baseline_contracts.get(key), hybrid_contracts.get(key)
        ):
            raise ReplayError(
                f"targeted hybrid {source!r} changed model_contracts.{key}"
            )

    baseline_fields = baseline.get("fields")
    hybrid_fields = hybrid.get("fields")
    if not isinstance(baseline_fields, Mapping) or not isinstance(
        hybrid_fields, Mapping
    ):
        raise ReplayError(f"targeted hybrid {source!r} has no fields object")
    for field in INVARIANT_RESULT_FIELDS:
        if not _type_sensitive_equal(
            baseline_fields.get(field), hybrid_fields.get(field)
        ):
            raise ReplayError(f"targeted hybrid {source!r} changed fields.{field}")
    baseline_recipient = baseline_fields.get("recipient")
    hybrid_recipient = hybrid_fields.get("recipient")
    if not isinstance(baseline_recipient, Mapping) or not isinstance(
        hybrid_recipient, Mapping
    ):
        raise ReplayError(f"targeted hybrid {source!r} has no recipient field")
    baseline_score = _canonical_detector_score(
        baseline_recipient, description="baseline recipient"
    )
    hybrid_score = _canonical_detector_score(
        hybrid_recipient, description="hybrid recipient"
    )
    if not _type_sensitive_equal(baseline_score, hybrid_score):
        raise ReplayError(f"targeted hybrid {source!r} changed recipient detector score")
    for key in ("delivery_policy", "delivery_value", "value"):
        if not _type_sensitive_equal(
            baseline_recipient.get(key), hybrid_recipient.get(key)
        ):
            raise ReplayError(f"targeted hybrid {source!r} changed recipient {key}")

    baseline_detections = _strict_detections(
        baseline, description="targeted baseline result"
    )
    hybrid_detections = _strict_detections(
        hybrid, description="targeted hybrid result"
    )
    if set(baseline_detections) != set(hybrid_detections):
        raise ReplayError(f"targeted hybrid {source!r} changed detection labels")
    for label, baseline_detection in baseline_detections.items():
        hybrid_detection = hybrid_detections[label]
        if label == "recipient_field":
            baseline_stable = {
                key: value for key, value in baseline_detection.items() if key != "ocr"
            }
            hybrid_stable = {
                key: value for key, value in hybrid_detection.items() if key != "ocr"
            }
        else:
            baseline_stable = baseline_detection
            hybrid_stable = hybrid_detection
        if not _type_sensitive_equal(baseline_stable, hybrid_stable):
            raise ReplayError(
                f"targeted hybrid {source!r} changed detection {label}"
            )


def _load_target_ab(
    comparison: Path,
    *,
    prepared: Mapping[str, Any],
    baseline: Mapping[str, Any],
    hybrid: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = comparison / "summary.json"
    rows_path = comparison / "comparisons.jsonl"
    summary_identity = _file_identity(summary_path, description="targeted A/B summary")
    rows_identity = _file_identity(rows_path, description="targeted A/B comparisons")
    summary = _load_json(summary_path, description="targeted A/B summary")
    rows = _load_jsonl(rows_path, description="targeted A/B comparisons")
    if not isinstance(summary, Mapping):
        raise ReplayError("targeted A/B summary must be an object")
    exact = {
        "schema_version": AB_SCHEMA_VERSION,
        "kind": AB_KIND,
        "evaluation_mode": "pilot",
        "records": TARGET_RECORDS,
        "input_set_identical": True,
        "cli_summary_counts_verified": True,
        "invariant_records": TARGET_RECORDS,
        "recipient_candidate_coverage": 1.0,
        "accepted": True,
        "failures": [],
    }
    for key, expected in exact.items():
        if not _type_sensitive_equal(summary.get(key), expected):
            raise ReplayError(f"targeted A/B summary {key} must be {expected!r}")
    cli_build = _validate_cli_build(summary)
    if len(rows) != TARGET_RECORDS:
        raise ReplayError(f"targeted A/B comparisons must contain {TARGET_RECORDS} rows")
    input_set = summary.get("input_set")
    if not isinstance(input_set, Mapping) or input_set.get("records") != TARGET_RECORDS:
        raise ReplayError("targeted A/B summary has no exact input-set binding")
    _assert_bound_identity(
        input_set.get("input_manifest"),
        {
            **prepared["identities"]["inputs"],
            "records": TARGET_RECORDS,
            "normalized_source_set_sha256": _normalized_source_set_sha256(
                tuple(prepared["keys"])
            ),
        },
        description="targeted input list",
    )
    if input_set.get("normalized_source_set_sha256") != _normalized_source_set_sha256(
        tuple(prepared["keys"])
    ):
        raise ReplayError("targeted A/B source-set hash differs from the prepared selection")
    run_manifests = summary.get("run_manifests")
    run_summaries = summary.get("run_summaries")
    if not isinstance(run_manifests, Mapping) or not isinstance(run_summaries, Mapping):
        raise ReplayError("targeted A/B summary lacks run bindings")
    _assert_bound_identity(
        run_manifests.get("baseline"),
        baseline["manifest_identity"],
        description="targeted baseline manifest",
    )
    _assert_bound_identity(
        run_manifests.get("hybrid"),
        hybrid["manifest_identity"],
        description="targeted hybrid manifest",
    )
    _assert_bound_identity(
        run_summaries.get("baseline"),
        baseline["summary_identity"],
        description="targeted baseline summary",
    )
    _assert_bound_identity(
        run_summaries.get("hybrid"),
        hybrid["summary_identity"],
        description="targeted hybrid summary",
    )
    cpu = summary.get("cpu")
    if not isinstance(cpu, Mapping):
        raise ReplayError("targeted A/B summary has no CPU evidence")
    overhead = _finite_number(
        cpu.get("p95_overhead_ms"), description="targeted CPU p95 overhead"
    )
    ceiling = _finite_number(
        cpu.get("max_p95_overhead_ms"), description="targeted CPU p95 ceiling"
    )
    if ceiling < 0 or ceiling > MAX_P95_OVERHEAD_MS or overhead > ceiling or overhead > MAX_P95_OVERHEAD_MS:
        raise ReplayError(
            f"targeted CPU p95 overhead {overhead:.4f} exceeds the fixed {MAX_P95_OVERHEAD_MS:.1f} ms ceiling"
        )
    recomputed_overhead = _finite_number(
        hybrid["summary"]["inference_latency_ms"].get("p95"),
        description="targeted hybrid p95",
    ) - _finite_number(
        baseline["summary"]["inference_latency_ms"].get("p95"),
        description="targeted baseline p95",
    )
    if abs(overhead - recomputed_overhead) > 0.00011:
        raise ReplayError("targeted CPU p95 overhead differs from the bound run summaries")
    if not _type_sensitive_equal(
        baseline["summary"].get("detector_intra_op_threads"),
        hybrid["summary"].get("detector_intra_op_threads"),
    ):
        raise ReplayError("targeted baseline/hybrid detector thread settings differ")
    by_source: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        source = _nonempty_string(
            row.get("source"), description=f"targeted comparison row {index} source"
        )
        key = _source_key(source)
        candidate = row.get("recipient_candidate")
        if (
            row.get("invariant") is not True
            or row.get("failures") != []
            or not isinstance(candidate, str)
            or not candidate
        ):
            raise ReplayError("every targeted A/B comparison must be invariant with a recipient candidate")
        if key in by_source:
            raise ReplayError(f"duplicate targeted comparison source: {source!r}")
        recipient = _field(hybrid["results"][key]["payload"], "recipient")
        if recipient.get("candidate") != candidate or recipient.get("ctc_candidate") != candidate:
            raise ReplayError("targeted comparator candidate differs from hybrid candidate/CTC")
        by_source[key] = row
    if set(by_source) != set(prepared["keys"]):
        raise ReplayError("targeted A/B comparison source set differs from selection")
    return {
        "summary": dict(summary),
        "summary_identity": summary_identity,
        "rows_identity": rows_identity,
        "p95_overhead_ms": overhead,
        "cli_build": cli_build,
    }


def _load_score(
    directory: Path,
    *,
    prepared: Mapping[str, Any],
    expected_manifest: Path,
    expected_results_root: Path,
    expected_entries: Mapping[str, Mapping[str, Any]],
    description: str,
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    rows_path = directory / "comparisons.jsonl"
    summary_identity = _file_identity(summary_path, description=f"{description} summary")
    rows_identity = _file_identity(rows_path, description=f"{description} comparisons")
    summary = _load_json(summary_path, description=f"{description} summary")
    rows = _load_jsonl(rows_path, description=f"{description} comparisons")
    if not isinstance(summary, Mapping):
        raise ReplayError(f"{description} summary must be an object")
    if summary.get("schema_version") != 1 or summary.get("kind") != SCORE_KIND:
        raise ReplayError(f"{description} is not a unified candidate scorer report")
    if (
        summary.get("formal_delivery_gate") is not False
        or summary.get("accepted") is not False
        or not isinstance(summary.get("acceptance"), Mapping)
        or summary["acceptance"].get("formal_delivery_gate") is not False
        or summary.get("evaluation_split") != "val"
    ):
        raise ReplayError(f"{description} must remain formal_delivery_gate=false on val")
    scope = summary.get("evaluation_scope")
    if not isinstance(scope, Mapping):
        raise ReplayError(f"{description} has no evaluation scope")
    # Score the derived records manifest as an *unbound* full split.  Passing
    # the targeted list with a positive limit would invoke the scorer's own
    # five-field pilot ordering, which is intentionally different from this
    # formal-order replay selection.  The gate itself supplies the stronger
    # records/manifest/hash closure while the scorer stays diagnostic-only.
    scope_exact = {
        "kind": "full_split",
        "requested_limit": None,
        "evaluated_expected_receipts": TARGET_RECORDS,
        "full_split_expected_receipts": TARGET_RECORDS,
        "input_list_sha256": None,
        "formal_delivery_gate": False,
    }
    for key, expected in scope_exact.items():
        if type(scope.get(key)) is not type(expected) or scope.get(key) != expected:
            raise ReplayError(f"{description} evaluation_scope.{key} must be {expected!r}")
    if scope.get("input_list_path") is not None:
        raise ReplayError(f"{description} must be an unbound diagnostic score")
    selection = summary.get("input_selection")
    if selection is not None:
        raise ReplayError(f"{description} must not claim a formal/hash-bound input selection")
    if summary.get("records_sha256") != prepared["identities"]["subset_records"]["sha256"] or not _same_path(
        summary.get("records"), prepared["paths"]["subset_records"]
    ):
        raise ReplayError(f"{description} subset-records binding mismatch")
    model_hashes: set[str] = set()
    for source_key, entry in expected_entries.items():
        result = entry.get("payload")
        if not isinstance(result, Mapping):
            raise ReplayError(f"{description} has no bound result for {source_key!r}")
        contracts = result.get("model_contracts")
        model_hash = (
            contracts.get("unified_ocr_model_sha256")
            if isinstance(contracts, Mapping)
            else None
        )
        if not isinstance(model_hash, str) or LOWER_SHA256.fullmatch(model_hash) is None:
            raise ReplayError(
                f"{description} bound result {source_key!r} has no valid unified model hash"
            )
        model_hashes.add(model_hash)
    if len(model_hashes) != 1 or summary.get("model_sha256") != next(iter(model_hashes)):
        raise ReplayError(f"{description} model hash differs from its bound results")
    model_path_value = summary.get("model")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise ReplayError(f"{description} has no unified model path")
    model_identity = _file_identity(
        Path(model_path_value), description=f"{description} unified model"
    )
    if model_identity["sha256"] != summary.get("model_sha256"):
        raise ReplayError(f"{description} unified model bytes differ from model_sha256")
    if not _same_path(summary.get("manifest"), expected_manifest):
        raise ReplayError(f"{description} points to the wrong inference manifest")
    if summary.get("manifest_sha256") != _sha256(expected_manifest):
        raise ReplayError(f"{description} inference manifest hash mismatch")
    if not _same_path(summary.get("results_root"), expected_results_root):
        raise ReplayError(f"{description} results root mismatch")
    floors = summary.get("floors")
    if not isinstance(floors, Mapping):
        raise ReplayError(f"{description} has no fixed floors")
    for field, expected in FIXED_FLOORS.items():
        if type(floors.get(field)) not in (int, float) or isinstance(
            floors.get(field), bool
        ) or float(floors[field]) != expected:
            raise ReplayError(f"{description} changed the fixed {field} floor")
    coverage = summary.get("coverage")
    missing = summary.get("missing")
    artifact_audit = summary.get("artifact_audit")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("expected_receipts") != TARGET_RECORDS
        or coverage.get("matched_result_receipts") != TARGET_RECORDS
        or coverage.get("result_coverage") != 1.0
        or coverage.get("extra_manifest_sources") != []
        or not isinstance(missing, Mapping)
        or missing.get("result_receipts") != 0
        or not isinstance(artifact_audit, Mapping)
        or artifact_audit.get("manifest_records") != TARGET_RECORDS
        or artifact_audit.get("usable_manifest_sources") != TARGET_RECORDS
        or artifact_audit.get("all_results_match_model") is not True
    ):
        raise ReplayError(f"{description} does not have complete result/model coverage")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    expected_domain = prepared.get("score_references")
    if not isinstance(expected_domain, Mapping):
        raise ReplayError(f"{description} has no frozen scorer reference domain")
    for index, row in enumerate(rows):
        source = _nonempty_string(
            row.get("source"), description=f"{description} comparison row {index} source"
        )
        field = _nonempty_string(
            row.get("field"), description=f"{description} comparison row {index} field"
        )
        source_key = _source_key(source)
        if source_key not in set(prepared["keys"]):
            raise ReplayError(f"{description} contains a source outside the targeted selection")
        key = (source_key, field)
        if key in by_key:
            raise ReplayError(f"{description} contains duplicate source/field comparison")
        expected_reference = expected_domain.get(key)
        if not isinstance(expected_reference, Mapping):
            raise ReplayError(
                f"{description} contains a comparison outside the frozen reference domain"
            )
        for reference_key in (
            "schema_version",
            "kind",
            "id",
            "group_id",
            "split",
            "source",
            "teacher_result_json",
            "field",
            "reference_text",
            "reference_crop_sha256",
            "reference_detector_score",
            "reference_bbox_rectified",
        ):
            if not _type_sensitive_equal(
                row.get(reference_key), expected_reference.get(reference_key)
            ):
                raise ReplayError(
                    f"{description} {field} {reference_key} differs from subset-records"
                )
        if type(row.get("candidate_present")) is not bool or type(row.get("raw_exact")) is not bool:
            raise ReplayError(f"{description} comparison booleans are invalid")
        result_field = SCORE_RESULT_FIELDS.get(field)
        if result_field is None:
            raise ReplayError(f"{description} contains unsupported score field {field!r}")
        entry = expected_entries.get(source_key)
        result = entry.get("payload") if isinstance(entry, Mapping) else None
        if not isinstance(entry, Mapping) or not isinstance(result, Mapping):
            raise ReplayError(f"{description} has no bound result for {source!r}")
        result_path = Path(
            _nonempty_string(
                entry.get("path"), description=f"{description} bound result path"
            )
        )
        if not _same_path(row.get("result_json"), result_path):
            raise ReplayError(f"{description} result_json differs from its manifest")
        if not _type_sensitive_equal(row.get("manifest_status"), entry.get("status")):
            raise ReplayError(f"{description} manifest status differs from its manifest")
        contracts = result.get("model_contracts")
        result_model_sha256 = (
            contracts.get("unified_ocr_model_sha256")
            if isinstance(contracts, Mapping)
            else None
        )
        if not _type_sensitive_equal(
            row.get("unified_model_sha256"), result_model_sha256
        ):
            raise ReplayError(f"{description} comparison model hash differs from its result")
        expected_candidate = _candidate(result, result_field)
        expected_present = expected_candidate is not None
        result_field_payload = _field(result, result_field)
        expected_ctc = result_field_payload.get("ctc_candidate")
        expected_ctc = expected_ctc if isinstance(expected_ctc, str) else None
        expected_structured = result_field_payload.get("structured_candidate")
        expected_structured = (
            expected_structured if isinstance(expected_structured, str) else None
        )
        if (
            row.get("candidate_text") != expected_candidate
            or row.get("candidate_present") is not expected_present
            or row.get("ctc_candidate_text") != expected_ctc
            or row.get("structured_candidate_text") != expected_structured
            or row.get("raw_exact")
            is not (expected_present and expected_candidate == row.get("reference_text"))
        ):
            raise ReplayError(
                f"{description} {field} comparison differs from its bound result/reference"
            )
        if field == "transfer_status":
            if not _type_sensitive_equal(
                row.get("reference_status_class"),
                expected_reference.get("reference_status_class"),
            ):
                raise ReplayError(
                    f"{description} transfer-status truth class differs from subset-records"
                )
            expected_candidate_class = (
                normalize_status(expected_candidate) if expected_candidate is not None else None
            )
            if row.get("candidate_status_class") != expected_candidate_class:
                raise ReplayError(
                    f"{description} transfer-status candidate class is inconsistent"
                )
            expected_transition = (
                row.get("reference_status_class") in {"pending", "failed"}
                and expected_candidate_class == "success"
            )
            if row.get("non_success_to_success") is not expected_transition:
                raise ReplayError(
                    f"{description} transfer-status transition boolean is inconsistent"
                )
        by_key[key] = row
        metrics = counts.setdefault(field, {"records": 0, "candidates": 0, "exact": 0})
        metrics["records"] += 1
        metrics["candidates"] += int(row["candidate_present"])
        metrics["exact"] += int(row["raw_exact"])
    if set(by_key) != set(expected_domain):
        missing_rows = set(expected_domain) - set(by_key)
        raise ReplayError(
            f"{description} scorer domain is incomplete: missing={len(missing_rows)}"
        )
    summary_by_field = summary.get("by_field")
    if not isinstance(summary_by_field, Mapping) or set(summary_by_field) != set(
        FIXED_FLOORS
    ):
        raise ReplayError(f"{description} has no by_field metrics")
    if set(counts) != set(FIXED_FLOORS):
        raise ReplayError(f"{description} does not score all five frozen fields")
    for field in FIXED_FLOORS:
        metrics = counts[field]
        observed = summary_by_field.get(field)
        if not isinstance(observed, Mapping):
            raise ReplayError(f"{description} has no {field} metrics")
        for name, expected in (
            ("records", metrics["records"]),
            ("candidate_records", metrics["candidates"]),
            ("raw_exact_matches", metrics["exact"]),
        ):
            if observed.get(name) != expected:
                raise ReplayError(f"{description} {field}.{name} differs from comparisons")
        expected_exact = metrics["exact"] / metrics["records"]
        expected_coverage = metrics["candidates"] / metrics["records"]
        if abs(
            _finite_number(
                observed.get("raw_exact_match"),
                description=f"{description} {field}.raw_exact_match",
            )
            - expected_exact
        ) > 1e-12 or abs(
            _finite_number(
                observed.get("candidate_coverage"),
                description=f"{description} {field}.candidate_coverage",
            )
            - expected_coverage
        ) > 1e-12:
            raise ReplayError(f"{description} {field} ratios differ from comparisons")
    status_rows = [
        row for (source_key, field), row in by_key.items() if field == "transfer_status"
    ]
    expected_non_success = sum(
        row.get("reference_status_class") in {"pending", "failed"}
        for row in status_rows
    )
    expected_unsafe = sum(bool(row.get("non_success_to_success")) for row in status_rows)
    status_metrics = summary_by_field["transfer_status"]
    if (
        status_metrics.get("non_success_truth_records") != expected_non_success
        or status_metrics.get("non_success_to_success") != expected_unsafe
    ):
        raise ReplayError(f"{description} transfer-status metrics differ from comparisons")
    denominators = summary.get("accuracy_denominators")
    expected_denominators = {
        field: counts[field]["records"] for field in FIXED_FLOORS
    }
    if (
        not isinstance(denominators, Mapping)
        or denominators.get("scope") != "selected_reference_records"
        or denominators.get("hash_bound") is not False
        or denominators.get("source")
        != "records_manifest_selected_field_reference_counts"
        or not _type_sensitive_equal(
            denominators.get("by_field"), expected_denominators
        )
    ):
        raise ReplayError(f"{description} accuracy denominators are not subset-bound")
    return {
        "summary": dict(summary),
        "summary_identity": summary_identity,
        "rows_identity": rows_identity,
        "rows": by_key,
        "model_identity": model_identity,
        "counts": counts,
    }


def gate(
    *,
    prepared: Path,
    baseline: Path,
    hybrid: Path,
    comparison: Path,
    old_score: Path,
    new_score: Path,
    output: Path,
) -> dict[str, Any]:
    """Gate already-produced targeted evidence; never run OCR or publish formal."""

    if output.is_symlink():
        raise ReplayError(f"refusing a symbolic-link targeted replay gate: {output}")
    output = output.resolve()
    if output.exists():
        raise ReplayError(f"refusing to overwrite targeted replay gate: {output}")
    frozen = _load_prepared(prepared)
    disallowed_roots = {
        frozen["root"].resolve(),
        Path(
            str(
                frozen["selection"]["source_evidence"]["formal_input_list"][
                    "path"
                ]
            )
        ).parent.resolve(),
        Path(str(frozen["selection"]["source_evidence"]["formal_baseline_manifest"]["path"])).parent.resolve(),
        Path(str(frozen["selection"]["source_evidence"]["formal_hybrid_manifest"]["path"])).parent.resolve(),
    }
    baseline_root = baseline.resolve(strict=True)
    hybrid_root = hybrid.resolve(strict=True)
    if _paths_overlap(baseline_root, hybrid_root) or any(
        _paths_overlap(candidate, old_root)
        for candidate in (baseline_root, hybrid_root)
        for old_root in disallowed_roots
    ):
        raise ReplayError("new targeted runs must be distinct from each other and all old formal runs")
    new_baseline = _load_run(
        baseline_root, expected_sources=frozen["sources"], hybrid=False
    )
    new_hybrid = _load_run(
        hybrid_root, expected_sources=frozen["sources"], hybrid=True
    )
    old_baseline_manifest_path = Path(
        str(
            frozen["selection"]["source_evidence"]["formal_baseline_manifest"][
                "path"
            ]
        )
    )
    old_baseline_manifest = _load_json(
        old_baseline_manifest_path, description="old formal baseline manifest"
    )
    if not isinstance(old_baseline_manifest, list):
        raise ReplayError("old formal baseline manifest is not an array")
    old_baseline_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(old_baseline_manifest):
        if not isinstance(row, Mapping):
            raise ReplayError(f"old formal baseline manifest row {index} is invalid")
        key = _source_key(row.get("source"))
        if key in old_baseline_rows:
            raise ReplayError("old formal baseline manifest contains duplicate sources")
        old_baseline_rows[key] = row
    old_baseline_summary_path = Path(
        str(
            frozen["selection"]["source_evidence"]["formal_baseline_summary"][
                "path"
            ]
        )
    )
    old_baseline_summary = _load_json(
        old_baseline_summary_path, description="old formal baseline summary"
    )
    if not isinstance(old_baseline_summary, Mapping):
        raise ReplayError("old formal baseline summary is not an object")
    for key in (
        "requested_device",
        "paddle_ocr_provider",
        "unified_provider",
        "detector_intra_op_threads",
    ):
        if not _type_sensitive_equal(
            old_baseline_summary.get(key), new_baseline["summary"].get(key)
        ):
            raise ReplayError(f"old baseline -> new baseline summary setting {key} changed")
    baseline_differences = 0
    for key in frozen["keys"]:
        selected = frozen["selection_by_source"][key]
        old_path = Path(str(selected["old_baseline_result"]["path"])).resolve(strict=True)
        old_payload = _load_json(old_path, description="old formal baseline result")
        new_entry = new_baseline["results"][key]
        if new_entry["path"] == old_path:
            raise ReplayError("old formal baseline result was reused as a new targeted result")
        if not isinstance(old_payload, Mapping) or not _type_sensitive_equal(
            _prediction_payload(old_payload),
            _prediction_payload(new_entry["payload"]),
        ):
            baseline_differences += 1
        old_manifest_row = old_baseline_rows.get(key)
        new_manifest_row = new_baseline["rows"][key]
        if not isinstance(old_manifest_row, Mapping) or not _type_sensitive_equal(
            _manifest_stable_payload(old_manifest_row),
            _manifest_stable_payload(new_manifest_row),
        ):
            baseline_differences += 1
    if baseline_differences:
        raise ReplayError(
            f"old baseline -> new baseline prediction differences: {baseline_differences}"
        )
    for key in frozen["keys"]:
        new_path = new_hybrid["results"][key]["path"]
        old_path = Path(
            str(frozen["selection_by_source"][key]["old_hybrid_result"]["path"])
        ).resolve(strict=True)
        if new_path == old_path:
            raise ReplayError("old formal hybrid result was reused as a new targeted result")
        recipient = _field(new_hybrid["results"][key]["payload"], "recipient")
        candidate = recipient.get("candidate")
        if not isinstance(candidate, str) or not candidate or recipient.get("ctc_candidate") != candidate:
            raise ReplayError("new targeted hybrid result lacks an agreeing recipient candidate/CTC")
        _assert_hybrid_invariants(
            new_baseline["results"][key]["payload"],
            new_hybrid["results"][key]["payload"],
            source=new_hybrid["results"][key]["source"],
        )
    ab = _load_target_ab(
        comparison,
        prepared=frozen,
        baseline=new_baseline,
        hybrid=new_hybrid,
    )
    old = _load_score(
        old_score,
        prepared=frozen,
        expected_manifest=frozen["paths"]["old_hybrid_subset_manifest"],
        expected_results_root=frozen["paths"]["old_hybrid_subset_manifest"].parent,
        expected_entries={
            key: {
                "payload": _load_json(
                    Path(
                        str(
                            frozen["selection_by_source"][key][
                                "old_hybrid_result"
                            ]["path"]
                        )
                    ),
                    description="old formal hybrid result",
                ),
                "path": frozen["selection_by_source"][key]["old_hybrid_result"][
                    "path"
                ],
                "status": frozen["old_manifest_by_source"][key].get("status"),
            }
            for key in frozen["keys"]
        },
        description="old targeted score",
    )
    new = _load_score(
        new_score,
        prepared=frozen,
        expected_manifest=new_hybrid["manifest_path"],
        expected_results_root=new_hybrid["root"],
        expected_entries={
            key: {
                "payload": new_hybrid["results"][key]["payload"],
                "path": new_hybrid["results"][key]["path"].as_posix(),
                "status": new_hybrid["rows"][key].get("status"),
            }
            for key in frozen["keys"]
        },
        description="new targeted score",
    )
    if set(old["rows"]) != set(new["rows"]):
        raise ReplayError("old/new targeted score comparison domains differ")
    if old["model_identity"]["sha256"] != new["model_identity"]["sha256"]:
        raise ReplayError("old/new targeted scores used different unified models")
    reference_keys = (
        "schema_version",
        "kind",
        "id",
        "group_id",
        "split",
        "source",
        "teacher_result_json",
        "field",
        "reference_text",
        "reference_crop_sha256",
        "reference_detector_score",
        "reference_bbox_rectified",
        "reference_status_class",
    )
    control_regressions: list[str] = []
    correct_to_wrong: list[tuple[str, str]] = []
    status_non_success_to_success: list[str] = []
    for key, old_row in old["rows"].items():
        new_row = new["rows"][key]
        for reference_key in reference_keys:
            if not _type_sensitive_equal(
                old_row.get(reference_key), new_row.get(reference_key)
            ):
                raise ReplayError(
                    f"old/new score reference evidence changed for {key[1]}"
                )
        source_key, field = key
        role = frozen["selection_by_source"][source_key]["role"]
        if old_row.get("raw_exact") is True and new_row.get("raw_exact") is not True:
            correct_to_wrong.append((str(new_row.get("source")), field))
        if (
            role == "control"
            and field == "recipient_field"
            and old_row.get("raw_exact") is True
            and new_row.get("raw_exact") is not True
        ):
            control_regressions.append(str(new_row.get("source")))
        if field == "transfer_status" and new_row.get("non_success_to_success") is True:
            status_non_success_to_success.append(str(new_row.get("source")))
    if control_regressions:
        raise ReplayError(
            f"recipient correct->wrong control regressions: {len(control_regressions)}"
        )
    if correct_to_wrong:
        raise ReplayError(
            f"old/new score correct->wrong regressions: {len(correct_to_wrong)}"
        )
    if status_non_success_to_success:
        raise ReplayError(
            "transfer status non-success->success regressions: "
            f"{len(status_non_success_to_success)}"
        )
    new_status_metrics = new["summary"].get("by_field", {}).get("transfer_status")
    if (
        not isinstance(new_status_metrics, Mapping)
        or type(new_status_metrics.get("non_success_truth_records")) is not int
        or new_status_metrics["non_success_truth_records"] <= 0
    ):
        raise ReplayError("new targeted score has no non-success status truth coverage")
    if new_status_metrics.get("non_success_to_success") != 0:
        raise ReplayError("new score summary reports a non-success->success status regression")
    score_metrics: dict[str, dict[str, Any]] = {}
    for field, floor in FIXED_FLOORS.items():
        metrics = new["counts"].get(field)
        if not isinstance(metrics, Mapping) or metrics.get("records", 0) <= 0:
            raise ReplayError(f"new targeted score has no {field} references")
        exact = int(metrics["exact"])
        records = int(metrics["records"])
        exact_match = exact / records
        if exact_match < floor:
            raise ReplayError(
                f"new targeted score {field} raw_exact_match={exact_match:.4f} < {floor:.4f}"
            )
        score_metrics[field] = {
            "records": records,
            "raw_exact_matches": exact,
            "raw_exact_match": exact_match,
            "floor": floor,
        }

    report = {
        "schema_version": 1,
        "kind": GATE_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "targeted_replay_only": True,
        "formal_delivery_gate": False,
        "accepted": True,
        "counts": {
            "selected": TARGET_RECORDS,
            "recipient_missing_recovered": MISSING_RECORDS,
            "controls": CONTROL_RECORDS,
            "baseline_prediction_differences": 0,
            "all_field_correct_to_wrong": 0,
            "control_recipient_correct_to_wrong": 0,
            "status_non_success_to_success": 0,
        },
        "cpu": {
            "p95_overhead_ms": ab["p95_overhead_ms"],
            "max_p95_overhead_ms": MAX_P95_OVERHEAD_MS,
        },
        "new_score_by_field": score_metrics,
        "warning": (
            "This 332-record targeted replay is diagnostic-only.  It is not a fresh "
            "10016-record formal run and cannot be used as delivery evidence."
        ),
        "source_evidence": {
            "prepared_summary": _file_identity(
                frozen["root"] / "summary.json", description="prepared summary"
            ),
            "prepared_selection": frozen["identities"]["selection"],
            "prepared_inputs": frozen["identities"]["inputs"],
            "prepared_subset_records": frozen["identities"]["subset_records"],
            "prepared_old_hybrid_subset_manifest": frozen["identities"][
                "old_hybrid_subset_manifest"
            ],
            "new_baseline_manifest": new_baseline["manifest_identity"],
            "new_baseline_summary": new_baseline["summary_identity"],
            "new_hybrid_manifest": new_hybrid["manifest_identity"],
            "new_hybrid_summary": new_hybrid["summary_identity"],
            "targeted_ab_summary": ab["summary_identity"],
            "targeted_ab_comparisons": ab["rows_identity"],
            "targeted_cli_assembly": ab["cli_build"]["assembly"],
            "targeted_cli_closure_manifest": ab["cli_build"][
                "closure_manifest"
            ],
            "old_score_summary": old["summary_identity"],
            "old_score_comparisons": old["rows_identity"],
            "new_score_summary": new["summary_identity"],
            "new_score_comparisons": new["rows_identity"],
            "unified_model": new["model_identity"],
        },
    }
    for identity in report["source_evidence"].values():
        _assert_identity(identity, description="targeted gate source evidence")
    for name, identity in frozen["selection"]["source_evidence"].items():
        _assert_identity(
            identity, description=f"targeted gate prepared source evidence {name}"
        )
    # Re-walk the exact CLI closure at publication time; re-hashing only its
    # manifest and primary assembly would miss a late mutation to another DLL.
    _validate_cli_build(ab["summary"])
    for run in (new_baseline, new_hybrid):
        errors_path = run["root"] / "inference_errors.jsonl"
        if not errors_path.is_file() or errors_path.read_text(encoding="utf-8-sig") != "":
            raise ReplayError("targeted inference error evidence changed before publication")
    for key in frozen["keys"]:
        selected = frozen["selection_by_source"][key]
        for identity_name in ("image", "old_baseline_result", "old_hybrid_result"):
            _assert_identity(
                selected[identity_name],
                description=f"targeted gate {identity_name}",
            )
        _assert_identity(
            new_baseline["results"][key]["identity"],
            description="targeted gate new baseline result",
        )
        _assert_identity(
            new_hybrid["results"][key]["identity"],
            description="targeted gate new hybrid result",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        _write_json(stage / "summary.json", report)
        _publish_directory(stage, output, description="targeted replay gate")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {**report, "output_directory": output.as_posix()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="freeze the strict 204 missing + 128 control selection"
    )
    prepare_parser.add_argument("--formal-root", required=True, type=Path)
    prepare_parser.add_argument("--diagnostic", required=True, type=Path)
    prepare_parser.add_argument("--records", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    gate_parser = subparsers.add_parser(
        "gate", help="validate already-produced targeted CPU A/B and score evidence"
    )
    gate_parser.add_argument("--prepared", required=True, type=Path)
    gate_parser.add_argument("--baseline", required=True, type=Path)
    gate_parser.add_argument("--hybrid", required=True, type=Path)
    gate_parser.add_argument("--comparison", required=True, type=Path)
    gate_parser.add_argument("--old-score", required=True, type=Path)
    gate_parser.add_argument("--new-score", required=True, type=Path)
    gate_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare(
                formal_root=args.formal_root,
                diagnostic=args.diagnostic,
                records=args.records,
                output=args.output,
            )
        else:
            report = gate(
                prepared=args.prepared,
                baseline=args.baseline,
                hybrid=args.hybrid,
                comparison=args.comparison,
                old_score=args.old_score,
                new_score=args.new_score,
                output=args.output,
            )
    except (ReplayError, OSError, ValueError) as error:
        print(f"Targeted hybrid replay failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "formal_delivery_gate": False,
                "output_directory": report["output_directory"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
