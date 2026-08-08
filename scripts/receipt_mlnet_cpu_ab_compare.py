#!/usr/bin/env python3
"""Fail-closed consistency and performance analysis for ML.NET CPU A/B runs.

The PowerShell orchestrator writes an immutable run plan and invokes the same
protected unified or hybrid-recipient workload for a baseline and a candidate.
This analyzer validates every run, compares every prediction with type-sensitive
JSON semantics, and pools per-image timing from the manifests. Paths and
top-level timing metadata are the only result properties excluded from
prediction comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEGACY_PLAN_KIND = "receipt_mlnet_cpu_ab_plan_v1"
PLAN_KIND = "receipt_mlnet_cpu_ab_plan_v2"
REPORT_KIND = "receipt_mlnet_cpu_ab_report_v2"
VARIANTS = ("baseline", "candidate")
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
EXPECTED_FIELDS = frozenset(
    {"time", "amount", "transfer_status", "recipient", "payment_method"}
)
EXPECTED_DETECTIONS = frozenset(
    {"time", "amount", "transfer_status", "recipient_field", "payment_method_field"}
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
ALL_STAGES = (
    "image_load",
    "device",
    "detector_preprocess",
    "detector_inference",
    "detector_postprocess",
    "paddle_ocr",
    "unified_ocr_preprocess",
    "unified_ocr_inference",
    "unified_ocr_postprocess",
    "result_assembly",
)
MODEL_HASH_FIELDS = {
    "detector": "detector_sha256",
    "detector_contract": "detector_contract_sha256",
    "device": "device_sha256",
    "device_contract": "device_contract_sha256",
    "unified_ocr": "unified_ocr_model_sha256",
    "unified_labels": "unified_ocr_labels_sha256",
    "unified_contract": "unified_ocr_contract_sha256",
}
PADDLE_BUNDLE_CONTRACT = "paddle_ocr_delivery.contract.json"
PADDLE_RESULT_HASH_FIELD = "ocr_bundle_contract_sha256"
WALL_CLOCK_KIND = "receipt_mlnet_cpu_ab_wall_clock_v1"
ROUTE_FIELDS = ("hybrid_ocr_route", "hybrid_ocr_third_route")
MAX_REPORTED_DIFFERENCES = 200
MINIMUM_THROUGHPUT_GAIN_PERCENT = 2.0
MAXIMUM_P50_REGRESSION_PERCENT = 0.0
MAXIMUM_P95_REGRESSION_PERCENT = 0.0
MAXIMUM_DETECTOR_INTRA_OP_THREADS = 256


class ValidationError(RuntimeError):
    """The A/B evidence is incomplete, inconsistent, or unsafe to accept."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"JSON contains non-finite numeric constant: {value}")


def _load_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise ValidationError(f"missing {description}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exception:
        raise ValidationError(
            f"invalid {description} {path}: {exception.msg}"
        ) from exception


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{description} must be one JSON object")
    return value


def _finite_number(value: Any, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{description} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        qualifier = "positive finite" if positive else "non-negative finite"
        raise ValidationError(f"{description} must be {qualifier}")
    return number


def _verify_file_evidence(evidence: Any, description: str) -> Path:
    row = _require_mapping(evidence, description)
    raw_path = row.get("path")
    expected_hash = row.get("sha256")
    expected_bytes = row.get("bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValidationError(f"{description} has no path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValidationError(f"{description} has no valid SHA-256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValidationError(f"{description} has no valid byte count")
    path = Path(raw_path)
    if not path.is_file():
        raise ValidationError(f"missing {description}: {path}")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_hash.casefold():
        raise ValidationError(f"{description} changed after the A/B plan was frozen: {path}")
    return path


def _safe_relative_payload_path(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{description} has no relative path")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or len(parts) == 0
        or any(part in ("", ".", "..") for part in parts)
        or ":" in parts[0]
    ):
        raise ValidationError(f"{description} has unsafe relative path: {value}")
    return "/".join(parts)


def _verify_app_payload(variant: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    executable = _verify_file_evidence(
        payload.get("executable"), f"{variant} executable"
    )
    root_raw = payload.get("app_root")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValidationError(f"variant {variant} has no app_root")
    app_root = Path(root_raw)
    if not app_root.is_dir():
        raise ValidationError(f"variant {variant} app_root is missing: {app_root}")
    app_root = app_root.resolve()
    if executable.parent.resolve() != app_root:
        raise ValidationError(f"variant {variant} executable is not directly under app_root")
    manifest_path = _verify_file_evidence(
        payload.get("app_payload"), f"{variant} app payload manifest"
    )
    rows = _load_json(manifest_path, f"{variant} app payload manifest")
    if not isinstance(rows, list) or not rows:
        raise ValidationError(f"variant {variant} app payload manifest is empty")
    listed: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"{variant} app payload row {index + 1}")
        relative = _safe_relative_payload_path(
            row.get("path"), f"{variant} app payload row {index + 1}"
        )
        key = relative.casefold()
        if key in listed:
            raise ValidationError(f"variant {variant} app payload has duplicate path: {relative}")
        sha = row.get("sha256")
        byte_count = row.get("bytes")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValidationError(f"variant {variant} app payload has invalid SHA-256: {relative}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValidationError(f"variant {variant} app payload has invalid bytes: {relative}")
        target = (app_root / Path(*relative.split("/"))).resolve()
        try:
            common = os.path.commonpath([_path_key(target), _path_key(app_root)])
        except ValueError as exception:
            raise ValidationError(
                f"variant {variant} app payload escapes to another volume: {relative}"
            ) from exception
        if common != _path_key(app_root) or target == app_root or not target.is_file():
            raise ValidationError(f"variant {variant} app payload path is missing/unsafe: {relative}")
        if target.stat().st_size != byte_count or _sha256(target) != sha.casefold():
            raise ValidationError(
                f"variant {variant} app payload changed during A/B execution: {relative}"
            )
        listed[key] = row

    actual: dict[str, str] = {}
    for target in app_root.rglob("*"):
        if target.is_symlink():
            raise ValidationError(
                f"variant {variant} app payload contains a symbolic link: {target}"
            )
        if not target.is_file():
            continue
        relative = target.relative_to(app_root).as_posix()
        key = relative.casefold()
        if key in actual:
            raise ValidationError(
                f"variant {variant} app payload has duplicate canonical path: {relative}"
            )
        actual[key] = relative
    missing = sorted(set(listed) - set(actual))
    extra = sorted(set(actual) - set(listed))
    if missing or extra:
        raise ValidationError(
            f"variant {variant} app payload manifest is not closed: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    expected_actual: dict[str, str] = {}
    for property_name in (
        "executable_relative_path",
        "managed_entrypoint_relative_path",
        "deps_json_relative_path",
        "runtimeconfig_json_relative_path",
    ):
        relative = _safe_relative_payload_path(
            payload.get(property_name), f"variant {variant} {property_name}"
        )
        if relative.casefold() not in listed:
            raise ValidationError(
                f"variant {variant} app payload omits required {property_name}: {relative}"
            )
        expected_actual[property_name] = relative
    if _path_key(app_root / expected_actual["executable_relative_path"]) != _path_key(executable):
        raise ValidationError(f"variant {variant} executable evidence and payload differ")
    managed_row = listed[expected_actual["managed_entrypoint_relative_path"].casefold()]
    executable_row = listed[expected_actual["executable_relative_path"].casefold()]
    executable_evidence = _require_mapping(payload["executable"], f"{variant} executable")
    if (
        executable_row.get("sha256") != executable_evidence.get("sha256")
        or executable_row.get("bytes") != executable_evidence.get("bytes")
    ):
        raise ValidationError(f"variant {variant} executable row is not bound to its evidence")
    dll_paths = [row.get("path") for row in rows if str(row.get("path", "")).casefold().endswith(".dll")]
    if len(dll_paths) < 2:
        raise ValidationError(
            f"variant {variant} app payload does not include the managed entrypoint and runtime/native DLLs"
        )
    return {
        "app_root": str(app_root),
        "payload_manifest": str(manifest_path.resolve()),
        "payload_sha256": _sha256(manifest_path),
        "payload_file_count": len(rows),
        "executable_sha256": str(executable_row["sha256"]),
        "managed_entrypoint": expected_actual["managed_entrypoint_relative_path"],
        "managed_entrypoint_sha256": str(managed_row["sha256"]),
    }


def _verify_paddle_bundle(payload: Any) -> dict[str, Any]:
    bundle = _require_mapping(payload, "fixed Paddle OCR bundle")
    root_raw = bundle.get("bundle_root")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValidationError("fixed Paddle OCR bundle has no bundle_root")
    root = Path(root_raw)
    if not root.is_dir():
        raise ValidationError(f"fixed Paddle OCR bundle root is missing: {root}")
    root = root.resolve()
    manifest_path = _verify_file_evidence(
        bundle.get("bundle_payload"), "fixed Paddle OCR bundle payload manifest"
    )
    rows = _load_json(manifest_path, "fixed Paddle OCR bundle payload manifest")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("fixed Paddle OCR bundle payload manifest is empty")

    listed: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"Paddle OCR bundle payload row {index + 1}")
        relative = _safe_relative_payload_path(
            row.get("path"), f"Paddle OCR bundle payload row {index + 1}"
        )
        key = relative.casefold()
        if key in listed:
            raise ValidationError(
                f"Paddle OCR bundle payload has duplicate path: {relative}"
            )
        sha = row.get("sha256")
        byte_count = row.get("bytes")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValidationError(
                f"Paddle OCR bundle payload has invalid SHA-256: {relative}"
            )
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValidationError(
                f"Paddle OCR bundle payload has invalid bytes: {relative}"
            )
        target = (root / Path(*relative.split("/"))).resolve()
        try:
            common = os.path.commonpath([_path_key(target), _path_key(root)])
        except ValueError as exception:
            raise ValidationError(
                f"Paddle OCR bundle payload escapes to another volume: {relative}"
            ) from exception
        if common != _path_key(root) or target == root or not target.is_file():
            raise ValidationError(
                f"Paddle OCR bundle payload path is missing/unsafe: {relative}"
            )
        if target.stat().st_size != byte_count or _sha256(target) != sha.casefold():
            raise ValidationError(
                f"Paddle OCR bundle payload changed during A/B execution: {relative}"
            )
        listed[key] = row

    actual: dict[str, str] = {}
    for target in root.rglob("*"):
        if target.is_symlink():
            raise ValidationError(
                f"Paddle OCR bundle payload contains a symbolic link: {target}"
            )
        if not target.is_file():
            continue
        relative = target.relative_to(root).as_posix()
        key = relative.casefold()
        if key in actual:
            raise ValidationError(
                f"Paddle OCR bundle payload has duplicate canonical path: {relative}"
            )
        actual[key] = relative
    missing = sorted(set(listed) - set(actual))
    extra = sorted(set(actual) - set(listed))
    if missing or extra:
        raise ValidationError(
            "Paddle OCR bundle payload manifest is not closed: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    contract_relative = _safe_relative_payload_path(
        bundle.get("contract_relative_path"),
        "fixed Paddle OCR bundle contract_relative_path",
    )
    if contract_relative != PADDLE_BUNDLE_CONTRACT:
        raise ValidationError(
            "fixed Paddle OCR bundle contract path changed: "
            f"expected {PADDLE_BUNDLE_CONTRACT}, found {contract_relative}"
        )
    contract_key = contract_relative.casefold()
    if contract_key not in listed:
        raise ValidationError("fixed Paddle OCR bundle payload omits its contract")
    contract_path = root / Path(*contract_relative.split("/"))
    contract = _require_mapping(
        _load_json(contract_path, "fixed Paddle OCR bundle contract"),
        "fixed Paddle OCR bundle contract",
    )
    if (
        type(contract.get("schema_version")) is not int
        or contract.get("schema_version") != 1
        or contract.get("kind") != "paddle_ocr_v2_delivery"
    ):
        raise ValidationError("fixed Paddle OCR bundle contract has the wrong schema/kind")
    models = _require_mapping(
        contract.get("models"), "fixed Paddle OCR bundle models"
    )
    if set(models) != {"det", "cls", "rec"}:
        raise ValidationError(
            "fixed Paddle OCR bundle contract must contain exactly det/cls/rec models"
        )
    package_payload_bytes = 0
    for role in ("det", "cls", "rec"):
        model = _require_mapping(
            models.get(role), f"fixed Paddle OCR bundle {role} model"
        )
        relative = _safe_relative_payload_path(
            model.get("path"), f"fixed Paddle OCR bundle {role} model path"
        )
        row = listed.get(relative.casefold())
        if row is None or not relative.casefold().endswith(".onnx"):
            raise ValidationError(
                f"fixed Paddle OCR bundle {role} model is not a frozen ONNX payload"
            )
        model_sha = model.get("sha256")
        model_bytes = model.get("size_bytes")
        if (
            not isinstance(model_sha, str)
            or model_sha.casefold() != str(row.get("sha256", "")).casefold()
            or isinstance(model_bytes, bool)
            or not isinstance(model_bytes, int)
            or model_bytes != row.get("bytes")
        ):
            raise ValidationError(
                f"fixed Paddle OCR bundle {role} contract evidence differs from its payload"
            )
        package_payload_bytes += model_bytes
    dictionary = _require_mapping(
        contract.get("dictionary"), "fixed Paddle OCR bundle dictionary"
    )
    dictionary_relative = _safe_relative_payload_path(
        dictionary.get("path"), "fixed Paddle OCR bundle dictionary path"
    )
    dictionary_row = listed.get(dictionary_relative.casefold())
    dictionary_sha = dictionary.get("sha256")
    dictionary_bytes = dictionary.get("size_bytes")
    if (
        dictionary_row is None
        or not isinstance(dictionary_sha, str)
        or dictionary_sha.casefold()
        != str(dictionary_row.get("sha256", "")).casefold()
        or isinstance(dictionary_bytes, bool)
        or not isinstance(dictionary_bytes, int)
        or dictionary_bytes != dictionary_row.get("bytes")
    ):
        raise ValidationError(
            "fixed Paddle OCR bundle dictionary contract evidence differs from its payload"
        )
    package_payload_bytes += dictionary_bytes
    package_size = contract.get("package_size_bytes")
    if (
        isinstance(package_size, bool)
        or not isinstance(package_size, int)
        or package_size != package_payload_bytes
    ):
        raise ValidationError(
            "fixed Paddle OCR bundle package_size_bytes differs from its models/dictionary"
        )
    return {
        "bundle_root": str(root),
        "payload_manifest": str(manifest_path.resolve()),
        "payload_sha256": _sha256(manifest_path),
        "payload_file_count": len(rows),
        "contract": contract_relative,
        "contract_sha256": _sha256(contract_path),
        "package_size_bytes": package_size,
    }


def _read_fixed_inputs(plan: Mapping[str, Any]) -> tuple[list[str], Mapping[str, Any]]:
    fixed_list = _verify_file_evidence(plan.get("fixed_input_list"), "fixed input list")
    evidence_path = _verify_file_evidence(plan.get("input_evidence"), "input evidence")
    inputs = [
        line.strip()
        for line in fixed_list.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_count = plan.get("input_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValidationError("plan input_count must be a positive integer")
    if len(inputs) != expected_count:
        raise ValidationError(
            f"fixed input list count differs from plan: list={len(inputs)} plan={expected_count}"
        )
    keys = [_path_key(item) for item in inputs]
    if len(set(keys)) != len(keys):
        raise ValidationError("fixed input list contains duplicate paths")

    rows = _load_json(evidence_path, "input evidence")
    if not isinstance(rows, list) or len(rows) != len(inputs):
        raise ValidationError("input evidence does not cover the fixed input list exactly")
    evidence_by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"input evidence row {index + 1}")
        source = row.get("source")
        sha = row.get("sha256")
        byte_count = row.get("bytes")
        if not isinstance(source, str) or not source:
            raise ValidationError(f"input evidence row {index + 1} has no source")
        key = _path_key(source)
        if key in evidence_by_key:
            raise ValidationError(f"input evidence contains duplicate source: {source}")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValidationError(f"input evidence has invalid SHA-256: {source}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValidationError(f"input evidence has invalid byte count: {source}")
        evidence_by_key[key] = row
    if set(evidence_by_key) != set(keys):
        raise ValidationError("input evidence sources differ from the fixed input list")

    # Re-hash after all benchmark processes have finished.  This makes the
    # repeated measurements evidence for one immutable set of image bytes.
    for source, key in zip(inputs, keys):
        path = Path(source)
        row = evidence_by_key[key]
        if not path.is_file():
            raise ValidationError(f"fixed input disappeared during A/B execution: {source}")
        if path.stat().st_size != row["bytes"] or _sha256(path) != str(row["sha256"]).casefold():
            raise ValidationError(f"fixed input changed during A/B execution: {source}")
    return inputs, {key: evidence_by_key[key] for key in keys}


def _validate_input_selection(
    plan: Mapping[str, Any], fixed_inputs: Sequence[str], schema_version: int = 1
) -> Mapping[str, Any]:
    selection = _require_mapping(plan.get("input_selection"), "input selection")
    if selection.get("rule") != "deduplicate_in_order_then_first_n":
        raise ValidationError("input selection rule changed")
    requested = selection.get("input_limit_requested")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValidationError("input_limit_requested must be a non-negative integer")
    if selection.get("selected_count") != len(fixed_inputs):
        raise ValidationError("input selection selected_count differs from fixed inputs")
    if schema_version >= 2:
        if "expected_input_count" not in selection:
            raise ValidationError("v2 input selection omits expected_input_count")
        expected_count = selection.get("expected_input_count")
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
            or expected_count != len(fixed_inputs)
        ):
            raise ValidationError(
                "input selection expected_input_count differs from fixed inputs"
            )
    source_list = _verify_file_evidence(
        selection.get("source_input_list"), "source input list"
    )
    source_directory = source_list.parent
    available: list[str] = []
    seen: set[str] = set()
    for raw_line in source_list.read_text(encoding="utf-8-sig").splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = source_directory / candidate
        full = os.path.abspath(os.fspath(candidate))
        key = _path_key(full)
        if key not in seen:
            seen.add(key)
            available.append(full)
    if selection.get("available_count") != len(available):
        raise ValidationError("input selection available_count differs from source input list")
    selected = available if requested == 0 else available[:requested]
    if [_path_key(item) for item in selected] != [_path_key(item) for item in fixed_inputs]:
        raise ValidationError("fixed inputs are not the declared first-N canonical selection")
    return selection


def _validate_cli_contract(
    plan: Mapping[str, Any], schema_version: int
) -> tuple[str, dict[str, int | None]]:
    contract = _require_mapping(plan.get("cli_contract"), "CLI contract")
    exact = {
        "device": "cpu",
        "unified_provider": "cpu",
        "score_threshold": 0.5,
        "rectification": "max-side-1600",
        "annotate": "none",
        "require_complete": True,
        "continue_on_error": False,
        "skip_existing": False,
        "includes_device_model": True,
    }
    for name, expected in exact.items():
        value = contract.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValidationError(
                f"CLI protection setting {name} changed: expected {expected!r}, found {value!r}"
            )
    ocr_mode = contract.get("ocr")
    allowed_modes = {"unified"} if schema_version == 1 else {"unified", "hybrid-recipient"}
    if type(ocr_mode) is not str or ocr_mode not in allowed_modes:
        raise ValidationError(
            f"CLI protection setting ocr changed: expected one of {sorted(allowed_modes)!r}, "
            f"found {ocr_mode!r}"
        )
    if schema_version >= 2:
        expected_paddle_provider = "cpu" if ocr_mode == "hybrid-recipient" else None
        if "paddle_ocr_provider" not in contract:
            raise ValidationError("CLI contract omits paddle_ocr_provider")
        paddle_provider = contract.get("paddle_ocr_provider")
        if type(paddle_provider) is not type(expected_paddle_provider) or paddle_provider != expected_paddle_provider:
            raise ValidationError(
                "CLI Paddle OCR provider contract changed: "
                f"expected {expected_paddle_provider!r}, found {paddle_provider!r}"
            )
        expected_bundle = ocr_mode == "hybrid-recipient"
        if (
            "includes_paddle_ocr_bundle" not in contract
            or type(contract.get("includes_paddle_ocr_bundle")) is not bool
            or contract.get("includes_paddle_ocr_bundle") is not expected_bundle
        ):
            raise ValidationError(
                "CLI Paddle OCR bundle contract is inconsistent with the OCR mode"
            )
    thread_contract = _require_mapping(
        contract.get("detector_intra_op_threads"),
        "detector intra-op thread contract",
    )
    if set(thread_contract) != set(VARIANTS):
        raise ValidationError(
            "detector intra-op thread contract must contain exactly baseline and candidate"
        )
    baseline_threads = thread_contract.get("baseline")
    candidate_threads = thread_contract.get("candidate")
    if baseline_threads is not None:
        raise ValidationError(
            "baseline detector intra-op threads must remain null/default"
        )
    if candidate_threads is not None and (
        isinstance(candidate_threads, bool)
        or not isinstance(candidate_threads, int)
        or not 1 <= candidate_threads <= MAXIMUM_DETECTOR_INTRA_OP_THREADS
    ):
        raise ValidationError(
            "candidate detector intra-op threads must be null/default or an integer in "
            f"[1, {MAXIMUM_DETECTOR_INTRA_OP_THREADS}]"
        )
    return ocr_mode, {"baseline": None, "candidate": candidate_threads}


def _validate_performance_gate(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = _require_mapping(plan.get("performance_gate"), "performance gate")
    exact = {
        "minimum_throughput_gain_percent": MINIMUM_THROUGHPUT_GAIN_PERCENT,
        "maximum_p50_regression_percent": MAXIMUM_P50_REGRESSION_PERCENT,
        "maximum_p95_regression_percent": MAXIMUM_P95_REGRESSION_PERCENT,
    }
    for name, expected in exact.items():
        value = gate.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise ValidationError(
                f"CPU performance gate {name} changed: expected {expected!r}, found {value!r}"
            )
    return gate


def _validate_plan(
    plan: Mapping[str, Any],
) -> tuple[int, int, int, int, str, dict[str, int | None]]:
    schema_version = plan.get("schema_version")
    kind = plan.get("kind")
    if type(schema_version) is not int or (schema_version, kind) not in (
        (1, LEGACY_PLAN_KIND),
        (2, PLAN_KIND),
    ):
        raise ValidationError("unsupported CPU A/B plan schema/kind")
    repetitions = plan.get("repetitions")
    warmup_runs = plan.get("warmup_runs")
    warmup_limit = plan.get("warmup_limit")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 3:
        raise ValidationError("CPU A/B requires at least three measured repetitions")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 1:
        raise ValidationError("CPU A/B requires at least one warmup run per variant")
    if isinstance(warmup_limit, bool) or not isinstance(warmup_limit, int) or warmup_limit < 1:
        raise ValidationError("CPU A/B warmup_limit must be positive")
    ocr_mode, detector_thread_contract = _validate_cli_contract(plan, schema_version)
    _validate_performance_gate(plan)
    return (
        schema_version,
        repetitions,
        warmup_runs,
        warmup_limit,
        ocr_mode,
        detector_thread_contract,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * (position - lower)


def _summarize(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "p50": round(_percentile(ordered, 0.50), 4),
        "p95": round(_percentile(ordered, 0.95), 4),
    }


def _assert_summary_matches(
    observed: Any,
    values: Sequence[float],
    description: str,
) -> None:
    payload = _require_mapping(observed, description)
    expected = _summarize(values)
    if payload.get("count") != expected["count"]:
        raise ValidationError(f"{description} count differs from its manifest")
    for name in ("mean", "p50", "p95"):
        actual = payload.get(name)
        wanted = expected[name]
        if wanted is None:
            if actual is not None:
                raise ValidationError(f"{description}.{name} must be null")
        else:
            number = _finite_number(actual, f"{description}.{name}")
            if abs(number - wanted) > 0.00011:
                raise ValidationError(
                    f"{description}.{name} differs from its manifest: "
                    f"summary={number} recomputed={wanted}"
                )


def _contained_result_path(raw_path: Any, output: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValidationError(f"manifest has no result path under {output}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = output / candidate
    candidate = candidate.resolve()
    output_resolved = output.resolve()
    try:
        common = os.path.commonpath([_path_key(candidate), _path_key(output_resolved)])
    except ValueError as exception:
        raise ValidationError(f"result path is on a different volume: {candidate}") from exception
    if common != _path_key(output_resolved) or candidate == output_resolved:
        raise ValidationError(f"manifest result escapes its unique output directory: {candidate}")
    if not candidate.is_file():
        raise ValidationError(f"manifest result is missing: {candidate}")
    return candidate


def _validate_geometry(result: Mapping[str, Any], description: str) -> None:
    geometry = _require_mapping(result.get("geometry"), f"{description} geometry")
    if geometry.get("rectification") != "max-side-1600":
        raise ValidationError(f"{description} changed the production rectification mode")
    source_size = _require_mapping(geometry.get("source_size"), f"{description} source_size")
    rectified_size = _require_mapping(
        geometry.get("rectified_size"), f"{description} rectified_size"
    )
    width = source_size.get("width")
    height = source_size.get("height")
    rectified_width = rectified_size.get("width")
    rectified_height = rectified_size.get("height")
    for value, label in (
        (width, "source width"),
        (height, "source height"),
        (rectified_width, "rectified width"),
        (rectified_height, "rectified height"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValidationError(f"{description} has invalid {label}")
    expected_rotation = 90 if width > height else 0
    expected_width, expected_height = (height, width) if expected_rotation == 90 else (width, height)
    longest = max(expected_width, expected_height)
    if longest > 1600:
        scale = 1600.0 / longest
        expected_width = max(2, round(expected_width * scale))
        expected_height = max(2, round(expected_height * scale))
    if type(geometry.get("screen_detected")) is not bool or geometry.get("screen_detected") is not False:
        raise ValidationError(f"{description} changed the screen-detection protection line")
    if geometry.get("rotation_degrees") != expected_rotation:
        raise ValidationError(f"{description} changed the portrait orientation rule")
    if (rectified_width, rectified_height) != (expected_width, expected_height):
        raise ValidationError(
            f"{description} has unexpected rectified size: "
            f"{rectified_width}x{rectified_height}, expected {expected_width}x{expected_height}"
        )


def _validate_result_contract(
    result: Mapping[str, Any],
    source: str,
    artifact_hashes: Mapping[str, str],
    ocr_mode: str,
    description: str,
) -> None:
    if _path_key(str(result.get("source", ""))) != _path_key(source):
        raise ValidationError(f"{description} source differs from its manifest")
    if result.get("inference_engine") != "mlnet":
        raise ValidationError(f"{description} did not use ML.NET")
    _validate_geometry(result, description)
    device = result.get("device")
    if not isinstance(device, Mapping) or not device:
        raise ValidationError(f"{description} skipped the device model result")
    fields = result.get("fields")
    if not isinstance(fields, Mapping) or not EXPECTED_FIELDS.issubset(fields):
        raise ValidationError(f"{description} omitted unified OCR field outputs")
    for field_name in EXPECTED_FIELDS:
        field = fields[field_name]
        if not isinstance(field, Mapping) or not isinstance(field.get("delivery_policy"), str):
            raise ValidationError(f"{description} omitted {field_name} delivery policy")
    detections = result.get("detections")
    if not isinstance(detections, list):
        raise ValidationError(f"{description} omitted detector outputs")
    # The persisted detector list contains only emitted boxes; the stable
    # five-field result shape above is the completeness contract. Keep this
    # structural check independent from the later exact cross-run comparison.
    labels: set[str] = set()
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise ValidationError(
                f"{description} detection {index + 1} must be one JSON object"
            )
        label = detection.get("label")
        if not isinstance(label, str) or label not in EXPECTED_DETECTIONS:
            raise ValidationError(
                f"{description} detection {index + 1} has an unknown label: {label!r}"
            )
        if label in labels:
            raise ValidationError(
                f"{description} contains duplicate detector label: {label}"
            )
        labels.add(label)
    contracts = _require_mapping(result.get("model_contracts"), f"{description} model contracts")
    for artifact_name, result_name in MODEL_HASH_FIELDS.items():
        if contracts.get(result_name) != artifact_hashes[artifact_name]:
            raise ValidationError(
                f"{description} {result_name} differs from the fixed A/B artifact"
            )
    if ocr_mode == "hybrid-recipient":
        if contracts.get(PADDLE_RESULT_HASH_FIELD) != artifact_hashes.get(
            "paddle_ocr_contract"
        ):
            raise ValidationError(
                f"{description} {PADDLE_RESULT_HASH_FIELD} differs from the frozen Paddle OCR bundle"
            )
        if contracts.get("ocr_bundle") != PADDLE_BUNDLE_CONTRACT:
            raise ValidationError(
                f"{description} did not bind the expected Paddle OCR bundle contract"
            )
    elif (
        "ocr_bundle" in contracts
        or PADDLE_RESULT_HASH_FIELD in contracts
    ):
        raise ValidationError(
            f"{description} unified-only result unexpectedly binds a Paddle OCR bundle"
        )


def _load_outer_wall_seconds(
    descriptor: Mapping[str, Any],
    output: Path,
    expected_count: int,
    schema_version: int,
) -> float | None:
    if schema_version == 1:
        return None
    run_id = str(descriptor["id"])
    raw_path = descriptor.get("wall_clock_evidence")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValidationError(f"run {run_id} has no outer wall-clock evidence path")
    path = Path(raw_path).resolve()
    if path.parent != output.resolve().parent:
        raise ValidationError(
            f"run {run_id} wall-clock evidence is not beside its fresh output directory"
        )
    evidence = _require_mapping(
        _load_json(path, f"run {run_id} outer wall-clock evidence"),
        f"run {run_id} outer wall-clock evidence",
    )
    exact = {
        "schema_version": 1,
        "kind": WALL_CLOCK_KIND,
        "run_id": run_id,
        "phase": descriptor.get("phase"),
        "variant": descriptor.get("variant"),
        "iteration": descriptor.get("iteration"),
        "expected_count": expected_count,
        "exit_code": 0,
    }
    for name, expected in exact.items():
        value = evidence.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValidationError(
                f"run {run_id} wall-clock evidence {name} differs from its frozen descriptor"
            )
    for name in ("started_utc", "finished_utc"):
        value = evidence.get(name)
        if not isinstance(value, str) or not value:
            raise ValidationError(
                f"run {run_id} wall-clock evidence has no {name} timestamp"
            )
    return _finite_number(
        evidence.get("elapsed_seconds"),
        f"run {run_id} outer wall elapsed_seconds",
        positive=True,
    )


def _recipient_route_counts(
    results: Mapping[str, Mapping[str, Any]], ocr_mode: str, description: str
) -> dict[str, dict[str, int]]:
    if ocr_mode != "hybrid-recipient":
        return {}
    counts: dict[str, dict[str, int]] = {
        field: defaultdict(int) for field in ROUTE_FIELDS
    }
    for result in results.values():
        fields = _require_mapping(result.get("fields"), f"{description} fields")
        recipient = _require_mapping(
            fields.get("recipient"), f"{description} recipient field"
        )
        primary = recipient.get("hybrid_ocr_route")
        if not isinstance(primary, str) or not primary:
            raise ValidationError(
                f"{description} hybrid recipient has no string hybrid_ocr_route"
            )
        counts["hybrid_ocr_route"][primary] += 1
        third = recipient.get("hybrid_ocr_third_route")
        if third is None:
            counts["hybrid_ocr_third_route"]["<missing>"] += 1
        elif isinstance(third, str) and third:
            counts["hybrid_ocr_third_route"][third] += 1
        else:
            raise ValidationError(
                f"{description} hybrid recipient has an invalid hybrid_ocr_third_route"
            )
    return {
        field: dict(sorted(field_counts.items()))
        for field, field_counts in counts.items()
    }


def _load_run(
    descriptor: Mapping[str, Any],
    expected_sources: Sequence[str],
    artifact_hashes: Mapping[str, str],
    expected_detector_intra_op_threads: int | None,
    ocr_mode: str,
    schema_version: int,
) -> dict[str, Any]:
    run_id = descriptor.get("id")
    output_raw = descriptor.get("output_directory")
    if not isinstance(run_id, str) or not run_id:
        raise ValidationError("run descriptor has no id")
    if not isinstance(output_raw, str) or not output_raw:
        raise ValidationError(f"run {run_id} has no output directory")
    output = Path(output_raw)
    if not output.is_dir():
        raise ValidationError(f"run {run_id} output is missing: {output}")
    summary = _require_mapping(
        _load_json(output / "inference_summary.json", f"run {run_id} summary"),
        f"run {run_id} summary",
    )
    manifest = _load_json(output / "inference_manifest.json", f"run {run_id} manifest")
    if not isinstance(manifest, list):
        raise ValidationError(f"run {run_id} manifest must be one JSON array")
    errors_path = output / "inference_errors.jsonl"
    if not errors_path.is_file() or errors_path.read_text(encoding="utf-8-sig").strip():
        raise ValidationError(f"run {run_id} contains inference errors")
    expected_count = len(expected_sources)
    expected_paddle_provider = "cpu" if ocr_mode == "hybrid-recipient" else None
    if (
        (schema_version >= 2 and "paddle_ocr_provider" not in summary)
        or summary.get("requested_device") != "cpu"
        or summary.get("unified_provider") != "cpu"
        or summary.get("paddle_ocr_provider") != expected_paddle_provider
        or summary.get("input") != expected_count
        or summary.get("written") != expected_count
        or summary.get("skipped") != 0
        or summary.get("errors") != 0
        or len(manifest) != expected_count
    ):
        raise ValidationError(f"run {run_id} accounting/provider evidence is inconsistent")
    actual_detector_threads = summary.get("detector_intra_op_threads")
    if expected_detector_intra_op_threads is None:
        if actual_detector_threads is not None:
            raise ValidationError(
                f"run {run_id} baseline unexpectedly set detector intra-op threads"
            )
    elif (
        isinstance(actual_detector_threads, bool)
        or not isinstance(actual_detector_threads, int)
        or actual_detector_threads != expected_detector_intra_op_threads
    ):
        raise ValidationError(
            f"run {run_id} detector intra-op thread summary differs from the frozen plan: "
            f"expected={expected_detector_intra_op_threads!r} "
            f"found={actual_detector_threads!r}"
        )

    expected_by_key = {_path_key(source): source for source in expected_sources}
    results: dict[str, Mapping[str, Any]] = {}
    inference_values: list[float] = []
    stage_values: dict[str, list[float]] = defaultdict(list)
    for index, raw_record in enumerate(manifest):
        record = _require_mapping(raw_record, f"run {run_id} manifest record {index + 1}")
        if record.get("status") != "written":
            raise ValidationError(f"run {run_id} contains a non-written manifest record")
        source = record.get("source")
        if not isinstance(source, str):
            raise ValidationError(f"run {run_id} manifest record has no source")
        key = _path_key(source)
        if key not in expected_by_key or key in results:
            raise ValidationError(f"run {run_id} has unexpected or duplicate source: {source}")
        result_path = _contained_result_path(record.get("result"), output)
        result = _require_mapping(
            _load_json(result_path, f"run {run_id} result"),
            f"run {run_id} result",
        )
        _validate_result_contract(
            result,
            expected_by_key[key],
            artifact_hashes,
            ocr_mode,
            f"run {run_id} result {result_path}",
        )
        results[key] = result
        inference_values.append(
            _finite_number(record.get("inference_ms"), f"run {run_id} inference_ms")
        )
        stages = _require_mapping(
            record.get("stage_latency_ms"), f"run {run_id} stage latency"
        )
        for stage in REQUIRED_STAGES:
            stage_values[stage].append(
                _finite_number(stages.get(stage), f"run {run_id} {stage} latency")
            )
        if ocr_mode == "hybrid-recipient":
            stage_values["paddle_ocr"].append(
                _finite_number(
                    stages.get("paddle_ocr"), f"run {run_id} paddle OCR latency"
                )
            )
        elif stages.get("paddle_ocr") is not None:
            raise ValidationError(
                f"run {run_id} unified-only record unexpectedly used Paddle OCR"
            )
    if set(results) != set(expected_by_key):
        raise ValidationError(f"run {run_id} omitted fixed input sources")

    _assert_summary_matches(
        summary.get("inference_latency_ms"), inference_values, f"run {run_id} inference summary"
    )
    summary_stages = _require_mapping(
        summary.get("stage_latency_ms"), f"run {run_id} stage summary"
    )
    for stage in ALL_STAGES:
        _assert_summary_matches(
            summary_stages.get(stage), stage_values.get(stage, []), f"run {run_id} {stage} summary"
        )
    total_seconds = _finite_number(
        summary.get("total_seconds"), f"run {run_id} total_seconds", positive=True
    )
    outer_wall_seconds = _load_outer_wall_seconds(
        descriptor, output, expected_count, schema_version
    )
    if (
        outer_wall_seconds is not None
        and outer_wall_seconds + 0.0001 < total_seconds
    ):
        raise ValidationError(
            f"run {run_id} outer wall time is shorter than CLI total_seconds"
        )
    throughput_seconds = (
        total_seconds if outer_wall_seconds is None else outer_wall_seconds
    )
    route_counts = _recipient_route_counts(results, ocr_mode, f"run {run_id}")
    return {
        "descriptor": dict(descriptor),
        "summary": dict(summary),
        "results": results,
        "inference_values": inference_values,
        "stage_values": dict(stage_values),
        "route_counts": route_counts,
        "metrics": {
            "input": expected_count,
            "total_seconds": total_seconds,
            "cli_total_seconds": total_seconds,
            "outer_wall_seconds": outer_wall_seconds,
            "throughput_seconds": throughput_seconds,
            "throughput_images_per_second": round(
                expected_count / throughput_seconds, 6
            ),
            "inference_latency_ms": _summarize(inference_values),
            "stage_latency_ms": {
                stage: _summarize(stage_values.get(stage, [])) for stage in ALL_STAGES
            },
        },
    }


def _prediction_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in RESULT_EXCLUDED_TOP_LEVEL_KEYS
    }


def _json_pointer(path: Sequence[str | int]) -> str:
    if not path:
        return "/"
    return "/" + "/".join(
        str(item).replace("~", "~0").replace("/", "~1") for item in path
    )


def _compact_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 500:
        return value[:497] + "..."
    return value


def _collect_differences(
    left: Any,
    right: Any,
    path: tuple[str | int, ...],
    emit: Any,
) -> None:
    if type(left) is not type(right):
        emit(path, "type", type(left).__name__, type(right).__name__)
        return
    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            emit(path + (key,), "missing_from_compared", _compact_value(left[key]), None)
        for key in sorted(right_keys - left_keys):
            emit(path + (key,), "extra_in_compared", None, _compact_value(right[key]))
        for key in sorted(left_keys & right_keys):
            _collect_differences(left[key], right[key], path + (key,), emit)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            emit(path, "list_length", len(left), len(right))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _collect_differences(left_item, right_item, path + (index,), emit)
        return
    if left != right:
        emit(path, "value", _compact_value(left), _compact_value(right))


def _aggregate_variant(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inference_values: list[float] = []
    stage_values: dict[str, list[float]] = defaultdict(list)
    cli_total_seconds: list[float] = []
    throughput_seconds: list[float] = []
    outer_wall_seconds: list[float] = []
    throughputs: list[float] = []
    total_images = 0
    run_metrics: list[Mapping[str, Any]] = []
    for run in runs:
        inference_values.extend(run["inference_values"])
        for stage, values in run["stage_values"].items():
            stage_values[stage].extend(values)
        metrics = run["metrics"]
        total_images += metrics["input"]
        cli_total_seconds.append(metrics["cli_total_seconds"])
        throughput_seconds.append(metrics["throughput_seconds"])
        if metrics["outer_wall_seconds"] is not None:
            outer_wall_seconds.append(metrics["outer_wall_seconds"])
        throughputs.append(metrics["throughput_images_per_second"])
        run_metrics.append(
            {
                "id": run["descriptor"]["id"],
                "iteration": run["descriptor"]["iteration"],
                **metrics,
            }
        )
    summed_throughput_seconds = sum(throughput_seconds)
    return {
        "repetitions": len(runs),
        "total_images": total_images,
        "sum_total_seconds": round(sum(cli_total_seconds), 4),
        "total_seconds_per_run": _summarize(cli_total_seconds),
        "sum_throughput_seconds": round(summed_throughput_seconds, 6),
        "cli_total_seconds_per_run": _summarize(cli_total_seconds),
        "outer_wall_seconds_per_run": _summarize(outer_wall_seconds),
        "throughput_images_per_second": {
            "aggregate": round(total_images / summed_throughput_seconds, 6),
            "per_run": _summarize(throughputs),
            "measurement": (
                "external_process_wall_clock"
                if len(outer_wall_seconds) == len(runs)
                else "legacy_inference_summary_total_seconds"
            ),
        },
        "inference_latency_ms": _summarize(inference_values),
        "stage_latency_ms": {
            stage: _summarize(stage_values.get(stage, [])) for stage in ALL_STAGES
        },
        "runs": run_metrics,
    }


def _metric_delta(baseline: Any, candidate: Any) -> dict[str, Any]:
    baseline_number = float(baseline)
    candidate_number = float(candidate)
    return {
        "baseline": baseline_number,
        "candidate": candidate_number,
        "absolute": round(candidate_number - baseline_number, 6),
        "percent": (
            None
            if baseline_number == 0
            else round((candidate_number - baseline_number) * 100.0 / baseline_number, 4)
        ),
    }


def _performance_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    latency = {
        name: _metric_delta(
            baseline["inference_latency_ms"][name], candidate["inference_latency_ms"][name]
        )
        for name in ("mean", "p50", "p95")
    }
    stages: dict[str, Any] = {}
    for stage in ALL_STAGES:
        stage_delta: dict[str, Any] = {}
        for name in ("mean", "p50", "p95"):
            baseline_value = baseline["stage_latency_ms"][stage][name]
            candidate_value = candidate["stage_latency_ms"][stage][name]
            if baseline_value is not None and candidate_value is not None:
                stage_delta[name] = _metric_delta(baseline_value, candidate_value)
        stages[stage] = stage_delta
    return {
        "throughput_images_per_second": _metric_delta(
            baseline["throughput_images_per_second"]["aggregate"],
            candidate["throughput_images_per_second"]["aggregate"],
        ),
        "inference_latency_ms": latency,
        "stage_latency_ms": stages,
        "interpretation": (
            "Throughput uses pooled external process wall time for v2 evidence. "
            "Positive throughput percent is faster; negative latency percent is faster. "
            "Performance is reported, not used to weaken prediction or accuracy gates."
        ),
    }


def _evaluate_performance_gate(
    gate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    throughput_gain = float(delta["throughput_images_per_second"]["percent"])
    p50_regression = float(delta["inference_latency_ms"]["p50"]["percent"])
    p95_regression = float(delta["inference_latency_ms"]["p95"]["percent"])
    baseline_throughput = float(
        baseline["throughput_images_per_second"]["aggregate"]
    )
    candidate_throughput = float(
        candidate["throughput_images_per_second"]["aggregate"]
    )
    baseline_p50 = float(baseline["inference_latency_ms"]["p50"])
    candidate_p50 = float(candidate["inference_latency_ms"]["p50"])
    baseline_p95 = float(baseline["inference_latency_ms"]["p95"])
    candidate_p95 = float(candidate["inference_latency_ms"]["p95"])
    failures: list[str] = []
    if candidate_throughput < baseline_throughput * (
        1.0 + float(gate["minimum_throughput_gain_percent"]) / 100.0
    ):
        failures.append(
            "aggregate throughput gain is below the required minimum"
        )
    if candidate_p50 > baseline_p50 * (
        1.0 + float(gate["maximum_p50_regression_percent"]) / 100.0
    ):
        failures.append("inference p50 regressed")
    if candidate_p95 > baseline_p95 * (
        1.0 + float(gate["maximum_p95_regression_percent"]) / 100.0
    ):
        failures.append("inference p95 regressed")
    return {
        "accepted": not failures,
        "requirements": dict(gate),
        "observed": {
            "throughput_gain_percent": throughput_gain,
            "p50_regression_percent": p50_regression,
            "p95_regression_percent": p95_regression,
        },
        "failures": failures,
    }


def analyze_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _require_mapping(_load_json(plan_path, "CPU A/B plan"), "CPU A/B plan")
    (
        schema_version,
        repetitions,
        warmup_runs,
        warmup_limit,
        ocr_mode,
        detector_thread_contract,
    ) = _validate_plan(plan)
    fixed_inputs, _ = _read_fixed_inputs(plan)
    input_selection = _validate_input_selection(plan, fixed_inputs, schema_version)
    if warmup_limit > len(fixed_inputs):
        raise ValidationError("warmup_limit exceeds the fixed input count")

    artifacts = _require_mapping(plan.get("artifacts"), "fixed artifacts")
    artifact_hashes: dict[str, str] = {}
    for name in MODEL_HASH_FIELDS:
        path = _verify_file_evidence(artifacts.get(name), f"fixed artifact {name}")
        artifact_hashes[name] = _sha256(path)
    paddle_bundle_identity: dict[str, Any] | None = None
    if ocr_mode == "hybrid-recipient":
        paddle_bundle_identity = _verify_paddle_bundle(
            artifacts.get("paddle_ocr_bundle")
        )
        artifact_hashes["paddle_ocr_contract"] = paddle_bundle_identity[
            "contract_sha256"
        ]
    elif "paddle_ocr_bundle" in artifacts:
        raise ValidationError(
            "unified-only CPU A/B plan unexpectedly freezes a Paddle OCR bundle"
        )
    variants = _require_mapping(plan.get("variants"), "A/B variants")
    variant_identities: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_payload = _require_mapping(variants.get(variant), f"variant {variant}")
        variant_identities[variant] = _verify_app_payload(variant, variant_payload)
    if (
        variant_identities["baseline"]["payload_sha256"]
        == variant_identities["candidate"]["payload_sha256"]
    ):
        raise ValidationError(
            "baseline and candidate app payloads are byte-identical; refusing a meaningless A/B"
        )
    if (
        variant_identities["baseline"]["managed_entrypoint_sha256"]
        == variant_identities["candidate"]["managed_entrypoint_sha256"]
    ):
        raise ValidationError(
            "baseline and candidate managed entrypoints are byte-identical; refusing a meaningless A/B"
        )

    raw_runs = plan.get("runs")
    if not isinstance(raw_runs, list):
        raise ValidationError("plan runs must be one JSON array")
    expected_descriptors = {
        (phase, variant, iteration)
        for phase, count in (("warmup", warmup_runs), ("measured", repetitions))
        for variant in VARIANTS
        for iteration in range(1, count + 1)
    }
    descriptors: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    output_keys: set[str] = set()
    wall_clock_keys: set[str] = set()
    order_values: set[int] = set()
    for raw_descriptor in raw_runs:
        descriptor = _require_mapping(raw_descriptor, "run descriptor")
        phase = descriptor.get("phase")
        variant = descriptor.get("variant")
        iteration = descriptor.get("iteration")
        order = descriptor.get("execution_order")
        output = descriptor.get("output_directory")
        wall_clock = descriptor.get("wall_clock_evidence")
        if (
            phase not in ("warmup", "measured")
            or variant not in VARIANTS
            or isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or isinstance(order, bool)
            or not isinstance(order, int)
            or not isinstance(output, str)
            or (
                schema_version >= 2
                and (not isinstance(wall_clock, str) or not wall_clock)
            )
        ):
            raise ValidationError("plan contains a malformed run descriptor")
        key = (phase, variant, iteration)
        output_key = _path_key(output)
        wall_clock_key = _path_key(wall_clock) if schema_version >= 2 else None
        if (
            key in descriptors
            or output_key in output_keys
            or order in order_values
            or (wall_clock_key is not None and wall_clock_key in wall_clock_keys)
        ):
            raise ValidationError("plan contains duplicate run identity/output/order")
        descriptors[key] = descriptor
        output_keys.add(output_key)
        if wall_clock_key is not None:
            wall_clock_keys.add(wall_clock_key)
        order_values.add(order)
    if set(descriptors) != expected_descriptors:
        raise ValidationError("plan does not contain every required warmup/measured A/B run")
    if order_values != set(range(1, len(raw_runs) + 1)):
        raise ValidationError("run execution_order must be contiguous and unique")
    expected_execution_sequence: list[tuple[str, str, int]] = []
    for phase, count in (("warmup", warmup_runs), ("measured", repetitions)):
        for iteration in range(1, count + 1):
            variants_in_order = (
                VARIANTS if iteration % 2 == 1 else tuple(reversed(VARIANTS))
            )
            expected_execution_sequence.extend(
                (phase, variant, iteration) for variant in variants_in_order
            )
    actual_execution_sequence = [
        key
        for key, _ in sorted(
            descriptors.items(), key=lambda item: item[1]["execution_order"]
        )
    ]
    if actual_execution_sequence != expected_execution_sequence:
        raise ValidationError(
            "run execution_order does not preserve the frozen alternating AB/BA schedule"
        )

    loaded_runs: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for key, descriptor in descriptors.items():
        expected_sources = fixed_inputs[:warmup_limit] if key[0] == "warmup" else fixed_inputs
        if descriptor.get("expected_count") != len(expected_sources):
            raise ValidationError(f"run {descriptor.get('id')} expected_count changed")
        expected_detector_threads = detector_thread_contract[key[1]]
        if (
            "detector_intra_op_threads" not in descriptor
            or type(descriptor.get("detector_intra_op_threads"))
            is not type(expected_detector_threads)
            or descriptor.get("detector_intra_op_threads") != expected_detector_threads
        ):
            raise ValidationError(
                f"run {descriptor.get('id')} detector intra-op threads differ from the frozen CLI contract"
            )
        loaded_runs[key] = _load_run(
            descriptor,
            expected_sources,
            artifact_hashes,
            expected_detector_threads,
            ocr_mode,
            schema_version,
        )

    reference_key = ("measured", "baseline", 1)
    reference = loaded_runs[reference_key]
    differences: list[dict[str, Any]] = []
    difference_count = 0

    for key in sorted(loaded_runs, key=lambda item: loaded_runs[item]["descriptor"]["execution_order"]):
        if key == reference_key:
            continue
        compared = loaded_runs[key]
        for source_key, compared_result in compared["results"].items():
            reference_result = reference["results"][source_key]

            def emit(
                path: Sequence[str | int],
                reason: str,
                reference_value: Any,
                compared_value: Any,
            ) -> None:
                nonlocal difference_count
                difference_count += 1
                if len(differences) < MAX_REPORTED_DIFFERENCES:
                    differences.append(
                        {
                            "source": reference_result["source"],
                            "reference_run": reference["descriptor"]["id"],
                            "compared_run": compared["descriptor"]["id"],
                            "json_pointer": _json_pointer(path),
                            "reason": reason,
                            "reference": reference_value,
                            "compared": compared_value,
                        }
                    )

            _collect_differences(
                _prediction_payload(reference_result),
                _prediction_payload(compared_result),
                (),
                emit,
            )

    route_count_failures: list[str] = []
    route_counts_by_run: dict[str, Any] = {}
    if ocr_mode == "hybrid-recipient":
        for phase, count in (("warmup", warmup_runs), ("measured", repetitions)):
            phase_reference = loaded_runs[(phase, "baseline", 1)]["route_counts"]
            for iteration in range(1, count + 1):
                for variant in VARIANTS:
                    run = loaded_runs[(phase, variant, iteration)]
                    run_id = run["descriptor"]["id"]
                    route_counts_by_run[run_id] = run["route_counts"]
                    if run["route_counts"] != phase_reference:
                        route_count_failures.append(
                            f"{run_id} route counts differ from {phase}-01-baseline"
                        )

    measured_by_variant = {
        variant: [loaded_runs[("measured", variant, index)] for index in range(1, repetitions + 1)]
        for variant in VARIANTS
    }
    aggregate = {
        variant: _aggregate_variant(measured_by_variant[variant]) for variant in VARIANTS
    }
    route_counts_accepted = not route_count_failures
    prediction_accepted = difference_count == 0 and route_counts_accepted
    performance_delta = _performance_delta(
        aggregate["baseline"], aggregate["candidate"]
    )
    performance_acceptance = _evaluate_performance_gate(
        _require_mapping(plan["performance_gate"], "performance gate"),
        aggregate["baseline"],
        aggregate["candidate"],
        performance_delta,
    )
    accepted = prediction_accepted and performance_acceptance["accepted"]
    report = {
        "schema_version": 2,
        "kind": REPORT_KIND,
        "created_utc": _utc_now(),
        "accepted": accepted,
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "input_count": len(fixed_inputs),
        "input_selection": dict(input_selection),
        "warmup_runs_per_variant": warmup_runs,
        "warmup_images_per_run": warmup_limit,
        "measured_repetitions_per_variant": repetitions,
        "cli_contract": dict(plan["cli_contract"]),
        "performance_gate": dict(plan["performance_gate"]),
        "fixed_paddle_ocr_bundle": paddle_bundle_identity,
        "variant_identities": variant_identities,
        "prediction_consistency": {
            "accepted": prediction_accepted,
            "reference_run": reference["descriptor"]["id"],
            "compared_runs": len(loaded_runs) - 1,
            "difference_count": difference_count,
            "reported_difference_count": len(differences),
            "excluded_result_metadata": sorted(RESULT_EXCLUDED_TOP_LEVEL_KEYS),
            "comparison": (
                "type-sensitive deep JSON comparison; arrays are order-sensitive; "
                "all measured repeats and warmups compare to baseline measured repeat 1"
            ),
        },
        "route_consistency": {
            "applicable": ocr_mode == "hybrid-recipient",
            "accepted": route_counts_accepted,
            "fields": list(ROUTE_FIELDS) if ocr_mode == "hybrid-recipient" else [],
            "by_run": route_counts_by_run,
            "failures": route_count_failures,
        },
        "performance": {
            "accepted": performance_acceptance["accepted"],
            "gate": performance_acceptance,
            "baseline": aggregate["baseline"],
            "candidate": aggregate["candidate"],
            "candidate_vs_baseline": performance_delta,
        },
    }
    return report, differences


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--differences", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report.exists() or args.differences.exists():
        print("refusing to overwrite an existing A/B report or differences file", file=sys.stderr)
        return 2
    try:
        report, differences = analyze_plan(args.plan)
    except Exception as exception:
        failure = {
            "schema_version": 1,
            "kind": REPORT_KIND,
            "created_utc": _utc_now(),
            "accepted": False,
            "plan": str(args.plan.resolve()),
            "validation_error": f"{type(exception).__name__}: {exception}",
        }
        _write_json_atomic(args.report, failure)
        _write_jsonl_atomic(args.differences, [])
        print(f"CPU A/B validation failed: {exception}", file=sys.stderr)
        return 1
    _write_json_atomic(args.report, report)
    _write_jsonl_atomic(args.differences, differences)
    baseline = report["performance"]["baseline"]
    candidate = report["performance"]["candidate"]
    print(
        "baseline: "
        f"mean={baseline['inference_latency_ms']['mean']:.4f} ms, "
        f"p50={baseline['inference_latency_ms']['p50']:.4f} ms, "
        f"p95={baseline['inference_latency_ms']['p95']:.4f} ms, "
        f"throughput={baseline['throughput_images_per_second']['aggregate']:.6f} img/s"
    )
    print(
        "candidate: "
        f"mean={candidate['inference_latency_ms']['mean']:.4f} ms, "
        f"p50={candidate['inference_latency_ms']['p50']:.4f} ms, "
        f"p95={candidate['inference_latency_ms']['p95']:.4f} ms, "
        f"throughput={candidate['throughput_images_per_second']['aggregate']:.6f} img/s"
    )
    if not report["prediction_consistency"]["accepted"]:
        print(
            f"prediction consistency failed: {report['prediction_consistency']['difference_count']} difference(s); "
            f"see {args.differences}",
            file=sys.stderr,
        )
    if not report["performance"]["accepted"]:
        failures = "; ".join(report["performance"]["gate"]["failures"])
        print(f"CPU performance gate failed: {failures}", file=sys.stderr)
    if not report["accepted"]:
        return 1
    print("PASS: CPU predictions are exactly consistent and the candidate is measurably faster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
