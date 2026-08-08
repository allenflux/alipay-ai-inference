#!/usr/bin/env python3
"""Analyze frozen CPU Paddle layout records without producing field candidates.

The analyzer consumes the formal missing-field audit, the frozen time-339
selection, and the completed LayoutShadow output.  It publishes only raw
anchor, geometry, confidence, ambiguity, and coverage evidence.  It never
writes receipt fields and cannot act as a formal delivery gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import unicodedata
from typing import Any
from uuid import uuid4


EXPECTED_RECORDS = 339
FORMAL_RECORDS = 10016
SELECTION_KIND = "receipt_mlnet_layout_shadow_time_selection_v1"
AUDIT_SUMMARY_KIND = "receipt_mlnet_formal_missing_fields_audit_summary_v1"
AUDIT_FINDING_KIND = "receipt_mlnet_formal_missing_fields_audit_finding_v1"
LAYOUT_SUMMARY_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_summary_v1"
LAYOUT_RECORD_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_record_v1"
EVIDENCE_SUMMARY_KIND = "receipt_mlnet_layout_shadow_field_evidence_summary_v1"
EVIDENCE_RECORD_KIND = "receipt_mlnet_layout_shadow_field_evidence_record_v1"
QUAD_COORDINATE_SPACE = "max_side_1600_rectified_tl_tr_br_bl"
QUAD_NORMALIZATION = "x/(rectified_width-1),y/(rectified_height-1)"
CONFIDENCE_SEMANTICS = "ctc_emitted_character_mean"
RECTIFICATION = "max-side-1600"
STATUS_BAR_FRACTION = 0.08

FIELD_SPECS = {
    "time": (339, 339, 0),
    "payment_method_field": (1, 1, 0),
    "transfer_status": (1, 1, 0),
}
PAYMENT_LABELS = ("付款方式", "交易方式", "付款渠道", "支付方式")
PAYMENT_FIXED_VALUES = {"余额": "balance", "余额宝": "yuebao", "花呗": "huabei"}
PAYMENT_CARD_PATTERN = re.compile(
    r"^(?P<prefix>[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z0-9·]{2,48}"
    r"(?:银行卡|储蓄卡|信用卡)) ?(?P<open>\(|（)(?P<tail>[0-9]{4})(?P<close>\)|）)$"
)
CLOCK_PATTERN = re.compile(r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})$")
TIME_LIKE_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,2})[:：][0-9]{2}(?::[0-9]{2})?(?![0-9])")
STATUS_SUBJECTS = ("转账", "交易", "付款", "支付", "转帐")
STATUS_PHRASES_BY_CLASS = {
    "success": tuple(f"{subject}成功" for subject in STATUS_SUBJECTS),
    "failed": tuple(
        f"{subject}{suffix}"
        for suffix in ("失败", "未成功", "已撤销")
        for subject in STATUS_SUBJECTS
    ),
    "pending": tuple(
        f"{subject}{suffix}"
        for suffix in ("处理中", "待处理", "进行中")
        for subject in STATUS_SUBJECTS
    ),
}
STATUS_PHRASE_CLASS = {
    phrase: status_class
    for status_class, phrases in STATUS_PHRASES_BY_CLASS.items()
    for phrase in phrases
}
STATUS_SUCCESS_BLOCKERS = (
    "未", "不", "非", "无", "否", "没", "没有", "未能", "不是", "并未",
    "尚未", "不能", "无法", "没能", "未曾", "从未", "并非", "吗", "么",
    "待确认", "待核实", "未知", "不确定", "疑似",
)
WHITESPACE = re.compile(r"\s+")
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(ValueError):
    """Raised when an input closure or diagnostic schema is unsafe to use."""


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
        raise EvidenceError(f"invalid JSON at {location}: {error}") from error


def _read_bytes(path: Path, *, description: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvidenceError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise EvidenceError(f"{description} is not a regular file: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read {description}: {resolved}: {error}") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path, payload: bytes | None = None) -> dict[str, Any]:
    data = _read_bytes(path, description="bound input") if payload is None else payload
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _load_json(path: Path, payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{description} is not UTF-8: {path}") from error
    value = _loads(text, location=str(path))
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be one JSON object")
    return value


def _load_jsonl(path: Path, payload: bytes, *, description: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{description} is not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise EvidenceError(f"{description} has a blank line at {path}:{line_number}")
        value = _loads(line, location=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise EvidenceError(f"{description} row must be an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise EvidenceError(f"{description} is empty: {path}")
    return rows


def _require_directory(path: Path, *, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvidenceError(f"{description} does not exist: {path}") from error
    if not resolved.is_dir():
        raise EvidenceError(f"{description} is not a directory: {resolved}")
    return resolved


def _path_key(path: Path | str) -> str:
    value = Path(path).resolve(strict=True)
    return os.path.normcase(os.path.normpath(str(value)))


def _require_bool(value: object, expected: bool, *, description: str) -> None:
    if value is not expected:
        raise EvidenceError(f"{description} must be {str(expected).lower()}")


def _require_int(value: object, expected: int | None = None, *, description: str) -> int:
    if type(value) is not int:
        raise EvidenceError(f"{description} must be an integer")
    if expected is not None and value != expected:
        raise EvidenceError(f"{description} must be {expected}, found {value}")
    return value


def _require_number(value: object, *, description: str, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{description} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{description} is below {minimum}")
    if maximum is not None and result > maximum:
        raise EvidenceError(f"{description} is above {maximum}")
    return result


def _require_sha(value: object, *, description: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise EvidenceError(f"{description} must be a lowercase SHA-256")
    return value


def _same_identity(contract: object, observed: Mapping[str, Any], *, description: str,
                   require_path: bool = True) -> None:
    if not isinstance(contract, Mapping):
        raise EvidenceError(f"{description} identity is missing")
    if require_path:
        path = contract.get("path")
        if not isinstance(path, str) or _path_key(path) != _path_key(str(observed["path"])):
            raise EvidenceError(f"{description} path differs")
    if _require_sha(contract.get("sha256"), description=f"{description} sha256") != observed["sha256"]:
        raise EvidenceError(f"{description} SHA-256 differs")
    if _require_int(contract.get("size_bytes"), description=f"{description} size") != observed["size_bytes"]:
        raise EvidenceError(f"{description} size differs")


def _parse_input_list(path: Path, payload: bytes) -> list[Path]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise EvidenceError("selection inputs.txt must be UTF-8 without BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError("selection inputs.txt is not strict UTF-8") from error
    if not text.endswith("\n"):
        raise EvidenceError("selection inputs.txt must end with one newline")
    raw_lines = text.splitlines()
    if len(raw_lines) != EXPECTED_RECORDS:
        raise EvidenceError(f"selection inputs.txt must contain {EXPECTED_RECORDS} records")
    sources: list[Path] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_lines):
        if not raw or raw != raw.strip():
            raise EvidenceError(f"selection input[{index}] is blank or has surrounding whitespace")
        source = Path(raw)
        if not source.is_absolute():
            raise EvidenceError(f"selection input[{index}] is not absolute: {raw}")
        try:
            source = source.resolve(strict=True)
        except FileNotFoundError as error:
            raise EvidenceError(f"selection input[{index}] does not exist: {raw}") from error
        if not source.is_file():
            raise EvidenceError(f"selection input[{index}] is not a file: {source}")
        key = _path_key(source)
        if key in seen:
            raise EvidenceError(f"selection inputs contain duplicate source: {source}")
        seen.add(key)
        sources.append(source)
    return sources


def _validate_selection(selection_directory: Path, audit_directory: Path) -> tuple[
    dict[str, Any], list[Path], list[dict[str, Any]], dict[str, Any]
]:
    selection_path = selection_directory / "selection.json"
    input_path = selection_directory / "inputs.txt"
    selection_bytes = _read_bytes(selection_path, description="layout selection contract")
    input_bytes = _read_bytes(input_path, description="layout selection input list")
    selection = _load_json(selection_path, selection_bytes, description="layout selection contract")
    if selection.get("schema_version") != 1 or selection.get("kind") != SELECTION_KIND:
        raise EvidenceError("layout selection schema/kind is unsupported")
    _require_bool(selection.get("diagnostic_only"), True, description="selection diagnostic_only")
    _require_bool(selection.get("formal_delivery_gate"), False, description="selection formal_delivery_gate")
    _require_int(selection.get("records"), EXPECTED_RECORDS, description="selection records")
    _require_int(selection.get("external_reference_present_records"), 0,
                 description="selection external-reference-present records")
    _require_int(selection.get("external_reference_missing_records"), EXPECTED_RECORDS,
                 description="selection external-reference-missing records")
    if selection.get("selection_field") != "time" or selection.get("selection_order") \
            != "formal_audit_missing_by_field_time_sources_order":
        raise EvidenceError("layout selection field/order is unsupported")
    sources = _parse_input_list(input_path, input_bytes)
    input_contract = selection.get("input_list")
    if not isinstance(input_contract, Mapping):
        raise EvidenceError("selection input_list contract is missing")
    if input_contract.get("relative_path") != "inputs.txt":
        raise EvidenceError("selection input_list relative_path is unsupported")
    if _require_sha(input_contract.get("sha256"), description="selection input-list sha256") != _sha256(input_bytes):
        raise EvidenceError("selection input-list SHA-256 differs")
    _require_int(input_contract.get("size_bytes"), len(input_bytes), description="selection input-list size")
    _require_int(input_contract.get("records"), EXPECTED_RECORDS, description="selection input-list records")
    if input_contract.get("encoding") != "utf-8-no-bom" or input_contract.get("terminal_newline") is not True:
        raise EvidenceError("selection input-list encoding contract is unsupported")

    source_contracts = selection.get("source_files")
    if not isinstance(source_contracts, list) or len(source_contracts) != EXPECTED_RECORDS:
        raise EvidenceError("selection source_files must contain exactly 339 identities")
    identities: list[dict[str, Any]] = []
    for index, (source, contract) in enumerate(zip(sources, source_contracts, strict=True)):
        payload = _read_bytes(source, description=f"selection source[{index}]")
        observed = _identity(source, payload)
        _same_identity(contract, observed, description=f"selection source[{index}]")
        identities.append(observed)
    if _require_int(selection.get("source_total_bytes"), description="selection source total bytes") \
            != sum(int(item["size_bytes"]) for item in identities):
        raise EvidenceError("selection source_total_bytes differs")
    selection_closure = hashlib.sha256()
    for item in identities:
        selection_closure.update(
            (
                f"{_path_key(str(item['path']))}\0{item['path']}\0{item['sha256']}\0"
                f"{item['size_bytes']}\n"
            ).encode("utf-8")
        )
    if _require_sha(selection.get("source_closure_sha256"), description="selection source closure") \
            != selection_closure.hexdigest():
        raise EvidenceError("selection source_closure_sha256 differs")
    audit_contract = selection.get("formal_audit")
    if not isinstance(audit_contract, Mapping):
        raise EvidenceError("selection formal_audit contract is missing")
    audit_path = audit_contract.get("directory")
    if not isinstance(audit_path, str) or _path_key(audit_path) != _path_key(audit_directory):
        raise EvidenceError("selection formal_audit directory differs from --audit-directory")
    _require_int(audit_contract.get("records"), FORMAL_RECORDS, description="selection formal audit records")
    bindings = {
        "selection": _identity(selection_path, selection_bytes),
        "input_list": _identity(input_path, input_bytes),
        "sources": identities,
    }
    return selection, sources, identities, bindings


def _validate_audit(
    audit_directory: Path,
    selection: Mapping[str, Any],
    sources: Sequence[Path],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    summary_path = audit_directory / "summary.json"
    findings_path = audit_directory / "findings.jsonl"
    summary_bytes = _read_bytes(summary_path, description="formal missing-fields summary")
    findings_bytes = _read_bytes(findings_path, description="formal missing-fields findings")
    summary_identity = _identity(summary_path, summary_bytes)
    findings_identity = _identity(findings_path, findings_bytes)
    formal_contract = selection.get("formal_audit")
    assert isinstance(formal_contract, Mapping)
    _same_identity(formal_contract.get("summary"), summary_identity, description="formal audit summary")
    _same_identity(formal_contract.get("findings"), findings_identity, description="formal audit findings")
    summary = _load_json(summary_path, summary_bytes, description="formal missing-fields summary")
    findings = _load_jsonl(findings_path, findings_bytes, description="formal missing-fields findings")
    if summary.get("schema_version") != 1 or summary.get("kind") != AUDIT_SUMMARY_KIND:
        raise EvidenceError("formal missing-fields summary schema/kind is unsupported")
    _require_bool(summary.get("read_only_existing_results"), True, description="formal audit read_only_existing_results")
    _require_bool(summary.get("ocr_rerun"), False, description="formal audit ocr_rerun")
    _require_bool(summary.get("formal_required"), True, description="formal audit formal_required")
    _require_int(summary.get("records"), FORMAL_RECORDS, description="formal audit records")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("summary") != "summary.json" or artifacts.get("findings") != "findings.jsonl":
        raise EvidenceError("formal audit artifact names are unsupported")

    selected_keys = {_path_key(source) for source in sources}
    field_sets: dict[str, set[str]] = {}
    missing = summary.get("missing_by_field")
    if not isinstance(missing, Mapping):
        raise EvidenceError("formal audit missing_by_field is absent")
    for field, (records, reference_missing, reference_present) in FIELD_SPECS.items():
        contract = missing.get(field)
        if not isinstance(contract, Mapping):
            raise EvidenceError(f"formal audit field {field} is missing")
        _require_int(contract.get("records"), records, description=f"formal audit {field} records")
        _require_int(contract.get("reference_missing_records"), reference_missing,
                     description=f"formal audit {field} reference-missing records")
        _require_int(contract.get("reference_present_records"), reference_present,
                     description=f"formal audit {field} reference-present records")
        raw_sources = contract.get("sources")
        if not isinstance(raw_sources, list) or len(raw_sources) != records:
            raise EvidenceError(f"formal audit {field} sources count differs")
        keys: set[str] = set()
        for index, raw in enumerate(raw_sources):
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise EvidenceError(f"formal audit {field} source[{index}] is not absolute")
            key = _path_key(raw)
            if key in keys:
                raise EvidenceError(f"formal audit {field} has duplicate source: {raw}")
            keys.add(key)
        if not keys.issubset(selected_keys):
            raise EvidenceError(f"formal audit {field} sources are outside the frozen time-339 selection")
        field_sets[field] = keys
    if field_sets["time"] != selected_keys:
        raise EvidenceError("formal audit time source set differs from the frozen selection")

    finding_sets = {field: set() for field in FIELD_SPECS}
    seen_findings: set[str] = set()
    for index, row in enumerate(findings):
        if row.get("schema_version") != 1 or row.get("kind") != AUDIT_FINDING_KIND:
            raise EvidenceError(f"formal audit finding[{index}] schema/kind is unsupported")
        source_raw = row.get("source")
        if not isinstance(source_raw, str) or not Path(source_raw).is_absolute():
            raise EvidenceError(f"formal audit finding[{index}] source is not absolute")
        key = _path_key(source_raw)
        if key in seen_findings:
            raise EvidenceError(f"formal audit has duplicate finding source: {source_raw}")
        seen_findings.add(key)
        missing_fields = row.get("missing_fields")
        if not isinstance(missing_fields, list) or not all(isinstance(value, str) for value in missing_fields):
            raise EvidenceError(f"formal audit finding[{index}] missing_fields is invalid")
        reference_map = row.get("reference_present_by_field")
        by_field = row.get("by_missing_field")
        if not isinstance(reference_map, Mapping) or not isinstance(by_field, Mapping):
            raise EvidenceError(f"formal audit finding[{index}] field evidence is missing")
        for field in FIELD_SPECS:
            if field not in missing_fields:
                continue
            if key not in selected_keys:
                raise EvidenceError(f"formal audit {field} finding is outside the frozen selection")
            if reference_map.get(field) is not False:
                raise EvidenceError(f"formal audit {field} finding unexpectedly has a reference")
            detail = by_field.get(field)
            if not isinstance(detail, Mapping) or detail.get("reference_present") is not False \
                    or detail.get("reference_text") is not None or detail.get("score_comparison") is not None:
                raise EvidenceError(f"formal audit {field} reference evidence disagrees")
            finding_sets[field].add(key)
    for field, expected in field_sets.items():
        if finding_sets[field] != expected:
            raise EvidenceError(f"formal audit {field} summary/findings source sets differ")
    return field_sets, {
        "audit_summary": summary_identity,
        "audit_findings": findings_identity,
    }


def _clean_text(value: str) -> str:
    # Match ReceiptFieldNormalizer.CleanText exactly for accepted_text binding.
    # Field-specific diagnostic parsers may normalize NFC independently.
    return WHITESPACE.sub(" ", value).strip()


def _rect(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    return {
        "x_min": round(x_min, 6),
        "y_min": round(y_min, 6),
        "x_max": round(x_max, 6),
        "y_max": round(y_max, 6),
        "x_center": round((x_min + x_max) / 2, 6),
        "y_center": round((y_min + y_max) / 2, 6),
        "width": round(x_max - x_min, 6),
        "height": round(y_max - y_min, 6),
    }


def _matrix3(value: object, *, description: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise EvidenceError(f"{description} must be a 3x3 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise EvidenceError(f"{description} must be a 3x3 matrix")
        matrix.append([
            _require_number(item, description=description) for item in row
        ])
    return matrix


def _project(point: tuple[float, float], matrix: Sequence[Sequence[float]], *,
             description: str) -> tuple[float, float]:
    x, y = point
    divisor = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if not math.isfinite(divisor) or abs(divisor) < 1e-12:
        raise EvidenceError(f"{description} projects a point to infinity")
    projected = (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / divisor,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / divisor,
    )
    if not all(math.isfinite(item) for item in projected):
        raise EvidenceError(f"{description} produces a non-finite point")
    return projected


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _matrix_infinity_norm(matrix: Sequence[Sequence[float]]) -> float:
    return max(sum(abs(value) for value in row) for row in matrix)


def _require_homography_pair(
    source_to_rectified: Sequence[Sequence[float]],
    rectified_to_source: Sequence[Sequence[float]],
    *,
    source_width: int,
    source_height: int,
    rectified_width: int,
    rectified_height: int,
    rotation_degrees: int,
    description: str,
) -> None:
    # A literal A*A^-1 elementwise comparison is unnecessarily brittle for
    # homographies with a large translation: subtracting two ~image-size
    # values can leave a harmless residual above a fixed 1e-6. Validate the
    # observable geometry instead, while retaining explicit singularity and
    # conditioning guards.
    for name, matrix, inverse in (
        ("original_to_rectified", source_to_rectified, rectified_to_source),
        ("rectified_to_original", rectified_to_source, source_to_rectified),
    ):
        determinant = _determinant3(matrix)
        norm = _matrix_infinity_norm(matrix)
        inverse_norm = _matrix_infinity_norm(inverse)
        if not math.isfinite(determinant) or abs(determinant) <= 1e-15:
            raise EvidenceError(f"{description} {name} matrix is singular")
        reciprocal_condition_bound = 1.0 / (norm * inverse_norm)
        if not math.isfinite(reciprocal_condition_bound) or reciprocal_condition_bound < 1e-12:
            raise EvidenceError(f"{description} {name} matrix is ill-conditioned")

    source_corners = [
        (0.0, 0.0),
        (float(source_width - 1), 0.0),
        (float(source_width - 1), float(source_height - 1)),
        (0.0, float(source_height - 1)),
    ]
    rectified_corners = [
        (0.0, 0.0),
        (float(rectified_width - 1), 0.0),
        (float(rectified_width - 1), float(rectified_height - 1)),
        (0.0, float(rectified_height - 1)),
    ]
    expected_forward = rectified_corners if rotation_degrees == 0 else [
        rectified_corners[1],
        rectified_corners[2],
        rectified_corners[3],
        rectified_corners[0],
    ]
    tolerance = max(
        0.02,
        max(source_width, source_height, rectified_width, rectified_height) * 1e-6,
    )

    def require_close(observed: tuple[float, float], expected: tuple[float, float], label: str) -> None:
        if any(
            not math.isclose(observed[axis], expected[axis], rel_tol=0, abs_tol=tolerance)
            for axis in range(2)
        ):
            raise EvidenceError(
                f"{description} {label} differs by more than {tolerance:.6g} pixel(s)"
            )

    for index, (source, expected) in enumerate(zip(source_corners, expected_forward, strict=True)):
        require_close(
            _project(source, source_to_rectified, description=description),
            expected,
            f"source corner[{index}] projection",
        )

    source_probes = [
        *source_corners,
        ((source_width - 1) / 2.0, (source_height - 1) / 2.0),
        ((source_width - 1) / 4.0, (source_height - 1) / 4.0),
        ((source_width - 1) * 0.75, (source_height - 1) * 0.75),
    ]
    rectified_probes = [
        *rectified_corners,
        ((rectified_width - 1) / 2.0, (rectified_height - 1) / 2.0),
        ((rectified_width - 1) / 4.0, (rectified_height - 1) / 4.0),
        ((rectified_width - 1) * 0.75, (rectified_height - 1) * 0.75),
    ]
    for index, source in enumerate(source_probes):
        rectified = _project(source, source_to_rectified, description=description)
        require_close(
            _project(rectified, rectified_to_source, description=description),
            source,
            f"source round-trip[{index}]",
        )
    for index, rectified in enumerate(rectified_probes):
        source = _project(rectified, rectified_to_source, description=description)
        require_close(
            _project(source, source_to_rectified, description=description),
            rectified,
            f"rectified round-trip[{index}]",
        )


def _status_bar_limit(size: int) -> int:
    return max(1, round(size * STATUS_BAR_FRACTION)) - 1


def _convex_hull_area(points: Sequence[tuple[float, float]]) -> float:
    unique = sorted(set(points))
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    return abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    ) / 2.0


def _quad_has_self_intersection(
    points: Sequence[tuple[float, float]],
) -> bool:
    """Return true when non-adjacent edges cross, touch, or overlap."""

    if len(points) != 4:
        return False

    def cross(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])

    def intersects(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> bool:
        first_left = cross(first_start, first_end, second_start)
        first_right = cross(first_start, first_end, second_end)
        second_left = cross(second_start, second_end, first_start)
        second_right = cross(second_start, second_end, first_end)

        def strictly_opposite_sign(first: float, second: float) -> bool:
            # Do not use an epsilon: a near-endpoint crossing remains a real
            # bow-tie and must stay invalid at any nonzero scale.
            return (first < 0 < second) or (second < 0 < first)

        if strictly_opposite_sign(first_left, first_right) and (
            strictly_opposite_sign(second_left, second_right)
        ):
            return True

        def lies_on_segment(
            start: tuple[float, float],
            end: tuple[float, float],
            point: tuple[float, float],
            orientation: float,
        ) -> bool:
            return orientation == 0.0 and (
                min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
                and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
            )

        return (
            lies_on_segment(first_start, first_end, second_start, first_left)
            or lies_on_segment(first_start, first_end, second_end, first_right)
            or lies_on_segment(second_start, second_end, first_start, second_left)
            or lies_on_segment(second_start, second_end, first_end, second_right)
        )

    return intersects(points[0], points[1], points[2], points[3]) or intersects(
        points[1], points[2], points[3], points[0]
    )


def _quad_geometry(
    line: Mapping[str, Any],
    *,
    record_index: int,
    line_index: int,
    rectified_width: int,
    rectified_height: int,
    source_width: int,
    source_height: int,
    rectified_to_source: Sequence[Sequence[float]],
    allow_invalid_quad_contract_violation: bool = False,
) -> dict[str, Any]:
    raw = line.get("quad_rectified")
    normalized = line.get("quad_rectified_normalized")
    if not isinstance(raw, list) or not isinstance(normalized, list) or len(raw) != 4 or len(normalized) != 4:
        raise EvidenceError(f"layout record[{record_index}] line[{line_index}] quad must have four points")
    raw_points: list[tuple[float, float]] = []
    normalized_points: list[tuple[float, float]] = []
    for point_index, (raw_point, normalized_point) in enumerate(zip(raw, normalized, strict=True)):
        if not isinstance(raw_point, list) or not isinstance(normalized_point, list) \
                or len(raw_point) != 2 or len(normalized_point) != 2:
            raise EvidenceError(f"layout record[{record_index}] line[{line_index}] point[{point_index}] is invalid")
        x = _require_number(raw_point[0], description="quad x", minimum=0,
                            maximum=rectified_width - 1)
        y = _require_number(raw_point[1], description="quad y", minimum=0,
                            maximum=rectified_height - 1)
        nx = _require_number(normalized_point[0], description="normalized quad x", minimum=0, maximum=1)
        ny = _require_number(normalized_point[1], description="normalized quad y", minimum=0, maximum=1)
        if not math.isclose(nx, x / (rectified_width - 1), rel_tol=0, abs_tol=2e-5) \
                or not math.isclose(ny, y / (rectified_height - 1), rel_tol=0, abs_tol=2e-5):
            raise EvidenceError(
                f"layout record[{record_index}] line[{line_index}] normalized quad disagrees with rectified quad"
            )
        raw_points.append((x, y))
        normalized_points.append((nx, ny))
    source_points = [
        _project(point, rectified_to_source, description="H_rectified_to_original")
        for point in raw_points
    ]
    for x, y in source_points:
        if x < -1e-3 or x > source_width - 1 + 1e-3 or y < -1e-3 or y > source_height - 1 + 1e-3:
            raise EvidenceError(
                f"layout record[{record_index}] line[{line_index}] projects outside source bounds"
            )
    polygon_area = abs(sum(
        raw_points[item][0] * raw_points[(item + 1) % 4][1]
        - raw_points[(item + 1) % 4][0] * raw_points[item][1]
        for item in range(4)
    )) / 2.0
    hull_area = _convex_hull_area(raw_points)
    self_intersects = _quad_has_self_intersection(raw_points)
    turn_crosses = [
        (
            raw_points[(index + 1) % 4][0] - raw_points[index][0]
        ) * (
            raw_points[(index + 2) % 4][1]
            - raw_points[(index + 1) % 4][1]
        ) - (
            raw_points[(index + 1) % 4][1] - raw_points[index][1]
        ) * (
            raw_points[(index + 2) % 4][0]
            - raw_points[(index + 1) % 4][0]
        )
        for index in range(4)
    ]
    turn_tolerance = max(1e-3, hull_area * 1e-9)
    consistently_cyclic_convex = all(
        cross > turn_tolerance for cross in turn_crosses
    ) or all(cross < -turn_tolerance for cross in turn_crosses)
    unique_points = len(set(raw_points))
    bounding_width = max(point[0] for point in raw_points) - min(
        point[0] for point in raw_points
    )
    bounding_height = max(point[1] for point in raw_points) - min(
        point[1] for point in raw_points
    )
    degenerate_quad: dict[str, Any] | None = None
    classification: str | None = None
    if unique_points < 4:
        classification = "repeated_points"
    elif bounding_width <= 1e-9 or bounding_height <= 1e-9:
        classification = "axis_collapsed"
    elif hull_area <= 1e-9:
        classification = "collinear_points"
    elif polygon_area <= 1e-3 and hull_area > 1e-3:
        classification = "order_cancels_nondegenerate_hull"
    elif polygon_area <= 1e-3:
        classification = "sub_millipixel_area"
    elif self_intersects:
        classification = "self_intersects_nondegenerate_hull"
    elif not consistently_cyclic_convex or (
        hull_area - polygon_area > max(1e-3, hull_area * 1e-6)
    ):
        # A cyclic convex quadrilateral has the same shoelace and convex-hull
        # area in either traversal direction.  A material gap proves that a
        # four-point DB rectangle contract became concave/non-convex; retain
        # only the diagnostic and quarantine its complete record.
        classification = "nonconvex_nondegenerate_hull"

    if classification is not None:
        if not allow_invalid_quad_contract_violation:
            diagnostic = json.dumps(
                {
                    "classification": classification,
                    "quad_rectified": [list(point) for point in raw_points],
                    "polygon_area_pixels2": polygon_area,
                    "convex_hull_area_pixels2": hull_area,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if classification == "order_cancels_nondegenerate_hull":
                message = "quad order cancels a non-degenerate hull"
            elif classification == "self_intersects_nondegenerate_hull":
                message = "quad is self-intersecting with nonzero ordered area"
            elif classification == "nonconvex_nondegenerate_hull":
                message = "quad is non-convex relative to its hull"
            else:
                message = "quad is degenerate"
            raise EvidenceError(
                f"layout record[{record_index}] line[{line_index}] {message}: {diagnostic}"
            )
        degenerate_quad = {
            "classification": classification,
            "polygon_area_pixels2": polygon_area,
            "convex_hull_area_pixels2": hull_area,
            "unique_points": unique_points,
            "bounding_width_pixels": bounding_width,
            "bounding_height_pixels": bounding_height,
            "candidate_eligible": False,
            "producer_contract_violation": True,
            "record_candidate_eligible": False,
            "canonicalized": False,
        }
        if classification == "nonconvex_nondegenerate_hull":
            degenerate_quad["turn_crosses_pixels2"] = turn_crosses
    source_normalized = [
        (
            min(1.0, max(0.0, x / (source_width - 1))),
            min(1.0, max(0.0, y / (source_height - 1))),
        )
        for x, y in source_points
    ]
    rectified_membership = max(point[1] for point in raw_points) <= _status_bar_limit(rectified_height)
    source_membership = max(point[1] for point in source_points) <= _status_bar_limit(source_height) + 1e-3
    return {
        "rectified_normalized": _rect(normalized_points),
        "source_normalized": _rect(source_normalized),
        "rectified_top8_membership": rectified_membership,
        "source_top8_membership": source_membership,
        "top8_membership_disagrees": rectified_membership != source_membership,
        "rectified_top8_y_max_normalized": round(
            _status_bar_limit(rectified_height) / (rectified_height - 1), 6
        ),
        "source_top8_y_max_normalized": round(
            _status_bar_limit(source_height) / (source_height - 1), 6
        ),
        "degenerate_quad": degenerate_quad,
    }


def _line_evidence(line: Mapping[str, Any]) -> dict[str, Any]:
    geometry = line["_geometry"]
    return {
        "line_index": line["index"],
        "text": line["text"],
        "confidence": line["confidence"],
        "passes_drop_score": line["passes_drop_score"],
        "quad_rectified_normalized": line["quad_rectified_normalized"],
        "rect_normalized": dict(geometry["rectified_normalized"]),
        "source_rect_normalized": dict(geometry["source_normalized"]),
        "rectified_top8_membership": geometry["rectified_top8_membership"],
        "source_top8_membership": geometry["source_top8_membership"],
        "top8_membership_disagrees": geometry["top8_membership_disagrees"],
        "rotation_degrees": line["_rotation_degrees"],
    }


def _clock_value(text: str) -> str | None:
    visible = unicodedata.normalize("NFC", text).strip().replace("：", ":")
    match = CLOCK_PATTERN.fullmatch(visible)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None
    return visible


def _body_anchor_support(lines: Sequence[Mapping[str, Any]], clock_line: Mapping[str, Any]) -> list[dict[str, Any]]:
    clock_bottom = clock_line["_geometry"]["rectified_normalized"]["y_max"]
    support: list[dict[str, Any]] = []
    for line in lines:
        if line["index"] == clock_line["index"] or line["passes_drop_score"] is not True:
            continue
        rect = line["_geometry"]["rectified_normalized"]
        if rect["y_center"] <= clock_bottom:
            continue
        cleaned = _clean_text(str(line["text"]))
        for label in PAYMENT_LABELS:
            if cleaned.startswith(label):
                support.append({
                    "line_index": line["index"],
                    "anchor_type": "payment_label_below_clock",
                    "matched_token": label,
                    "confidence": line["confidence"],
                    "rect_normalized": dict(rect),
                })
                break
        segments, _hard_boundary = _cjk_segments(cleaned)
        status_hits = [
            (phrase, status_class)
            for segment in segments
            for phrase, status_class in STATUS_PHRASE_CLASS.items()
            if phrase in segment
            and not (status_class == "success" and any(token in segment for token in STATUS_SUCCESS_BLOCKERS))
        ]
        if len(set(status_hits)) == 1:
            phrase, status_class = status_hits[0]
            support.append({
                "line_index": line["index"],
                "anchor_type": "visible_status_phrase_below_clock",
                "matched_token": phrase,
                "status_class": status_class,
                "confidence": line["confidence"],
                "rect_normalized": dict(rect),
            })
    return support


def _time_evidence(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for line in lines:
        clock = _clock_value(str(line["text"]))
        if clock is None:
            text = str(line["text"])
            if TIME_LIKE_PATTERN.search(text) is not None:
                item = _line_evidence(line)
                item["exclusion_reason"] = (
                    "not_one_full_line_h_mm_or_hh_mm_status_bar_clock"
                )
                excluded.append(item)
            continue
        geometry = line["_geometry"]
        support = _body_anchor_support(lines, line)
        anchor = _line_evidence(line)
        anchor.update({
            "anchor_kind": "strict_visible_clock_full_line",
            "visible_clock": clock,
            "status_bar_geometry_evidence": (
                geometry["rectified_top8_membership"]
                and geometry["source_top8_membership"]
                and not geometry["top8_membership_disagrees"]
                and line["_rotation_degrees"] == 0
            ),
            "accepted_body_anchors_below": support,
        })
        anchors.append(anchor)
    top = [anchor for anchor in anchors if anchor["status_bar_geometry_evidence"]]
    accepted_top = [anchor for anchor in top if anchor["passes_drop_score"]]
    if not anchors:
        ambiguity = "excluded_time_like_only" if excluded else "no_strict_clock_anchor"
    elif any(anchor["top8_membership_disagrees"] for anchor in anchors):
        ambiguity = "source_rectified_top8_region_disagreement"
    elif any(
        anchor["rotation_degrees"] != 0
        and (anchor["source_top8_membership"] or anchor["rectified_top8_membership"])
        for anchor in anchors
    ):
        ambiguity = "nonzero_rotation_status_bar_region_ambiguous"
    elif not top:
        ambiguity = "clock_anchor_outside_source_and_rectified_top8"
    elif len(top) == 1 and not top[0]["passes_drop_score"]:
        ambiguity = "single_top8_clock_below_drop_score"
    elif len(top) == 1 and not top[0]["accepted_body_anchors_below"]:
        ambiguity = "single_top8_clock_without_body_anchor"
    elif len(top) == 1:
        ambiguity = "unique_top8_status_bar_anchor_with_body_support"
    else:
        ambiguity = "multiple_top8_status_bar_anchors"
    return {
        "semantic_scope": "visible_screen_status_bar_clock_only",
        "transaction_time_labels_accepted": False,
        "seconds_or_datetime_accepted": False,
        "embedded_clock_accepted": False,
        "reverse_clock_repair_applied": False,
        "status_bar_fraction": STATUS_BAR_FRACTION,
        "anchor_count": len(anchors),
        "accepted_anchor_count": sum(anchor["passes_drop_score"] for anchor in anchors),
        "top8_status_bar_anchor_count": len(top),
        "accepted_top8_status_bar_anchor_count": len(accepted_top),
        "excluded_time_like_count": len(excluded),
        "ambiguity": ambiguity,
        "unique_diagnostic_coverage": len(top) == 1
        and len(accepted_top) == 1
        and bool(top[0]["accepted_body_anchors_below"]),
        "anchors": anchors,
        "excluded_time_like_lines": excluded,
    }


def _payment_value_grammar(text: str) -> str | None:
    cleaned = _clean_text(text)
    if cleaned in PAYMENT_FIXED_VALUES:
        return PAYMENT_FIXED_VALUES[cleaned]
    match = PAYMENT_CARD_PATTERN.fullmatch(cleaned)
    if match is None:
        return None
    if (match.group("open"), match.group("close")) not in {("(", ")"), ("（", "）")}:
        return None
    return "bank_card_tail4"


def _vertical_overlap(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    overlap = max(0.0, min(left["y_max"], right["y_max"]) - max(left["y_min"], right["y_min"]))
    denominator = min(left["height"], right["height"])
    return 0.0 if denominator <= 0 else overlap / denominator


def _payment_evidence(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for line in lines:
        cleaned = _clean_text(str(line["text"]))
        matching = [label for label in PAYMENT_LABELS if label in cleaned]
        if not matching:
            continue
        label = matching[0]
        rect = line["_geometry"]["rectified_normalized"]
        if len(matching) != 1:
            relation = "multiple_labels_conflict"
            remainder = ""
        elif cleaned == label:
            relation = "exact_label_line"
            remainder = ""
        elif cleaned.startswith(label):
            relation = "label_prefix_same_line"
            remainder = cleaned[len(label):].lstrip(" :：-—")
        else:
            relation = "nonprefix_label_conflict"
            remainder = ""
        anchor = _line_evidence(line)
        anchor.update({
            "anchor_label": label,
            "matched_labels": matching,
            "anchor_relation": relation,
        })
        anchors.append(anchor)
        if relation == "label_prefix_same_line" and remainder:
            grammar = _payment_value_grammar(remainder)
            if grammar is not None:
                values.append({
                    "source": "same_line_after_label",
                    "anchor_line_index": line["index"],
                    "value_line_index": line["index"],
                    "visible_text": remainder,
                    "strict_value_grammar": grammar,
                    "confidence": line["confidence"],
                    "passes_drop_score": line["passes_drop_score"],
                    "rect_normalized": dict(rect),
                })
        if relation == "exact_label_line":
            for other in lines:
                if other["index"] == line["index"]:
                    continue
                grammar = _payment_value_grammar(str(other["text"]))
                if grammar is None:
                    continue
                other_rect = other["_geometry"]["rectified_normalized"]
                overlap = _vertical_overlap(rect, other_rect)
                gap = other_rect["x_min"] - rect["x_max"]
                if other_rect["x_center"] <= rect["x_center"] or overlap < 0.50 or gap < -0.03 or gap > 0.65:
                    continue
                values.append({
                    "source": "same_row_right_neighbor",
                    "anchor_line_index": line["index"],
                    "value_line_index": other["index"],
                    "visible_text": _clean_text(str(other["text"])),
                    "strict_value_grammar": grammar,
                    "confidence": other["confidence"],
                    "passes_drop_score": other["passes_drop_score"],
                    "vertical_overlap_ratio": round(overlap, 6),
                    "horizontal_gap_normalized": round(gap, 6),
                    "rect_normalized": dict(other_rect),
                })
    distinct_values = {(value["visible_text"], value["strict_value_grammar"]) for value in values}
    strict_anchors = [
        anchor for anchor in anchors
        if anchor["anchor_relation"] in {"exact_label_line", "label_prefix_same_line"}
    ]
    accepted_strict_anchors = [anchor for anchor in strict_anchors if anchor["passes_drop_score"]]
    accepted_values = [value for value in values if value["passes_drop_score"]]
    if not anchors:
        ambiguity = "no_payment_label_anchor"
    elif len(strict_anchors) != 1:
        ambiguity = "multiple_or_conflicting_payment_labels"
    elif len(accepted_strict_anchors) != 1:
        ambiguity = "payment_label_below_drop_score"
    elif not values:
        ambiguity = "unique_label_without_strict_value_evidence"
    elif len(values) == 1 and len(distinct_values) == 1:
        ambiguity = "unique_strict_payment_value_evidence"
    else:
        ambiguity = "multiple_strict_payment_values"
    return {
        "anchor_count": len(anchors),
        "strict_anchor_count": len(strict_anchors),
        "accepted_strict_anchor_count": len(accepted_strict_anchors),
        "strict_value_evidence_count": len(values),
        "accepted_strict_value_evidence_count": len(accepted_values),
        "ambiguity": ambiguity,
        "unique_diagnostic_coverage": len(strict_anchors) == 1
        and len(accepted_strict_anchors) == 1
        and len(values) == 1 and len(distinct_values) == 1 and len(accepted_values) == 1,
        "anchors": anchors,
        "value_geometry_evidence": values,
    }


def _is_cjk(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _cjk_segments(value: str) -> tuple[list[str], bool]:
    segments: list[str] = []
    current: list[str] = []
    has_hard_boundary = False
    for character in unicodedata.normalize("NFC", value):
        if _is_cjk(character):
            current.append(character)
        elif character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        else:
            has_hard_boundary = True
            if current:
                segments.append("".join(current))
                current = []
    if current:
        segments.append("".join(current))
    return segments, has_hard_boundary


def _status_evidence(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    all_classes: set[str] = set()
    blocked_success = False
    for line in lines:
        segments, hard_boundary = _cjk_segments(str(line["text"]))
        hits: list[dict[str, Any]] = []
        for segment in segments:
            for phrase, status_class in STATUS_PHRASE_CLASS.items():
                if phrase not in segment:
                    continue
                blocked = status_class == "success" and any(
                    token in segment for token in STATUS_SUCCESS_BLOCKERS
                )
                blocked_success = blocked_success or blocked
                hits.append({
                    "phrase": phrase,
                    "phrase_class": status_class,
                    "same_cjk_stream_success_blocker": blocked,
                    "exact_isolated_line": not hard_boundary and len(segments) == 1 and segment == phrase,
                })
                if not blocked:
                    all_classes.add(status_class)
        if not hits:
            continue
        anchor = _line_evidence(line)
        anchor.update({"anchor_kind": "audited_long_visible_status_phrase", "phrase_hits": hits})
        anchors.append(anchor)
    unblocked_pairs = [
        (anchor, hit)
        for anchor in anchors
        for hit in anchor["phrase_hits"]
        if not hit["same_cjk_stream_success_blocker"]
    ]
    unblocked = [hit for _anchor, hit in unblocked_pairs]
    exact_pairs = [
        (anchor, hit) for anchor, hit in unblocked_pairs if hit["exact_isolated_line"]
    ]
    exact = [hit for _anchor, hit in exact_pairs]
    accepted_exact = [
        hit for anchor, hit in exact_pairs if anchor["passes_drop_score"]
    ]
    distinct = {(hit["phrase"], hit["phrase_class"]) for hit in unblocked}
    if not anchors:
        ambiguity = "no_audited_long_status_phrase"
    elif blocked_success:
        ambiguity = "success_phrase_has_blocking_context"
    elif len(distinct) != 1:
        ambiguity = "multiple_distinct_status_phrases"
    elif len(exact) != 1:
        ambiguity = "status_phrase_not_one_exact_isolated_line"
    elif len(accepted_exact) != 1:
        ambiguity = "exact_status_phrase_below_drop_score"
    else:
        ambiguity = "unique_exact_isolated_status_phrase_evidence"
    return {
        "long_phrase_anchor_count": len(anchors),
        "unblocked_phrase_hit_count": len(unblocked),
        "exact_isolated_phrase_hit_count": len(exact),
        "accepted_exact_isolated_phrase_hit_count": len(accepted_exact),
        "observed_phrase_classes": sorted(all_classes),
        "blocked_success_evidence": blocked_success,
        "ambiguity": ambiguity,
        "unique_diagnostic_coverage": len(distinct) == 1 and len(exact) == 1
        and len(accepted_exact) == 1 and not blocked_success,
        "requires_independent_tight_crop_confirmation": True,
        "success_acceptance_enabled": False,
        "anchors": anchors,
    }


def _validate_layout(
    layout_directory: Path,
    selection_directory: Path,
    selection: Mapping[str, Any],
    sources: Sequence[Path],
    source_identities: Sequence[Mapping[str, Any]],
    missing_sets: Mapping[str, set[str]],
    *,
    allow_invalid_quad_contract_violation_lines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = layout_directory / "summary.json"
    records_path = layout_directory / "records.jsonl"
    summary_bytes = _read_bytes(summary_path, description="layout shadow summary")
    records_bytes = _read_bytes(records_path, description="layout shadow records")
    summary = _load_json(summary_path, summary_bytes, description="layout shadow summary")
    records = _load_jsonl(records_path, records_bytes, description="layout shadow records")
    if summary.get("schema_version") != 1 or summary.get("kind") != LAYOUT_SUMMARY_KIND:
        raise EvidenceError("layout shadow summary schema/kind is unsupported")
    _require_bool(summary.get("diagnostic_only"), True, description="layout diagnostic_only")
    _require_bool(summary.get("formal_delivery_gate"), False, description="layout formal_delivery_gate")
    _require_bool(summary.get("candidate_write_enabled"), False, description="layout candidate_write_enabled")
    _require_int(summary.get("expected_records"), EXPECTED_RECORDS, description="layout expected_records")
    _require_int(summary.get("records"), EXPECTED_RECORDS, description="layout records")
    _require_int(summary.get("errors"), 0, description="layout errors")
    if summary.get("execution_provider") != "cpu":
        raise EvidenceError("layout shadow execution_provider must be cpu")
    if summary.get("rectification") != RECTIFICATION or summary.get("quad_coordinate_space") != QUAD_COORDINATE_SPACE \
            or summary.get("quad_normalization") != QUAD_NORMALIZATION \
            or summary.get("confidence_semantics") != CONFIDENCE_SEMANTICS:
        raise EvidenceError("layout shadow geometry/confidence contract is unsupported")
    drop_score = _require_number(summary.get("paddle_drop_score"), description="layout paddle_drop_score",
                                 minimum=0, maximum=1)
    input_contract = summary.get("input_list")
    selection_input = selection.get("input_list")
    if not isinstance(input_contract, Mapping) or not isinstance(selection_input, Mapping):
        raise EvidenceError("layout/selection input-list contract is missing")
    input_path = selection_directory / "inputs.txt"
    if not isinstance(input_contract.get("path"), str) or _path_key(input_contract["path"]) != _path_key(input_path):
        raise EvidenceError("layout input-list path differs from selection inputs.txt")
    if _require_sha(input_contract.get("sha256"), description="layout input-list sha256") != selection_input.get("sha256"):
        raise EvidenceError("layout input-list SHA-256 differs from selection")
    _require_int(input_contract.get("size_bytes"), selection_input.get("size_bytes"),
                 description="layout input-list size")
    _require_int(input_contract.get("records"), EXPECTED_RECORDS, description="layout input-list records")
    artifacts = summary.get("artifacts")
    records_contract = artifacts.get("records_jsonl") if isinstance(artifacts, Mapping) else None
    if not isinstance(records_contract, Mapping) or records_contract.get("relative_path") != "records.jsonl":
        raise EvidenceError("layout records artifact contract is unsupported")
    if _require_sha(records_contract.get("sha256"), description="layout records sha256") != _sha256(records_bytes):
        raise EvidenceError("layout records SHA-256 differs from summary")
    _require_int(records_contract.get("size_bytes"), len(records_bytes), description="layout records size")
    if len(records) != EXPECTED_RECORDS:
        raise EvidenceError(f"layout records.jsonl must contain {EXPECTED_RECORDS} records")

    evidence_rows: list[dict[str, Any]] = []
    degenerate_lines: list[dict[str, Any]] = []
    for index, (record, source, source_identity) in enumerate(
        zip(records, sources, source_identities, strict=True)
    ):
        if record.get("schema_version") != 1 or record.get("kind") != LAYOUT_RECORD_KIND:
            raise EvidenceError(f"layout record[{index}] schema/kind is unsupported")
        _require_bool(record.get("diagnostic_only"), True, description=f"layout record[{index}] diagnostic_only")
        _require_bool(record.get("formal_delivery_gate"), False, description=f"layout record[{index}] formal_delivery_gate")
        _require_bool(record.get("candidate_write_enabled"), False,
                      description=f"layout record[{index}] candidate_write_enabled")
        _require_int(record.get("index"), index, description=f"layout record[{index}] index")
        source_raw = record.get("source")
        if not isinstance(source_raw, str) or _path_key(source_raw) != _path_key(source):
            raise EvidenceError(f"layout record[{index}] source/order differs from selection")
        if _require_sha(record.get("source_image_sha256"), description=f"layout record[{index}] source sha256") \
                != source_identity["sha256"]:
            raise EvidenceError(f"layout record[{index}] source image SHA-256 differs")
        _require_int(record.get("source_image_size_bytes"), int(source_identity["size_bytes"]),
                     description=f"layout record[{index}] source size")
        if record.get("execution_provider") != "cpu" or record.get("quad_coordinate_space") != QUAD_COORDINATE_SPACE \
                or record.get("quad_normalization") != QUAD_NORMALIZATION \
                or record.get("confidence_semantics") != CONFIDENCE_SEMANTICS:
            raise EvidenceError(f"layout record[{index}] provider/geometry/confidence contract differs")
        geometry = record.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("rectification") != RECTIFICATION:
            raise EvidenceError(f"layout record[{index}] rectification geometry is invalid")
        rectified_size = geometry.get("rectified_size")
        source_size = geometry.get("source_size")
        if not isinstance(rectified_size, Mapping) or not isinstance(source_size, Mapping):
            raise EvidenceError(f"layout record[{index}] source/rectified size is missing")
        width = _require_int(rectified_size.get("width"), description="rectified width")
        height = _require_int(rectified_size.get("height"), description="rectified height")
        source_width = _require_int(source_size.get("width"), description="source width")
        source_height = _require_int(source_size.get("height"), description="source height")
        if width < 2 or height < 2 or source_width < 2 or source_height < 2 \
                or max(width, height) > 1600:
            raise EvidenceError(f"layout record[{index}] rectified dimensions are invalid")
        rotation = _require_int(geometry.get("rotation_degrees"), description="rotation_degrees")
        if rotation not in {0, 90}:
            raise EvidenceError(f"layout record[{index}] rotation_degrees is unsupported")
        if geometry.get("screen_detected") is not False:
            raise EvidenceError(f"layout record[{index}] screen_detected must be false in max-side-1600 mode")
        rectified_to_source = _matrix3(
            geometry.get("H_rectified_to_original"),
            description="H_rectified_to_original",
        )
        source_to_rectified = _matrix3(
            geometry.get("H_original_to_rectified"),
            description="H_original_to_rectified",
        )
        _require_homography_pair(
            source_to_rectified,
            rectified_to_source,
            source_width=source_width,
            source_height=source_height,
            rectified_width=width,
            rectified_height=height,
            rotation_degrees=rotation,
            description=f"layout record[{index}] homography",
        )
        screen_quad = geometry.get("screen_quad_original")
        expected_screen_quad = [
            [0.0, 0.0],
            [float(source_width - 1), 0.0],
            [float(source_width - 1), float(source_height - 1)],
            [0.0, float(source_height - 1)],
        ]
        if not isinstance(screen_quad, list) or len(screen_quad) != 4:
            raise EvidenceError(f"layout record[{index}] screen_quad_original is invalid")
        for point_index, (actual, expected) in enumerate(zip(screen_quad, expected_screen_quad, strict=True)):
            if not isinstance(actual, list) or len(actual) != 2 or any(
                not math.isclose(
                    _require_number(actual[axis], description="screen quad coordinate"),
                    expected[axis], rel_tol=0, abs_tol=1e-5,
                )
                for axis in range(2)
            ):
                raise EvidenceError(
                    f"layout record[{index}] screen_quad_original[{point_index}] is not the full image"
                )
        raw_lines = record.get("lines")
        if not isinstance(raw_lines, list):
            raise EvidenceError(f"layout record[{index}] lines is not a list")
        _require_int(record.get("raw_line_count"), len(raw_lines), description="raw_line_count")
        prepared_lines: list[dict[str, Any]] = []
        for line_index, line in enumerate(raw_lines):
            if not isinstance(line, dict):
                raise EvidenceError(f"layout record[{index}] line[{line_index}] is not an object")
            _require_int(line.get("index"), line_index, description="layout line index")
            if not isinstance(line.get("text"), str):
                raise EvidenceError(f"layout record[{index}] line[{line_index}] text is invalid")
            confidence = _require_number(line.get("confidence"), description="layout line confidence",
                                         minimum=0, maximum=1)
            passes = line.get("passes_drop_score")
            if type(passes) is not bool or passes != (confidence >= drop_score):
                raise EvidenceError(f"layout record[{index}] line[{line_index}] drop-score flag differs")
            line_geometry = _quad_geometry(
                line,
                record_index=index,
                line_index=line_index,
                rectified_width=width,
                rectified_height=height,
                source_width=source_width,
                source_height=source_height,
                rectified_to_source=rectified_to_source,
                allow_invalid_quad_contract_violation=(
                    allow_invalid_quad_contract_violation_lines
                ),
            )
            prepared = dict(line)
            prepared["_geometry"] = line_geometry
            prepared["_rotation_degrees"] = rotation
            prepared_lines.append(prepared)
            degenerate_quad = line_geometry.get("degenerate_quad")
            if isinstance(degenerate_quad, Mapping):
                degenerate_lines.append(
                    {
                        "record_index": index,
                        "line_index": line_index,
                        "source": str(source),
                        "text": line["text"],
                        "confidence": confidence,
                        "passes_drop_score": passes,
                        "quad_rectified": line.get("quad_rectified"),
                        "quad_rectified_normalized": line.get(
                            "quad_rectified_normalized"
                        ),
                        **dict(degenerate_quad),
                    }
                )
        accepted = [line for line in prepared_lines if line["passes_drop_score"]]
        _require_int(record.get("accepted_line_count"), len(accepted), description="accepted_line_count")
        accepted_text = " ".join(
            cleaned for cleaned in (_clean_text(line["text"]) for line in accepted) if cleaned
        )
        if record.get("accepted_text") != accepted_text:
            raise EvidenceError(f"layout record[{index}] accepted_text projection differs")
        accepted_confidence = record.get("accepted_confidence")
        if not accepted:
            if accepted_confidence is not None:
                raise EvidenceError(f"layout record[{index}] accepted_confidence must be null")
        else:
            observed_confidence = _require_number(accepted_confidence, description="accepted confidence",
                                                  minimum=0, maximum=1)
            expected_confidence = statistics.fmean(line["confidence"] for line in accepted)
            if not math.isclose(observed_confidence, expected_confidence, rel_tol=0, abs_tol=2e-6):
                raise EvidenceError(f"layout record[{index}] accepted_confidence projection differs")
        timing = record.get("timing_ms")
        if not isinstance(timing, Mapping):
            raise EvidenceError(f"layout record[{index}] timing_ms is missing")
        for stage in ("image_load", "rectification", "layout_ocr", "total"):
            _require_number(timing.get(stage), description=f"layout timing {stage}", minimum=0)

        source_key = _path_key(source)
        record_has_invalid_quad_contract = any(
            isinstance(line["_geometry"].get("degenerate_quad"), Mapping)
            for line in prepared_lines
        )
        analysis_lines = [] if record_has_invalid_quad_contract else [
            line
            for line in prepared_lines
            if line["_geometry"].get("degenerate_quad") is None
        ]
        evidence_rows.append({
            "schema_version": 1,
            "kind": EVIDENCE_RECORD_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "index": index,
            "source": str(source),
            "source_image_sha256": source_identity["sha256"],
            "audit_missing_fields": [field for field in FIELD_SPECS if source_key in missing_sets[field]],
            "evidence_by_field": {
                "time": _time_evidence(analysis_lines),
                "payment_method_field": _payment_evidence(analysis_lines),
                "transfer_status": _status_evidence(analysis_lines),
            },
        })
    bindings = {
        "layout_summary": _identity(summary_path, summary_bytes),
        "layout_records": _identity(records_path, records_bytes),
        "paddle_bundle": summary.get("paddle_bundle"),
        "paddle_drop_score": drop_score,
        "layout_latency_ms": summary.get("latency_ms"),
    }
    if allow_invalid_quad_contract_violation_lines:
        bindings["excluded_quad_contract_lines"] = degenerate_lines
    return evidence_rows, bindings


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] if lower == upper else (
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )
    return round(value, 6)


def _distribution(values: Iterable[float]) -> dict[str, Any] | None:
    observed = list(values)
    if not observed:
        return None
    return {
        "count": len(observed),
        "min": round(min(observed), 6),
        "mean": round(statistics.fmean(observed), 6),
        "p50": _percentile(observed, 0.50),
        "p95": _percentile(observed, 0.95),
        "max": round(max(observed), 6),
    }


def _coverage(rows: Sequence[Mapping[str, Any]], field: str, target_keys: set[str]) -> dict[str, Any]:
    selected = [row for row in rows if _path_key(str(row["source"])) in target_keys]

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        evidence = [row["evidence_by_field"][field] for row in group]
        anchors = [anchor for item in evidence for anchor in item["anchors"]]
        result = {
            "records": len(group),
            "records_with_anchor": sum(bool(item["anchors"]) for item in evidence),
            "records_with_unique_diagnostic_coverage": sum(
                item["unique_diagnostic_coverage"] is True for item in evidence
            ),
            "anchor_coverage": round(
                sum(bool(item["anchors"]) for item in evidence) / len(group), 6
            ) if group else 0.0,
            "unique_diagnostic_coverage": round(
                sum(item["unique_diagnostic_coverage"] is True for item in evidence) / len(group), 6
            ) if group else 0.0,
            "ambiguity_counts": dict(sorted(Counter(item["ambiguity"] for item in evidence).items())),
            "anchor_confidence": _distribution(float(anchor["confidence"]) for anchor in anchors),
            "anchor_geometry": {
                axis: _distribution(float(anchor["rect_normalized"][axis]) for anchor in anchors)
                for axis in ("x_min", "x_max", "y_min", "y_max", "x_center", "y_center", "width", "height")
            },
        }
        if field == "time":
            metric_counts = {
                "records_with_exact_clock_anywhere": sum(item["anchor_count"] > 0 for item in evidence),
                "records_with_source_top8_clock": sum(
                    any(anchor["source_top8_membership"] for anchor in item["anchors"])
                    for item in evidence
                ),
                "records_with_rectified_top8_clock": sum(
                    any(anchor["rectified_top8_membership"] for anchor in item["anchors"])
                    for item in evidence
                ),
                "records_with_accepted_source_and_rectified_top8_clock": sum(
                    any(
                        anchor["passes_drop_score"]
                        and anchor["source_top8_membership"]
                        and anchor["rectified_top8_membership"]
                        and not anchor["top8_membership_disagrees"]
                        for anchor in item["anchors"]
                    )
                    for item in evidence
                ),
                "records_with_multiple_source_and_rectified_top8_clocks": sum(
                    item["top8_status_bar_anchor_count"] > 1 for item in evidence
                ),
                "records_with_excluded_time_like_text": sum(
                    item["excluded_time_like_count"] > 0 for item in evidence
                ),
                "records_with_source_rectified_region_disagreement": sum(
                    any(anchor["top8_membership_disagrees"] for anchor in item["anchors"])
                    for item in evidence
                ),
            }
            result["time_evidence_counts"] = metric_counts
            result["time_evidence_fractions"] = {
                key: round(value / len(group), 6) if group else 0.0
                for key, value in metric_counts.items()
            }
        elif field == "payment_method_field":
            result["payment_evidence_counts"] = {
                "records_with_strict_label": sum(item["strict_anchor_count"] > 0 for item in evidence),
                "records_with_strict_value_geometry": sum(
                    item["strict_value_evidence_count"] > 0 for item in evidence
                ),
                "records_with_multiple_strict_values": sum(
                    item["ambiguity"] == "multiple_strict_payment_values" for item in evidence
                ),
            }
        elif field == "transfer_status":
            result["status_evidence_counts"] = {
                "records_with_long_phrase": sum(item["long_phrase_anchor_count"] > 0 for item in evidence),
                "records_with_exact_isolated_phrase": sum(
                    item["exact_isolated_phrase_hit_count"] > 0 for item in evidence
                ),
                "records_with_blocked_success_evidence": sum(
                    item["blocked_success_evidence"] is True for item in evidence
                ),
                "success_acceptances": 0,
            }
        return result

    return {
        "selection": summarize(rows),
        "audit_missing_target": summarize(selected),
    }


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _source_closure(identities: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in identities:
        digest.update(
            f"{_path_key(str(item['path']))}\0{item['sha256']}\0{item['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def prepare_analysis(
    *,
    selection_directory: Path,
    audit_directory: Path,
    layout_directory: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    selection_dir = _require_directory(selection_directory, description="selection directory")
    audit_dir = _require_directory(audit_directory, description="formal audit directory")
    layout_dir = _require_directory(layout_directory, description="LayoutShadow directory")
    selection, sources, source_identities, selection_bindings = _validate_selection(
        selection_dir, audit_dir
    )
    missing_sets, audit_bindings = _validate_audit(audit_dir, selection, sources)
    rows, layout_bindings = _validate_layout(
        layout_dir,
        selection_dir,
        selection,
        sources,
        source_identities,
        missing_sets,
    )
    evidence_bytes = _serialize_jsonl(rows)
    evidence_identity = {
        "relative_path": "evidence.jsonl",
        "sha256": _sha256(evidence_bytes),
        "size_bytes": len(evidence_bytes),
        "records": len(rows),
    }
    summary = {
        "schema_version": 1,
        "kind": EVIDENCE_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "records": len(rows),
        "errors": 0,
        "execution_provider": "cpu",
        "analysis_contract": {
            "time": {
                "semantic_scope": "visible_screen_status_bar_clock_only",
                "full_line_ascii_or_fullwidth_colon_clock": True,
                "transaction_time_labels_accepted": False,
                "seconds_datetime_or_embedded_clock_accepted": False,
                "reverse_clock_repair_applied": False,
                "source_and_rectified_full_image_top_fraction": STATUS_BAR_FRACTION,
                "full_image_top_strip_is_physical_screen_claim": False,
            },
            "payment_method_field": {
                "labels": list(PAYMENT_LABELS),
                "strict_fixed_values": sorted(PAYMENT_FIXED_VALUES),
                "strict_bank_card_tail4": True,
                "value_before_label_accepted": False,
                "geometry_is_diagnostic_only": True,
            },
            "transfer_status": {
                "audited_long_phrase_count": len(STATUS_PHRASE_CLASS),
                "short_bare_status_phrases_accepted": False,
                "success_blocker_scan": "same_visible_cjk_stream",
                "requires_independent_tight_crop_confirmation": True,
                "success_acceptance_enabled": False,
            },
        },
        "status_safety": {
            "field_candidate_writes": 0,
            "success_acceptances": 0,
            "non_success_to_success": 0,
        },
        "coverage_by_field": {
            field: _coverage(rows, field, missing_sets[field]) for field in FIELD_SPECS
        },
        "input_evidence": {
            **selection_bindings,
            **audit_bindings,
            **layout_bindings,
            "source_closure_sha256": _source_closure(source_identities),
        },
        "artifacts": {"evidence_jsonl": evidence_identity},
    }
    bindings = {
        **selection_bindings,
        **audit_bindings,
        **{key: layout_bindings[key] for key in ("layout_summary", "layout_records")},
    }
    _assert_bindings_current(bindings)
    return summary, evidence_bytes, bindings


def _assert_bindings_current(bindings: Mapping[str, Any]) -> None:
    for name in (
        "selection", "input_list", "audit_summary", "audit_findings",
        "layout_summary", "layout_records",
    ):
        expected = bindings.get(name)
        if not isinstance(expected, Mapping):
            raise EvidenceError(f"missing binding {name}")
        observed = _identity(Path(str(expected["path"])))
        if observed != expected:
            raise EvidenceError(f"bound input changed while analysis was being published: {name}")
    sources = bindings.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_RECORDS:
        raise EvidenceError("source bindings are incomplete")
    for expected in sources:
        observed = _identity(Path(str(expected["path"])))
        if observed != expected:
            raise EvidenceError(f"source image changed while analysis was being published: {expected['path']}")


def write_atomic(
    output_directory: Path,
    *,
    summary: Mapping[str, Any],
    evidence_bytes: bytes,
    bindings: Mapping[str, Any],
) -> None:
    output = output_directory.absolute()
    if not output.name or os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite layout evidence output: {output}")
    if summary.get("diagnostic_only") is not True or summary.get("formal_delivery_gate") is not False \
            or summary.get("candidate_write_enabled") is not False:
        raise EvidenceError("layout evidence safety flags are invalid")
    artifact = summary.get("artifacts", {}).get("evidence_jsonl") \
        if isinstance(summary.get("artifacts"), Mapping) else None
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != _sha256(evidence_bytes) \
            or artifact.get("size_bytes") != len(evidence_bytes):
        raise EvidenceError("layout evidence artifact identity differs")
    _assert_bindings_current(bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        (stage / "evidence.jsonl").write_bytes(evidence_bytes)
        (stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _assert_bindings_current(bindings)
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite layout evidence output: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-directory", required=True, type=Path)
    parser.add_argument("--audit-directory", required=True, type=Path)
    parser.add_argument("--layout-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary, evidence_bytes, bindings = prepare_analysis(
            selection_directory=args.selection_directory,
            audit_directory=args.audit_directory,
            layout_directory=args.layout_directory,
        )
        write_atomic(
            args.output_directory,
            summary=summary,
            evidence_bytes=evidence_bytes,
            bindings=bindings,
        )
    except (EvidenceError, FileExistsError, OSError, UnicodeError) as error:
        print(f"Layout shadow evidence failed: {error}")
        return 2
    print(
        f"layout_shadow_evidence records={summary['records']} "
        f"time_target_unique={summary['coverage_by_field']['time']['audit_missing_target']['records_with_unique_diagnostic_coverage']} "
        f"output={args.output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
