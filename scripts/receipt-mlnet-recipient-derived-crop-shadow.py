#!/usr/bin/env python3
"""Prepare and evaluate diagnostic-only fourth/fifth recipient crop shadows.

``prepare`` consumes the frozen formal recipient diagnostic plus the strict
truth-free v4 probe.  It selects only remaining records whose detector,
ordinary-geometry and alternative-envelope gates are all clear, then freezes
two deterministic rectified-space crop plans.  It never runs OCR and never
writes a field candidate.

``evaluate`` consumes separately produced CPU PP-OCR layout evidence for those
crops.  Crop 4 may form exact consensus with one existing crop; otherwise
crop 4 and crop 5 must contain the same unique strict line.  The result remains
a shadow candidate only and is never a formal delivery gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


FORMAL_FAILURES = 204
V4_CANDIDATES = 75
V4_REMAINING = 129
V4_GLOBAL_GATE_FAILED_REMAINING = 66
V4_GLOBAL_GATE_CLEAR_REMAINING = 63
ATTEMPTS = ("first", "retry", "right_value")
CROP4 = "crop4_interrow_value_corridor"
CROP5 = "crop5_recipient_value_core"

DIAGNOSTIC_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_diagnostic_summary_v1"
DIAGNOSTIC_FINDING_KIND = "receipt_mlnet_hybrid_failure_diagnostic_finding_v1"
TRUTH_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
TRUTH_FINDING_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
PLAN_SUMMARY_KIND = "receipt_mlnet_recipient_derived_crop_plan_summary_v1"
PLAN_RECORD_KIND = "receipt_mlnet_recipient_derived_crop_plan_record_v1"
LAYOUT_SUMMARY_KIND = "receipt_mlnet_recipient_derived_crop_layout_summary_v1"
LAYOUT_RECORD_KIND = "receipt_mlnet_recipient_derived_crop_layout_record_v1"
EVALUATION_SUMMARY_KIND = "receipt_mlnet_recipient_derived_crop_shadow_summary_v1"
EVALUATION_RECORD_KIND = "receipt_mlnet_recipient_derived_crop_shadow_record_v1"


class ShadowError(ValueError):
    """Raised when frozen diagnostic or derived-layout evidence is unsafe."""


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ShadowError(f"missing {description}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ShadowError(f"{description} must be an object")
    return payload


def _load_jsonl(path: Path, *, description: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as error:
        raise ShadowError(f"missing {description}: {path}") from error
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ShadowError(f"{description}[{index}] must be an object")
        rows.append(payload)
    return rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ShadowError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise ShadowError(f"{description} is not a file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ShadowError(f"{description} must be non-empty: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": size,
    }


def _assert_identities(identities: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in identities.items():
        actual = _identity(Path(str(expected.get("path") or "")), description=name)
        if actual != dict(expected):
            raise ShadowError(f"{name} changed while derived-crop shadow was reading it")


def _source_key(value: object) -> str:
    return os.path.normpath(str(value or "").replace("\\", "/")).casefold()


def _nonempty(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShadowError(f"{description} must be a non-empty string")
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShadowError(f"{description} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowError(f"{description} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ShadowError(f"{description} must be finite")
    return result


def _box(value: object, *, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ShadowError(f"{description} must contain four coordinates")
    result = [_finite(item, description=description) for item in value]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ShadowError(f"{description} has non-positive area")
    return result


def _rows_by_source(
    rows: Sequence[Mapping[str, Any]], *, kind: str, description: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != 1 or row.get("kind") != kind:
            raise ShadowError(f"{description}[{index}] has unsupported schema/kind")
        source = _nonempty(row.get("source"), description=f"{description}[{index}].source")
        key = _source_key(source)
        if key in result:
            raise ShadowError(f"duplicate {description} source: {source!r}")
        result[key] = row
    return result


def _count_map(rows: object, *, description: str) -> dict[str, int]:
    if not isinstance(rows, list):
        raise ShadowError(f"{description} must be an array")
    output: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ShadowError(f"{description}[{index}] must be an object")
        name = _nonempty(row.get("name"), description=f"{description}[{index}].name")
        records = _integer(
            row.get("records"), description=f"{description}[{index}].records"
        )
        if name in output:
            raise ShadowError(f"duplicate {description} name: {name}")
        output[name] = records
    return output


def _truth_probe_script() -> Path:
    return Path(__file__).resolve().with_name(
        "receipt-mlnet-hybrid-failure-truth-probe.py"
    )


def _load_filter_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "receipt_mlnet_hybrid_failure_truth_probe_for_derived_crop", path
    )
    if spec is None or spec.loader is None:
        raise ShadowError(f"cannot load strict truth-probe filter contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_frozen_inputs(
    diagnostic: Path, truth_probe: Path
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]],
]:
    paths = {
        "diagnostic_summary": diagnostic / "summary.json",
        "diagnostic_findings": diagnostic / "findings.jsonl",
        "truth_probe_summary": truth_probe / "summary.json",
        "truth_probe_findings": truth_probe / "findings.jsonl",
        "truth_probe_filter_script": _truth_probe_script(),
    }
    identities = {
        name: _identity(path, description=name) for name, path in paths.items()
    }
    diagnostic_summary = _load_json(paths["diagnostic_summary"], description="diagnostic summary")
    truth_summary = _load_json(paths["truth_probe_summary"], description="truth-probe summary")
    if (
        diagnostic_summary.get("schema_version") != 1
        or diagnostic_summary.get("kind") != DIAGNOSTIC_SUMMARY_KIND
        or diagnostic_summary.get("comparison_evaluation_mode") != "formal"
        or diagnostic_summary.get("failed_records") != FORMAL_FAILURES
        or diagnostic_summary.get("recipient_missing_only_records") != FORMAL_FAILURES
    ):
        raise ShadowError("diagnostic summary is not the frozen formal 204 missing-only evidence")
    if truth_summary.get("schema_version") != 1 or truth_summary.get("kind") != TRUTH_SUMMARY_KIND:
        raise ShadowError("truth-probe summary has unsupported schema/kind")
    teacher = truth_summary.get("paddle_teacher_consensus")
    remaining = truth_summary.get("remaining_failure_analysis")
    overlay = truth_summary.get("remaining_global_gate_overlay_analysis")
    if not isinstance(teacher, Mapping) or not isinstance(remaining, Mapping) or not isinstance(overlay, Mapping):
        raise ShadowError("truth-probe summary lacks v4 teacher/remaining/gate evidence")
    expected_states = {
        "ambiguous": 15,
        "candidate": 75,
        "rejected_by_global_gate": 30,
        "unresolved": 84,
    }
    if (
        teacher.get("records") != V4_CANDIDATES
        or _count_map(teacher.get("by_state"), description="teacher.by_state") != expected_states
        or remaining.get("records") != V4_REMAINING
        or remaining.get("strict_candidate_records") != V4_CANDIDATES
        or overlay.get("records") != V4_REMAINING
        or overlay.get("any_global_gate_failure_records") != V4_GLOBAL_GATE_FAILED_REMAINING
        or overlay.get("clear_global_gate_records") != V4_GLOBAL_GATE_CLEAR_REMAINING
    ):
        raise ShadowError("truth-probe summary does not match frozen v4 75/129 and 66/63 contract")

    diagnostic_rows = _load_jsonl(paths["diagnostic_findings"], description="diagnostic findings")
    truth_rows = _load_jsonl(paths["truth_probe_findings"], description="truth-probe findings")
    if len(diagnostic_rows) != FORMAL_FAILURES or len(truth_rows) != FORMAL_FAILURES:
        raise ShadowError("diagnostic and truth-probe findings must each contain exactly 204 rows")
    diagnostic_by_source = _rows_by_source(
        diagnostic_rows, kind=DIAGNOSTIC_FINDING_KIND, description="diagnostic findings"
    )
    truth_by_source = _rows_by_source(
        truth_rows, kind=TRUTH_FINDING_KIND, description="truth-probe findings"
    )
    if set(diagnostic_by_source) != set(truth_by_source):
        raise ShadowError("diagnostic and truth-probe source sets differ")
    _assert_identities(identities)
    return diagnostic_by_source, truth_by_source, identities


def _derive_crops(geometry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    width = _integer(geometry.get("rectified_width"), description="rectified_width", minimum=2)
    height = _integer(geometry.get("rectified_height"), description="rectified_height", minimum=2)
    amount = _box(geometry.get("amount_box"), description="rectified amount_box")
    recipient = _box(geometry.get("recipient_box"), description="rectified recipient_box")
    payment = _box(geometry.get("payment_box"), description="rectified payment_box")
    for name, value in (("amount", amount), ("recipient", recipient), ("payment", payment)):
        if value[0] < -1 or value[1] < -1 or value[2] > width + 1 or value[3] > height + 1:
            raise ShadowError(f"{name} box escapes rectified bounds")

    recipient_width = recipient[2] - recipient[0]
    recipient_height = recipient[3] - recipient[1]
    amount_center_y = (amount[1] + amount[3]) * 0.5
    recipient_center_x = (recipient[0] + recipient[2]) * 0.5
    recipient_center_y = (recipient[1] + recipient[3]) * 0.5
    payment_center_y = (payment[1] + payment[3]) * 0.5
    if not amount_center_y < recipient_center_y < payment_center_y:
        raise ShadowError("derived crops require amount < recipient < payment centers")
    upper_midpoint = (amount_center_y + recipient_center_y) * 0.5
    lower_midpoint = (recipient_center_y + payment_center_y) * 0.5
    right = min(float(width), recipient[2] + 0.08 * recipient_width)

    def rectangle(name: str, left: float, top: float, bottom: float) -> dict[str, Any]:
        pixel_box = [
            max(0, math.floor(left)),
            max(0, math.floor(top)),
            min(width, math.ceil(right)),
            min(height, math.ceil(bottom)),
        ]
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise ShadowError(f"derived {name} has non-positive area")
        if not (
            pixel_box[0] <= recipient_center_x < pixel_box[2]
            and pixel_box[1] <= recipient_center_y < pixel_box[3]
        ):
            raise ShadowError(f"derived {name} does not contain recipient center")
        if pixel_box[1] <= amount_center_y < pixel_box[3]:
            raise ShadowError(f"derived {name} contains amount-row center")
        if pixel_box[1] <= payment_center_y < pixel_box[3]:
            raise ShadowError(f"derived {name} contains payment-row center")
        return {
            "name": name,
            "rectified_box": pixel_box,
            "width": pixel_box[2] - pixel_box[0],
            "height": pixel_box[3] - pixel_box[1],
            "pixel_box_semantics": "left_top_inclusive_right_bottom_exclusive",
        }

    crop4 = rectangle(
        CROP4,
        max(width * 0.28, recipient[0] + 0.20 * recipient_width),
        max(0.0, upper_midpoint, recipient[1] - 0.35 * recipient_height),
        min(float(height), lower_midpoint, recipient[3] + 0.35 * recipient_height),
    )
    crop5 = rectangle(
        CROP5,
        max(width * 0.36, recipient[0] + 0.32 * recipient_width),
        max(0.0, upper_midpoint, recipient[1] - 0.08 * recipient_height),
        min(float(height), lower_midpoint, recipient[3] + 0.08 * recipient_height),
    )
    minimum_left_delta = max(2, math.floor(width * 0.04))
    if crop5["rectified_box"][0] - crop4["rectified_box"][0] < minimum_left_delta:
        raise ShadowError("derived crop4/crop5 do not have independent horizontal context")
    if crop4["rectified_box"] == crop5["rectified_box"]:
        raise ShadowError("derived crop4/crop5 must not be identical")
    return crop4, crop5


def _canonical_plan_id(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(rendered)


def _build_plans(
    diagnostic_by_source: Mapping[str, Mapping[str, Any]],
    truth_by_source: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    remaining = [
        row for row in truth_by_source.values() if row.get("remaining_failure_cluster") is not None
    ]
    gate_clear = []
    gate_failed = 0
    for row in remaining:
        shadow = row.get("shadow_candidate_truth_free")
        if not isinstance(shadow, Mapping):
            raise ShadowError("truth-probe finding lacks truth-free shadow")
        failures = shadow.get("global_gate_failures")
        if not isinstance(failures, list) or any(not isinstance(value, str) for value in failures):
            raise ShadowError("truth-probe global_gate_failures must be a string array")
        if failures:
            gate_failed += 1
        else:
            gate_clear.append(row)
    if (
        len(remaining) != V4_REMAINING
        or len(gate_clear) != V4_GLOBAL_GATE_CLEAR_REMAINING
        or gate_failed != V4_GLOBAL_GATE_FAILED_REMAINING
    ):
        raise ShadowError("truth-probe findings do not reproduce v4 remaining 129 / gate 66+63")

    plans: list[dict[str, Any]] = []
    for truth in gate_clear:
        source = _nonempty(truth.get("source"), description="truth-probe source")
        diagnostic = diagnostic_by_source[_source_key(source)]
        if _source_key(diagnostic.get("source")) != _source_key(source):
            raise ShadowError("diagnostic/truth source mismatch")
        geometry = diagnostic.get("geometry_evidence")
        if not isinstance(geometry, Mapping):
            raise ShadowError(f"global-clear source has no rectified geometry: {source}")
        if truth.get("geometry_reasons") != [] or truth.get("alternative_envelope") is not True:
            raise ShadowError(f"global-clear source has inconsistent geometry/envelope: {source}")
        score = _finite(truth.get("recipient_detector_score"), description="recipient detector score")
        if score < 0.68 or score > 1.0:
            raise ShadowError(f"global-clear source violates detector floor: {source}")
        source_identity = _identity(Path(source), description=f"source image {source}")
        crop4, crop5 = _derive_crops(geometry)
        attempts = truth.get("attempts")
        if not isinstance(attempts, Mapping) or set(attempts) != set(ATTEMPTS):
            raise ShadowError(f"truth-probe attempts are incomplete: {source}")
        plan_without_id = {
            "schema_version": 1,
            "kind": PLAN_RECORD_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "source": source,
            "source_image": source_identity,
            "rectification": "max_side_1600",
            "rectified_size": {
                "width": _integer(geometry.get("rectified_width"), description="rectified width", minimum=2),
                "height": _integer(geometry.get("rectified_height"), description="rectified height", minimum=2),
            },
            "detector_geometry": {
                "amount_box": geometry.get("amount_box"),
                "recipient_box": geometry.get("recipient_box"),
                "payment_box": geometry.get("payment_box"),
            },
            "global_gate_evidence": {
                "recipient_detector_score": score,
                "minimum_recipient_detector_score": 0.68,
                "ordinary_25pct_geometry_verified": True,
                "alternative_envelope_verified": True,
                "global_gate_failures": [],
            },
            "existing_attempts": {name: attempts[name] for name in ATTEMPTS},
            "crops": [crop4, crop5],
        }
        plans.append({**plan_without_id, "plan_id": _canonical_plan_id(plan_without_id)})
    return plans


def _write_atomic_directory(output: Path, files: Mapping[str, bytes]) -> None:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite derived-crop shadow output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        for name, payload in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite derived-crop shadow output: {output}")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare(diagnostic: Path, truth_probe: Path, output: Path) -> None:
    diagnostic_by_source, truth_by_source, identities = _validate_frozen_inputs(
        diagnostic.resolve(), truth_probe.resolve()
    )
    plans = _build_plans(diagnostic_by_source, truth_by_source)
    if len(plans) != V4_GLOBAL_GATE_CLEAR_REMAINING:
        raise ShadowError("internal derived-crop plan count differs from 63")
    plans_bytes = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        for row in plans
    ).encode("utf-8")
    inputs_bytes = "".join(f"{row['source']}\n" for row in plans).encode("utf-8")
    summary = {
        "schema_version": 1,
        "kind": PLAN_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "ocr_rerun": False,
        "production_output_changed": False,
        "frozen_v4": {
            "formal_failures": FORMAL_FAILURES,
            "candidate_records": V4_CANDIDATES,
            "remaining_records": V4_REMAINING,
            "remaining_with_global_gate_failures": V4_GLOBAL_GATE_FAILED_REMAINING,
            "remaining_with_clear_global_gates": V4_GLOBAL_GATE_CLEAR_REMAINING,
        },
        "records": len(plans),
        "crop_names": [CROP4, CROP5],
        "route_contract": {
            "crop4_requires_exact_match_with_existing_strict_crop": True,
            "crop5_requires_unique_exact_crop4_crop5_agreement": True,
            "minimum_line_confidence": 0.80,
            "minimum_recipient_detector_score": 0.68,
            "requires_ordinary_25pct_geometry": True,
            "requires_alternative_envelope": True,
            "candidate_write_enabled": False,
        },
        "required_layout_producer": {
            "api": "PaddleOcrEngine.RecognizeLayoutDiagnostic",
            "execution_provider": "cpu",
            "rectification": "max_side_1600",
            "requires_raw_quad_crop_and_rectified_coordinates": True,
            "requires_verified_paddle_bundle_identity": True,
            "required_summary_kind": LAYOUT_SUMMARY_KIND,
            "required_record_kind": LAYOUT_RECORD_KIND,
        },
        "filter_contract": identities["truth_probe_filter_script"],
        "source_evidence": identities,
        "artifacts": {
            "plans": {
                "path": "plans.jsonl",
                "sha256": _sha256_bytes(plans_bytes),
                "size_bytes": len(plans_bytes),
                "records": len(plans),
            },
            "inputs": {
                "path": "inputs.txt",
                "sha256": _sha256_bytes(inputs_bytes),
                "size_bytes": len(inputs_bytes),
                "records": len(plans),
            },
        },
    }
    _assert_identities(identities)
    summary_bytes = (json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    _write_atomic_directory(
        output,
        {"summary.json": summary_bytes, "plans.jsonl": plans_bytes, "inputs.txt": inputs_bytes},
    )


def _verify_artifact(directory: Path, record: Mapping[str, Any], *, description: str) -> Path:
    relative = _nonempty(record.get("path"), description=f"{description}.path")
    path = (directory / relative).resolve(strict=True)
    try:
        path.relative_to(directory.resolve(strict=True))
    except ValueError as error:
        raise ShadowError(f"{description} escapes its directory") from error
    actual = _identity(path, description=description)
    if actual["sha256"] != record.get("sha256") or actual["size_bytes"] != record.get("size_bytes"):
        raise ShadowError(f"{description} identity differs from summary")
    return path


def _load_plan(plan_directory: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    summary_path = plan_directory / "summary.json"
    summary = _load_json(summary_path, description="derived-crop plan summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != PLAN_SUMMARY_KIND
        or summary.get("diagnostic_only") is not True
        or summary.get("formal_delivery_gate") is not False
        or summary.get("candidate_write_enabled") is not False
        or summary.get("records") != V4_GLOBAL_GATE_CLEAR_REMAINING
    ):
        raise ShadowError("derived-crop plan summary violates the diagnostic-only 63-record contract")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("plans"), Mapping):
        raise ShadowError("derived-crop plan summary has no plans artifact")
    plans_path = _verify_artifact(plan_directory, artifacts["plans"], description="plans artifact")
    plans = _load_jsonl(plans_path, description="derived-crop plans")
    if len(plans) != V4_GLOBAL_GATE_CLEAR_REMAINING:
        raise ShadowError("derived-crop plan artifact must contain exactly 63 rows")
    by_source = _rows_by_source(plans, kind=PLAN_RECORD_KIND, description="derived-crop plans")
    for source, plan in by_source.items():
        if (
            plan.get("diagnostic_only") is not True
            or plan.get("formal_delivery_gate") is not False
            or plan.get("candidate_write_enabled") is not False
        ):
            raise ShadowError(f"derived-crop plan {source!r} is not diagnostic-only")
        if any(
            forbidden in plan
            for forbidden in ("candidate", "shadow_candidate", "delivery_value", "fields")
        ):
            raise ShadowError(f"derived-crop plan {source!r} contains a production-value field")
        gates = plan.get("global_gate_evidence")
        score = (
            _finite(
                gates.get("recipient_detector_score"),
                description="plan recipient detector score",
            )
            if isinstance(gates, Mapping)
            else float("nan")
        )
        if (
            not isinstance(gates, Mapping)
            or gates.get("global_gate_failures") != []
            or gates.get("ordinary_25pct_geometry_verified") is not True
            or gates.get("alternative_envelope_verified") is not True
            or gates.get("minimum_recipient_detector_score") != 0.68
            or score < 0.68
            or score > 1.0
        ):
            raise ShadowError(f"derived-crop plan {source!r} does not preserve all global gates")
        attempts = plan.get("existing_attempts")
        if not isinstance(attempts, Mapping) or set(attempts) != set(ATTEMPTS):
            raise ShadowError(f"derived-crop plan {source!r} has invalid existing attempts")
        for attempt_name in ATTEMPTS:
            attempt = attempts[attempt_name]
            lines = attempt.get("lines") if isinstance(attempt, Mapping) else None
            if not isinstance(lines, list):
                raise ShadowError(f"derived-crop plan {source!r} {attempt_name} lines are invalid")
            for line_index, line in enumerate(lines):
                if (
                    not isinstance(line, Mapping)
                    or line.get("index") != line_index
                    or not isinstance(line.get("text"), str)
                ):
                    raise ShadowError(
                        f"derived-crop plan {source!r} {attempt_name} line schema is invalid"
                    )
                confidence = _finite(
                    line.get("confidence"), description="existing line confidence"
                )
                if confidence < 0 or confidence > 1:
                    raise ShadowError("existing line confidence must be within [0,1]")
        rectified_size = plan.get("rectified_size")
        detector_geometry = plan.get("detector_geometry")
        if not isinstance(rectified_size, Mapping) or not isinstance(detector_geometry, Mapping):
            raise ShadowError(f"derived-crop plan {source!r} lacks rectified geometry")
        recomputed_crop4, recomputed_crop5 = _derive_crops(
            {
                "rectified_width": rectified_size.get("width"),
                "rectified_height": rectified_size.get("height"),
                **detector_geometry,
            }
        )
        if plan.get("crops") != [recomputed_crop4, recomputed_crop5]:
            raise ShadowError(f"derived-crop plan {source!r} crop geometry is not canonical")
        without_id = {key: value for key, value in plan.items() if key != "plan_id"}
        if plan.get("plan_id") != _canonical_plan_id(without_id):
            raise ShadowError(f"derived-crop plan {source!r} has invalid plan_id")
    return summary, plans, _identity(summary_path, description="plan summary")


def _quad(value: object, *, description: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ShadowError(f"{description} must contain four points")
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ShadowError(f"{description} points must contain x,y")
        points.append([
            _finite(point[0], description=description),
            _finite(point[1], description=description),
        ])
    return points


def _validate_layout_crop(
    crop: Mapping[str, Any], plan_crop: Mapping[str, Any], *, drop_score: float
) -> list[dict[str, Any]]:
    if crop.get("name") != plan_crop.get("name") or crop.get("rectified_box") != plan_crop.get("rectified_box"):
        raise ShadowError("layout crop name/rectified_box differs from frozen plan")
    box = [int(value) for value in plan_crop["rectified_box"]]
    width = box[2] - box[0]
    height = box[3] - box[1]
    if crop.get("width") != width or crop.get("height") != height:
        raise ShadowError("layout crop dimensions differ from frozen plan")
    raw_lines = crop.get("lines")
    if not isinstance(raw_lines, list):
        raise ShadowError("layout crop lines must be an array")
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(raw_lines):
        if not isinstance(line, Mapping) or line.get("index") != index:
            raise ShadowError("layout crop line indices must be contiguous")
        text = line.get("text")
        if not isinstance(text, str):
            raise ShadowError("layout crop line text must be a string")
        confidence = _finite(line.get("confidence"), description="layout line confidence")
        if confidence < 0 or confidence > 1:
            raise ShadowError("layout line confidence must be within [0,1]")
        if line.get("passes_drop_score") is not (confidence >= drop_score):
            raise ShadowError("layout line passes_drop_score differs from bundle floor")
        quad_crop = _quad(line.get("quad_crop"), description="quad_crop")
        quad_rectified = _quad(line.get("quad_rectified"), description="quad_rectified")
        for crop_point, rectified_point in zip(quad_crop, quad_rectified, strict=True):
            if not (-1 <= crop_point[0] <= width + 1 and -1 <= crop_point[1] <= height + 1):
                raise ShadowError("layout line quad escapes crop bounds")
            if not (
                math.isclose(rectified_point[0], crop_point[0] + box[0], abs_tol=1e-4)
                and math.isclose(rectified_point[1], crop_point[1] + box[1], abs_tol=1e-4)
            ):
                raise ShadowError("layout crop/rectified quadrilaterals disagree")
        lines.append(
            {
                "text": text,
                "confidence": confidence,
                "passes_drop_score": bool(line["passes_drop_score"]),
            }
        )
    return lines


def _strict_lines(lines: Iterable[Mapping[str, Any]], filter_module) -> dict[str, float]:
    output: dict[str, float] = {}
    for line in lines:
        text = " ".join(str(line.get("text") or "").split())
        confidence = float(line["confidence"])
        allowed, _ = filter_module._shadow_line_allowed(text)
        if line.get("passes_drop_score") is False or confidence < 0.80 or not allowed:
            continue
        output[text] = max(output.get(text, 0.0), confidence)
    return output


def _existing_strict_lines(plan: Mapping[str, Any], filter_module) -> dict[str, dict[str, float]]:
    by_text: dict[str, dict[str, float]] = defaultdict(dict)
    attempts = plan.get("existing_attempts")
    if not isinstance(attempts, Mapping):
        raise ShadowError("plan has no existing attempts")
    for name in ATTEMPTS:
        attempt = attempts.get(name)
        if not isinstance(attempt, Mapping) or not isinstance(attempt.get("lines"), list):
            raise ShadowError(f"plan existing attempt {name} is invalid")
        strict = _strict_lines(attempt["lines"], filter_module)
        for text, confidence in strict.items():
            by_text[text][name] = confidence
    return by_text


def _evaluate_one(
    plan: Mapping[str, Any], layout: Mapping[str, Any], filter_module, *, drop_score: float
) -> dict[str, Any]:
    source = plan["source"]
    if _identity(Path(source), description=f"evaluation source image {source}") != plan["source_image"]:
        raise ShadowError(f"source image changed after derived-crop planning: {source}")
    if (
        layout.get("schema_version") != 1
        or layout.get("kind") != LAYOUT_RECORD_KIND
        or layout.get("diagnostic_only") is not True
        or layout.get("formal_delivery_gate") is not False
        or layout.get("candidate_write_enabled") is not False
        or layout.get("execution_provider") != "cpu"
        or layout.get("plan_id") != plan.get("plan_id")
        or _source_key(layout.get("source")) != _source_key(source)
        or layout.get("source_image_sha256") != plan["source_image"]["sha256"]
        or layout.get("rectified_size") != plan.get("rectified_size")
    ):
        raise ShadowError(f"layout record is not bound to diagnostic plan: {source}")
    crops = layout.get("crops")
    if not isinstance(crops, list) or len(crops) != 2:
        raise ShadowError("layout record must contain exactly crop4 and crop5")
    by_name = {crop.get("name"): crop for crop in crops if isinstance(crop, Mapping)}
    plan_crops = {crop["name"]: crop for crop in plan["crops"]}
    if set(by_name) != {CROP4, CROP5}:
        raise ShadowError("layout record crop names differ from crop4/crop5 contract")
    crop4_lines = _validate_layout_crop(by_name[CROP4], plan_crops[CROP4], drop_score=drop_score)
    crop5_lines = _validate_layout_crop(by_name[CROP5], plan_crops[CROP5], drop_score=drop_score)
    existing = _existing_strict_lines(plan, filter_module)
    crop4 = _strict_lines(crop4_lines, filter_module)
    crop5 = _strict_lines(crop5_lines, filter_module)

    shadow_candidate: str | None = None
    route: str | None = None
    confidence: float | None = None
    evidence_crops: list[str] = []
    if len(crop4) == 1:
        text, crop4_confidence = next(iter(crop4.items()))
        old_crops = existing.get(text, {})
        if old_crops:
            old_crop, old_confidence = max(
                old_crops.items(), key=lambda item: (item[1], item[0])
            )
            shadow_candidate = text
            route = "derived_crop4_existing_exact_shadow"
            confidence = min(crop4_confidence, old_confidence)
            evidence_crops = [old_crop, CROP4]
    if shadow_candidate is None and len(crop4) == 1 and len(crop5) == 1:
        crop4_text, crop4_confidence = next(iter(crop4.items()))
        crop5_text, crop5_confidence = next(iter(crop5.items()))
        if crop4_text == crop5_text:
            shadow_candidate = crop4_text
            route = "derived_crop4_crop5_exact_shadow"
            confidence = min(crop4_confidence, crop5_confidence)
            evidence_crops = [CROP4, CROP5]
    return {
        "schema_version": 1,
        "kind": EVALUATION_RECORD_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "production_output_changed": False,
        "source": source,
        "plan_id": plan["plan_id"],
        "state": "shadow_candidate" if shadow_candidate is not None else "unresolved",
        "shadow_candidate": shadow_candidate,
        "shadow_route": route,
        "minimum_confidence": confidence,
        "evidence_crops": evidence_crops,
        "strict_existing_lines": [
            {"text": text, "crops": sorted(crops), "crop_confidences": dict(sorted(crops.items()))}
            for text, crops in sorted(existing.items())
        ],
        "strict_crop4_lines": dict(sorted(crop4.items())),
        "strict_crop5_lines": dict(sorted(crop5.items())),
    }


def _validate_bundle_evidence(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ShadowError("layout summary has no verified Paddle bundle evidence")
    output: dict[str, dict[str, Any]] = {}
    for role in ("detector", "classifier", "recognizer", "dictionary"):
        record = value.get(role)
        if not isinstance(record, Mapping):
            raise ShadowError(f"layout Paddle bundle has no {role} identity")
        path = Path(_nonempty(record.get("path"), description=f"Paddle {role} path"))
        expected = {
            "path": path.resolve(strict=True).as_posix(),
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        }
        actual = _identity(path, description=f"Paddle {role}")
        if actual != expected:
            raise ShadowError(f"Paddle {role} identity differs from layout summary")
        output[role] = actual
    return output


def evaluate(plan_directory: Path, layout_directory: Path, output: Path) -> None:
    plan_directory = plan_directory.resolve()
    layout_directory = layout_directory.resolve()
    plan_summary, plans, plan_summary_identity = _load_plan(plan_directory)
    plan_records_identity = _identity(
        plan_directory / str(plan_summary["artifacts"]["plans"]["path"]),
        description="plan records",
    )
    filter_identity = plan_summary.get("filter_contract")
    if not isinstance(filter_identity, Mapping):
        raise ShadowError("plan has no strict filter identity")
    _assert_identities({"truth_probe_filter_script": filter_identity})
    filter_module = _load_filter_module(Path(str(filter_identity["path"])))

    layout_summary_path = layout_directory / "summary.json"
    layout_summary_identity = _identity(
        layout_summary_path, description="layout summary"
    )
    layout_summary = _load_json(layout_summary_path, description="derived-crop layout summary")
    if (
        layout_summary.get("schema_version") != 1
        or layout_summary.get("kind") != LAYOUT_SUMMARY_KIND
        or layout_summary.get("diagnostic_only") is not True
        or layout_summary.get("formal_delivery_gate") is not False
        or layout_summary.get("candidate_write_enabled") is not False
        or layout_summary.get("execution_provider") != "cpu"
        or layout_summary.get("records") != V4_GLOBAL_GATE_CLEAR_REMAINING
        or layout_summary.get("errors") != 0
        or layout_summary.get("rectification") != "max_side_1600"
    ):
        raise ShadowError("layout summary violates CPU diagnostic-only 63-record contract")
    drop_score = _finite(layout_summary.get("paddle_drop_score"), description="Paddle drop score")
    if drop_score < 0 or drop_score > 1:
        raise ShadowError("Paddle drop score must be within [0,1]")
    paddle_bundle = _validate_bundle_evidence(layout_summary.get("paddle_bundle"))
    input_plan = layout_summary.get("input_plan")
    if not isinstance(input_plan, Mapping):
        raise ShadowError("layout summary has no input_plan binding")
    plan_artifact = plan_summary["artifacts"]["plans"]
    if (
        input_plan.get("sha256") != plan_artifact.get("sha256")
        or input_plan.get("size_bytes") != plan_artifact.get("size_bytes")
        or input_plan.get("records") != V4_GLOBAL_GATE_CLEAR_REMAINING
    ):
        raise ShadowError("layout input_plan differs from frozen crop plan")
    artifacts = layout_summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("records"), Mapping):
        raise ShadowError("layout summary has no records artifact")
    records_path = _verify_artifact(layout_directory, artifacts["records"], description="layout records")
    layout_records_identity = _identity(records_path, description="layout records")
    layout_rows = _load_jsonl(records_path, description="layout records")
    if len(layout_rows) != V4_GLOBAL_GATE_CLEAR_REMAINING:
        raise ShadowError("layout records must contain exactly 63 rows")
    plans_by_source = {_source_key(row["source"]): row for row in plans}
    layouts_by_source = _rows_by_source(layout_rows, kind=LAYOUT_RECORD_KIND, description="layout records")
    if set(plans_by_source) != set(layouts_by_source):
        raise ShadowError("layout and crop-plan source sets differ")

    findings = [
        _evaluate_one(
            plans_by_source[_source_key(plan["source"])],
            layouts_by_source[_source_key(plan["source"])],
            filter_module,
            drop_score=drop_score,
        )
        for plan in plans
    ]
    findings_bytes = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        for row in findings
    ).encode("utf-8")
    states = Counter(row["state"] for row in findings)
    routes = Counter(row["shadow_route"] or "none" for row in findings)
    summary = {
        "schema_version": 1,
        "kind": EVALUATION_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "production_output_changed": False,
        "accuracy_claimed": False,
        "truth_used_for_candidate_selection": False,
        "records": len(findings),
        "shadow_candidate_records": states["shadow_candidate"],
        "unresolved_records": states["unresolved"],
        "by_state": [{"name": name, "records": count} for name, count in sorted(states.items())],
        "by_shadow_route": [{"name": name, "records": count} for name, count in sorted(routes.items())],
        "input_plan_summary": plan_summary_identity,
        "input_layout_summary": layout_summary_identity,
        "filter_contract": dict(filter_identity),
        "paddle_bundle": paddle_bundle,
        "artifacts": {
            "findings": {
                "path": "findings.jsonl",
                "sha256": _sha256_bytes(findings_bytes),
                "size_bytes": len(findings_bytes),
                "records": len(findings),
            }
        },
    }
    closing_identities: dict[str, Mapping[str, Any]] = {
        "truth_probe_filter_script": filter_identity,
        "plan_summary": plan_summary_identity,
        "plan_records": plan_records_identity,
        "layout_summary": layout_summary_identity,
        "layout_records": layout_records_identity,
        **{f"paddle_{name}": identity for name, identity in paddle_bundle.items()},
        **{
            f"source_image_{index:03d}": plan["source_image"]
            for index, plan in enumerate(plans)
        },
    }
    _assert_identities(closing_identities)
    summary_bytes = (json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    _write_atomic_directory(output, {"summary.json": summary_bytes, "findings.jsonl": findings_bytes})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="freeze the global-clear crop4/crop5 plans")
    prepare_parser.add_argument("--diagnostic", type=Path, required=True)
    prepare_parser.add_argument("--truth-probe", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate bound CPU layout evidence")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--layout-evidence", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.diagnostic, args.truth_probe, args.output)
    else:
        evaluate(args.plan, args.layout_evidence, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
