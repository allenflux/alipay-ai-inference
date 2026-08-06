#!/usr/bin/env python3
"""Audit ML.NET recipient text and detector geometry against teacher evidence.

This is intentionally a standard-library-only diagnostic.  It consumes the
``comparisons.jsonl`` emitted by ``receipt_mlnet_unified_evaluate.py`` and the
``teacher_result_json`` referenced by each recipient comparison.  The report
is diagnostic evidence only; it does not alter or relax the delivery gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RECIPIENT_FIELD = "recipient_field"
FIELD_ORDER = ("amount", "time", "payment_method_field", "recipient_field")


class RecipientAuditInputError(ValueError):
    """Raised when audit evidence is malformed or unusable."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exception:
        raise RecipientAuditInputError(f"Cannot read JSON {path}: {exception}") from exception
    if not isinstance(payload, Mapping):
        raise RecipientAuditInputError(f"{path}: JSON root must be an object")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exception:
                    raise RecipientAuditInputError(
                        f"{path}:{line_number}: invalid JSON: {exception}"
                    ) from exception
                if not isinstance(row, Mapping):
                    raise RecipientAuditInputError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeError) as exception:
        raise RecipientAuditInputError(f"Cannot read JSONL {path}: {exception}") from exception
    return rows


def _normalise_nfkc_trim(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _levenshtein(left: str, right: str) -> int:
    """Return Unicode-code-point Levenshtein distance using O(min(m, n)) memory."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    bbox = [_finite_number(component) for component in value]
    if any(component is None for component in bbox):
        return None
    result = [float(component) for component in bbox if component is not None]
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _valid_homography(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            return None
        parsed = [_finite_number(component) for component in row]
        if any(component is None for component in parsed):
            return None
        matrix.append([float(component) for component in parsed if component is not None])
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    return matrix


def _project_bbox(bbox: Sequence[float], homography: Sequence[Sequence[float]]) -> list[float]:
    """Project all four bbox corners and return their rectified axis-aligned bounds."""

    corners = (
        (float(bbox[0]), float(bbox[1])),
        (float(bbox[2]), float(bbox[1])),
        (float(bbox[2]), float(bbox[3])),
        (float(bbox[0]), float(bbox[3])),
    )
    projected: list[tuple[float, float]] = []
    for x_coordinate, y_coordinate in corners:
        denominator = (
            homography[2][0] * x_coordinate
            + homography[2][1] * y_coordinate
            + homography[2][2]
        )
        if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
            raise RecipientAuditInputError("bbox corner projects to infinity")
        projected_x = (
            homography[0][0] * x_coordinate
            + homography[0][1] * y_coordinate
            + homography[0][2]
        ) / denominator
        projected_y = (
            homography[1][0] * x_coordinate
            + homography[1][1] * y_coordinate
            + homography[1][2]
        ) / denominator
        if not math.isfinite(projected_x) or not math.isfinite(projected_y):
            raise RecipientAuditInputError("bbox corner projection is non-finite")
        projected.append((projected_x, projected_y))
    x_coordinates = [point[0] for point in projected]
    y_coordinates = [point[1] for point in projected]
    result = [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise RecipientAuditInputError("projected bbox has no positive area")
    return result


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "records": len(finite_values),
        "min": min(finite_values) if finite_values else None,
        "p05": _percentile(finite_values, 0.05),
        "p50": _percentile(finite_values, 0.50),
        "p95": _percentile(finite_values, 0.95),
        "max": max(finite_values) if finite_values else None,
        "mean": statistics.fmean(finite_values) if finite_values else None,
    }


def _edit_distance_summary(distances: Sequence[int], reference_characters: int) -> dict[str, Any]:
    counts = Counter(int(distance) for distance in distances)
    bins = {
        "0": counts.get(0, 0),
        "1": counts.get(1, 0),
        "2": counts.get(2, 0),
        "3": counts.get(3, 0),
        "4": counts.get(4, 0),
        "5+": sum(count for distance, count in counts.items() if distance >= 5),
    }
    total = sum(distances)
    return {
        "records": len(distances),
        "total_edits": total,
        "reference_characters": reference_characters,
        "micro_cer": total / reference_characters if reference_characters else None,
        "mean_edits_per_record": statistics.fmean(distances) if distances else None,
        "max_edits": max(distances) if distances else None,
        "distribution": {str(distance): counts[distance] for distance in sorted(counts)},
        "bins": bins,
    }


def _size_key(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "missing"
    width = _finite_number(value.get("width"))
    height = _finite_number(value.get("height"))
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        return "invalid"

    def render(number: float) -> str:
        return str(int(number)) if number.is_integer() else format(number, ".6g")

    return f"{render(width)}x{render(height)}"


def _new_geometry_accumulator() -> dict[str, Any]:
    return {
        "records": 0,
        "teacher_result_records": 0,
        "geometry_records": 0,
        "rotation_degrees": Counter(),
        "screen_detected": Counter(),
        "source_size": Counter(),
        "rectified_size": Counter(),
    }


def _update_geometry_accumulator(
    accumulator: dict[str, Any], teacher_payload: Mapping[str, Any] | None
) -> None:
    accumulator["records"] += 1
    if teacher_payload is None:
        accumulator["rotation_degrees"]["missing"] += 1
        accumulator["screen_detected"]["missing"] += 1
        accumulator["source_size"]["missing"] += 1
        accumulator["rectified_size"]["missing"] += 1
        return
    accumulator["teacher_result_records"] += 1
    geometry = teacher_payload.get("geometry")
    if not isinstance(geometry, Mapping):
        accumulator["rotation_degrees"]["missing"] += 1
        accumulator["screen_detected"]["missing"] += 1
        accumulator["source_size"]["missing"] += 1
        accumulator["rectified_size"]["missing"] += 1
        return
    accumulator["geometry_records"] += 1
    rotation = geometry.get("rotation_degrees")
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)) or not math.isfinite(float(rotation)):
        rotation_key = "missing" if rotation is None else "invalid"
    elif float(rotation).is_integer():
        rotation_key = str(int(rotation))
    else:
        rotation_key = "invalid"
    screen_detected = geometry.get("screen_detected")
    if isinstance(screen_detected, bool):
        screen_key = str(screen_detected).lower()
    else:
        screen_key = "missing" if screen_detected is None else "invalid"
    accumulator["rotation_degrees"][rotation_key] += 1
    accumulator["screen_detected"][screen_key] += 1
    accumulator["source_size"][_size_key(geometry.get("source_size"))] += 1
    accumulator["rectified_size"][_size_key(geometry.get("rectified_size"))] += 1


def _finalise_geometry_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    records = int(accumulator["records"])
    geometry_records = int(accumulator["geometry_records"])
    return {
        "records": records,
        "teacher_result_records": int(accumulator["teacher_result_records"]),
        "geometry_records": geometry_records,
        "geometry_coverage": geometry_records / records if records else None,
        "rotation_degrees": dict(sorted(accumulator["rotation_degrees"].items())),
        "screen_detected": dict(sorted(accumulator["screen_detected"].items())),
        "source_size": dict(sorted(accumulator["source_size"].items())),
        "rectified_size": dict(sorted(accumulator["rectified_size"].items())),
    }


def _new_alignment_accumulator() -> dict[str, Any]:
    return {
        "records": 0,
        "missing_by_reason": Counter(),
        "ious": [],
        "record_edge_mae": [],
        "signed_edges": {edge: [] for edge in ("left", "top", "right", "bottom")},
        "absolute_edges": {edge: [] for edge in ("left", "top", "right", "bottom")},
        "samples": [],
    }


def _alignment_missing(accumulator: dict[str, Any], reason: str) -> None:
    accumulator["missing_by_reason"][reason] += 1


def _add_alignment(
    accumulator: dict[str, Any],
    *,
    iou: float,
    signed_edges: Mapping[str, float],
    sample: Mapping[str, Any],
) -> None:
    accumulator["ious"].append(iou)
    absolute_values: list[float] = []
    for edge, value in signed_edges.items():
        accumulator["signed_edges"][edge].append(value)
        absolute = abs(value)
        accumulator["absolute_edges"][edge].append(absolute)
        absolute_values.append(absolute)
    accumulator["record_edge_mae"].append(statistics.fmean(absolute_values))
    accumulator["samples"].append(dict(sample))


def _finalise_alignment_accumulator(
    accumulator: Mapping[str, Any], *, worst_limit: int
) -> dict[str, Any]:
    ious = accumulator["ious"]
    records = int(accumulator["records"])
    available = len(ious)
    all_absolute_edges = [
        value for values in accumulator["absolute_edges"].values() for value in values
    ]
    worst_samples = sorted(
        accumulator["samples"], key=lambda sample: (float(sample["iou"]), str(sample.get("id", "")))
    )[:worst_limit]
    return {
        "records": records,
        "available_records": available,
        "coverage": available / records if records else None,
        "missing_records": records - available,
        "missing_by_reason": dict(sorted(accumulator["missing_by_reason"].items())),
        "iou": _numeric_summary(ious),
        "mean_absolute_edge_deviation_per_record_px": _numeric_summary(
            accumulator["record_edge_mae"]
        ),
        "absolute_edge_deviation_px": _numeric_summary(all_absolute_edges),
        "signed_edge_deviation_px": {
            edge: _numeric_summary(values)
            for edge, values in accumulator["signed_edges"].items()
        },
        "absolute_edge_deviation_by_edge_px": {
            edge: _numeric_summary(values)
            for edge, values in accumulator["absolute_edges"].items()
        },
        "worst_iou_samples": worst_samples,
    }


def _find_teacher_bbox(teacher_payload: Mapping[str, Any]) -> list[float] | None:
    detections = teacher_payload.get("detections")
    if not isinstance(detections, list):
        return None
    for detection in detections:
        if not isinstance(detection, Mapping) or detection.get("label") != RECIPIENT_FIELD:
            continue
        return _valid_bbox(detection.get("bbox_rectified"))
    return None


def _resolve_teacher_path(raw_path: str, comparisons_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    beside_comparisons = comparisons_path.parent / path
    if beside_comparisons.is_file():
        return beside_comparisons
    return path


def _load_teacher_cached(
    raw_path: Any,
    *,
    comparisons_path: Path,
    cache: dict[str, tuple[Mapping[str, Any] | None, str | None, str | None]],
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "teacher_result_path_missing", None
    path = _resolve_teacher_path(raw_path, comparisons_path)
    cache_key = str(path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            result = (None, "teacher_result_root_not_object", path.as_posix())
        else:
            result = (payload, None, path.as_posix())
    except FileNotFoundError:
        result = (None, "teacher_result_not_found", path.as_posix())
    except PermissionError:
        result = (None, "teacher_result_permission_denied", path.as_posix())
    except UnicodeError:
        result = (None, "teacher_result_encoding_error", path.as_posix())
    except json.JSONDecodeError:
        result = (None, "teacher_result_invalid_json", path.as_posix())
    except OSError:
        result = (None, "teacher_result_io_error", path.as_posix())
    cache[cache_key] = result
    return result


def _evaluation_snapshot(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None or not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    by_field = summary.get("by_field")
    fields: dict[str, Any] = {}
    for field in FIELD_ORDER:
        metrics = by_field.get(field) if isinstance(by_field, Mapping) else None
        if not isinstance(metrics, Mapping):
            continue
        fields[field] = {
            "raw_exact_matches": metrics.get("raw_exact_matches"),
            "records": metrics.get("records"),
            "raw_exact_match": metrics.get("raw_exact_match"),
            "candidate_records": metrics.get("candidate_records"),
            "candidate_coverage": metrics.get("candidate_coverage"),
        }
    amount_semantic = summary.get("amount_semantic")
    return {
        "summary": summary_path.as_posix(),
        "kind": summary.get("kind"),
        "formal_delivery_gate": summary.get("formal_delivery_gate"),
        "pilot_thresholds_passed": summary.get("pilot_thresholds_passed"),
        "accepted": summary.get("accepted"),
        "by_field": fields,
        "amount_semantic": dict(amount_semantic) if isinstance(amount_semantic, Mapping) else None,
    }


def build_recipient_audit(
    *,
    comparisons_path: Path,
    summary_path: Path | None = None,
    worst_limit: int = 20,
) -> dict[str, Any]:
    if worst_limit < 0:
        raise RecipientAuditInputError("worst_limit must be non-negative")
    rows = [row for row in _load_jsonl(comparisons_path) if row.get("field") == RECIPIENT_FIELD]
    if not rows:
        raise RecipientAuditInputError(
            f"{comparisons_path}: no {RECIPIENT_FIELD!r} comparisons"
        )

    strict_matches = 0
    normalised_matches = 0
    candidate_records = 0
    raw_distances: list[int] = []
    normalised_distances: list[int] = []
    raw_reference_characters = 0
    normalised_reference_characters = 0
    teacher_errors: Counter[str] = Counter()
    teacher_cache: dict[str, tuple[Mapping[str, Any] | None, str | None, str | None]] = {}
    geometry_accumulators = {
        "all": _new_geometry_accumulator(),
        "strict_exact": _new_geometry_accumulator(),
        "strict_mismatch": _new_geometry_accumulator(),
    }
    alignment_accumulators = {
        "all": _new_alignment_accumulator(),
        "strict_exact": _new_alignment_accumulator(),
        "strict_mismatch": _new_alignment_accumulator(),
    }

    for row_index, row in enumerate(rows, 1):
        reference = row.get("reference_text")
        if not isinstance(reference, str):
            raise RecipientAuditInputError(
                f"{comparisons_path}: recipient row {row_index} has no string reference_text"
            )
        candidate_value = row.get("candidate_text")
        candidate = candidate_value if isinstance(candidate_value, str) else ""
        candidate_present = isinstance(candidate_value, str)
        if candidate_present:
            candidate_records += 1
        strict_exact = candidate_present and candidate == reference
        normalised_reference = _normalise_nfkc_trim(reference)
        normalised_candidate = _normalise_nfkc_trim(candidate)
        normalised_exact = candidate_present and normalised_candidate == normalised_reference
        strict_matches += int(strict_exact)
        normalised_matches += int(normalised_exact)
        raw_distances.append(_levenshtein(reference, candidate))
        normalised_distances.append(_levenshtein(normalised_reference, normalised_candidate))
        raw_reference_characters += len(reference)
        normalised_reference_characters += len(normalised_reference)

        classification = "strict_exact" if strict_exact else "strict_mismatch"
        group_names = ("all", classification)
        teacher_payload, teacher_error, teacher_path = _load_teacher_cached(
            row.get("teacher_result_json"),
            comparisons_path=comparisons_path,
            cache=teacher_cache,
        )
        if teacher_error is not None:
            teacher_errors[teacher_error] += 1
        for group_name in group_names:
            _update_geometry_accumulator(geometry_accumulators[group_name], teacher_payload)
            alignment_accumulators[group_name]["records"] += 1

        current_bbox = _valid_bbox(row.get("detection_bbox_image"))
        if current_bbox is None:
            missing_reason = "current_detection_bbox_missing_or_invalid"
        elif teacher_payload is None:
            missing_reason = teacher_error or "teacher_result_unavailable"
        else:
            teacher_geometry = teacher_payload.get("geometry")
            if not isinstance(teacher_geometry, Mapping):
                missing_reason = "teacher_geometry_missing"
            else:
                homography = _valid_homography(teacher_geometry.get("H_original_to_rectified"))
                if homography is None:
                    missing_reason = "teacher_homography_missing_or_invalid"
                else:
                    teacher_bbox = _find_teacher_bbox(teacher_payload)
                    if teacher_bbox is None:
                        missing_reason = "teacher_recipient_bbox_missing_or_invalid"
                    else:
                        try:
                            projected_bbox = _project_bbox(current_bbox, homography)
                        except RecipientAuditInputError:
                            missing_reason = "current_bbox_projection_invalid"
                        else:
                            missing_reason = ""
        if missing_reason:
            for group_name in group_names:
                _alignment_missing(alignment_accumulators[group_name], missing_reason)
            continue

        edge_names = ("left", "top", "right", "bottom")
        signed_edges = {
            edge: projected_bbox[index] - teacher_bbox[index]
            for index, edge in enumerate(edge_names)
        }
        iou = _bbox_iou(projected_bbox, teacher_bbox)
        sample = {
            "id": row.get("id"),
            "source": row.get("source"),
            "reference_text": reference,
            "candidate_text": candidate_value,
            "strict_exact": strict_exact,
            "nfkc_trim_exact": normalised_exact,
            "teacher_result_json": teacher_path,
            "detection_bbox_image": current_bbox,
            "projected_bbox_rectified": projected_bbox,
            "teacher_bbox_rectified": teacher_bbox,
            "iou": iou,
            "signed_edge_deviation_px": signed_edges,
            "absolute_edge_deviation_px": {
                edge: abs(value) for edge, value in signed_edges.items()
            },
        }
        for group_name in group_names:
            _add_alignment(
                alignment_accumulators[group_name],
                iou=iou,
                signed_edges=signed_edges,
                sample=sample,
            )

    records = len(rows)
    resolved_summary = summary_path
    if resolved_summary is None:
        candidate_summary = comparisons_path.parent / "summary.json"
        resolved_summary = candidate_summary if candidate_summary.is_file() else None
    return {
        "schema_version": 1,
        "kind": "receipt_mlnet_recipient_audit_v1",
        "comparisons": comparisons_path.as_posix(),
        "field": RECIPIENT_FIELD,
        "diagnostic_only": True,
        "evaluation_snapshot": _evaluation_snapshot(resolved_summary),
        "text": {
            "records": records,
            "candidate_records": candidate_records,
            "candidate_coverage": candidate_records / records,
            "strict_exact_matches": strict_matches,
            "strict_exact_match": strict_matches / records,
            "nfkc_trim_exact_matches": normalised_matches,
            "nfkc_trim_exact_match": normalised_matches / records,
            "nfkc_trim_recovered_matches": normalised_matches - strict_matches,
            "normalization": "Unicode NFKC followed by surrounding-whitespace trim",
            "raw_edit_distance": _edit_distance_summary(
                raw_distances, raw_reference_characters
            ),
            "nfkc_trim_edit_distance": _edit_distance_summary(
                normalised_distances, normalised_reference_characters
            ),
        },
        "teacher_result_errors": dict(sorted(teacher_errors.items())),
        "teacher_geometry": {
            group_name: _finalise_geometry_accumulator(accumulator)
            for group_name, accumulator in geometry_accumulators.items()
        },
        "bbox_alignment": {
            "method": (
                "project the four detection_bbox_image corners with teacher "
                "H_original_to_rectified; compare their axis-aligned bounds with teacher "
                "recipient_field bbox_rectified"
            ),
            "edge_sign": "projected_current_minus_teacher in rectified pixels",
            **{
                group_name: _finalise_alignment_accumulator(
                    accumulator, worst_limit=worst_limit
                )
                for group_name, accumulator in alignment_accumulators.items()
            },
        },
    }


def _format_rate(value: Any) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number:.2%}"


def _format_number(value: Any, digits: int = 3) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _compact_counts(value: Any, *, limit: int = 5) -> str:
    if not isinstance(value, Mapping):
        return "{}"
    items = sorted(value.items(), key=lambda item: (-int(item[1]), str(item[0])))
    rendered = ", ".join(f"{key}:{count}" for key, count in items[:limit])
    remaining = len(items) - limit
    if remaining > 0:
        rendered = f"{rendered}, +{remaining} keys"
    return "{" + rendered + "}"


def format_recipient_audit(report: Mapping[str, Any]) -> str:
    lines = ["mlnet_cpu_evaluation_snapshot"]
    snapshot = report.get("evaluation_snapshot")
    if isinstance(snapshot, Mapping):
        by_field = snapshot.get("by_field")
        for field in FIELD_ORDER:
            metrics = by_field.get(field) if isinstance(by_field, Mapping) else None
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"  {field}={metrics.get('raw_exact_matches')}/{metrics.get('records')}="
                f"{_format_rate(metrics.get('raw_exact_match'))}"
            )
        amount_semantic = snapshot.get("amount_semantic")
        if isinstance(amount_semantic, Mapping):
            lines.append(
                "  amount_semantic="
                f"{amount_semantic.get('exact_matches')}/{amount_semantic.get('records')}="
                f"{_format_rate(amount_semantic.get('exact_match'))}; "
                f"diagnostic_only={amount_semantic.get('diagnostic_only')}; "
                f"affects_acceptance={amount_semantic.get('affects_acceptance')}"
            )
        lines.append(
            "  formal_delivery_gate="
            f"{snapshot.get('formal_delivery_gate')}; "
            f"pilot_thresholds_passed={snapshot.get('pilot_thresholds_passed')}"
        )
    else:
        lines.append("  unavailable (summary.json not found)")

    text = report["text"]
    lines.extend(
        [
            "recipient_text_audit",
            (
                f"  strict={text['strict_exact_matches']}/{text['records']}="
                f"{_format_rate(text['strict_exact_match'])}; "
                f"nfkc_trim={text['nfkc_trim_exact_matches']}/{text['records']}="
                f"{_format_rate(text['nfkc_trim_exact_match'])}; "
                f"recovered={text['nfkc_trim_recovered_matches']}; "
                f"coverage={_format_rate(text['candidate_coverage'])}"
            ),
            (
                "  raw_edit_bins="
                f"{_compact_counts(text['raw_edit_distance']['bins'], limit=6)}; "
                f"mean={_format_number(text['raw_edit_distance']['mean_edits_per_record'])}; "
                f"micro_cer={_format_rate(text['raw_edit_distance']['micro_cer'])}"
            ),
            (
                "  nfkc_trim_edit_bins="
                f"{_compact_counts(text['nfkc_trim_edit_distance']['bins'], limit=6)}; "
                f"mean={_format_number(text['nfkc_trim_edit_distance']['mean_edits_per_record'])}; "
                f"micro_cer={_format_rate(text['nfkc_trim_edit_distance']['micro_cer'])}"
            ),
        ]
    )

    geometry = report["teacher_geometry"]["all"]
    lines.extend(
        [
            "teacher_geometry",
            (
                f"  available={geometry['geometry_records']}/{geometry['records']}="
                f"{_format_rate(geometry['geometry_coverage'])}; "
                f"rotation={_compact_counts(geometry['rotation_degrees'])}; "
                f"screen_detected={_compact_counts(geometry['screen_detected'])}"
            ),
            (
                f"  source_size={_compact_counts(geometry['source_size'])}; "
                f"rectified_size={_compact_counts(geometry['rectified_size'])}"
            ),
        ]
    )

    lines.append("recipient_bbox_alignment")
    for group_name in ("all", "strict_exact", "strict_mismatch"):
        metrics = report["bbox_alignment"][group_name]
        lines.append(
            f"  {group_name}: available={metrics['available_records']}/{metrics['records']}="
            f"{_format_rate(metrics['coverage'])}; "
            f"iou_mean={_format_number(metrics['iou']['mean'])}; "
            f"iou_p05={_format_number(metrics['iou']['p05'])}; "
            f"iou_p50={_format_number(metrics['iou']['p50'])}; "
            "edge_mae_mean_px="
            f"{_format_number(metrics['mean_absolute_edge_deviation_per_record_px']['mean'])}; "
            f"missing={metrics['missing_by_reason']}"
        )
    if report.get("teacher_result_errors"):
        lines.append(f"teacher_result_errors={report['teacher_result_errors']}")
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="scorer evaluation directory containing comparisons.jsonl and optional summary.json",
    )
    parser.add_argument(
        "--comparisons",
        type=Path,
        help="override comparisons.jsonl path",
    )
    parser.add_argument("--summary", type=Path, help="override summary.json path")
    parser.add_argument("--output", type=Path, help="defaults to EVALUATION-DIR/recipient-audit.json")
    parser.add_argument("--worst-limit", type=int, default=20)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    evaluation_dir: Path = args.evaluation_dir
    comparisons_path = args.comparisons or evaluation_dir / "comparisons.jsonl"
    summary_path = args.summary
    if summary_path is None:
        candidate_summary = evaluation_dir / "summary.json"
        summary_path = candidate_summary if candidate_summary.is_file() else None
    output_path = args.output or evaluation_dir / "recipient-audit.json"
    report = build_recipient_audit(
        comparisons_path=comparisons_path,
        summary_path=summary_path,
        worst_limit=args.worst_limit,
    )
    _atomic_write_json(output_path, report)
    print(format_recipient_audit(report))
    print(f"recipient_audit_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
