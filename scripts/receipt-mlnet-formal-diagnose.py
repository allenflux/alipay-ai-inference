#!/usr/bin/env python3
"""Print a compact, read-only diagnosis for one completed CPU formal run.

The production PowerShell wrapper is intentionally fail-closed, so a scorer
failure can leave a complete inference directory without publishing a package.
This helper reads that evidence in place.  It never rewrites results, changes
acceptance floors, or creates a delivery directory.
"""

from __future__ import annotations

import argparse
import json
import math
import ntpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FIELD_RESULT_KEYS = {
    "amount": "amount",
    "time": "time",
    "payment_method_field": "payment_method",
    "recipient_field": "recipient",
}


class DiagnosisError(RuntimeError):
    """Raised when completed-run evidence is missing or malformed."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosisError(f"missing {description}: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exception:
        raise DiagnosisError(f"invalid {description} {path}: {exception.msg}") from exception
    if not isinstance(payload, dict):
        raise DiagnosisError(f"invalid {description} {path}: expected one JSON object")
    return payload


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DiagnosisError(f"missing {description}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exception:
                raise DiagnosisError(
                    f"invalid {description} {path}:{line_number}: {exception.msg}"
                ) from exception
            if not isinstance(value, dict):
                raise DiagnosisError(
                    f"invalid {description} {path}:{line_number}: expected one JSON object"
                )
            rows.append(value)
    return rows


def _normalise_windows_path(value: object) -> str:
    return ntpath.normcase(ntpath.normpath(str(value).replace("/", "\\")))


def _render(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def diagnose(*, data_root: Path, tag: str) -> list[str]:
    output = data_root / "delivery-validation" / f"mlnet-wide1536-cpu-full-{tag}"
    evaluation = data_root / "delivery-validation" / f"mlnet-wide1536-cpu-full-e2e-{tag}"
    delivery = data_root / "delivery" / f"ReceiptMlNet-wide1536-cpu-production-{tag}"
    runtime = _load_json(output / "inference_summary.json", "inference summary")
    score = _load_json(evaluation / "summary.json", "evaluation summary")
    comparisons = _load_jsonl(evaluation / "comparisons.jsonl", "comparisons")

    lines = [
        "receipt_mlnet_formal_diagnose_v1",
        f"tag={tag}",
        f"output={output}",
        f"evaluation={evaluation}",
        f"delivery={delivery}",
        f"delivery_exists={delivery.is_dir()}",
        (
            "runtime "
            f"requested_device={runtime.get('requested_device')} "
            f"unified_provider={runtime.get('unified_provider')} "
            f"input={runtime.get('input')} written={runtime.get('written')} "
            f"skipped={runtime.get('skipped')} errors={runtime.get('errors')}"
        ),
    ]
    latency = _mapping(runtime.get("inference_latency_ms"))
    lines.append(
        "latency_ms "
        f"mean={latency.get('mean')} p50={latency.get('p50')} p95={latency.get('p95')} "
        f"total_seconds={runtime.get('total_seconds')}"
    )
    lines.append(f"score accepted={score.get('accepted')} kind={score.get('kind')}")

    by_field = _mapping(score.get("by_field"))
    floors = _mapping(score.get("floors"))
    missing_root = _mapping(score.get("missing"))
    missing_by_field = _mapping(missing_root.get("field_candidates"))
    for field in FIELD_RESULT_KEYS:
        metric = _mapping(by_field.get(field))
        rate = _finite_number(metric.get("raw_exact_match"))
        floor = _finite_number(floors.get(field))
        coverage = _finite_number(metric.get("candidate_coverage"))
        missing = _mapping(missing_by_field.get(field))
        lines.append(
            f"field={field} matches={metric.get('raw_exact_matches')}/{metric.get('records')} "
            f"rate={rate} floor={floor} rate_pass={rate is not None and floor is not None and rate >= floor} "
            f"candidates={metric.get('candidate_records')} coverage={coverage} "
            f"missing={missing.get('records')} coverage_pass={coverage == 1.0}"
        )

    failures = score.get("failures")
    if not isinstance(failures, list):
        raise DiagnosisError("evaluation summary failures is not an array")
    for failure in failures:
        lines.append(f"failure={failure}")

    comparison_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for comparison in comparisons:
        field = comparison.get("field")
        source = comparison.get("source")
        if isinstance(field, str) and isinstance(source, str):
            comparison_index[(field, _normalise_windows_path(source))] = comparison

    for field in FIELD_RESULT_KEYS:
        missing = _mapping(missing_by_field.get(field))
        sources = missing.get("sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            key = (field, _normalise_windows_path(source))
            comparison = comparison_index.get(key)
            if comparison is None:
                lines.append(f"missing_detail field={field} source={source} comparison=not_found")
                continue
            result_path_value = comparison.get("result_json")
            result_path = Path(str(result_path_value)) if isinstance(result_path_value, str) else None
            result_field: Mapping[str, Any] = {}
            matching_detection: Mapping[str, Any] = {}
            result_geometry: Mapping[str, Any] = {}
            if result_path is not None and result_path.is_file():
                result = _load_json(result_path, "result JSON")
                result_field = _mapping(_mapping(result.get("fields")).get(FIELD_RESULT_KEYS[field]))
                result_geometry = _mapping(result.get("geometry"))
                detections = result.get("detections")
                if isinstance(detections, list):
                    matching_detection = next(
                        (
                            detection
                            for detection in detections
                            if isinstance(detection, Mapping) and detection.get("label") == field
                        ),
                        {},
                    )
            teacher_path_value = comparison.get("teacher_result_json")
            teacher_path = Path(str(teacher_path_value)) if isinstance(teacher_path_value, str) else None
            teacher_geometry: Mapping[str, Any] = {}
            teacher_detection: Mapping[str, Any] = {}
            if teacher_path is not None and teacher_path.is_file():
                teacher = _load_json(teacher_path, "teacher result JSON")
                teacher_geometry = _mapping(teacher.get("geometry"))
                teacher_detections = teacher.get("detections")
                if isinstance(teacher_detections, list):
                    teacher_detection = next(
                        (
                            detection
                            for detection in teacher_detections
                            if isinstance(detection, Mapping) and detection.get("label") == field
                        ),
                        {},
                    )
            lines.append(
                f"missing_detail field={field} source={source} "
                f"reference={_render(comparison.get('reference_text'))} "
                f"reason={comparison.get('missing_reason')} result={result_path_value} "
                f"field_state={result_field.get('state')} candidate={_render(result_field.get('candidate'))} "
                f"ctc={_render(result_field.get('ctc_candidate'))} "
                f"structured={_render(result_field.get('structured_candidate'))} "
                f"detector_score={matching_detection.get('score')} "
                f"bbox={_render(matching_detection.get('bbox_image'))}"
            )
            lines.append(
                f"missing_geometry field={field} "
                f"reference_detector_score={comparison.get('reference_detector_score')} "
                f"reference_bbox_rectified={_render(comparison.get('reference_bbox_rectified'))} "
                f"teacher_detection_score={teacher_detection.get('score')} "
                f"teacher_detection_bbox={_render(teacher_detection.get('bbox_image'))} "
                f"teacher_geometry={_render(teacher_geometry)} "
                f"result_geometry={_render(result_geometry)}"
            )
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="formal run tag, for example 20260806-165128")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\alipay-ai-data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for line in diagnose(data_root=args.data_root, tag=args.tag):
            print(line)
    except (DiagnosisError, OSError) as exception:
        print(f"diagnosis_error={exception}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
