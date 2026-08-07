#!/usr/bin/env python3
"""Compare teacher-path and current ML.NET geometry on two detector misses.

This CPU-only diagnostic runs the same detector artifact twice in Python:
once with the teacher pipeline's portrait-orientation rule and once with the
current ML.NET pipeline's forced zero-degree orientation.  That separates an
orientation/rectification mismatch from an ONNX Runtime or resize mismatch.
It writes a new evidence directory and never changes formal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transfer_receipt_ai.geometry import (  # noqa: E402
    RectificationOptions,
    bbox_to_polygon,
    load_upright_rgb,
    rectify_receipt,
    save_rgb,
    transform_points,
)
from transfer_receipt_ai.onnx_runtime import (  # noqa: E402
    OnnxLRCNNPredictor,
    prepare_detector_input,
)


FORMAL_TAG = "20260806-165128"
CASES = (
    {
        "field": "amount",
        "source": Path(r"D:\download\TempFakeImages\s3_voucher_GWCZ2072762506148974592_20260703032240.jpg"),
        "token": "2072762506148974592",
    },
    {
        "field": "payment_method_field",
        "source": Path(r"D:\download\TempFakeImages\s3_voucher_GWCZ2072894140638695424_20260703120459.jpg"),
        "token": "2072894140638695424",
    },
)


class ParityError(RuntimeError):
    """Raised when diagnostic evidence is missing or inconsistent."""


def _windows_path(value: object) -> str:
    return str(value).replace("/", "\\").rstrip("\\").casefold()


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{description} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ParityError(f"{description} is not finite")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ParityError(f"missing {description}: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exception:
        raise ParityError(f"invalid {description} {path}: {exception.msg}") from exception
    if not isinstance(payload, dict):
        raise ParityError(f"invalid {description} {path}: expected one JSON object")
    return payload


def _matching_comparison(path: Path, *, token: str, field: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    if not path.is_file():
        raise ParityError(f"missing formal comparisons: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if token not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exception:
                raise ParityError(f"invalid comparisons {path}:{line_number}: {exception.msg}") from exception
            if isinstance(row, dict) and row.get("field") == field:
                matches.append(row)
    if len(matches) != 1:
        raise ParityError(f"expected one comparison for {field}/{token}, found {len(matches)}")
    return matches[0]


def _latest_mlnet_report(validation_root: Path) -> Path:
    candidates = [
        path
        for path in validation_root.glob("mlnet-missing-detector-pilot-*/report.json")
        if path.is_file()
    ]
    if not candidates:
        raise ParityError(f"no completed ML.NET detector pilot report under {validation_root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def _mlnet_observation(report: Mapping[str, Any], *, field: str, source: Path) -> Mapping[str, Any]:
    zero = report.get("zero_threshold")
    observations = zero.get("observations") if isinstance(zero, Mapping) else None
    if not isinstance(observations, list):
        raise ParityError("ML.NET detector pilot has no zero-threshold observations")
    expected_source = _windows_path(source)
    matches = [
        item
        for item in observations
        if isinstance(item, Mapping)
        and item.get("field") == field
        and _windows_path(item.get("source", "")) == expected_source
    ]
    if len(matches) != 1:
        raise ParityError(f"expected one ML.NET zero-threshold observation for {field}, found {len(matches)}")
    return matches[0]


def _validate_pilot_run(
    payload: object,
    *,
    name: str,
    expected_threshold: float,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ParityError(f"ML.NET detector pilot has no {name} run")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ParityError(f"ML.NET detector pilot {name} has no summary")
    expected_summary = {
        "requested_device": "cpu",
        "unified_provider": "cpu",
        "input": 2,
        "written": 2,
        "skipped": 0,
        "errors": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ParityError(
                f"ML.NET detector pilot {name} summary {key}={summary.get(key)!r}; "
                f"expected {expected!r}"
            )
    observations = payload.get("observations")
    if not isinstance(observations, list) or len(observations) != len(CASES):
        raise ParityError(f"ML.NET detector pilot {name} does not contain two observations")
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ParityError(f"ML.NET detector pilot {name} has an invalid observation")
        threshold = _finite_number(observation.get("threshold"), f"{name} observation threshold")
        if threshold != expected_threshold:
            raise ParityError(
                f"ML.NET detector pilot {name} threshold={threshold}; expected {expected_threshold}"
            )
    return payload


def _validate_mlnet_result(
    observation: Mapping[str, Any],
    *,
    model_hash: str,
    source: Path,
) -> tuple[dict[str, Any], Path]:
    result_path_raw = observation.get("result")
    if not isinstance(result_path_raw, str) or not result_path_raw:
        raise ParityError(f"ML.NET observation has no result path for {source}")
    result_path = Path(result_path_raw)
    result = _load_json(result_path, "ML.NET detector result")
    if _windows_path(result.get("source", "")) != _windows_path(source):
        raise ParityError(f"ML.NET detector result source differs from parity source: {source}")

    contracts = result.get("model_contracts")
    if not isinstance(contracts, Mapping):
        raise ParityError(f"ML.NET detector result omitted model contracts: {result_path}")
    detector_hash = contracts.get("detector_sha256")
    if not isinstance(detector_hash, str) or detector_hash.casefold() != model_hash:
        raise ParityError(f"ML.NET detector result used a different detector: {result_path}")
    for key in ("device_sha256", "unified_ocr_model_sha256"):
        if not isinstance(contracts.get(key), str) or not contracts[key]:
            raise ParityError(f"ML.NET detector result omitted {key}: {result_path}")

    geometry = result.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ParityError(f"ML.NET detector result omitted geometry: {result_path}")
    if geometry.get("rectification") != "max-side-1600":
        raise ParityError(f"ML.NET detector result used different rectification: {result_path}")
    if geometry.get("rotation_degrees") != 0 or geometry.get("screen_detected") is not False:
        raise ParityError(f"ML.NET detector result is not current zero-degree full-image geometry: {result_path}")
    return result, result_path


def _source_bbox(detection: object, rectification: object) -> list[float] | None:
    if detection is None:
        return None
    polygon = bbox_to_polygon(detection.bbox_xyxy)
    projected = transform_points(polygon, rectification.rectified_to_original)
    source_height, source_width = rectification.source_rgb.shape[:2]
    return [
        float(np.clip(projected[:, 0].min(), 0.0, source_width)),
        float(np.clip(projected[:, 1].min(), 0.0, source_height)),
        float(np.clip(projected[:, 0].max(), 0.0, source_width)),
        float(np.clip(projected[:, 1].max(), 0.0, source_height)),
    ]


def _bbox_max_abs_delta(first: object, second: object) -> float | None:
    if not isinstance(first, list) or not isinstance(second, list) or len(first) != 4 or len(second) != 4:
        return None
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        return None
    return float(np.abs(first_values - second_values).max())


def run(*, data_root: Path, model: Path, output: Path | None = None) -> tuple[dict[str, Any], Path]:
    validation_root = data_root / "delivery-validation"
    comparisons_path = validation_root / f"mlnet-wide1536-cpu-full-e2e-{FORMAL_TAG}" / "comparisons.jsonl"
    model = model.resolve()
    contract_path = model.with_suffix(".contract.json")
    contract = _load_json(contract_path, "detector contract")
    if contract.get("kind") != "receipt_lrcnn_v1":
        raise ParityError(f"unexpected detector contract kind: {contract.get('kind')}")
    detector_input = contract.get("input")
    if not isinstance(detector_input, Mapping):
        raise ParityError("detector contract omitted input metadata")
    if detector_input.get("layout") != "CHW" or detector_input.get("shape") != [3, 1536, 864]:
        raise ParityError(
            "detector contract must use the production CHW [3,1536,864] input; "
            f"found layout={detector_input.get('layout')!r} shape={detector_input.get('shape')!r}"
        )
    expected_model_hash = (
        contract.get("onnx", {}).get("sha256")
        if isinstance(contract.get("onnx"), Mapping)
        else None
    )
    model_hash = _sha256(model)
    if not isinstance(expected_model_hash, str) or expected_model_hash.casefold() != model_hash:
        raise ParityError("detector model SHA-256 differs from its contract")

    mlnet_report_path = _latest_mlnet_report(validation_root)
    mlnet_report = _load_json(mlnet_report_path, "ML.NET detector pilot report")
    if mlnet_report.get("kind") != "receipt_mlnet_missing_detector_pilot_v1":
        raise ParityError(f"unexpected ML.NET detector pilot kind: {mlnet_report.get('kind')}")
    if mlnet_report.get("formal_tag") != FORMAL_TAG:
        raise ParityError("ML.NET detector pilot is not bound to the requested formal tag")
    expected_formal_evaluation = validation_root / f"mlnet-wide1536-cpu-full-e2e-{FORMAL_TAG}"
    if _windows_path(mlnet_report.get("formal_evaluation", "")) != _windows_path(expected_formal_evaluation):
        raise ParityError("ML.NET detector pilot is not bound to the fixed formal evaluation")
    if mlnet_report.get("runtime") != "cpu":
        raise ParityError("ML.NET detector pilot is not CPU-only")
    if mlnet_report.get("rectification") != "max-side-1600":
        raise ParityError("ML.NET detector pilot used different rectification")
    if mlnet_report.get("includes_device_model") is not True:
        raise ParityError("ML.NET detector pilot omitted the device model")
    _validate_pilot_run(mlnet_report.get("baseline"), name="baseline", expected_threshold=0.50)
    _validate_pilot_run(mlnet_report.get("zero_threshold"), name="zero_threshold", expected_threshold=0.0)

    predictor = OnnxLRCNNPredictor(
        model,
        device="cpu",
        score_threshold=0.0,
        resize_mode="letterbox",
    )
    if predictor.providers != ["CPUExecutionProvider"]:
        raise ParityError(f"Python ONNX provider is not CPU-only: {predictor.providers}")
    if predictor.input_width != 864 or predictor.input_height != 1536:
        raise ParityError(
            "Python ONNX detector input differs from production 864x1536; "
            f"found {predictor.input_width}x{predictor.input_height}"
        )

    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = output or validation_root / f"mlnet-detector-python-parity-{tag}"
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, case in enumerate(CASES, start=1):
        field = str(case["field"])
        source = Path(case["source"])
        token = str(case["token"])
        if not source.is_file():
            raise ParityError(f"missing parity source: {source}")
        comparison = _matching_comparison(comparisons_path, token=token, field=field)
        if _windows_path(comparison.get("source", "")) != _windows_path(source):
            raise ParityError(f"formal comparison source differs from parity source: {source}")
        reference = comparison.get("reference_text")
        if not isinstance(reference, str) or not reference:
            raise ParityError(f"formal comparison has no reference text for {field}")

        mlnet = _mlnet_observation(mlnet_report, field=field, source=source)
        mlnet_result, mlnet_result_path = _validate_mlnet_result(
            mlnet,
            model_hash=model_hash,
            source=source,
        )
        mlnet_score_raw = mlnet.get("detection_score")
        mlnet_score = (
            float(mlnet_score_raw)
            if isinstance(mlnet_score_raw, (int, float)) and not isinstance(mlnet_score_raw, bool)
            else None
        )
        if mlnet_score is None or not math.isfinite(mlnet_score):
            raise ParityError(f"ML.NET zero-threshold observation has no finite detector score for {field}")

        source_rgb = load_upright_rgb(source)
        python_runs: dict[str, dict[str, Any]] = {}
        for geometry_name, options in (
            (
                "teacher_portrait",
                RectificationOptions(
                    orientation_degrees=None,
                    prefer_portrait=True,
                    auto_screen=False,
                    max_side=1600,
                ),
            ),
            (
                "current_mlnet_forced_zero",
                RectificationOptions(
                    orientation_degrees=0,
                    prefer_portrait=False,
                    auto_screen=False,
                    max_side=1600,
                ),
            ),
        ):
            rectification = rectify_receipt(source_rgb, options)
            rectified_path = output / f"case-{index}-{field}-{geometry_name}-rectified.png"
            save_rgb(rectified_path, rectification.rectified_rgb)
            tensor, mapping = prepare_detector_input(
                rectification.rectified_rgb,
                input_width=predictor.input_width,
                input_height=predictor.input_height,
                resize_mode="letterbox",
            )
            detections = predictor.predict(rectification.rectified_rgb)
            detection = next((item for item in detections if item.label == field), None)
            detection_source_bbox = _source_bbox(detection, rectification)
            python_runs[geometry_name] = {
                "score": None if detection is None else float(detection.score),
                "bbox_rectified": None if detection is None else list(detection.bbox_xyxy),
                "bbox_source": detection_source_bbox,
                "all_detections": [
                    {
                        "label": item.label,
                        "score": float(item.score),
                        "bbox_rectified": list(item.bbox_xyxy),
                        "bbox_source": _source_bbox(item, rectification),
                    }
                    for item in detections
                ],
                "geometry": {
                    **rectification.manifest(),
                    "detector_canvas": {"width": predictor.input_width, "height": predictor.input_height},
                    "resize_mode": "letterbox",
                    "scale_x": mapping.scale_x,
                    "scale_y": mapping.scale_y,
                    "offset_x": mapping.offset_x,
                    "offset_y": mapping.offset_y,
                },
                "detector_tensor_sha256": hashlib.sha256(
                    np.ascontiguousarray(tensor).tobytes()
                ).hexdigest(),
                "rectified": str(rectified_path),
            }

        teacher = python_runs["teacher_portrait"]
        forced_zero = python_runs["current_mlnet_forced_zero"]
        teacher_score = teacher["score"]
        forced_zero_score = forced_zero["score"]
        reference_score_raw = comparison.get("reference_detector_score")
        reference_score = _finite_number(reference_score_raw, f"formal reference detector score for {field}")
        teacher_minus_reference = (
            teacher_score - reference_score if teacher_score is not None else None
        )
        forced_zero_minus_mlnet = (
            forced_zero_score - mlnet_score if forced_zero_score is not None else None
        )
        for label, value in (
            ("teacher/reference score delta", teacher_minus_reference),
            ("forced-zero/ML.NET score delta", forced_zero_minus_mlnet),
        ):
            if value is not None and not math.isfinite(value):
                raise ParityError(f"non-finite {label} for {field}")

        forced_geometry = forced_zero["geometry"]
        mlnet_geometry = mlnet_result["geometry"]
        for size_key in ("source_size", "rectified_size"):
            if forced_geometry.get(size_key) != mlnet_geometry.get(size_key):
                raise ParityError(f"forced-zero Python {size_key} differs from ML.NET for {field}")
        if forced_geometry.get("rotation_degrees") != mlnet_geometry.get("rotation_degrees"):
            raise ParityError(f"forced-zero Python rotation differs from ML.NET for {field}")
        if forced_geometry.get("screen_detected") != mlnet_geometry.get("screen_detected"):
            raise ParityError(f"forced-zero Python screen flag differs from ML.NET for {field}")

        bbox_delta = _bbox_max_abs_delta(forced_zero.get("bbox_source"), mlnet.get("detection_bbox"))
        record = {
            "field": field,
            "source": str(source),
            "source_sha256": _sha256(source),
            "reference": reference,
            "reference_detector_score": reference_score,
            "reference_crop_sha256": comparison.get("reference_crop_sha256"),
            "python": python_runs,
            "mlnet": dict(mlnet),
            "mlnet_result": str(mlnet_result_path),
            "comparison": {
                "teacher_minus_reference_score": teacher_minus_reference,
                "forced_zero_minus_mlnet_score": forced_zero_minus_mlnet,
                "forced_zero_source_bbox_minus_mlnet_max_abs": bbox_delta,
                "bbox_coordinate_system": "EXIF-upright source pixels",
            },
        }
        records.append(record)
        print(
            f"field={field} teacher_score={teacher_score} forced0_score={forced_zero_score} "
            f"mlnet_score={mlnet_score} reference_score={reference_score} "
            f"teacher_reference_delta={teacher_minus_reference} forced0_mlnet_delta={forced_zero_minus_mlnet}"
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "receipt_mlnet_detector_python_parity_v2",
        "formal_tag": FORMAL_TAG,
        "runtime": "cpu",
        "providers": predictor.providers,
        "model": str(model),
        "model_sha256": model_hash,
        "model_contract": str(contract_path),
        "mlnet_report": str(mlnet_report_path),
        "output": str(output),
        "records": records,
    }
    report_path = output / "report.json"
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return report, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\alipay-ai-data"))
    parser.add_argument("--model", type=Path, default=REPOSITORY_ROOT / "artifacts" / "receipt_lrcnn_v1.onnx")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _, report_path = run(data_root=args.data_root, model=args.model, output=args.output)
    except (ParityError, OSError, ValueError, ImportError) as exception:
        print(f"parity_error={exception}")
        return 2
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
