#!/usr/bin/env python3
"""Compare Python ONNX detector output with the completed ML.NET miss pilot.

This is a two-image, CPU-only diagnostic.  It uses the same detector artifact,
portrait orientation rule, full-image max-side-1600 rectification and black
letterbox contract as production.  It writes a new evidence directory and
never changes the formal inference or evaluation directories.
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
    load_upright_rgb,
    rectify_receipt,
    save_rgb,
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
    expected_source = str(source).replace("/", "\\").casefold()
    matches = [
        item
        for item in observations
        if isinstance(item, Mapping)
        and item.get("field") == field
        and str(item.get("source", "")).replace("/", "\\").casefold() == expected_source
    ]
    if len(matches) != 1:
        raise ParityError(f"expected one ML.NET zero-threshold observation for {field}, found {len(matches)}")
    return matches[0]


def run(*, data_root: Path, model: Path, output: Path | None = None) -> tuple[dict[str, Any], Path]:
    validation_root = data_root / "delivery-validation"
    comparisons_path = validation_root / f"mlnet-wide1536-cpu-full-e2e-{FORMAL_TAG}" / "comparisons.jsonl"
    model = model.resolve()
    contract_path = model.with_suffix(".contract.json")
    contract = _load_json(contract_path, "detector contract")
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

    predictor = OnnxLRCNNPredictor(
        model,
        device="cpu",
        score_threshold=0.0,
        resize_mode="letterbox",
    )
    if predictor.providers != ["CPUExecutionProvider"]:
        raise ParityError(f"Python ONNX provider is not CPU-only: {predictor.providers}")

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
        if str(comparison.get("source", "")).replace("/", "\\").casefold() != str(source).replace("/", "\\").casefold():
            raise ParityError(f"formal comparison source differs from parity source: {source}")
        reference = comparison.get("reference_text")
        if not isinstance(reference, str) or not reference:
            raise ParityError(f"formal comparison has no reference text for {field}")

        source_rgb = load_upright_rgb(source)
        rectification = rectify_receipt(
            source_rgb,
            RectificationOptions(
                orientation_degrees=None,
                prefer_portrait=True,
                auto_screen=False,
                max_side=1600,
            ),
        )
        rectified_path = output / f"case-{index}-{field}-rectified.png"
        save_rgb(rectified_path, rectification.rectified_rgb)
        tensor, mapping = prepare_detector_input(
            rectification.rectified_rgb,
            input_width=predictor.input_width,
            input_height=predictor.input_height,
            resize_mode="letterbox",
        )
        detections = predictor.predict(rectification.rectified_rgb)
        detection = next((item for item in detections if item.label == field), None)
        mlnet = _mlnet_observation(mlnet_report, field=field, source=source)
        python_score = None if detection is None else float(detection.score)
        mlnet_score_raw = mlnet.get("detection_score")
        mlnet_score = (
            float(mlnet_score_raw)
            if isinstance(mlnet_score_raw, (int, float)) and not isinstance(mlnet_score_raw, bool)
            else None
        )
        score_delta = (
            python_score - mlnet_score
            if python_score is not None and mlnet_score is not None
            else None
        )
        if score_delta is not None and not math.isfinite(score_delta):
            raise ParityError(f"non-finite detector score delta for {field}")
        record = {
            "field": field,
            "source": str(source),
            "source_sha256": _sha256(source),
            "reference": reference,
            "reference_detector_score": comparison.get("reference_detector_score"),
            "reference_crop_sha256": comparison.get("reference_crop_sha256"),
            "python": {
                "score": python_score,
                "bbox_rectified": None if detection is None else list(detection.bbox_xyxy),
                "all_detections": [
                    {"label": item.label, "score": float(item.score), "bbox_rectified": list(item.bbox_xyxy)}
                    for item in detections
                ],
            },
            "mlnet": dict(mlnet),
            "python_minus_mlnet_score": score_delta,
            "geometry": {
                **rectification.manifest(),
                "detector_canvas": {"width": predictor.input_width, "height": predictor.input_height},
                "resize_mode": "letterbox",
                "scale_x": mapping.scale_x,
                "scale_y": mapping.scale_y,
                "offset_x": mapping.offset_x,
                "offset_y": mapping.offset_y,
            },
            "detector_tensor_sha256": hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest(),
            "rectified": str(rectified_path),
        }
        records.append(record)
        print(
            f"field={field} python_score={python_score} mlnet_score={mlnet_score} "
            f"delta={score_delta} reference_score={comparison.get('reference_detector_score')}"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "receipt_mlnet_detector_python_parity_v1",
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
