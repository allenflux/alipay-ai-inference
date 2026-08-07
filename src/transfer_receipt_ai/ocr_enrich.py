"""Paddle-only OCR completion worker for Windows GPU inference.

The public receipt CLI starts this module only after the Torch detector child
has exited.  Keep imports that can reach OpenCV or the detector inside
``run_ocr_enrichment`` so Paddle is the first GPU framework loaded here.
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Any

from .result_semantics import has_current_result_semantics


_STAGE_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete receipt OCR from private detector-stage artifacts")
    parser.add_argument("--stage-output", type=Path, required=True, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="Paddle device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--annotate", choices=("flagged", "all", "none"), default="all",
        help="标注哪些图:all=每张 / flagged=只标缺核心字段的 / none=都不标",
    )
    parser.add_argument("--_quiet", action="store_true", help=argparse.SUPPRESS)
    return parser


def _require_paddleocr_v2() -> None:
    """Reject PaddleOCR 3 before importing its PaddleX/ModelScope dependency chain."""
    try:
        version = metadata.version("paddleocr")
    except metadata.PackageNotFoundError as error:
        raise ImportError(
            "PaddleOCR is not installed. Install PaddlePaddle GPU first, then run "
            "`python -m pip install -r requirements-ocr.txt`."
        ) from error
    if version != "2.10.0":
        raise RuntimeError(
            f"This Windows GPU OCR worker requires paddleocr==2.10.0, but found {version}. "
            "Run `python -m pip uninstall -y paddleocr paddlex modelscope albumentations albucore` "
            "then `python -m pip install --no-cache-dir --force-reinstall -r requirements-ocr.txt`."
        )
    try:
        albumentations_version = metadata.version("albumentations")
    except metadata.PackageNotFoundError as error:
        raise ImportError(
            "Windows GPU OCR requires albumentations==1.4.10. Run "
            "`python -m pip install --no-cache-dir --force-reinstall -r requirements-ocr.txt`."
        ) from error
    if albumentations_version != "1.4.10":
        raise RuntimeError(
            "This Windows GPU OCR worker requires albumentations==1.4.10 so PaddleOCR does not import Torch. "
            f"Found {albumentations_version}. Run `python -m pip uninstall -y albumentations albucore` then "
            "`python -m pip install --no-cache-dir --force-reinstall -r requirements-ocr.txt`."
        )
    try:
        albucore_version = metadata.version("albucore")
    except metadata.PackageNotFoundError as error:
        raise ImportError(
            "Windows GPU OCR requires albucore==0.0.13. Run "
            "`python -m pip install --no-cache-dir --force-reinstall -r requirements-ocr.txt`."
        ) from error
    if albucore_version != "0.0.13":
        raise RuntimeError(
            "This Windows GPU OCR worker requires albucore==0.0.13. "
            f"Found {albucore_version}. Run `python -m pip uninstall -y albumentations albucore` then "
            "`python -m pip install --no-cache-dir --force-reinstall -r requirements-ocr.txt`."
        )


def _create_reader(device: str) -> Any:
    _require_paddleocr_v2()
    # Importing this module is safe; Paddle itself is imported lazily by the
    # reader below, before this worker imports image/pipeline modules.
    from .ocr import PaddleOCRReader

    return PaddleOCRReader(device=device, require_v2=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "inference_manifest.json",
        output_dir / "inference_manifest.jsonl",
        output_dir / "inference_errors.jsonl",
    )


def _written_record(source_path: Path, output_stem: Path, *, status: str) -> dict[str, str]:
    return {
        "source": source_path.resolve().as_posix(),
        "result": output_stem.with_suffix(".json").resolve().as_posix(),
        "annotated_rectified": output_stem.with_name(
            output_stem.name + "_rectified_annotated.jpg"
        ).resolve().as_posix(),
        "annotated_original": output_stem.with_name(
            output_stem.name + "_original_annotated.jpg"
        ).resolve().as_posix(),
        "status": status,
    }


def _ocr_committed_result_exists(source_path: Path, output_stem: Path) -> bool:
    """Return whether a final result bundle was completed by the OCR stage."""
    result_path = output_stem.with_suffix(".json")
    # 只看 JSON(写在标注图之后的完成标记);标注图在 flagged/none 下可能缺,不作为续跑依据。
    if not result_path.is_file():
        return False
    if result_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not has_current_result_semantics(payload)
        or not isinstance(payload, dict)
        or not isinstance(payload.get("fields"), dict)
    ):
        return False
    if payload.get("source") != source_path.resolve().as_posix():
        return False
    detections = payload.get("detections")
    if not isinstance(detections, list):
        return False
    # A crop can only lack OCR when it is geometrically empty, in which case
    # retrying it is safer than treating a detector-only result as complete.
    # Empty detection lists are also retried: without an extra public-schema
    # marker they are indistinguishable from a prior ``--ocr none`` run.
    return bool(detections) and all(isinstance(item, dict) and "ocr" in item for item in detections)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read detector-stage artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Detector-stage artifact {path} must contain a JSON object")
    if payload.get("schema_version") != _STAGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported detector-stage artifact schema in {path}")
    return payload


def _stage_paths(stage_stem: Path) -> tuple[Path, Path]:
    return (
        stage_stem.with_name(stage_stem.name + "_ocr_stage.json"),
        stage_stem.with_name(stage_stem.name + "_ocr_stage_rectified.png"),
    )


def _source_path(payload: dict[str, Any]) -> Path:
    value = payload.get("source")
    if not isinstance(value, str) or not value:
        raise ValueError("Detector-stage artifact is missing its source path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Source image no longer exists: {path}")
    fingerprint = payload.get("source_fingerprint")
    if isinstance(fingerprint, dict):
        stat = path.stat()
        if fingerprint.get("size") != stat.st_size or fingerprint.get("mtime_ns") != stat.st_mtime_ns:
            raise ValueError(f"Source image changed after detector stage: {path}")
    return path


def _relative_stage_stem(record: dict[str, Any], stage_output_dir: Path) -> Path:
    value = record.get("result")
    if not isinstance(value, str):
        raise ValueError("Detector manifest record is missing result path")
    result_path = Path(value).resolve()
    try:
        return result_path.relative_to(stage_output_dir.resolve()).with_suffix("")
    except ValueError as error:
        raise ValueError(f"Detector result is outside the stage directory: {result_path}") from error


def _skipped_record(record: dict[str, Any]) -> dict[str, str]:
    required = ("source", "result", "annotated_rectified", "annotated_original", "status")
    if any(not isinstance(record.get(key), str) for key in required):
        raise ValueError("Detector manifest has an invalid skipped-existing record")
    return {key: record[key] for key in required}


def _copy_detector_errors(stage_output_dir: Path, errors_file: Any) -> int:
    source = stage_output_dir / "inference_errors.jsonl"
    if not source.is_file():
        return 0
    count = 0
    with source.open("r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                errors_file.write(line.rstrip("\n") + "\n")
                count += 1
    errors_file.flush()
    return count


def _rehydrate_result(payload: dict[str, Any], rectified_path: Path, reader: Any) -> tuple[Any, Path]:
    """Build a final ReceiptResult from exact detector-stage state and OCR."""
    import numpy as np
    from PIL import Image

    from .geometry import RectificationResult, load_upright_rgb
    from .labels import DETECTION_CLASSES
    from .model import Detection
    from .pipeline import ExtractedDetection, ReceiptResult, _build_fields, _crop_with_margin

    source_path = _source_path(payload)
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Detector-stage artifact is missing geometry")
    source_rgb = load_upright_rgb(source_path)
    source_size = geometry.get("source_size")
    if not isinstance(source_size, dict) or (
        source_size.get("width"),
        source_size.get("height"),
    ) != (source_rgb.shape[1], source_rgb.shape[0]):
        raise ValueError(f"Source image dimensions changed after detector stage: {source_path}")
    if not rectified_path.is_file():
        raise FileNotFoundError(f"Detector-stage rectified image is missing: {rectified_path}")
    with Image.open(rectified_path) as image:
        rectified_rgb = np.asarray(image.convert("RGB")).copy()
    rectified_size = geometry.get("rectified_size")
    if not isinstance(rectified_size, dict) or (
        rectified_size.get("width"),
        rectified_size.get("height"),
    ) != (rectified_rgb.shape[1], rectified_rgb.shape[0]):
        raise ValueError(f"Detector-stage rectified image has unexpected dimensions: {rectified_path}")

    def matrix(name: str) -> Any:
        values = np.asarray(geometry.get(name), dtype=np.float64)
        if values.shape != (3, 3) or not np.isfinite(values).all() or abs(float(np.linalg.det(values))) < 1e-12:
            raise ValueError(f"Detector-stage geometry has invalid {name}")
        return values

    screen_quad = np.asarray(geometry.get("screen_quad_original"), dtype=np.float32)
    if screen_quad.shape != (4, 2) or not np.isfinite(screen_quad).all():
        raise ValueError("Detector-stage geometry has invalid screen_quad_original")
    original_to_rectified = matrix("H_original_to_rectified")
    rectified_to_original = matrix("H_rectified_to_original")
    if not np.allclose(original_to_rectified @ rectified_to_original, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("Detector-stage geometry transforms are not inverses")
    rotation_degrees = int(geometry.get("rotation_degrees", 0))
    if rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError("Detector-stage geometry has invalid rotation_degrees")
    rectification = RectificationResult(
        source_rgb=source_rgb,
        rectified_rgb=rectified_rgb,
        original_to_rectified=original_to_rectified,
        rectified_to_original=rectified_to_original,
        screen_quad_original=screen_quad,
        rotation_degrees=rotation_degrees,
        screen_detected=bool(geometry.get("screen_detected", False)),
    )

    staged_detections = payload.get("detections")
    if not isinstance(staged_detections, list):
        raise ValueError("Detector-stage artifact has invalid detections")
    detections: list[Any] = []
    seen_labels: set[str] = set()
    for item in staged_detections:
        if not isinstance(item, dict):
            raise ValueError("Detector-stage artifact contains a non-object detection")
        label = item.get("label")
        score = item.get("score")
        bbox = item.get("bbox_xyxy")
        polygon = np.asarray(item.get("original_polygon"), dtype=np.float32)
        if (
            not isinstance(label, str)
            or not isinstance(score, (int, float))
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
            or polygon.shape != (4, 2)
        ):
            raise ValueError("Detector-stage artifact contains an invalid detection")
        values = tuple(float(value) for value in bbox)
        if (
            label not in DETECTION_CLASSES
            or label in seen_labels
            or not np.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or values[2] <= values[0]
            or values[3] <= values[1]
            or not np.isfinite(np.asarray(values)).all()
            or not np.isfinite(polygon).all()
        ):
            raise ValueError("Detector-stage artifact contains non-finite detection coordinates")
        seen_labels.add(label)
        detection = Detection(label, float(score), values)
        ocr_result = None
        crop = _crop_with_margin(rectified_rgb, detection.bbox_xyxy)
        if crop.size:
            ocr_result = reader.recognize(crop)
        detections.append(ExtractedDetection(detection, ocr_result, polygon))

    status_style = payload.get("status_style")
    tags = payload.get("tags")
    device = payload.get("device")
    if status_style is not None and not isinstance(status_style, dict):
        raise ValueError("Detector-stage artifact has invalid status_style")
    if tags is not None and not isinstance(tags, dict):
        raise ValueError("Detector-stage artifact has invalid tags")
    if device is not None and not isinstance(device, dict):
        raise ValueError("Detector-stage artifact has invalid device")
    return (
        ReceiptResult(
            source_path=source_path.resolve().as_posix(),
            rectification=rectification,
            detections=detections,
            fields=_build_fields(detections),
            status_style=status_style,
            tags=tags,
            device=device,
        ),
        source_path,
    )


def run_ocr_enrichment(
    *,
    stage_output_dir: Path,
    output_dir: Path,
    reader: Any,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    annotate: str = "all",
) -> list[dict[str, str]]:
    """Complete one detector batch without importing or instantiating Torch."""
    # These imports occur only after PaddleOCR has been initialised by ``main``.
    from tqdm import tqdm

    from .pipeline import write_receipt_result

    stage_output_dir = stage_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_output_dir / "inference_manifest.json"
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read detector manifest {manifest_path}: {error}") from error
    if not isinstance(manifest_payload, list):
        raise ValueError(f"Detector manifest {manifest_path} must contain a JSON list")

    final_manifest_path, final_manifest_jsonl_path, final_errors_path = _manifest_paths(output_dir)
    final_manifest: list[dict[str, str]] = []
    failure_count = 0
    with final_manifest_jsonl_path.open("w", encoding="utf-8") as manifest_jsonl, final_errors_path.open(
        "w", encoding="utf-8"
    ) as errors_jsonl:
        failure_count += _copy_detector_errors(stage_output_dir, errors_jsonl)
        for raw_record in tqdm(manifest_payload, desc="Running PaddleOCR completion"):
            if not isinstance(raw_record, dict):
                error = ValueError("Detector manifest contains a non-object record")
                failure_count += 1
                errors_jsonl.write(
                    json.dumps({"source": None, "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False)
                    + "\n"
                )
                errors_jsonl.flush()
                if not continue_on_error:
                    raise error
                continue
            try:
                if raw_record.get("status") == "skipped_existing":
                    record = _skipped_record(raw_record)
                    final_manifest.append(record)
                    manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest_jsonl.flush()
                    continue
                relative_stem = _relative_stage_stem(raw_record, stage_output_dir)
                stage_stem = stage_output_dir / relative_stem
                artifact_path, rectified_path = _stage_paths(stage_stem)
                payload = _load_json(artifact_path)
                source_path = _source_path(payload)
                output_stem = output_dir / relative_stem
                if skip_existing and _ocr_committed_result_exists(source_path, output_stem):
                    record = _written_record(source_path, output_stem, status="skipped_existing")
                else:
                    result, source_path = _rehydrate_result(payload, rectified_path, reader)
                    write_receipt_result(result, output_stem, annotate_mode=annotate)
                    record = _written_record(source_path, output_stem, status="written")
                final_manifest.append(record)
                manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_jsonl.flush()
            except Exception as error:
                failure_count += 1
                source_value = raw_record.get("source")
                errors_jsonl.write(
                    json.dumps(
                        {
                            "source": source_value if isinstance(source_value, str) else None,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors_jsonl.flush()
                if not continue_on_error:
                    raise
    _write_json_atomic(final_manifest_path, final_manifest)
    if failure_count:
        print(f"WARNING: {failure_count} image(s) failed; see {final_errors_path}")
    return final_manifest


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Construct PaddleOCR before importing the pipeline, and never construct a
    # Torch predictor in this process.
    reader = _create_reader(args.device)
    outputs = run_ocr_enrichment(
        stage_output_dir=args.stage_output,
        output_dir=args.output,
        reader=reader,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        annotate=args.annotate,
    )
    if not args._quiet:
        print(f"Wrote {len(outputs)} inference result bundle(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
