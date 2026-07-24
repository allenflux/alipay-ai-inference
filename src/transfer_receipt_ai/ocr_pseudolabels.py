"""Build a reviewable OCR training set from completed PaddleOCR result bundles.

This module never imports PaddleOCR.  It consumes the JSON bundles produced by
the existing Python ``--ocr paddle`` pipeline, recreates a clean rectified
image from the recorded final-result geometry, and applies the same field-crop
rule as the Paddle worker.  Final JSON coordinates are rounded, so this is a
clean reproducible approximation rather than a pixel-for-pixel copy of the
private full-precision stage crop.  Paddle text is therefore a *pseudo label*,
not a ground truth label: the exporter applies conservative filters and always
writes enough provenance for a human review pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import load_upright_rgb, save_rgb
from .labels import DETECTION_CLASSES
from .ocr import (
    clean_text,
    extract_field_value,
    normalize_amount,
    normalize_payment_method,
    normalize_status,
    normalize_time,
)
from .pipeline import _crop_with_margin
from .status_crops import (
    _group_id,
    _load_json_document,
    _paths_overlap,
    _result_payload,
    _selection_key,
    _source_path,
    reconstruct_rectified,
)


DATASET_SCHEMA_VERSION = 1
DEFAULT_MIN_DETECTOR_SCORE = 0.90
DEFAULT_MIN_OCR_CONFIDENCE = 0.98
DEFAULT_MAX_TEXT_LENGTH = 64
DEFAULT_MAX_PER_LABEL_TEXT = 250


class UnsafeOcrDatasetOutputError(ValueError):
    """Raised before a data-set export could overwrite results or source images."""


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalise_text(value: str) -> str:
    """Keep a visually faithful, portable label string for CTC training."""
    return unicodedata.normalize("NFC", clean_text(value))


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return score


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("bbox_rectified must contain four numeric values")
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("bbox_rectified must contain four numeric values") from None
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_rectified must be a finite non-empty xyxy box")
    return x1, y1, x2, y2


def _semantic_value(label: str, text: str) -> str | None:
    """Return a useful structured value, or reject a clearly bad pseudo label.

    The training label remains the visible ``text``.  The semantic value is
    retained only for auditing; replacing visible text with normalised text
    would teach a recognizer characters that are not in the crop.
    """
    if label == "amount":
        amount = normalize_amount(text)
        return str(amount["normalized"]) if amount is not None else None
    if label == "time":
        return normalize_time(text)
    if label == "transfer_status":
        status = normalize_status(text)
        return None if status == "unknown" else status
    if label == "recipient_field":
        value = extract_field_value(text, "recipient")
        return value or None
    if label == "payment_method_field":
        value = extract_field_value(text, "payment_method")
        return normalize_payment_method(value)["normalized"] if value else None
    raise ValueError(f"Unsupported OCR field label: {label}")


def _split_for_group(group_id: str, *, validation_ratio: float, test_ratio: float, split_seed: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"{split_seed}\0{group_id}".encode("utf-8")).digest()[:8], "big"
    ) / float(2**64)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "val"
    return "train"


def _review_selected(sample_id: str, ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    bucket = int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big") / float(2**64)
    return bucket < ratio


def _character_coverage(records: Iterable[Mapping[str, object]], *, labels: Sequence[str]) -> dict[str, object]:
    """Count characters by field and split so free-text coverage is inspectable."""
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = {
        label: {split: [] for split in ("train", "val", "test")} for label in labels
    }
    for record in records:
        label = str(record["field"])
        split = str(record["split"])
        if label in grouped and split in grouped[label]:
            grouped[label][split].append(record)
    coverage: dict[str, object] = {}
    for label, splits in grouped.items():
        label_coverage: dict[str, object] = {}
        for split, split_records in splits.items():
            characters = Counter(character for record in split_records for character in str(record["text"]))
            label_coverage[split] = {
                "records": len(split_records),
                "unique_characters": len(characters),
                "characters": dict(sorted(characters.items())),
            }
        coverage[label] = label_coverage
    return coverage


def _sample_id(result_json: Path, label: str, bbox: Sequence[float], text: str) -> str:
    payload = json.dumps(
        {
            "result_json": result_json.resolve().as_posix(),
            "label": label,
            "bbox_rectified": [round(float(value), 4) for value in bbox],
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _crop_digest(crop_rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(crop_rgb.shape).encode("ascii"))
    digest.update(crop_rgb.tobytes(order="C"))
    return digest.hexdigest()


def _rejection(
    result_json: Path,
    *,
    label: object = None,
    reason: str,
    detail: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "result_json": result_json.resolve().as_posix(),
        "label": label if isinstance(label, str) else None,
        "reason": reason,
    }
    if detail:
        record["detail"] = detail
    return record


def _parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("at least one field label is required")
    invalid = sorted(set(labels) - set(DETECTION_CLASSES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown labels: {','.join(invalid)}; expected: {','.join(DETECTION_CLASSES)}"
        )
    return labels


def _validate_build_options(
    *,
    min_detector_score: float,
    min_ocr_confidence: float,
    validation_ratio: float,
    test_ratio: float,
    review_ratio: float,
    max_text_length: int,
    max_per_label_text: int,
    limit: int | None,
) -> None:
    for name, value in (
        ("min_detector_score", min_detector_score),
        ("min_ocr_confidence", min_ocr_confidence),
        ("validation_ratio", validation_ratio),
        ("test_ratio", test_ratio),
        ("review_ratio", review_ratio),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")
    if max_text_length <= 0:
        raise ValueError("max_text_length must be positive")
    if max_per_label_text <= 0:
        raise ValueError("max_per_label_text must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")


def _assert_safe_output(results_dir: Path, output_dir: Path) -> None:
    if _paths_overlap(results_dir, output_dir):
        raise UnsafeOcrDatasetOutputError(
            "dataset output and inference results directories must not overlap in either direction"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UnsafeOcrDatasetOutputError(
            f"dataset output already contains files: {output_dir}. Choose a new empty output directory."
        )


def _preflight_source_output_overlap(result_paths: Sequence[Path], output_dir: Path) -> None:
    """Reject an unsafe output tree before writing the first crop."""
    for result_json in result_paths:
        try:
            payload = _result_payload(_load_json_document(result_json))
            if payload is None:
                continue
            source = _source_path(payload, result_json)
        except (OSError, ValueError):
            # Normal processing records malformed inputs in rejected/errors.
            continue
        if _paths_overlap(output_dir, source.parent):
            raise UnsafeOcrDatasetOutputError(
                "dataset output and raw source image directory must not overlap in either direction: "
                f"output={output_dir}, source_directory={source.parent}"
            )


def build_pseudo_label_dataset(
    *,
    results_dir: Path,
    output_dir: Path,
    labels: Sequence[str] = DETECTION_CLASSES,
    min_detector_score: float = DEFAULT_MIN_DETECTOR_SCORE,
    min_ocr_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    max_per_label_text: int = DEFAULT_MAX_PER_LABEL_TEXT,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.0,
    review_ratio: float = 0.10,
    split_seed: str = "receipt-ocr-pseudo-v1",
    limit: int | None = None,
    allow_source_newer: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, object]]:
    """Export high-confidence PaddleOCR pseudo labels and their clean crops.

    ``results_dir`` must contain final JSON bundles generated by the Python
    PaddleOCR workflow.  ML.NET JSON output does not include OCR or rectified
    geometry, so it is intentionally not accepted as an input source.
    """
    labels = tuple(labels)
    invalid_labels = sorted(set(labels) - set(DETECTION_CLASSES))
    if not labels or invalid_labels:
        raise ValueError(f"labels must be a non-empty subset of: {','.join(DETECTION_CLASSES)}")
    _validate_build_options(
        min_detector_score=min_detector_score,
        min_ocr_confidence=min_ocr_confidence,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        review_ratio=review_ratio,
        max_text_length=max_text_length,
        max_per_label_text=max_per_label_text,
        limit=limit,
    )
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)
    output_dir = output_dir.resolve()
    _assert_safe_output(results_dir, output_dir)
    result_paths = sorted(
        results_dir.rglob("*.json"),
        key=lambda path: _selection_key(path.relative_to(results_dir)),
    )
    if limit is not None:
        result_paths = result_paths[:limit]
    _preflight_source_output_overlap(result_paths, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    seen_crops: dict[str, tuple[str, str]] = {}
    per_label_text: Counter[tuple[str, str]] = Counter()

    for result_json in result_paths:
        relative_result = result_json.relative_to(results_dir)
        try:
            payload = _result_payload(_load_json_document(result_json))
            if payload is None:
                continue
            source = _source_path(payload, result_json)
            source_stat = source.stat()
            if not allow_source_newer and source_stat.st_mtime_ns > result_json.stat().st_mtime_ns:
                raise ValueError(
                    "source image is newer than its PaddleOCR result JSON; do not create pseudo labels from "
                    "a changed source image. Re-run the Python --ocr paddle inference first."
                )
            if _paths_overlap(output_dir, source.parent):
                raise UnsafeOcrDatasetOutputError(
                    "dataset output and raw source image directory must not overlap in either direction"
                )
            source_rgb = load_upright_rgb(source)
            rectified_rgb = reconstruct_rectified(payload, source_rgb)
            detections = payload.get("detections")
            if not isinstance(detections, list):
                rejected.append(_rejection(result_json, reason="invalid_detections"))
                continue
            group_id = _group_id(source, relative_result)
            for detection in detections:
                label: object = None
                try:
                    if not isinstance(detection, Mapping):
                        raise ValueError("detection must be an object")
                    label = detection.get("label")
                    if not isinstance(label, str) or label not in labels:
                        continue
                    detector_score = _finite_score(detection.get("score"), "detector score")
                    if detector_score < min_detector_score:
                        rejected.append(
                            _rejection(result_json, label=label, reason="low_detector_score", detail=str(detector_score))
                        )
                        continue
                    ocr = detection.get("ocr")
                    if not isinstance(ocr, Mapping):
                        rejected.append(_rejection(result_json, label=label, reason="missing_ocr"))
                        continue
                    raw_text = ocr.get("text")
                    if not isinstance(raw_text, str):
                        rejected.append(_rejection(result_json, label=label, reason="invalid_ocr_text"))
                        continue
                    text = _normalise_text(raw_text)
                    if not text:
                        rejected.append(_rejection(result_json, label=label, reason="empty_text"))
                        continue
                    if len(text) > max_text_length:
                        rejected.append(
                            _rejection(result_json, label=label, reason="text_too_long", detail=str(len(text)))
                        )
                        continue
                    if any(not character.isprintable() for character in text):
                        rejected.append(_rejection(result_json, label=label, reason="non_printable_text"))
                        continue
                    ocr_confidence = _finite_score(ocr.get("confidence"), "OCR confidence")
                    if ocr_confidence < min_ocr_confidence:
                        rejected.append(
                            _rejection(result_json, label=label, reason="low_ocr_confidence", detail=str(ocr_confidence))
                        )
                        continue
                    semantic_value = _semantic_value(label, text)
                    if semantic_value is None:
                        rejected.append(_rejection(result_json, label=label, reason="field_semantic_validation_failed"))
                        continue
                    bbox = _bbox(detection.get("bbox_rectified"))
                    crop_rgb = _crop_with_margin(rectified_rgb, bbox)
                    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 8:
                        rejected.append(_rejection(result_json, label=label, reason="empty_or_tiny_crop"))
                        continue
                    if int(crop_rgb.max()) == int(crop_rgb.min()):
                        rejected.append(_rejection(result_json, label=label, reason="blank_crop"))
                        continue
                    crop_sha256 = _crop_digest(crop_rgb)
                    previous = seen_crops.get(crop_sha256)
                    if previous is not None:
                        if previous == (label, text):
                            rejected.append(_rejection(result_json, label=label, reason="duplicate_crop"))
                        else:
                            rejected.append(_rejection(result_json, label=label, reason="conflicting_duplicate_crop"))
                        continue
                    text_key = (label, text)
                    if per_label_text[text_key] >= max_per_label_text:
                        rejected.append(_rejection(result_json, label=label, reason="per_label_text_limit"))
                        continue
                    sample_id = _sample_id(result_json, label, bbox, text)
                    relative_image = Path("images") / label / f"{crop_sha256}.png"
                    image_path = output_dir / relative_image
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    save_rgb(image_path, crop_rgb)
                    seen_crops[crop_sha256] = (label, text)
                    per_label_text[text_key] += 1
                    record = {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "id": sample_id,
                        "image": relative_image.as_posix(),
                        "field": label,
                        "text": text,
                        "paddle_text": raw_text,
                        "paddle_confidence": round(ocr_confidence, 6),
                        "detector_score": round(detector_score, 6),
                        "semantic_value": semantic_value,
                        "bbox_rectified": [round(value, 3) for value in bbox],
                        "source": source.resolve().as_posix(),
                        "source_size_bytes": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                        "result_json": result_json.resolve().as_posix(),
                        "group_id": group_id,
                        "crop_sha256": crop_sha256,
                        "split": _split_for_group(
                            group_id,
                            validation_ratio=validation_ratio,
                            test_ratio=test_ratio,
                            split_seed=split_seed,
                        ),
                        "label_source": "paddle_pseudo",
                    }
                    accepted.append(record)
                except Exception as error:
                    rejected.append(
                        _rejection(
                            result_json,
                            label=label,
                            reason="invalid_detection",
                            detail=f"{type(error).__name__}: {error}",
                        )
                    )
        except UnsafeOcrDatasetOutputError:
            raise
        except Exception as error:
            error_record = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "result_json": result_json.resolve().as_posix(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
            errors.append(error_record)
            if not continue_on_error:
                raise

    accepted.sort(key=lambda record: str(record["id"]))
    review = [record for record in accepted if _review_selected(str(record["id"]), review_ratio)]
    split_records = {
        split: [record for record in accepted if record["split"] == split]
        for split in ("train", "val", "test")
    }
    character_coverage = _character_coverage(accepted, labels=labels)
    characters = sorted({character for record in accepted for character in str(record["text"])})
    by_field = Counter(str(record["field"]) for record in accepted)
    config = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": "receipt_ocr_paddle_pseudo_v1",
        "results_dir": results_dir.as_posix(),
        "labels": list(labels),
        "selection": {
            "min_detector_score": min_detector_score,
            "min_ocr_confidence": min_ocr_confidence,
            "max_text_length": max_text_length,
            "max_per_label_text": max_per_label_text,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "review_ratio": review_ratio,
            "split_seed": split_seed,
            "allow_source_newer": allow_source_newer,
        },
        "counts": {
            "accepted": len(accepted),
            "review_candidates": len(review),
            "rejected": len(rejected),
            "errors": len(errors),
            "by_field": dict(sorted(by_field.items())),
            "by_split": {name: len(records) for name, records in split_records.items()},
            "charset_size": len(characters),
        },
        "character_coverage_file": "character_coverage.json",
        "warning": (
            "Records are PaddleOCR pseudo labels. Review a stratified sample and keep an independent "
            "human-labelled evaluation set before making accuracy claims. Source images must remain unchanged "
            "after PaddleOCR inference; legacy result JSON can only detect modifications whose source mtime changes."
        ),
    }
    _atomic_write_jsonl(output_dir / "pseudo_labels.jsonl", accepted)
    _atomic_write_jsonl(output_dir / "review_candidates.jsonl", review)
    _atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    _atomic_write_jsonl(output_dir / "build_errors.jsonl", errors)
    for split, records in split_records.items():
        _atomic_write_jsonl(output_dir / "splits" / f"{split}.jsonl", records)
    (output_dir / "charset.txt").write_text("".join(characters) + "\n", encoding="utf-8")
    _atomic_write_json(output_dir / "character_coverage.json", character_coverage)
    _atomic_write_json(output_dir / "dataset_config.json", config)
    return accepted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export reviewable clean OCR crops and high-confidence PaddleOCR pseudo labels"
    )
    parser.add_argument("--results", type=Path, required=True, help="Python/Paddle inference result directory")
    parser.add_argument("--output", type=Path, required=True, help="New empty pseudo-label dataset directory")
    parser.add_argument(
        "--labels",
        type=_parse_labels,
        default=DETECTION_CLASSES,
        help="Comma-separated fields; default is all receipt fields",
    )
    parser.add_argument("--min-detector-score", type=float, default=DEFAULT_MIN_DETECTOR_SCORE)
    parser.add_argument("--min-ocr-confidence", type=float, default=DEFAULT_MIN_OCR_CONFIDENCE)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--max-per-label-text", type=int, default=DEFAULT_MAX_PER_LABEL_TEXT)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--review-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", default="receipt-ocr-pseudo-v1")
    parser.add_argument("--limit", type=int, help="Deterministic maximum number of result JSON candidates")
    parser.add_argument(
        "--allow-source-newer",
        action="store_true",
        help="Allow source files newer than the result JSON (unsafe unless you verified their content is unchanged)",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        records = build_pseudo_label_dataset(
            results_dir=args.results,
            output_dir=args.output,
            labels=args.labels,
            min_detector_score=args.min_detector_score,
            min_ocr_confidence=args.min_ocr_confidence,
            max_text_length=args.max_text_length,
            max_per_label_text=args.max_per_label_text,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            review_ratio=args.review_ratio,
            split_seed=args.split_seed,
            limit=args.limit,
            allow_source_newer=args.allow_source_newer,
            continue_on_error=args.continue_on_error,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"OCR pseudo-label export failed:\n{error}") from None
    config = json.loads((args.output / "dataset_config.json").read_text(encoding="utf-8"))
    counts = config["counts"]
    print(
        f"Exported {len(records)} PaddleOCR pseudo-label crop(s) to {args.output} "
        f"(rejected={counts['rejected']}, errors={counts['errors']}, by_field={counts['by_field']})"
    )
    if counts["errors"]:
        print("WARNING: inspect build_errors.jsonl before training; failed result bundles were excluded.")


if __name__ == "__main__":  # pragma: no cover
    main()
