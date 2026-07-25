"""Create receipt-field training crops from detector results and local truth.

Unlike :mod:`ocr_pseudolabels`, this module neither imports nor consumes
PaddleOCR.  It combines existing Python detector result JSON (which contains
rectification geometry) with a local JSONL transaction table.  This lets a
large historical image set become supervised data while keeping the entire
pipeline offline.

The transaction table is deliberately simple and explicit::

    {"receipt_key":"GWCZ207...", "amount":"100.00", "time":"12:06",
     "transfer_status":"success", "payment_method":"bank_card",
     "recipient":"交易商家"}

``receipt_key`` is extracted from the source filename with a caller-controlled
regular expression.  A filename lookup is training provenance only; production
must still use the image detector and put image/lookup disagreements into
review rather than treating a rename as proof of authenticity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import load_upright_rgb, save_rgb
from .labels import DETECTION_CLASSES
from .ocr import clean_text, normalize_payment_method, normalize_status, normalize_time
from .pipeline import crop_field_with_margin
from .status_crops import _load_json_document, _paths_overlap, _result_payload, _selection_key, _source_path, reconstruct_rectified


SCHEMA_VERSION = 1
DEFAULT_MIN_DETECTOR_SCORE = 0.80
DEFAULT_SOURCE_KEY_REGEX = r"(?P<key>GWCZ[0-9A-Za-z]+)"

FIELD_TO_TRUTH_KEY = {
    "amount": "amount",
    "time": "time",
    "transfer_status": "transfer_status",
    "recipient_field": "recipient",
    "payment_method_field": "payment_method",
}


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _split_for_group(group_id: str, *, validation_ratio: float, test_ratio: float, split_seed: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"{split_seed}\0{group_id}".encode("utf-8")).digest()[:8], "big"
    ) / float(2**64)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "val"
    return "train"


def _crop_digest(crop_rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(crop_rgb.shape).encode("ascii"))
    digest.update(crop_rgb.tobytes(order="C"))
    return digest.hexdigest()


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
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


def _normalise_amount(value: object, record: Mapping[str, object]) -> str | None:
    raw_fen = record.get("amount_fen")
    if raw_fen is not None:
        if isinstance(raw_fen, bool):
            return None
        try:
            fen_decimal = Decimal(str(raw_fen))
        except (InvalidOperation, TypeError, ValueError):
            return None
        if not fen_decimal.is_finite() or fen_decimal != fen_decimal.to_integral_value():
            return None
        fen = int(fen_decimal)
        if fen < 0:
            return None
        return f"¥{Decimal(fen) / Decimal(100):.2f}"
    if isinstance(value, bool) or value is None:
        return None
    compact = clean_text(str(value)).replace(" ", "")
    if compact.startswith(("¥", "￥")):
        compact = compact[1:]
    # Transaction truth must describe the entire value.  Do not reuse the
    # permissive OCR normalizer here: it intentionally accepts a numeric
    # substring from noisy text, which could turn a malformed truth value such
    # as ``12.345`` into a plausible but wrong training label.
    if not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?", compact):
        return None
    try:
        decimal = Decimal(compact.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite() or decimal < 0:
        return None
    try:
        decimal = decimal.quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
    return f"¥{decimal:.2f}"


def _normalise_truth_value(field: str, record: Mapping[str, object]) -> str | None:
    truth_key = FIELD_TO_TRUTH_KEY[field]
    value = record.get(truth_key)
    if field == "amount":
        return _normalise_amount(value, record)
    if not isinstance(value, str) or not clean_text(value):
        return None
    if field == "time":
        return normalize_time(value)
    if field == "transfer_status":
        compact = clean_text(value).lower()
        if compact in {"success", "pending", "failed"}:
            return compact
        normalized = normalize_status(value)
        return normalized if normalized != "unknown" else None
    if field == "payment_method_field":
        compact = clean_text(value).lower()
        if compact in {"yuebao", "balance", "huabei", "bank_card", "other"}:
            return compact
        return normalize_payment_method(value)["normalized"]
    if field == "recipient_field":
        return clean_text(value) or None
    raise ValueError(f"Unsupported field {field}")


def _load_truth_records(path: Path, *, key_field: str) -> dict[str, dict[str, object]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    truth_by_key: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            raw_key = value.get(key_field)
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError(f"{path}:{line_number}: {key_field!r} must be a non-empty string")
            key = raw_key.strip()
            prior = truth_by_key.get(key)
            current = dict(value)
            if prior is not None and prior != current:
                raise ValueError(f"{path}:{line_number}: conflicting truth rows for {key_field}={key!r}")
            truth_by_key[key] = current
    if not truth_by_key:
        raise ValueError(f"No truth records in {path}")
    return truth_by_key


def _source_key(source: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(source.name)
    if match is None:
        raise ValueError(f"source filename does not match receipt-key pattern: {source.name}")
    try:
        value = match.group("key")
    except IndexError:
        raise ValueError("receipt-key regex must define a named group '(?P<key>...)'") from None
    if not value:
        raise ValueError("receipt-key regex produced an empty key")
    return value


def _best_detection(
    payload: Mapping[str, object], field: str, *, min_detector_score: float
) -> tuple[tuple[tuple[float, float, float, float], float] | None, list[str]]:
    """Choose the best usable box while retaining malformed candidates for audit."""
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("result has no detections list")
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    invalid_candidates: list[str] = []
    for detection in detections:
        if not isinstance(detection, Mapping) or detection.get("label") != field:
            continue
        try:
            score = _finite_score(detection.get("score"), "detector score")
            bbox = _bbox(detection.get("bbox_rectified"))
        except ValueError as error:
            invalid_candidates.append(str(error))
            continue
        if score < min_detector_score:
            continue
        candidates.append((bbox, score))
    return (max(candidates, key=lambda item: item[1]) if candidates else None), invalid_candidates


def _sample_id(result_json: Path, field: str, receipt_key: str, bbox: Sequence[float]) -> str:
    payload = json.dumps(
        {
            "result_json": result_json.resolve().as_posix(),
            "field": field,
            "receipt_key": receipt_key,
            "bbox_rectified": [round(float(value), 4) for value in bbox],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_safe_output(results_dir: Path, output_dir: Path, result_paths: Sequence[Path]) -> None:
    if _paths_overlap(results_dir, output_dir):
        raise ValueError("output directory and detector result directory must not overlap")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory already contains files: {output_dir}")
    for result_json in result_paths:
        try:
            payload = _result_payload(_load_json_document(result_json))
            if payload is None:
                continue
            source = _source_path(payload, result_json)
        except (OSError, ValueError):
            continue
        if _paths_overlap(output_dir, source.parent):
            raise ValueError(
                "output directory and source image directory must not overlap: "
                f"output={output_dir}, source_directory={source.parent}"
            )


def _new_staging_directory(output_dir: Path) -> Path:
    """Create an unpublished sibling directory for an all-or-nothing export."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))


def _write_jsonl_record(stream: Any, record: Mapping[str, object]) -> None:
    stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def build_truth_dataset(
    *,
    results_dir: Path,
    truth_path: Path,
    output_dir: Path,
    receipt_key_field: str = "receipt_key",
    source_key_regex: str = DEFAULT_SOURCE_KEY_REGEX,
    fields: Sequence[str] = DETECTION_CLASSES,
    min_detector_score: float = DEFAULT_MIN_DETECTOR_SCORE,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    split_seed: str = "receipt-truth-v1",
    limit: int | None = None,
    continue_on_error: bool = False,
    allow_source_newer: bool = False,
) -> dict[str, object]:
    """Build crop/semantic records from local transaction truth and detector JSON.

    The output is first written to a sibling staging directory and only
    published after every JSONL file and crop has been completed.  This keeps
    an interrupted large build from looking like a valid training dataset.
    """
    if not receipt_key_field.strip():
        raise ValueError("receipt_key_field must not be empty")
    fields = tuple(fields)
    invalid_fields = sorted(set(fields) - set(DETECTION_CLASSES))
    if not fields or invalid_fields:
        raise ValueError(f"fields must be a non-empty subset of: {','.join(DETECTION_CLASSES)}")
    if not math.isfinite(min_detector_score) or not 0.0 <= min_detector_score <= 1.0:
        raise ValueError("min_detector_score must be between 0 and 1")
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in (validation_ratio, test_ratio)):
        raise ValueError("validation_ratio and test_ratio must be between 0 (inclusive) and 1 (exclusive)")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    try:
        key_pattern = re.compile(source_key_regex)
    except re.error as error:
        raise ValueError(f"invalid source_key_regex: {error}") from None
    if "key" not in key_pattern.groupindex:
        raise ValueError("source_key_regex must define a named group '(?P<key>...)'")

    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)
    truth_by_key = _load_truth_records(truth_path, key_field=receipt_key_field)
    output_dir = output_dir.resolve()
    result_paths = sorted(results_dir.rglob("*.json"), key=lambda path: _selection_key(path.relative_to(results_dir)))
    if limit is not None:
        result_paths = result_paths[:limit]
    _assert_safe_output(results_dir, output_dir, result_paths)
    staging_dir = _new_staging_directory(output_dir)
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    error_count = 0
    by_field: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    # Store compact binary digests rather than full recipient strings.  The
    # JSONL rows themselves are streamed so a 120k-receipt build does not
    # retain hundreds of thousands of Python dictionaries in RAM.
    seen_crops: dict[bytes, bytes] = {}
    conflicting_crops: set[bytes] = set()
    published = False
    try:
        with (
            (staging_dir / "candidates.jsonl").open("w", encoding="utf-8") as candidate_stream,
            (staging_dir / "rejected.jsonl").open("w", encoding="utf-8") as rejected_stream,
            (staging_dir / "build_errors.jsonl").open("w", encoding="utf-8") as errors_stream,
        ):
            for result_json in result_paths:
                result_path_text = result_json.resolve().as_posix()
                try:
                    payload = _result_payload(_load_json_document(result_json))
                    if payload is None:
                        continue
                    source = _source_path(payload, result_json)
                    if not allow_source_newer and source.stat().st_mtime_ns > result_json.stat().st_mtime_ns:
                        raise ValueError(
                            "source image is newer than its detector result; rerun detector inference or pass "
                            "--allow-source-newer only after auditing the change"
                        )
                    receipt_key = _source_key(source, key_pattern)
                    truth = truth_by_key.get(receipt_key)
                    if truth is None:
                        _write_jsonl_record(
                            rejected_stream,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "result_json": result_path_text,
                                "receipt_key": receipt_key,
                                "reason": "receipt_key_missing_from_truth",
                            },
                        )
                        rejected_count += 1
                        continue
                    source_rgb = load_upright_rgb(source)
                    rectified_rgb = reconstruct_rectified(payload, source_rgb)
                    # The local truth key is the stronger grouping signal: a
                    # receipt may be captured multiple times under different
                    # directories or timestamps.  Keep all of them in one
                    # split to prevent transaction-level leakage.
                    group_id = f"receipt_key:{receipt_key}"
                    split = _split_for_group(
                        group_id,
                        validation_ratio=validation_ratio,
                        test_ratio=test_ratio,
                        split_seed=split_seed,
                    )
                    for field in fields:
                        semantic_value = _normalise_truth_value(field, truth)
                        if semantic_value is None:
                            _write_jsonl_record(
                                rejected_stream,
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "result_json": result_path_text,
                                    "receipt_key": receipt_key,
                                    "field": field,
                                    "reason": "missing_or_invalid_truth_value",
                                },
                            )
                            rejected_count += 1
                            continue
                        match, invalid_candidates = _best_detection(
                            payload, field, min_detector_score=min_detector_score
                        )
                        for detail in invalid_candidates:
                            _write_jsonl_record(
                                rejected_stream,
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "result_json": result_path_text,
                                    "receipt_key": receipt_key,
                                    "field": field,
                                    "reason": "invalid_detection",
                                    "detail": detail,
                                },
                            )
                            rejected_count += 1
                        if match is None:
                            _write_jsonl_record(
                                rejected_stream,
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "result_json": result_path_text,
                                    "receipt_key": receipt_key,
                                    "field": field,
                                    "reason": "missing_or_low_score_detection",
                                },
                            )
                            rejected_count += 1
                            continue
                        bbox, score = match
                        crop_rgb = crop_field_with_margin(rectified_rgb, bbox)
                        if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 8:
                            _write_jsonl_record(
                                rejected_stream,
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "result_json": result_path_text,
                                    "receipt_key": receipt_key,
                                    "field": field,
                                    "reason": "empty_or_tiny_crop",
                                },
                            )
                            rejected_count += 1
                            continue
                        crop_sha256 = _crop_digest(crop_rgb)
                        crop_key = bytes.fromhex(crop_sha256)
                        label_signature = hashlib.sha256(
                            f"{field}\0{semantic_value}".encode("utf-8")
                        ).digest()
                        prior_signature = seen_crops.get(crop_key)
                        if prior_signature is not None:
                            reason = (
                                "conflicting_duplicate_crop"
                                if crop_key in conflicting_crops or prior_signature != label_signature
                                else "duplicate_crop"
                            )
                            if reason == "conflicting_duplicate_crop":
                                conflicting_crops.add(crop_key)
                            _write_jsonl_record(
                                rejected_stream,
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "result_json": result_path_text,
                                    "receipt_key": receipt_key,
                                    "field": field,
                                    "reason": reason,
                                    "crop_sha256": crop_sha256,
                                },
                            )
                            rejected_count += 1
                            continue
                        sample_id = _sample_id(result_json, field, receipt_key, bbox)
                        relative_image = Path("images") / field / f"{crop_sha256}.png"
                        image_path = staging_dir / relative_image
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        save_rgb(image_path, crop_rgb)
                        seen_crops[crop_key] = label_signature
                        _write_jsonl_record(
                            candidate_stream,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "id": sample_id,
                                "image": relative_image.as_posix(),
                                "field": field,
                                "text": semantic_value,
                                "semantic_value": semantic_value,
                                "receipt_key": receipt_key,
                                "bbox_rectified": [round(value, 3) for value in bbox],
                                "detector_score": round(score, 6),
                                "source": source.resolve().as_posix(),
                                "result_json": result_path_text,
                                "group_id": group_id,
                                "crop_sha256": crop_sha256,
                                "split": split,
                                "label_source": "transaction_truth",
                            },
                        )
                        candidate_count += 1
                except Exception as error:
                    _write_jsonl_record(
                        errors_stream,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "result_json": result_path_text,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        },
                    )
                    error_count += 1
                    if not continue_on_error:
                        raise

            # A later exact-pixel conflict invalidates the first candidate as
            # well.  Make a small second streaming pass so neither side can
            # become a training label, while keeping peak memory bounded.
            candidate_stream.flush()
            candidate_stream.close()
            with (
                (staging_dir / "candidates.jsonl").open("r", encoding="utf-8") as candidate_input,
                (staging_dir / "pseudo_labels.jsonl").open("w", encoding="utf-8") as accepted_stream,
            ):
                for line in candidate_input:
                    record: Any = json.loads(line)
                    if not isinstance(record, Mapping):  # Defensive: this file was written above.
                        raise ValueError("internal candidate manifest contains a non-object")
                    crop_sha256 = str(record.get("crop_sha256", ""))
                    try:
                        crop_key = bytes.fromhex(crop_sha256)
                    except ValueError:
                        raise ValueError("internal candidate manifest has an invalid crop digest") from None
                    if crop_key in conflicting_crops:
                        image_value = record.get("image")
                        field_value = record.get("field")
                        if not isinstance(field_value, str) or field_value not in DETECTION_CLASSES:
                            raise ValueError("internal candidate manifest has an invalid field")
                        expected_image = (Path("images") / field_value / f"{crop_sha256}.png").as_posix()
                        if image_value != expected_image:
                            raise ValueError("internal candidate manifest has an unexpected crop path")
                        image_path = (staging_dir / expected_image).resolve()
                        try:
                            image_path.relative_to(staging_dir.resolve())
                        except ValueError:
                            raise ValueError("internal candidate crop path escapes the staging directory") from None
                        image_path.unlink(missing_ok=True)
                        _write_jsonl_record(
                            rejected_stream,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "result_json": record.get("result_json"),
                                "receipt_key": record.get("receipt_key"),
                                "field": record.get("field"),
                                "reason": "conflicting_duplicate_crop_excluded",
                                "crop_sha256": crop_sha256,
                            },
                        )
                        rejected_count += 1
                        continue
                    _write_jsonl_record(accepted_stream, record)
                    accepted_count += 1
                    by_field[str(record["field"])] += 1
                    by_split[str(record["split"])] += 1
            (staging_dir / "candidates.jsonl").unlink()

        summary: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "receipt_field_transaction_truth_v1",
            "results_dir": results_dir.as_posix(),
            "truth_path": truth_path.resolve().as_posix(),
            "receipt_key_field": receipt_key_field,
            "source_key_regex": source_key_regex,
            "fields": list(fields),
            "selection": {
                "min_detector_score": min_detector_score,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "split_seed": split_seed,
                "allow_source_newer": allow_source_newer,
            },
            "counts": {
                "accepted": accepted_count,
                "candidate_accepted_before_conflict_filter": candidate_count,
                "rejected": rejected_count,
                "errors": error_count,
                "by_field": dict(sorted(by_field.items())),
                "by_split": {split: int(by_split[split]) for split in ("train", "val", "test")},
            },
            "warning": (
                "Local transaction truth is suitable for supervised training and held-out acceptance only when the "
                "receipt-key mapping is independently trustworthy. Filename keys are not proof that an image is authentic."
            ),
        }
        _atomic_write_json(staging_dir / "dataset_config.json", summary)
        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)
        published = True
        return summary
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid_fields = sorted(set(fields) - set(DETECTION_CLASSES))
    if not fields or invalid_fields:
        raise argparse.ArgumentTypeError(
            f"fields must be a non-empty subset of: {','.join(DETECTION_CLASSES)}"
        )
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Paddle-free field crops from detector JSON and local transaction truth")
    parser.add_argument("--results", type=Path, required=True, help="Python detector result directory produced with --ocr none")
    parser.add_argument("--truth", type=Path, required=True, help="Local JSONL transaction table")
    parser.add_argument("--output", type=Path, required=True, help="New empty crop/label dataset directory")
    parser.add_argument("--receipt-key-field", default="receipt_key")
    parser.add_argument("--source-key-regex", default=DEFAULT_SOURCE_KEY_REGEX)
    parser.add_argument(
        "--fields",
        type=_parse_fields,
        default=DETECTION_CLASSES,
        help="Comma-separated field subset; omit recipient_field for the four-model first phase",
    )
    parser.add_argument("--min-detector-score", type=float, default=DEFAULT_MIN_DETECTOR_SCORE)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", default="receipt-truth-v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-source-newer",
        action="store_true",
        help="Allow crops from images modified after the detector JSON was written; audit the source change first",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_truth_dataset(
            results_dir=args.results,
            truth_path=args.truth,
            output_dir=args.output,
            receipt_key_field=args.receipt_key_field,
            source_key_regex=args.source_key_regex,
            fields=args.fields,
            min_detector_score=args.min_detector_score,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            limit=args.limit,
            continue_on_error=args.continue_on_error,
            allow_source_newer=args.allow_source_newer,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"OCR transaction-truth dataset build failed:\n{error}") from None
    print(f"Exported {dict(summary['counts'])['accepted']} local-truth field crop(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
