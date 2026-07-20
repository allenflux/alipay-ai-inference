"""Batch inference CLI: raw image → circles + extracted transfer fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from .geometry import RectificationOptions
from .labels import DETECTION_CLASSES
from .model import LRCNNPredictor
from .ocr import PaddleOCRReader
from .pipeline import ReceiptPipeline, write_receipt_result
from .prepare import iter_image_paths, load_corrections, parse_max_side
from .status_style import (
    STATUS_STYLE_SCHEMA_VERSION,
    StatusStylePredictor,
    status_style_checkpoint_signature,
    status_style_tags,
)


STATUS_STYLE_MARGIN_RATIO = 0.30


def _parse_orientation(value: str) -> int | None:
    if value == "auto":
        return None
    degrees = int(value)
    if degrees not in {0, 90, 180, 270}:
        raise argparse.ArgumentTypeError("orientation must be auto, 0, 90, 180 or 270")
    return degrees


def _override_for(corrections: dict[str, dict[str, Any]], relative_path: Path, source_path: Path) -> dict[str, Any]:
    return corrections.get(relative_path.as_posix(), corrections.get(source_path.name, {}))


def _shard_for(relative_path: Path, shard_count: int) -> int:
    """Assign a path to a stable shard, even when new input files are added."""
    digest = _selection_key(relative_path)
    return int.from_bytes(digest[:8], "big") % shard_count


def _selection_key(relative_path: Path) -> bytes:
    """Give each relative path a deterministic pseudo-random ordering key."""
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).digest()


def _manifest_paths(output_dir: Path, shard_index: int, shard_count: int) -> tuple[Path, Path, Path]:
    suffix = "" if shard_count == 1 else f".shard-{shard_index:03d}-of-{shard_count:03d}"
    return (
        output_dir / f"inference_manifest{suffix}.json",
        output_dir / f"inference_manifest{suffix}.jsonl",
        output_dir / f"inference_errors{suffix}.jsonl",
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_output_names(image_paths: list[Path], root: Path) -> None:
    seen: dict[str, Path] = {}
    for image_path in image_paths:
        relative_stem = image_path.relative_to(root).with_suffix("")
        key = relative_stem.as_posix().casefold()
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Input images would overwrite the same output stem: "
                f"{previous} and {image_path}. Rename one source image before bulk inference."
            )
        seen[key] = image_path


def _committed_result_exists(
    source_path: Path,
    output_stem: Path,
    *,
    status_style_model: Mapping[str, object] | None = None,
    status_style_inference_config: Mapping[str, object] | None = None,
) -> bool:
    result_path = output_stem.with_suffix(".json")
    rectified_path = output_stem.with_name(output_stem.name + "_rectified_annotated.jpg")
    original_path = output_stem.with_name(output_stem.name + "_original_annotated.jpg")
    if not all(path.is_file() for path in (result_path, rectified_path, original_path)):
        return False
    if result_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("fields"), dict):
        return False
    if status_style_model is None:
        return True
    status_style = payload.get("status_style")
    tags = payload.get("tags")
    return (
        isinstance(status_style, dict)
        and status_style.get("schema_version") == STATUS_STYLE_SCHEMA_VERSION
        and status_style.get("model") == dict(status_style_model)
        and status_style.get("inference_config") == dict(status_style_inference_config or {})
        and isinstance(tags, dict)
        and tags == status_style_tags(status_style)
    )


def _written_record(source_path: Path, output_stem: Path, *, status: str) -> dict[str, str]:
    return {
        "source": source_path.resolve().as_posix(),
        "result": output_stem.with_suffix(".json").resolve().as_posix(),
        "annotated_rectified": output_stem.with_name(output_stem.name + "_rectified_annotated.jpg").resolve().as_posix(),
        "annotated_original": output_stem.with_name(output_stem.name + "_original_annotated.jpg").resolve().as_posix(),
        "status": status,
    }


def _require_five_fields(result: Any) -> None:
    detected = {item.detection.label for item in result.detections}
    missing = [label for label in DETECTION_CLASSES if label not in detected]
    if missing or len(result.detections) != len(DETECTION_CLASSES):
        raise ValueError(
            "incomplete detection: expected exactly five fields; "
            f"found={len(result.detections)}, missing={','.join(missing) or 'none'}"
        )


def run_inference(
    *,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    score_threshold: float = 0.50,
    device: str = "auto",
    use_ocr: bool = True,
    corrections: dict[str, dict[str, Any]] | None = None,
    orientation_degrees: int | None = None,
    prefer_portrait: bool = True,
    auto_screen: bool = True,
    max_side: int = 1600,
    use_ocr_orientation: bool = False,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    require_complete: bool = False,
    limit: int | None = None,
    status_style_checkpoint: Path | None = None,
    status_confidence_threshold: float = 0.80,
    status_absent_confidence_threshold: float = 0.95,
) -> list[dict[str, str]]:
    """Process an image or image tree and write one result bundle per raw image."""
    all_image_paths = list(iter_image_paths(input_path))
    if not all_image_paths:
        raise ValueError(f"No supported images found under {input_path}")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count must be positive and shard_index must be in [0, shard_count)")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= status_confidence_threshold <= 1.0:
        raise ValueError("status_confidence_threshold must be between 0 and 1")
    if not 0.0 <= status_absent_confidence_threshold <= 1.0:
        raise ValueError("status_absent_confidence_threshold must be between 0 and 1")
    if status_absent_confidence_threshold < status_confidence_threshold:
        raise ValueError("status_absent_confidence_threshold must be at least status_confidence_threshold")
    root = input_path.parent if input_path.is_file() else input_path
    _validate_output_names(all_image_paths, root)
    image_paths = sorted(
        (
            path
            for path in all_image_paths
            if _shard_for(path.relative_to(root), shard_count) == shard_index
        ),
        key=lambda path: (_selection_key(path.relative_to(root)), path.relative_to(root).as_posix()),
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    corrections = corrections or {}
    ocr = PaddleOCRReader() if use_ocr else None
    predictor = LRCNNPredictor(checkpoint, device=device, score_threshold=score_threshold)
    status_style_predictor: StatusStylePredictor | None = None
    status_style_model: dict[str, object] | None = None
    status_style_inference_config: dict[str, object] | None = None
    if status_style_checkpoint is not None:
        # Hash and load exactly once per batch.  The signature makes resume
        # safe when a checkpoint or one of its decision thresholds changes.
        status_style_model = status_style_checkpoint_signature(status_style_checkpoint)
        status_style_inference_config = {
            "confidence_threshold": float(status_confidence_threshold),
            "absent_confidence_threshold": float(status_absent_confidence_threshold),
            "margin_ratio": STATUS_STYLE_MARGIN_RATIO,
        }
        status_style_predictor = StatusStylePredictor(
            status_style_checkpoint,
            device=device,
            confidence_threshold=status_confidence_threshold,
            absent_confidence_threshold=status_absent_confidence_threshold,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_jsonl_path, errors_jsonl_path = _manifest_paths(output_dir, shard_index, shard_count)
    manifest: list[dict[str, str]] = []
    failure_count = 0
    with manifest_jsonl_path.open("w", encoding="utf-8") as manifest_jsonl, errors_jsonl_path.open(
        "w", encoding="utf-8"
    ) as errors_jsonl:
        for source_path in tqdm(image_paths, desc=f"Running LRCNN inference shard {shard_index + 1}/{shard_count}"):
            relative_path = source_path.relative_to(root)
            override = _override_for(corrections, relative_path, source_path)
            if override.get("skip") is True:
                continue
            output_stem = (output_dir / relative_path).with_suffix("")
            if skip_existing and _committed_result_exists(
                source_path,
                output_stem,
                status_style_model=status_style_model,
                status_style_inference_config=status_style_inference_config,
            ):
                record = _written_record(source_path, output_stem, status="skipped_existing")
                manifest.append(record)
                manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_jsonl.flush()
                continue
            try:
                options = RectificationOptions(
                    orientation_degrees=override.get("orientation_degrees", orientation_degrees),
                    prefer_portrait=bool(override.get("prefer_portrait", prefer_portrait)),
                    auto_screen=bool(override.get("auto_screen", auto_screen)),
                    screen_quad=override.get("screen_quad"),
                    max_side=int(override.get("max_side", max_side)),
                    orientation_scorer=ocr.orientation_score if ocr and use_ocr_orientation else None,
                )
                pipeline = ReceiptPipeline(
                    predictor,
                    ocr=ocr,
                    rectification_options=options,
                    status_style_predictor=status_style_predictor,
                    status_style_model=status_style_model,
                    status_style_inference_config=status_style_inference_config,
                    status_style_margin_ratio=STATUS_STYLE_MARGIN_RATIO,
                )
                result = pipeline.run(source_path)
                if require_complete:
                    _require_five_fields(result)
                write_receipt_result(result, output_stem)
                record = _written_record(source_path, output_stem, status="written")
                manifest.append(record)
                manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_jsonl.flush()
            except Exception as error:
                failure_count += 1
                error_record = {
                    "source": source_path.resolve().as_posix(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                errors_jsonl.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                errors_jsonl.flush()
                if not continue_on_error:
                    raise
    _write_json_atomic(manifest_path, manifest)
    if failure_count:
        print(f"WARNING: {failure_count} image(s) failed; see {errors_jsonl_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, OCR and circle transfer receipt fields")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--status-style-checkpoint",
        type=Path,
        help="Optional status-style classifier checkpoint; omit to keep the original five-field v1 output",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw image or raw-image directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument(
        "--status-confidence-threshold",
        type=float,
        default=0.80,
        help="Minimum confidence for Android/iOS status-style tags",
    )
    parser.add_argument(
        "--status-absent-confidence-threshold",
        type=float,
        default=0.95,
        help="Stricter minimum confidence for the no-check suspected-fake tag",
    )
    parser.add_argument("--ocr", choices=("paddle", "none"), default="paddle")
    parser.add_argument("--corrections", type=Path, help="Optional per-image manual correction JSON")
    parser.add_argument("--orientation", type=_parse_orientation, default=None)
    parser.add_argument("--landscape-ok", action="store_true")
    parser.add_argument("--no-auto-screen", action="store_true")
    parser.add_argument("--ocr-orientation", action="store_true", help="Use OCR to distinguish all four orientations")
    parser.add_argument("--skip-existing", action="store_true", help="Resume by skipping images whose result JSON exists")
    parser.add_argument("--continue-on-error", action="store_true", help="Record a bad image and continue the batch")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based stable shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of stable input shards")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only write result bundles when all five required fields are detected",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many images from the selected shard")
    parser.add_argument(
        "--max-side",
        type=parse_max_side,
        default=1600,
        help="Maximum long edge after correction; use 0 to keep original resolution",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if not 0.0 <= args.status_confidence_threshold <= 1.0:
        raise SystemExit("--status-confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.status_absent_confidence_threshold <= 1.0:
        raise SystemExit("--status-absent-confidence-threshold must be between 0 and 1")
    if args.status_absent_confidence_threshold < args.status_confidence_threshold:
        raise SystemExit(
            "--status-absent-confidence-threshold must be at least --status-confidence-threshold"
        )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-count must be positive and --shard-index must be in [0, shard-count)")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    outputs = run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        score_threshold=args.score_threshold,
        use_ocr=args.ocr == "paddle",
        corrections=load_corrections(args.corrections),
        orientation_degrees=args.orientation,
        prefer_portrait=not args.landscape_ok,
        auto_screen=not args.no_auto_screen,
        max_side=args.max_side,
        use_ocr_orientation=args.ocr_orientation,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        require_complete=args.require_complete,
        limit=args.limit,
        status_style_checkpoint=args.status_style_checkpoint,
        status_confidence_threshold=args.status_confidence_threshold,
        status_absent_confidence_threshold=args.status_absent_confidence_threshold,
    )
    print(f"Wrote {len(outputs)} inference result bundle(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
