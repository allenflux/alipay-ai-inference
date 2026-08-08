"""Audit the frozen Paddle recognizer on the production recipient value view.

The existing Paddle recipient baseline feeds the complete detector crop through
DB detection before parsing a visible ``收款方`` anchor.  That does not answer the
narrow fallback question: can the already-frozen recognizer read the same
right-side value view that the recipient-only student receives, without a text
detector or a row parser?

This module is deliberately evaluation-only.  It trims the immutable held-out
recipient crop by the v12/v13 contract's fixed 30 percent, calls the pinned
Paddle reader with ``det=False`` (angle classifier + recognizer), and compares
the cleaned complete recognition directly with the Paddle-derived recipient
target.  No parser, lexicon, full-layout OCR, test-time lookup, training, or
model mutation is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .ocr import OCRResult, clean_text
from .paddle_recipient_evaluate import (
    _atomic_json,
    _atomic_jsonl,
    _crop_sha256,
    _default_reader_factory,
    _default_runtime_probe,
    _load_recipient_records,
    _percentile,
    _sha256,
    _validate_device,
    _verify_reader_matches_bundle,
)


SCHEMA_VERSION = 1
KIND = "receipt_recipient_value_view_teacher_ceiling_v1"
COMPARISON_KIND = "receipt_recipient_value_view_teacher_comparison_v1"
VALUE_VIEW_LEFT_TRIM = 0.30
MINIMUM_CONFIDENCE = 0.80
TARGET_EXACT_MATCH = 0.90
INFERENCE_MODE = "value_view_cls_rec_no_parser"
EXPECTED_FULL_VAL_RECORDS = 6789
EVALUATION_SPLIT = "val"
_PUBLISHED_FILENAMES = (
    "comparisons.jsonl",
    "disagreements.jsonl",
    "summary.json",
)


FileBinding = tuple[int, int, int, str]


def _value_view(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the exact v12/v13 left-trim rounding to decoded RGB pixels."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("recipient crop must be an RGB image")
    height, width, _channels = image_rgb.shape
    if height <= 0 or width <= 0:
        raise ValueError("recipient crop must be non-empty")
    left = min(width - 1, max(0, int(round(width * VALUE_VIEW_LEFT_TRIM))))
    return np.ascontiguousarray(image_rgb[:, left:, :])


def _recognize_value(reader: object, image_rgb: np.ndarray) -> OCRResult:
    recognize = getattr(reader, "recognize", None)
    if not callable(recognize):
        raise TypeError("Paddle reader must expose recognize(image_rgb, det=False)")
    result = recognize(image_rgb, det=False)
    if not isinstance(result, OCRResult):
        raise TypeError("Paddle reader must return OCRResult")
    return result


def _validated_confidence(value: object, *, description: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a finite probability in [0, 1]")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} must be a finite probability in [0, 1]"
        ) from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{description} must be a finite probability in [0, 1]")
    return confidence


def _assert_crop_files_current(records: Sequence[Mapping[str, object]]) -> None:
    """Re-hash every selected crop after inference and again before publish."""

    for record in records:
        image_path = Path(str(record["image"]))
        if _sha256(image_path) != record["crop_file_sha256"]:
            raise ValueError(
                f"Recipient crop changed during value-view teacher evaluation: {image_path}"
            )


def _path_contains_or_is_contained_by(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _assert_fresh_nonreparse_output(raw_output: Path, resolved_output: Path) -> None:
    """Keep a requested output identity before resolving links/junctions."""

    if os.path.lexists(os.fspath(raw_output)):
        raise ValueError(
            f"Refusing to overwrite existing, symlink, or reparse value-view teacher output: "
            f"{raw_output}"
        )
    for ancestor in raw_output.parents:
        if os.path.lexists(os.fspath(ancestor)) and _is_reparse_path(ancestor):
            raise ValueError(
                f"Value-view teacher output must not traverse a symlink/junction/reparse point: "
                f"{ancestor}"
            )
    if raw_output.resolve() != resolved_output:
        raise ValueError("Value-view teacher output identity changed while resolving its path")


def _assert_evidence_path_isolated(
    *,
    evidence_path: Path,
    protected_paths: Mapping[str, Path],
) -> None:
    """Reject an evidence directory that can contain or mutate an input asset."""

    for label, protected in protected_paths.items():
        if _path_contains_or_is_contained_by(evidence_path, protected):
            raise ValueError(
                f"Value-view teacher evidence path overlaps protected {label}: "
                f"evidence={evidence_path}, protected={protected}"
            )


def _reader_live_asset_paths(reader: object) -> dict[str, Path]:
    """Resolve every live model/dictionary path exposed by PaddleOCR."""

    engine = getattr(reader, "_engine", None)
    args_object = getattr(engine, "args", None)
    if isinstance(args_object, Mapping):
        args = dict(args_object)
    else:
        try:
            args = vars(args_object) if args_object is not None else {}
        except TypeError:
            args = {}
    names = {
        "live detector model": "det_model_dir",
        "live classifier model": "cls_model_dir",
        "live recognizer model": "rec_model_dir",
        "live recognition dictionary": "rec_char_dict_path",
    }
    resolved: dict[str, Path] = {}
    for label, argument in names.items():
        value = args.get(argument)
        try:
            raw_path = os.fspath(value) if value is not None else ""
        except TypeError:
            raw_path = ""
        if isinstance(raw_path, str) and raw_path.strip():
            resolved[label] = Path(raw_path).expanduser().resolve()
    return resolved


def _bind_file(path: Path) -> FileBinding:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, _sha256(path))


def _matches_binding(path: Path, binding: FileBinding) -> bool:
    try:
        stat = path.stat()
        actual = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            _sha256(path),
        )
    except OSError:
        return False
    return actual == binding


def _remove_bound_files_and_empty_directory(
    directory: Path,
    bindings: Mapping[str, FileBinding],
) -> None:
    """Remove only unchanged files created by this run; never recurse."""

    for name, binding in bindings.items():
        path = directory / name
        if _matches_binding(path, binding):
            path.unlink()
    try:
        directory.rmdir()
    except (FileNotFoundError, OSError):
        # A non-cooperating file/process may now occupy the directory.  It is
        # not owned by this run and must never be recursively deleted.
        pass


def _publish_fresh_summary_last(
    *,
    stage: Path,
    output: Path,
    stage_bindings: Mapping[str, FileBinding] | None = None,
) -> None:
    """Publish into a newly reserved directory without POSIX rename overwrite.

    Reserve the destination with the filesystem's no-clobber ``mkdir``, then
    use same-directory hard links whose creation fails if a competing file is
    already present.  Link the two data files first and expose ``summary.json``
    last as the commit marker.  A directory without the summary is never valid
    evidence.  Failure cleanup removes only unchanged files bound to this
    publisher and never recursively deletes a directory.
    """

    bindings = (
        dict(stage_bindings)
        if stage_bindings is not None
        else {name: _bind_file(stage / name) for name in _PUBLISHED_FILENAMES}
    )
    if set(bindings) != set(_PUBLISHED_FILENAMES):
        raise ValueError("Value-view teacher stage bindings are incomplete")
    created = False
    linked: dict[str, FileBinding] = {}
    try:
        output.mkdir(parents=False, exist_ok=False)
        created = True
    except FileExistsError as error:
        raise ValueError(
            f"Refusing to overwrite existing value-view teacher output: {output}"
        ) from error
    try:
        for name in _PUBLISHED_FILENAMES:
            if not _matches_binding(stage / name, bindings[name]):
                raise ValueError(f"Value-view teacher staged file changed before publish: {name}")
            try:
                os.link(stage / name, output / name)
            except FileExistsError as error:
                raise ValueError(
                    f"Refusing to overwrite competing value-view teacher file: {output / name}"
                ) from error
            linked[name] = bindings[name]
        for name, binding in linked.items():
            if not _matches_binding(output / name, binding):
                raise ValueError(f"Published value-view teacher file changed: {name}")
        _remove_bound_files_and_empty_directory(stage, bindings)
    except Exception:
        if created:
            _remove_bound_files_and_empty_directory(output, linked)
        raise


def evaluate_value_view_teacher(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    split: str = EVALUATION_SPLIT,
    device: str = "cuda",
    limit: int | None = None,
    target_exact_match: float = TARGET_EXACT_MATCH,
    progress_every: int = 25,
    bundle_dir: Path | None = None,
    reader_factory: Callable[[str], object] = _default_reader_factory,
    runtime_probe: Callable[[object], Mapping[str, object]] = _default_runtime_probe,
) -> tuple[dict[str, object], bool]:
    """Evaluate the parser-free value view and summary-last publish evidence."""

    if split != EVALUATION_SPLIT:
        raise ValueError("recipient value-view teacher evaluation is hard-locked to val")
    normalized_device = _validate_device(device)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer when present")
    if limit is not None and limit >= EXPECTED_FULL_VAL_RECORDS:
        raise ValueError(
            "pilot limit must be smaller than 6789; omit limit and bind the "
            "immutable Paddle audit bundle for a full validation"
        )
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every <= 0
    ):
        raise ValueError("progress_every must be a positive integer")
    try:
        target = float(target_exact_match)
    except (TypeError, ValueError) as error:
        raise ValueError("target_exact_match must be a finite probability in (0, 1]") from error
    if not math.isfinite(target) or target != TARGET_EXACT_MATCH:
        raise ValueError(
            f"target_exact_match is hard-locked to exactly {TARGET_EXACT_MATCH:.2f}"
        )
    if limit is None:
        if bundle_dir is None:
            raise ValueError(
                "A full 6789-record value-view val ceiling must bind an immutable Paddle audit bundle"
            )

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    raw_output = Path(
        os.path.abspath(os.fspath(Path(output_dir).expanduser()))
    )
    if os.path.lexists(os.fspath(raw_output)):
        raise ValueError(
            f"Refusing to overwrite existing, symlink, or reparse value-view teacher output: "
            f"{raw_output}"
        )
    output = raw_output.resolve()
    _assert_fresh_nonreparse_output(raw_output, output)
    manifest = Path(manifest_path).expanduser().resolve()
    bundle_path = Path(bundle_dir).expanduser().resolve() if bundle_dir is not None else None
    protected_paths = {
        "manifest": manifest,
        "dataset root": root,
        **({"frozen bundle": bundle_path} if bundle_path is not None else {}),
    }
    _assert_evidence_path_isolated(
        evidence_path=output,
        protected_paths=protected_paths,
    )
    if os.path.lexists(output):
        raise ValueError(f"Refusing to overwrite existing value-view teacher output: {output}")
    manifest_sha256 = _sha256(manifest)
    records = _load_recipient_records(
        manifest_path=manifest,
        dataset_root=root,
        split=split,
        limit=limit,
        require_crop_hash=bundle_dir is not None,
    )
    if limit is not None and len(records) != limit:
        raise ValueError(
            f"Requested limit={limit} but selected exactly {len(records)} val recipient records"
        )
    if limit is None and len(records) != EXPECTED_FULL_VAL_RECORDS:
        raise ValueError(
            "Full value-view val ceiling must contain exactly "
            f"{EXPECTED_FULL_VAL_RECORDS} records, got {len(records)}"
        )

    reader = reader_factory(normalized_device)
    protected_paths.update(_reader_live_asset_paths(reader))
    _assert_evidence_path_isolated(
        evidence_path=output,
        protected_paths=protected_paths,
    )
    runtime = dict(runtime_probe(reader))
    active_device = str(runtime.get("active_paddle_device", ""))
    if normalized_device.startswith("cuda") and not active_device.startswith("gpu"):
        raise RuntimeError(
            f"Paddle CUDA was requested but active device is {active_device or 'unknown'}"
        )
    if bool(runtime.get("torch_imported")):
        raise RuntimeError("Paddle value-view teacher detected Torch in its worker process")
    bundle: dict[str, object] | None = None
    if bundle_dir is not None:
        bundle = _verify_reader_matches_bundle(reader, bundle_path)

    comparisons: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    for number, record in enumerate(records, start=1):
        image_path = Path(str(record["image"]))
        if _sha256(image_path) != record["crop_file_sha256"]:
            raise ValueError(f"Recipient crop changed after manifest validation: {image_path}")
        with Image.open(image_path) as image:
            image_rgb = np.asarray(image.convert("RGB")).copy()
        if _crop_sha256(image_rgb) != record["crop_sha256"]:
            raise ValueError(
                f"Decoded recipient crop changed before value-view recognition: {image_path}"
            )
        if _sha256(image_path) != record["crop_file_sha256"]:
            raise ValueError(
                f"Recipient crop file changed while decoding value view: {image_path}"
            )
        view = _value_view(image_rgb)
        started = perf_counter()
        result = _recognize_value(reader, view)
        elapsed_ms = (perf_counter() - started) * 1000.0
        confidence = _validated_confidence(
            result.confidence,
            description="Paddle recipient candidate confidence",
        )
        paddle_lines = []
        for line_number, (line_text, line_confidence) in enumerate(result.lines, start=1):
            paddle_lines.append(
                {
                    "text": line_text,
                    "confidence": _validated_confidence(
                        line_confidence,
                        description=f"Paddle recipient line {line_number} confidence",
                    ),
                }
            )
        confidence_eligible = (
            confidence is not None
            and confidence >= MINIMUM_CONFIDENCE
        )
        candidate = clean_text(result.text) or None
        if not confidence_eligible:
            candidate = None
        reference = str(record["reference_text"])
        exact = candidate == reference
        comparisons.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": COMPARISON_KIND,
                **record,
                "inference_mode": INFERENCE_MODE,
                "value_view_left_trim": VALUE_VIEW_LEFT_TRIM,
                "value_view_width": int(view.shape[1]),
                "raw_paddle_text": result.text,
                "candidate_text": candidate,
                "candidate_confidence": confidence,
                "minimum_candidate_confidence": MINIMUM_CONFIDENCE,
                "confidence_eligible": confidence_eligible,
                "paddle_lines": paddle_lines,
                "raw_exact": exact,
                "inference_ms": round(elapsed_ms, 4),
            }
        )
        latencies_ms.append(elapsed_ms)
        if number == 1 or number == len(records) or number % progress_every == 0:
            exact_count = sum(bool(row["raw_exact"]) for row in comparisons)
            print(
                f"recipient value-view teacher {number}/{len(records)} "
                f"exact={exact_count}/{number}={exact_count / number:.2%}"
            )

    _assert_crop_files_current(records)
    if _sha256(manifest) != manifest_sha256:
        raise ValueError(f"Manifest changed during value-view teacher evaluation: {manifest}")
    if bundle_dir is not None:
        bundle_after = _verify_reader_matches_bundle(reader, bundle_path)
        if bundle_after != bundle:
            raise ValueError("Frozen Paddle bundle or live source assets changed during evaluation")
        bundle = {**bundle_after, "verified_before_and_after": True}

    exact_matches = sum(bool(row["raw_exact"]) for row in comparisons)
    candidate_records = sum(row["candidate_text"] is not None for row in comparisons)
    exact_match = exact_matches / len(comparisons)
    candidate_coverage = candidate_records / len(comparisons)
    accepted = exact_match >= target and candidate_records == len(comparisons)

    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_fresh_nonreparse_output(raw_output, output)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    _assert_evidence_path_isolated(
        evidence_path=stage,
        protected_paths=protected_paths,
    )
    stage.mkdir(parents=False, exist_ok=False)
    comparisons_path = stage / "comparisons.jsonl"
    disagreements_path = stage / "disagreements.jsonl"
    summary: dict[str, object]
    stage_bindings: dict[str, FileBinding] = {}
    try:
        _atomic_jsonl(comparisons_path, comparisons)
        stage_bindings["comparisons.jsonl"] = _bind_file(comparisons_path)
        _atomic_jsonl(
            disagreements_path,
            [row for row in comparisons if not bool(row["raw_exact"])],
        )
        stage_bindings["disagreements.jsonl"] = _bind_file(disagreements_path)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "analysis_only": True,
            "production_route_authorized": False,
            "manifest": manifest.as_posix(),
            "manifest_sha256": manifest_sha256,
            "comparisons_sha256": _sha256(comparisons_path),
            "disagreements_sha256": _sha256(disagreements_path),
            "dataset_root": root.as_posix(),
            "evaluation_split": split,
            "records": len(comparisons),
            "limit": limit,
            "full_validation": limit is None,
            "expected_full_val_records": EXPECTED_FULL_VAL_RECORDS,
            "requested_device": normalized_device,
            "runtime": runtime,
            "frozen_bundle": bundle,
            "inference_mode": {
                "name": INFERENCE_MODE,
                "experimental": True,
                "input": "anchored_recipient_crop_left_trim_30_percent",
                "detection_enabled": False,
                "angle_classifier_enabled": True,
                "recognizer_enabled": True,
                "parser_enabled": False,
                "lexicon_enabled": False,
                "full_layout_enabled": False,
                "minimum_candidate_confidence": MINIMUM_CONFIDENCE,
            },
            "value_view_left_trim": VALUE_VIEW_LEFT_TRIM,
            "candidate_records": candidate_records,
            "candidate_coverage": candidate_coverage,
            "exact_matches": exact_matches,
            "exact_match": exact_match,
            "latency_ms": {
                "mean": sum(latencies_ms) / len(latencies_ms),
                "p50": _percentile(latencies_ms, 0.50),
                "p95": _percentile(latencies_ms, 0.95),
                "max": max(latencies_ms),
            },
            "acceptance": {
                "target_exact_match": TARGET_EXACT_MATCH,
                "requires_candidate_coverage": 1.0,
                "passed": accepted,
            },
            "decision": (
                "analysis_only_teacher_ceiling_pass_for_separate_shadow_experiment"
                if accepted
                else "analysis_only_teacher_ceiling_fail_stop"
            ),
            "warning": (
                "Teacher-parity uses held-out Paddle-derived labels, not independent human truth. "
                "A pass permits only a separately guarded recognizer-only shadow; it does not "
                "authorize a production fallback or training on this held-out split."
            ),
        }
        _assert_crop_files_current(records)
        if _sha256(manifest) != manifest_sha256:
            raise ValueError(
                f"Manifest changed while staging value-view teacher evidence: {manifest}"
            )
        if bundle_dir is not None:
            bundle_before_publish = _verify_reader_matches_bundle(reader, bundle_path)
            if bundle_before_publish != bundle_after:
                raise ValueError(
                    "Frozen Paddle bundle or live source assets changed before final publish"
                )
            bundle = {
                **bundle_before_publish,
                "verification_passes": 4,
                "verified_before_inference_after_inference_before_summary_and_before_publish": True,
            }
            summary["frozen_bundle"] = bundle
        _atomic_json(stage / "summary.json", summary)
        stage_bindings["summary.json"] = _bind_file(stage / "summary.json")
        _assert_crop_files_current(records)
        if _sha256(manifest) != manifest_sha256:
            raise ValueError(
                f"Manifest changed before value-view teacher evidence publish: {manifest}"
            )
        if bundle_dir is not None:
            bundle_at_publish = _verify_reader_matches_bundle(reader, bundle_path)
            if bundle_at_publish != bundle_before_publish:
                raise ValueError(
                    "Frozen Paddle bundle or live source assets changed at final publish closure"
                )
        _assert_evidence_path_isolated(
            evidence_path=stage,
            protected_paths=protected_paths,
        )
        _publish_fresh_summary_last(
            stage=stage,
            output=output,
            stage_bindings=stage_bindings,
        )
    finally:
        if stage.exists():
            _remove_bound_files_and_empty_directory(
                stage,
                stage_bindings,
            )
    return summary, accepted


def format_summary(summary: Mapping[str, object]) -> str:
    latency = summary.get("latency_ms")
    acceptance = summary.get("acceptance")
    if not isinstance(latency, Mapping) or not isinstance(acceptance, Mapping):
        raise ValueError("value-view teacher summary is invalid")
    return "\n".join(
        [
            "recipient_value_view_teacher_ceiling",
            "  route=value-view 30% left trim -> cls+rec; det=False; parser=False",
            f"  fixed candidate confidence floor={MINIMUM_CONFIDENCE:.2f}",
            f"  exact={summary.get('exact_matches')}/{summary.get('records')}="
            f"{float(summary.get('exact_match') or 0.0):.2%}; "
            f"coverage={summary.get('candidate_records')}/{summary.get('records')}="
            f"{float(summary.get('candidate_coverage') or 0.0):.2%}",
            f"  latency_ms=mean:{float(latency.get('mean') or 0.0):.2f}, "
            f"p50:{float(latency.get('p50') or 0.0):.2f}, "
            f"p95:{float(latency.get('p95') or 0.0):.2f}",
            f"  target={float(acceptance.get('target_exact_match') or 0.0):.2%}; "
            f"passed={acceptance.get('passed')}; decision={summary.get('decision')}",
            f"  output remains analysis-only; production_route_authorized="
            f"{summary.get('production_route_authorized')}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen Paddle cls+rec on the fixed recipient value view"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary, accepted = evaluate_value_view_teacher(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output,
            device=args.device,
            limit=args.limit,
            progress_every=args.progress_every,
            bundle_dir=args.bundle,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"Recipient value-view teacher audit failed: {error}") from error
    print(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if args.json
        else format_summary(summary)
    )
    if not accepted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
