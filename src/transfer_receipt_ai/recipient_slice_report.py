"""Read-only held-out error slices for the v12 recipient CTC head.

The unified ONNX evaluator deliberately reports one strict-exact value per
field.  That is the acceptance value, but it does not answer *why* an
open-text recipient field misses.  This module joins the evaluator's existing
``comparisons.jsonl`` with the original unified manifest and reports the
smallest useful diagnostic slices without running inference, changing a
checkpoint, or rewriting either input.

In particular, the report keeps two different notions separate:

* ``reference_has_oov_character`` comes from the evaluator / exported model
  charset.  A warm-start can retain a character that is absent from the
  current manifest's train split.
* ``min_train_character_support`` is recomputed from the manifest's train
  recipient labels.  A character observed once is in-vocabulary, but is still
  a long-tail learning problem.

The optional image-only geometry pass uses the same static trim arithmetic as
the v12 preprocessing contract.  It is evidence about the crop boundary, not
an OCR result and never changes the delivery preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .recipient_audit import RecipientCropGeometryAudit, audit_recipient_crop


REPORT_KIND = "receipt_recipient_slice_report_v1"
REPORT_SCHEMA_VERSION = 1
_RECIPIENT_FIELD = "recipient_field"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file while rejecting malformed/ambiguous diagnostic input."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{source}: no JSONL rows")
    return rows


def _require_nonempty_string(value: object, *, source: Path, row_label: str, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{row_label}: {key} must be a non-empty string")
    return value


def _optional_probability(value: object, *, source: Path, row_label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source}:{row_label}: paddle_confidence must be a finite probability")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}:{row_label}: paddle_confidence must be a finite probability") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{source}:{row_label}: paddle_confidence must be a finite probability")
    return probability


def _recipient_slot(record: Mapping[str, object]) -> Mapping[str, object] | None:
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        return None
    slot = slots.get(_RECIPIENT_FIELD)
    return slot if isinstance(slot, Mapping) else None


def _load_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, object]], Counter[str], dict[str, int]]:
    """Return recipient slots by receipt id and train-only character support."""
    source = Path(manifest_path).expanduser().resolve()
    records = _read_jsonl(source)
    slots_by_id: dict[str, dict[str, object]] = {}
    train_character_support: Counter[str] = Counter()
    summary: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        row_label = str(line_number)
        receipt_id = _require_nonempty_string(record.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate manifest id {receipt_id!r}")
        seen_ids.add(receipt_id)
        summary["manifest_records"] += 1
        slot = _recipient_slot(record)
        if slot is None:
            continue
        text = _require_nonempty_string(slot.get("text"), source=source, row_label=row_label, key="recipient text")
        split = _require_nonempty_string(record.get("split"), source=source, row_label=row_label, key="split")
        confidence = _optional_probability(slot.get("paddle_confidence"), source=source, row_label=row_label)
        image = slot.get("image")
        if image is not None and (not isinstance(image, str) or not image):
            raise ValueError(f"{source}:{row_label}: recipient image must be a non-empty string when present")
        slots_by_id[receipt_id] = {
            "text": text,
            "split": split,
            "paddle_confidence": confidence,
            "image": image,
        }
        summary["manifest_recipient_records"] += 1
        if split == "train":
            train_character_support.update(text)
            summary["manifest_train_recipient_records"] += 1
    if not train_character_support:
        raise ValueError(f"{source}: no train recipient characters")
    return slots_by_id, train_character_support, {key: int(value) for key, value in summary.items()}


def _levenshtein_distance(reference: str, candidate: str) -> int:
    """Return a small dependency-free Unicode edit distance fallback."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def _comparison_boolean(value: object, *, source: Path, row_label: str, key: str, fallback: bool) -> bool:
    if value is None:
        return fallback
    if not isinstance(value, bool):
        raise ValueError(f"{source}:{row_label}: {key} must be a boolean")
    return value


def _comparison_cer_edits(
    value: object,
    *,
    reference_text: str,
    candidate_text: str,
    source: Path,
    row_label: str,
) -> int:
    if value is None:
        return _levenshtein_distance(reference_text, candidate_text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}:{row_label}: cer_edits must be a non-negative integer")
    return value


def _resolve_image_path(
    *,
    comparison: Mapping[str, object],
    manifest_slot: Mapping[str, object],
    dataset_root: Path | None,
) -> Path | None:
    """Resolve the evaluator's absolute image first, then a manifest fallback."""
    raw_image = comparison.get("image")
    if not isinstance(raw_image, str) or not raw_image:
        raw_image = manifest_slot.get("image")
    if not isinstance(raw_image, str) or not raw_image:
        return None
    candidate = Path(raw_image).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if dataset_root is None:
        return None
    return (dataset_root / candidate).resolve()


def _bucket_oov(row: Mapping[str, object]) -> str:
    return "oov" if bool(row["reference_has_oov_character"]) else "in_vocab"


def _bucket_length(row: Mapping[str, object]) -> str:
    length = int(row["reference_length"])
    if length <= 4:
        return "1-4"
    if length <= 8:
        return "5-8"
    if length <= 12:
        return "9-12"
    return "13+"


def _bucket_min_character_support(row: Mapping[str, object]) -> str:
    support = int(row["min_train_character_support"])
    if support <= 0:
        return "0"
    if support == 1:
        return "1"
    if support <= 3:
        return "2-3"
    if support <= 9:
        return "4-9"
    return "10+"


def _bucket_confidence(row: Mapping[str, object]) -> str:
    confidence = row["paddle_confidence"]
    if confidence is None:
        return "missing"
    value = float(confidence)
    if value < 0.95:
        return "<0.95"
    if value < 0.98:
        return "0.95-<0.98"
    return ">=0.98"


def _bucket_cer_edits(row: Mapping[str, object]) -> str:
    edits = int(row["cer_edits"])
    if edits == 0:
        return "0"
    if edits == 1:
        return "1"
    return "2+"


def _bucket_candidate_empty(row: Mapping[str, object]) -> str:
    return "empty" if bool(row["candidate_empty"]) else "nonempty"


def _bucket_cut_window_ink(row: Mapping[str, object]) -> str:
    value = row["cut_window_has_ink"]
    if value is None:
        return "unavailable"
    return "ink" if bool(value) else "no_ink"


def _slice_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate strict exact, edit distance, empties, and geometry evidence."""
    count = len(rows)
    if not count:
        return {
            "records": 0,
            "exact_matches": 0,
            "raw_exact_match": None,
            "cer_edits": 0,
            "reference_characters": 0,
            "micro_cer": None,
            "macro_cer": None,
            "empty_candidate_records": 0,
            "empty_candidate_rate": None,
            "confidence_records": 0,
            "mean_paddle_confidence": None,
            "geometry_records": 0,
            "cut_window_ink_records": 0,
            "cut_window_ink_rate": None,
            "nearest_blank_gap_touch_records": 0,
            "nearest_blank_gap_touch_rate": None,
        }
    exact_matches = sum(bool(row["raw_exact"]) for row in rows)
    edits = sum(int(row["cer_edits"]) for row in rows)
    characters = sum(int(row["reference_length"]) for row in rows)
    macro_cer = sum(int(row["cer_edits"]) / max(1, int(row["reference_length"])) for row in rows) / count
    empty = sum(bool(row["candidate_empty"]) for row in rows)
    confidences = [float(row["paddle_confidence"]) for row in rows if row["paddle_confidence"] is not None]
    geometry_rows = [row for row in rows if row["cut_window_has_ink"] is not None]
    cut_ink = sum(bool(row["cut_window_has_ink"]) for row in geometry_rows)
    gap_touch = sum(bool(row["nearest_blank_gap_touches_trim"]) for row in geometry_rows)
    geometry_count = len(geometry_rows)
    return {
        "records": count,
        "exact_matches": exact_matches,
        "raw_exact_match": exact_matches / count,
        "cer_edits": edits,
        "reference_characters": characters,
        "micro_cer": edits / max(1, characters),
        "macro_cer": macro_cer,
        "empty_candidate_records": empty,
        "empty_candidate_rate": empty / count,
        "confidence_records": len(confidences),
        "mean_paddle_confidence": sum(confidences) / len(confidences) if confidences else None,
        "geometry_records": geometry_count,
        "cut_window_ink_records": cut_ink,
        "cut_window_ink_rate": cut_ink / geometry_count if geometry_count else None,
        "nearest_blank_gap_touch_records": gap_touch,
        "nearest_blank_gap_touch_rate": gap_touch / geometry_count if geometry_count else None,
    }


def _group_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    bucket: Callable[[Mapping[str, object]], str],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(bucket(row), []).append(row)
    return {name: _slice_metrics(grouped[name]) for name in sorted(grouped)}


def _geometry_result(
    image_path: Path | None,
    *,
    left_trim_fraction: float,
    audit_crop: Callable[..., RecipientCropGeometryAudit],
) -> tuple[bool | None, bool | None, str | None]:
    """Return image-only trim evidence without making broken images fatal."""
    if image_path is None:
        return None, None, "image_path_unavailable"
    try:
        audit = audit_crop(image_path, left_trim_ratio=left_trim_fraction)
    except (OSError, ValueError) as error:
        return None, None, type(error).__name__
    gap = audit.nearest_blank_gap
    return bool(audit.cut_window_has_ink), bool(gap is not None and gap.touches_trim_boundary), None


def build_recipient_slice_report(
    *,
    comparisons_path: Path,
    manifest_path: Path,
    dataset_root: Path | None = None,
    left_trim_fraction: float = 0.30,
    include_geometry: bool = True,
    audit_crop: Callable[..., RecipientCropGeometryAudit] = audit_recipient_crop,
) -> dict[str, object]:
    """Build a side-effect-free recipient diagnostic report from existing files.

    ``audit_crop`` is injectable solely to keep the aggregation testable; the
    CLI always uses :func:`recipient_audit.audit_recipient_crop`.
    """
    try:
        trim = float(left_trim_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("left_trim_fraction must be a finite value in [0, 1)") from error
    if not math.isfinite(trim) or not 0.0 <= trim < 1.0:
        raise ValueError("left_trim_fraction must be a finite value in [0, 1)")
    source = Path(comparisons_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    slots_by_id, train_support, manifest_summary = _load_manifest(manifest_path)
    comparison_rows = _read_jsonl(source)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    geometry_error_counts: Counter[str] = Counter()
    oov_source_counts: Counter[str] = Counter()
    manifest_reference_mismatches = 0
    for line_number, comparison in enumerate(comparison_rows, start=1):
        if comparison.get("field") != _RECIPIENT_FIELD:
            continue
        row_label = str(line_number)
        receipt_id = _require_nonempty_string(comparison.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate recipient comparison id {receipt_id!r}")
        seen_ids.add(receipt_id)
        manifest_slot = slots_by_id.get(receipt_id)
        if manifest_slot is None:
            raise ValueError(f"{source}:{row_label}: recipient comparison id {receipt_id!r} is absent from manifest")
        reference_text = _require_nonempty_string(
            comparison.get("reference_text"), source=source, row_label=row_label, key="reference_text"
        )
        raw_candidate = comparison.get("candidate_text")
        if raw_candidate is None:
            candidate_text = ""
        elif isinstance(raw_candidate, str):
            candidate_text = raw_candidate
        else:
            raise ValueError(f"{source}:{row_label}: candidate_text must be a string or null")
        raw_exact = _comparison_boolean(
            comparison.get("raw_exact"),
            source=source,
            row_label=row_label,
            key="raw_exact",
            fallback=reference_text == candidate_text,
        )
        cer_edits = _comparison_cer_edits(
            comparison.get("cer_edits"),
            reference_text=reference_text,
            candidate_text=candidate_text,
            source=source,
            row_label=row_label,
        )
        min_support = min((int(train_support[character]) for character in reference_text), default=0)
        mean_support = sum(int(train_support[character]) for character in reference_text) / max(1, len(reference_text))
        raw_oov = comparison.get("reference_has_oov_character")
        if raw_oov is None:
            reference_has_oov = min_support == 0
            oov_source_counts["manifest_train_support_fallback"] += 1
        elif isinstance(raw_oov, bool):
            reference_has_oov = raw_oov
            oov_source_counts["exported_evaluator"] += 1
        else:
            raise ValueError(f"{source}:{row_label}: reference_has_oov_character must be a boolean")
        cut_window_has_ink: bool | None = None
        nearest_blank_gap_touches_trim: bool | None = None
        if include_geometry:
            image_path = _resolve_image_path(
                comparison=comparison,
                manifest_slot=manifest_slot,
                dataset_root=root,
            )
            cut_window_has_ink, nearest_blank_gap_touches_trim, geometry_error = _geometry_result(
                image_path,
                left_trim_fraction=trim,
                audit_crop=audit_crop,
            )
            if geometry_error is not None:
                geometry_error_counts[geometry_error] += 1
        else:
            geometry_error_counts["disabled"] += 1
        manifest_text = str(manifest_slot["text"])
        manifest_reference_mismatches += int(manifest_text != reference_text)
        rows.append(
            {
                "id": receipt_id,
                "reference_length": len(reference_text),
                "raw_exact": raw_exact,
                "cer_edits": cer_edits,
                "candidate_empty": not candidate_text,
                "reference_has_oov_character": reference_has_oov,
                "min_train_character_support": min_support,
                "mean_train_character_support": mean_support,
                "paddle_confidence": manifest_slot["paddle_confidence"],
                "cut_window_has_ink": cut_window_has_ink,
                "nearest_blank_gap_touches_trim": nearest_blank_gap_touches_trim,
            }
        )
    if not rows:
        raise ValueError(f"{source}: no recipient_field comparisons")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "comparisons": source.as_posix(),
        "manifest": Path(manifest_path).expanduser().resolve().as_posix(),
        "dataset_root": root.as_posix() if root is not None else None,
        "recipient_field": _RECIPIENT_FIELD,
        "left_trim_fraction": trim,
        "geometry_requested": include_geometry,
        "manifest_summary": {
            **manifest_summary,
            "train_recipient_character_count": len(train_support),
            "manifest_reference_mismatch_records": manifest_reference_mismatches,
        },
        "comparison_summary": {
            "recipient_comparison_records": len(rows),
            "oov_source_counts": dict(sorted(oov_source_counts.items())),
            "geometry_error_counts": dict(sorted(geometry_error_counts.items())),
        },
        "slice_definitions": {
            "oov": "exported evaluator's OOV flag when available; otherwise current manifest train support == 0",
            "reference_length": "Unicode codepoint count: 1-4, 5-8, 9-12, 13+",
            "min_train_character_support": "minimum current-manifest train count over characters: 0, 1, 2-3, 4-9, 10+",
            "paddle_confidence": "recipient slot teacher confidence: <0.95, 0.95-<0.98, >=0.98, missing",
            "cer_edits": "whole-string Levenshtein edits: 0, 1, 2+",
            "candidate_empty": "whether greedy CTC emitted an empty candidate",
            "cut_window_ink": "image-only foreground at the current static left-trim boundary: ink, no_ink, unavailable",
        },
        "overall": _slice_metrics(rows),
        "slices": {
            "oov": _group_metrics(rows, bucket=_bucket_oov),
            "reference_length": _group_metrics(rows, bucket=_bucket_length),
            "min_train_character_support": _group_metrics(rows, bucket=_bucket_min_character_support),
            "paddle_confidence": _group_metrics(rows, bucket=_bucket_confidence),
            "cer_edits": _group_metrics(rows, bucket=_bucket_cer_edits),
            "candidate_empty": _group_metrics(rows, bucket=_bucket_candidate_empty),
            "cut_window_ink": _group_metrics(rows, bucket=_bucket_cut_window_ink),
        },
        "warning": (
            "This is a read-only slice of held-out Paddle-teacher parity, not human-truth accuracy. "
            "Geometry is image-only evidence about the current trim and does not modify preprocessing or ONNX."
        ),
    }


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def format_recipient_slice_report(report: Mapping[str, object]) -> str:
    """Render a concise terminal report suitable for the requested screenshot."""
    overall = report.get("overall")
    slices = report.get("slices")
    comparison_summary = report.get("comparison_summary")
    if not isinstance(overall, Mapping) or not isinstance(slices, Mapping) or not isinstance(comparison_summary, Mapping):
        raise ValueError("recipient slice report is invalid")
    lines = [
        "recipient_slice_report",
        f"  comparisons={report.get('comparisons')}",
        f"  manifest={report.get('manifest')}",
        f"  records={overall.get('records')} exact={overall.get('exact_matches')} "
        f"strict={_format_rate(overall.get('raw_exact_match'))} "
        f"micro_cer={float(overall.get('micro_cer') or 0.0):.4f} "
        f"empty={overall.get('empty_candidate_records')}/{overall.get('records')}="
        f"{_format_rate(overall.get('empty_candidate_rate'))}",
        f"  trim={float(report.get('left_trim_fraction') or 0.0):g} "
        f"geometry={overall.get('geometry_records')}/{overall.get('records')} "
        f"cut_ink={overall.get('cut_window_ink_records')}/{overall.get('geometry_records')}="
        f"{_format_rate(overall.get('cut_window_ink_rate'))}",
    ]
    geometry_errors = comparison_summary.get("geometry_error_counts")
    if isinstance(geometry_errors, Mapping) and geometry_errors:
        lines.append("  geometry_unavailable=" + ", ".join(f"{key}:{value}" for key, value in geometry_errors.items()))
    for name in (
        "oov",
        "reference_length",
        "min_train_character_support",
        "paddle_confidence",
        "cer_edits",
        "candidate_empty",
        "cut_window_ink",
    ):
        groups = slices.get(name)
        if not isinstance(groups, Mapping):
            continue
        lines.append(f"  [{name}]")
        for bucket, metrics in groups.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"    {bucket}: {metrics.get('exact_matches')}/{metrics.get('records')}="
                f"{_format_rate(metrics.get('raw_exact_match'))}; "
                f"micro_cer={float(metrics.get('micro_cer') or 0.0):.4f}; "
                f"empty={_format_rate(metrics.get('empty_candidate_rate'))}"
            )
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only recipient error slices from existing ONNX comparisons and a unified manifest"
    )
    parser.add_argument("--comparisons", type=Path, required=True, help="ONNX evaluator comparisons.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="unified_fields.jsonl used by that evaluator")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="crop root used only if comparisons/manifest image paths are relative",
    )
    parser.add_argument(
        "--left-trim",
        type=float,
        default=0.30,
        help="current v12 recipient left-crop fraction used for image-only geometry evidence",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="do not read crop images; emit unavailable geometry rows instead",
    )
    parser.add_argument("--output", type=Path, help="new JSON report path; refuses to overwrite")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report instead of the compact text table")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = build_recipient_slice_report(
            comparisons_path=args.comparisons,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            left_trim_fraction=args.left_trim,
            include_geometry=not args.skip_geometry,
        )
        if args.output is not None:
            _atomic_write_json(args.output, report)
    except (OSError, ValueError) as error:
        raise SystemExit(f"recipient slice report failed: {error}") from error
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_recipient_slice_report(report))
        if args.output is not None:
            print(f"Wrote recipient slice report to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":  # pragma: no cover - exercised through the hyphenated wrapper.
    main()
