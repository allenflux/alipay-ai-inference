"""Read-only feasibility and integrity audit for v12 recipient recognition.

The held-out ONNX evaluator tells us the current strict-exact result, but a
low result alone cannot distinguish between a recognizer capacity problem and
a target set that is mostly unseen by the training data.  This module joins an
existing ``comparisons.jsonl`` with the immutable unified manifest and answers
that question without loading a model, opening an image, changing a checkpoint,
or rewriting either input.

Its *closed-set ceiling* is deliberately an oracle, not a deployed decoding
algorithm: it retains every current exact prediction and assumes a perfect
train-only selector could correct every remaining held-out reference whose
full name occurs in the training split.  A ceiling below the requested target
therefore rules out a train-only name dictionary as a route to that target;
a ceiling above it only establishes that a new candidate-selection design is
worth testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT_KIND = "receipt_recipient_data_audit_v1"
REPORT_SCHEMA_VERSION = 1
_RECIPIENT_FIELD = "recipient_field"
_SPLITS = frozenset(("train", "val", "test"))
_SUPPORT_THRESHOLDS = (1, 2, 3, 5, 10, 25)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load one non-empty JSONL file with unambiguous error messages."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{source}: no JSONL rows")
    return rows


def _require_string(value: object, *, source: Path, row_label: str, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{row_label}: {key} must be a non-empty string")
    return value


def _optional_string(value: object, *, source: Path, row_label: str, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{row_label}: {key} must be a non-empty string when present")
    return value


def _recipient_slot(record: Mapping[str, object]) -> Mapping[str, object] | None:
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        return None
    slot = slots.get(_RECIPIENT_FIELD)
    return slot if isinstance(slot, Mapping) else None


def _record_key(
    values: dict[str, set[str]],
    *,
    key: str | None,
    split: str,
) -> None:
    """Track non-empty split identities while treating absent optional keys as absent."""
    if key:
        values.setdefault(key, set()).add(split)


def _collision_summary(values: Mapping[str, set[str]]) -> dict[str, object]:
    collisions = [(key, sorted(splits)) for key, splits in values.items() if len(splits) > 1]
    collisions.sort(key=lambda item: item[0])
    return {
        "distinct_keys": len(values),
        "cross_split_keys": len(collisions),
        "examples": [
            {"key": key, "splits": splits}
            for key, splits in collisions[:5]
        ],
    }


def _label_conflict_summary(values: Mapping[str, set[str]]) -> dict[str, object]:
    conflicts = [(key, sorted(texts)) for key, texts in values.items() if len(texts) > 1]
    conflicts.sort(key=lambda item: item[0])
    return {
        "distinct_keys": len(values),
        "conflicting_keys": len(conflicts),
        "examples": [
            {"key": key, "texts": texts}
            for key, texts in conflicts[:5]
        ],
    }


def _load_manifest(manifest_path: Path) -> tuple[dict[str, dict[str, object]], Counter[str], dict[str, object]]:
    """Load v12 recipient labels and recompute split-integrity evidence.

    The validator intentionally traverses every slot for crop hashes: a crop
    shared by another field across a split is still evidence of a receipt-level
    split problem.  Recipient-only source-row checks and label conflicts are
    additionally reported because they control any future name-based design.
    """
    source = Path(manifest_path).expanduser().resolve()
    records = _read_jsonl(source)
    slots_by_id: dict[str, dict[str, object]] = {}
    train_name_support: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = {}
    source_splits: dict[str, set[str]] = {}
    result_splits: dict[str, set[str]] = {}
    receipt_key_splits: dict[str, set[str]] = {}
    crop_splits: dict[str, set[str]] = {}
    recipient_source_record_splits: dict[str, set[str]] = {}
    crop_texts: dict[str, set[str]] = {}
    source_record_texts: dict[str, set[str]] = {}
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    summary: Counter[str] = Counter()

    for line_number, record in enumerate(records, start=1):
        row_label = str(line_number)
        receipt_id = _require_string(record.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate manifest id {receipt_id!r}")
        seen_ids.add(receipt_id)
        split = _require_string(record.get("split"), source=source, row_label=row_label, key="split")
        if split not in _SPLITS:
            raise ValueError(f"{source}:{row_label}: split must be train, val, or test")
        group_id = _require_string(record.get("group_id"), source=source, row_label=row_label, key="group_id")
        slots = record.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(f"{source}:{row_label}: slots must be an object")
        summary["manifest_records"] += 1
        split_counts[split] += 1
        _record_key(group_splits, key=group_id, split=split)
        _record_key(
            source_splits,
            key=_optional_string(record.get("source"), source=source, row_label=row_label, key="source"),
            split=split,
        )
        _record_key(
            result_splits,
            key=_optional_string(record.get("result_json"), source=source, row_label=row_label, key="result_json"),
            split=split,
        )
        _record_key(
            receipt_key_splits,
            key=_optional_string(record.get("receipt_key"), source=source, row_label=row_label, key="receipt_key"),
            split=split,
        )

        for field, raw_slot in slots.items():
            if raw_slot is None:
                continue
            if not isinstance(field, str) or not isinstance(raw_slot, Mapping):
                raise ValueError(f"{source}:{row_label}: slots must map names to objects or null")
            crop_sha256 = _optional_string(
                raw_slot.get("crop_sha256"),
                source=source,
                row_label=row_label,
                key=f"{field} crop_sha256",
            )
            _record_key(crop_splits, key=crop_sha256, split=split)

        slot = _recipient_slot(record)
        if slot is None:
            continue
        text = _require_string(slot.get("text"), source=source, row_label=row_label, key="recipient text")
        source_record_id = _optional_string(
            slot.get("source_record_id"),
            source=source,
            row_label=row_label,
            key="recipient source_record_id",
        )
        crop_sha256 = _optional_string(
            slot.get("crop_sha256"),
            source=source,
            row_label=row_label,
            key="recipient crop_sha256",
        )
        slots_by_id[receipt_id] = {"text": text, "split": split}
        summary["manifest_recipient_records"] += 1
        if split == "train":
            train_name_support[text] += 1
            summary["manifest_train_recipient_records"] += 1
        _record_key(recipient_source_record_splits, key=source_record_id, split=split)
        if crop_sha256:
            crop_texts.setdefault(crop_sha256, set()).add(text)
        if source_record_id:
            source_record_texts.setdefault(source_record_id, set()).add(text)

    if not train_name_support:
        raise ValueError(f"{source}: no train recipient labels")
    integrity = {
        "cross_split_collisions": {
            "group_id": _collision_summary(group_splits),
            "source": _collision_summary(source_splits),
            "result_json": _collision_summary(result_splits),
            "receipt_key": _collision_summary(receipt_key_splits),
            "crop_sha256": _collision_summary(crop_splits),
            "recipient_source_record_id": _collision_summary(recipient_source_record_splits),
        },
        "recipient_label_conflicts": {
            "crop_sha256": _label_conflict_summary(crop_texts),
            "source_record_id": _label_conflict_summary(source_record_texts),
        },
    }
    cross_split_count = sum(
        int(item["cross_split_keys"])
        for item in integrity["cross_split_collisions"].values()
        if isinstance(item, Mapping)
    )
    conflict_count = sum(
        int(item["conflicting_keys"])
        for item in integrity["recipient_label_conflicts"].values()
        if isinstance(item, Mapping)
    )
    integrity["clean"] = cross_split_count == 0 and conflict_count == 0
    integrity["cross_split_collision_keys"] = cross_split_count
    integrity["recipient_label_conflict_keys"] = conflict_count
    return slots_by_id, train_name_support, {
        **{key: int(value) for key, value in summary.items()},
        "records_by_split": {key: int(split_counts[key]) for key in sorted(split_counts)},
        "train_distinct_recipient_names": len(train_name_support),
        "integrity": integrity,
    }


def _comparison_exact(
    value: object,
    *,
    expected: bool,
    source: Path,
    row_label: str,
) -> bool:
    if value is None:
        return expected
    if not isinstance(value, bool):
        raise ValueError(f"{source}:{row_label}: raw_exact must be a boolean")
    if value != expected:
        raise ValueError(
            f"{source}:{row_label}: raw_exact disagrees with candidate_text == reference_text; "
            "the evaluator comparison is not trustworthy"
        )
    return value


def _load_comparison_rows(
    comparisons_path: Path,
    *,
    slots_by_id: Mapping[str, Mapping[str, object]],
    train_name_support: Mapping[str, int],
) -> list[dict[str, object]]:
    source = Path(comparisons_path).expanduser().resolve()
    comparisons = _read_jsonl(source)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, comparison in enumerate(comparisons, start=1):
        if comparison.get("field") != _RECIPIENT_FIELD:
            continue
        row_label = str(line_number)
        receipt_id = _require_string(comparison.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate recipient comparison id {receipt_id!r}")
        seen_ids.add(receipt_id)
        slot = slots_by_id.get(receipt_id)
        if slot is None:
            raise ValueError(f"{source}:{row_label}: recipient comparison id {receipt_id!r} is absent from manifest")
        split = str(slot["split"])
        if split == "train":
            raise ValueError(f"{source}:{row_label}: recipient comparison id {receipt_id!r} belongs to train, not held-out data")
        comparison_split = comparison.get("split")
        if comparison_split is not None and comparison_split != split:
            raise ValueError(
                f"{source}:{row_label}: comparison split {comparison_split!r} disagrees with manifest {split!r}"
            )
        reference_text = _require_string(
            comparison.get("reference_text"), source=source, row_label=row_label, key="reference_text"
        )
        manifest_text = str(slot["text"])
        if reference_text != manifest_text:
            raise ValueError(
                f"{source}:{row_label}: reference_text disagrees with manifest recipient text for id {receipt_id!r}; "
                "use the manifest that produced this ONNX evaluation"
            )
        raw_candidate = comparison.get("candidate_text")
        if raw_candidate is None:
            candidate_text = ""
        elif isinstance(raw_candidate, str):
            candidate_text = raw_candidate
        else:
            raise ValueError(f"{source}:{row_label}: candidate_text must be a string or null")
        exact = _comparison_exact(
            comparison.get("raw_exact"),
            expected=candidate_text == reference_text,
            source=source,
            row_label=row_label,
        )
        support = int(train_name_support.get(reference_text, 0))
        rows.append(
            {
                "id": receipt_id,
                "split": split,
                "reference_text": reference_text,
                "candidate_text": candidate_text,
                "raw_exact": exact,
                "reference_train_support": support,
                "reference_seen_train": support > 0,
                "candidate_seen_train": candidate_text in train_name_support,
            }
        )
    if not rows:
        raise ValueError(f"{source}: no recipient_field comparisons")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _support_bucket(support: int) -> str:
    if support <= 0:
        return "0"
    if support == 1:
        return "1"
    if support <= 3:
        return "2-3"
    if support <= 9:
        return "4-9"
    if support <= 24:
        return "10-24"
    return "25+"


def _bucket_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(rows)
    records = len(values)
    exact = sum(bool(row["raw_exact"]) for row in values)
    correctable = sum(not bool(row["raw_exact"]) and bool(row["reference_seen_train"]) for row in values)
    return {
        "records": records,
        "current_exact_matches": exact,
        "current_raw_exact_match": _rate(exact, records),
        "incorrect_records": records - exact,
        "oracle_correctable_incorrect_records": correctable,
        "oracle_closed_set_exact_matches": exact + correctable,
        "oracle_closed_set_exact_match": _rate(exact + correctable, records),
    }


def _quality_audit_summary(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"available": False, "path": None}
    source = Path(path).expanduser().resolve()
    rows = _read_jsonl(source)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_by_reason: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    retained = 0
    for line_number, row in enumerate(rows, start=1):
        row_label = str(line_number)
        split = _require_string(row.get("split"), source=source, row_label=row_label, key="split")
        if split not in _SPLITS:
            raise ValueError(f"{source}:{row_label}: split must be train, val, or test")
        decision = _require_string(
            row.get("quality_decision"), source=source, row_label=row_label, key="quality_decision"
        )
        if decision not in {"accepted", "rejected"}:
            raise ValueError(f"{source}:{row_label}: quality_decision must be accepted or rejected")
        decisions[decision] += 1
        by_split[split]["source_records"] += 1
        by_split[split][f"quality_{decision}"] += 1
        if bool(row.get("retained_in_unified_manifest")):
            retained += 1
            by_split[split]["retained_slot_records"] += 1
        if decision == "rejected":
            reason = _require_string(
                row.get("quality_reason"), source=source, row_label=row_label, key="quality_reason"
            )
            rejected_by_reason[reason] += 1
    return {
        "available": True,
        "path": source.as_posix(),
        "source_records": len(rows),
        "quality_accepted": int(decisions["accepted"]),
        "quality_rejected": int(decisions["rejected"]),
        "retained_slot_records": retained,
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "by_split": {
            split: {
                "source_records": int(values["source_records"]),
                "quality_accepted": int(values["quality_accepted"]),
                "quality_rejected": int(values["quality_rejected"]),
                "retained_slot_records": int(values["retained_slot_records"]),
            }
            for split, values in sorted(by_split.items())
        },
    }


def build_recipient_data_audit(
    *,
    comparisons_path: Path,
    manifest_path: Path,
    target_raw_exact_match: float = 0.90,
    quality_audit_path: Path | None = None,
) -> dict[str, object]:
    """Return a side-effect-free data/coverage audit for a held-out run."""
    try:
        target = float(target_raw_exact_match)
    except (TypeError, ValueError) as error:
        raise ValueError("target_raw_exact_match must be a finite probability in (0, 1]") from error
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("target_raw_exact_match must be a finite probability in (0, 1]")

    slots_by_id, train_name_support, manifest_summary = _load_manifest(manifest_path)
    rows = _load_comparison_rows(
        comparisons_path,
        slots_by_id=slots_by_id,
        train_name_support=train_name_support,
    )
    records = len(rows)
    overall = _bucket_metrics(rows)
    support_thresholds = {
        f">={threshold}": _bucket_metrics(
            row for row in rows if int(row["reference_train_support"]) >= threshold
        )
        for threshold in _SUPPORT_THRESHOLDS
    }
    support_bins: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        support_bins[_support_bucket(int(row["reference_train_support"]))].append(row)
    support_bin_metrics = {
        name: _bucket_metrics(support_bins.get(name, []))
        for name in ("0", "1", "2-3", "4-9", "10-24", "25+")
    }
    exact = int(overall["current_exact_matches"])
    correctable = int(overall["oracle_correctable_incorrect_records"])
    references_seen = sum(bool(row["reference_seen_train"]) for row in rows)
    incorrect_candidate_known = sum(
        not bool(row["raw_exact"]) and bool(row["candidate_seen_train"])
        for row in rows
    )
    split_counts = Counter(str(row["split"]) for row in rows)
    integrity = manifest_summary["integrity"]
    if not isinstance(integrity, Mapping):  # defensive: this is produced above, but never let a malformed report lie.
        raise ValueError("manifest integrity summary is invalid")
    oracle_rate = _rate(exact + correctable, records)
    clean_integrity = bool(integrity["clean"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "comparisons": Path(comparisons_path).expanduser().resolve().as_posix(),
        "manifest": Path(manifest_path).expanduser().resolve().as_posix(),
        "recipient_field": _RECIPIENT_FIELD,
        "target_raw_exact_match": target,
        "manifest_summary": manifest_summary,
        "quality_audit": _quality_audit_summary(quality_audit_path),
        "evaluation": {
            **overall,
            "records_by_split": {key: int(split_counts[key]) for key in sorted(split_counts)},
            "incorrect_candidates_equal_some_train_name": incorrect_candidate_known,
        },
        "train_reference_coverage": {
            "train_distinct_recipient_names": len(train_name_support),
            "held_out_references_seen_in_train": references_seen,
            "held_out_references_seen_in_train_rate": _rate(references_seen, records),
            "thresholds": support_thresholds,
            "support_bins": support_bin_metrics,
        },
        "omniscient_train_only_closed_set_ceiling": {
            "current_exact_matches": exact,
            "current_exact_match": _rate(exact, records),
            "incorrect_references_seen_in_train": correctable,
            "max_exact_matches": exact + correctable,
            "max_raw_exact_match": oracle_rate,
            "target_raw_exact_match": target,
            "target_reachable_under_oracle": bool(clean_integrity and oracle_rate is not None and oracle_rate >= target),
            "integrity_clean": clean_integrity,
            "interpretation": (
                "Oracle only: retain current exact predictions and perfectly select the correct full train-only name "
                "for every remaining held-out reference seen in train. It is an upper bound, not a lexicon result."
            ),
        },
        "decision": {
            "train_only_closed_set_route_eligible_for_target": bool(
                clean_integrity and oracle_rate is not None and oracle_rate >= target
            ),
            "reason": (
                "The held-out manifest has split/label integrity conflicts; repair or explain those before using coverage "
                "to choose a model route."
                if not clean_integrity
                else "The oracle ceiling meets the target; a train-only candidate-selection experiment is justified, but not proven."
                if oracle_rate is not None and oracle_rate >= target
                else "Even a perfect train-only closed-set selector cannot reach the target on this held-out set; use a stronger "
                "open-text recognizer or new supervision instead of another same-recipe tail run."
            ),
        },
        "warning": (
            "This is a read-only data and teacher-parity feasibility audit. The compared labels are not independent human "
            "truth, and the closed-set figure is not a deployable accuracy claim."
        ),
    }


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def format_recipient_data_audit(report: Mapping[str, object]) -> str:
    """Render a compact terminal report for an RDP screenshot or run log."""
    evaluation = report.get("evaluation")
    coverage = report.get("train_reference_coverage")
    ceiling = report.get("omniscient_train_only_closed_set_ceiling")
    manifest = report.get("manifest_summary")
    decision = report.get("decision")
    if not all(isinstance(value, Mapping) for value in (evaluation, coverage, ceiling, manifest, decision)):
        raise ValueError("recipient data audit report is invalid")
    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("recipient data audit integrity is invalid")
    lines = [
        "recipient_data_audit",
        f"  comparisons={report.get('comparisons')}",
        f"  manifest={report.get('manifest')}",
        f"  records={evaluation.get('current_exact_matches')}/{evaluation.get('records')}="
        f"{_format_rate(evaluation.get('current_raw_exact_match'))}; "
        f"held_out_splits={evaluation.get('records_by_split')}",
        f"  train-name coverage={coverage.get('held_out_references_seen_in_train')}/{evaluation.get('records')}="
        f"{_format_rate(coverage.get('held_out_references_seen_in_train_rate'))}; "
        f"train-distinct-names={coverage.get('train_distinct_recipient_names')}",
        f"  oracle closed-set ceiling={ceiling.get('max_exact_matches')}/{evaluation.get('records')}="
        f"{_format_rate(ceiling.get('max_raw_exact_match'))}; "
        f"target={_format_rate(ceiling.get('target_raw_exact_match'))}; "
        f"eligible={ceiling.get('target_reachable_under_oracle')}",
        f"  integrity clean={integrity.get('clean')}; cross-split-keys={integrity.get('cross_split_collision_keys')}; "
        f"label-conflict-keys={integrity.get('recipient_label_conflict_keys')}",
        f"  decision={decision.get('reason')}",
        "  [reference_train_support]",
    ]
    thresholds = coverage.get("thresholds")
    if isinstance(thresholds, Mapping):
        for threshold, metrics in thresholds.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"    {threshold}: {metrics.get('records')}/{evaluation.get('records')} references; "
                f"current={_format_rate(metrics.get('current_raw_exact_match'))}; "
                f"oracle={_format_rate(metrics.get('oracle_closed_set_exact_match'))}"
            )
    lines.append("  [support_bins]")
    bins = coverage.get("support_bins")
    if isinstance(bins, Mapping):
        for name, metrics in bins.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"    {name}: records={metrics.get('records')}; current={_format_rate(metrics.get('current_raw_exact_match'))}; "
                f"potential_gain={metrics.get('oracle_correctable_incorrect_records')}"
            )
    quality = report.get("quality_audit")
    if isinstance(quality, Mapping):
        if bool(quality.get("available")):
            lines.append(
                f"  quality-audit accepted={quality.get('quality_accepted')}; rejected={quality.get('quality_rejected')}; "
                f"retained={quality.get('retained_slot_records')}"
            )
        else:
            lines.append("  quality-audit=unavailable")
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
        description="Read-only recipient train-name coverage, closed-set ceiling, and manifest-integrity audit"
    )
    parser.add_argument("--comparisons", type=Path, required=True, help="existing ONNX evaluator comparisons.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="unified_fields.jsonl used by that evaluator")
    parser.add_argument(
        "--quality-audit",
        type=Path,
        help="optional sibling recipient_quality_audit.jsonl from the same v11/v12 manifest build",
    )
    parser.add_argument("--target", type=float, default=0.90, help="strict-exact target for feasibility, default: 0.90")
    parser.add_argument("--output", type=Path, help="new JSON report path; refuses to overwrite")
    parser.add_argument("--json", action="store_true", help="print complete JSON instead of compact report")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = build_recipient_data_audit(
            comparisons_path=args.comparisons,
            manifest_path=args.manifest,
            target_raw_exact_match=args.target,
            quality_audit_path=args.quality_audit,
        )
        if args.output is not None:
            _atomic_write_json(args.output, report)
    except (OSError, ValueError) as error:
        raise SystemExit(f"recipient data audit failed: {error}") from error
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_recipient_data_audit(report))
        if args.output is not None:
            print(f"Wrote recipient data audit to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":  # pragma: no cover - invoked via the hyphenated checkout wrapper.
    main()
