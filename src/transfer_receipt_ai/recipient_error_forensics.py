"""Read-only edit forensics for held-out recipient OCR comparisons.

The normal unified evaluator intentionally keeps its acceptance output small:
strict exact match, CER, and one row per field/receipt.  Those metrics establish
that a recipe missed its gate, but do not distinguish a rare-character problem
from long-sequence deletions or a small set of systematic substitutions.

This module consumes the evaluator's already-written ``comparisons.jsonl`` and
the exact unified manifest that produced it.  It validates their join, derives
a deterministic Unicode-codepoint Levenshtein alignment, and reports:

* the most frequent substitutions, deletions, and insertions;
* strict/CER/edit counts by reference length and train-character support;
* reference-character error rates by train support; and
* a compact, stratified set of image paths with labels and predictions.

It never imports the model runtime, opens an image, changes a checkpoint, or
rewrites either input.  A requested JSON output is created atomically and an
existing path is always rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


REPORT_KIND = "receipt_recipient_error_forensics_v1"
REPORT_SCHEMA_VERSION = 1
_RECIPIENT_FIELD = "recipient_field"
_HELD_OUT_SPLITS = frozenset(("val", "test"))
_LENGTH_BUCKETS = ("1-4", "5-8", "9-12", "13+")
_SUPPORT_BUCKETS = ("0", "1", "2-3", "4-9", "10+")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
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


def _required_string(value: object, *, source: Path, row_label: str, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{row_label}: {key} must be a non-empty string")
    return value


def _optional_candidate(value: object, *, source: Path, row_label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{source}:{row_label}: candidate_text must be a string or null")
    return value


def _optional_confidence(value: object, *, source: Path, row_label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source}:{row_label}: confidence must be a finite probability")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}:{row_label}: confidence must be a finite probability") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{source}:{row_label}: confidence must be a finite probability")
    return confidence


def _recipient_slot(record: Mapping[str, object]) -> Mapping[str, object] | None:
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        return None
    slot = slots.get(_RECIPIENT_FIELD)
    return slot if isinstance(slot, Mapping) else None


def _load_manifest(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, object]], Counter[str], dict[str, object]]:
    source = Path(manifest_path).expanduser().resolve()
    records = _read_jsonl(source)
    by_id: dict[str, dict[str, object]] = {}
    train_character_support: Counter[str] = Counter()
    train_name_support: Counter[str] = Counter()
    recipient_by_split: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        row_label = str(line_number)
        receipt_id = _required_string(record.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate manifest id {receipt_id!r}")
        seen_ids.add(receipt_id)
        split = _required_string(record.get("split"), source=source, row_label=row_label, key="split")
        if split not in {"train", *_HELD_OUT_SPLITS}:
            raise ValueError(f"{source}:{row_label}: split must be train, val, or test")
        slot = _recipient_slot(record)
        if slot is None:
            continue
        text = _required_string(slot.get("text"), source=source, row_label=row_label, key="recipient text")
        raw_image = slot.get("image")
        if raw_image is not None and (not isinstance(raw_image, str) or not raw_image):
            raise ValueError(f"{source}:{row_label}: recipient image must be a non-empty string when present")
        by_id[receipt_id] = {
            "split": split,
            "text": text,
            "image": raw_image,
        }
        recipient_by_split[split] += 1
        if split == "train":
            train_character_support.update(text)
            train_name_support[text] += 1
    if not train_character_support:
        raise ValueError(f"{source}: no train recipient characters")
    return by_id, train_character_support, {
        "manifest_records": len(records),
        "manifest_recipient_records": len(by_id),
        "recipient_records_by_split": {
            split: int(recipient_by_split[split]) for split in ("train", "val", "test")
        },
        "train_distinct_recipient_characters": len(train_character_support),
        "train_distinct_recipient_names": len(train_name_support),
    }


def align_recipient_text(reference: str, candidate: str) -> list[dict[str, object]]:
    """Return one deterministic minimum-edit Unicode-codepoint alignment.

    Equal diagonals are always retained.  Remaining ties prefer substitution,
    then deletion, then insertion.  That keeps a one-glyph visual confusion as
    a substitution instead of an arbitrary delete+insert pair while remaining
    fully deterministic for repeated Chinese characters.
    """
    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise TypeError("reference and candidate must be strings")
    rows = len(reference) + 1
    columns = len(candidate) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row, reference_character in enumerate(reference, start=1):
        for column, candidate_character in enumerate(candidate, start=1):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + (reference_character != candidate_character),
            )

    aligned: list[dict[str, object]] = []
    row = len(reference)
    column = len(candidate)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == candidate[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            aligned.append(
                {
                    "operation": "equal",
                    "reference_character": reference[row - 1],
                    "candidate_character": candidate[column - 1],
                    "reference_index": row - 1,
                    "candidate_index": column - 1,
                }
            )
            row -= 1
            column -= 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            aligned.append(
                {
                    "operation": "substitution",
                    "reference_character": reference[row - 1],
                    "candidate_character": candidate[column - 1],
                    "reference_index": row - 1,
                    "candidate_index": column - 1,
                }
            )
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            aligned.append(
                {
                    "operation": "deletion",
                    "reference_character": reference[row - 1],
                    "candidate_character": None,
                    "reference_index": row - 1,
                    "candidate_index": column,
                }
            )
            row -= 1
        elif column and costs[row][column] == costs[row][column - 1] + 1:
            aligned.append(
                {
                    "operation": "insertion",
                    "reference_character": None,
                    "candidate_character": candidate[column - 1],
                    "reference_index": row,
                    "candidate_index": column - 1,
                }
            )
            column -= 1
        else:  # pragma: no cover - guarded by the recurrence above.
            raise AssertionError("Levenshtein backtrace lost its predecessor")
    aligned.reverse()
    return aligned


def _length_bucket(length: int) -> str:
    if length <= 4:
        return "1-4"
    if length <= 8:
        return "5-8"
    if length <= 12:
        return "9-12"
    return "13+"


def _support_bucket(support: int) -> str:
    if support <= 0:
        return "0"
    if support == 1:
        return "1"
    if support <= 3:
        return "2-3"
    if support <= 9:
        return "4-9"
    return "10+"


def _resolve_image(
    comparison: Mapping[str, object],
    manifest_slot: Mapping[str, object],
    *,
    dataset_root: Path | None,
) -> str | None:
    raw = comparison.get("image")
    if not isinstance(raw, str) or not raw:
        raw = manifest_slot.get("image")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.as_posix()
    if dataset_root is not None:
        return (dataset_root / path).resolve().as_posix()
    return path.as_posix()


def _load_comparison_rows(
    comparisons_path: Path,
    *,
    manifest_by_id: Mapping[str, Mapping[str, object]],
    train_character_support: Mapping[str, int],
    split: str,
    dataset_root: Path | None,
) -> list[dict[str, object]]:
    source = Path(comparisons_path).expanduser().resolve()
    comparisons = _read_jsonl(source)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, comparison in enumerate(comparisons, start=1):
        if comparison.get("field") != _RECIPIENT_FIELD:
            continue
        row_label = str(line_number)
        receipt_id = _required_string(comparison.get("id"), source=source, row_label=row_label, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{row_label}: duplicate recipient comparison id {receipt_id!r}")
        seen_ids.add(receipt_id)
        manifest_slot = manifest_by_id.get(receipt_id)
        if manifest_slot is None:
            raise ValueError(f"{source}:{row_label}: recipient comparison id {receipt_id!r} is absent from manifest")
        manifest_split = str(manifest_slot["split"])
        if manifest_split != split:
            raise ValueError(
                f"{source}:{row_label}: recipient comparison id {receipt_id!r} belongs to {manifest_split!r}, "
                f"not requested split {split!r}"
            )
        comparison_split = comparison.get("split")
        if comparison_split is not None and comparison_split != manifest_split:
            raise ValueError(
                f"{source}:{row_label}: comparison split {comparison_split!r} disagrees with manifest {manifest_split!r}"
            )
        reference = _required_string(
            comparison.get("reference_text"), source=source, row_label=row_label, key="reference_text"
        )
        if reference != manifest_slot["text"]:
            raise ValueError(
                f"{source}:{row_label}: reference_text disagrees with manifest recipient text for {receipt_id!r}"
            )
        candidate = _optional_candidate(comparison.get("candidate_text"), source=source, row_label=row_label)
        exact = reference == candidate
        declared_exact = comparison.get("raw_exact")
        if declared_exact is not None and (not isinstance(declared_exact, bool) or declared_exact != exact):
            raise ValueError(f"{source}:{row_label}: raw_exact disagrees with reference_text == candidate_text")
        alignment = align_recipient_text(reference, candidate)
        edit_count = sum(item["operation"] != "equal" for item in alignment)
        declared_edits = comparison.get("cer_edits")
        if declared_edits is not None and (
            isinstance(declared_edits, bool) or not isinstance(declared_edits, int) or declared_edits != edit_count
        ):
            raise ValueError(f"{source}:{row_label}: cer_edits disagrees with deterministic edit distance")
        declared_characters = comparison.get("reference_characters")
        if declared_characters is not None and (
            isinstance(declared_characters, bool)
            or not isinstance(declared_characters, int)
            or declared_characters != len(reference)
        ):
            raise ValueError(f"{source}:{row_label}: reference_characters disagrees with reference_text")
        character_support = [int(train_character_support.get(character, 0)) for character in reference]
        substitutions = sum(item["operation"] == "substitution" for item in alignment)
        deletions = sum(item["operation"] == "deletion" for item in alignment)
        insertions = sum(item["operation"] == "insertion" for item in alignment)
        rows.append(
            {
                "id": receipt_id,
                "split": manifest_split,
                "image": _resolve_image(comparison, manifest_slot, dataset_root=dataset_root),
                "reference_text": reference,
                "candidate_text": candidate,
                "confidence": _optional_confidence(
                    comparison.get("confidence"), source=source, row_label=row_label
                ),
                "raw_exact": exact,
                "reference_length": len(reference),
                "reference_length_bucket": _length_bucket(len(reference)),
                "min_train_character_support": min(character_support, default=0),
                "mean_train_character_support": (
                    sum(character_support) / len(character_support) if character_support else 0.0
                ),
                "min_train_character_support_bucket": _support_bucket(min(character_support, default=0)),
                "cer_edits": edit_count,
                "cer": edit_count / max(1, len(reference)),
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "alignment": alignment,
            }
        )
    if not rows:
        raise ValueError(f"{source}: no recipient_field comparisons for split {split!r}")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _record_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    selected = list(rows)
    records = len(selected)
    exact = sum(bool(row["raw_exact"]) for row in selected)
    reference_characters = sum(int(row["reference_length"]) for row in selected)
    edits = sum(int(row["cer_edits"]) for row in selected)
    substitutions = sum(int(row["substitutions"]) for row in selected)
    deletions = sum(int(row["deletions"]) for row in selected)
    insertions = sum(int(row["insertions"]) for row in selected)
    empty_candidates = sum(not str(row["candidate_text"]) for row in selected)
    return {
        "records": records,
        "exact_matches": exact,
        "raw_exact_match": _rate(exact, records),
        "incorrect_records": records - exact,
        "reference_characters": reference_characters,
        "cer_edits": edits,
        "micro_cer": _rate(edits, reference_characters),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "empty_candidate_records": empty_candidates,
        "empty_candidate_rate": _rate(empty_candidates, records),
        "operation_count_consistent": edits == substitutions + deletions + insertions,
    }


def _sample(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "split": row["split"],
        "image": row["image"],
        "reference_text": row["reference_text"],
        "candidate_text": row["candidate_text"],
        "confidence": row["confidence"],
        "reference_length": row["reference_length"],
        "min_train_character_support": row["min_train_character_support"],
        "cer_edits": row["cer_edits"],
        "cer": row["cer"],
        "substitutions": row["substitutions"],
        "deletions": row["deletions"],
        "insertions": row["insertions"],
    }


def _sample_sort_key(row: Mapping[str, object]) -> tuple[float, int, int, str]:
    return (
        -float(row["cer"]),
        -int(row["cer_edits"]),
        -int(row["reference_length"]),
        str(row["id"]),
    )


def _top_edit_operations(
    rows: Sequence[Mapping[str, object]],
    *,
    train_character_support: Mapping[str, int],
    limit: int,
    examples_per_operation: int,
) -> dict[str, list[dict[str, object]]]:
    counters: dict[str, Counter[object]] = {
        "substitutions": Counter(),
        "deletions": Counter(),
        "insertions": Counter(),
    }
    examples: dict[str, dict[object, dict[str, Mapping[str, object]]]] = {
        name: defaultdict(dict) for name in counters
    }
    for row in rows:
        per_row_counts: Counter[tuple[str, object]] = Counter()
        for item in row["alignment"]:  # type: ignore[index]
            operation = str(item["operation"])
            if operation == "substitution":
                group = "substitutions"
                key: object = (str(item["reference_character"]), str(item["candidate_character"]))
            elif operation == "deletion":
                group = "deletions"
                key = str(item["reference_character"])
            elif operation == "insertion":
                group = "insertions"
                key = str(item["candidate_character"])
            else:
                continue
            counters[group][key] += 1
            per_row_counts[(group, key)] += 1
        for (group, key), occurrences in per_row_counts.items():
            sample = {**_sample(row), "operation_instances_in_sample": int(occurrences)}
            examples[group][key][str(row["id"])] = sample

    rendered: dict[str, list[dict[str, object]]] = {}
    for group, counter in counters.items():
        total = sum(counter.values())
        ordered = sorted(counter.items(), key=lambda item: (-item[1], repr(item[0])))[:limit]
        group_rows: list[dict[str, object]] = []
        for key, count in ordered:
            if group == "substitutions":
                reference_character, candidate_character = key
                payload: dict[str, object] = {
                    "reference_character": reference_character,
                    "candidate_character": candidate_character,
                    "reference_train_support": int(train_character_support.get(reference_character, 0)),
                }
            elif group == "deletions":
                payload = {
                    "reference_character": key,
                    "reference_train_support": int(train_character_support.get(str(key), 0)),
                }
            else:
                payload = {"candidate_character": key}
            operation_examples = sorted(examples[group][key].values(), key=_sample_sort_key)
            group_rows.append(
                {
                    **payload,
                    "count": int(count),
                    "share_of_operation_type": _rate(int(count), total),
                    "examples": operation_examples[:examples_per_operation],
                }
            )
        rendered[group] = group_rows
    return rendered


def _record_slices(
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
    buckets: Sequence[str],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {bucket: _record_metrics(grouped.get(bucket, [])) for bucket in buckets}


def _character_support_slices(
    rows: Sequence[Mapping[str, object]],
    *,
    train_character_support: Mapping[str, int],
) -> dict[str, dict[str, object]]:
    counters: dict[str, Counter[str]] = {bucket: Counter() for bucket in _SUPPORT_BUCKETS}
    for row in rows:
        for item in row["alignment"]:  # type: ignore[index]
            reference_character = item["reference_character"]
            if reference_character is None:
                continue
            support = int(train_character_support.get(str(reference_character), 0))
            values = counters[_support_bucket(support)]
            values["reference_characters"] += 1
            operation = str(item["operation"])
            if operation == "equal":
                values["correct_characters"] += 1
            elif operation == "substitution":
                values["substitutions"] += 1
            elif operation == "deletion":
                values["deletions"] += 1
    return {
        bucket: {
            "reference_characters": int(values["reference_characters"]),
            "correct_characters": int(values["correct_characters"]),
            "substitutions": int(values["substitutions"]),
            "deletions": int(values["deletions"]),
            "reference_character_error_rate": _rate(
                int(values["substitutions"] + values["deletions"]),
                int(values["reference_characters"]),
            ),
        }
        for bucket in _SUPPORT_BUCKETS
        for values in (counters[bucket],)
    }


def _representative_samples(rows: Sequence[Mapping[str, object]], *, limit: int) -> list[dict[str, object]]:
    misses = [row for row in rows if not bool(row["raw_exact"])]
    if not misses or limit <= 0:
        return []
    selected: list[Mapping[str, object]] = []
    selected_ids: set[str] = set()

    def add_best(candidates: Iterable[Mapping[str, object]]) -> None:
        ordered = sorted(candidates, key=_sample_sort_key)
        if not ordered:
            return
        best = ordered[0]
        receipt_id = str(best["id"])
        if receipt_id not in selected_ids and len(selected) < limit:
            selected.append(best)
            selected_ids.add(receipt_id)

    # Preserve at least one hard miss from every populated length and support
    # slice before filling by global severity.
    for bucket in _LENGTH_BUCKETS:
        add_best(row for row in misses if row["reference_length_bucket"] == bucket)
    for bucket in _SUPPORT_BUCKETS:
        add_best(row for row in misses if row["min_train_character_support_bucket"] == bucket)
    for row in sorted(misses, key=_sample_sort_key):
        if len(selected) >= limit:
            break
        if str(row["id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["id"]))
    return [_sample(row) for row in selected]


def _recipe_signals(
    *,
    overall: Mapping[str, object],
    length_slices: Mapping[str, Mapping[str, object]],
    character_support_slices: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    short_records = sum(int(length_slices[bucket]["records"]) for bucket in ("1-4", "5-8"))
    short_exact = sum(int(length_slices[bucket]["exact_matches"]) for bucket in ("1-4", "5-8"))
    long_records = sum(int(length_slices[bucket]["records"]) for bucket in ("9-12", "13+"))
    long_exact = sum(int(length_slices[bucket]["exact_matches"]) for bucket in ("9-12", "13+"))
    short_rate = _rate(short_exact, short_records)
    long_rate = _rate(long_exact, long_records)
    rare_characters = sum(
        int(character_support_slices[bucket]["reference_characters"]) for bucket in ("0", "1", "2-3")
    )
    rare_errors = sum(
        int(character_support_slices[bucket]["substitutions"])
        + int(character_support_slices[bucket]["deletions"])
        for bucket in ("0", "1", "2-3")
    )
    common_characters = int(character_support_slices["10+"]["reference_characters"])
    common_errors = int(character_support_slices["10+"]["substitutions"]) + int(
        character_support_slices["10+"]["deletions"]
    )
    rare_error_rate = _rate(rare_errors, rare_characters)
    common_error_rate = _rate(common_errors, common_characters)
    substitutions = int(overall["substitutions"])
    deletions = int(overall["deletions"])
    insertions = int(overall["insertions"])
    dominant_operation = max(
        (("substitution", substitutions), ("deletion", deletions), ("insertion", insertions)),
        key=lambda item: (item[1], item[0]),
    )[0]
    priorities: list[str] = []
    if (
        short_rate is not None
        and long_rate is not None
        and short_records >= 20
        and long_records >= 20
        and short_rate - long_rate >= 0.15
    ):
        priorities.append("length-aware recipient curriculum/loss A/B before a same-recipe epoch extension")
    if (
        rare_error_rate is not None
        and common_error_rate is not None
        and rare_characters >= 20
        and common_characters >= 20
        and rare_error_rate - common_error_rate >= 0.05
    ):
        priorities.append("rare-character sampling/loss A/B with the frozen held-out split")
    if deletions > substitutions and deletions > insertions:
        priorities.append("inspect long-sequence CTC deletion pressure and visual time resolution")
    elif substitutions >= deletions and substitutions >= insertions:
        priorities.append("inspect the top visual glyph confusions and their train support")
    if not priorities:
        priorities.append("use the ranked edit examples to choose one bounded recipient-only A/B")
    return {
        "short_length_1_8_records": short_records,
        "short_length_1_8_exact_match": short_rate,
        "long_length_9_plus_records": long_records,
        "long_length_9_plus_exact_match": long_rate,
        "long_minus_short_exact_match": (
            long_rate - short_rate if short_rate is not None and long_rate is not None else None
        ),
        "rare_support_0_3_reference_characters": rare_characters,
        "rare_support_0_3_character_error_rate": rare_error_rate,
        "common_support_10_plus_reference_characters": common_characters,
        "common_support_10_plus_character_error_rate": common_error_rate,
        "dominant_edit_operation": dominant_operation,
        "bounded_next_experiment_priorities": priorities,
        "interpretation_limit": (
            "These are correlation signals from a frozen held-out teacher-parity comparison, not causal proof. "
            "Test one recipient-only change at a time; do not tune on the held-out examples."
        ),
    }


def build_recipient_error_forensics(
    *,
    comparisons_path: Path,
    manifest_path: Path,
    split: str = "test",
    dataset_root: Path | None = None,
    top: int = 15,
    examples_per_operation: int = 2,
    representative_limit: int = 12,
) -> dict[str, object]:
    """Build a validated, side-effect-free recipient edit-forensics report."""
    if split not in _HELD_OUT_SPLITS:
        raise ValueError("split must be val or test")
    for name, value, minimum in (
        ("top", top, 1),
        ("examples_per_operation", examples_per_operation, 0),
        ("representative_limit", representative_limit, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    manifest_by_id, train_support, manifest_summary = _load_manifest(manifest_path)
    rows = _load_comparison_rows(
        comparisons_path,
        manifest_by_id=manifest_by_id,
        train_character_support=train_support,
        split=split,
        dataset_root=root,
    )
    expected_records = int(dict(manifest_summary["recipient_records_by_split"])[split])
    if len(rows) != expected_records:
        raise ValueError(
            f"recipient comparison denominator {len(rows)} disagrees with manifest {split} recipient records "
            f"{expected_records}; refusing partial forensics"
        )
    overall = _record_metrics(rows)
    length_slices = _record_slices(
        rows,
        key="reference_length_bucket",
        buckets=_LENGTH_BUCKETS,
    )
    support_slices = _record_slices(
        rows,
        key="min_train_character_support_bucket",
        buckets=_SUPPORT_BUCKETS,
    )
    character_support_slices = _character_support_slices(rows, train_character_support=train_support)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "comparisons": Path(comparisons_path).expanduser().resolve().as_posix(),
        "manifest": Path(manifest_path).expanduser().resolve().as_posix(),
        "dataset_root": root.as_posix() if root is not None else None,
        "field": _RECIPIENT_FIELD,
        "split": split,
        "alignment_policy": {
            "unit": "Unicode codepoint",
            "distance": "Levenshtein",
            "tie_break": "equal, substitution, deletion, insertion",
        },
        "manifest_summary": manifest_summary,
        "overall": overall,
        "record_slices": {
            "reference_length": length_slices,
            "minimum_train_character_support": support_slices,
        },
        "reference_character_support_slices": character_support_slices,
        "top_edit_operations": _top_edit_operations(
            rows,
            train_character_support=train_support,
            limit=top,
            examples_per_operation=examples_per_operation,
        ),
        "representative_misses": _representative_samples(rows, limit=representative_limit),
        "recipe_signals": _recipe_signals(
            overall=overall,
            length_slices=length_slices,
            character_support_slices=character_support_slices,
        ),
        "warning": (
            "Read-only Paddle-teacher parity forensics, not independent human-truth accuracy. "
            "The held-out split must remain evaluation-only."
        ),
    }


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _quoted(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def format_recipient_error_forensics(report: Mapping[str, object]) -> str:
    overall = report.get("overall")
    record_slices = report.get("record_slices")
    character_slices = report.get("reference_character_support_slices")
    operations = report.get("top_edit_operations")
    examples = report.get("representative_misses")
    signals = report.get("recipe_signals")
    if not all(
        isinstance(value, Mapping)
        for value in (overall, record_slices, character_slices, operations, signals)
    ) or not isinstance(examples, Sequence):
        raise ValueError("recipient error forensics report is invalid")
    lines = [
        "recipient_error_forensics",
        f"  comparisons={report.get('comparisons')}",
        f"  manifest={report.get('manifest')}",
        f"  split={report.get('split')} exact={overall.get('exact_matches')}/{overall.get('records')}="
        f"{_format_rate(overall.get('raw_exact_match'))}; micro_cer={float(overall.get('micro_cer') or 0.0):.4f}",
        f"  edits={overall.get('cer_edits')}; substitutions={overall.get('substitutions')}; "
        f"deletions={overall.get('deletions')}; insertions={overall.get('insertions')}; "
        f"empty={overall.get('empty_candidate_records')}/{overall.get('records')}="
        f"{_format_rate(overall.get('empty_candidate_rate'))}",
    ]
    for group, arrow in (("substitutions", "->"), ("deletions", "-> empty"), ("insertions", "empty ->")):
        values = operations.get(group)
        lines.append(f"  [top_{group}]")
        if not isinstance(values, Sequence) or not values:
            lines.append("    none")
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            if group == "substitutions":
                label = f"{_quoted(item.get('reference_character'))} {arrow} {_quoted(item.get('candidate_character'))}"
            elif group == "deletions":
                label = f"{_quoted(item.get('reference_character'))} {arrow}"
            else:
                label = f"{arrow} {_quoted(item.get('candidate_character'))}"
            support = item.get("reference_train_support")
            support_text = "" if support is None else f"; train_support={support}"
            lines.append(f"    {label}: {item.get('count')}{support_text}")

    for slice_name, bucket_order in (
        ("reference_length", _LENGTH_BUCKETS),
        ("minimum_train_character_support", _SUPPORT_BUCKETS),
    ):
        values = record_slices.get(slice_name)
        if not isinstance(values, Mapping):
            continue
        lines.append(f"  [{slice_name}]")
        for bucket in bucket_order:
            metrics = values.get(bucket)
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"    {bucket}: {metrics.get('exact_matches')}/{metrics.get('records')}="
                f"{_format_rate(metrics.get('raw_exact_match'))}; "
                f"micro_cer={float(metrics.get('micro_cer') or 0.0):.4f}; "
                f"S/D/I={metrics.get('substitutions')}/{metrics.get('deletions')}/{metrics.get('insertions')}"
            )
    lines.append("  [reference_character_train_support]")
    for bucket in _SUPPORT_BUCKETS:
        metrics = character_slices.get(bucket)
        if not isinstance(metrics, Mapping):
            continue
        lines.append(
            f"    {bucket}: chars={metrics.get('reference_characters')}; "
            f"error={_format_rate(metrics.get('reference_character_error_rate'))}; "
            f"sub/del={metrics.get('substitutions')}/{metrics.get('deletions')}"
        )
    priorities = signals.get("bounded_next_experiment_priorities")
    if isinstance(priorities, Sequence):
        lines.append("  [bounded_next_experiment_priorities]")
        lines.extend(f"    - {priority}" for priority in priorities)
    lines.append("  [representative_misses]")
    if not examples:
        lines.append("    none")
    for item in examples:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"    {item.get('id')}: {_quoted(item.get('reference_text'))} -> "
            f"{_quoted(item.get('candidate_text'))}; edits={item.get('cer_edits')}; "
            f"len={item.get('reference_length')}; min_support={item.get('min_train_character_support')}"
        )
        lines.append(f"      image={item.get('image')}")
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only recipient substitutions/deletions/insertions from existing held-out comparisons"
    )
    parser.add_argument("--comparisons", type=Path, required=True, help="existing ONNX comparisons.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="exact unified_fields.jsonl used in evaluation")
    parser.add_argument("--split", choices=sorted(_HELD_OUT_SPLITS), default="test")
    parser.add_argument("--dataset-root", type=Path, help="optional root for relative fallback image paths")
    parser.add_argument("--top", type=int, default=15, help="maximum rows per edit-operation ranking")
    parser.add_argument("--examples-per-operation", type=int, default=2)
    parser.add_argument("--representative-limit", type=int, default=12)
    parser.add_argument("--output", type=Path, help="new JSON path; existing files are rejected")
    parser.add_argument("--json", action="store_true", help="print full JSON instead of compact terminal text")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = build_recipient_error_forensics(
            comparisons_path=args.comparisons,
            manifest_path=args.manifest,
            split=args.split,
            dataset_root=args.dataset_root,
            top=args.top,
            examples_per_operation=args.examples_per_operation,
            representative_limit=args.representative_limit,
        )
        if args.output is not None:
            _atomic_write_json(args.output, report)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"recipient error forensics failed: {error}") from error
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_recipient_error_forensics(report))
        if args.output is not None:
            print(f"Wrote recipient error forensics to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    main()
