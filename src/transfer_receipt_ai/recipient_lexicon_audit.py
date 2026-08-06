"""Read-only recipient lexicon correction audit.

This module evaluates whether a local Paddle-teacher recipient catalogue can
improve an existing student comparison file without running a model or
modifying a checkpoint.  It is intentionally an audit: the output reports both
helped and hurt rewrites so an unsafe recipe cannot hide behind a better top
line.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transfer_receipt_ai.recipient_lexicon import (
    RecipientLexiconCandidate,
    build_recipient_lexicon,
)


REPORT_KIND = "receipt_recipient_lexicon_audit_v1"
REPORT_SCHEMA_VERSION = 1
RECIPIENT_FIELD = "recipient_field"


@dataclass(frozen=True)
class _LexiconName:
    text: str
    key: str
    occurrences: int
    characters: frozenset[str]


@dataclass(frozen=True)
class _Policy:
    name: str
    max_edit_distance: int
    min_similarity: float
    min_support: int


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


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _key(value: str) -> str:
    return unicodedata.normalize("NFKC", value.strip()).casefold()


def _bounded_levenshtein(left: str, right: str, *, maximum: int) -> int | None:
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if abs(len(left) - len(right)) > maximum:
        return None
    if left == right:
        return 0
    if not left or not right:
        distance = max(len(left), len(right))
        return distance if distance <= maximum else None
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_character in enumerate(right, start=1):
            cost = 0 if left_character == right_character else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= maximum else None


def _similarity(left: str, right: str, distance: int) -> float:
    return 1.0 - (distance / max(len(left), len(right), 1))


def _recipient_slot(record: Mapping[str, object]) -> Mapping[str, object] | None:
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        return None
    slot = slots.get(RECIPIENT_FIELD)
    return slot if isinstance(slot, Mapping) else None


def _load_lexicon_names(manifest: Path, *, lexicon_splits: frozenset[str]) -> tuple[list[_LexiconName], Counter[str]]:
    counts: Counter[str] = Counter()
    for row in _read_jsonl(manifest):
        split = _text(row.get("split"))
        if split not in lexicon_splits:
            continue
        slot = _recipient_slot(row)
        if slot is None:
            continue
        text = _text(slot.get("text"))
        if text:
            counts[text] += 1
    if not counts:
        raise ValueError(f"{manifest}: no recipient labels for lexicon splits {sorted(lexicon_splits)}")
    names = [
        _LexiconName(
            text=text,
            key=_key(text),
            occurrences=occurrences,
            characters=frozenset(_key(text)),
        )
        for text, occurrences in counts.items()
        if _key(text)
    ]
    names.sort(key=lambda item: (-item.occurrences, item.text))
    return names, counts


def _load_recipient_comparisons(comparisons: Path) -> list[dict[str, object]]:
    rows = [
        row for row in _read_jsonl(comparisons)
        if row.get("field") == RECIPIENT_FIELD
    ]
    if not rows:
        raise ValueError(f"{comparisons}: no recipient_field comparisons")
    return rows


def _candidate_pool(
    candidate_key: str,
    *,
    policy: _Policy,
    by_length: Mapping[int, Sequence[_LexiconName]],
) -> Iterable[_LexiconName]:
    candidate_characters = frozenset(candidate_key)
    for length in range(max(0, len(candidate_key) - policy.max_edit_distance), len(candidate_key) + policy.max_edit_distance + 1):
        for name in by_length.get(length, ()):
            if name.occurrences < policy.min_support:
                continue
            # A cheap lower bound: if two strings have too few shared distinct
            # characters, they cannot be within a small edit distance.  This
            # keeps the audit fast enough for tens of thousands of names.
            if len(candidate_characters & name.characters) < min(len(candidate_key), len(name.key)) - policy.max_edit_distance:
                continue
            yield name


def _nearest_unique_match(
    candidate_text: str,
    *,
    policy: _Policy,
    by_length: Mapping[int, Sequence[_LexiconName]],
) -> tuple[_LexiconName, int, float] | None:
    candidate_key = _key(candidate_text)
    if not candidate_key:
        return None
    best: list[tuple[_LexiconName, int, float]] = []
    for name in _candidate_pool(candidate_key, policy=policy, by_length=by_length):
        distance = _bounded_levenshtein(candidate_key, name.key, maximum=policy.max_edit_distance)
        if distance is None:
            continue
        similarity = _similarity(candidate_key, name.key, distance)
        if similarity < policy.min_similarity:
            continue
        if not best or distance < best[0][1] or (distance == best[0][1] and name.occurrences > best[0][0].occurrences):
            best = [(name, distance, similarity)]
        elif distance == best[0][1] and name.occurrences == best[0][0].occurrences:
            best.append((name, distance, similarity))
    unique_texts = {item[0].text for item in best}
    if len(unique_texts) != 1:
        return None
    return best[0]


def _empty_metrics(name: str, total: int, current_exact: int) -> dict[str, object]:
    return {
        "policy": name,
        "records": total,
        "current_exact": current_exact,
        "current_exact_rate": current_exact / total,
        "corrected_exact": current_exact,
        "corrected_exact_rate": current_exact / total,
        "rewrites": 0,
        "helped": 0,
        "hurt": 0,
        "unchanged_wrong": total - current_exact,
        "net_exact_gain": 0,
        "examples": [],
    }


def _evaluate_policy(
    rows: Sequence[Mapping[str, object]],
    *,
    policy: _Policy,
    by_length: Mapping[int, Sequence[_LexiconName]],
) -> dict[str, object]:
    total = len(rows)
    current_exact = sum(1 for row in rows if _text(row.get("candidate_text")) == _text(row.get("reference_text")))
    metrics = _empty_metrics(policy.name, total, current_exact)
    examples: list[dict[str, object]] = []
    corrected_exact = 0
    rewrites = helped = hurt = 0
    for row in rows:
        reference = _text(row.get("reference_text"))
        candidate = _text(row.get("candidate_text"))
        before_exact = candidate == reference
        resolved = candidate
        match = _nearest_unique_match(candidate, policy=policy, by_length=by_length)
        if match is not None:
            name, distance, similarity = match
            resolved = name.text
            if resolved != candidate:
                rewrites += 1
                after_exact = resolved == reference
                if after_exact and not before_exact:
                    helped += 1
                elif before_exact and not after_exact:
                    hurt += 1
                elif not before_exact and not after_exact and resolved != reference:
                    hurt += 1
                if len(examples) < 10 and after_exact != before_exact:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "candidate": candidate,
                            "resolved": resolved,
                            "reference": reference,
                            "before_exact": before_exact,
                            "after_exact": after_exact,
                            "edit_distance": distance,
                            "similarity": round(similarity, 6),
                            "support": name.occurrences,
                        }
                    )
        if resolved == reference:
            corrected_exact += 1
    metrics.update(
        {
            "corrected_exact": corrected_exact,
            "corrected_exact_rate": corrected_exact / total,
            "rewrites": rewrites,
            "helped": helped,
            "hurt": hurt,
            "unchanged_wrong": total - corrected_exact,
            "net_exact_gain": corrected_exact - current_exact,
            "examples": examples,
            "parameters": {
                "max_edit_distance": policy.max_edit_distance,
                "min_similarity": policy.min_similarity,
                "min_support": policy.min_support,
            },
        }
    )
    return metrics


def _evaluate_safe_one_edit(rows: Sequence[Mapping[str, object]], names: Sequence[_LexiconName]) -> dict[str, object]:
    lexicon = build_recipient_lexicon(name.text for name in names)
    total = len(rows)
    current_exact = sum(1 for row in rows if _text(row.get("candidate_text")) == _text(row.get("reference_text")))
    metrics = _empty_metrics("safe_one_edit_high_confidence", total, current_exact)
    corrected_exact = 0
    rewrites = helped = hurt = 0
    examples: list[dict[str, object]] = []
    support = {name.text: name.occurrences for name in names}
    for row in rows:
        reference = _text(row.get("reference_text"))
        candidate = _text(row.get("candidate_text"))
        confidence_raw = row.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool) and math.isfinite(float(confidence_raw)) else None
        before_exact = candidate == reference
        resolved = candidate
        match = lexicon.rerank((RecipientLexiconCandidate(candidate, confidence=confidence),))
        if match is not None:
            resolved = match.resolved_text
            if resolved != candidate:
                rewrites += 1
                after_exact = resolved == reference
                if after_exact and not before_exact:
                    helped += 1
                elif before_exact and not after_exact:
                    hurt += 1
                elif not before_exact and not after_exact:
                    hurt += 1
                if len(examples) < 10 and after_exact != before_exact:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "candidate": candidate,
                            "resolved": resolved,
                            "reference": reference,
                            "before_exact": before_exact,
                            "after_exact": after_exact,
                            "edit_distance": match.edit_distance,
                            "similarity": round(match.similarity, 6),
                            "support": support.get(resolved, 0),
                            "confidence": confidence,
                        }
                    )
        if resolved == reference:
            corrected_exact += 1
    metrics.update(
        {
            "corrected_exact": corrected_exact,
            "corrected_exact_rate": corrected_exact / total,
            "rewrites": rewrites,
            "helped": helped,
            "hurt": hurt,
            "unchanged_wrong": total - corrected_exact,
            "net_exact_gain": corrected_exact - current_exact,
            "examples": examples,
        }
    )
    return metrics


def build_recipient_lexicon_audit(
    *,
    comparisons: Path,
    manifest: Path,
    target: float = 0.70,
    lexicon_splits: frozenset[str] = frozenset(("train",)),
) -> dict[str, object]:
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    rows = _load_recipient_comparisons(comparisons)
    names, counts = _load_lexicon_names(manifest, lexicon_splits=lexicon_splits)
    by_length: dict[int, list[_LexiconName]] = defaultdict(list)
    for name in names:
        by_length[len(name.key)].append(name)
    policies = [
        _Policy(name=f"edit{distance}_sim{similarity:.2f}_support{support}", max_edit_distance=distance, min_similarity=similarity, min_support=support)
        for distance in (1, 2)
        for similarity in (0.80, 0.85, 0.90)
        for support in (1, 2, 3, 5, 10, 25)
    ]
    policy_reports = [_evaluate_policy(rows, policy=policy, by_length=by_length) for policy in policies]
    safe_report = _evaluate_safe_one_edit(rows, names)
    current_exact = int(safe_report["current_exact"])
    train_coverage = sum(1 for row in rows if _text(row.get("reference_text")) in counts)
    oracle_exact = current_exact + sum(
        1
        for row in rows
        if _text(row.get("candidate_text")) != _text(row.get("reference_text"))
        and _text(row.get("reference_text")) in counts
    )
    best = max(policy_reports + [safe_report], key=lambda item: (float(item["corrected_exact_rate"]), -int(item["hurt"])))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "comparisons": str(Path(comparisons).expanduser().resolve()),
        "manifest": str(Path(manifest).expanduser().resolve()),
        "target": target,
        "target_reached": float(best["corrected_exact_rate"]) >= target,
        "lexicon_splits": sorted(lexicon_splits),
        "lexicon_names": len(names),
        "lexicon_records": sum(counts.values()),
        "records": len(rows),
        "current_exact": current_exact,
        "current_exact_rate": current_exact / len(rows),
        "train_reference_coverage": train_coverage,
        "train_reference_coverage_rate": train_coverage / len(rows),
        "oracle_train_closed_set_exact": oracle_exact,
        "oracle_train_closed_set_exact_rate": oracle_exact / len(rows),
        "safe_one_edit": safe_report,
        "best_policy": best,
        "policy_reports": sorted(policy_reports, key=lambda item: (-float(item["corrected_exact_rate"]), int(item["hurt"]), item["policy"]))[:12],
    }


def format_recipient_lexicon_audit(report: Mapping[str, object]) -> str:
    def pct(value: object) -> str:
        return f"{float(value):.2%}"

    lines = [
        "recipient lexicon audit",
        f"  target={pct(report['target'])} reached={report['target_reached']}",
        f"  current={report['current_exact']}/{report['records']}={pct(report['current_exact_rate'])}",
        f"  train_reference_coverage={report['train_reference_coverage']}/{report['records']}={pct(report['train_reference_coverage_rate'])}",
        f"  oracle_train_closed_set={report['oracle_train_closed_set_exact']}/{report['records']}={pct(report['oracle_train_closed_set_exact_rate'])}",
    ]
    for label in ("safe_one_edit", "best_policy"):
        item = report[label]
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "  {0}: {1}/{2}={3}; rewrites={4}, helped={5}, hurt={6}, net={7}; policy={8}".format(
                label,
                item["corrected_exact"],
                item["records"],
                pct(item["corrected_exact_rate"]),
                item["rewrites"],
                item["helped"],
                item["hurt"],
                item["net_exact_gain"],
                item["policy"],
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit Paddle-teacher recipient lexicon correction on comparisons.jsonl")
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.70)
    parser.add_argument("--lexicon-splits", nargs="+", default=["train"], choices=("train", "val", "test"))
    args = parser.parse_args(argv)

    report = build_recipient_lexicon_audit(
        comparisons=args.comparisons,
        manifest=args.manifest,
        target=args.target,
        lexicon_splits=frozenset(args.lexicon_splits),
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recipient lexicon audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_recipient_lexicon_audit(report))
    print(f"Wrote recipient lexicon audit to {output}")


if __name__ == "__main__":
    main()
