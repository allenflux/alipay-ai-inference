#!/usr/bin/env python3
"""Audit candidate and strict-exact changes between two scorer directories.

This helper is deliberately read-only.  It validates both
``comparisons.jsonl`` files, rejects input-set or reference-truth drift, and
then reports candidate/``raw_exact`` changes.  Public field aliases keep the
PowerShell command easy to type (for example, ``--field payment``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FIELD_ALIASES = {
    "amount": "amount",
    "time": "time",
    "payment": "payment_method_field",
    "recipient": "recipient_field",
    "status": "transfer_status",
}

_ABSENT_ID = object()


class ScoreDiffError(RuntimeError):
    """Raised when scorer evidence cannot be compared safely."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _validate_comparison(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    location = f"{path}:{line_number}"
    for key in ("source", "field", "reference_text"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ScoreDiffError(
                f"invalid comparisons {location}: {key} must be a non-empty string"
            )

    if "id" in row and (not isinstance(row["id"], str) or not row["id"]):
        raise ScoreDiffError(
            f"invalid comparisons {location}: id must be a non-empty string when present"
        )

    for key in ("candidate_present", "raw_exact"):
        if not isinstance(row.get(key), bool):
            raise ScoreDiffError(
                f"invalid comparisons {location}: {key} must be a boolean"
            )

    candidate = row.get("candidate_text")
    missing_reason = row.get("missing_reason")
    if row["candidate_present"]:
        if not isinstance(candidate, str) or not candidate:
            raise ScoreDiffError(
                f"invalid comparisons {location}: present candidate_text must be a "
                "non-empty string"
            )
        if missing_reason is not None:
            raise ScoreDiffError(
                f"invalid comparisons {location}: missing_reason must be null when "
                "candidate is present"
            )
    elif candidate is not None:
        raise ScoreDiffError(
            f"invalid comparisons {location}: candidate_text must be null when candidate "
            "is missing"
        )
    elif not isinstance(missing_reason, str) or not missing_reason:
        raise ScoreDiffError(
            f"invalid comparisons {location}: missing candidate requires a non-empty "
            "missing_reason"
        )

    expected_exact = row["candidate_present"] and candidate == row["reference_text"]
    if row["raw_exact"] is not expected_exact:
        raise ScoreDiffError(
            f"invalid comparisons {location}: raw_exact disagrees with strict text equality"
        )


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, object]:
    return (row["source"], row["field"], row.get("id", _ABSENT_ID))


def _render_key(key: tuple[str, str, object]) -> dict[str, Any]:
    source, field, row_id = key
    rendered: dict[str, Any] = {"source": source, "field": field}
    if row_id is not _ABSENT_ID:
        rendered["id"] = row_id
    return rendered


def _render(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_comparisons(path: Path) -> dict[tuple[str, str, object], dict[str, Any]]:
    if not path.is_file():
        raise ScoreDiffError(f"missing comparisons: {path}")

    rows: dict[tuple[str, str, object], dict[str, Any]] = {}
    id_modes: set[bool] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ScoreDiffError(
                    f"invalid comparisons {path}:{line_number}: blank JSONL line"
                )
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except (json.JSONDecodeError, ValueError) as exception:
                raise ScoreDiffError(
                    f"invalid comparisons {path}:{line_number}: {exception}"
                ) from exception
            if not isinstance(value, dict):
                raise ScoreDiffError(
                    f"invalid comparisons {path}:{line_number}: expected one JSON object"
                )
            _validate_comparison(value, path=path, line_number=line_number)
            id_modes.add("id" in value)
            key = _row_key(value)
            if key in rows:
                raise ScoreDiffError(
                    f"invalid comparisons {path}:{line_number}: duplicate comparison key "
                    f"{_render(_render_key(key))}"
                )
            rows[key] = value

    if not rows:
        raise ScoreDiffError(f"invalid comparisons {path}: file is empty")
    if len(id_modes) != 1:
        raise ScoreDiffError(
            f"invalid comparisons {path}: id must be present on every row or no rows"
        )
    return rows


def _sorted_keys(keys: set[tuple[str, str, object]]) -> list[tuple[str, str, object]]:
    return sorted(
        keys,
        key=lambda key: (
            key[0],
            key[1],
            "" if key[2] is _ABSENT_ID else str(key[2]),
        ),
    )


def _reject_collection_drift(
    before: Mapping[tuple[str, str, object], dict[str, Any]],
    after: Mapping[tuple[str, str, object], dict[str, Any]],
) -> None:
    before_keys = set(before)
    after_keys = set(after)
    if before_keys != after_keys:
        removed = _sorted_keys(before_keys - after_keys)
        added = _sorted_keys(after_keys - before_keys)
        raise ScoreDiffError(
            "comparison collection changed: "
            f"removed={len(removed)} added={len(added)} "
            f"removed_sample={_render([_render_key(key) for key in removed[:3]])} "
            f"added_sample={_render([_render_key(key) for key in added[:3]])}"
        )

    changed_references = [
        key
        for key in _sorted_keys(before_keys)
        if before[key]["reference_text"] != after[key]["reference_text"]
    ]
    if changed_references:
        key = changed_references[0]
        raise ScoreDiffError(
            "reference truth changed: "
            f"count={len(changed_references)} key={_render(_render_key(key))} "
            f"before={_render(before[key]['reference_text'])} "
            f"after={_render(after[key]['reference_text'])}"
        )


def _state(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_text": row.get("candidate_text"),
        "candidate_present": row["candidate_present"],
        "raw_exact": row["raw_exact"],
    }


def compare(
    *, before_dir: Path, after_dir: Path, field: str | None = None
) -> list[str]:
    before_path = before_dir / "comparisons.jsonl"
    after_path = after_dir / "comparisons.jsonl"
    before = _load_comparisons(before_path)
    after = _load_comparisons(after_path)
    _reject_collection_drift(before, after)

    comparison_field = FIELD_ALIASES[field] if field is not None else None
    keys = _sorted_keys(set(before))
    if comparison_field is not None:
        keys = [key for key in keys if key[1] == comparison_field]
        if not keys:
            raise ScoreDiffError(
                f"comparisons contain no rows for field {field!r} ({comparison_field})"
            )

    changes: list[dict[str, Any]] = []
    correct_to_wrong = 0
    wrong_to_correct = 0
    for key in keys:
        before_state = _state(before[key])
        after_state = _state(after[key])
        if before_state == after_state:
            continue
        before_exact = before_state["raw_exact"]
        after_exact = after_state["raw_exact"]
        if before_exact and not after_exact:
            transition = "correct_to_wrong"
            correct_to_wrong += 1
        elif not before_exact and after_exact:
            transition = "wrong_to_correct"
            wrong_to_correct += 1
        else:
            transition = "accuracy_unchanged"
        changes.append(
            {
                **_render_key(key),
                "reference_text": before[key]["reference_text"],
                "before": before_state,
                "after": after_state,
                "candidate_changed": (
                    before_state["candidate_present"] != after_state["candidate_present"]
                    or before_state["candidate_text"] != after_state["candidate_text"]
                ),
                "raw_exact_changed": before_exact != after_exact,
                "transition": transition,
            }
        )

    lines = [
        "receipt_mlnet_score_diff_v1",
        f"before={before_dir}",
        f"after={after_dir}",
        "field=" + (field if field is not None else "all"),
    ]
    lines.extend(f"change={_render(change)}" for change in changes)
    lines.append(
        "summary="
        + _render(
            {
                "field": field if field is not None else "all",
                "comparison_field": comparison_field,
                "records": len(keys),
                "changed": len(changes),
                "correct_to_wrong": correct_to_wrong,
                "wrong_to_correct": wrong_to_correct,
                "unchanged": len(keys) - len(changes),
            }
        )
    )
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        help="scorer output directory before the candidate change",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        help="scorer output directory after the candidate change",
    )
    parser.add_argument(
        "--field",
        choices=tuple(FIELD_ALIASES),
        help="optional public field alias; use payment for payment_method_field",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for line in compare(before_dir=args.before, after_dir=args.after, field=args.field):
            print(line)
    except (OSError, UnicodeError, ScoreDiffError) as exception:
        print(f"score_diff_error={exception}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
