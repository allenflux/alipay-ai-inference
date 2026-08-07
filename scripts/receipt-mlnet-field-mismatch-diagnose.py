#!/usr/bin/env python3
"""Print strict, read-only field mismatches from a scorer directory.

The helper reads only ``comparisons.jsonl``.  It does not rewrite scorer
evidence, change acceptance floors, or publish delivery artifacts.  Public
field names intentionally avoid internal underscore-heavy names so the command
is straightforward to type in Windows PowerShell.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


FIELD_ALIASES = {
    "amount": "amount",
    "time": "time",
    "payment": "payment_method_field",
    "recipient": "recipient_field",
    "status": "transfer_status",
}


class DiagnosisError(RuntimeError):
    """Raised when scorer evidence is missing, malformed, or inconsistent."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_comparisons(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DiagnosisError(f"missing comparisons: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise DiagnosisError(f"invalid comparisons {path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except (json.JSONDecodeError, ValueError) as exception:
                raise DiagnosisError(
                    f"invalid comparisons {path}:{line_number}: {exception}"
                ) from exception
            if not isinstance(value, dict):
                raise DiagnosisError(
                    f"invalid comparisons {path}:{line_number}: expected one JSON object"
                )
            _validate_comparison(value, path=path, line_number=line_number)
            rows.append(value)

    if not rows:
        raise DiagnosisError(f"invalid comparisons {path}: file is empty")
    return rows


def _validate_comparison(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    location = f"{path}:{line_number}"
    for key in ("field", "source", "reference_text"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise DiagnosisError(f"invalid comparisons {location}: {key} must be a non-empty string")
    for key in ("candidate_present", "raw_exact"):
        if not isinstance(row.get(key), bool):
            raise DiagnosisError(f"invalid comparisons {location}: {key} must be a boolean")

    candidate_present = row["candidate_present"]
    candidate = row.get("candidate_text")
    missing_reason = row.get("missing_reason")
    if candidate_present:
        if not isinstance(candidate, str) or not candidate:
            raise DiagnosisError(
                f"invalid comparisons {location}: present candidate_text must be a non-empty string"
            )
        if missing_reason is not None:
            raise DiagnosisError(
                f"invalid comparisons {location}: missing_reason must be null when candidate is present"
            )
    else:
        if candidate is not None:
            raise DiagnosisError(
                f"invalid comparisons {location}: candidate_text must be null when candidate is missing"
            )
        if not isinstance(missing_reason, str) or not missing_reason:
            raise DiagnosisError(
                f"invalid comparisons {location}: missing candidate requires a non-empty missing_reason"
            )

    expected_exact = candidate_present and candidate == row["reference_text"]
    if row["raw_exact"] is not expected_exact:
        raise DiagnosisError(
            f"invalid comparisons {location}: raw_exact disagrees with strict text equality"
        )


def _render(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mismatch_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_exact": False,
        "source": row["source"],
        "reference_text": row["reference_text"],
        "candidate_text": row.get("candidate_text"),
        "candidate_present": row["candidate_present"],
        "missing_reason": row.get("missing_reason"),
        "detector_diagnostics": {
            "detection_score": row.get("detection_score"),
            "detection_bbox_image": row.get("detection_bbox_image"),
            "reference_detector_score": row.get("reference_detector_score"),
            "reference_bbox_rectified": row.get("reference_bbox_rectified"),
            "reference_crop_sha256": row.get("reference_crop_sha256"),
        },
        "field_diagnostics": {
            "ctc_candidate_text": row.get("ctc_candidate_text"),
            "structured_candidate_text": row.get("structured_candidate_text"),
            "result_geometry": row.get("result_geometry"),
            "manifest_status": row.get("manifest_status"),
            "result_json": row.get("result_json"),
            "teacher_result_json": row.get("teacher_result_json"),
        },
    }


def diagnose(*, score_dir: Path, field: str) -> list[str]:
    comparison_field = FIELD_ALIASES[field]
    comparisons_path = score_dir / "comparisons.jsonl"
    rows = _load_comparisons(comparisons_path)
    selected = [row for row in rows if row["field"] == comparison_field]
    if not selected:
        raise DiagnosisError(
            f"comparisons contain no rows for field {field!r} ({comparison_field})"
        )

    mismatches = [row for row in selected if row["raw_exact"] is False]
    missing = sum(not row["candidate_present"] for row in mismatches)
    lines = [
        "receipt_mlnet_field_mismatch_diagnose_v1",
        f"score_dir={score_dir}",
        f"field={field} comparison_field={comparison_field}",
    ]
    lines.extend(f"mismatch={_render(_mismatch_payload(row))}" for row in mismatches)
    lines.append(
        "summary="
        + _render(
            {
                "field": field,
                "comparison_field": comparison_field,
                "records": len(selected),
                "raw_exact_matches": len(selected) - len(mismatches),
                "raw_exact_mismatches": len(mismatches),
                "candidate_missing_mismatches": missing,
            }
        )
    )
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-dir",
        required=True,
        type=Path,
        help="scorer output directory containing comparisons.jsonl",
    )
    parser.add_argument(
        "--field",
        required=True,
        choices=tuple(FIELD_ALIASES),
        help="public field name; use payment for payment_method_field",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for line in diagnose(score_dir=args.score_dir, field=args.field):
            print(line)
    except (DiagnosisError, OSError) as exception:
        print(f"diagnosis_error={exception}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
