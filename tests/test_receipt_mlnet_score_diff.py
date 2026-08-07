from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "receipt-mlnet-score-diff.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_score_diff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    row_id: str,
    *,
    source: str | None = None,
    field: str = "payment_method_field",
    reference: str = "储蓄卡（8885）",
    candidate: str | None = "储蓄卡（8885)",
) -> dict[str, object]:
    return {
        "id": row_id,
        "source": source or rf"D:\receipts\{row_id}.jpg",
        "field": field,
        "reference_text": reference,
        "candidate_text": candidate,
        "candidate_present": candidate is not None,
        "missing_reason": None if candidate is not None else "candidate_missing",
        "raw_exact": candidate is not None and candidate == reference,
    }


def _write_score(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True)
    (directory / "comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary(lines: list[str]) -> dict[str, object]:
    return json.loads(
        next(line.removeprefix("summary=") for line in lines if line.startswith("summary="))
    )


def _changes(lines: list[str]) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("change="))
        for line in lines
        if line.startswith("change=")
    ]


def test_payment_diff_binds_reordered_rows_and_classifies_transitions(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before_rows = [
        _row("fixed"),
        _row("stable", reference="余额", candidate="余额"),
        _row("different-wrong", reference="余额", candidate="银行卡"),
        _row("regression", reference="余额", candidate="余额"),
        _row("amount", field="amount", reference="1.00", candidate="1.00"),
    ]
    after_rows = [
        _row("amount", field="amount", reference="1.00", candidate="2.00"),
        _row("regression", reference="余额", candidate="银行卡"),
        _row("different-wrong", reference="余额", candidate="储蓄卡"),
        _row("stable", reference="余额", candidate="余额"),
        _row("fixed", candidate="储蓄卡（8885）"),
    ]
    _write_score(before, before_rows)
    _write_score(after, after_rows)
    before_bytes = (before / "comparisons.jsonl").read_bytes()
    after_bytes = (after / "comparisons.jsonl").read_bytes()

    lines = MODULE.compare(before_dir=before, after_dir=after, field="payment")

    changes = _changes(lines)
    assert [change["id"] for change in changes] == [
        "different-wrong",
        "fixed",
        "regression",
    ]
    fixed = next(change for change in changes if change["id"] == "fixed")
    assert fixed["before"] == {
        "candidate_present": True,
        "candidate_text": "储蓄卡（8885)",
        "raw_exact": False,
    }
    assert fixed["after"] == {
        "candidate_present": True,
        "candidate_text": "储蓄卡（8885）",
        "raw_exact": True,
    }
    assert fixed["candidate_changed"] is True
    assert fixed["raw_exact_changed"] is True
    assert fixed["transition"] == "wrong_to_correct"
    assert _summary(lines) == {
        "changed": 3,
        "comparison_field": "payment_method_field",
        "correct_to_wrong": 1,
        "field": "payment",
        "records": 4,
        "unchanged": 1,
        "wrong_to_correct": 1,
    }
    assert (before / "comparisons.jsonl").read_bytes() == before_bytes
    assert (after / "comparisons.jsonl").read_bytes() == after_bytes


def test_unfiltered_diff_exposes_changes_in_other_fields(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_score(
        before,
        [
            _row("payment", reference="余额", candidate="余额"),
            _row("amount", field="amount", reference="1.00", candidate="1.00"),
        ],
    )
    _write_score(
        after,
        [
            _row("payment", reference="余额", candidate="余额"),
            _row("amount", field="amount", reference="1.00", candidate="2.00"),
        ],
    )

    lines = MODULE.compare(before_dir=before, after_dir=after)

    assert [change["field"] for change in _changes(lines)] == ["amount"]
    assert _summary(lines) == {
        "changed": 1,
        "comparison_field": None,
        "correct_to_wrong": 1,
        "field": "all",
        "records": 2,
        "unchanged": 1,
        "wrong_to_correct": 0,
    }


def test_diff_binds_by_source_and_field_when_id_is_absent(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before_row = _row("one", reference="余额", candidate="银行卡")
    after_row = _row("one", reference="余额", candidate="余额")
    before_row.pop("id")
    after_row.pop("id")
    _write_score(before, [before_row])
    _write_score(after, [after_row])

    changes = _changes(MODULE.compare(before_dir=before, after_dir=after))

    assert len(changes) == 1
    assert "id" not in changes[0]
    assert changes[0]["source"] == r"D:\receipts\one.jpg"
    assert changes[0]["transition"] == "wrong_to_correct"


@pytest.mark.parametrize("drift", ["removed", "added", "id"])
def test_diff_rejects_comparison_collection_drift(tmp_path: Path, drift: str) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    original = [_row("one"), _row("two")]
    changed = [_row("one"), _row("two")]
    if drift == "removed":
        changed.pop()
    elif drift == "added":
        changed.append(_row("three"))
    else:
        changed[1] = _row("replacement", source=changed[1]["source"])
    _write_score(before, original)
    _write_score(after, changed)

    with pytest.raises(MODULE.ScoreDiffError, match="comparison collection changed"):
        MODULE.compare(before_dir=before, after_dir=after)


def test_diff_rejects_reference_truth_drift_even_when_field_is_filtered(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_score(
        before,
        [
            _row("payment", reference="余额", candidate="余额"),
            _row("amount", field="amount", reference="1.00", candidate="1.00"),
        ],
    )
    _write_score(
        after,
        [
            _row("payment", reference="余额", candidate="余额"),
            _row("amount", field="amount", reference="2.00", candidate="1.00"),
        ],
    )

    with pytest.raises(MODULE.ScoreDiffError, match="reference truth changed"):
        MODULE.compare(before_dir=before, after_dir=after, field="payment")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"source":"x","source":"y"}\n', "duplicate JSON key"),
        ('{"score":NaN}\n', "non-standard JSON constant"),
        ("[]\n", "expected one JSON object"),
        ("\n", "blank JSONL line"),
    ],
)
def test_diff_rejects_non_strict_jsonl(
    tmp_path: Path, contents: str, message: str
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    (before / "comparisons.jsonl").write_text(contents, encoding="utf-8")
    _write_score(after, [_row("one")])

    with pytest.raises(MODULE.ScoreDiffError, match=message):
        MODULE.compare(before_dir=before, after_dir=after)


def test_diff_rejects_inconsistent_raw_exact_and_mixed_id_mode(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    inconsistent = _row("one", reference="余额", candidate="余额")
    inconsistent["raw_exact"] = False
    _write_score(before, [inconsistent])
    _write_score(after, [_row("one")])
    with pytest.raises(MODULE.ScoreDiffError, match="raw_exact disagrees"):
        MODULE.compare(before_dir=before, after_dir=after)

    mixed = tmp_path / "mixed"
    row_without_id = _row("two")
    row_without_id.pop("id")
    _write_score(mixed, [_row("one"), row_without_id])
    with pytest.raises(MODULE.ScoreDiffError, match="id must be present on every row"):
        MODULE.compare(before_dir=mixed, after_dir=after)


def test_diff_rejects_inconsistent_missing_reason(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    invalid = _row("one")
    invalid["missing_reason"] = "should_not_be_set"
    _write_score(before, [invalid])
    _write_score(after, [_row("one")])

    with pytest.raises(MODULE.ScoreDiffError, match="missing_reason must be null"):
        MODULE.compare(before_dir=before, after_dir=after)


def test_main_returns_two_for_missing_field_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_score(before, [_row("one", field="amount", reference="1.00", candidate="1.00")])
    _write_score(after, [_row("one", field="amount", reference="1.00", candidate="1.00")])

    exit_code = MODULE.main(
        ["--before", str(before), "--after", str(after), "--field", "payment"]
    )

    assert exit_code == 2
    assert (
        "score_diff_error=comparisons contain no rows for field 'payment'"
        in capsys.readouterr().out
    )
