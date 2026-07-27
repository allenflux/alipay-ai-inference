from __future__ import annotations

import json
from pathlib import Path
import csv

import pytest
from PIL import Image

from transfer_receipt_ai.ocr_review import (
    DECISION_CANDIDATE,
    DECISION_CUSTOM,
    ReviewConfigurationError,
    ReviewStore,
    prepare_review_records,
)


def _comparison(*, record_id: str, image: Path, reference: str, candidate: str) -> dict[str, object]:
    return {
        "id": record_id,
        "field": "amount",
        "reference_text": reference,
        "candidate_text": candidate,
        "ctc_candidate_text": candidate,
        "structured_candidate_text": candidate,
        "confidence": 0.99,
        "structured_confidence": 0.98,
        "image": image.as_posix(),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_review_preparation_filters_matches_and_resumes_decisions(tmp_path: Path) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (16, 8), "white").save(first_image)
    Image.new("RGB", (16, 8), "white").save(second_image)
    comparisons = tmp_path / "comparisons.jsonl"
    _write_rows(
        comparisons,
        [
            _comparison(record_id="one", image=first_image, reference="99.99", candidate="99.98"),
            _comparison(record_id="two", image=second_image, reference="12.00", candidate="12.00"),
        ],
    )
    labels = tmp_path / "truth.jsonl"
    records = prepare_review_records(input_path=comparisons, output_path=labels)
    assert len(records) == 1
    assert records[0]["decision"] is None
    assert records[0]["kind"] == "receipt_ocr_manual_review_v1"
    assert isinstance(records[0]["input_row_sha256"], str)

    store = ReviewStore(records=records, output_path=labels)
    saved = store.save_decision(index=0, decision=DECISION_CANDIDATE)
    assert saved["truth_text"] == "99.98"
    assert store.progress()["reviewed"] == 1

    resumed = prepare_review_records(input_path=comparisons, output_path=labels)
    assert resumed[0]["decision"] == DECISION_CANDIDATE
    assert resumed[0]["truth_text"] == "99.98"


def test_review_custom_truth_and_safe_image_lookup(tmp_path: Path) -> None:
    image = tmp_path / "crop.png"
    Image.new("RGB", (16, 8), "white").save(image)
    comparisons = tmp_path / "comparisons.jsonl"
    _write_rows(comparisons, [_comparison(record_id="one", image=image, reference="08:42", candidate="08:4")])
    labels = tmp_path / "truth.jsonl"
    store = ReviewStore(
        records=prepare_review_records(input_path=comparisons, output_path=labels),
        output_path=labels,
    )
    assert store.image_path(0) == image.resolve()
    saved = store.save_decision(index=0, decision=DECISION_CUSTOM, truth_text="08:42")
    assert saved["truth_text"] == "08:42"
    with pytest.raises(ReviewConfigurationError, match="must not be empty"):
        store.save_decision(index=0, decision=DECISION_CUSTOM, truth_text="  ")
    with pytest.raises(IndexError):
        store.image_path(1)


def test_review_accepts_the_csv_export_shape_and_keeps_numeric_confidence(tmp_path: Path) -> None:
    image = tmp_path / "crop.png"
    Image.new("RGB", (16, 8), "white").save(image)
    comparisons = tmp_path / "manual-review.csv"
    with comparisons.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("id", "field", "reference_text", "candidate_text", "confidence", "image"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "one",
                "field": "time",
                "reference_text": "00:09",
                "candidate_text": "00:06",
                "confidence": "0.999977",
                "image": image.as_posix(),
            }
        )
    records = prepare_review_records(input_path=comparisons, output_path=tmp_path / "truth.jsonl")
    assert records[0]["confidence"] == "0.999977"
