from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from transfer_receipt_ai.ocr_truth_dataset import build_truth_dataset


def test_transaction_truth_builder_creates_paddle_free_field_crops(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source = source_dir / "s3_voucher_GWCZ123_20260701000000.png"
    source_dir.mkdir()
    pixels = np.full((100, 180, 3), 255, dtype=np.uint8)
    for index in range(5):
        pixels[10 + index * 15 : 20 + index * 15, 10:150] = 30 + index * 20
    Image.fromarray(pixels).save(source)
    result_dir = tmp_path / "detector-results"
    result_dir.mkdir()
    labels = ("time", "amount", "transfer_status", "recipient_field", "payment_method_field")
    result = {
        "source": source.as_posix(),
        "geometry": {
            "source_size": {"width": 180, "height": 100},
            "rectified_size": {"width": 180, "height": 100},
            "H_original_to_rectified": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "detections": [
            {"label": label, "score": 0.99, "bbox_rectified": [10.0, 10.0 + index * 15, 150.0, 20.0 + index * 15]}
            for index, label in enumerate(labels)
        ],
    }
    (result_dir / "receipt.json").write_text(json.dumps(result), encoding="utf-8")
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        json.dumps(
            {
                "receipt_key": "GWCZ123",
                "amount": "100.00",
                "time": "12:06",
                "transfer_status": "success",
                "payment_method": "bank_card",
                "recipient": "交易商家",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "truth-dataset"
    summary = build_truth_dataset(
        results_dir=result_dir,
        truth_path=truth,
        output_dir=output,
        validation_ratio=0.0,
        test_ratio=0.0,
    )

    assert summary["counts"]["accepted"] == 5
    records = [
        json.loads(line)
        for line in (output / "pseudo_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    values = {str(record["field"]): str(record["semantic_value"]) for record in records}
    assert values == {
        "time": "12:06",
        "amount": "¥100.00",
        "transfer_status": "success",
        "recipient_field": "交易商家",
        "payment_method_field": "bank_card",
    }
    assert {record["label_source"] for record in records} == {"transaction_truth"}
    assert {record["group_id"] for record in records} == {"receipt_key:GWCZ123"}
    assert all((output / str(record["image"])).is_file() for record in records)


def test_transaction_truth_amount_fen_must_be_an_integer() -> None:
    from transfer_receipt_ai.ocr_truth_dataset import _normalise_amount

    assert _normalise_amount(None, {"amount_fen": 100}) == "¥1.00"
    assert _normalise_amount(None, {"amount_fen": 100.5}) is None
    assert _normalise_amount("12.345", {}) is None
    assert _normalise_amount("amount: ¥12.34", {}) is None
    assert _normalise_amount("¥1,234.5", {}) == "¥1234.50"


def test_transaction_truth_same_receipt_key_never_crosses_a_split(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    result_dir = tmp_path / "detector-results"
    first_result_dir = result_dir / "first"
    second_result_dir = result_dir / "second"
    first_result_dir.mkdir(parents=True)
    second_result_dir.mkdir(parents=True)
    result_template = {
        "geometry": {
            "source_size": {"width": 100, "height": 50},
            "rectified_size": {"width": 100, "height": 50},
            "H_original_to_rectified": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "detections": [{"label": "amount", "score": 0.99, "bbox_rectified": [10.0, 10.0, 90.0, 40.0]}],
    }
    for index, (name, result_path, shade) in enumerate(
        (
            ("copy_GWCZsame_20260701000000.png", first_result_dir / "first.json", 90),
            ("different_GWCZsame_20260702000000.png", second_result_dir / "second.json", 150),
        )
    ):
        source = source_dir / name
        Image.fromarray(np.full((50, 100, 3), shade + index, dtype=np.uint8)).save(source)
        result_path.write_text(json.dumps({**result_template, "source": source.as_posix()}), encoding="utf-8")
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps({"receipt_key": "GWCZsame", "amount": "100.00"}) + "\n", encoding="utf-8")

    output = tmp_path / "truth-dataset"
    build_truth_dataset(
        results_dir=result_dir,
        truth_path=truth,
        output_dir=output,
        validation_ratio=0.50,
        test_ratio=0.0,
    )
    records = [
        json.loads(line)
        for line in (output / "pseudo_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    amount_records = [record for record in records if record["field"] == "amount"]
    assert len(amount_records) == 2
    assert {record["group_id"] for record in amount_records} == {"receipt_key:GWCZsame"}
    assert len({record["split"] for record in amount_records}) == 1


def test_transaction_truth_conflicting_identical_crop_excludes_both_labels(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    result_dir = tmp_path / "detector-results"
    result_dir.mkdir()
    result_template = {
        "geometry": {
            "source_size": {"width": 100, "height": 50},
            "rectified_size": {"width": 100, "height": 50},
            "H_original_to_rectified": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "detections": [{"label": "amount", "score": 0.99, "bbox_rectified": [10.0, 10.0, 90.0, 40.0]}],
    }
    for key in ("GWCZfirst", "GWCZsecond"):
        source = source_dir / f"copy_{key}_20260701000000.png"
        Image.fromarray(np.full((50, 100, 3), 123, dtype=np.uint8)).save(source)
        (result_dir / f"{key}.json").write_text(
            json.dumps({**result_template, "source": source.as_posix()}), encoding="utf-8"
        )
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"receipt_key": "GWCZfirst", "amount": "100.00"},
                {"receipt_key": "GWCZsecond", "amount": "200.00"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "truth-dataset"
    summary = build_truth_dataset(
        results_dir=result_dir,
        truth_path=truth,
        output_dir=output,
        validation_ratio=0.0,
        test_ratio=0.0,
    )

    assert summary["counts"]["candidate_accepted_before_conflict_filter"] == 1
    assert summary["counts"]["accepted"] == 0
    assert not list((output / "images" / "amount").glob("*.png"))
    reasons = {
        json.loads(line)["reason"]
        for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }
    assert "conflicting_duplicate_crop" in reasons
    assert "conflicting_duplicate_crop_excluded" in reasons
