from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from transfer_receipt_ai.ocr_lite_dataset import UNKNOWN_RECIPIENT_CLASS, build_lite_dataset
from transfer_receipt_ai.ocr_train import load_records


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 80, 3), shade, dtype=np.uint8)).save(path)


def _record(index: int, field: str, value: str, split: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"sample-{index}",
        "image": f"images/{field}/{index}.png",
        "field": field,
        "text": value,
        "semantic_value": value,
        "split": split,
        "group_id": f"receipt-{index}",
        "label_source": "paddle_pseudo",
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_build_lite_dataset_creates_specialised_targets_and_recipient_unknown_policy(tmp_path: Path) -> None:
    dataset = tmp_path / "pseudo"
    records = [
        _record(1, "amount", "¥100.00", "train"),
        _record(2, "amount", "¥200.10", "val"),
        _record(3, "time", "12:06", "train"),
        _record(4, "time", "09:05", "val"),
        _record(5, "transfer_status", "success", "train"),
        _record(6, "transfer_status", "pending", "val"),
        _record(7, "payment_method_field", "bank_card", "train"),
        _record(8, "payment_method_field", "balance", "val"),
        _record(9, "recipient_field", "商户甲", "train"),
        _record(10, "recipient_field", "商户甲", "train"),
        _record(11, "recipient_field", "商户甲", "val"),
        _record(12, "recipient_field", "长尾商户", "test"),
    ]
    for index, record in enumerate(records):
        _write_image(dataset / str(record["image"]), 25 + index)
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    output = tmp_path / "lite"
    summary = build_lite_dataset(
        records_path=records_path,
        output_dir=output,
        recipient_top_k=1,
        recipient_min_train_count=2,
        recipient_unknown_to_known_ratio=2.0,
    )

    amount = _read_jsonl(output / "amount_ctc.jsonl")
    time = _read_jsonl(output / "time_ctc.jsonl")
    status = _read_jsonl(output / "transfer_status_classifier.jsonl")
    payment = _read_jsonl(output / "payment_method_classifier.jsonl")
    recipients = _read_jsonl(output / "recipient_classifier.jsonl")
    assert {record["text"] for record in amount} == {"100.00", "200.10"}
    assert {record["text"] for record in time} == {"12:06", "09:05"}
    assert {record["class_name"] for record in status} == {"success", "pending"}
    assert {record["class_name"] for record in payment} == {"bank_card", "balance"}
    assert {record["class_name"] for record in recipients if record["id"] == "sample-11:recipient_classifier"} == {"known_0001"}
    assert {record["class_name"] for record in recipients if record["id"] == "sample-12:recipient_classifier"} == {
        UNKNOWN_RECIPIENT_CLASS
    }
    catalog = json.loads((output / "recipient_catalog.json").read_text(encoding="utf-8"))
    assert catalog["entries"] == [{"class_name": "known_0001", "recipient_value": "商户甲", "train_records": 2}]
    assert summary["tasks"]["recipient_classifier"]["records"] == len(recipients)


def test_lite_ctc_manifest_can_reference_a_separate_dataset_root(tmp_path: Path) -> None:
    dataset = tmp_path / "pseudo"
    record = _record(1, "amount", "¥100.00", "train")
    second = _record(2, "amount", "¥200.00", "val")
    for offset, item in enumerate((record, second)):
        _write_image(dataset / str(item["image"]), 30 + offset)
    records_path = dataset / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in (record, second)), encoding="utf-8"
    )
    output = tmp_path / "lite"
    build_lite_dataset(records_path=records_path, output_dir=output, recipient_top_k=1, recipient_min_train_count=1)

    loaded = load_records(output / "amount_ctc.jsonl", fields=("amount",), dataset_root=dataset)
    assert [record["text"] for record in loaded] == ["100.00", "200.00"]
    assert all(Path(record["image_path"]).is_file() for record in loaded)
